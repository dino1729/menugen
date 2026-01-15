"""
NVIDIA NIM API Image Generation Client.

Generates images using NVIDIA's Stable Diffusion 3 API.
Includes automatic fallback from large to medium model on 404,
and flexible response parsing for different API response formats.

Configuration is loaded from config.py module.
"""
import asyncio
import base64
import logging
import os
import random
from typing import Dict, Optional, Tuple

import aiofiles
import httpx

from config import get_config, get_nvidia_model_params
from client_utils import ImageGenerationError, IMAGE_SAVE_DIR, sanitize_filename

logger = logging.getLogger("menugen.nvidia_image_generation")

# Alternative URLs for fallback (large -> medium)
NVIDIA_SD3_LARGE_URL = "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-3-large"
NVIDIA_SD3_MEDIUM_URL = "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-3-medium"


def build_nvidia_url(model_id: str) -> str:
    """
    Build the full NVIDIA NIM API URL from a model ID.

    Args:
        model_id: Model identifier like 'black-forest-labs/flux.1-schnell'
                  or 'stabilityai/stable-diffusion-3.5-large'

    Returns:
        Full API URL like 'https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-schnell'
    """
    config = get_config()
    base_url = config.nvidia.base_url.rstrip("/")
    return f"{base_url}/{model_id}"


async def _calculate_backoff(attempt: int, response: Optional[httpx.Response] = None) -> float:
    """Calculate backoff time with exponential increase and jitter."""
    config = get_config()
    retry_cfg = config.retry

    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass

    backoff = min(
        retry_cfg.initial_backoff_seconds * (2 ** attempt),
        retry_cfg.max_backoff_seconds
    )
    jitter = backoff * retry_cfg.jitter_factor * random.random()
    return backoff + jitter


def _extract_image_from_response(response_body: Dict) -> str:
    """
    Extract base64 image data from NVIDIA API response.
    
    Handles multiple response formats:
      - {"image": "<base64>"} (current format)
      - {"images": ["<base64>", ...]} (alternative format)
      - {"data": [{"b64_json": "..."}]} (OpenAI-like format)
      
    Returns:
        Base64 encoded image string
        
    Raises:
        ImageGenerationError: If no image data found in response
    """
    # Format 1: Direct image field
    if "image" in response_body:
        return response_body["image"]
    
    # Format 2: images array
    if "images" in response_body and len(response_body["images"]) > 0:
        return response_body["images"][0]
    
    # Format 3: OpenAI-like data array with b64_json
    if "data" in response_body and len(response_body["data"]) > 0:
        item = response_body["data"][0]
        if "b64_json" in item:
            return item["b64_json"]
        if "image" in item:
            return item["image"]
    
    # Format 4: artifacts array (some SD3 APIs)
    if "artifacts" in response_body and len(response_body["artifacts"]) > 0:
        artifact = response_body["artifacts"][0]
        if "base64" in artifact:
            return artifact["base64"]
    
    raise ImageGenerationError(
        f"NVIDIA API response missing image data. "
        f"Expected 'image', 'images', 'data[].b64_json', or 'artifacts[].base64'. "
        f"Got: {list(response_body.keys())}"
    )


async def _make_nvidia_request(
    url: str,
    api_key: str,
    payload: Dict,
    timeout: float = 120.0,
) -> Tuple[int, Dict]:
    """
    Make a request to NVIDIA API with retry logic.

    Returns:
        Tuple of (status_code, response_body)
    """
    config = get_config()
    max_retries = config.retry.max_retries
    retry_status_codes = set(config.retry.retry_status_codes)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    last_exception: Optional[Exception] = None
    last_response: Optional[httpx.Response] = None

    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(max_retries):
            try:
                response = await client.post(url, headers=headers, json=payload)

                # Return immediately on success or non-retriable error
                if response.status_code not in retry_status_codes:
                    try:
                        return response.status_code, response.json()
                    except Exception:
                        return response.status_code, {"error": response.text}

                # Retriable status code
                last_response = response
                backoff = await _calculate_backoff(attempt, response)
                logger.warning(
                    f"NVIDIA API returned {response.status_code}, "
                    f"attempt {attempt + 1}/{max_retries}, backing off {backoff:.2f}s"
                )
                await asyncio.sleep(backoff)

            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as e:
                last_exception = e
                backoff = await _calculate_backoff(attempt)
                logger.warning(
                    f"NVIDIA API connection error: {e}, "
                    f"attempt {attempt + 1}/{max_retries}, backing off {backoff:.2f}s"
                )
                await asyncio.sleep(backoff)

    # All retries exhausted
    if last_response is not None:
        try:
            return last_response.status_code, last_response.json()
        except Exception:
            return last_response.status_code, {"error": last_response.text}
    elif last_exception is not None:
        raise ImageGenerationError(
            f"NVIDIA API request failed after {max_retries} retries: {last_exception}. "
            f"Tip: You can switch to LiteLLM provider by setting IMAGE_PROVIDER=litellm"
        )
    else:
        raise ImageGenerationError(
            f"NVIDIA API request failed after {max_retries} retries. "
            f"Tip: You can switch to LiteLLM provider by setting IMAGE_PROVIDER=litellm"
        )


async def generate_image_nvidia_raw(
    prompt: str,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    model_id: Optional[str] = None,
    timeout: float = 120.0,
    fallback_on_404: bool = True,
    **kwargs,
) -> bytes:
    """
    Generate an image using NVIDIA NIM API and return raw bytes.

    Supports multiple model types with different parameter requirements:
    - Stable Diffusion models: cfg_scale, aspect_ratio, steps, negative_prompt
    - FLUX models: cfg_scale (fixed at 0), width, height, steps (max 4 for Schnell)

    Args:
        prompt: Text description for image generation
        api_key: NVIDIA API key (defaults to config)
        api_url: NVIDIA API URL (defaults to config-based URL from model_id)
        model_id: Model identifier to determine parameter format (e.g., 'black-forest-labs/flux.1-schnell')
        timeout: Request timeout in seconds
        fallback_on_404: If True, try medium model when large SD model returns 404
        **kwargs: Model-specific parameters (cfg_scale, aspect_ratio, width, height, seed, steps, negative_prompt)

    Returns:
        Raw image bytes (decoded from base64)

    Raises:
        ImageGenerationError: On API errors or missing image data
    """
    config = get_config()
    key = api_key or config.nvidia.api_key
    url = api_url or os.getenv("NVIDIA_IMAGE_GEN_URL")

    if not key:
        raise ImageGenerationError(
            "NVIDIA_API_KEY is not set. "
            "Consider using LiteLLM provider (IMAGE_PROVIDER=litellm) instead."
        )
    if not url:
        raise ImageGenerationError(
            "NVIDIA_IMAGE_GEN_URL is not set. "
            "Consider using LiteLLM provider (IMAGE_PROVIDER=litellm) instead."
        )

    # Determine if this is a FLUX model
    is_flux = model_id and "flux" in model_id.lower()

    # Build payload based on model type
    if is_flux:
        # FLUX models use width/height instead of aspect_ratio, and don't support negative_prompt
        payload = {
            "prompt": prompt,
            "cfg_scale": kwargs.get("cfg_scale", 0),
            "width": kwargs.get("width", 1024),
            "height": kwargs.get("height", 1024),
            "seed": kwargs.get("seed", 0),
            "steps": kwargs.get("steps", 4),
        }
    else:
        # Stable Diffusion models
        payload = {
            "prompt": prompt,
            "cfg_scale": kwargs.get("cfg_scale", 5),
            "aspect_ratio": kwargs.get("aspect_ratio", "1:1"),
            "seed": kwargs.get("seed", 0),
            "steps": kwargs.get("steps", 50),
            "negative_prompt": kwargs.get("negative_prompt", ""),
        }

    logger.info(f"Generating image with NVIDIA API: {url} (model_id={model_id}, is_flux={is_flux})")
    logger.info(f"Payload: {payload}")
    status_code, response_body = await _make_nvidia_request(url, key, payload, timeout)
    
    # Handle 404 with fallback - ONLY for Stable Diffusion models
    # FLUX models should not fall back to SD models as they have incompatible APIs
    if status_code == 404 and fallback_on_404 and not is_flux:
        # Check if we're using large and can fall back to medium
        if "stable-diffusion-3-large" in url:
            fallback_url = url.replace("stable-diffusion-3-large", "stable-diffusion-3-medium")
            logger.warning(f"NVIDIA API returned 404 for large model, falling back to: {fallback_url}")
            status_code, response_body = await _make_nvidia_request(fallback_url, key, payload, timeout)
        elif "stable-diffusion" in url and url != NVIDIA_SD3_MEDIUM_URL:
            # Try the known medium URL as last resort for SD models only
            logger.warning(f"NVIDIA API returned 404, trying known medium endpoint: {NVIDIA_SD3_MEDIUM_URL}")
            status_code, response_body = await _make_nvidia_request(NVIDIA_SD3_MEDIUM_URL, key, payload, timeout)
    
    # Check for errors
    if status_code != 200:
        error_msg = response_body.get("error", response_body.get("detail", str(response_body)))
        raise ImageGenerationError(
            f"NVIDIA API returned {status_code}: {error_msg}. "
            f"Tip: You can switch to LiteLLM provider by setting IMAGE_PROVIDER=litellm"
        )
    
    # Extract and decode image
    try:
        base64_image = _extract_image_from_response(response_body)
        return base64.b64decode(base64_image)
    except Exception as e:
        if isinstance(e, ImageGenerationError):
            raise
        raise ImageGenerationError(f"Failed to decode image: {e}")


async def generate_menu_item_image_nvidia(item: Dict, session_config: dict = None) -> str:
    """
    Generate an image using NVIDIA NIM API.

    This is the main entry point for menu item image generation.
    Supports multiple NVIDIA models including:
    - Stable Diffusion 3/3.5 (stabilityai/stable-diffusion-3-medium, stabilityai/stable-diffusion-3.5-large)
    - FLUX.1 models (black-forest-labs/flux.1-dev, flux.1-schnell, flux.1-kontext-dev)

    Args:
        item: Dictionary containing 'name' and 'description' of menu item
        session_config: Optional dict with 'image_gen_model' specifying the NVIDIA model to use

    Returns:
        Local filename of the saved image

    Raises:
        ImageGenerationError: If image generation or saving fails
    """
    item_name = item.get('name', 'Unknown')
    description = item.get('description', '')

    logger.info(f"generate_menu_item_image_nvidia called with session_config={session_config}")

    # Get model from session config or fall back to environment variable
    model_id = None
    if session_config:
        model_id = session_config.get('image_gen_model')
        logger.info(f"Extracted model_id from session_config: {model_id}")

    # Determine the API URL
    if model_id:
        # Build URL from model ID
        api_url = build_nvidia_url(model_id)
        logger.info(f"Using model from session config: {model_id} -> {api_url}")
    else:
        # Fall back to environment variable
        api_url = os.getenv("NVIDIA_IMAGE_GEN_URL")
        logger.info(f"Using default NVIDIA URL from environment: {api_url}")

    # Create prompt for the menu item
    prompt = f"Create a high-quality, appetizing photo of this menu item: {item_name}. Description: {description}"

    # Get model-specific parameters from config
    model_params = get_nvidia_model_params(model_id or "")
    logger.info(f"Generating image with NVIDIA API for item: {item_name} using params: {model_params}")

    try:
        # Generate image with model-specific parameters
        image_data = await generate_image_nvidia_raw(
            prompt=prompt,
            api_url=api_url,
            model_id=model_id,
            **model_params,
        )

        # Save locally
        filename = sanitize_filename(item_name) + ".png"
        filepath = os.path.join(IMAGE_SAVE_DIR, filename)

        async with aiofiles.open(filepath, mode='wb') as f:
            await f.write(image_data)

        logger.info(f"Successfully saved NVIDIA-generated image locally: {filepath}")
        return filename

    except ImageGenerationError:
        raise
    except Exception as e:
        raise ImageGenerationError(f"Unexpected error generating image for {item_name}: {e}")
