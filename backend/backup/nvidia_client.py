import os
import logging
import httpx
import base64
import aiofiles
from typing import Dict
from openai_client import ImageGenerationError, IMAGE_SAVE_DIR, sanitize_filename

# Set up logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("menugen.nvidia_client")

# Get NVIDIA API key from environment
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# NVIDIA API endpoint
NVIDIA_INVOKE_URL = "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-3-medium"

# Fixed parameters for consistent image generation
NVIDIA_DEFAULTS = {
    "cfg_scale": 5,
    "aspect_ratio": "1:1",  # Square images to match DALL-E 3
    "seed": 0,
    "steps": 50,
    "negative_prompt": ""
}

async def generate_image_nvidia(item: Dict) -> str:
    """
    Generate an image using NVIDIA NIM API (Stable Diffusion 3).
    
    Args:
        item: Dictionary containing 'name' and 'description' of menu item
        
    Returns:
        Local filename of the saved image
        
    Raises:
        ImageGenerationError: If image generation or saving fails
    """
    item_name = item.get('name', 'Unknown')
    description = item.get('description', '')
    
    if not NVIDIA_API_KEY:
        error_msg = "NVIDIA_API_KEY environment variable is not set"
        logger.error(error_msg)
        raise ImageGenerationError(error_msg)
    
    # Create prompt for the menu item
    prompt = f"Create a high-quality, appetizing photo of this menu item: {item_name}. Description: {description}"
    
    logger.info(f"Generating image with NVIDIA API for item: {item_name}")
    
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Accept": "application/json",
    }
    
    payload = {
        "prompt": prompt,
        **NVIDIA_DEFAULTS
    }
    
    try:
        # Make async request to NVIDIA API using httpx
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(NVIDIA_INVOKE_URL, headers=headers, json=payload)
            response.raise_for_status()
            response_body = response.json()
        
        # Extract base64 image from response
        if 'image' not in response_body:
            error_msg = f"NVIDIA API response missing 'image' field. Response: {response_body}"
            logger.error(error_msg)
            raise ImageGenerationError(error_msg)
        
        base64_image = response_body['image']
        logger.info(f"Successfully received image data from NVIDIA API for item: {item_name}")
        
        # Decode base64 and save locally
        image_data = base64.b64decode(base64_image)
        filename = sanitize_filename(item_name) + ".png"
        filepath = os.path.join(IMAGE_SAVE_DIR, filename)
        
        # Save image asynchronously
        async with aiofiles.open(filepath, mode='wb') as f:
            await f.write(image_data)
        
        logger.info(f"Successfully saved NVIDIA-generated image locally: {filepath}")
        return filename
        
    except httpx.TimeoutException:
        error_msg = f"NVIDIA API request timed out for item: {item_name}"
        logger.error(error_msg)
        raise ImageGenerationError(error_msg)
    except httpx.HTTPStatusError as e:
        error_msg = f"NVIDIA API returned error status {e.response.status_code} for item {item_name}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ImageGenerationError(error_msg)
    except httpx.RequestError as e:
        error_msg = f"NVIDIA API request failed for item {item_name}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ImageGenerationError(error_msg)
    except base64.binascii.Error as e:
        error_msg = f"Failed to decode base64 image data for item {item_name}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ImageGenerationError(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error generating image with NVIDIA API for item {item_name}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ImageGenerationError(error_msg)

