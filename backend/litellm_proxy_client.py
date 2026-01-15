"""
LiteLLM Proxy Client - Direct httpx-based client for OpenAI-compatible endpoints.

This module provides a clean, reliable way to call the LiteLLM proxy without
the litellm SDK's provider inference logic that can bypass the proxy.

All calls go through:
  - /v1/chat/completions (text generation)
  - /v1/images/generations (image generation)
  - /v1/video/generations (video generation, with fallback to images endpoint)

Includes built-in retry/backoff for transient errors (429, 5xx).
Configuration is loaded from config.py module.
"""
import asyncio
import base64
import json
import logging
import random
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

from config import get_config, get_base_url_with_v1, get_retry_status_codes

logger = logging.getLogger("menugen.litellm_proxy_client")


class ProxyClientError(Exception):
    """Raised when the proxy client encounters an error."""
    def __init__(self, message: str, status_code: Optional[int] = None, response_body: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


async def _calculate_backoff(attempt: int, response: Optional[httpx.Response] = None) -> float:
    """
    Calculate backoff time with exponential increase and jitter.
    Respects Retry-After header if present.
    """
    config = get_config()
    retry_cfg = config.retry

    # Check for Retry-After header
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass  # Fall through to exponential backoff

    # Exponential backoff with jitter
    backoff = min(
        retry_cfg.initial_backoff_seconds * (2 ** attempt),
        retry_cfg.max_backoff_seconds
    )
    jitter = backoff * retry_cfg.jitter_factor * random.random()
    return backoff + jitter


async def _request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: Dict[str, str],
    json_body: Optional[Dict[str, Any]] = None,
    timeout: float = 120.0,
) -> httpx.Response:
    """
    Make an HTTP request with automatic retry on transient errors.

    Retries on: 429 (rate limit), 408 (timeout), 500/502/503/504 (server errors)
    Uses exponential backoff with jitter, respects Retry-After header.
    """
    config = get_config()
    max_retries = config.retry.max_retries
    retry_status_codes = get_retry_status_codes()

    last_exception: Optional[Exception] = None
    last_response: Optional[httpx.Response] = None

    for attempt in range(max_retries):
        try:
            response = await client.request(
                method=method,
                url=url,
                headers=headers,
                json=json_body,
                timeout=timeout,
            )

            if response.status_code not in retry_status_codes:
                return response

            # Retriable status code - log and backoff
            last_response = response
            backoff = await _calculate_backoff(attempt, response)
            logger.warning(
                f"Retriable status {response.status_code} on {url}, "
                f"attempt {attempt + 1}/{max_retries}, backing off {backoff:.2f}s"
            )
            await asyncio.sleep(backoff)

        except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as e:
            last_exception = e
            backoff = await _calculate_backoff(attempt)
            logger.warning(
                f"Connection error on {url}: {e}, "
                f"attempt {attempt + 1}/{max_retries}, backing off {backoff:.2f}s"
            )
            await asyncio.sleep(backoff)

    # All retries exhausted
    if last_response is not None:
        raise ProxyClientError(
            f"Max retries exceeded for {url}, last status: {last_response.status_code}",
            status_code=last_response.status_code,
            response_body=last_response.text,
        )
    elif last_exception is not None:
        raise ProxyClientError(f"Max retries exceeded for {url}: {last_exception}")
    else:
        raise ProxyClientError(f"Max retries exceeded for {url}")


def _get_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    """Build standard headers for proxy requests."""
    config = get_config()
    key = api_key or config.openai_api_key
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


# =============================================================================
# Chat Completions
# =============================================================================

async def chat_completions(
    model: str,
    messages: List[Dict[str, Any]],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    response_format: Optional[Dict[str, str]] = None,
    timeout: float = 120.0,
    **kwargs,
) -> Dict[str, Any]:
    """
    Call the LiteLLM proxy's /v1/chat/completions endpoint.

    Args:
        model: Model name (as configured in your LiteLLM proxy)
        messages: List of message dicts with 'role' and 'content'
        base_url: Override default base URL from config
        api_key: Override default API key from config
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        response_format: Optional {"type": "json_object"} for JSON mode
        timeout: Request timeout in seconds
        **kwargs: Additional parameters to pass to the API

    Returns:
        Raw JSON response dict from the proxy

    Raises:
        ProxyClientError: On non-retriable errors or max retries exceeded
    """
    effective_base = base_url or get_base_url_with_v1()
    # Ensure URL ends with /v1 for the endpoint
    if not effective_base.endswith("/v1"):
        effective_base = f"{effective_base.rstrip('/')}/v1"
    url = f"{effective_base}/chat/completions"
    headers = _get_headers(api_key)
    
    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
        # Gemini uses max_output_tokens, add both for compatibility
        body["max_output_tokens"] = max_tokens
    if temperature is not None:
        body["temperature"] = temperature
    if response_format is not None:
        body["response_format"] = response_format
    body.update(kwargs)
    
    logger.debug(f"chat_completions: POST {url} model={model}")
    
    async with httpx.AsyncClient() as client:
        response = await _request_with_retry(client, "POST", url, headers, body, timeout)
        
        if response.status_code != 200:
            raise ProxyClientError(
                f"Chat completions failed: {response.status_code} {response.text}",
                status_code=response.status_code,
                response_body=response.text,
            )
        
        return response.json()


def extract_chat_content(response: Dict[str, Any]) -> str:
    """
    Extract the text content from a chat completions response.

    Args:
        response: Raw JSON response from chat_completions()

    Returns:
        The assistant's message content as a string

    Raises:
        ProxyClientError: If content cannot be extracted
    """
    try:
        content = response["choices"][0]["message"]["content"]
        if content is None:
            raise ProxyClientError(f"Chat content is None, response: {response}")
        return content.strip()
    except (KeyError, IndexError, TypeError) as e:
        raise ProxyClientError(f"Failed to extract chat content: {e}, response: {response}")


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract JSON from text that may contain markdown fences or extra content.
    
    Handles:
      - ```json ... ``` fences
      - Raw JSON objects
      - JSON embedded in other text
    """
    # Strip markdown fences
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    
    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Try to find first JSON object
    match = re.search(r'\{[\s\S]*\}', cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    raise ProxyClientError(f"Failed to extract JSON from text: {text[:200]}...")


# =============================================================================
# Image Generations
# =============================================================================

async def image_generations(
    model: str,
    prompt: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    size: str = "1024x1024",
    n: int = 1,
    response_format: str = "url",  # "url" or "b64_json"
    timeout: float = 180.0,
    **kwargs,
) -> Dict[str, Any]:
    """
    Call the LiteLLM proxy's /v1/images/generations endpoint.

    Args:
        model: Image model name (as configured in your LiteLLM proxy)
        prompt: Text description of the image to generate
        base_url: Override default base URL from config
        api_key: Override default API key from config
        size: Image size (e.g., "1024x1024")
        n: Number of images to generate
        response_format: "url" for URL or "b64_json" for base64
        timeout: Request timeout in seconds
        **kwargs: Additional parameters

    Returns:
        Raw JSON response dict from the proxy
    """
    effective_base = base_url or get_base_url_with_v1()
    if not effective_base.endswith("/v1"):
        effective_base = f"{effective_base.rstrip('/')}/v1"
    url = f"{effective_base}/images/generations"
    headers = _get_headers(api_key)
    
    body: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "n": n,
        "response_format": response_format,
    }
    body.update(kwargs)
    
    logger.debug(f"image_generations: POST {url} model={model}")
    
    async with httpx.AsyncClient() as client:
        response = await _request_with_retry(client, "POST", url, headers, body, timeout)
        
        if response.status_code != 200:
            raise ProxyClientError(
                f"Image generations failed: {response.status_code} {response.text}",
                status_code=response.status_code,
                response_body=response.text,
            )
        
        return response.json()


async def extract_image_bytes(
    response: Dict[str, Any],
    http_client: Optional[httpx.AsyncClient] = None,
) -> Tuple[bytes, Optional[str]]:
    """
    Extract image bytes from an image generations response.
    
    Handles:
      - b64_json field (base64 encoded)
      - data: URLs (base64 encoded)
      - HTTP URLs (downloads the image)
      
    Args:
        response: Raw JSON response from image_generations()
        http_client: Optional httpx client for downloading URLs
        
    Returns:
        Tuple of (image_bytes, original_url_or_none)
    """
    try:
        image_data = response["data"][0]
    except (KeyError, IndexError, TypeError) as e:
        raise ProxyClientError(f"No image data in response: {e}, response: {response}")
    
    # Handle base64 encoded
    if "b64_json" in image_data and image_data["b64_json"]:
        return base64.b64decode(image_data["b64_json"]), None
    
    # Handle URL
    url = image_data.get("url")
    if not url:
        raise ProxyClientError(f"No url or b64_json in image data: {image_data}")
    
    # Data URL
    if url.startswith("data:"):
        try:
            _, encoded = url.split(",", 1)
            return base64.b64decode(encoded), url
        except ValueError as e:
            raise ProxyClientError(f"Failed to parse data URL: {e}")
    
    # HTTP URL - download
    should_close = False
    if http_client is None:
        http_client = httpx.AsyncClient()
        should_close = True
    
    try:
        img_response = await http_client.get(url, timeout=60.0)
        img_response.raise_for_status()
        return img_response.content, url
    except httpx.HTTPError as e:
        raise ProxyClientError(f"Failed to download image from {url}: {e}")
    finally:
        if should_close:
            await http_client.aclose()


# =============================================================================
# Video Generations
# =============================================================================

async def video_generations(
    model: str,
    prompt: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    seconds: str = "8",
    size: str = "1280x720",
    timeout: float = 300.0,
    **kwargs,
) -> Dict[str, Any]:
    """
    Call the LiteLLM proxy's /v1/videos endpoint for video generation.

    See: https://docs.litellm.ai/docs/providers/openai/videos

    Args:
        model: Video model name (e.g., "sora-2", "veo-3.0-generate-001")
        prompt: Text description of the video to generate
        base_url: Override default base URL from config
        api_key: Override default API key from config
        seconds: Video duration in seconds (e.g., "8", "16")
        size: Video dimensions (e.g., "720x1280", "1280x720")
        timeout: Request timeout (video generation can be slow)
        **kwargs: Additional parameters

    Returns:
        Raw JSON response dict from the proxy containing video_id and status
    """
    config = get_config()
    base = base_url or config.openai_base_url
    headers = _get_headers(api_key)
    
    body: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "seconds": seconds,
        "size": size,
    }
    body.update(kwargs)
    
    async with httpx.AsyncClient() as client:
        # Use the correct LiteLLM video endpoint: /v1/videos
        video_url = f"{base}/v1/videos"
        logger.debug(f"video_generations: POST {video_url} model={model}")
        
        response = await _request_with_retry(client, "POST", video_url, headers, body, timeout)
        
        if response.status_code not in (200, 201, 202):
            raise ProxyClientError(
                f"Video generations failed: {response.status_code} {response.text}",
                status_code=response.status_code,
                response_body=response.text,
            )
        
        return response.json()


async def video_status(
    video_id: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    custom_llm_provider: Optional[str] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """
    Check video generation status via /v1/videos/{video_id}.

    Args:
        video_id: The video ID returned from video_generations()
        base_url: Override default base URL from config
        api_key: Override default API key from config
        custom_llm_provider: Provider name (e.g., "openai", "vertex_ai")
        timeout: Request timeout

    Returns:
        Status response dict with video status
    """
    config = get_config()
    base = base_url or config.openai_base_url
    headers = _get_headers(api_key)
    if custom_llm_provider:
        headers["custom-llm-provider"] = custom_llm_provider
    
    async with httpx.AsyncClient() as client:
        url = f"{base}/v1/videos/{video_id}"
        response = await _request_with_retry(client, "GET", url, headers, timeout=timeout)
        
        if response.status_code != 200:
            raise ProxyClientError(
                f"Video status check failed: {response.status_code} {response.text}",
                status_code=response.status_code,
                response_body=response.text,
            )
        
        return response.json()


async def video_content(
    video_id: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    custom_llm_provider: Optional[str] = None,
    timeout: float = 120.0,
) -> bytes:
    """
    Download video content via /v1/videos/{video_id}/content.

    Args:
        video_id: The video ID returned from video_generations()
        base_url: Override default base URL from config
        api_key: Override default API key from config
        custom_llm_provider: Provider name (e.g., "openai", "vertex_ai")
        timeout: Request timeout

    Returns:
        Raw video bytes
    """
    config = get_config()
    base = base_url or config.openai_base_url
    headers = _get_headers(api_key)
    headers["Accept"] = "application/octet-stream"
    if custom_llm_provider:
        headers["custom-llm-provider"] = custom_llm_provider
    
    async with httpx.AsyncClient() as client:
        url = f"{base}/v1/videos/{video_id}/content"
        response = await client.get(url, headers=headers, timeout=timeout)
        
        if response.status_code != 200:
            raise ProxyClientError(
                f"Video content download failed: {response.status_code}",
                status_code=response.status_code,
                response_body=response.text,
            )
        
        return response.content


def extract_video_url(response: Dict[str, Any]) -> Optional[str]:
    """
    Extract video URL from a video generations response.
    
    Handles various response formats:
      - {"data": [{"url": "..."}]}
      - {"data": [{"video_url": "..."}]}
      - {"video_url": "..."}
    """
    # Try data array format (OpenAI-like)
    if "data" in response and len(response["data"]) > 0:
        item = response["data"][0]
        return item.get("url") or item.get("video_url")
    
    # Try direct video_url
    if "video_url" in response:
        return response["video_url"]
    
    return None


# =============================================================================
# Health Check
# =============================================================================

async def check_proxy_health(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """
    Check if the LiteLLM proxy is reachable and responding.

    Tries /health first, then /v1/models as fallback.

    Returns:
        Dict with 'healthy' boolean and 'models' list if available

    Raises:
        ProxyClientError: If proxy is unreachable
    """
    config = get_config()
    base = base_url or config.openai_base_url
    headers = _get_headers(api_key)
    
    result: Dict[str, Any] = {"healthy": False, "models": []}
    
    async with httpx.AsyncClient() as client:
        # Try /health endpoint
        try:
            health_response = await client.get(f"{base}/health", headers=headers, timeout=timeout)
            if health_response.status_code == 200:
                result["healthy"] = True
                result["health_response"] = health_response.json() if health_response.text else {}
        except (httpx.HTTPError, json.JSONDecodeError):
            pass
        
        # Try /v1/models to get available models
        try:
            models_response = await client.get(f"{base}/v1/models", headers=headers, timeout=timeout)
            if models_response.status_code == 200:
                result["healthy"] = True
                models_data = models_response.json()
                if "data" in models_data:
                    result["models"] = [m.get("id") for m in models_data["data"] if m.get("id")]
        except (httpx.HTTPError, json.JSONDecodeError):
            pass
        
        if not result["healthy"]:
            raise ProxyClientError(f"LiteLLM proxy at {base} is not reachable or not responding")
        
        return result

