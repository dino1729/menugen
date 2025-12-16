"""
Pytest configuration and shared fixtures for LiteLLM tests.

All tests use the LiteLLM proxy configured via environment variables (no hardcoded defaults):
- LITELLM_BASE_URL: Base URL for LiteLLM proxy (default: http://localhost:4000)
- LITELLM_API_KEY: API key for LiteLLM proxy (required)
- LLM_MODEL: Model for text generation (required)
- VISION_MODEL: Model for image parsing/vision tasks (falls back to LLM_MODEL if not set)
- IMAGE_GEN_MODEL: Model for image generation (required)
- VIDEO_GEN_MODEL: Model for video generation (required for video tests)

Includes a session-scoped preflight check that verifies proxy connectivity
before running any tests, providing clear error messages on failure.
"""
import os
import sys
from pathlib import Path

import httpx
import pytest
from dotenv import load_dotenv

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

# Load environment from root .env
load_dotenv(PROJECT_ROOT / ".env")

# Test output directory
OUTPUTS_DIR = PROJECT_ROOT / "tests" / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


class ProxyPreflightError(Exception):
    """Raised when the LiteLLM proxy preflight check fails."""
    pass


def _check_proxy_health(base_url: str, api_key: str, timeout: float = 10.0) -> dict:
    """
    Synchronous check if the LiteLLM proxy is reachable.
    
    Returns dict with 'healthy' status and 'models' if available.
    Raises ProxyPreflightError with actionable message on failure.
    """
    result = {"healthy": False, "models": [], "error": None}
    
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    # Try /health endpoint
    try:
        health_response = httpx.get(f"{base_url}/health", headers=headers, timeout=timeout)
        if health_response.status_code == 200:
            result["healthy"] = True
            result["health_response"] = health_response.json() if health_response.text else {}
    except (httpx.HTTPError, Exception) as e:
        result["error"] = str(e)
    
    # Try /v1/models to get available models
    if not result["healthy"]:
        try:
            models_response = httpx.get(f"{base_url}/v1/models", headers=headers, timeout=timeout)
            if models_response.status_code == 200:
                result["healthy"] = True
                models_data = models_response.json()
                if "data" in models_data:
                    result["models"] = [m.get("id") for m in models_data["data"] if m.get("id")]
        except (httpx.HTTPError, Exception) as e:
            if not result["error"]:
                result["error"] = str(e)
    
    return result


@pytest.fixture(scope="session", autouse=True)
def preflight_check():
    """
    Session-scoped preflight check that runs once before all tests.
    
    Verifies:
    1. Required environment variables are set
    2. LiteLLM proxy is reachable
    
    Fails fast with actionable error messages if checks fail.
    """
    base_url = os.getenv("LITELLM_BASE_URL", "http://localhost:4000").rstrip("/")
    api_key = os.getenv("LITELLM_API_KEY", "")
    text_model = os.getenv("LLM_MODEL")
    
    # Check required env vars
    missing_vars = []
    if not text_model:
        missing_vars.append("LLM_MODEL")
    
    if missing_vars:
        raise ProxyPreflightError(
            f"\n{'='*60}\n"
            f"PREFLIGHT CHECK FAILED: Missing required environment variables\n"
            f"{'='*60}\n"
            f"Missing: {', '.join(missing_vars)}\n\n"
            f"Please set these in your .env file:\n"
            f"  LLM_MODEL=<your-text-model>  (e.g., gpt-4o, claude-3-sonnet)\n"
            f"  IMAGE_GEN_MODEL=<your-image-model>  (e.g., dall-e-3, gemini-3-pro-image-preview)\n"
            f"  VIDEO_GEN_MODEL=<your-video-model>  (e.g., veo-3.0-generate-001)\n"
            f"{'='*60}"
        )
    
    # Check proxy connectivity
    print(f"\n🔍 Preflight: Checking LiteLLM proxy at {base_url}...")
    health = _check_proxy_health(base_url, api_key)
    
    if not health["healthy"]:
        error_details = health.get("error", "Unknown error")
        raise ProxyPreflightError(
            f"\n{'='*60}\n"
            f"PREFLIGHT CHECK FAILED: LiteLLM proxy not reachable\n"
            f"{'='*60}\n"
            f"URL: {base_url}\n"
            f"Error: {error_details}\n\n"
            f"Troubleshooting steps:\n"
            f"1. Ensure your LiteLLM proxy is running:\n"
            f"   docker ps | grep litellm\n"
            f"   or: litellm --config /path/to/config.yaml\n\n"
            f"2. Check the proxy URL in your .env:\n"
            f"   LITELLM_BASE_URL={base_url}\n\n"
            f"3. Verify the proxy is accepting connections:\n"
            f"   curl {base_url}/health\n"
            f"{'='*60}"
        )
    
    # Report success
    model_count = len(health.get("models", []))
    if model_count > 0:
        print(f"✓ Preflight: Proxy healthy, {model_count} models available")
    else:
        print(f"✓ Preflight: Proxy healthy")
    
    return health


@pytest.fixture
def litellm_config():
    """Fixture providing LiteLLM configuration from environment (no hardcoded model defaults)."""
    text_model = os.getenv("LLM_MODEL")
    if not text_model:
        pytest.skip("LLM_MODEL environment variable not set")
    
    return {
        "base_url": os.getenv("LITELLM_BASE_URL", "http://localhost:4000").rstrip("/"),
        "api_key": os.getenv("LITELLM_API_KEY"),
        "text_model": text_model,
        "vision_model": os.getenv("VISION_MODEL") or text_model,  # Falls back to text_model
        "image_model": os.getenv("IMAGE_GEN_MODEL"),
        "video_model": os.getenv("VIDEO_GEN_MODEL"),
    }


@pytest.fixture
def outputs_dir():
    """Fixture providing the test outputs directory path."""
    return OUTPUTS_DIR


@pytest.fixture
def text_outputs_dir(outputs_dir):
    """Fixture providing directory for text generation outputs."""
    path = outputs_dir / "text"
    path.mkdir(exist_ok=True)
    return path


@pytest.fixture
def image_outputs_dir(outputs_dir):
    """Fixture providing directory for image generation outputs."""
    path = outputs_dir / "images"
    path.mkdir(exist_ok=True)
    return path


@pytest.fixture
def video_outputs_dir(outputs_dir):
    """Fixture providing directory for video generation outputs."""
    path = outputs_dir / "videos"
    path.mkdir(exist_ok=True)
    return path
