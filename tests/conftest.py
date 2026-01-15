"""
Pytest configuration and shared fixtures for MenuGen tests.

All tests use the unified config module which loads from:
- Environment variables (OPENAI_BASE_URL, OPENAI_API_KEY, etc.)
- config.json (if present)
- Built-in defaults

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
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_DIR))

# Change to backend directory for config loading
os.chdir(str(BACKEND_DIR))

# Load environment from backend/.env (where the actual config lives)
load_dotenv(BACKEND_DIR / ".env")

# Import config module after setting up paths
from config import load_config, get_config, AppConfig

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

    # Ensure base_url has /v1 suffix for OpenAI-compatible API
    if not base_url.endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"

    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Try /v1/models to get available models (OpenAI-compatible endpoint)
    try:
        models_response = httpx.get(f"{base_url}/models", headers=headers, timeout=timeout)
        if models_response.status_code == 200:
            result["healthy"] = True
            models_data = models_response.json()
            if "data" in models_data:
                result["models"] = [m.get("id") for m in models_data["data"] if m.get("id")]
    except (httpx.HTTPError, Exception) as e:
        result["error"] = str(e)

    # Try /health endpoint as fallback (without /v1)
    if not result["healthy"]:
        health_url = base_url.replace("/v1", "") + "/health"
        try:
            health_response = httpx.get(health_url, headers=headers, timeout=timeout)
            if health_response.status_code == 200:
                result["healthy"] = True
                result["health_response"] = health_response.json() if health_response.text else {}
        except (httpx.HTTPError, Exception) as e:
            if not result["error"]:
                result["error"] = str(e)

    return result


@pytest.fixture(scope="session", autouse=True)
def preflight_check():
    """
    Session-scoped preflight check that runs once before all tests.

    Verifies:
    1. Config module loads successfully
    2. LiteLLM proxy is reachable

    Fails fast with actionable error messages if checks fail.
    """
    # Load config
    load_config()
    cfg = get_config()

    base_url = cfg.openai_base_url
    api_key = cfg.openai_api_key

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
            f"2. Check the proxy URL in backend/.env:\n"
            f"   OPENAI_BASE_URL={base_url}\n\n"
            f"3. Verify the proxy is accepting connections:\n"
            f"   curl {base_url}/v1/models\n"
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
def app_config() -> AppConfig:
    """Fixture providing the loaded AppConfig instance."""
    load_config()
    return get_config()


@pytest.fixture
def litellm_config(app_config):
    """Fixture providing LiteLLM configuration from the config module."""
    return {
        "base_url": app_config.openai_base_url.rstrip("/"),
        "api_key": app_config.openai_api_key,
        "text_model": app_config.description_model,
        "vision_model": app_config.vision_model,
        "image_model": app_config.image_gen_model,
        "video_model": app_config.video_gen_model,
        "description_model": app_config.description_model,
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


@pytest.fixture
def config_outputs_dir(outputs_dir):
    """Fixture providing directory for config test outputs."""
    path = outputs_dir / "config"
    path.mkdir(exist_ok=True)
    return path
