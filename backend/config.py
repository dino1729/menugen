"""
Configuration management for MenuGen backend.

Loads configuration from:
1. config.json (curated model whitelists, NVIDIA params, retry settings)
2. Environment variables (API keys, URLs, default model overrides)

Environment variables take precedence over config.json values.
Validates LiteLLM proxy connectivity on startup (fail-fast).
"""
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("menugen.config")

# Configuration file paths
CONFIG_DIR = Path(__file__).parent
CONFIG_FILE = CONFIG_DIR / "config.json"
CONFIG_EXAMPLE_FILE = CONFIG_DIR / "config.example.json"


@dataclass
class RetryConfig:
    """Retry/backoff configuration for API calls."""
    max_retries: int = 5
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 30.0
    jitter_factor: float = 0.1
    retry_status_codes: List[int] = field(
        default_factory=lambda: [429, 408, 500, 502, 503, 504]
    )


@dataclass
class NvidiaModelConfig:
    """Configuration for a specific NVIDIA model."""
    api_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NvidiaConfig:
    """NVIDIA provider configuration."""
    base_url: str = "https://ai.api.nvidia.com/v1/genai"
    api_key: Optional[str] = None
    models: Dict[str, NvidiaModelConfig] = field(default_factory=dict)


@dataclass
class AppConfig:
    """Main application configuration."""
    # API endpoints (using OPENAI_* prefix for compatibility)
    openai_base_url: str = "http://localhost:4000"
    openai_api_key: str = ""

    # Default models
    vision_model: str = "gpt-4o"
    description_model: str = "gemini-3-flash-preview"
    image_gen_model: str = "gemini-3-pro-image-preview"
    video_gen_model: str = "veo-3.1-generate-001"

    # Provider selection
    image_provider: str = "litellm"

    # Curated model whitelists
    vision_models: List[str] = field(default_factory=list)
    text_models: List[str] = field(default_factory=list)
    image_models: List[str] = field(default_factory=list)
    video_models: List[str] = field(default_factory=list)

    # Sub-configs
    nvidia: NvidiaConfig = field(default_factory=NvidiaConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)

    # Runtime state
    litellm_healthy: bool = False
    available_models: List[str] = field(default_factory=list)


# Global config instance
_config: Optional[AppConfig] = None


def load_config() -> AppConfig:
    """
    Load configuration from config.json and environment variables.

    Priority: Environment variables > config.json > defaults

    Returns:
        Loaded AppConfig instance
    """
    global _config

    config = AppConfig()

    # Load from config.json if it exists
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                data = json.load(f)

            # Load defaults section
            defaults = data.get("defaults", {})
            if "vision_model" in defaults:
                config.vision_model = defaults["vision_model"]
            if "description_model" in defaults:
                config.description_model = defaults["description_model"]
            if "image_gen_model" in defaults:
                config.image_gen_model = defaults["image_gen_model"]
            if "video_gen_model" in defaults:
                config.video_gen_model = defaults["video_gen_model"]

            # Load whitelists
            whitelists = data.get("whitelists", {})
            config.vision_models = whitelists.get("vision_models", [])
            config.text_models = whitelists.get("text_models", [])
            config.image_models = whitelists.get("image_models", [])
            config.video_models = whitelists.get("video_models", [])

            # Load NVIDIA config
            nvidia_data = data.get("nvidia", {})
            config.nvidia.base_url = nvidia_data.get("base_url", config.nvidia.base_url)
            for model_id, model_data in nvidia_data.get("models", {}).items():
                config.nvidia.models[model_id] = NvidiaModelConfig(
                    api_params=model_data.get("api_params", {})
                )

            # Load retry config
            retry_data = data.get("retry", {})
            config.retry = RetryConfig(
                max_retries=retry_data.get("max_retries", 5),
                initial_backoff_seconds=retry_data.get("initial_backoff_seconds", 1.0),
                max_backoff_seconds=retry_data.get("max_backoff_seconds", 30.0),
                jitter_factor=retry_data.get("jitter_factor", 0.1),
                retry_status_codes=retry_data.get(
                    "retry_status_codes", [429, 408, 500, 502, 503, 504]
                ),
            )

            logger.info(f"Loaded configuration from {CONFIG_FILE}")
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in config.json: {e}, using defaults")
        except Exception as e:
            logger.warning(f"Failed to load config.json: {e}, using defaults")
    else:
        logger.info(f"No config.json found at {CONFIG_FILE}, using defaults")

    # Environment variable overrides (highest priority)
    # Handle OPENAI_BASE_URL with or without /v1 suffix
    base_url = os.getenv("OPENAI_BASE_URL", config.openai_base_url)
    config.openai_base_url = base_url.rstrip("/").rstrip("/v1").rstrip("/")

    config.openai_api_key = os.getenv("OPENAI_API_KEY", config.openai_api_key)

    # Model overrides from environment
    config.vision_model = os.getenv("VISION_MODEL", config.vision_model)
    config.description_model = os.getenv("DESCRIPTION_MODEL", config.description_model)
    config.image_gen_model = os.getenv("IMAGE_GEN_MODEL", config.image_gen_model)
    config.video_gen_model = os.getenv("VIDEO_GEN_MODEL", config.video_gen_model)

    # Provider selection
    config.image_provider = os.getenv("IMAGE_PROVIDER", config.image_provider).lower()

    # NVIDIA API key
    config.nvidia.api_key = os.getenv("NVIDIA_API_KEY")

    _config = config
    return config


def get_config() -> AppConfig:
    """
    Get the current configuration instance.

    Loads config if not already loaded.

    Returns:
        Current AppConfig instance
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_base_url_with_v1() -> str:
    """
    Get the OpenAI-compatible base URL with /v1 suffix.

    Returns:
        Base URL ending with /v1
    """
    config = get_config()
    base = config.openai_base_url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


async def validate_litellm_connectivity() -> bool:
    """
    Validate LiteLLM proxy is reachable. Called at startup.

    Tries /health first, then /v1/models as fallback.
    Updates config with available models on success.

    Returns:
        True if proxy is healthy

    Raises:
        RuntimeError: If proxy is not reachable (fail-fast)
    """
    config = get_config()
    base_url = config.openai_base_url

    headers = {"Content-Type": "application/json"}
    if config.openai_api_key:
        headers["Authorization"] = f"Bearer {config.openai_api_key}"

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Try /health endpoint first
        try:
            resp = await client.get(f"{base_url}/health", headers=headers)
            if resp.status_code == 200:
                logger.info(f"LiteLLM proxy at {base_url} is healthy (via /health)")
                config.litellm_healthy = True
        except Exception as e:
            logger.debug(f"Health check failed: {e}")

        # Try /v1/models to get available models
        try:
            models_url = f"{base_url}/v1/models"
            resp = await client.get(models_url, headers=headers)
            if resp.status_code == 200:
                config.litellm_healthy = True
                data = resp.json()
                config.available_models = [
                    m.get("id") for m in data.get("data", []) if m.get("id")
                ]
                logger.info(
                    f"LiteLLM proxy at {base_url} is reachable, "
                    f"fetched {len(config.available_models)} models"
                )
                return True
        except Exception as e:
            logger.error(f"Models endpoint check failed: {e}")

    if not config.litellm_healthy:
        raise RuntimeError(
            f"LiteLLM proxy at {base_url} is not reachable. "
            f"Please ensure LiteLLM is running and OPENAI_BASE_URL is correct."
        )

    return True


def get_nvidia_model_params(model_id: str) -> Dict[str, Any]:
    """
    Get API parameters for a NVIDIA model from config.

    If model is defined in config.json, returns those params.
    Otherwise, infers defaults based on model type (FLUX vs SD).

    Args:
        model_id: Model identifier (e.g., 'black-forest-labs/flux.1-schnell')

    Returns:
        Dictionary of API parameters for the model
    """
    config = get_config()

    # Check if model is explicitly configured
    if model_id in config.nvidia.models:
        return config.nvidia.models[model_id].api_params.copy()

    # Infer defaults based on model name
    model_lower = model_id.lower() if model_id else ""

    if "flux" in model_lower:
        # FLUX models use different params
        return {
            "cfg_scale": 0,
            "width": 1024,
            "height": 1024,
            "seed": 0,
            "steps": 4 if "schnell" in model_lower else 28,
        }
    else:
        # Stable Diffusion models
        return {
            "cfg_scale": 5,
            "aspect_ratio": "1:1",
            "seed": 0,
            "steps": 50,
            "negative_prompt": "",
        }


def get_retry_status_codes() -> set:
    """Get the set of HTTP status codes that should trigger retry."""
    config = get_config()
    return set(config.retry.retry_status_codes)
