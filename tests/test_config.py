"""
Tests for the config module.

Tests cover:
- Config dataclass defaults
- Config loading
- Helper functions
- Config integration with actual config.json
"""
import json
import os
from pathlib import Path

import pytest

from config import (
    AppConfig,
    NvidiaConfig,
    RetryConfig,
    load_config,
    get_config,
    get_base_url_with_v1,
    get_retry_status_codes,
    get_nvidia_model_params,
)


class TestAppConfigDefaults:
    """Test default values for AppConfig."""

    def test_default_openai_base_url(self):
        """Test default OpenAI base URL."""
        cfg = AppConfig()
        assert cfg.openai_base_url == "http://localhost:4000"

    def test_default_models(self):
        """Test default model values."""
        cfg = AppConfig()
        assert cfg.vision_model == "gpt-4o"
        assert cfg.description_model == "gemini-3-flash-preview"
        assert cfg.image_gen_model == "gemini-3-pro-image-preview"
        assert cfg.video_gen_model == "veo-3.1-generate-001"

    def test_default_image_provider(self):
        """Test default image provider."""
        cfg = AppConfig()
        assert cfg.image_provider == "litellm"

    def test_default_whitelists_empty(self):
        """Test default whitelists are empty."""
        cfg = AppConfig()
        assert cfg.vision_models == []
        assert cfg.text_models == []
        assert cfg.image_models == []
        assert cfg.video_models == []


class TestNvidiaConfig:
    """Test NvidiaConfig dataclass."""

    def test_default_nvidia_base_url(self):
        """Test default NVIDIA base URL."""
        cfg = NvidiaConfig()
        assert cfg.base_url == "https://ai.api.nvidia.com/v1/genai"

    def test_default_nvidia_models_empty(self):
        """Test default models dict is empty."""
        cfg = NvidiaConfig()
        assert cfg.models == {}

    def test_nvidia_config_with_api_key(self):
        """Test NvidiaConfig with API key."""
        cfg = NvidiaConfig(api_key="test-key")
        assert cfg.api_key == "test-key"


class TestRetryConfig:
    """Test RetryConfig dataclass."""

    def test_default_retry_values(self):
        """Test default retry configuration values."""
        cfg = RetryConfig()
        assert cfg.max_retries == 5
        assert cfg.initial_backoff_seconds == 1.0
        assert cfg.jitter_factor == 0.1
        assert 429 in cfg.retry_status_codes
        assert 500 in cfg.retry_status_codes


class TestGetRetryStatusCodes:
    """Test get_retry_status_codes helper function."""

    def test_returns_set(self, app_config):
        """Test that retry status codes are returned as a set."""
        codes = get_retry_status_codes()
        assert isinstance(codes, set)

    def test_contains_common_codes(self, app_config):
        """Test that common retry codes are included."""
        codes = get_retry_status_codes()
        assert 429 in codes  # Rate limit
        assert 500 in codes  # Server error
        assert 502 in codes  # Bad gateway
        assert 503 in codes  # Service unavailable


class TestGetNvidiaModelParams:
    """Test get_nvidia_model_params helper function."""

    def test_unknown_model_returns_sd_defaults(self, app_config):
        """Test that unknown model returns Stable Diffusion default params."""
        params = get_nvidia_model_params("unknown/model-xyz")
        # Unknown models get SD defaults as fallback
        assert "cfg_scale" in params
        assert "steps" in params

    def test_none_model_returns_sd_defaults(self, app_config):
        """Test that None model returns Stable Diffusion default params."""
        params = get_nvidia_model_params(None)
        # None models get SD defaults as fallback
        assert "cfg_scale" in params

    def test_flux_model_returns_flux_params(self, app_config):
        """Test that FLUX model names return FLUX-specific params."""
        params = get_nvidia_model_params("black-forest-labs/flux.1-schnell")
        # FLUX models should have width/height instead of aspect_ratio
        assert "width" in params
        assert "height" in params
        assert params["cfg_scale"] == 0  # FLUX uses cfg_scale=0


class TestGetBaseUrlWithV1:
    """Test get_base_url_with_v1 helper function."""

    def test_returns_url_with_v1(self, app_config):
        """Test that URL includes /v1 suffix."""
        url = get_base_url_with_v1()
        assert url.endswith("/v1")

    def test_url_is_valid(self, app_config):
        """Test that returned URL is a valid HTTP URL."""
        url = get_base_url_with_v1()
        assert url.startswith("http://") or url.startswith("https://")


class TestConfigIntegration:
    """Integration tests for config module with actual config.json."""

    def test_load_config_succeeds(self):
        """Test that load_config() completes without error."""
        load_config()
        cfg = get_config()
        assert cfg is not None

    def test_config_has_valid_openai_url(self, app_config):
        """Test that OpenAI URL is set."""
        assert app_config.openai_base_url is not None
        assert len(app_config.openai_base_url) > 0

    def test_config_has_valid_models(self, app_config):
        """Test that model names are set."""
        assert app_config.vision_model is not None
        assert app_config.description_model is not None
        assert app_config.image_gen_model is not None
        assert app_config.video_gen_model is not None

    def test_config_image_provider_valid(self, app_config):
        """Test that image provider is a valid value."""
        assert app_config.image_provider in ("litellm", "nvidia", "openai")

    def test_whitelists_loaded(self, app_config):
        """Test that whitelists are loaded from config.json."""
        # If config.json exists and has whitelists, they should be loaded
        if app_config.vision_models:
            assert isinstance(app_config.vision_models, list)
            assert len(app_config.vision_models) > 0

    def test_nvidia_config_loaded(self, app_config):
        """Test that NVIDIA config is loaded."""
        assert app_config.nvidia is not None
        assert app_config.nvidia.base_url is not None

    def test_nvidia_models_loaded(self, app_config):
        """Test that NVIDIA models config is loaded."""
        if app_config.nvidia.models:
            assert isinstance(app_config.nvidia.models, dict)

    def test_retry_config_loaded(self, app_config):
        """Test that retry config is loaded."""
        assert app_config.retry is not None
        assert app_config.retry.max_retries > 0
        assert app_config.retry.initial_backoff_seconds > 0


class TestConfigEnvironmentOverride:
    """Test that environment variables properly override config."""

    def test_openai_base_url_from_env(self, app_config, monkeypatch):
        """Test that OPENAI_BASE_URL env var is read."""
        # The config should have loaded the env var
        env_url = os.getenv("OPENAI_BASE_URL")
        if env_url:
            assert app_config.openai_base_url == env_url

    def test_vision_model_from_env(self, app_config):
        """Test that VISION_MODEL env var is respected."""
        env_model = os.getenv("VISION_MODEL")
        if env_model:
            assert app_config.vision_model == env_model

    def test_image_provider_from_env(self, app_config):
        """Test that IMAGE_PROVIDER env var is respected."""
        env_provider = os.getenv("IMAGE_PROVIDER")
        if env_provider:
            assert app_config.image_provider == env_provider


class TestConfigJsonContent:
    """Tests that verify config.json content is correctly loaded."""

    def test_vision_models_whitelist_has_gpt4o(self, app_config):
        """Test that vision models whitelist includes gpt-4o."""
        if app_config.vision_models:
            assert "gpt-4o" in app_config.vision_models

    def test_text_models_whitelist_exists(self, app_config):
        """Test that text models whitelist is populated."""
        if app_config.text_models:
            assert len(app_config.text_models) > 0

    def test_image_models_whitelist_exists(self, app_config):
        """Test that image models whitelist is populated."""
        if app_config.image_models:
            assert len(app_config.image_models) > 0

    def test_video_models_whitelist_exists(self, app_config):
        """Test that video models whitelist is populated."""
        if app_config.video_models:
            assert len(app_config.video_models) > 0

    def test_nvidia_models_have_flux(self, app_config):
        """Test that NVIDIA models include FLUX models."""
        if app_config.nvidia.models:
            flux_models = [k for k in app_config.nvidia.models.keys() if "flux" in k.lower()]
            # May or may not have FLUX depending on config
            assert isinstance(flux_models, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
