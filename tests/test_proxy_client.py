"""
Unit tests for litellm_proxy_client module.

Tests cover:
- Helper functions (extract_chat_content, extract_json_from_text, etc.)
- Request retry logic
- Error handling
- Response parsing
"""
import base64
import json
from unittest import mock

import httpx
import pytest

from litellm_proxy_client import (
    ProxyClientError,
    chat_completions,
    extract_chat_content,
    extract_json_from_text,
    image_generations,
    extract_image_bytes,
    video_generations,
    extract_video_url,
    check_proxy_health,
    _get_headers,
    _calculate_backoff,
)


class TestExtractChatContent:
    """Tests for extract_chat_content function."""

    def test_extracts_content_from_valid_response(self):
        """Test extracting content from a standard OpenAI-like response."""
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "  Hello, World!  "
                    }
                }
            ]
        }
        content = extract_chat_content(response)
        assert content == "Hello, World!"

    def test_raises_on_missing_choices(self):
        """Test that missing choices raises ProxyClientError."""
        response = {"data": "something"}
        with pytest.raises(ProxyClientError):
            extract_chat_content(response)

    def test_raises_on_empty_choices(self):
        """Test that empty choices raises ProxyClientError."""
        response = {"choices": []}
        with pytest.raises(ProxyClientError):
            extract_chat_content(response)

    def test_raises_on_missing_message(self):
        """Test that missing message raises ProxyClientError."""
        response = {"choices": [{"index": 0}]}
        with pytest.raises(ProxyClientError):
            extract_chat_content(response)

    def test_raises_on_missing_content(self):
        """Test that missing content raises ProxyClientError."""
        response = {"choices": [{"message": {"role": "assistant"}}]}
        with pytest.raises(ProxyClientError):
            extract_chat_content(response)


class TestExtractJsonFromText:
    """Tests for extract_json_from_text function."""

    def test_parses_raw_json(self):
        """Test parsing raw JSON without markdown."""
        text = '{"name": "Pizza", "price": 12.99}'
        result = extract_json_from_text(text)
        assert result["name"] == "Pizza"
        assert result["price"] == 12.99

    def test_parses_json_with_markdown_fence(self):
        """Test parsing JSON wrapped in markdown fence."""
        text = '```json\n{"name": "Burger"}\n```'
        result = extract_json_from_text(text)
        assert result["name"] == "Burger"

    def test_parses_json_with_plain_fence(self):
        """Test parsing JSON wrapped in plain fence."""
        text = '```\n{"items": [1, 2, 3]}\n```'
        result = extract_json_from_text(text)
        assert result["items"] == [1, 2, 3]

    def test_extracts_json_from_surrounding_text(self):
        """Test extracting JSON embedded in other text."""
        text = 'Here is the data: {"key": "value"} Hope this helps!'
        result = extract_json_from_text(text)
        assert result["key"] == "value"

    def test_raises_on_invalid_json(self):
        """Test that invalid JSON raises ProxyClientError."""
        text = "This is not JSON at all"
        with pytest.raises(ProxyClientError):
            extract_json_from_text(text)

    def test_handles_nested_json(self):
        """Test parsing nested JSON objects."""
        text = '{"user": {"name": "Alice", "scores": [10, 20]}}'
        result = extract_json_from_text(text)
        assert result["user"]["name"] == "Alice"
        assert result["user"]["scores"] == [10, 20]


class TestExtractVideoUrl:
    """Tests for extract_video_url function."""

    def test_extracts_url_from_data_array(self):
        """Test extracting URL from OpenAI-like data array."""
        response = {"data": [{"url": "https://example.com/video.mp4"}]}
        url = extract_video_url(response)
        assert url == "https://example.com/video.mp4"

    def test_extracts_video_url_from_data_array(self):
        """Test extracting video_url from data array."""
        response = {"data": [{"video_url": "https://example.com/clip.mp4"}]}
        url = extract_video_url(response)
        assert url == "https://example.com/clip.mp4"

    def test_extracts_direct_video_url(self):
        """Test extracting direct video_url field."""
        response = {"video_url": "https://example.com/direct.mp4"}
        url = extract_video_url(response)
        assert url == "https://example.com/direct.mp4"

    def test_returns_none_for_empty_response(self):
        """Test that empty response returns None."""
        response = {}
        url = extract_video_url(response)
        assert url is None

    def test_returns_none_for_empty_data_array(self):
        """Test that empty data array returns None."""
        response = {"data": []}
        url = extract_video_url(response)
        assert url is None


class TestGetHeaders:
    """Tests for _get_headers helper function."""

    def test_includes_content_type(self, app_config):
        """Test that Content-Type header is included."""
        headers = _get_headers()
        assert headers["Content-Type"] == "application/json"

    def test_includes_accept(self, app_config):
        """Test that Accept header is included."""
        headers = _get_headers()
        assert headers["Accept"] == "application/json"

    def test_includes_authorization_when_key_provided(self, app_config):
        """Test that Authorization header is set when API key provided."""
        headers = _get_headers(api_key="test-key-123")
        assert headers["Authorization"] == "Bearer test-key-123"

    def test_uses_config_key_when_not_provided(self, app_config):
        """Test that config API key is used when not explicitly provided."""
        headers = _get_headers()
        if app_config.openai_api_key:
            assert "Authorization" in headers


class TestCalculateBackoff:
    """Tests for _calculate_backoff function."""

    @pytest.mark.asyncio
    async def test_respects_retry_after_header(self, app_config):
        """Test that Retry-After header is respected."""
        mock_response = mock.MagicMock()
        mock_response.headers = {"Retry-After": "5.0"}

        backoff = await _calculate_backoff(0, mock_response)
        assert backoff == 5.0

    @pytest.mark.asyncio
    async def test_exponential_backoff(self, app_config):
        """Test that backoff increases exponentially."""
        backoff_0 = await _calculate_backoff(0)
        backoff_1 = await _calculate_backoff(1)
        backoff_2 = await _calculate_backoff(2)

        # Backoff should increase (allowing for jitter)
        assert backoff_1 > backoff_0 * 0.9  # Allow some variance for jitter
        assert backoff_2 > backoff_1 * 0.9

    @pytest.mark.asyncio
    async def test_backoff_has_jitter(self, app_config):
        """Test that backoff includes jitter (randomness)."""
        # Run multiple times and check for variance
        backoffs = [await _calculate_backoff(1) for _ in range(5)]
        # With jitter, not all values should be exactly the same
        assert len(set(round(b, 2) for b in backoffs)) > 1


class TestExtractImageBytes:
    """Tests for extract_image_bytes function."""

    @pytest.mark.asyncio
    async def test_extracts_b64_json(self):
        """Test extracting base64 encoded image data."""
        image_data = base64.b64encode(b"fake image data").decode()
        response = {"data": [{"b64_json": image_data}]}

        img_bytes, url = await extract_image_bytes(response)
        assert img_bytes == b"fake image data"
        assert url is None

    @pytest.mark.asyncio
    async def test_handles_data_url(self):
        """Test extracting image from data: URL."""
        image_data = base64.b64encode(b"test image").decode()
        data_url = f"data:image/png;base64,{image_data}"
        response = {"data": [{"url": data_url}]}

        img_bytes, returned_url = await extract_image_bytes(response)
        assert img_bytes == b"test image"
        assert returned_url == data_url

    @pytest.mark.asyncio
    async def test_raises_on_empty_response(self):
        """Test that empty response raises error."""
        response = {}
        with pytest.raises(ProxyClientError):
            await extract_image_bytes(response)

    @pytest.mark.asyncio
    async def test_raises_on_empty_data_array(self):
        """Test that empty data array raises error."""
        response = {"data": []}
        with pytest.raises(ProxyClientError):
            await extract_image_bytes(response)

    @pytest.mark.asyncio
    async def test_raises_on_missing_url_and_b64(self):
        """Test that missing url and b64_json raises error."""
        response = {"data": [{"something": "else"}]}
        with pytest.raises(ProxyClientError):
            await extract_image_bytes(response)


class TestProxyClientError:
    """Tests for ProxyClientError exception class."""

    def test_error_with_message_only(self):
        """Test creating error with message only."""
        error = ProxyClientError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.status_code is None
        assert error.response_body is None

    def test_error_with_status_code(self):
        """Test creating error with status code."""
        error = ProxyClientError("Failed", status_code=500)
        assert error.status_code == 500

    def test_error_with_response_body(self):
        """Test creating error with response body."""
        error = ProxyClientError("Failed", response_body='{"error": "bad"}')
        assert error.response_body == '{"error": "bad"}'

    def test_error_with_all_fields(self):
        """Test creating error with all fields."""
        error = ProxyClientError(
            "API Error",
            status_code=429,
            response_body='{"message": "rate limited"}'
        )
        assert str(error) == "API Error"
        assert error.status_code == 429
        assert error.response_body == '{"message": "rate limited"}'


class TestChatCompletionsIntegration:
    """Integration tests for chat_completions (requires running proxy)."""

    @pytest.mark.asyncio
    async def test_simple_completion(self, litellm_config):
        """Test a simple chat completion request."""
        response = await chat_completions(
            model=litellm_config["text_model"],
            messages=[{"role": "user", "content": "Say 'test' and nothing else"}],
            base_url=litellm_config["base_url"],
            api_key=litellm_config["api_key"],
            max_tokens=10,
        )

        assert "choices" in response
        assert len(response["choices"]) > 0
        # Check response structure (content might be None for some models)
        message = response["choices"][0].get("message", {})
        assert "content" in message or "role" in message

    @pytest.mark.asyncio
    async def test_completion_with_system_message(self, litellm_config):
        """Test completion with system message."""
        response = await chat_completions(
            model=litellm_config["text_model"],
            messages=[
                {"role": "system", "content": "You only respond with 'OK'."},
                {"role": "user", "content": "Hello"},
            ],
            base_url=litellm_config["base_url"],
            api_key=litellm_config["api_key"],
            max_tokens=10,
        )

        # Verify response structure
        assert "choices" in response
        assert len(response["choices"]) > 0


class TestCheckProxyHealth:
    """Tests for check_proxy_health function."""

    @pytest.mark.asyncio
    async def test_healthy_proxy(self, litellm_config):
        """Test checking health of a running proxy."""
        result = await check_proxy_health(
            base_url=litellm_config["base_url"],
            api_key=litellm_config["api_key"],
        )

        assert result["healthy"] is True

    @pytest.mark.asyncio
    async def test_returns_models_list(self, litellm_config):
        """Test that health check returns available models."""
        result = await check_proxy_health(
            base_url=litellm_config["base_url"],
            api_key=litellm_config["api_key"],
        )

        assert "models" in result
        # Should have some models if proxy is configured
        if result["models"]:
            assert isinstance(result["models"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
