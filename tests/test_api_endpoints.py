"""
Tests for FastAPI endpoints in main.py.

Tests cover:
- /config endpoint
- /models endpoint
- /health endpoint
- Static file serving
"""
import pytest
from httpx import AsyncClient, ASGITransport

# Import the FastAPI app
from main import app


@pytest.fixture
def anyio_backend():
    return 'asyncio'


class TestConfigEndpoint:
    """Tests for the /config endpoint."""

    @pytest.mark.asyncio
    async def test_config_returns_200(self, app_config):
        """Test that /config returns 200 OK."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/config")
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_config_returns_json(self, app_config):
        """Test that /config returns valid JSON."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/config")
            data = response.json()
            assert isinstance(data, dict)

    @pytest.mark.asyncio
    async def test_config_contains_required_fields(self, app_config):
        """Test that /config response contains required fields."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/config")
            data = response.json()

            # Check required fields
            assert "image_provider" in data
            assert "vision_model" in data
            assert "description_model" in data
            assert "image_gen_model" in data
            assert "video_gen_model" in data

    @pytest.mark.asyncio
    async def test_config_contains_whitelists(self, app_config):
        """Test that /config response contains whitelists."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/config")
            data = response.json()

            assert "whitelists" in data
            whitelists = data["whitelists"]
            assert "vision" in whitelists
            assert "text" in whitelists
            assert "image" in whitelists
            assert "video" in whitelists

    @pytest.mark.asyncio
    async def test_config_nvidia_available_field(self, app_config):
        """Test that /config includes nvidia_available field."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/config")
            data = response.json()

            assert "nvidia_available" in data
            assert isinstance(data["nvidia_available"], bool)


class TestModelsEndpoint:
    """Tests for the /models endpoint."""

    @pytest.mark.asyncio
    async def test_models_returns_200(self, app_config):
        """Test that /models returns 200 OK."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/models")
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_models_returns_json(self, app_config):
        """Test that /models returns valid JSON."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/models")
            data = response.json()
            assert isinstance(data, dict)

    @pytest.mark.asyncio
    async def test_models_contains_success_field(self, app_config):
        """Test that /models response has success field."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/models")
            data = response.json()
            assert "success" in data

    @pytest.mark.asyncio
    async def test_models_contains_models_object(self, app_config):
        """Test that /models response has models object."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/models")
            data = response.json()

            if data.get("success"):
                assert "models" in data
                models = data["models"]
                # Check model category keys
                assert "all" in models
                assert "vision" in models
                assert "image" in models
                assert "video" in models
                assert "text" in models


class TestHealthEndpoint:
    """Tests for the /health endpoint (via root /)."""

    @pytest.mark.asyncio
    async def test_root_can_serve_as_health_check(self):
        """Test that / endpoint can be used as a health check."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/")
            # Root endpoint returns 200 when server is healthy
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_config_endpoint_as_health_check(self):
        """Test that /config can indicate server health."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/config")
            # Config endpoint returns 200 when server is healthy
            assert response.status_code == 200


class TestRootEndpoint:
    """Tests for the root / endpoint."""

    @pytest.mark.asyncio
    async def test_root_returns_200(self):
        """Test that / returns 200 OK."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/")
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_root_returns_welcome_message(self):
        """Test that / returns a welcome message."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/")
            data = response.json()
            assert "message" in data


class TestUploadEndpoints:
    """Tests for upload endpoints (basic validation only)."""

    @pytest.mark.asyncio
    async def test_upload_menu_requires_file(self):
        """Test that /upload_menu/ requires a file."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/upload_menu/")
            # Should return 422 Unprocessable Entity for missing file
            assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_parse_menu_only_requires_file(self):
        """Test that /parse_menu_only/ requires a file."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/parse_menu_only/")
            # Should return 422 Unprocessable Entity for missing file
            assert response.status_code == 422


class TestCORSConfiguration:
    """Tests for CORS configuration."""

    @pytest.mark.asyncio
    async def test_cors_allows_any_origin(self):
        """Test that CORS allows requests from any origin."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.options(
                "/config",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "GET",
                }
            )
            # FastAPI's default CORS returns 200 for preflight
            assert response.status_code in (200, 204)


class TestStaticFiles:
    """Tests for static file serving."""

    @pytest.mark.asyncio
    async def test_images_endpoint_exists(self):
        """Test that /images/ endpoint is mounted."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Requesting a non-existent image should return 404, not 500
            response = await client.get("/images/nonexistent.png")
            assert response.status_code in (404, 200)  # 404 if file not found


class TestConfigModelValues:
    """Tests to verify config values are correctly reflected in API."""

    @pytest.mark.asyncio
    async def test_config_values_match_app_config(self, app_config):
        """Test that /config values match loaded AppConfig."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/config")
            data = response.json()

            assert data["vision_model"] == app_config.vision_model
            assert data["description_model"] == app_config.description_model
            assert data["image_gen_model"] == app_config.image_gen_model
            assert data["video_gen_model"] == app_config.video_gen_model
            assert data["image_provider"] == app_config.image_provider


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
