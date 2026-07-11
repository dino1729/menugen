"""Regression tests for MenuGen's local security boundaries."""

from io import BytesIO

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

import main
from config import AppConfig, NvidiaModelConfig


def jpeg_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (4, 4), "white").save(output, format="JPEG")
    return output.getvalue()


@pytest.mark.asyncio
async def test_paid_upload_requires_trusted_request_header(monkeypatch):
    async def no_op_process_menu(*args, **kwargs):
        return None

    monkeypatch.setattr(main, "process_menu", no_op_process_menu)

    async with AsyncClient(
        transport=ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/upload_menu/",
            files={"file": ("menu.jpg", b"not-an-image", "image/jpeg")},
        )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_upload_rejects_unsupported_content_type(monkeypatch):
    async def no_op_process_menu(*args, **kwargs):
        return None

    monkeypatch.setattr(main, "process_menu", no_op_process_menu)

    async with AsyncClient(
        transport=ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/upload_menu/",
            headers={"X-MenuGen-Request": "1"},
            files={"file": ("menu.pdf", b"%PDF-1.7", "application/pdf")},
        )

    assert response.status_code == 415


@pytest.mark.asyncio
async def test_upload_rejects_files_larger_than_ten_megabytes(monkeypatch):
    async def no_op_process_menu(*args, **kwargs):
        return None

    monkeypatch.setattr(main, "process_menu", no_op_process_menu)
    oversized = b"\xff\xd8" + (b"0" * (10 * 1024 * 1024))

    async with AsyncClient(
        transport=ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/upload_menu/",
            headers={"X-MenuGen-Request": "1"},
            files={"file": ("large.jpg", oversized, "image/jpeg")},
        )

    assert response.status_code == 413


@pytest.mark.asyncio
async def test_upload_rejects_spoofed_image_content(monkeypatch):
    async def no_op_process_menu(*args, **kwargs):
        return None

    monkeypatch.setattr(main, "process_menu", no_op_process_menu)

    async with AsyncClient(
        transport=ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/upload_menu/",
            headers={"X-MenuGen-Request": "1"},
            files={"file": ("fake.jpg", b"not really an image", "image/jpeg")},
        )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_upload_rejects_model_outside_configured_whitelist(monkeypatch):
    async def no_op_process_menu(*args, **kwargs):
        return None

    monkeypatch.setattr(main, "process_menu", no_op_process_menu)

    async with AsyncClient(
        transport=ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/upload_menu/",
            headers={"X-MenuGen-Request": "1"},
            data={"vision_model": "unapproved-expensive-model"},
            files={"file": ("menu.jpg", jpeg_bytes(), "image/jpeg")},
        )

    assert response.status_code == 422


def test_nvidia_provider_uses_a_configured_nvidia_model_by_default(monkeypatch):
    cfg = AppConfig(image_gen_model="litellm/image-model")
    cfg.nvidia.api_key = "configured"
    cfg.nvidia.models = {"nvidia/image-model": NvidiaModelConfig()}
    monkeypatch.setattr(main, "get_config", lambda: cfg)

    session_config = main.build_session_config(image_provider="nvidia")

    assert session_config["image_gen_model"] == "nvidia/image-model"
