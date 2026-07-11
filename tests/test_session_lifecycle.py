"""Regression tests for isolated generation sessions and lifecycle cleanup."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import WebSocketDisconnect
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

import client_utils
import main


class FakeWebSocket:
    def __init__(self):
        self.messages = []
        self.client_state = SimpleNamespace(name="CONNECTED")
        self.closed = False

    async def send_json(self, data):
        self.messages.append(data)

    async def close(self, **kwargs):
        self.closed = True


@pytest.mark.asyncio
async def test_processing_a_session_does_not_clear_other_session_images(monkeypatch):
    clear_images = Mock()
    monkeypatch.setattr(client_utils, "clear_images_folder", clear_images)
    monkeypatch.setattr(
        main, "parse_menu_image", lambda *args, **kwargs: _async_value({"items": []})
    )
    websocket = FakeWebSocket()
    main.sessions["session-a"] = websocket

    await main.process_menu("session-a", b"image", {})

    clear_images.assert_not_called()


@pytest.mark.asyncio
async def test_processing_closes_websocket_after_done(monkeypatch):
    monkeypatch.setattr(
        main, "parse_menu_image", lambda *args, **kwargs: _async_value({"items": []})
    )
    websocket = FakeWebSocket()
    main.sessions["session-done"] = websocket

    await main.process_menu("session-done", b"image", {})

    assert websocket.closed


@pytest.mark.asyncio
async def test_duplicate_item_names_receive_distinct_session_scoped_urls(monkeypatch):
    async def parse(*args, **kwargs):
        return {"items": [{"name": "Taco"}, {"name": "Taco"}]}

    async def simplify(*args, **kwargs):
        return "A taco."

    async def generate(item, session_config):
        return session_config.get("output_filename", "Taco.png")

    monkeypatch.setattr(main, "parse_menu_image", parse)
    monkeypatch.setattr(main, "simplify_menu_item_description", simplify)
    monkeypatch.setattr(main, "generate_menu_item_image", generate)
    websocket = FakeWebSocket()
    main.sessions["session-b"] = websocket

    await main.process_menu("session-b", b"image", {})

    image_messages = [
        message
        for message in websocket.messages
        if message.get("type") == "image_generated"
    ]
    urls = [message["url"] for message in image_messages]
    assert len(urls) == 2
    assert len(set(urls)) == 2
    assert all(url.startswith("/images/session-b/") for url in urls)
    assert [message["index"] for message in image_messages] == [0, 1]


@pytest.mark.asyncio
async def test_cancel_endpoint_is_idempotent_for_finished_session():
    async with AsyncClient(
        transport=ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        response = await client.delete(
            "/sessions/already-finished",
            headers={"X-MenuGen-Request": "1"},
        )

    assert response.status_code == 200
    assert response.json() == {"status": "cancelled"}


@pytest.mark.asyncio
async def test_cancel_endpoint_cancels_and_forgets_running_job():
    blocker = asyncio.Event()

    async def wait_forever():
        await blocker.wait()

    task = asyncio.create_task(wait_forever())
    main.jobs["running"] = task

    async with AsyncClient(
        transport=ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        response = await client.delete(
            "/sessions/running",
            headers={"X-MenuGen-Request": "1"},
        )
    await asyncio.sleep(0)

    assert response.status_code == 200
    assert task.cancelled()
    assert "running" not in main.jobs


def test_websocket_rejects_untrusted_origin(monkeypatch):
    async def healthy_proxy():
        return True

    monkeypatch.setattr(main, "validate_litellm_connectivity", healthy_proxy)

    with TestClient(main.app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                "/ws/untrusted", headers={"origin": "https://evil.example"}
            ):
                pass


def test_websocket_rejects_unknown_session(monkeypatch):
    async def healthy_proxy():
        return True

    monkeypatch.setattr(main, "validate_litellm_connectivity", healthy_proxy)

    with TestClient(main.app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                "/ws/unknown",
                headers={"origin": "http://localhost:3000"},
            ):
                pass


def test_websocket_disconnect_removes_session(monkeypatch):
    async def healthy_proxy():
        return True

    monkeypatch.setattr(main, "validate_litellm_connectivity", healthy_proxy)
    main.jobs["known"] = Mock()

    with TestClient(main.app) as client:
        with client.websocket_connect(
            "/ws/known", headers={"origin": "http://localhost:3000"}
        ):
            assert "known" in main.sessions

    assert "known" not in main.sessions


async def _async_value(value):
    return value
