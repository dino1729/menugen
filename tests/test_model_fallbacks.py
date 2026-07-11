"""Unit tests for MenuGen's role-specific NVIDIA NIM fallbacks."""

import pytest

import litellm_client


@pytest.mark.asyncio
async def test_menu_parsing_attaches_vision_nim_fallback(app_config, monkeypatch):
    calls = []
    app_config.nvidia_nim_api_key = "nim-key"

    async def fake_chat_completions(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise litellm_client.ProxyClientError("primary failed", status_code=503)
        return {"choices": [{"message": {"content": '{"items": []}'}}]}

    monkeypatch.setattr(litellm_client, "chat_completions", fake_chat_completions)

    await litellm_client.parse_menu_image_litellm(b"image")

    assert calls[0]["model"] == app_config.vision_model
    assert calls[1]["model"] == app_config.nim_vision_fallback_model
    assert calls[1]["base_url"] == app_config.nvidia_nim_base_url
    assert calls[1]["api_key"] == "nim-key"


@pytest.mark.asyncio
async def test_description_attaches_text_nim_fallback(app_config, monkeypatch):
    calls = []
    app_config.nvidia_nim_api_key = "nim-key"

    async def fake_chat_completions(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise litellm_client.ProxyClientError("primary failed", status_code=503)
        return {"choices": [{"message": {"content": "A delicious tomato soup."}}]}

    monkeypatch.setattr(litellm_client, "chat_completions", fake_chat_completions)

    await litellm_client.simplify_menu_item_description_litellm(
        {"name": "Tomato Soup"}
    )

    assert calls[0]["model"] == app_config.description_model
    assert calls[1]["model"] == app_config.nim_text_fallback_model
    assert calls[1]["base_url"] == app_config.nvidia_nim_base_url
    assert calls[1]["api_key"] == "nim-key"
