"""
Tests for text generation using LiteLLM proxy (via direct httpx calls).

These tests verify text completion capabilities through the LiteLLM proxy.
Results are saved to tests/outputs/text/ for inspection.
"""
import json
from datetime import datetime
from pathlib import Path

import pytest

# Import our proxy client instead of litellm SDK
from backend.litellm_proxy_client import (
    chat_completions,
    extract_chat_content,
    extract_json_from_text,
    ProxyClientError,
)


class TestTextGeneration:
    """Test suite for LiteLLM text generation."""

    @pytest.mark.asyncio
    async def test_simple_completion(self, litellm_config, text_outputs_dir):
        """Test basic text completion with a simple prompt."""
        response = await chat_completions(
            model=litellm_config["text_model"],
            messages=[
                {"role": "user", "content": "Say 'Hello, World!' and nothing else."}
            ],
            base_url=litellm_config["base_url"],
            api_key=litellm_config["api_key"],
            max_tokens=50,
        )

        content = extract_chat_content(response)
        
        # Save output - response is already a plain dict, no serialization issues
        output_file = text_outputs_dir / f"simple_completion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file.write_text(json.dumps({
            "model": litellm_config["text_model"],
            "prompt": "Say 'Hello, World!' and nothing else.",
            "response": content,
            "usage": response.get("usage"),  # Plain dict from JSON response
        }, indent=2))

        assert content is not None
        assert len(content) > 0
        print(f"\n✓ Response: {content}")
        print(f"✓ Output saved to: {output_file}")

    @pytest.mark.asyncio
    async def test_json_structured_output(self, litellm_config, text_outputs_dir):
        """Test structured JSON output generation."""
        prompt = (
            "Generate a JSON object describing a pizza with fields: "
            "name, ingredients (array), price (number), is_vegetarian (boolean). "
            "Respond with ONLY valid JSON, no markdown fences or extra text."
        )

        response = await chat_completions(
            model=litellm_config["text_model"],
            messages=[{"role": "user", "content": prompt}],
            base_url=litellm_config["base_url"],
            api_key=litellm_config["api_key"],
            # Note: response_format may not be supported by all proxies/models
            # We use extract_json_from_text as fallback
            max_tokens=200,
        )

        content = extract_chat_content(response)
        
        # Use our robust JSON extractor that handles markdown fences
        parsed = extract_json_from_text(content)
        
        # Save output
        output_file = text_outputs_dir / f"json_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file.write_text(json.dumps({
            "model": litellm_config["text_model"],
            "prompt": prompt,
            "raw_response": content,
            "parsed_response": parsed,
        }, indent=2))

        assert "name" in parsed or "ingredients" in parsed
        print(f"\n✓ Generated JSON: {json.dumps(parsed, indent=2)}")
        print(f"✓ Output saved to: {output_file}")

    @pytest.mark.asyncio
    async def test_system_prompt(self, litellm_config, text_outputs_dir):
        """Test text generation with system prompt."""
        system_prompt = "You are a helpful chef assistant. Always respond with cooking tips."
        user_prompt = "How do I make pasta taste better?"

        response = await chat_completions(
            model=litellm_config["text_model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            base_url=litellm_config["base_url"],
            api_key=litellm_config["api_key"],
            max_tokens=300,
        )

        content = extract_chat_content(response)
        
        # Save output
        output_file = text_outputs_dir / f"system_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file.write_text(json.dumps({
            "model": litellm_config["text_model"],
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response": content,
        }, indent=2))

        assert len(content) > 20
        print(f"\n✓ Response preview: {content[:200]}...")
        print(f"✓ Output saved to: {output_file}")

    @pytest.mark.asyncio
    async def test_menu_item_description(self, litellm_config, text_outputs_dir):
        """Test generating menu item descriptions (similar to app usage)."""
        item_name = "Truffle Risotto"
        
        prompt = (
            f"Generate a simple, concise, and appetizing description for the menu item "
            f"named '{item_name}' as a single, complete sentence in simple English."
        )

        response = await chat_completions(
            model=litellm_config["text_model"],
            messages=[
                {"role": "system", "content": "You generate simple and appetizing menu descriptions."},
                {"role": "user", "content": prompt},
            ],
            base_url=litellm_config["base_url"],
            api_key=litellm_config["api_key"],
            max_tokens=100,
        )

        content = extract_chat_content(response).strip('"')
        
        # Save output
        output_file = text_outputs_dir / f"menu_description_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file.write_text(json.dumps({
            "model": litellm_config["text_model"],
            "item_name": item_name,
            "description": content,
        }, indent=2))

        assert len(content) > 10
        print(f"\n✓ Menu description for '{item_name}': {content}")
        print(f"✓ Output saved to: {output_file}")

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, litellm_config, text_outputs_dir):
        """Test multi-turn conversation capability."""
        messages = [
            {"role": "user", "content": "My name is Alice."},
        ]

        # First turn
        response1 = await chat_completions(
            model=litellm_config["text_model"],
            messages=messages,
            base_url=litellm_config["base_url"],
            api_key=litellm_config["api_key"],
            max_tokens=100,
        )
        
        assistant_response = extract_chat_content(response1)
        messages.append({"role": "assistant", "content": assistant_response})
        messages.append({"role": "user", "content": "What is my name?"})

        # Second turn
        response2 = await chat_completions(
            model=litellm_config["text_model"],
            messages=messages,
            base_url=litellm_config["base_url"],
            api_key=litellm_config["api_key"],
            max_tokens=100,
        )
        
        final_response = extract_chat_content(response2)
        
        # Save output
        output_file = text_outputs_dir / f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file.write_text(json.dumps({
            "model": litellm_config["text_model"],
            "conversation": messages + [{"role": "assistant", "content": final_response}],
        }, indent=2))

        assert "alice" in final_response.lower()
        print(f"\n✓ Final response: {final_response}")
        print(f"✓ Output saved to: {output_file}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
