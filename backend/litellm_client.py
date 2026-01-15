"""
LiteLLM Client for MenuGen Application.

This module provides high-level functions for menu parsing, description generation,
and image generation using the LiteLLM proxy via direct httpx calls.

All calls go through the configured LiteLLM proxy endpoint, avoiding SDK bypass issues.
Configuration is loaded from config.py module.
"""
import base64
import logging
import os
from typing import Dict

from config import get_config, get_base_url_with_v1
from litellm_proxy_client import (
    chat_completions,
    image_generations,
    extract_chat_content,
    extract_json_from_text,
    extract_image_bytes,
    ProxyClientError,
)
from client_utils import ImageGenerationError, save_image_locally

logger = logging.getLogger("menugen.litellm_client")


async def parse_menu_image_litellm(image_content: bytes, session_config: dict = None) -> Dict:
    """
    Parse menu image using LiteLLM (routing to configured proxy).

    Args:
        image_content: Raw image bytes
        session_config: Optional session configuration with model overrides

    Returns:
        Parsed menu data with 'items' array
    """
    config = get_config()

    # Use session config if provided, otherwise use config defaults
    vision_model = (session_config or {}).get("vision_model", config.vision_model)

    logger.info(f"parse_menu_image_litellm called with vision model: {vision_model}")

    try:
        # Encode image to base64
        base64_image = base64.b64encode(image_content).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{base64_image}"

        prompt = (
            "You are a helpful assistant that extracts structured menu data from images. "
            "Given a photo of a restaurant menu, return a JSON object with this exact structure: "
            '{"items": [{"name": "Item Name", "description": "Item description if available", "section": "Section name if available"}, ...]}. '
            "Each item in the 'items' array should be a flat object with 'name', optional 'description', and optional 'section' fields. "
            "Do NOT nest items under sections. Flatten all items into a single 'items' array. "
            "Respond ONLY with valid JSON in this exact format."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]

        base_url = get_base_url_with_v1()
        logger.info(f"Calling LiteLLM proxy at {base_url}")

        response = await chat_completions(
            model=vision_model,
            messages=messages,
            base_url=base_url,
            api_key=config.openai_api_key,
        )

        content = extract_chat_content(response)
        logger.info(f"Raw menu parsing response (model={vision_model}): {content[:100]}...")

        return extract_json_from_text(content)

    except ProxyClientError as e:
        logger.error(f"Error in parse_menu_image_litellm: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in parse_menu_image_litellm: {e}")
        raise


async def simplify_menu_item_description_litellm(item: Dict, session_config: dict = None) -> str:
    """
    Simplify menu item description using LiteLLM.

    Args:
        item: Menu item dict with 'name' and optional 'description'
        session_config: Optional session configuration with model overrides

    Returns:
        Simplified/generated description string
    """
    config = get_config()
    item_name = item.get('name', 'Unknown')
    description = item.get('description')

    # Use session config if provided, otherwise use config defaults
    description_model = (session_config or {}).get("description_model", config.description_model)

    logger.info(f"simplify_menu_item_description_litellm called for: {item_name} using model: {description_model}")

    try:
        if description:
            prompt = (
                f"Rephrase this menu item as ONE complete sentence (15-30 words). "
                f"Item: '{item_name}'. Description: '{description}'. "
                f"Reply with ONLY the sentence, nothing else."
            )
            system_message = "You write concise menu descriptions. Always respond with exactly one complete sentence."
        else:
            prompt = (
                f"Write ONE appetizing sentence (15-30 words) describing the dish '{item_name}'. "
                f"Reply with ONLY the sentence, nothing else."
            )
            system_message = "You write concise menu descriptions. Always respond with exactly one complete sentence."

        response = await chat_completions(
            model=description_model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            base_url=get_base_url_with_v1(),
            api_key=config.openai_api_key,
            max_tokens=500,
            temperature=0.7,
            max_completion_tokens=500  # Some models use this instead
        )

        # Log finish reason to debug truncation
        finish_reason = response.get("choices", [{}])[0].get("finish_reason", "unknown")
        logger.debug(f"Description response for {item_name}: finish_reason={finish_reason}")

        generated_description = extract_chat_content(response)
        generated_description = generated_description.strip('\"')

        # Log if description seems truncated
        if not generated_description.endswith(('.', '!', '?')):
            logger.warning(f"Description for {item_name} may be truncated: '{generated_description}' (finish_reason={finish_reason})")

        return generated_description

    except ProxyClientError as e:
        logger.error(f"Error in simplify_menu_item_description_litellm: {e}")
        # Fallback to original description if available
        return description.strip().strip('\"') if description else ""
    except Exception as e:
        logger.error(f"Error in simplify_menu_item_description_litellm: {e}")
        # Fallback to original description if available
        return description.strip().strip('\"') if description else ""


async def generate_menu_item_image_litellm(item: Dict, session_config: dict = None) -> str:
    """
    Generate menu item image using LiteLLM (via Proxy).

    Args:
        item: Menu item dict with 'name' and optional 'description'
        session_config: Optional session configuration with model overrides

    Returns:
        Local filename of the saved image

    Raises:
        ImageGenerationError: If image generation fails
    """
    config = get_config()
    item_name = item.get('name', 'Unknown')
    description = item.get('description', '')

    # Use session config if provided, otherwise use config defaults
    image_gen_model = (session_config or {}).get("image_gen_model", config.image_gen_model)

    logger.info(f"Generating image with LiteLLM ({image_gen_model}) for: {item_name}")

    prompt = f"Create a high-quality, appetizing photo of this menu item: {item_name}. Description: {description}"

    try:
        response = await image_generations(
            model=image_gen_model,
            prompt=prompt,
            base_url=get_base_url_with_v1(),
            api_key=config.openai_api_key,
            size="1024x1024"
        )

        # Extract image bytes or URL
        image_bytes, image_url = await extract_image_bytes(response)

        if image_url and not image_url.startswith('data:'):
            # We have an HTTP URL, use the existing save function
            local_filename = await save_image_locally(image_url, item_name)
        else:
            # We have raw bytes, save directly
            from client_utils import IMAGE_SAVE_DIR, sanitize_filename
            import aiofiles

            filename = sanitize_filename(item_name) + ".png"
            filepath = os.path.join(IMAGE_SAVE_DIR, filename)

            async with aiofiles.open(filepath, mode='wb') as f:
                await f.write(image_bytes)

            local_filename = filename

        logger.info(f"Image generated successfully: {local_filename}")
        return local_filename

    except ProxyClientError as e:
        logger.error(f"Error in generate_menu_item_image_litellm: {e}", exc_info=True)
        raise ImageGenerationError(f"LiteLLM image generation failed: {e}")
    except Exception as e:
        logger.error(f"Error in generate_menu_item_image_litellm: {e}", exc_info=True)
        raise ImageGenerationError(f"LiteLLM image generation failed: {e}")
