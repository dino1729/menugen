"""
LiteLLM Client for MenuGen Application.

This module provides high-level functions for menu parsing, description generation,
and image generation using the LiteLLM proxy via direct httpx calls.

All calls go through the configured LiteLLM proxy endpoint, avoiding SDK bypass issues.
"""
import os
import base64
import logging
import json
from typing import Dict

from litellm_proxy_client import (
    chat_completions,
    image_generations,
    extract_chat_content,
    extract_json_from_text,
    extract_image_bytes,
    ProxyClientError,
    LITELLM_BASE_URL,
    LITELLM_API_KEY,
)
from client_utils import ImageGenerationError, save_image_locally

# Set up logging
logger = logging.getLogger("menugen.litellm_client")

# Configuration - All model names must be set via environment variables
LLM_MODEL = os.getenv("LLM_MODEL")  # Model for text generation (descriptions)
VISION_MODEL = os.getenv("VISION_MODEL") or LLM_MODEL  # Model for image parsing (falls back to LLM_MODEL)
IMAGE_GEN_MODEL = os.getenv("IMAGE_GEN_MODEL")  # Model for image generation


async def parse_menu_image_litellm(image_content: bytes, session_config: dict = None) -> Dict:
    """
    Parse menu image using LiteLLM (routing to configured proxy).
    
    Args:
        image_content: Raw image bytes
        session_config: Optional session configuration with model overrides
        
    Returns:
        Parsed menu data with 'items' array
    """
    # Use session config if provided, otherwise use environment defaults
    vision_model = (session_config or {}).get("vision_model", VISION_MODEL)
    
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

        logger.info(f"Calling LiteLLM proxy at {LITELLM_BASE_URL}")
        
        response = await chat_completions(
            model=vision_model,
            messages=messages,
            base_url=LITELLM_BASE_URL,
            api_key=LITELLM_API_KEY,
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
    item_name = item.get('name', 'Unknown')
    description = item.get('description')
    
    # Use session config if provided, otherwise use environment defaults
    llm_model = (session_config or {}).get("llm_model", LLM_MODEL)
    
    logger.info(f"simplify_menu_item_description_litellm called for: {item_name} using model: {llm_model}")

    try:
        if description:
            prompt = (
                f"Rephrase the following menu item description as a single, complete sentence in simple English. "
                f"Explain any potentially unfamiliar culinary terms. "
                f"Original item name: '{item_name}'. Original description: '{description}'"
                f"Rephrased sentence in simple English:"
            )
            system_message = "You rephrase menu descriptions into single, simple English sentences."
        else:
            prompt = (
                f"Generate a simple, concise, and appetizing description for the menu item named '{item_name}' as a single, complete sentence in simple English. "
                f"Generated sentence in simple English:"
            )
            system_message = "You generate simple and appetizing menu descriptions as single, complete sentences."

        response = await chat_completions(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            base_url=LITELLM_BASE_URL,
            api_key=LITELLM_API_KEY,
            max_tokens=100
        )

        generated_description = extract_chat_content(response)
        generated_description = generated_description.strip('\"')
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
    item_name = item.get('name', 'Unknown')
    description = item.get('description', '')
    
    # Use session config if provided, otherwise use environment defaults
    image_gen_model = (session_config or {}).get("image_gen_model", IMAGE_GEN_MODEL)
    
    logger.info(f"Generating image with LiteLLM ({image_gen_model}) for: {item_name}")

    prompt = f"Create a high-quality, appetizing photo of this menu item: {item_name}. Description: {description}"

    try:
        response = await image_generations(
            model=image_gen_model,
            prompt=prompt,
            base_url=LITELLM_BASE_URL,
            api_key=LITELLM_API_KEY,
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
