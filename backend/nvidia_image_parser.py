import os
import base64
import logging
from typing import Dict
from openai import AsyncOpenAI

# Set up logging
logger = logging.getLogger("menugen.nvidia_image_parser")

# Read NVIDIA API key and base URL from environment
nvidia_api_key = os.getenv("NVIDIA_API_KEY")
nvidia_base_url = os.getenv("NVIDIA_BASE_URL")

# NVIDIA client using OpenAI-compatible API
nvidia_client = AsyncOpenAI(
    api_key=nvidia_api_key,
    base_url=nvidia_base_url
)

# Read NVIDIA vision model from environment variable
NVIDIA_VISION_MODEL = os.getenv("NVIDIA_VISION_MODEL")

async def parse_menu_image_nvidia(image_content: bytes) -> Dict:
    """Parse menu image using NVIDIA vision model.
    
    Args:
        image_content: Raw image bytes
        
    Returns:
        Dictionary with 'items' array containing parsed menu data
        
    Raises:
        ValueError: If API returns empty or invalid response
    """
    logger.info(f"parse_menu_image_nvidia called with model: {NVIDIA_VISION_MODEL}")
    
    try:
        prompt = (
            "You are a helpful assistant that extracts structured menu data from images. "
            "Given a photo of a restaurant menu, return a JSON object with this exact structure: "
            '{"items": [{"name": "Item Name", "description": "Item description if available", "section": "Section name if available"}, ...]}. '
            "Each item in the 'items' array should be a flat object with 'name', optional 'description', and optional 'section' fields. "
            "Do NOT nest items under sections. Flatten all items into a single 'items' array. "
            "Respond ONLY with valid JSON in this exact format."
        )
        
        logger.info("Encoding image content to base64.")
        base64_image = base64.b64encode(image_content).decode('utf-8')
        
        logger.info(f"Calling NVIDIA API for menu parsing with model: {NVIDIA_VISION_MODEL}")
        
        # NVIDIA-specific parameters for phi-4-multimodal-instruct
        # Note: NVIDIA models don't support response_format parameter
        response = await nvidia_client.chat.completions.create(
            model=NVIDIA_VISION_MODEL,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": f"{prompt}\n\nExtract the menu items from this image. Remember: return a flat 'items' array, do not nest items under sections. Return ONLY valid JSON following the exact format specified, no other text or markdown."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]}
            ],
            max_tokens=2048,
            temperature=0.10,
            top_p=0.70
        )
        
        if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
            logger.error("NVIDIA API returned an empty or invalid response.")
            raise ValueError("NVIDIA API returned an empty or invalid response.")
        
        menu_json = response.choices[0].message.content.strip()
        logger.info(f"Raw menu parsing response string (stripped): {menu_json}")
        
        # Check if the string is empty after stripping
        if not menu_json:
            logger.error("NVIDIA API response content was empty after stripping whitespace.")
            raise ValueError("NVIDIA API response content was empty after stripping whitespace.")
        
        # Clean potential markdown fences
        if menu_json.startswith("```json"):
            menu_json = menu_json[7:]  # Remove ```json\n
        if menu_json.endswith("```"):
            menu_json = menu_json[:-3]  # Remove ```
        menu_json = menu_json.strip()  # Strip again after removing fences
        
        import json
        try:
            parsed = json.loads(menu_json)
            logger.info(f"Successfully parsed menu JSON with NVIDIA.")
            return parsed
        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to decode JSON response: {json_err}")
            logger.error(f"Invalid JSON string received (stripped): {menu_json}")
            raise ValueError(f"Failed to decode JSON response from NVIDIA: {json_err}")
    
    except Exception as e:
        logger.error(f"Error in parse_menu_image_nvidia: {e}")
        raise

