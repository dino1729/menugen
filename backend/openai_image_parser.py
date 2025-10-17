import os
import base64
import logging
from typing import Dict
from openai import AsyncOpenAI

# Set up logging
logger = logging.getLogger("menugen.openai_image_parser")

# Read OpenAI API key and initialize client
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")

# Initialize the async OpenAI client
client = AsyncOpenAI(api_key=api_key, base_url=base_url)

# Read OpenAI vision model from environment variable
VISION_MODEL = os.getenv("VISION_MODEL", "model-router")

async def parse_menu_image_openai(image_content: bytes) -> Dict:
    """Parse menu image using OpenAI vision model.
    
    Args:
        image_content: Raw image bytes
        
    Returns:
        Dictionary with 'items' array containing parsed menu data
        
    Raises:
        ValueError: If API returns empty or invalid response
    """
    logger.info(f"parse_menu_image_openai called with model: {VISION_MODEL}")
    
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
        
        logger.info(f"Calling OpenAI API for menu parsing with model: {VISION_MODEL}")
        
        # OpenAI parameters with JSON response format
        response = await client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract the menu items from this image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "auto"}}
                ]}
            ],
            max_tokens=2048,
            response_format={"type": "json_object"}
        )
        
        if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
            logger.error("OpenAI API returned an empty or invalid response.")
            raise ValueError("OpenAI API returned an empty or invalid response.")
        
        menu_json = response.choices[0].message.content.strip()
        logger.info(f"Raw menu parsing response string (stripped): {menu_json}")
        
        # Check if the string is empty after stripping
        if not menu_json:
            logger.error("OpenAI API response content was empty after stripping whitespace.")
            raise ValueError("OpenAI API response content was empty after stripping whitespace.")
        
        # Clean potential markdown fences
        if menu_json.startswith("```json"):
            menu_json = menu_json[7:]  # Remove ```json\n
        if menu_json.endswith("```"):
            menu_json = menu_json[:-3]  # Remove ```
        menu_json = menu_json.strip()  # Strip again after removing fences
        
        import json
        try:
            parsed = json.loads(menu_json)
            logger.info(f"Successfully parsed menu JSON with OpenAI.")
            return parsed
        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to decode JSON response: {json_err}")
            logger.error(f"Invalid JSON string received (stripped): {menu_json}")
            raise ValueError(f"Failed to decode JSON response from OpenAI: {json_err}")
    
    except Exception as e:
        logger.error(f"Error in parse_menu_image_openai: {e}")
        raise

