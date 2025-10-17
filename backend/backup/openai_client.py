import os
from typing import Dict
import base64
from openai import AsyncOpenAI
import asyncio
import logging
import sys
import aiohttp # Add aiohttp for async HTTP requests
import aiofiles # Add aiofiles for async file operations
import re # Add re for sanitizing filenames
import glob

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")
nvidia_api_key = os.getenv("NVIDIA_API_KEY")

# Initialize the async clients
client = AsyncOpenAI(api_key=api_key, base_url=base_url)

# NVIDIA client using OpenAI-compatible API
nvidia_client = AsyncOpenAI(
    api_key=nvidia_api_key,
    base_url="https://integrate.api.nvidia.com/v1"
)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("menugen.openai_client")

# Custom Exception for Image Generation Failure
class ImageGenerationError(Exception):
    pass

# Read model names from environment variables with defaults
VISION_MODEL = os.getenv("VISION_MODEL", "model-router")
DESCRIPTION_MODEL = os.getenv("DESCRIPTION_MODEL", "model-router")
IMAGE_GEN_MODEL = os.getenv("IMAGE_GEN_MODEL", "dall-e-3")

# NVIDIA model configurations
NVIDIA_VISION_MODEL = os.getenv("NVIDIA_VISION_MODEL", "microsoft/phi-4-multimodal-instruct")
NVIDIA_TEXT_MODEL = os.getenv("NVIDIA_TEXT_MODEL", "openai/gpt-oss-20b")

# Define the path for saving images
IMAGE_SAVE_DIR = os.path.join(os.path.dirname(__file__), "data", "images")

# Ensure the image save directory exists
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

def sanitize_filename(name):
    """Remove or replace characters that are invalid in filenames."""
    # Remove invalid characters
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    # Replace whitespace with underscores
    name = re.sub(r'\s+', '_', name)
    # Limit length (optional)
    return name[:100] # Limit to 100 chars

def clear_images_folder():
    for f in glob.glob(os.path.join(IMAGE_SAVE_DIR, "*")):
        try:
            os.remove(f)
            logger.info(f"Deleted old image: {f}")
        except Exception as e:
            logger.error(f"Failed to delete {f}: {e}")

async def save_image_locally(url: str, item_name: str) -> str:
    """Downloads an image from a URL and saves it locally. Returns the filename."""
    filename = sanitize_filename(item_name) + ".png" # Assume png, adjust if needed
    filepath = os.path.join(IMAGE_SAVE_DIR, filename)
    logger.info(f"Attempting to save image for '{item_name}' locally to {filepath}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if (response.status == 200):
                    async with aiofiles.open(filepath, mode='wb') as f:
                        await f.write(await response.read())
                    logger.info(f"Successfully saved image locally: {filepath}")
                    return filename # Return the filename
                else:
                    logger.error(f"Failed to download image from {url}. Status: {response.status}")
                    raise ImageGenerationError(f"Failed to download image from {url}. Status: {response.status}") # Raise error on download failure
    except Exception as e:
        logger.error(f"Error saving image locally for {item_name} from {url}: {e}", exc_info=True)
        raise ImageGenerationError(f"Error saving image locally: {e}") # Re-raise as specific error

async def parse_menu_image(image_content: bytes, model_provider: str = "nvidia") -> Dict:
    logger.info(f"parse_menu_image called with provider: {model_provider}")
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
        
        # Select client and model based on provider
        if model_provider == "nvidia":
            active_client = nvidia_client
            vision_model = NVIDIA_VISION_MODEL
            logger.info(f"Using NVIDIA vision model: {vision_model}")
        else:
            active_client = client
            vision_model = VISION_MODEL
            logger.info(f"Using OpenAI vision model: {vision_model}")
        
        logger.info(f"Calling {model_provider} API for menu parsing.")
        # Use await with the async client
        if model_provider == "nvidia":
            # NVIDIA-specific parameters for phi-4-multimodal-instruct
            # Note: NVIDIA models don't support response_format parameter
            response = await active_client.chat.completions.create(
                model=vision_model,
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
        else:
            # OpenAI parameters
            response = await active_client.chat.completions.create(
                model=vision_model,
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

        menu_json = response.choices[0].message.content.strip() # Strip whitespace
        logger.info(f"Raw menu parsing response string (stripped): {menu_json}") # Log stripped string
        
        # Check if the string is empty after stripping
        if not menu_json:
            logger.error("OpenAI API response content was empty after stripping whitespace.")
            raise ValueError("OpenAI API response content was empty after stripping whitespace.")

        # Clean potential markdown fences
        if menu_json.startswith("```json"):
            menu_json = menu_json[7:] # Remove ```json\n
        if menu_json.endswith("```"):
            menu_json = menu_json[:-3] # Remove ```
        menu_json = menu_json.strip() # Strip again after removing fences

        import json
        try:
            parsed = json.loads(menu_json)
            logger.info(f"Successfully parsed menu JSON.")
            return parsed
        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to decode JSON response: {json_err}")
            logger.error(f"Invalid JSON string received (stripped): {menu_json}")
            raise ValueError(f"Failed to decode JSON response from OpenAI: {json_err}")

    except Exception as e:
        logger.error(f"Error in parse_menu_image: {e}")
        raise

# Clear images folder before generating images for the first item
_images_cleared = False
async def generate_menu_item_image(item: Dict, model_provider: str = "nvidia") -> str:
    """
    Generate an image for a menu item using the specified model provider.
    
    Args:
        item: Dictionary containing menu item details (name, description)
        model_provider: Either "nvidia" or "openai" (default: "nvidia")
        
    Returns:
        Local filename of the generated image
        
    Raises:
        ImageGenerationError: If image generation fails
    """
    global _images_cleared
    if not _images_cleared:
        clear_images_folder()
        _images_cleared = True
    
    item_name = item.get('name', 'Unknown')
    logger.info(f"generate_menu_item_image called for item: {item_name} with provider: {model_provider}")
    
    # Route to appropriate provider
    if model_provider == "nvidia":
        # Import NVIDIA client and use it
        from nvidia_client import generate_image_nvidia
        return await generate_image_nvidia(item)
    
    elif model_provider == "openai":
        # Use existing OpenAI logic
        image_model = IMAGE_GEN_MODEL
        description = item.get('description', '')
        prompt = f"Create a high-quality, appetizing photo of this menu item: {item_name}. Description: {description}"

        try:
            logger.info(f"Attempting image generation with OpenAI model ({image_model}) for item: {item_name}")
            response = await client.images.generate(
                model=image_model,
                prompt=prompt,
                n=1,
                size="1024x1024"
            )
            image_url = response.data[0].url if response.data and len(response.data) > 0 else None
            if not image_url:
                logger.warning(f"Image model ({image_model}) response missing or empty URL. Full response data: {response.data}")
                raise ImageGenerationError(f"Image model ({image_model}) returned empty or missing URL.")
            logger.info(f"Image generated successfully with {image_model} for item: {item_name}, URL: {image_url}")

            # Save the image locally and get the filename
            local_filename = await save_image_locally(image_url, item_name)
            return local_filename

        except Exception as e:
            error_message = f"Error using OpenAI image model ({image_model}) for item {item_name}: {e}"
            logger.error(error_message, exc_info=True)
            raise ImageGenerationError(error_message)
    
    else:
        error_message = f"Unknown model provider: {model_provider}. Must be 'nvidia' or 'openai'."
        logger.error(error_message)
        raise ImageGenerationError(error_message)

async def simplify_menu_item_description(item: Dict, model_provider: str = "nvidia") -> str:
    """Takes a menu item dictionary and returns a simplified description using AI.
    If no description exists, it generates one based on the item name.
    Ensures the description is a full sentence and removes surrounding quotes.
    """
    item_name = item.get('name', 'Unknown')
    description = item.get('description')
    logger.info(f"simplify_menu_item_description called for item: {item_name} with provider: {model_provider}")

    try:
        if (description):
            logger.info(f"Simplifying existing description for: {item_name}")
            prompt = (
                f"Rephrase the following menu item description as a single, complete sentence in simple English. "
                f"Explain any potentially unfamiliar culinary terms (like picadillo, aioli, etc.) or the dish name itself (like Pastelón) in simple terms within the sentence. "
                f"The goal is for someone completely unfamiliar with the dish or terms to understand what it is. "
                f"Focus on key ingredients and preparation. Avoid jargon. Do not include any quotation marks in the final output. "
                f"Example 1: If the original description is 'baked Roman-style, semolina gnocchi; gorgonzola cheese, rosemary; salsa rossa', "
                f"the rephrased sentence should be like 'These Roman-style baked dumplings made from semolina flour are served with gorgonzola cheese, rosemary, and a vibrant red sauce.'. "
                f"Example 2: If the item name is 'Pastelón' and description is 'beef picadillo, sweet plantain, cheese fondue', "
                f"the rephrased sentence should be like 'Pastelón is a layered casserole, similar to lasagna, made with seasoned ground beef (picadillo), sweet plantains, and melted cheese.'. "
                f"Original item name: '{item_name}'. Original description: '{description}'"
                f"Rephrased sentence in simple English:"
            )
            system_message = "You rephrase menu descriptions into single, simple English sentences, explaining unfamiliar terms clearly and avoiding quotes."
        else:
            logger.info(f"Generating description for item with no description: {item_name}")
            prompt = (
                f"Generate a simple, concise, and appetizing description for the menu item named '{item_name}' as a single, complete sentence in simple English. "
                f"If the item name itself might be unfamiliar (like 'Pastelón'), briefly explain what it is. "
                f"Focus on likely key ingredients and preparation method based on the name. Avoid jargon. "
                f"Do not include any quotation marks in the final output. "
                f"Example: For an item named 'Focaccia', the generated sentence could be 'Enjoy our freshly baked Italian flatbread, known as Focaccia, perfect for starting your meal.'. "
                f"Generate a description for '{item_name}'."
                f"Generated sentence in simple English:"
            )
            system_message = "You generate simple and appetizing menu descriptions as single, complete sentences in simple English, explaining unfamiliar item names clearly and avoiding quotes."

        # Select client and model based on provider
        if model_provider == "nvidia":
            active_client = nvidia_client
            text_model = NVIDIA_TEXT_MODEL
            logger.info(f"Using NVIDIA text model: {text_model}")
        else:
            active_client = client
            text_model = DESCRIPTION_MODEL
            logger.info(f"Using OpenAI text model: {text_model}")

        logger.info(f"Calling {model_provider} API for description processing: {item_name}")
        # Use await with the async client
        if model_provider == "nvidia":
            # NVIDIA-specific parameters for openai/gpt-oss-20b
            response = await active_client.chat.completions.create(
                model=text_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4096,
                temperature=1.0,
                top_p=1.0
            )
        else:
            # OpenAI parameters
            response = await active_client.chat.completions.create(
                model=text_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.6
            )

        if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
            logger.warning(f"OpenAI API returned no description/simplification for: {item_name}. Returning original or empty.")
            # Clean quotes even from fallback
            clean_description = description.strip().strip('\"') if description else ""
            return clean_description

        generated_description = response.choices[0].message.content.strip()
        # Explicitly remove leading/trailing quotes
        generated_description = generated_description.strip('\"')
        logger.info(f"Successfully generated/simplified description for: {item_name}")
        return generated_description

    except Exception as e:
        logger.error(f"Error processing description for item {item_name}: {e}")
        # Fallback: Clean quotes from original description (if exists) or return empty string
        # clean_description = description.strip().strip('\"') if description else ""
        # return clean_description
        logger.critical(f"Terminating application due to critical error in simplify_menu_item_description for item {item_name}.")
        sys.exit(1) # Terminate the application on error
