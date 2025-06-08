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
# Initialize the async client
client = AsyncOpenAI(api_key=api_key, base_url=base_url)

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

async def parse_menu_image(image_content: bytes) -> Dict:
    logger.info("parse_menu_image called.")
    try:
        prompt = (
            "You are a helpful assistant that extracts structured menu data from images. "
            "Given a photo of a restaurant menu, return a JSON object with a list of menu items. "
            "Each item should have a 'name' and, if available, a 'description'. "
            "If the menu has sections, include a 'section' field for each item. "
            "Respond ONLY with the JSON object."
        )
        logger.info("Encoding image content to base64.")
        base64_image = base64.b64encode(image_content).decode('utf-8')
        logger.info("Calling model-router API for menu parsing.")
        # Use await with the async client
        response = await client.chat.completions.create(
            model=VISION_MODEL, # Use environment variable
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract the menu items from this image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "auto"}}
                ]}
            ],
            max_tokens=2048, # Increased max_tokens
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
async def generate_menu_item_image(item: Dict) -> str:
    global _images_cleared
    if not _images_cleared:
        clear_images_folder()
        _images_cleared = True
    item_name = item.get('name', 'Unknown') # Get item name for logging
    logger.info(f"generate_menu_item_image called for item: {item_name}")
    image_model = IMAGE_GEN_MODEL # Use the single image model variable
    description = item.get('description', '')
    prompt = f"Create a high-quality, appetizing photo of this menu item: {item_name}. Description: {description}"

    try:
        logger.info(f"Attempting image generation with model ({image_model}) for item: {item_name}")
        response = await client.images.generate(
            model=image_model, # Use the single model variable
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        image_url = response.data[0].url if response.data and len(response.data) > 0 else None # Safer access
        if not image_url:
             logger.warning(f"Image model ({image_model}) response missing or empty URL. Full response data: {response.data}")
             # Raise error directly if the model fails
             raise ImageGenerationError(f"Image model ({image_model}) returned empty or missing URL.")
        logger.info(f"Image generated successfully with {image_model} for item: {item_name}, URL: {image_url}")

        # --- Save the image locally and get the filename ---
        local_filename = await save_image_locally(image_url, item_name)
        # ---------------------------------------------------

        return local_filename # Return the local filename instead of the original URL

    except Exception as e:
        # Catch any exception during the attempt (API error, network issue, etc.)
        error_message = f"Error using image model ({image_model}) for item {item_name}: {e}"
        logger.error(error_message, exc_info=True) # Log traceback
        # Raise specific exception
        raise ImageGenerationError(error_message)
    # Fallback logic was already removed

async def simplify_menu_item_description(item: Dict) -> str:
    """Takes a menu item dictionary and returns a simplified description using OpenAI.
    If no description exists, it generates one based on the item name.
    Ensures the description is a full sentence and removes surrounding quotes.
    """
    item_name = item.get('name', 'Unknown')
    description = item.get('description')
    logger.info(f"simplify_menu_item_description called for item: {item_name}")

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

        logger.info(f"Calling OpenAI API for: {item_name}")
        # Use await with the async client
        response = await client.chat.completions.create(
            model=DESCRIPTION_MODEL, # Use environment variable
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.6,
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
