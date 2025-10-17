import os
import logging
from typing import Dict
from openai import AsyncOpenAI
from client_utils import ImageGenerationError, save_image_locally

# Set up logging
logger = logging.getLogger("menugen.openai_image_generation")

# Read OpenAI API key and initialize client
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")

# Initialize the async OpenAI client
client = AsyncOpenAI(api_key=api_key, base_url=base_url)

# Read OpenAI image generation model from environment variable
IMAGE_GEN_MODEL = os.getenv("IMAGE_GEN_MODEL", "dall-e-3")

async def generate_menu_item_image_openai(item: Dict) -> str:
    """
    Generate an image using OpenAI DALL-E 3.
    
    Args:
        item: Dictionary containing 'name' and 'description' of menu item
        
    Returns:
        Local filename of the saved image
        
    Raises:
        ImageGenerationError: If image generation or saving fails
    """
    item_name = item.get('name', 'Unknown')
    description = item.get('description', '')
    
    # Create prompt for the menu item
    prompt = f"Create a high-quality, appetizing photo of this menu item: {item_name}. Description: {description}"
    
    logger.info(f"Generating image with OpenAI model ({IMAGE_GEN_MODEL}) for item: {item_name}")
    
    try:
        response = await client.images.generate(
            model=IMAGE_GEN_MODEL,
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        
        image_url = response.data[0].url if response.data and len(response.data) > 0 else None
        if not image_url:
            logger.warning(f"Image model ({IMAGE_GEN_MODEL}) response missing or empty URL. Full response data: {response.data}")
            raise ImageGenerationError(f"Image model ({IMAGE_GEN_MODEL}) returned empty or missing URL.")
        
        logger.info(f"Image generated successfully with {IMAGE_GEN_MODEL} for item: {item_name}, URL: {image_url}")
        
        # Save the image locally and get the filename
        local_filename = await save_image_locally(image_url, item_name)
        return local_filename
    
    except Exception as e:
        error_message = f"Error using OpenAI image model ({IMAGE_GEN_MODEL}) for item {item_name}: {e}"
        logger.error(error_message, exc_info=True)
        raise ImageGenerationError(error_message)

