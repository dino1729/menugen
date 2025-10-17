import os
import re
import glob
import logging
import aiohttp
import aiofiles

# Set up logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("menugen.client_utils")

# Custom Exception for Image Generation Failure
class ImageGenerationError(Exception):
    pass

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
    return name[:100]  # Limit to 100 chars

def clear_images_folder():
    """Clear all images from the images folder."""
    for f in glob.glob(os.path.join(IMAGE_SAVE_DIR, "*")):
        try:
            os.remove(f)
            logger.info(f"Deleted old image: {f}")
        except Exception as e:
            logger.error(f"Failed to delete {f}: {e}")

async def save_image_locally(url: str, item_name: str) -> str:
    """Downloads an image from a URL and saves it locally. Returns the filename."""
    filename = sanitize_filename(item_name) + ".png"  # Assume png, adjust if needed
    filepath = os.path.join(IMAGE_SAVE_DIR, filename)
    logger.info(f"Attempting to save image for '{item_name}' locally to {filepath}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    async with aiofiles.open(filepath, mode='wb') as f:
                        await f.write(await response.read())
                    logger.info(f"Successfully saved image locally: {filepath}")
                    return filename  # Return the filename
                else:
                    logger.error(f"Failed to download image from {url}. Status: {response.status}")
                    raise ImageGenerationError(f"Failed to download image from {url}. Status: {response.status}")
    except Exception as e:
        logger.error(f"Error saving image locally for {item_name} from {url}: {e}", exc_info=True)
        raise ImageGenerationError(f"Error saving image locally: {e}")

