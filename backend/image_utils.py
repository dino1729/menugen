import base64
from PIL import Image
from io import BytesIO
import logging
import os

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("menugen.image_utils")

def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to a base64 string."""
    logger.info(f"Encoding image to base64: {image_path}")
    try:
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            logger.info(f"Image encoded to base64, length={len(b64)}")
            return b64
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        raise
