import os
from dotenv import load_dotenv

# Load environment variables from .env file BEFORE other imports
load_dotenv()

import asyncio
import logging
import uuid
from typing import Optional

from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect, Request, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Import config module for centralized configuration
from config import get_config, load_config, validate_litellm_connectivity

# Import the custom exception and utilities
from client_utils import ImageGenerationError, clear_images_folder as clear_images_util

# Import provider-specific implementations
from litellm_client import (
    parse_menu_image_litellm,
    simplify_menu_item_description_litellm,
    generate_menu_item_image_litellm
)
from nvidia_image_generation import generate_menu_item_image_nvidia 

# Logging setup
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("menugen.main")


# Wrapper functions to route to appropriate provider (with session config support)
async def parse_menu_image(image_content: bytes, session_config: dict = None):
    """Route menu image parsing to LiteLLM with optional session config."""
    return await parse_menu_image_litellm(image_content, session_config)


async def simplify_menu_item_description(item, session_config: dict = None):
    """Route description simplification to LiteLLM with optional session config."""
    return await simplify_menu_item_description_litellm(item, session_config)


async def generate_menu_item_image(item, session_config: dict = None):
    """Route image generation based on session configuration or global default."""
    config = get_config()
    provider = (session_config or {}).get("image_provider", config.image_provider)
    logger.info(f"generate_menu_item_image: provider={provider}, session_config={session_config}")

    if provider == "nvidia":
        return await generate_menu_item_image_nvidia(item, session_config)
    else:
        # Use litellm (supports openai, gemini, etc. via proxy)
        return await generate_menu_item_image_litellm(item, session_config)


app = FastAPI(title="MenuGen API", description="Menu parsing and illustration API")

# Allow CORS for frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static images from /images
app.mount("/images", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "data", "images")), name="images")

# In-memory session-to-connection mapping (for demo; not for prod)
sessions = {}


@app.on_event("startup")
async def startup_event():
    """Validate LiteLLM connectivity on startup (fail-fast)."""
    logger.info("Starting MenuGen backend...")
    load_config()  # Load configuration first

    try:
        await validate_litellm_connectivity()
        logger.info("Startup validation passed - LiteLLM proxy is reachable")
    except RuntimeError as e:
        logger.critical(f"Startup validation FAILED: {e}")
        raise  # This will prevent the app from starting


@app.get("/")
def read_root():
    logger.info("Received request at root endpoint.")
    return {"message": "Welcome to the Menu Parser & Illustrator API!"}


@app.get("/config")
async def get_config_endpoint():
    """
    Return current backend configuration including available providers,
    selected models, and curated whitelists.
    Frontend uses this to display current settings and populate model dropdowns.
    """
    logger.info("Received request for configuration.")
    cfg = get_config()

    return {
        "image_provider": cfg.image_provider,
        "vision_model": cfg.vision_model,
        "description_model": cfg.description_model,
        "image_gen_model": cfg.image_gen_model,
        "video_gen_model": cfg.video_gen_model,
        "openai_base_url": cfg.openai_base_url,
        "nvidia_available": bool(cfg.nvidia.api_key),
        "litellm_healthy": cfg.litellm_healthy,
        "whitelists": {
            "vision": cfg.vision_models,
            "text": cfg.text_models,
            "image": cfg.image_models,
            "video": cfg.video_models,
        }
    }

@app.get("/models")
async def get_models():
    """
    Return curated model whitelists from config.json.

    If whitelists are defined in config.json, returns those curated lists.
    Falls back to fetching from LiteLLM proxy if whitelists are empty.
    """
    logger.info("Fetching available models.")
    cfg = get_config()

    # If we have curated whitelists, use them
    if cfg.vision_models or cfg.text_models or cfg.image_models or cfg.video_models:
        all_models = list(set(
            cfg.vision_models + cfg.text_models + cfg.image_models + cfg.video_models
        ))
        logger.info(f"Returning curated whitelists: {len(cfg.vision_models)} vision, "
                   f"{len(cfg.image_models)} image, {len(cfg.video_models)} video, {len(cfg.text_models)} text")
        return {
            "success": True,
            "models": {
                "all": sorted(all_models),
                "vision": cfg.vision_models,
                "image": cfg.image_models,
                "video": cfg.video_models,
                "text": cfg.text_models,
            }
        }

    # Fallback: fetch from LiteLLM proxy
    logger.info("No curated whitelists found, fetching from LiteLLM proxy.")
    import httpx

    try:
        headers = {}
        if cfg.openai_api_key:
            headers["Authorization"] = f"Bearer {cfg.openai_api_key}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{cfg.openai_base_url}/v1/models", headers=headers)

            if response.status_code != 200:
                logger.error(f"Failed to fetch models from LiteLLM: {response.status_code}")
                return {
                    "success": False,
                    "error": f"LiteLLM returned status {response.status_code}",
                    "models": {"all": [], "vision": [], "image": [], "video": [], "text": []}
                }

            data = response.json()
            models = data.get("data", [])

            # Extract and categorize model IDs
            model_set = set()
            vision_models = set()
            image_models = set()
            video_models = set()
            text_models = set()

            for model in models:
                model_id = model.get("id", "")
                if not model_id or model_id.endswith("-perflab"):
                    continue

                model_set.add(model_id)
                model_lower = model_id.lower()

                # Vision models
                if any(kw in model_lower for kw in [
                    "gpt-4", "gpt-5", "gemini-2", "gemini-3", "claude-sonnet", "claude-opus",
                    "vision", "multimodal"
                ]):
                    vision_models.add(model_id)

                # Image generation models
                if any(kw in model_lower for kw in [
                    "dall-e", "imagen", "flux", "stable-diffusion", "image-preview"
                ]):
                    image_models.add(model_id)

                # Video generation models
                if any(kw in model_lower for kw in ["veo", "sora", "video"]):
                    video_models.add(model_id)

                # Text models
                if any(kw in model_lower for kw in [
                    "gpt", "claude", "gemini", "llama", "mistral", "deepseek"
                ]):
                    text_models.add(model_id)

            logger.info(f"Fetched {len(model_set)} models from LiteLLM proxy.")
            return {
                "success": True,
                "models": {
                    "all": sorted(list(model_set)),
                    "vision": sorted(list(vision_models)),
                    "image": sorted(list(image_models)),
                    "video": sorted(list(video_models)),
                    "text": sorted(list(text_models)),
                }
            }

    except Exception as e:
        logger.error(f"Error fetching models from LiteLLM: {e}")
        return {
            "success": False,
            "error": str(e),
            "models": {
                "all": [],
                "vision": [],
                "image": [],
                "video": [],
                "text": []
            }
        }

@app.post("/upload_menu/")
async def upload_menu(
    file: UploadFile,
    request: Request,
    image_provider: Optional[str] = Form(None),
    vision_model: Optional[str] = Form(None),
    image_gen_model: Optional[str] = Form(None),
    video_gen_model: Optional[str] = Form(None),
    description_model: Optional[str] = Form(None)
):
    """
    Upload a menu image and start processing with optional model/provider overrides.

    Form parameters:
    - file: The menu image file
    - image_provider: Provider for image generation ("litellm" or "nvidia")
    - vision_model: Model for menu parsing (vision/multimodal)
    - image_gen_model: Model for generating menu item images
    - video_gen_model: Model for video generation (if needed)
    - description_model: Model for text generation (descriptions)
    """
    logger.info(f"Received file upload: filename={file.filename}, content_type={file.content_type}")
    logger.info(f"Provider settings: image_provider={image_provider}, vision={vision_model}, image_gen={image_gen_model}")

    content = await file.read()

    session_id = str(uuid.uuid4())
    logger.info(f"Generated session_id={session_id} for upload.")

    # Store model preferences with session (use config defaults)
    cfg = get_config()
    session_config = {
        "image_provider": image_provider or cfg.image_provider,
        "vision_model": vision_model or cfg.vision_model,
        "image_gen_model": image_gen_model or cfg.image_gen_model,
        "video_gen_model": video_gen_model or cfg.video_gen_model,
        "description_model": description_model or cfg.description_model,
    }
    
    asyncio.create_task(process_menu(session_id, content, session_config))
    logger.info(f"Started background task for session_id={session_id}.")
    return JSONResponse(content={"status": "processing", "sessionId": session_id})

@app.post("/parse_menu_only/")
async def parse_menu_only(
    file: UploadFile,
    request: Request,
    vision_model: Optional[str] = Form(None),
    description_model: Optional[str] = Form(None)
):
    """
    Parse menu image only (no image generation) with optional model overrides.

    Form parameters:
    - file: The menu image file
    - vision_model: Model for menu parsing (vision/multimodal)
    - description_model: Model for text generation (descriptions)
    """
    logger.info(f"Received file for parsing only: filename={file.filename}, content_type={file.content_type}")
    logger.info(f"Model settings: vision={vision_model}, description={description_model}")

    content = await file.read()

    # Create session config using config defaults
    cfg = get_config()
    session_config = {
        "vision_model": vision_model or cfg.vision_model,
        "description_model": description_model or cfg.description_model,
    }
    
    try:
        logger.info("Calling parse_menu_image for parsing only.")
        parsed_data = await parse_menu_image(content, session_config)
        logger.info(f"Menu parsed successfully (parsing only): {parsed_data}")

        # Determine the list of items
        items = []
        if isinstance(parsed_data, dict):
            items = parsed_data.get("items", [])
            logger.info("Parsed data is a dictionary. Extracted items.")
        elif isinstance(parsed_data, list):
            items = parsed_data
            logger.info("Parsed data is a list. Using it directly as items.")
        else:
            logger.warning("Parsed data is neither a dict nor a list. Cannot process items.")

        # Simplify or generate descriptions for all items
        if items:
            logger.info(f"Processing descriptions for {len(items)} items.")
            simplification_tasks = [simplify_menu_item_description(item, session_config) for item in items]
            
            # Run simplification/generation tasks concurrently
            processed_descriptions = await asyncio.gather(*simplification_tasks)
            
            # Update item descriptions with processed versions
            for i, item in enumerate(items):
                item['description'] = processed_descriptions[i]
            logger.info("Descriptions processed successfully.")
        else:
            logger.info("No items found in parsed data to process descriptions for.")

        # Return the original structure (dict or list) with updated descriptions
        return JSONResponse(content={"status": "success", "data": parsed_data})
    except Exception as e:
        logger.error(f"Error during parsing only: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    logger.info(f"WebSocket connection attempt for session_id={session_id}.")
    await websocket.accept()
    sessions[session_id] = websocket
    logger.info(f"WebSocket accepted and registered for session_id={session_id}.")
    try:
        while True:
            await asyncio.sleep(10)  # Keep connection alive
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session_id={session_id}.")
        sessions.pop(session_id, None)
    except Exception as e:
        logger.error(f"WebSocket error for session_id={session_id}: {e}")
        sessions.pop(session_id, None)

def safe_send_json(websocket, data):
    try:
        logger.info(f"Sending data over WebSocket: {data}")
        return asyncio.create_task(websocket.send_json(data))
    except Exception as e:
        logger.error(f"Send error: {e}")

async def process_menu(session_id, image_content, session_config: dict = None):
    """
    Process menu with optional session configuration for model/provider overrides.
    
    Args:
        session_id: Unique session identifier
        image_content: Raw image bytes
        session_config: Optional dict with model/provider overrides
    """
    logger.info(f"Begin processing menu for session_id={session_id}.")
    if session_config:
        logger.info(f"Using session config: {session_config}")

    # Clear images folder before generating new images
    clear_images_util()

    # Wait for WebSocket connection to be established
    websocket = None
    for _ in range(10):  # Wait up to 5 seconds (10 * 0.5s)
        websocket = sessions.get(session_id)
        if websocket:
            logger.info(f"WebSocket connection found for session_id={session_id}.")
            break
        logger.info(f"Waiting for WebSocket connection for session_id={session_id}...")
        await asyncio.sleep(0.5)

    if not websocket:
        logger.error(f"No WebSocket found for session_id={session_id} after waiting.")
        return

    try:
        # Step 1: Parse menu
        await safe_send_json(websocket, {"type": "status", "message": f"Parsing menu..."})
        logger.info(f"Calling parse_menu_image for session_id={session_id}.")
        parsed_menu = await parse_menu_image(image_content, session_config)
        logger.info(f"Menu parsed for session_id={session_id}: {parsed_menu}")
        await safe_send_json(websocket, {"type": "menu_parsed", "data": parsed_menu})

        # Step 2: Generate images for each item
        items = []
        if isinstance(parsed_menu, dict):
            items = parsed_menu.get("items", [])
            logger.info(f"Parsed menu is a dict, extracted {len(items)} items.")
        elif isinstance(parsed_menu, list):
            items = parsed_menu
            logger.info(f"Parsed menu is a list, using it directly ({len(items)} items).")
        else:
            logger.warning(f"Parsed menu is neither dict nor list ({type(parsed_menu)}), cannot extract items.")

        if not items:
             logger.warning(f"No items found to generate images for session_id={session_id}.")
             await safe_send_json(websocket, {"type": "status", "message": "No menu items found to generate images for."})
             await safe_send_json(websocket, {"type": "done"})
             return 

        # Simplify descriptions before generating images
        logger.info(f"Processing descriptions for {len(items)} items before image generation.")
        
        tasks_for_gather = []
        indices_of_dict_items = []
        for i, item_to_process in enumerate(items):
            if isinstance(item_to_process, dict):
                tasks_for_gather.append(simplify_menu_item_description(item_to_process, session_config))
                indices_of_dict_items.append(i)
        
        if tasks_for_gather:
            processed_descriptions = await asyncio.gather(*tasks_for_gather)
            for i, original_item_index in enumerate(indices_of_dict_items):
                items[original_item_index]['description'] = processed_descriptions[i]
            logger.info("Descriptions processed successfully before image generation.")
            # Send the updated menu data again
            await safe_send_json(websocket, {"type": "menu_parsed", "data": parsed_menu})
        else:
            logger.info("No dictionary items found to process descriptions for.")

        default_provider = get_config().image_provider
        provider = (session_config or {}).get("image_provider", default_provider) if session_config else default_provider

        # Parallel image generation with max 5 concurrent calls
        MAX_CONCURRENT_IMAGES = 5
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_IMAGES)

        async def generate_image_for_item(idx: int, item: dict):
            """Generate image for a single item with semaphore limiting."""
            if not isinstance(item, dict):
                logger.warning(f"Skipping item at index {idx} because it's not a dictionary: {item}")
                return None

            item_name = item.get('name', f'Unknown Item {idx+1}')

            async with semaphore:
                logger.info(f"Generating image for item {item_name} (session_id={session_id}, idx={idx}) using {provider}.")
                await safe_send_json(websocket, {"type": "status", "message": f"Generating image for {item_name}..."})
                try:
                    local_filename = await generate_menu_item_image(item, session_config)
                    logger.info(f"Image generated and saved locally as {local_filename} for item {item_name} (session_id={session_id}).")
                    image_static_url = f"/images/{local_filename}"
                    await safe_send_json(websocket, {"type": "image_generated", "item": item_name, "url": image_static_url})
                    return {"success": True, "item": item_name, "url": image_static_url}
                except ImageGenerationError as img_err:
                    logger.error(f"Image generation failed permanently for item {item_name} (session_id={session_id}): {img_err}")
                    await safe_send_json(websocket, {"type": "image_generation_failed", "item": item_name, "message": str(img_err)})
                    return {"success": False, "item": item_name, "error": str(img_err)}
                except Exception as e:
                    logger.error(f"Unexpected error during image generation for item {item_name} (session_id={session_id}): {e}")
                    await safe_send_json(websocket, {"type": "image_error", "item": item_name, "message": f"Unexpected error: {str(e)}"})
                    return {"success": False, "item": item_name, "error": str(e)}

        # Launch all image generation tasks in parallel (limited by semaphore)
        await safe_send_json(websocket, {"type": "status", "message": f"Generating images for {len(items)} items (up to {MAX_CONCURRENT_IMAGES} in parallel)..."})
        image_tasks = [generate_image_for_item(idx, item) for idx, item in enumerate(items)]
        await asyncio.gather(*image_tasks)

        await safe_send_json(websocket, {"type": "done"})
        logger.info(f"Finished processing items for session_id={session_id}.")
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected during processing for session_id={session_id}.")

    except Exception as e:
        logger.critical(f"Critical error in process_menu for session_id={session_id}: {e}", exc_info=True)
        if websocket and websocket.client_state.name == 'CONNECTED':
            await safe_send_json(websocket, {"type": "error", "source": "processor", "message": f"An unexpected error occurred: {str(e)}"})
        else:
            logger.error(f"WebSocket for session {session_id} is closed or invalid, cannot send final error.")
    finally:
        logger.info(f"Cleaning up session {session_id}.")
        sessions.pop(session_id, None)
