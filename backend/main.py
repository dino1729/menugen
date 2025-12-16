import os
from dotenv import load_dotenv

# Load environment variables from .env file BEFORE other imports
load_dotenv() 

import logging
import asyncio
import uuid
import glob
from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect, BackgroundTasks, Request, Form
from fastapi.responses import JSONResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

# Import the custom exception and utilities
from client_utils import ImageGenerationError, clear_images_folder as clear_images_util

# Import provider-specific implementations
from litellm_client import (
    parse_menu_image_litellm,
    simplify_menu_item_description_litellm,
    generate_menu_item_image_litellm
)
from nvidia_image_generation import generate_menu_item_image_nvidia

# Configuration
# "litellm" (for Gemini/Others via proxy) or "nvidia"
IMAGE_PROVIDER = os.getenv("IMAGE_PROVIDER", "litellm").lower() 

# Wrapper functions to route to appropriate provider (with session config support)
async def parse_menu_image(image_content: bytes, session_config: dict = None):
    """Route menu image parsing to LiteLLM with optional session config."""
    return await parse_menu_image_litellm(image_content, session_config)

async def simplify_menu_item_description(item, session_config: dict = None):
    """Route description simplification to LiteLLM with optional session config."""
    return await simplify_menu_item_description_litellm(item, session_config)

async def generate_menu_item_image(item, session_config: dict = None):
    """Route image generation based on session configuration or global default."""
    provider = (session_config or {}).get("image_provider", IMAGE_PROVIDER)
    logger.info(f"[DEBUG] generate_menu_item_image: provider={provider}, session_config={session_config}")

    if provider == "nvidia":
        return await generate_menu_item_image_nvidia(item, session_config)
    else:
        # Use litellm (supports openai, gemini, etc. via proxy)
        return await generate_menu_item_image_litellm(item, session_config)

# Logging setup
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("menugen.main")

app = FastAPI()

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

@app.get("/")
def read_root():
    logger.info("Received request at root endpoint.")
    return {"message": "Welcome to the Menu Parser & Illustrator API!"}

@app.get("/config")
async def get_config():
    """
    Return current backend configuration including available providers and selected models.
    Frontend uses this to display current settings and populate model dropdowns.
    """
    logger.info("Received request for configuration.")
    return {
        "image_provider": os.getenv("IMAGE_PROVIDER", "litellm"),
        "vision_model": os.getenv("VISION_MODEL", "gpt-4o"),
        "image_gen_model": os.getenv("IMAGE_GEN_MODEL", "gemini-3-pro-image-preview"),
        "video_gen_model": os.getenv("VIDEO_GEN_MODEL", "veo-3.0-generate-001"),
        "llm_model": os.getenv("LLM_MODEL", "gpt-4o"),
        "litellm_base_url": os.getenv("LITELLM_BASE_URL", "http://localhost:4000"),
        "nvidia_available": bool(os.getenv("NVIDIA_API_KEY")),
    }

@app.get("/models")
async def get_models():
    """
    Fetch available models from LiteLLM proxy.
    Returns a categorized list of models by capability (vision, image, video, text).
    """
    logger.info("Fetching available models from LiteLLM proxy.")
    import httpx
    
    litellm_base_url = os.getenv("LITELLM_BASE_URL", "http://localhost:4000").rstrip("/")
    litellm_api_key = os.getenv("LITELLM_API_KEY", "")
    
    try:
        headers = {"Authorization": f"Bearer {litellm_api_key}"} if litellm_api_key else {}
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{litellm_base_url}/v1/models", headers=headers)
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch models from LiteLLM: {response.status_code}")
                return {
                    "success": False,
                    "error": f"LiteLLM returned status {response.status_code}",
                    "models": []
                }
            
            data = response.json()
            models = data.get("data", [])
            
            # Extract model IDs and categorize them (using sets to avoid duplicates)
            model_set = set()
            vision_models = set()
            image_models = set()
            video_models = set()
            text_models = set()
            
            for model in models:
                model_id = model.get("id", "")
                if not model_id:
                    continue
                
                # Skip perflab duplicates
                if model_id.endswith("-perflab"):
                    continue
                    
                model_set.add(model_id)
                model_lower = model_id.lower()
                
                # Categorize models based on known capabilities
                # Vision models (multimodal models that can understand images)
                # Include GPT-4+, GPT-5+, Gemini 1.5+, Claude 3+, and other vision models
                is_vision = (
                    "vision" in model_lower or
                    model_lower.startswith("gpt-4") or  # GPT-4, 4o, 4.1
                    model_lower.startswith("gpt-5") or  # GPT-5, 5.1, 5.2
                    "gpt-4o" in model_lower or
                    "gemini-1.5" in model_lower or
                    "gemini-2" in model_lower or  # Gemini 2.0, 2.5
                    "gemini-3" in model_lower or
                    model_lower.startswith("claude-3") or  # Claude 3 (Opus, Sonnet, Haiku)
                    "claude-4" in model_lower or  # Claude 4 (Haiku 4.5, Sonnet 4, Opus 4)
                    "claude-haiku-4" in model_lower or
                    "claude-sonnet-4" in model_lower or
                    "claude-opus-4" in model_lower or
                    "phi-4-multimodal" in model_lower or
                    "llama-3.2-90b-vision" in model_lower or
                    "pixtral" in model_lower or
                    "qwen-vl" in model_lower
                )
                if is_vision:
                    vision_models.add(model_id)
                
                # Image generation models
                if any(keyword in model_lower for keyword in [
                    "dall-e", "imagen", "flux", "stable-diffusion", "midjourney",
                    "image-preview", "imagen-3", "sd-", "sdxl"
                ]):
                    image_models.add(model_id)
                
                # Video generation models
                if any(keyword in model_lower for keyword in [
                    "veo", "sora", "gen-3", "runway", "video"
                ]):
                    video_models.add(model_id)
                
                # Text models (all models can do text, but we filter to common ones)
                if any(keyword in model_lower for keyword in [
                    "gpt", "claude", "gemini", "llama", "mistral", "phi", 
                    "haiku", "sonnet", "opus", "qwen", "deepseek", "nemotron",
                    "devstral", "ministral", "kimi", "o1", "o3", "o4"
                ]):
                    text_models.add(model_id)
            
            logger.info(f"Successfully fetched {len(model_set)} unique models from LiteLLM.")
            logger.info(f"Categorized: {len(vision_models)} vision, {len(image_models)} image, {len(video_models)} video, {len(text_models)} text")
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
    llm_model: Optional[str] = Form(None)
):
    """
    Upload a menu image and start processing with optional model/provider overrides.
    
    Form parameters:
    - file: The menu image file
    - image_provider: Provider for image generation ("litellm", "nvidia", "openai")
    - vision_model: Model for menu parsing (vision/multimodal)
    - image_gen_model: Model for generating menu item images
    - video_gen_model: Model for video generation (if needed)
    - llm_model: Model for text generation (descriptions)
    """
    logger.info(f"Received file upload: filename={file.filename}, content_type={file.content_type}")
    logger.info(f"Provider settings: image_provider={image_provider}, vision={vision_model}, image_gen={image_gen_model}")
    
    content = await file.read()
    
    session_id = str(uuid.uuid4())
    logger.info(f"Generated session_id={session_id} for upload.")
    
    # Store model preferences with session
    session_config = {
        "image_provider": image_provider or os.getenv("IMAGE_PROVIDER", "litellm"),
        "vision_model": vision_model or os.getenv("VISION_MODEL", "gpt-4o"),
        "image_gen_model": image_gen_model or os.getenv("IMAGE_GEN_MODEL", "gemini-3-pro-image-preview"),
        "video_gen_model": video_gen_model or os.getenv("VIDEO_GEN_MODEL", "veo-3.0-generate-001"),
        "llm_model": llm_model or os.getenv("LLM_MODEL", "gpt-4o"),
    }
    
    asyncio.create_task(process_menu(session_id, content, session_config))
    logger.info(f"Started background task for session_id={session_id}.")
    return JSONResponse(content={"status": "processing", "sessionId": session_id})

@app.post("/parse_menu_only/")
async def parse_menu_only(
    file: UploadFile,
    request: Request,
    vision_model: Optional[str] = Form(None),
    llm_model: Optional[str] = Form(None)
):
    """
    Parse menu image only (no image generation) with optional model overrides.
    
    Form parameters:
    - file: The menu image file
    - vision_model: Model for menu parsing (vision/multimodal)
    - llm_model: Model for text generation (descriptions)
    """
    logger.info(f"Received file for parsing only: filename={file.filename}, content_type={file.content_type}")
    logger.info(f"Model settings: vision={vision_model}, llm={llm_model}")
    
    content = await file.read()
    
    # Create session config
    session_config = {
        "vision_model": vision_model or os.getenv("VISION_MODEL", "gpt-4o"),
        "llm_model": llm_model or os.getenv("LLM_MODEL", "gpt-4o"),
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

        provider = (session_config or {}).get("image_provider", IMAGE_PROVIDER) if session_config else IMAGE_PROVIDER
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                logger.warning(f"Skipping item at index {idx} because it's not a dictionary: {item}")
                continue
            item_name = item.get('name', f'Unknown Item {idx+1}')
            logger.info(f"Generating image for item {item_name} (session_id={session_id}, idx={idx}) using {provider}.")
            await safe_send_json(websocket, {"type": "status", "message": f"Generating image for {item_name} ({idx+1}/{len(items)})..."})
            try:
                local_filename = await generate_menu_item_image(item, session_config)
                logger.info(f"Image generated and saved locally as {local_filename} for item {item_name} (session_id={session_id}).")

                image_static_url = f"/images/{local_filename}"
                await safe_send_json(websocket, {"type": "image_generated", "item": item_name, "url": image_static_url})
            except ImageGenerationError as img_err:
                logger.error(f"Image generation failed permanently for item {item_name} (session_id={session_id}): {img_err}")
                await safe_send_json(websocket, {"type": "image_generation_failed", "item": item_name, "message": str(img_err)})
            except Exception as e:
                logger.error(f"Unexpected error during image generation for item {item_name} (session_id={session_id}): {e}")
                await safe_send_json(websocket, {"type": "image_error", "item": item_name, "message": f"Unexpected error: {str(e)}"})
        
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
