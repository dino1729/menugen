import os
from dotenv import load_dotenv

# Load environment variables from .env file BEFORE other imports
load_dotenv() 

# --- Rest of your existing imports ---
import logging
from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect, BackgroundTasks, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
# Import the custom exception and utilities
from client_utils import ImageGenerationError, clear_images_folder as clear_images_util
from image_utils import encode_image_to_base64

# Import provider-specific implementations
from nvidia_image_parser import parse_menu_image_nvidia
from openai_image_parser import parse_menu_image_openai
from nvidia_description import simplify_menu_item_description_nvidia
from openai_description import simplify_menu_item_description_openai
from nvidia_image_generation import generate_menu_item_image_nvidia
from openai_image_generation import generate_menu_item_image_openai

# Wrapper functions to route to appropriate provider
async def parse_menu_image(image_content: bytes, model_provider: str = "nvidia"):
    """Route menu image parsing to appropriate provider."""
    if model_provider == "nvidia":
        return await parse_menu_image_nvidia(image_content)
    else:
        return await parse_menu_image_openai(image_content)

async def simplify_menu_item_description(item, model_provider: str = "nvidia"):
    """Route description simplification to appropriate provider."""
    if model_provider == "nvidia":
        return await simplify_menu_item_description_nvidia(item)
    else:
        return await simplify_menu_item_description_openai(item)

async def generate_menu_item_image(item, model_provider: str = "nvidia"):
    """Route image generation to appropriate provider."""
    if model_provider == "nvidia":
        return await generate_menu_item_image_nvidia(item)
    else:
        return await generate_menu_item_image_openai(item)
import asyncio
import uuid
from fastapi.staticfiles import StaticFiles
import glob
import httpx
from urllib.parse import urlparse, unquote

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

@app.post("/upload_menu/")
async def upload_menu(file: UploadFile, request: Request):
    logger.info(f"Received file upload: filename={file.filename}, content_type={file.content_type}")
    content = await file.read()
    
    # Try to get model_provider from form data
    form_data = await request.form()
    model_provider = form_data.get("model_provider", "nvidia")
    logger.info(f"Model provider selected: {model_provider}")
    
    session_id = str(uuid.uuid4())
    logger.info(f"Generated session_id={session_id} for upload.")
    asyncio.create_task(process_menu(session_id, content, model_provider))
    logger.info(f"Started background task for session_id={session_id}.")
    return JSONResponse(content={"status": "processing", "sessionId": session_id})

@app.post("/parse_menu_only/")
async def parse_menu_only(file: UploadFile, request: Request):
    logger.info(f"Received file for parsing only: filename={file.filename}, content_type={file.content_type}")
    content = await file.read()
    
    # Try to get model_provider from form data
    form_data = await request.form()
    model_provider = form_data.get("model_provider", "nvidia")
    logger.info(f"Model provider selected for parsing: {model_provider}")
    
    try:
        logger.info("Calling parse_menu_image for parsing only.")
        parsed_data = await parse_menu_image(content, model_provider)
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
            simplification_tasks = [simplify_menu_item_description(item, model_provider) for item in items]
            
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
        logger.error(f"Error during parsing only: {e}", exc_info=True) # Log traceback
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

# Using clear_images_util from client_utils (imported above as clear_images_util)

async def process_menu(session_id, image_content, model_provider="nvidia"):
    logger.info(f"Begin processing menu for session_id={session_id} with model_provider={model_provider}.")

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
        # Optionally, clean up or handle the case where the client never connects
        return

    try:
        # Step 1: Parse menu
        model_display = "NVIDIA" if model_provider == "nvidia" else "OpenAI"
        await safe_send_json(websocket, {"type": "status", "message": f"Parsing menu with {model_display}..."})
        logger.info(f"Calling parse_menu_image for session_id={session_id} with provider={model_provider}.")
        parsed_menu = await parse_menu_image(image_content, model_provider)
        logger.info(f"Menu parsed for session_id={session_id}: {parsed_menu}")
        await safe_send_json(websocket, {"type": "menu_parsed", "data": parsed_menu})

        # Step 2: Generate images for each item
        # Handle both dict and list responses for parsed_menu
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
             return # Exit if no items

        # Simplify descriptions before generating images
        logger.info(f"Processing descriptions for {len(items)} items before image generation with {model_provider}.")
        
        tasks_for_gather = []
        indices_of_dict_items = []
        for i, item_to_process in enumerate(items):
            if isinstance(item_to_process, dict):
                tasks_for_gather.append(simplify_menu_item_description(item_to_process, model_provider))
                indices_of_dict_items.append(i)
        
        if tasks_for_gather:
            processed_descriptions = await asyncio.gather(*tasks_for_gather)
            for i, original_item_index in enumerate(indices_of_dict_items):
                items[original_item_index]['description'] = processed_descriptions[i]
            logger.info("Descriptions processed successfully before image generation.")
            # Send the updated menu data (with simplified descriptions) again
            # `parsed_menu` has been updated in place because `items` refers to its contents.
            await safe_send_json(websocket, {"type": "menu_parsed", "data": parsed_menu})
        else:
            logger.info("No dictionary items found to process descriptions for.")

        for idx, item in enumerate(items):
            # Ensure item is a dictionary before accessing keys
            if not isinstance(item, dict):
                logger.warning(f"Skipping item at index {idx} because it's not a dictionary: {item}")
                continue
            item_name = item.get('name', f'Unknown Item {idx+1}') # Use get with default
            logger.info(f"Generating image for item {item_name} (session_id={session_id}, idx={idx}) using {model_provider}.")
            model_display = "NVIDIA Stable Diffusion 3" if model_provider == "nvidia" else "OpenAI DALL-E 3"
            await safe_send_json(websocket, {"type": "status", "message": f"Generating image for {item_name} ({idx+1}/{len(items)}) with {model_display}..."})
            try:
                # generate_menu_item_image now returns the local filename
                local_filename = await generate_menu_item_image(item, model_provider)
                logger.info(f"Image generated and saved locally as {local_filename} for item {item_name} (session_id={session_id}).")

                # Construct the relative URL for the static file server
                image_static_url = f"/images/{local_filename}"
                logger.info(f"Constructed static URL for {item_name}: {image_static_url}")

                # Send the static URL
                await safe_send_json(websocket, {"type": "image_generated", "item": item_name, "url": image_static_url})
            except ImageGenerationError as img_err: # Catch the specific error
                logger.error(f"Image generation failed permanently for item {item_name} (session_id={session_id}): {img_err}")
                # Send a specific error type for the UI to handle
                await safe_send_json(websocket, {"type": "image_generation_failed", "item": item_name, "message": str(img_err)})
            except Exception as e: # Catch other potential errors during generation for this item
                logger.error(f"Unexpected error during image generation for item {item_name} (session_id={session_id}): {e}")
                await safe_send_json(websocket, {"type": "image_error", "item": item_name, "message": f"Unexpected error: {str(e)}"})
        
        await safe_send_json(websocket, {"type": "done"})
        logger.info(f"Finished processing items for session_id={session_id}.") # Changed log message slightly
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected during processing for session_id={session_id}.")
        # No need to send error, connection is gone

    except Exception as e:
        # Catch all other errors (e.g., parsing, initial connection issues)
        logger.critical(f"Critical error in process_menu for session_id={session_id}: {e}", exc_info=True) # Add exc_info for traceback
        # Ensure websocket is still valid before sending error
        if websocket and websocket.client_state.name == 'CONNECTED':
            # Send a generic processor error
            await safe_send_json(websocket, {"type": "error", "source": "processor", "message": f"An unexpected error occurred: {str(e)}"})
        else:
            logger.error(f"WebSocket for session {session_id} is closed or invalid, cannot send final error.")
    finally:
        # Clean up session
        logger.info(f"Cleaning up session {session_id}.")
        sessions.pop(session_id, None)
