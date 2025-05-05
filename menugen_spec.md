# Architecture Specification: Menu Parser & Illustrator

**Version:** 2.1
**Date:** May 4, 2025

## 1. Introduction

This document outlines the architecture for the "Menu Parser & Illustrator" application. The primary goal is to provide a web-based interface where users can upload an image of a restaurant menu. The application will then intelligently parse the menu's contents (sections and items) using AI (specifically **GPT-4o**) and generate illustrative images for each identified menu item using an AI image generation model (specified as **gpt-image-1**).

This version focuses on a client-server architecture with a **React frontend** for user interaction and a **Python backend (using FastAPI)** to handle processing and AI interactions, featuring **real-time updates** for generated images.

## 2. Goals

* **Input:** Allow users to upload a menu image file (common formats like PNG, JPG/JPEG) via a web interface.
* **Parsing:** Utilize **GPT-4o's** vision capabilities via a backend API call to analyze the menu image and extract a structured representation (JSON) of its sections and items.
* **Image Generation:** For each parsed menu item, generate a relevant, visually appealing image using the **OpenAI Image Generation API (Model: gpt-image-1)** via asynchronous backend calls.
* **Web UI:** Provide a dynamic, responsive, and aesthetically pleasing React-based web interface for:
    * File uploads.
    * Displaying processing status (parsing, image generation progress).
    * Presenting the parsed menu structure clearly.
    * Displaying generated images alongside their corresponding menu items **as they become available**.
* **Styling:** Ensure the frontend features a modern design with smooth, potentially multicolored fonts, subtle CSS animations/transitions, and is fully responsive across devices (desktop, tablet, mobile).
* **Real-time Updates:** Implement mechanisms (e.g., WebSockets, SSE) for the backend to push generated image URLs (or related data) to the frontend as soon as they are ready.
* **Modularity:** Maintain separation of concerns between frontend, backend, and utility modules.

## 3. Non-Goals (for Version 2.1)

* Database storage for menus or images (results are transient per session).
* User accounts or authentication.
* Real-time collaborative editing.
* Advanced image editing or customization features within the app.
* Integration with external recipe databases or ordering systems.
* Payment processing or commercial features.
* Support for non-image menu formats (e.g., PDF text, DOCX) without external OCR pre-processing.
* Saving/Exporting generated menus or images beyond browser capabilities.

## 4. Architecture Overview

The application employs a client-server architecture with asynchronous updates:

1.  **Frontend (React):** User uploads a menu image file via the UI.
2.  **Upload Request:** React frontend sends the image file to a specific API endpoint (e.g., `/upload_menu/`) on the Python backend.
3.  **Backend Initiation:**
    * FastAPI backend receives the image file.
    * It validates and prepares the image (`image_utils`).
    * It initiates the parsing process by calling `openai_client.parse_menu_image` (using GPT-4o) asynchronously.
    * The initial API response might return a task ID or confirm acceptance.
    * Simultaneously, a persistent connection (WebSocket or SSE) is established between the frontend and backend (or the frontend connects after the initial upload response).
4.  **Backend Processing (Async):**
    * Once parsing is complete, the backend sends the parsed menu structure to the frontend via the WebSocket/SSE connection.
    * The backend then iterates through the parsed items, triggering **asynchronous** calls to `openai_client.generate_menu_item_image` (using `gpt-image-1`) for each item (potentially in parallel, respecting rate limits).
5.  **Streaming Image Results:** As each image generation call completes successfully, the backend immediately sends the item name and its corresponding image URL (or error message) to the frontend via the WebSocket/SSE connection.
6.  **Frontend Display:** The React frontend listens for messages on the WebSocket/SSE connection.
    * It updates the state with the parsed menu structure when received.
    * It updates the state for individual menu items with their generated image URLs as those messages arrive, causing the UI to re-render and display images incrementally. Status indicators show overall progress.

## 5. Components

* **React Frontend (`/frontend` directory):**
    * **Framework:** React (functional components, hooks).
    * **Styling:** Tailwind CSS, Custom CSS (animations, fonts).
    * **UI Components:** `ImageUploader`, `MenuDisplay`, `MenuItem`, `LoadingIndicator`, `ErrorMessage`.
    * **State Management:** React Context API or Zustand (managing upload state, connection status, parsed data, incrementally arriving image URLs, errors).
    * **API Client:** Functions for initial upload (`fetch`/`axios`).
    * **Real-time Client:** Implementation for WebSocket connection (`WebSocket` API) or Server-Sent Events (`EventSource` API) to receive updates from the backend. Logic to handle incoming messages and update state accordingly.
    * **Aesthetics:** Implement smooth transitions, gradient text, subtle animations, modern fonts (e.g., Google Fonts).
* **Python Backend (`/backend` directory):**
    * **Framework:** FastAPI.
    * **Server:** Uvicorn.
    * **API Endpoints:**
        * `/upload_menu/`: POST endpoint to receive the image, initiate parsing/generation tasks, potentially return a task ID.
        * `/ws` (or similar): WebSocket endpoint for persistent communication with the frontend. Alternatively, an SSE endpoint.
        * `/`: GET endpoint to serve static React build files.
    * **Core Logic:** Handles initial upload, manages background tasks for parsing and image generation (`async def`, `BackgroundTasks`, or potentially Celery for more complex scenarios). Manages WebSocket/SSE connections and broadcasts results.
    * **Dependencies:** `fastapi`, `uvicorn`, `websockets` (if using WebSockets), `python-multipart`, `openai`, `pillow`, `python-dotenv`.
* **`openai_client.py` (Backend Module):**
    * Initializes OpenAI client.
    * Contains **async** `parse_menu_image` (using `gpt-4o`).
    * Contains **async** `generate_menu_item_image` (using model specified as `gpt-image-1`). Includes rate-limiting delays (using `asyncio.sleep`).
* **`image_utils.py` (Backend Module):**
    * Contains `encode_image_to_base64`.
* **Configuration (`.env` file - Backend):**
    * Stores `OPENAI_API_KEY`.

## 6. Data Flow

1.  **User Action:** Uploads image file via React UI.
2.  **HTTP Request (POST):** `FormData` (image file) -> React Frontend -> Backend API (`/upload_menu/`).
3.  **HTTP Response:** Backend acknowledges request (e.g., `{"status": "processing", "taskId": "xyz"}`).
4.  **WebSocket/SSE Connection:** Frontend establishes connection to Backend (`/ws` or SSE endpoint).
5.  **Backend Processing (Async):**
    * Image File -> Base64 String.
    * Base64 String -> `parse_menu_image` (GPT-4o) -> Parsed `dict`.
6.  **WebSocket/SSE Message 1:** Backend sends `{"type": "menu_parsed", "data": parsed_dict}` -> Frontend.
7.  **Backend Processing (Async Loop):**
    * For each Item Name -> `generate_menu_item_image` (`gpt-image-1`) -> Image URL.
8.  **WebSocket/SSE Message(s) 2...N:** Backend sends `{"type": "image_generated", "item": item_name, "url": image_url}` (or error) -> Frontend (for each completed image).
9.  **React State Update:** Frontend listener updates state based on message `type`.
10. **UI Render:** React re-renders incrementally as parsed data and image URLs arrive.

## 7. APIs and Technologies

* **Frontend:**
    * React, HTML5, CSS3/Tailwind CSS, JavaScript (ES6+)
    * `fetch` API or `axios`
    * `WebSocket` API or `EventSource` API
* **Backend:**
    * Python 3.x, FastAPI, Uvicorn
    * `openai`, `Pillow`, `python-dotenv`, `python-multipart`, `websockets` (or SSE library)
* **External APIs:**
    * OpenAI Chat Completions API (Model: `gpt-4o`)
    * OpenAI Image Generation API (Model: `gpt-image-1`)
* **Communication Protocol:** RESTful API over HTTP/S (for initial upload), WebSockets or SSE (for real-time updates).
* **Data Formats:** JSON, Base64 (for image encoding).

## 8. Error Handling Strategy

* **Frontend:**
    * Validate file type/size.
    * Display clear loading/processing states (parsing, generating images X of Y).
    * Handle HTTP errors for initial upload.
    * Handle WebSocket/SSE connection errors and disconnections gracefully.
    * Display errors received via WebSocket/SSE (e.g., parsing failed, image generation failed for item X).
    * Show placeholders for items whose images haven't arrived or failed.
* **Backend:**
    * Validate uploads. Use Pydantic models.
    * Handle WebSocket/SSE connection lifecycle (connect, disconnect).
    * Robust `try...except` blocks for file processing, all API calls, JSON parsing.
    * Send specific error messages via WebSocket/SSE to the frontend (e.g., `{"type": "error", "source": "parser", "message": "..."}` or `{"type": "image_error", "item": "...", "message": "..."}`).
    * Log errors extensively.
    * Implement rate-limiting delays (`asyncio.sleep`) and potentially catch rate limit errors from OpenAI.

## 9. Scalability and Future Enhancements

* **Task Queues:** For higher load, replace `BackgroundTasks` with a dedicated task queue like Celery for managing parsing and image generation jobs.
* **Caching:** Implement backend caching (e.g., Redis) for OpenAI API responses.
* **Image Storage:** Store generated images in cloud storage (S3, GCS) and send persistent URLs via WebSocket/SSE.
* **Database Integration:** Add a database for persistence.
* **Improved UI/UX:** Add features like selecting items, customizing prompts, explicit progress bars.
* **Containerization:** Use Docker/Docker Compose.
* **Deployment:** Cloud platforms (Vercel, Heroku, AWS, GCP).
* **Model Confirmation:** Verify the exact identifier and API endpoint for `gpt-image-1` during development.
