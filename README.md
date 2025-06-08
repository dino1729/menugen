# MenuGen: Menu Parser & AI Image Generator

This project consists of a FastAPI backend (Python) and a React frontend. The backend parses menu images and generates dish illustrations using OpenAI APIs. The frontend provides a modern UI for uploading menus and viewing results in real time.

## Quick Start: Docker Compose

### 1. Prerequisites
- [Docker](https://www.docker.com/get-started) and [Docker Compose](https://docs.docker.com/compose/) installed
- An OpenAI API key (for backend)

### 2. Setup Environment Variables
Create a `.env` file in the `backend/` directory with at least:
```
OPENAI_API_KEY=sk-...
# Optionally:
# OPENAI_API_BASE=your_custom_base_url
# VISION_MODEL=gpt-4o
# IMAGE_GEN_MODEL=gpt-image-1
# DESCRIPTION_MODEL=gpt-4o-mini
```

### 3. Build and Run
From the project root, run:
```
docker compose up --build
```
- The backend will be available at [http://localhost:8000](http://localhost:8005)
- The frontend will be available at [http://localhost:3000](http://localhost:3005)

### 4. Usage
- Open the frontend in your browser: [http://localhost:3000](http://localhost:3005)
- Upload a menu image and watch as the app parses and generates images in real time.

### 5. Stopping
Press `Ctrl+C` in the terminal, or run:
```
docker-compose down
```

## Development Notes
- The backend serves generated images at `/images/` (mounted as a volume for persistence).
- For local development, you can run backend and frontend separately using `uvicorn` and `npm start`.
- Update dependencies in `backend/requirements.txt` and `frontend/package.json` as needed.

## Troubleshooting
- Ensure your OpenAI API key is valid and has access to the required models.
- If you change code, re-run `docker-compose up --build` to rebuild images.
- For logs, check the container output or use `docker-compose logs backend` / `docker-compose logs frontend`.

---

**Enjoy MenuGen!**
