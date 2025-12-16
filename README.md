# MenuGen: Menu Parser & AI Image Generator

This project consists of a FastAPI backend (Python) and a React frontend. The backend parses menu images and generates dish illustrations using AI models (NVIDIA Stable Diffusion 3 or OpenAI DALL-E 3). The frontend provides a modern UI for uploading menus and viewing results in real time.

## Quick Start: Docker Compose

### 1. Prerequisites
- [Docker](https://www.docker.com/get-started) and [Docker Compose](https://docs.docker.com/compose/) installed
- An NVIDIA API key (for image generation) or OpenAI API key (alternative)

### 2. Setup Environment Variables
Create a `.env` file in the **project root** by copying from `example.env`:
```bash
cp example.env .env
```

Then edit `.env` with your configuration:
```
# LiteLLM Configuration (Primary)
LITELLM_BASE_URL=http://localhost:4000
LITELLM_API_KEY=your_litellm_api_key_here

# Model Configuration
LLM_MODEL=gpt-4o
IMAGE_GEN_MODEL=gemini-3-pro-image-preview

# Image Provider Selection
# Options: "litellm" (uses IMAGE_GEN_MODEL via LiteLLM/Proxy) or "nvidia" (uses NVIDIA_IMAGE_GEN_URL)
IMAGE_PROVIDER=litellm

# NVIDIA Configuration (Backup)
NVIDIA_API_KEY=your_nvidia_api_key_here
NVIDIA_IMAGE_GEN_URL=https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-3-medium

# Server Configuration
LOG_LEVEL=INFO
```

### 3. Build and Run
From the project root, run:
```bash
docker compose up --build
```
- The backend will be available at [http://localhost:8005](http://localhost:8005)
- The frontend will be available at [http://localhost:3005](http://localhost:3005)

### 4. Usage
- Open the frontend in your browser: [http://localhost:3005](http://localhost:3005)
- Upload a menu image and watch as the app parses and generates images in real time.

### 5. Stopping
Press `Ctrl+C` in the terminal, or run:
```bash
docker compose down
```

## Local Development (without Docker)

Use the provided helper scripts to run locally:

```bash
# Start both backend and frontend
./start_menugen.sh

# Check status
./status_menugen.sh

# Stop all services
./stop_menugen.sh
```

The scripts will:
- Create a Python virtual environment (`.venv/`) if needed and install dependencies from `requirements.txt`
- Copy `example.env` to `.env` if not present
- Start the FastAPI backend on port 8005
- Install npm dependencies and start the React frontend on port 3000

## Project Structure

```
menugen/
├── backend/           # FastAPI Python backend (code only)
├── frontend/          # React TypeScript frontend (code only)
├── Dockerfile         # Multi-target Dockerfile (backend + frontend targets)
├── docker-compose.yml # Container orchestration
├── requirements.txt   # Python dependencies
├── example.env        # Environment template
├── start_menugen.sh   # Local dev start script
├── stop_menugen.sh    # Local dev stop script
└── status_menugen.sh  # Local dev status script
```

## Development Notes
- The backend serves generated images at `/images/` (mounted as a volume for persistence).
- Update Python dependencies in `requirements.txt` and frontend dependencies in `frontend/package.json`.

## Troubleshooting
- Ensure your API keys are valid and have access to the required models.
- If you change code, re-run `docker compose up --build` to rebuild images.
- For logs, check the container output or use `docker compose logs backend` / `docker compose logs frontend`.

---

**Enjoy MenuGen!**
