# MenuGen: Menu Parser & AI Image Generator

This project consists of a FastAPI backend (Python) and a React frontend. The backend parses menu images and generates dish illustrations using AI models (NVIDIA Stable Diffusion 3 or OpenAI DALL-E 3). The frontend provides a modern UI for uploading menus and viewing results in real time.

## Quick Start: Docker Compose

### 1. Prerequisites
- [Docker](https://www.docker.com/get-started) and [Docker Compose](https://docs.docker.com/compose/) installed
- An NVIDIA API key (for image generation) or OpenAI API key (alternative)

### 2. Setup Environment Variables
Create a `.env` file in the `backend/` directory with at least:
```
# Required: API Keys
NVIDIA_API_KEY=nvapi-...
OPENAI_API_KEY=sk-...

# Required: NVIDIA API Base URLs (must be explicitly set - no defaults)
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_IMAGE_GEN_URL=https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-3-medium

# Optional: OpenAI API Base URL
# OPENAI_API_BASE=https://api.openai.com/v1

# Optional: OpenAI Model Configuration
# VISION_MODEL=gpt-4o
# IMAGE_GEN_MODEL=dall-e-3
# DESCRIPTION_MODEL=gpt-4o-mini

# Optional: NVIDIA Model Configuration
# NVIDIA_VISION_MODEL=microsoft/phi-4-multimodal-instruct
# NVIDIA_TEXT_MODEL=openai/gpt-oss-20b

# Optional: Logging
# LOG_LEVEL=INFO
```

**Security Note:** NVIDIA API base URLs are now **required** environment variables with no hardcoded defaults. This enforces:
- Explicit configuration of API endpoints
- No outdated or insecure hardcoded URLs in source code
- Easy updates when NVIDIA changes their endpoints
- Environment-specific configurations (dev/staging/prod)

See `backend/ENV_CONFIGURATION.md` for detailed documentation of all environment variables.

**Note:** The app defaults to NVIDIA models for all AI operations:
- **Vision (menu parsing)**: Microsoft Phi-4 Multimodal Instruct
- **Text (descriptions)**: OpenAI GPT-OSS-20B
- **Image generation**: Stable Diffusion 3

You can switch to OpenAI models (GPT-4 Vision, GPT-4, DALL-E 3) using the model selector in the web interface.

### 3. Build and Run
From the project root, run:
```
docker compose up --build
```
- The backend will be available at [http://localhost:8005](http://localhost:8005)
- The frontend will be available at [http://localhost:3005](http://localhost:3005)

### 4. Usage
- Open the frontend in your browser: [http://localhost:3005](http://localhost:3005)
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
