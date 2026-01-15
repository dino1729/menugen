# MenuGen: Menu Parser & AI Image Generator

This project consists of a FastAPI backend (Python) and a React frontend. The backend parses menu images and generates dish illustrations using AI models via a LiteLLM proxy or direct NVIDIA API. The frontend provides a modern UI for uploading menus and viewing results in real time.

## Quick Start: Docker Compose

### 1. Prerequisites
- [Docker](https://www.docker.com/get-started) and [Docker Compose](https://docs.docker.com/compose/) installed
- A LiteLLM proxy running (for unified model access), or NVIDIA API key for direct image generation

### 2. Setup Environment Variables
Create a `.env` file in the **project root** by copying from `example.env`:
```bash
cp example.env .env
```

Then edit `.env` with your configuration:
```bash
# OpenAI-compatible API (LiteLLM proxy)
OPENAI_BASE_URL=http://localhost:4000
OPENAI_API_KEY=your_api_key_here

# Default models (optional - can be overridden in config.json or UI)
VISION_MODEL=gpt-4o
DESCRIPTION_MODEL=gemini-3-flash-preview
IMAGE_GEN_MODEL=gemini-3-pro-image-preview
VIDEO_GEN_MODEL=veo-3.1-generate-001

# Image provider: "litellm" or "nvidia"
IMAGE_PROVIDER=litellm

# NVIDIA Direct API (only needed if IMAGE_PROVIDER=nvidia)
NVIDIA_API_KEY=your_nvidia_api_key_here
NVIDIA_IMAGE_GEN_URL=https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-3-medium

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
├── backend/
│   ├── config.py              # Central configuration module
│   ├── config.example.json    # Config template (committed)
│   ├── config.json            # Local config (gitignored)
│   ├── main.py                # FastAPI application
│   ├── litellm_client.py      # High-level LLM functions
│   ├── litellm_proxy_client.py # Low-level httpx client
│   ├── nvidia_image_generation.py # NVIDIA NIM API client
│   ├── client_utils.py        # Shared utilities
│   └── archive/               # Deprecated implementations
├── frontend/                  # React TypeScript frontend
├── Dockerfile                 # Multi-target Dockerfile
├── docker-compose.yml         # Container orchestration
├── requirements.txt           # Python dependencies
├── example.env                # Environment template
├── start_menugen.sh           # Local dev start script
├── stop_menugen.sh            # Local dev stop script
└── status_menugen.sh          # Local dev status script
```

## Configuration

### Environment Variables

All LLM calls route through a LiteLLM proxy using OpenAI-compatible API format.

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_BASE_URL` | Yes | LiteLLM proxy URL (e.g., `http://localhost:4000`) |
| `OPENAI_API_KEY` | Yes | API key for the LiteLLM proxy |
| `VISION_MODEL` | No | Model for menu image parsing (default: `gpt-4o`) |
| `DESCRIPTION_MODEL` | No | Model for description generation (default: `gemini-3-flash-preview`) |
| `IMAGE_GEN_MODEL` | No | Model for image generation (default: `gemini-3-pro-image-preview`) |
| `VIDEO_GEN_MODEL` | No | Model for video generation (default: `veo-3.1-generate-001`) |
| `IMAGE_PROVIDER` | No | `litellm` or `nvidia` (default: `litellm`) |
| `NVIDIA_API_KEY` | No* | Required only if `IMAGE_PROVIDER=nvidia` |
| `NVIDIA_IMAGE_GEN_URL` | No* | NVIDIA API endpoint for image generation |
| `LOG_LEVEL` | No | Logging level (default: `INFO`) |

### Advanced Configuration (config.json)

For advanced settings, create `backend/config.json` (gitignored). Copy from the template:

```bash
cp backend/config.example.json backend/config.json
```

The config file supports:
- **Curated model whitelists** - Control which models appear in the UI dropdowns
- **NVIDIA model parameters** - Per-model settings for Stable Diffusion and FLUX models
- **Retry configuration** - Customize retry behavior for API calls

Example `config.json`:
```json
{
  "defaults": {
    "vision_model": "gpt-4o",
    "description_model": "gemini-3-flash-preview",
    "image_gen_model": "gemini-3-pro-image-preview",
    "video_gen_model": "veo-3.1-generate-001"
  },
  "whitelists": {
    "vision_models": ["gpt-4o", "gpt-4.1", "gemini-2.5-pro"],
    "text_models": ["gpt-4o", "gemini-3-flash-preview", "gemini-2.5-flash"],
    "image_models": ["gemini-3-pro-image-preview"],
    "video_models": ["veo-3.1-generate-001", "veo-3.0-generate-001"]
  },
  "nvidia": {
    "base_url": "https://ai.api.nvidia.com/v1/genai",
    "models": {
      "stabilityai/stable-diffusion-3.5-large": {
        "api_params": {"cfg_scale": 5, "steps": 50, "aspect_ratio": "1:1"}
      },
      "black-forest-labs/flux.1-schnell": {
        "api_params": {"cfg_scale": 0, "steps": 4, "width": 1024, "height": 1024}
      }
    }
  },
  "retry": {
    "max_retries": 5,
    "initial_backoff_seconds": 1.0,
    "max_backoff_seconds": 60.0,
    "jitter_factor": 0.1,
    "retry_status_codes": [429, 408, 500, 502, 503, 504]
  }
}
```

Configuration priority: Environment variables > config.json > built-in defaults

### Startup Validation

The backend validates LiteLLM proxy connectivity on startup. If the proxy is unreachable, the app will fail fast with a clear error message rather than failing at runtime.

## Development Notes
- The backend serves generated images at `/images/` (mounted as a volume for persistence).
- Update Python dependencies in `requirements.txt` and frontend dependencies in `frontend/package.json`.
- Legacy/deprecated implementations are archived in `backend/archive/` for reference.

## Troubleshooting
- Ensure your API keys are valid and have access to the required models.
- If you change code, re-run `docker compose up --build` to rebuild images.
- For logs, check the container output or use `docker compose logs backend` / `docker compose logs frontend`.

---

**Enjoy MenuGen!**
