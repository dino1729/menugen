# MenuGen: Menu Parser & AI Image Generator

MenuGen combines a FastAPI backend with a React/TypeScript frontend. It parses uploaded menu images through a LiteLLM-compatible vision model, improves item descriptions, and generates dish illustrations through LiteLLM or NVIDIA NIM.

## Prerequisites

- Python 3.11 or newer
- Node.js 20.19 or newer
- A reachable LiteLLM proxy and API key
- An NVIDIA API key only when using the direct NVIDIA provider

## Setup and Run

Copy the environment template and add real credentials:

```bash
cp example.env .env
```

Start both local services:

```bash
./start_menugen.sh
```

The script creates or repairs `.venv/`, installs Python and npm dependencies when needed, and binds both services to loopback:

- Frontend: <http://localhost:3000>
- Backend API: <http://localhost:8005>

Manage the processes with:

```bash
./status_menugen.sh
./stop_menugen.sh
```

## Testing and Building

```bash
# Backend unit and API tests
.venv/bin/python -m pytest -q

# Frontend tests and production build
cd frontend
npm test
npm run build
```

Tests that use the `litellm_config` fixture perform a live proxy preflight and may invoke configured models. Unit tests do not require the proxy.

## Project Structure

```text
backend/
  main.py                    FastAPI routes, sessions, and WebSockets
  config.py                  Environment and model configuration
  litellm_client.py          High-level parsing and generation workflow
  litellm_proxy_client.py    OpenAI-compatible HTTP client
  nvidia_image_generation.py NVIDIA NIM image generation
frontend/
  src/App.tsx                React interface and session client
  src/App.test.tsx           Vitest interaction regressions
  vite.config.ts             Development and test configuration
tests/                       Pytest unit, API, and integration tests
```

Generated images are isolated under `backend/data/images/<session-id>/`. Runtime logs and PID files live in `logs/` and `.pids/`; all are ignored by Git.

## Configuration and Security

Configuration precedence is environment variables, then `backend/config.json`, then built-in defaults. Copy `backend/config.example.json` to customize model allowlists and retry settings.

The paid upload endpoints accept only requests from the local frontend, validate model choices, and enforce a 10 MB JPEG/PNG/GIF limit. The backend binds to `127.0.0.1`. To add another trusted local origin, set `MENUGEN_ALLOWED_ORIGINS` to a comma-separated list. Never commit `.env`, `backend/config.json`, generated media, or logs.
