# Repository Guidelines

## Project Structure & Module Organization

MenuGen pairs a Python FastAPI service with a React/TypeScript client. Backend entrypoints and integrations live in `backend/`: `main.py` defines the API, `config.py` owns configuration, and `litellm_*` and `nvidia_image_generation.py` wrap model providers. Treat `backend/archive/` and `backend/backup/` as reference code, not active implementations. Frontend code is under `frontend/src/`, with static assets in `frontend/public/`. Python tests live in `tests/`; generated images and other runtime output belong in the ignored `backend/data/images/` and `tests/outputs/` directories.

## Build, Test, and Development Commands

- `./start_menugen.sh`: create/install local dependencies as needed, then start FastAPI on port 8005 and React on port 3000.
- `./status_menugen.sh` / `./stop_menugen.sh`: inspect or stop the local services.
- `.venv/bin/python -m pytest -q`: run the Python suite. Tests using `litellm_config` perform a live proxy preflight; unit tests remain offline.
- `cd frontend && npm test`: run the Vitest and React Testing Library suite once.
- `cd frontend && npm run build`: create a production frontend build and surface TypeScript/build errors.

## Coding Style & Naming Conventions

Use four-space indentation and standard Python conventions: `snake_case` for functions and modules, `PascalCase` for classes, and explicit type hints where they clarify API boundaries. In TypeScript, follow the existing two-space style, use `PascalCase` for React components and interfaces, and `camelCase` for variables and helpers. Keep provider-specific logic in its client module and centralize configuration in `backend/config.py`. The frontend uses Vite and TypeScript; no repository-wide autoformatter is configured.

## Testing Guidelines

Name Python files and functions `test_*.py` and `test_*`; use `pytest.mark.asyncio` for async API tests. Place shared fixtures in `tests/conftest.py`. Frontend tests use `*.test.tsx`. Add focused regression coverage for configuration, endpoints, and provider behavior. No numeric coverage threshold is enforced, but all relevant tests should pass before review.

## Commit & Pull Request Guidelines

Recent history favors concise imperative subjects, often with Conventional Commit prefixes such as `feat:`, `fix:`, and `refactor:`. Keep each commit scoped to one logical change. Pull requests should explain user-visible behavior, configuration changes, and verification commands; link related issues and include screenshots for UI changes. Never commit `.env`, API keys, `backend/config.json`, logs, generated media, or test outputs.
