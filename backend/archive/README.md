# Archive - Deprecated Implementations

This folder contains deprecated/legacy implementations that are kept for reference.
These files are NOT used by the current application.

## Contents

| File | Description | Replaced By |
|------|-------------|-------------|
| `nvidia_client.py` | Legacy NVIDIA API client | `nvidia_image_generation.py` |
| `nvidia_description.py` | Legacy NVIDIA description generation | `litellm_client.py` |
| `nvidia_image_parser.py` | Legacy NVIDIA image parsing | `litellm_client.py` |
| `openai_client.py` | Legacy OpenAI SDK direct client | `litellm_proxy_client.py` |
| `openai_description.py` | Legacy OpenAI description generation | `litellm_client.py` |
| `openai_image_generation.py` | Legacy OpenAI image generation | `litellm_client.py` |
| `openai_image_parser.py` | Legacy OpenAI image parsing | `litellm_client.py` |

## Current Architecture

The application now uses:

- **`litellm_proxy_client.py`** - Low-level httpx client for OpenAI-compatible API calls
- **`litellm_client.py`** - High-level functions for vision, text, and image generation
- **`nvidia_image_generation.py`** - Direct NVIDIA NIM API for image generation (optional)
- **`config.py`** - Centralized configuration management

All LLM calls are routed through a LiteLLM proxy for unified API access.

## Note

Do not delete these files without verifying no imports exist.
Git history preserves all changes if needed for reference.
