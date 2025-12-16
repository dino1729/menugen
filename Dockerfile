# ==============================================================================
# MenuGen Multi-target Dockerfile
# Build with: docker build --target backend -t menugen-backend .
#             docker build --target frontend -t menugen-frontend .
# ==============================================================================

# ------------------------------------------------------------------------------
# Backend target: Python FastAPI service
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS backend

WORKDIR /app

# Install dependencies from root requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY backend/ ./

# Create images directory
RUN mkdir -p data/images

EXPOSE 8005

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005"]

# ------------------------------------------------------------------------------
# Frontend build stage: Node.js to build React app
# ------------------------------------------------------------------------------
FROM node:20-alpine AS frontend-build

WORKDIR /app

# Install dependencies
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install

# Copy frontend source and build
COPY frontend/ ./
RUN npm run build

# ------------------------------------------------------------------------------
# Frontend target: nginx serving static React build
# ------------------------------------------------------------------------------
FROM nginx:alpine AS frontend

COPY --from=frontend-build /app/build /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]

