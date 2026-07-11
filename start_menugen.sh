#!/bin/bash
# start_menugen.sh - Start the MenuGen application (backend + frontend)
# Runs the local FastAPI and Vite development servers.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
VENV_DIR="$SCRIPT_DIR/.venv"
PID_DIR="$SCRIPT_DIR/.pids"
LOG_DIR="$SCRIPT_DIR/logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   Starting MenuGen Application${NC}"
echo -e "${GREEN}========================================${NC}"

# Create directories if they don't exist
mkdir -p "$PID_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$BACKEND_DIR/data/images"

# Check if already running
if [ -f "$PID_DIR/backend.pid" ]; then
    BACKEND_PID=$(cat "$PID_DIR/backend.pid")
    if kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo -e "${YELLOW}Backend is already running (PID: $BACKEND_PID)${NC}"
        BACKEND_RUNNING=true
    else
        rm "$PID_DIR/backend.pid"
        BACKEND_RUNNING=false
    fi
else
    BACKEND_RUNNING=false
fi

if [ -f "$PID_DIR/frontend.pid" ]; then
    FRONTEND_PID=$(cat "$PID_DIR/frontend.pid")
    if kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo -e "${YELLOW}Frontend is already running (PID: $FRONTEND_PID)${NC}"
        FRONTEND_RUNNING=true
    else
        rm "$PID_DIR/frontend.pid"
        FRONTEND_RUNNING=false
    fi
else
    FRONTEND_RUNNING=false
fi

# Start Backend
if [ "$BACKEND_RUNNING" = false ]; then
    echo -e "\n${GREEN}[1/2] Starting Backend...${NC}"
    
    cd "$SCRIPT_DIR"

    # Recreate missing or stale virtual environments instead of trusting the directory alone.
    if [ ! -x "$VENV_DIR/bin/python" ] || [ ! -x "$VENV_DIR/bin/uvicorn" ]; then
        echo -e "${YELLOW}Creating or repairing the Python virtual environment...${NC}"
        python3 -m venv --clear "$VENV_DIR"
        source "$VENV_DIR/bin/activate"
        echo "Installing dependencies..."
        pip install -r requirements.txt
    else
        echo "Activating Python virtual environment..."
        source "$VENV_DIR/bin/activate"
    fi
    
    # Check if root .env exists
    if [ ! -f ".env" ]; then
        if [ -f "example.env" ]; then
            echo -e "${YELLOW}No .env file found. Copying from example.env...${NC}"
            cp example.env .env
            echo -e "${YELLOW}Please review and update .env with your configuration${NC}"
        else
            echo -e "${RED}ERROR: No .env or example.env file found${NC}"
            exit 1
        fi
    fi
    
    # Start uvicorn in background (use full path to venv's uvicorn for nohup)
    echo "Starting uvicorn server on port 8005..."
    nohup "$VENV_DIR/bin/uvicorn" --app-dir backend main:app --host 127.0.0.1 --port 8005 > "$LOG_DIR/backend.log" 2>&1 &
    BACKEND_PID=$!
    echo "$BACKEND_PID" > "$PID_DIR/backend.pid"
    
    # Wait a moment and verify it started
    sleep 2
    if kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo -e "${GREEN}Backend started successfully (PID: $BACKEND_PID)${NC}"
        echo -e "  - API: http://localhost:8005"
        echo -e "  - Logs: $LOG_DIR/backend.log"
    else
        echo -e "${RED}ERROR: Backend failed to start. Check $LOG_DIR/backend.log${NC}"
        exit 1
    fi
fi

# Start Frontend
if [ "$FRONTEND_RUNNING" = false ]; then
    echo -e "\n${GREEN}[2/2] Starting Frontend...${NC}"
    
    cd "$FRONTEND_DIR"
    
    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        echo "Installing npm dependencies..."
        npm install
    fi
    
    # Start React dev server in background
    echo "Starting React development server on port 3000..."
    nohup npm start > "$LOG_DIR/frontend.log" 2>&1 &
    FRONTEND_PID=$!
    echo "$FRONTEND_PID" > "$PID_DIR/frontend.pid"
    
    # Wait a moment and verify it started
    sleep 3
    if kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo -e "${GREEN}Frontend started successfully (PID: $FRONTEND_PID)${NC}"
        echo -e "  - UI: http://localhost:3000"
        echo -e "  - Logs: $LOG_DIR/frontend.log"
    else
        echo -e "${RED}ERROR: Frontend failed to start. Check $LOG_DIR/frontend.log${NC}"
        exit 1
    fi
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}   MenuGen Application Started!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "\nAccess the application at: ${GREEN}http://localhost:3000${NC}"
echo -e "Backend API available at:  ${GREEN}http://localhost:8005${NC}"
echo -e "\nUse ${YELLOW}./stop_menugen.sh${NC} to stop the application"
echo -e "Use ${YELLOW}./status_menugen.sh${NC} to check status"
