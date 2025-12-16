#!/bin/bash
# stop_menugen.sh - Stop the MenuGen application (backend + frontend)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="$SCRIPT_DIR/.pids"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${RED}========================================${NC}"
echo -e "${RED}   Stopping MenuGen Application${NC}"
echo -e "${RED}========================================${NC}"

STOPPED_SOMETHING=false

# Stop Frontend
if [ -f "$PID_DIR/frontend.pid" ]; then
    FRONTEND_PID=$(cat "$PID_DIR/frontend.pid")
    if kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo -e "\n${YELLOW}Stopping Frontend (PID: $FRONTEND_PID)...${NC}"
        # Kill the process group to ensure all child processes are terminated
        kill -TERM -"$FRONTEND_PID" 2>/dev/null || kill -TERM "$FRONTEND_PID" 2>/dev/null
        sleep 1
        # Force kill if still running
        if kill -0 "$FRONTEND_PID" 2>/dev/null; then
            kill -9 "$FRONTEND_PID" 2>/dev/null
        fi
        echo -e "${GREEN}Frontend stopped${NC}"
        STOPPED_SOMETHING=true
    else
        echo -e "${YELLOW}Frontend process not found (stale PID file)${NC}"
    fi
    rm -f "$PID_DIR/frontend.pid"
else
    echo -e "${YELLOW}No frontend PID file found${NC}"
fi

# Stop Backend
if [ -f "$PID_DIR/backend.pid" ]; then
    BACKEND_PID=$(cat "$PID_DIR/backend.pid")
    if kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo -e "\n${YELLOW}Stopping Backend (PID: $BACKEND_PID)...${NC}"
        kill -TERM "$BACKEND_PID" 2>/dev/null
        sleep 1
        # Force kill if still running
        if kill -0 "$BACKEND_PID" 2>/dev/null; then
            kill -9 "$BACKEND_PID" 2>/dev/null
        fi
        echo -e "${GREEN}Backend stopped${NC}"
        STOPPED_SOMETHING=true
    else
        echo -e "${YELLOW}Backend process not found (stale PID file)${NC}"
    fi
    rm -f "$PID_DIR/backend.pid"
else
    echo -e "${YELLOW}No backend PID file found${NC}"
fi

# Also kill any orphaned processes on the ports
echo -e "\n${YELLOW}Checking for orphaned processes on ports...${NC}"

# Check port 8005 (backend)
ORPHAN_BACKEND=$(lsof -ti:8005 2>/dev/null || true)
if [ -n "$ORPHAN_BACKEND" ]; then
    echo -e "Killing orphaned process on port 8005 (PID: $ORPHAN_BACKEND)"
    kill -9 $ORPHAN_BACKEND 2>/dev/null || true
    STOPPED_SOMETHING=true
fi

# Check port 3000 (frontend)
ORPHAN_FRONTEND=$(lsof -ti:3000 2>/dev/null || true)
if [ -n "$ORPHAN_FRONTEND" ]; then
    echo -e "Killing orphaned process on port 3000 (PID: $ORPHAN_FRONTEND)"
    kill -9 $ORPHAN_FRONTEND 2>/dev/null || true
    STOPPED_SOMETHING=true
fi

if [ "$STOPPED_SOMETHING" = true ]; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}   MenuGen Application Stopped${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "\n${YELLOW}No running MenuGen processes found${NC}"
fi

