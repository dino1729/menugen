#!/bin/bash
# status_menugen.sh - Check status of the MenuGen application

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="$SCRIPT_DIR/.pids"
LOG_DIR="$SCRIPT_DIR/logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}   MenuGen Application Status${NC}"
echo -e "${CYAN}========================================${NC}"

# Check Backend
echo -e "\n${CYAN}Backend Status:${NC}"
BACKEND_RUNNING=false
if [ -f "$PID_DIR/backend.pid" ]; then
    BACKEND_PID=$(cat "$PID_DIR/backend.pid")
    if kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo -e "  Status: ${GREEN}RUNNING${NC}"
        echo -e "  PID: $BACKEND_PID"
        echo -e "  URL: http://localhost:8005"
        BACKEND_RUNNING=true
    else
        echo -e "  Status: ${RED}STOPPED${NC} (stale PID file)"
    fi
else
    echo -e "  Status: ${RED}STOPPED${NC}"
fi

# Also check if something is listening on port 8005
PORT_8005=$(lsof -ti:8005 2>/dev/null || true)
if [ -n "$PORT_8005" ] && [ "$BACKEND_RUNNING" = false ]; then
    echo -e "  ${YELLOW}Warning: Port 8005 is in use by PID $PORT_8005 (not managed by this script)${NC}"
fi

# Check Frontend
echo -e "\n${CYAN}Frontend Status:${NC}"
FRONTEND_RUNNING=false
if [ -f "$PID_DIR/frontend.pid" ]; then
    FRONTEND_PID=$(cat "$PID_DIR/frontend.pid")
    if kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo -e "  Status: ${GREEN}RUNNING${NC}"
        echo -e "  PID: $FRONTEND_PID"
        echo -e "  URL: http://localhost:3000"
        FRONTEND_RUNNING=true
    else
        echo -e "  Status: ${RED}STOPPED${NC} (stale PID file)"
    fi
else
    echo -e "  Status: ${RED}STOPPED${NC}"
fi

# Also check if something is listening on port 3000
PORT_3000=$(lsof -ti:3000 2>/dev/null || true)
if [ -n "$PORT_3000" ] && [ "$FRONTEND_RUNNING" = false ]; then
    echo -e "  ${YELLOW}Warning: Port 3000 is in use by PID $PORT_3000 (not managed by this script)${NC}"
fi

# Health check for backend API
echo -e "\n${CYAN}API Health Check:${NC}"
if [ "$BACKEND_RUNNING" = true ]; then
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8005/ 2>/dev/null || echo "000")
    if [ "$HTTP_STATUS" = "200" ]; then
        echo -e "  API Response: ${GREEN}OK (HTTP $HTTP_STATUS)${NC}"
    else
        echo -e "  API Response: ${YELLOW}HTTP $HTTP_STATUS${NC}"
    fi
else
    echo -e "  API Response: ${RED}N/A (backend not running)${NC}"
fi

# Log files info
echo -e "\n${CYAN}Log Files:${NC}"
if [ -f "$LOG_DIR/backend.log" ]; then
    BACKEND_LOG_SIZE=$(du -h "$LOG_DIR/backend.log" 2>/dev/null | cut -f1)
    BACKEND_LOG_LINES=$(wc -l < "$LOG_DIR/backend.log" 2>/dev/null)
    echo -e "  Backend: $LOG_DIR/backend.log ($BACKEND_LOG_SIZE, $BACKEND_LOG_LINES lines)"
else
    echo -e "  Backend: ${YELLOW}No log file${NC}"
fi

if [ -f "$LOG_DIR/frontend.log" ]; then
    FRONTEND_LOG_SIZE=$(du -h "$LOG_DIR/frontend.log" 2>/dev/null | cut -f1)
    FRONTEND_LOG_LINES=$(wc -l < "$LOG_DIR/frontend.log" 2>/dev/null)
    echo -e "  Frontend: $LOG_DIR/frontend.log ($FRONTEND_LOG_SIZE, $FRONTEND_LOG_LINES lines)"
else
    echo -e "  Frontend: ${YELLOW}No log file${NC}"
fi

# Configuration check (now using root .env)
echo -e "\n${CYAN}Configuration:${NC}"
if [ -f "$SCRIPT_DIR/.env" ]; then
    echo -e "  .env: ${GREEN}Present${NC}"
    # Show key config values (without revealing secrets)
    if grep -q "IMAGE_PROVIDER" "$SCRIPT_DIR/.env" 2>/dev/null; then
        IMG_PROVIDER=$(grep "IMAGE_PROVIDER" "$SCRIPT_DIR/.env" | cut -d'=' -f2)
        echo -e "  Image Provider: $IMG_PROVIDER"
    fi
    if grep -q "LLM_MODEL" "$SCRIPT_DIR/.env" 2>/dev/null; then
        LLM_MODEL=$(grep "LLM_MODEL" "$SCRIPT_DIR/.env" | cut -d'=' -f2)
        echo -e "  LLM Model: $LLM_MODEL"
    fi
    if grep -q "IMAGE_GEN_MODEL" "$SCRIPT_DIR/.env" 2>/dev/null; then
        IMG_MODEL=$(grep "IMAGE_GEN_MODEL" "$SCRIPT_DIR/.env" | cut -d'=' -f2)
        echo -e "  Image Gen Model: $IMG_MODEL"
    fi
else
    echo -e "  .env: ${RED}Missing${NC} (copy from example.env)"
fi

# Summary
echo -e "\n${CYAN}========================================${NC}"
if [ "$BACKEND_RUNNING" = true ] && [ "$FRONTEND_RUNNING" = true ]; then
    echo -e "${GREEN}All services are running!${NC}"
    echo -e "Access the app at: ${GREEN}http://localhost:3000${NC}"
elif [ "$BACKEND_RUNNING" = true ] || [ "$FRONTEND_RUNNING" = true ]; then
    echo -e "${YELLOW}Some services are not running${NC}"
    echo -e "Run ${YELLOW}./start_menugen.sh${NC} to start all services"
else
    echo -e "${RED}No services are running${NC}"
    echo -e "Run ${YELLOW}./start_menugen.sh${NC} to start the application"
fi
echo -e "${CYAN}========================================${NC}"
