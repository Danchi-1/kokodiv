#!/usr/bin/env bash
# Kokodiv — One-Click Launcher Script
# Starts llama.cpp LLM server + Kokodiv App Server + Opens Browser Automatically

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "=================================================="
echo " Starting Kokodiv On-Device Multimodal AI System"
echo " Working Directory: $BASE_DIR"
echo "=================================================="

# Select Python binary (prefer active virtualenv or system python)
PYTHON_BIN="python3"
if [ -d "$BASE_DIR/.venv" ]; then
    PYTHON_BIN="$BASE_DIR/.venv/bin/python3"
fi

# 1. Verify required Python dependencies
echo "[1/4] Checking Python dependencies..."
if ! $PYTHON_BIN -c "import numpy, onnxruntime, PIL" 2>/dev/null; then
    echo "=================================================="
    echo " [!] ERROR: Missing required Python packages."
    echo " Please install dependencies by running:"
    echo "     pip install -r requirements.txt"
    echo "=================================================="
    exit 1
fi
echo "[✓] Python dependencies verified."

# 2. Start llama.cpp LLM server on port 8080 (if model available)
MODEL_EXISTS=$(find "$BASE_DIR/models" -name "*.gguf" | head -n 1)
LLAMA_PID=""

if [ -n "$MODEL_EXISTS" ] || [ -f "$BASE_DIR/vendor/llama.cpp/build/bin/llama-server" ]; then
    echo "[2/4] Launching local llama.cpp server on http://localhost:8080..."
    ./scripts/start_llama_server.sh &
    LLAMA_PID=$!
else
    echo "[2/4] Note: GGUF model downloading or not yet ready. Running in RAG-direct mode."
    echo "      (To download LLM model, run: ./scripts/download_model.sh)"
fi

# 3. Start Kokodiv HTTP App Server in background
echo "[3/4] Launching Kokodiv Application Server on http://localhost:8090..."
pkill -f "src/server/app.py 8090" 2>/dev/null || true
fuser -k 8090/tcp 2>/dev/null || true
sleep 0.5
$PYTHON_BIN src/server/app.py 8090 &
SERVER_PID=$!

# Ensure cleanup on exit
cleanup() {
    echo "Stopping Kokodiv processes..."
    kill $SERVER_PID 2>/dev/null || true
    if [ -n "$LLAMA_PID" ]; then
        kill $LLAMA_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT

# 4. Poll server healthcheck endpoint to confirm successful startup
echo "[4/4] Waiting for Kokodiv Application Server to initialize..."
HEALTH_URL="http://127.0.0.1:8090/api/health"
SERVER_READY=false

for i in {1..10}; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[!] Server process exited unexpectedly."
        exit 1
    fi

    if command -v curl >/dev/null 2>&1; then
        if curl -s "$HEALTH_URL" | grep -q "healthy"; then
            SERVER_READY=true
            break
        fi
    elif command -v wget >/dev/null 2>&1; then
        if wget -qO- "$HEALTH_URL" | grep -q "healthy"; then
            SERVER_READY=true
            break
        fi
    fi
    sleep 0.5
done

if [ "$SERVER_READY" = false ]; then
    echo "=================================================="
    echo " [!] ERROR: Kokodiv server failed health check."
    echo " Stopping startup without opening browser."
    echo "=================================================="
    exit 1
fi

echo "[✓] Server is healthy and responding!"
echo "Launching Kokodiv Standalone Application Window..."

URL="http://localhost:8090"

# Launch in Native Desktop Window Mode (Standalone App Window without browser tabs/address bar)
if command -v google-chrome >/dev/null 2>&1; then
    google-chrome --app="$URL" --user-data-dir="/tmp/kokodiv_app_profile" >/dev/null 2>&1 &
elif command -v chromium-browser >/dev/null 2>&1; then
    chromium-browser --app="$URL" --user-data-dir="/tmp/kokodiv_app_profile" >/dev/null 2>&1 &
elif command -v chromium >/dev/null 2>&1; then
    chromium --app="$URL" --user-data-dir="/tmp/kokodiv_app_profile" >/dev/null 2>&1 &
elif command -v brave-browser >/dev/null 2>&1; then
    brave-browser --app="$URL" --user-data-dir="/tmp/kokodiv_app_profile" >/dev/null 2>&1 &
elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$URL"
elif command -v open >/dev/null 2>&1; then
    open "$URL"
else
    echo "Please open your browser manually and visit: $URL"
fi

echo "=================================================="
echo " Kokodiv is running! Press Ctrl+C to stop."
echo "=================================================="

wait $SERVER_PID
