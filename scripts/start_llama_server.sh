#!/usr/bin/env bash
# Script to launch local llama.cpp server on port 8080

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

MODEL_FILE="$BASE_DIR/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

if [ ! -f "$MODEL_FILE" ]; then
    # Fallback to any gguf in models directory
    ALT_MODEL=$(find "$BASE_DIR/models" -name "*.gguf" | head -n 1)
    if [ -n "$ALT_MODEL" ]; then
        MODEL_FILE="$ALT_MODEL"
    else
        echo "[!] Model file not found. Running download script..."
        ./scripts/download_model.sh
    fi
fi

echo "=================================================="
echo " Starting llama.cpp LLM Server"
echo " Model: $MODEL_FILE"
echo " Port: 8080"
echo "=================================================="

# Kill existing process on port 8080
pkill -f "llama-server" 2>/dev/null || true
pkill -f "llama_cpp.server" 2>/dev/null || true
fuser -k 8080/tcp 2>/dev/null || true
sleep 0.5

LLAMA_BIN=""
if [ -f "$BASE_DIR/vendor/llama.cpp/build/bin/llama-server" ]; then
    LLAMA_BIN="$BASE_DIR/vendor/llama.cpp/build/bin/llama-server"
elif command -v llama-server >/dev/null 2>&1; then
    LLAMA_BIN="llama-server"
fi

if [ -n "$LLAMA_BIN" ]; then
    echo "[✓] Found binary: $LLAMA_BIN"
    exec "$LLAMA_BIN" -m "$MODEL_FILE" --port 8080 -c 2048 -t 4 --host 127.0.0.1
else
    echo "[✓] Launching via python llama_cpp.server module..."
    PYTHON_BIN="python3"
    if [ -d "$BASE_DIR/.venv" ]; then
        PYTHON_BIN="$BASE_DIR/.venv/bin/python3"
    fi
    exec $PYTHON_BIN -m llama_cpp.server --model "$MODEL_FILE" --port 8080 --host 127.0.0.1 --n_threads 4 --n_ctx 2048
fi
