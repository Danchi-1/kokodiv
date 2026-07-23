#!/usr/bin/env bash
# Script to download ultra-fast GGUF LLM model for Kokodiv (CPU & low RAM optimized)

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$BASE_DIR/models"
MODEL_FILE="$MODEL_DIR/qwen2.5-0.5b-instruct-q4_k_m.gguf"
TMP_FILE="$MODEL_DIR/qwen2.5-0.5b-instruct-q4_k_m.gguf.tmp"
MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_FILE" ]; then
    echo "[✓] GGUF LLM model file already exists at: $MODEL_FILE"
    exit 0
fi

echo "=================================================="
echo " Downloading Qwen2.5-0.5B GGUF Model (~398 MB)..."
echo " Destination: $MODEL_FILE"
echo "=================================================="

if command -v curl >/dev/null 2>&1; then
    curl -L -C - --fail --progress-bar -o "$TMP_FILE" "$MODEL_URL"
elif command -v wget >/dev/null 2>&1; then
    wget -c -O "$TMP_FILE" "$MODEL_URL"
else
    echo "[!] ERROR: Neither curl nor wget found. Please download $MODEL_URL manually to $MODEL_FILE"
    exit 1
fi

mv "$TMP_FILE" "$MODEL_FILE"
echo "[✓] Model download complete!"
