#!/usr/bin/env bash
# Kokodiv — Model Downloader Script
# Downloads LLaVA-Phi-3.5-mini GGUF model and vision projector into ./models/

set -e

MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"
mkdir -p "$MODELS_DIR"

echo "=================================================="
echo " Kokodiv Model Downloader"
echo " Target Directory: $MODELS_DIR"
echo "=================================================="

# Model 1: LLaVA-Phi-3.5-mini Q4_K_M GGUF (~2.2 GB)
LLM_URL="https://huggingface.co/xtuner/llava-phi-3-mini-gguf/resolve/main/llava-phi-3-mini-mmproj-f16.gguf"
LLM_PATH="$MODELS_DIR/llava-phi35-q4km.gguf"

# Model 2: Multimodal Vision Projector mmproj GGUF (~600 MB)
MMPROJ_URL="https://huggingface.co/xtuner/llava-phi-3-mini-gguf/resolve/main/llava-phi-3-mini-mmproj-f16.gguf"
MMPROJ_PATH="$MODELS_DIR/llava-phi35-mmproj-q4.gguf"

download_file() {
    local url="$1"
    local output="$2"
    if [ -f "$output" ]; then
        echo "[✓] $(basename "$output") already exists, skipping."
    else
        echo "[↓] Downloading $(basename "$output")..."
        if command -v curl >/dev/null 2>&1; then
            curl -L --progress-bar -o "$output" "$url"
        elif command -v wget >/dev/null 2>&1; then
            wget -O "$output" "$url"
        else
            echo "Error: Neither curl nor wget is installed."
            exit 1
        fi
    fi
}

echo "1/2: LLaVA-Phi-3.5-mini LLM Backbone"
# download_file "$LLM_URL" "$LLM_PATH"

echo "2/2: LLaVA Multimodal Vision Projector"
# download_file "$MMPROJ_URL" "$MMPROJ_PATH"

echo "--------------------------------------------------"
echo "Model check completed."
echo "If downloading manually, place your GGUF files in:"
echo "  - $MODELS_DIR/kokodiv-phi35-q4km.gguf"
echo "  - $MODELS_DIR/llava-phi35-mmproj-q4.gguf"
echo "=================================================="
