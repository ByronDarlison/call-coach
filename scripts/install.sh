#!/bin/bash
# Install Call Coach on this Mac: whisper.cpp, the capture helper, Python
# packages and the Whisper model. It runs only when you start it.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
SUPPORT="$HOME/Library/Application Support/call-coach"
MODEL="${CALL_COACH_MODEL:-ggml-small.en.bin}"

command -v brew >/dev/null || { echo "Homebrew is required: https://brew.sh"; exit 1; }
command -v whisper-server >/dev/null || brew install whisper-cpp
command -v uv >/dev/null || brew install uv

mkdir -p bin "$SUPPORT/models"
swiftc -O -o bin/cc-capture capture/main.swift
uv venv -q --allow-existing .venv
uv pip install -q -p .venv/bin/python -e .

if [ ! -f "$SUPPORT/models/$MODEL" ]; then
  echo "Downloading Whisper model $MODEL (about 500 MB, once)..."
  curl -fL --progress-bar -o "$SUPPORT/models/$MODEL.part" \
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$MODEL"
  mv "$SUPPORT/models/$MODEL.part" "$SUPPORT/models/$MODEL"
fi

echo "Installed. Run: $ROOT/.venv/bin/call-coach   then open http://localhost:4800"
