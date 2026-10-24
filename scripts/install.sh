#!/bin/bash
# Install Call Coach on this Mac: whisper.cpp, the capture helper, Python
# packages, the Whisper model and (optionally) your Jev key. It runs only
# when you start it. Safe to run again.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
SUPPORT="$HOME/Library/Application Support/call-coach"
CONFIG_DIR="$HOME/.config/call-coach"
MODEL="${CALL_COACH_MODEL:-ggml-small.en.bin}"

command -v brew >/dev/null || { echo "Homebrew is required: https://brew.sh"; exit 1; }
if ! xcode-select -p >/dev/null 2>&1 || ! command -v swiftc >/dev/null; then
  echo "Apple's Command Line Tools are required to build the audio helper."
  echo "Run this, finish the installer window that opens, then run ./scripts/install.sh again:"
  echo "    xcode-select --install"
  exit 1
fi
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

# Save the Jev key once, so it does not need setting in every new terminal.
# An existing config (for example one pointing at a key proxy) is left alone.
if [ -f "$CONFIG_DIR/config.json" ]; then
  echo "Keeping your existing $CONFIG_DIR/config.json."
elif [ -t 0 ]; then
  echo
  echo "Paste your Jev API key from https://console.typesafe.ai/keys and press Return."
  read -rsp "(Typing is hidden. Press Return on its own to skip.) " key
  echo
  if [ -n "$key" ]; then
    mkdir -p "$CONFIG_DIR"
    chmod 700 "$CONFIG_DIR"
    ( umask 077; printf '%s\n' "$key" > "$CONFIG_DIR/jev-key" )
    printf '{"jev": {"api_key_file": "~/.config/call-coach/jev-key"}}\n' > "$CONFIG_DIR/config.json"
    echo "Saved. Only your user account can read $CONFIG_DIR/jev-key."
  else
    echo "Skipped. Without a key you tick questions by hand; run ./scripts/install.sh again to add one."
  fi
  unset key
fi

echo
echo "Installed. Start it with:  $ROOT/.venv/bin/call-coach"
echo "Then open http://localhost:4800"
