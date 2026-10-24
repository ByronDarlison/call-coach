#!/bin/bash
# Install Call Coach on this Mac: whisper.cpp, the capture helper, Python packages, the Whisper model
# and (optionally) your Jev key. It runs only when you start it, never at login. Safe to run again.
#
#   ./scripts/install.sh            install or update
#   ./scripts/install.sh --set-key  also replace the saved Jev key
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
SUPPORT="$HOME/Library/Application Support/call-coach"
CONFIG_DIR="$HOME/.config/call-coach"
MODEL="ggml-small.en.bin"
MODEL_SHA256="c6138d6d58ecc8322097e0f987c32f1be8bb0a18532a3f88f734d1bbf9c41e5d"
MODEL_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$MODEL"  # can change, so the checksum is checked

[ "$(uname)" = "Darwin" ] || { echo "Call Coach runs on macOS only."; exit 1; }
major="$(sw_vers -productVersion | cut -d. -f1)"
[ "$major" -ge 15 ] || { echo "Call Coach needs macOS 15 or later (it records the microphone through ScreenCaptureKit)."; exit 1; }
command -v brew >/dev/null || { echo "Homebrew is required: https://brew.sh"; exit 1; }
if ! xcode-select -p >/dev/null 2>&1 || ! command -v swiftc >/dev/null; then
  echo "Apple's Command Line Tools are required to build the audio helper."
  echo "Run this, finish the installer window that opens, then run ./scripts/install.sh again:"
  echo "    xcode-select --install"
  exit 1
fi
command -v whisper-server >/dev/null || brew install whisper-cpp
command -v uv >/dev/null || brew install uv

mkdir -p bin
mkdir -p -m 700 "$SUPPORT" "$SUPPORT/models"
swiftc -O -o bin/cc-capture capture/main.swift
bin/cc-capture --list-mics >/dev/null || { echo "The audio helper built but does not run."; exit 1; }
uv venv -q --allow-existing --python ">=3.10" .venv
uv pip sync -q -p .venv/bin/python requirements.lock
uv pip install -q -p .venv/bin/python --no-deps -e .

model_ok() { [ "$(shasum -a 256 "$1" | cut -d' ' -f1)" = "$MODEL_SHA256" ]; }
if [ -f "$SUPPORT/models/$MODEL" ] && ! model_ok "$SUPPORT/models/$MODEL"; then
  echo "The saved Whisper model is damaged; downloading it again."
  rm "$SUPPORT/models/$MODEL"
fi
if [ ! -f "$SUPPORT/models/$MODEL" ]; then
  echo "Downloading Whisper model $MODEL (about 500 MB, once)..."
  curl -fL --progress-bar -o "$SUPPORT/models/$MODEL.part" "$MODEL_URL"
  model_ok "$SUPPORT/models/$MODEL.part" || { rm -f "$SUPPORT/models/$MODEL.part"; echo "Download failed its checksum."; exit 1; }
  mv "$SUPPORT/models/$MODEL.part" "$SUPPORT/models/$MODEL"
fi

# Save the Jev key once, so it does not need setting in every new terminal. An existing config (for
# example one pointing at a key broker) is kept unless you pass --set-key.
mkdir -p -m 700 "$CONFIG_DIR"
if { [ ! -f "$CONFIG_DIR/config.json" ] || [ "${1:-}" = "--set-key" ]; } && [ -t 0 ]; then
  echo
  echo "Paste your Jev API key from https://console.typesafe.ai/keys and press Return."
  read -rsp "(Typing is hidden. Press Return on its own to skip.) " key
  echo
  if [ -n "$key" ]; then
    ( umask 077; printf '%s\n' "$key" > "$CONFIG_DIR/jev-key.new" && mv "$CONFIG_DIR/jev-key.new" "$CONFIG_DIR/jev-key" )
    .venv/bin/python - "$CONFIG_DIR/config.json" <<'PY'
import json, os, sys
path = sys.argv[1]
config = json.load(open(path)) if os.path.exists(path) else {}
config["jev"] = {"api_key_file": "~/.config/call-coach/jev-key"}
open(path, "w").write(json.dumps(config, indent=1) + "\n")
PY
    echo "Saved. Only your user account can read $CONFIG_DIR/jev-key."
  else
    echo "Skipped. Without a key you tick questions by hand; run ./scripts/install.sh --set-key to add one."
  fi
  unset key
else
  echo "Keeping your existing $CONFIG_DIR/config.json (run with --set-key to replace the Jev key)."
fi

if [ -t 0 ] && [ ! -e "$HOME/Desktop/Call Coach.command" ]; then
  read -rp "Put a Call Coach launcher on your Desktop? [y/N] " answer
  if [ "${answer:-n}" = "y" ]; then
    sed "s|\${CALL_COACH_ROOT:-\$HOME/call-coach}|\${CALL_COACH_ROOT:-$ROOT}|" "scripts/Call Coach.command" > "$HOME/Desktop/Call Coach.command"
    chmod +x "$HOME/Desktop/Call Coach.command"
    echo "Done. Double-click it to start; the first time, macOS asks to let Terminal record the screen and audio."
  fi
fi

echo
echo "Installed. Check this Mac with:  $ROOT/.venv/bin/call-coach doctor"
echo "Start it with:                  $ROOT/.venv/bin/call-coach --open"
