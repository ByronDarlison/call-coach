#!/bin/bash
# Install Call Coach on this Mac: whisper.cpp, the capture helper, Python
# packages and the Whisper model. Pass --login to start it at every login.
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

if [ "${1:-}" = "--login" ]; then
  PLIST="$HOME/Library/LaunchAgents/com.callcoach.service.plist"
  cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.callcoach.service</string>
  <key>ProgramArguments</key><array>
    <string>$ROOT/.venv/bin/python</string><string>-m</string><string>callcoach</string>
  </array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>EnvironmentVariables</key><dict><key>PATH</key><string>/opt/homebrew/bin:/usr/bin:/bin</string></dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$SUPPORT/service.log</string>
  <key>StandardErrorPath</key><string>$SUPPORT/service.log</string>
</dict></plist>
PLIST
  launchctl bootout "gui/$(id -u)" "$PLIST" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$PLIST"
  echo "Call Coach starts at login. Open http://localhost:4800"
else
  echo "Installed. Run: $ROOT/.venv/bin/call-coach   then open http://localhost:4800"
fi
