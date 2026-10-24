#!/bin/bash
# Double-click to start Call Coach. Keep this window open during the call. Closing it (or pressing
# Ctrl-C) stops Call Coach, and a running call is saved first.
#
# macOS gives Screen & System Audio Recording permission to the app that runs this file (Terminal),
# so Terminal needs that permission. Run `call-coach doctor` here to check.
ROOT="${CALL_COACH_ROOT:-$HOME/call-coach}"
LOG="${CALL_COACH_LOG:-$HOME/Library/Logs/call-coach.log}"
cd "$ROOT" || { echo "Call Coach is not installed at $ROOT"; read -r; exit 1; }
mkdir -p "$(dirname "$LOG")"
if [ -f "$LOG" ] && [ "$(stat -f%z "$LOG")" -gt 1000000 ]; then mv "$LOG" "$LOG.1"; fi
if curl -s -o /dev/null "http://127.0.0.1:${CALL_COACH_PORT:-4800}/"; then
  echo "Call Coach is already running; opening it."
  .venv/bin/call-coach open
  exit 0
fi
echo "Starting Call Coach. Leave this window open until the call is over. Log: $LOG"
# Call Coach writes to the log, not this window, so closing the window cannot break its output.
.venv/bin/call-coach $([ -n "${CALL_COACH_NO_OPEN:-}" ] || echo --open) >>"$LOG" 2>&1 </dev/null &
service=$!
tail -n 0 -f "$LOG" &
viewer=$!
stop() {
  trap '' HUP INT TERM
  kill -TERM "$service" 2>/dev/null
  wait "$service" 2>/dev/null   # Call Coach saves a running call before it exits
  kill "$viewer" 2>/dev/null
  exit 0
}
trap stop HUP INT TERM
wait "$service"
kill "$viewer" 2>/dev/null
