"""call-coach doctor: check this Mac before a call.

Run it from the same app you start Call Coach from (Terminal, or the Desktop launcher), because macOS
gives Screen & System Audio Recording permission to that app, not to Call Coach itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import select
import shutil
import struct
import subprocess
import time
from array import array
from pathlib import Path

from . import store

MODEL_SHA256 = {  # checked when installed; see scripts/install.sh
    "ggml-small.en.bin": "c6138d6d58ecc8322097e0f987c32f1be8bb0a18532a3f88f734d1bbf9c41e5d",
}


def _line(ok: bool | None, text: str) -> None:
    mark = {True: "ok  ", False: "FIX ", None: "note"}[ok]
    print(f"  [{mark}] {text}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(fd: int, n: int, deadline: float) -> bytes | None:
    """Read exactly n bytes before the deadline, or None."""
    out = b""
    while len(out) < n:
        left = deadline - time.time()
        if left <= 0 or not select.select([fd], [], [], left)[0]:
            return None
        chunk = os.read(fd, n - len(out))
        if not chunk:
            return None
        out += chunk
    return out


def listen(capture_bin: Path, args: list[str], seconds: float = 5.0) -> dict:
    """Run the helper for a few seconds with the same microphone and app as Start, and measure frames and
    loudness on each channel. Never waits past the deadline, even if the helper sends nothing."""
    proc = subprocess.Popen([str(capture_bin), *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    stats = {"S": [0, 0.0], "M": [0, 0.0]}
    deadline = time.time() + seconds
    fd = proc.stdout.fileno()
    try:
        while time.time() < deadline:
            header = _read(fd, 5, deadline)
            if header is None:
                break
            tag, (count,) = chr(header[0]), struct.unpack("<I", header[1:])
            if count > 32000:
                break
            payload = _read(fd, count * 4, deadline + 1)
            if payload is None:
                break
            samples = array("f")
            samples.frombytes(payload)
            if tag in stats and samples:
                stats[tag][0] += len(samples)
                stats[tag][1] = max(stats[tag][1], (sum(x * x for x in samples) / len(samples)) ** 0.5)
    finally:
        proc.terminate()
        try:
            err = proc.communicate(timeout=5)[1].decode(errors="replace").strip()
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            err = ""
    return {"call_audio_seconds": stats["S"][0] / 16000, "call_audio_peak": stats["S"][1],
            "mic_seconds": stats["M"][0] / 16000, "mic_peak": stats["M"][1], "log": err}


def run(config: dict, capture_bin: Path, settings: dict | None = None) -> int:
    settings = settings or {}
    problems = 0
    names = {"Apple_Terminal": "Terminal", "iTerm.app": "iTerm", "vscode": "Visual Studio Code"}
    host = names.get(os.environ.get("TERM_PROGRAM", ""), os.environ.get("TERM_PROGRAM") or "this app")
    print(f"Call Coach doctor (running from {host})\n")

    whisper = shutil.which("whisper-server")
    _line(bool(whisper), f"whisper-server: {whisper or 'missing; run: brew install whisper-cpp'}")
    problems += not whisper
    model = Path(os.path.expanduser(config.get("model", store.SUPPORT / "models" / "ggml-small.en.bin")))
    if model.exists():
        expected = MODEL_SHA256.get(model.name)
        if expected:
            good = sha256(model) == expected
            _line(good, f"Whisper model {model.name}" + ("" if good else ": damaged, delete it and run install.sh"))
            problems += not good
        else:
            _line(None, f"Whisper model {model.name} (no checksum on record)")
    else:
        _line(False, f"Whisper model missing at {model}; run scripts/install.sh")
        problems += 1

    if not capture_bin.exists():
        _line(False, f"Audio helper missing at {capture_bin}; run scripts/install.sh")
        return 1
    try:
        check = json.loads(subprocess.run([str(capture_bin), "--check"], capture_output=True, timeout=15).stdout)
    except (subprocess.TimeoutExpired, ValueError) as exc:
        _line(False, f"Audio helper did not answer: {exc}")
        return 1
    screen = check.get("screen_recording")
    _line(screen, "Screen & System Audio Recording permission" + ("" if screen else (
        f": not given to {host}. Open System Settings > Privacy & Security > Screen & System Audio "
        f"Recording, turn on {host}, then quit and reopen it.")))
    problems += not screen
    mic = check.get("microphone")
    _line(True if mic == "allowed" else None if mic == "not asked yet" else False,
          f"Microphone permission: {mic}" + (" (macOS asks at the first Start)" if mic == "not asked yet" else ""))
    problems += mic == "denied"

    args = []
    try:
        mics = json.loads(subprocess.run([str(capture_bin), "--list-mics"], capture_output=True, timeout=10).stdout)
        chosen = next((m for m in mics if m["id"] == settings.get("mic")), None)
        default = next((m["name"] for m in mics if m.get("default")), "none")
        if settings.get("mic") and not chosen:
            _line(None, f"Your chosen microphone is not connected; Start will use the default, {default}")
        elif chosen:
            args += ["--mic", chosen["id"]]
        _line(None, f"Microphone Start will use: {chosen['name'] if chosen else default}")
    except (subprocess.TimeoutExpired, ValueError):
        _line(False, "Could not list microphones")
        problems += 1
    if settings.get("app"):
        args += ["--app", settings["app"]]
    _line(None, f"Call audio Start will use: {settings.get('app') or 'all system sound'}")

    if screen:
        print("\n  Listening for 5 seconds: say a few words, and play some sound if you can…")
        heard = listen(capture_bin, args)
        got_call = heard["call_audio_seconds"] > 3
        got_mic = heard["mic_seconds"] > 3
        _line(got_call, f"Call audio: {heard['call_audio_seconds']:.1f} s received, loudest {heard['call_audio_peak']:.3f}")
        _line(got_mic, f"Microphone: {heard['mic_seconds']:.1f} s received, loudest {heard['mic_peak']:.3f}"
              + ("" if heard["mic_peak"] >= 0.005 else " (quiet: speak up, or choose another microphone)"))
        problems += (not got_call) + (not got_mic)
        if not (got_call and got_mic) and heard["log"]:
            print("    helper said: " + heard["log"][-300:])

    jev = config.get("jev") or {}
    has_key = bool(os.environ.get("TYPESAFE_API_KEY") or jev.get("broker_secret_file") or
                   (jev.get("api_key_file") and Path(os.path.expanduser(jev["api_key_file"])).exists()))
    _line(None if has_key else False, "Jev: " + ("set up; transcript text goes to " +
          (jev.get("url", "https://api.typesafe.ai").split("/")[2]) if has_key else
          "no key, so questions are ticked by hand. Run scripts/install.sh to add one."))
    print("\n" + ("Ready for a call." if not problems else f"{problems} thing(s) to fix before a call."))
    return 1 if problems else 0
