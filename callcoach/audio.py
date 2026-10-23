"""Run the capture helper and a local whisper-server, and turn audio into lines.

Each stream (system audio = Them, microphone = You) is buffered separately. A
simple energy check finds speech; a line is finalized when the speaker pauses
or the buffer reaches MAX_SECONDS. While someone is still talking, a partial
transcription is shown so the page feels live.
"""

from __future__ import annotations

import asyncio
import io
import re
import shutil
import struct
import time
import wave
from array import array
from pathlib import Path
from typing import Awaitable, Callable

import httpx

RATE = 16_000
SPEECH_RMS = 0.012
PAUSE_SECONDS = 0.7
MAX_SECONDS = 12.0
PARTIAL_EVERY = 1.5
SPEAKERS = {"S": "Them", "M": "You"}

Emit = Callable[[str, str, bool], Awaitable[None]]  # speaker, text, final


def _rms(samples: array) -> float:
    if not samples:
        return 0.0
    return (sum(x * x for x in samples) / len(samples)) ** 0.5


def _wav(samples: array) -> bytes:
    pcm = array("h", (max(-32768, min(32767, int(x * 32767))) for x in samples))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _clean(text: str) -> str:
    # Whisper labels non-speech as [music], (silence), *laughs* and so on; drop those.
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*", " ", text)
    return " ".join(text.split()).strip(" -")


class StreamBuffer:
    def __init__(self, tag: str):
        self.speaker = SPEAKERS[tag]
        self.samples = array("f")
        self.has_speech = False
        self.last_speech = 0.0
        self.last_partial = 0.0
        self.busy = False

    def add(self, chunk: array, now: float) -> None:
        self.samples.extend(chunk)
        if _rms(chunk) >= SPEECH_RMS:
            self.has_speech = True
            self.last_speech = now
        if not self.has_speech and len(self.samples) > RATE // 2:
            del self.samples[: len(self.samples) - RATE // 4]  # keep a little lead-in only


class Pipeline:
    def __init__(self, capture_bin: Path, model: Path, emit: Emit, whisper_port: int = 8178):
        self.capture_bin = capture_bin
        self.model = model
        self.emit = emit
        self.port = whisper_port
        self.capture: asyncio.subprocess.Process | None = None
        self.whisper: asyncio.subprocess.Process | None = None
        self.buffers = {tag: StreamBuffer(tag) for tag in SPEAKERS}
        self.tasks: list[asyncio.Task] = []
        self.client = httpx.AsyncClient(timeout=30)
        self.error: str | None = None

    async def start(self) -> None:
        whisper_bin = shutil.which("whisper-server")
        if not whisper_bin:
            raise RuntimeError("whisper-server not found; install it with: brew install whisper-cpp")
        if not self.model.exists():
            raise RuntimeError(f"Whisper model missing at {self.model}; run scripts/install.sh")
        self.whisper = await asyncio.create_subprocess_exec(
            whisper_bin, "-m", str(self.model), "--host", "127.0.0.1", "--port", str(self.port), "-nt",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        await self._wait_for_whisper()
        self.capture = await asyncio.create_subprocess_exec(
            str(self.capture_bin), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.tasks = [
            asyncio.create_task(self._read_capture()),
            asyncio.create_task(self._read_capture_errors()),
            asyncio.create_task(self._transcribe_loop()),
        ]

    async def _wait_for_whisper(self) -> None:
        for _ in range(100):
            try:
                await self.client.get(f"http://127.0.0.1:{self.port}/")
                return
            except httpx.HTTPError:
                await asyncio.sleep(0.2)
        raise RuntimeError("whisper-server did not start")

    async def _read_capture(self) -> None:
        assert self.capture and self.capture.stdout
        reader = self.capture.stdout
        try:
            while True:
                header = await reader.readexactly(5)
                tag = chr(header[0])
                (count,) = struct.unpack("<I", header[1:])
                payload = await reader.readexactly(count * 4)
                if tag in self.buffers:
                    chunk = array("f")
                    chunk.frombytes(payload)
                    self.buffers[tag].add(chunk, time.monotonic())
        except asyncio.IncompleteReadError:
            pass

    async def _read_capture_errors(self) -> None:
        assert self.capture and self.capture.stderr
        async for raw in self.capture.stderr:
            line = raw.decode(errors="replace").strip()
            if "could not start" in line or "stopped" in line:
                self.error = line
            print(line, flush=True)

    async def _transcribe_loop(self) -> None:
        while True:
            await asyncio.sleep(0.25)
            now = time.monotonic()
            for buf in self.buffers.values():
                if buf.busy or not buf.has_speech:
                    continue
                seconds = len(buf.samples) / RATE
                paused = now - buf.last_speech >= PAUSE_SECONDS
                if paused or seconds >= MAX_SECONDS:
                    samples = buf.samples
                    buf.samples, buf.has_speech = array("f"), False
                    asyncio.create_task(self._run(buf, samples, final=True))
                elif now - buf.last_partial >= PARTIAL_EVERY and seconds >= 1.0:
                    buf.last_partial = now
                    asyncio.create_task(self._run(buf, array("f", buf.samples), final=False))

    async def _run(self, buf: StreamBuffer, samples: array, final: bool) -> None:
        buf.busy = True
        try:
            resp = await self.client.post(
                f"http://127.0.0.1:{self.port}/inference",
                files={"file": ("audio.wav", _wav(samples), "audio/wav")},
                data={"response_format": "json", "temperature": "0.0"},
            )
            text = _clean(resp.json().get("text", ""))
        except (httpx.HTTPError, ValueError) as exc:
            self.error = f"transcription failed: {exc}"
            return
        finally:
            buf.busy = False
        if text or final:
            await self.emit(buf.speaker, text, final)

    async def stop(self) -> None:
        for task in self.tasks:
            task.cancel()
        # Flush whatever each speaker was in the middle of saying.
        for buf in self.buffers.values():
            if buf.has_speech and len(buf.samples) > RATE // 2:
                await self._run(buf, buf.samples, final=True)
            buf.samples, buf.has_speech = array("f"), False
        for proc in (self.capture, self.whisper):
            if proc and proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), 5)
                except asyncio.TimeoutError:
                    proc.kill()
        await self.client.aclose()
