"""Run the capture helper and a local whisper-server, and turn audio into transcript segments.

Each stream (call audio = Them, microphone = You) is buffered separately. A simple energy check finds
speech; a segment is finalized when the speaker pauses or the buffer reaches MAX_SECONDS. Finished segments
queue for transcription, so slow transcription never stops new audio being cut into segments. While someone
is still talking and nothing is queued, a partial transcription is shown so the page feels live.

Times are seconds of audio received on that stream since Start, so they describe when someone spoke rather
than when transcription finished. Both streams start together, so their times can be compared.
"""

from __future__ import annotations

import asyncio
import collections
import io
import os
import re
import shutil
import socket
import struct
import time
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

import httpx

RATE = 16_000
SPEECH_RMS = 0.005  # tested on a quiet MacBook mic: speech 0.008 to 0.02, room noise about 0.002
MIN_SPEECH_SECONDS = 0.4  # shorter bursts are clicks and bumps, not words
PAUSE_SECONDS = 1.0
LEAD_IN_SECONDS = 1.0  # audio kept from before speech is detected, so soft starts are not cut
MAX_SECONDS = 12.0
PARTIAL_EVERY = 1.5
MAX_FRAME_SAMPLES = RATE * 2  # the helper sends about 10 ms at a time; anything this big is a broken stream
STALL_SECONDS = 5.0  # both streams send frames continuously, even in silence
BACKLOG_WARN_SECONDS = 30.0
MAX_QUEUED_SECONDS = 120.0  # beyond this, speech is reported as not transcribed rather than held in memory
CAPTURE_READY_SECONDS = 10.0
WHISPER_READY_SECONDS = 60.0
SPEAKERS = {"S": "Them", "M": "You"}

# Whisper turns faint noise into these. Real short replies ("Yes", "No", "Twelve") are kept.
NOISE_PHRASES = {"you", "thank you", "thanks for watching", "bye"}
# Labels Whisper writes for non-speech. Other bracketed text, such as "(about 20%)", is kept.
NON_SPEECH = re.compile(
    r"[\[(*]\s*(?:blank[ _]audio|music[^\])*]*|silence|laugh\w*|applause|inaudible|noise|background noise|"
    r"static|sigh\w*|cough\w*|typing|clicking|beep\w*|no speech|sound|indistinct\w*|crosstalk)\s*[\])*]",
    re.IGNORECASE)


@dataclass
class Segment:
    speaker: str
    text: str
    start: float
    end: float
    final: bool = True
    cut: bool = False  # ended because the buffer was full, not because the speaker paused
    failed: bool = False  # speech that could not be transcribed
    raw: str = ""


OnSegment = Callable[[Segment], Awaitable[None]]
OnHealth = Callable[[str, str, str | None], Awaitable[None]]  # component, state, message


def _rms(samples: array) -> float:
    if not samples:
        return 0.0
    return (sum(x * x for x in samples) / len(samples)) ** 0.5


def _pcm16(samples: array) -> bytes:
    return array("h", (max(-32768, min(32767, int(x * 32767))) for x in samples)).tobytes()


def _wav(samples: array) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(_pcm16(samples))
    return buf.getvalue()


def clean(text: str) -> str:
    text = NON_SPEECH.sub(" ", text)
    return " ".join(text.split()).strip(" -")


def is_noise(text: str, speech_seconds: float) -> bool:
    words = re.sub(r"[^\w\s]", "", text.lower()).strip()
    return speech_seconds < 0.8 and words in NOISE_PHRASES


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def port_in_use(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


class StreamBuffer:
    def __init__(self, tag: str, debug_folder: str | None):
        self.tag = tag
        self.speaker = SPEAKERS[tag]
        self.samples = array("f")
        self.received = 0  # samples received on this stream since Start
        self.clock = 0.0  # seconds since Start (one clock for both streams) when the last frame arrived
        self.t0 = time.monotonic()
        self.pending: dict[int, tuple[float, float]] = {}  # segment id -> (start, seconds), queued or in flight
        self.next_id = 0
        self.problem: str | None = None
        self.has_speech = False
        self.last_speech = 0.0
        self.last_partial = 0.0
        self.last_frame = time.monotonic()
        self.speech_samples = 0
        self.queue: asyncio.Queue[tuple[array, float, float, bool, float]] = asyncio.Queue()
        self.queued_seconds = 0.0
        self.working = False
        self.utterance = 0  # bumped at each finalize so a late partial cannot overwrite a final line
        self.recording: wave.Wave_write | None = None
        if debug_folder:  # written as the call goes, not held in memory
            path = Path(debug_folder) / f"{time.strftime('%Y%m%d-%H%M%S')}-{self.speaker.lower()}.wav"
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            self.recording = wave.open(os.fdopen(fd, "wb"), "wb")
            self.recording.setnchannels(1)
            self.recording.setsampwidth(2)
            self.recording.setframerate(RATE)

    @property
    def now_seconds(self) -> float:
        return self.clock

    def speech_start(self) -> float | None:
        """When the speech now being buffered began, or None."""
        return self.clock - len(self.samples) / RATE if self.has_speech else None

    def add(self, chunk: array, now: float) -> None:
        self.samples.extend(chunk)
        self.received += len(chunk)
        self.last_frame = now
        self.clock = now - self.t0
        if self.recording:
            try:
                self.recording.writeframes(_pcm16(chunk))
            except OSError as exc:  # debug audio is optional: never let it stop the call
                self.problem = f"Debug audio stopped: {exc}"
                self.recording = None
        if _rms(chunk) >= SPEECH_RMS:
            self.has_speech = True
            self.last_speech = now
            self.speech_samples += len(chunk)
        lead_in = int(RATE * LEAD_IN_SECONDS)
        if not self.has_speech and len(self.samples) > 2 * lead_in:
            del self.samples[: len(self.samples) - lead_in]

    def take(self, cut: bool) -> tuple[float, float] | None:
        """Finalize what is buffered and queue it for transcription. Returns (start, end) of speech that
        could not be queued because transcription is too far behind."""
        samples, speech = self.samples, self.speech_samples / RATE
        end = self.now_seconds
        start = end - len(samples) / RATE
        self.samples, self.has_speech, self.speech_samples = array("f"), False, 0
        self.utterance += 1
        if speech < MIN_SPEECH_SECONDS:
            return None
        if self.queued_seconds + (end - start) > MAX_QUEUED_SECONDS:
            return start, end  # bounded memory: report the gap instead of holding audio without limit
        seg_id = self.next_id
        self.next_id += 1
        self.pending[seg_id] = (start, end - start)
        self.queue.put_nowait((seg_id, samples, start, end, cut, speech))
        self.queued_seconds += end - start
        return None

    def pending_start(self) -> float | None:
        """The earliest start of speech on this stream not yet turned into text."""
        starts = [start for start, _ in self.pending.values()]
        buffered = self.speech_start()
        if buffered is not None:
            starts.append(buffered)
        return min(starts) if starts else None


class Pipeline:
    def __init__(self, capture_bin: Path, model: Path, on_segment: OnSegment, on_health: OnHealth,
                 whisper_port: int | None = None, mic_id: str | None = None, app_id: str | None = None,
                 use_mic: bool = True):
        self.capture_bin = capture_bin
        self.model = model
        self.on_segment = on_segment
        self.on_health = on_health
        self.port = whisper_port or free_port()
        self.mic_id = mic_id
        self.app_id = app_id
        self.use_mic = use_mic
        self.capture: asyncio.subprocess.Process | None = None
        self.whisper: asyncio.subprocess.Process | None = None
        self.capture_log: collections.deque[str] = collections.deque(maxlen=40)
        self.whisper_log: collections.deque[str] = collections.deque(maxlen=40)
        debug = os.environ.get("CALL_COACH_DEBUG_AUDIO")
        if debug:
            Path(debug).mkdir(parents=True, exist_ok=True, mode=0o700)
        self.debug_folder = debug
        self.buffers = {tag: StreamBuffer(tag, debug) for tag in SPEAKERS}
        self.tasks: list[asyncio.Task] = []
        self.workers: list[asyncio.Task] = []
        self.partials: set[asyncio.Task] = set()
        self.problems: list[str] = []
        self.reader: asyncio.Task | None = None
        self.client = httpx.AsyncClient(timeout=30)
        self.stopping = False
        self.ready = asyncio.Event()
        self.start_error: str | None = None

    # ---------- start ----------

    async def start(self) -> None:
        """Start whisper-server and the capture helper, and return only once both are working."""
        try:
            await self._start_whisper()
            t0 = time.monotonic()
            for buf in self.buffers.values():
                buf.t0 = buf.last_frame = t0
            await self._start_capture()
        except BaseException:
            await self.stop(drain=False)
            raise
        self.tasks += [asyncio.create_task(self._segment_loop()), asyncio.create_task(self._watch())]
        self.workers = [asyncio.create_task(self._worker(buf)) for buf in self.buffers.values()]

    async def _start_whisper(self) -> None:
        whisper_bin = shutil.which("whisper-server")
        if not whisper_bin:
            raise RuntimeError("whisper-server not found; install it with: brew install whisper-cpp")
        if not self.model.exists():
            raise RuntimeError(f"Whisper model missing at {self.model}; run scripts/install.sh")
        if port_in_use(self.port):
            raise RuntimeError(f"Port {self.port} for whisper-server is already in use by another program.")
        self.whisper = await asyncio.create_subprocess_exec(
            whisper_bin, "-m", str(self.model), "--host", "127.0.0.1", "--port", str(self.port), "-nt",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        self.tasks.append(asyncio.create_task(self._drain_log(self.whisper.stderr, self.whisper_log)))
        deadline = time.monotonic() + WHISPER_READY_SECONDS
        while True:
            if self.whisper.returncode is not None:
                raise RuntimeError("whisper-server stopped while starting: " + self._tail(self.whisper_log))
            try:
                resp = await self.client.get(f"http://127.0.0.1:{self.port}/health", timeout=2)
                if resp.status_code == 200 and resp.json().get("status") == "ok":
                    break
            except (httpx.HTTPError, ValueError):
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("whisper-server did not become ready: " + self._tail(self.whisper_log))
            await asyncio.sleep(0.25)
        # Prove this process answers transcription requests, not just that something is listening.
        try:
            await asyncio.wait_for(self._transcribe(array("f", [0.0]) * (RATE // 2)),
                                   max(1.0, deadline - time.monotonic()))
        except (asyncio.TimeoutError, httpx.HTTPError, ValueError) as exc:
            raise RuntimeError(f"whisper-server is running but could not transcribe: {exc or 'timed out'}")

    async def _start_capture(self) -> None:
        args = []
        if self.mic_id:
            args += ["--mic", self.mic_id]
        if self.app_id:
            args += ["--app", self.app_id]
        if not self.use_mic:
            args += ["--no-mic"]
        if not self.capture_bin.exists():
            raise RuntimeError(f"Audio helper missing at {self.capture_bin}; run scripts/install.sh")
        self.capture = await asyncio.create_subprocess_exec(
            str(self.capture_bin), *args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        self.tasks.append(asyncio.create_task(self._read_capture_log()))
        self.reader = asyncio.create_task(self._read_capture())
        deadline = time.monotonic() + CAPTURE_READY_SECONDS
        try:
            await asyncio.wait_for(self.ready.wait(), CAPTURE_READY_SECONDS)
        except asyncio.TimeoutError:
            raise RuntimeError("The audio helper did not start within 10 seconds. " + self._tail(self.capture_log))
        if self.start_error:
            raise RuntimeError(self.start_error)
        # "capturing" is not enough: wait for real frames on every stream in use.
        wanted = [b for b in self.buffers.values() if b.tag != "M" or self.use_mic]
        while any(b.received == 0 for b in wanted):
            if self.capture.returncode is not None:
                raise RuntimeError("The audio helper stopped: " + self._tail(self.capture_log))
            if time.monotonic() > deadline:
                missing = " and ".join("the microphone" if b.tag == "M" else "the call audio"
                                       for b in wanted if b.received == 0)
                raise RuntimeError(f"No audio arrived from {missing}. Check the microphone choice, or run "
                                   f"`call-coach doctor`.")
            await asyncio.sleep(0.05)

    @staticmethod
    def _tail(log: collections.deque[str]) -> str:
        return " | ".join(list(log)[-3:]) or "(no output)"

    async def _drain_log(self, stream, log: collections.deque[str]) -> None:
        async for raw in stream:
            log.append(raw.decode(errors="replace").strip()[:300])

    async def _read_capture_log(self) -> None:
        assert self.capture and self.capture.stderr
        async for raw in self.capture.stderr:
            line = raw.decode(errors="replace").strip()[:300]
            self.capture_log.append(line)
            print(line, flush=True)
            if line.endswith("capturing"):
                self.ready.set()
            elif "could not start" in line:
                self.start_error = permission_help(line)
                self.ready.set()
        if not self.ready.is_set():
            self.start_error = "The audio helper stopped: " + self._tail(self.capture_log)
            self.ready.set()

    async def _read_capture(self) -> None:
        assert self.capture and self.capture.stdout
        reader = self.capture.stdout
        try:
            while True:
                header = await reader.readexactly(5)
                tag = chr(header[0])
                (count,) = struct.unpack("<I", header[1:])
                if tag not in self.buffers or count > MAX_FRAME_SAMPLES:
                    await self.on_health("capture", "failed", "The audio helper sent data Call Coach cannot read.")
                    return
                payload = await reader.readexactly(count * 4)
                chunk = array("f")
                chunk.frombytes(payload)
                self.buffers[tag].add(chunk, time.monotonic())
        except asyncio.IncompleteReadError:
            pass

    # ---------- during the call ----------

    def audio_seconds(self) -> float:
        """Seconds of audio received since Start. Both streams send frames continuously, even in silence."""
        return max(buf.now_seconds for buf in self.buffers.values())

    def pending_start(self, speaker: str) -> float | None:
        """The earliest start of this speaker's speech not yet turned into text, or None."""
        return next(b for b in self.buffers.values() if b.speaker == speaker).pending_start()

    def speaking(self, speaker: str) -> bool:
        """True while this speaker has speech being buffered or waiting to be transcribed."""
        buf = next(b for b in self.buffers.values() if b.speaker == speaker)
        return buf.has_speech or buf.queued_seconds > 0

    async def _watch(self) -> None:
        """Report a capture or transcription process that dies, and a stream that stops sending audio."""
        reported: dict[str, tuple[str, str | None]] = {}

        async def report(component: str, state: str, message: str | None) -> None:
            if reported.get(component) != (state, message):
                reported[component] = (state, message)
                await self.on_health(component, state, message)

        while not self.stopping:
            await asyncio.sleep(1)
            if self.stopping:
                return
            if self.capture and self.capture.returncode is not None:
                await report("capture", "failed", "Audio capture stopped: " + self._tail(self.capture_log)
                             + ". The transcript so far is kept; press Stop, then Start to carry on.")
                continue
            if self.whisper and self.whisper.returncode is not None:
                await report("whisper", "failed", "Transcription stopped: " + self._tail(self.whisper_log))
                continue
            now = time.monotonic()
            stalled = [buf.speaker for buf in self.buffers.values()
                       if (buf.tag != "M" or self.use_mic) and now - buf.last_frame > STALL_SECONDS]
            if stalled:
                names = " and ".join("the microphone" if s == "You" else "the call audio" for s in stalled)
                await report("capture", "degraded", f"No audio from {names} for over {int(STALL_SECONDS)} seconds.")
            else:
                await report("capture", "ok", None)
            behind = max(buf.queued_seconds for buf in self.buffers.values())
            if behind > BACKLOG_WARN_SECONDS:
                await report("whisper", "degraded", f"Transcription is {int(behind)} seconds behind.")
            elif reported.get("whisper", ("ok",))[0] != "failed":
                await report("whisper", "ok", None)

    async def _segment_loop(self) -> None:
        while not self.stopping:
            await asyncio.sleep(0.25)
            now = time.monotonic()
            for buf in self.buffers.values():
                if not buf.has_speech:
                    continue
                seconds = len(buf.samples) / RATE
                dropped = None
                if now - buf.last_speech >= PAUSE_SECONDS:
                    dropped = buf.take(cut=False)
                elif seconds >= MAX_SECONDS:
                    dropped = buf.take(cut=True)
                elif (not buf.working and buf.queue.empty() and now - buf.last_partial >= PARTIAL_EVERY
                      and seconds >= 1.0 and buf.speech_samples >= RATE * MIN_SPEECH_SECONDS):
                    buf.last_partial = now
                    task = asyncio.create_task(self._partial(buf, array("f", buf.samples), buf.utterance))
                    self.partials.add(task)
                    task.add_done_callback(self.partials.discard)
                if dropped:
                    await self.on_health("whisper", "failed", "Transcription has fallen more than two minutes "
                                         "behind; new speech is being skipped. Press Stop, then Start.")
                    await self.on_segment(Segment(buf.speaker, "", dropped[0], dropped[1], failed=True))

    async def _partial(self, buf: StreamBuffer, samples: array, utterance: int) -> None:
        if buf.working:
            return
        buf.working = True
        try:
            text = clean(await self._transcribe(samples))
        except (httpx.HTTPError, ValueError, RuntimeError):
            return
        finally:
            buf.working = False
        if utterance == buf.utterance and text and not self.stopping:
            end = buf.now_seconds
            await self.on_segment(Segment(buf.speaker, text, end - len(samples) / RATE, end, final=False))

    async def _worker(self, buf: StreamBuffer) -> None:
        while True:
            seg_id, samples, start, end, cut, speech = await buf.queue.get()
            while buf.working:  # let a partial in flight finish first
                await asyncio.sleep(0.05)
            buf.working = True
            finished = False  # a cancelled segment stays pending, so Stop can record it as not transcribed
            try:
                raw, failed = "", False
                for attempt in range(2):
                    try:
                        raw = await self._transcribe(samples)
                        break
                    except (httpx.HTTPError, ValueError, RuntimeError) as exc:
                        if attempt:
                            failed = True
                            await self.on_health("whisper", "degraded", f"A segment could not be transcribed: {exc}")
                text = clean(raw)
                if text and is_noise(text, speech):
                    text = ""
                if text or failed:
                    await self.on_segment(Segment(buf.speaker, text, start, end, cut=cut, failed=failed, raw=raw))
                finished = True
            finally:
                buf.working = False
                if finished:
                    buf.queued_seconds = max(0.0, buf.queued_seconds - (end - start))
                    buf.pending.pop(seg_id, None)
                buf.queue.task_done()

    async def _transcribe(self, samples: array) -> str:
        resp = await self.client.post(
            f"http://127.0.0.1:{self.port}/inference",
            files={"file": ("audio.wav", _wav(samples), "audio/wav")},
            data={"response_format": "json", "temperature": "0.0"})
        resp.raise_for_status()
        body = resp.json()
        if not isinstance(body, dict) or not isinstance(body.get("text"), str):
            raise ValueError("whisper-server replied without text")
        return body["text"]

    # ---------- stop ----------

    async def stop(self, drain: bool = True, drain_seconds: float = 30.0) -> list[str]:
        """Stop capture, transcribe what was already heard, then stop transcription. Returns problems."""
        self.stopping = True
        problems: list[str] = []
        try:
            if self.capture and self.capture.returncode is None:
                try:
                    if self.capture.stdin:
                        self.capture.stdin.close()
                    self.capture.terminate()
                except ProcessLookupError:
                    pass
            if self.reader:
                try:
                    await asyncio.wait_for(asyncio.shield(self.reader), 3)  # read the frames already sent
                except asyncio.TimeoutError:
                    self.reader.cancel()
                except Exception as exc:
                    problems.append(f"Reading audio failed at the end: {exc}")
            for task in list(self.partials):
                task.cancel()
            if drain and self.workers:
                for buf in self.buffers.values():
                    if buf.has_speech:
                        dropped = buf.take(cut=False)
                        if dropped:
                            await self.on_segment(Segment(buf.speaker, "", dropped[0], dropped[1], failed=True))
                try:
                    await asyncio.wait_for(asyncio.gather(*(b.queue.join() for b in self.buffers.values())),
                                           drain_seconds)
                except asyncio.TimeoutError:
                    for task in self.workers:
                        task.cancel()
                    await asyncio.gather(*self.workers, return_exceptions=True)
                    left = [(b.speaker, start, seconds) for b in self.buffers.values()
                            for start, seconds in b.pending.values()]  # queued and in flight
                    for b in self.buffers.values():
                        b.pending.clear()
                    for speaker, start, seconds in sorted(left, key=lambda x: x[1]):
                        await self.on_segment(Segment(speaker, "", start, start + seconds, failed=True))
                    problems.append(f"{len(left)} piece(s) of speech ({sum(x[2] for x in left):.0f} s) were not "
                                    f"transcribed before Stop timed out; they are marked in the transcript.")
        except Exception as exc:
            problems.append(f"Stopping audio hit a problem: {exc}")
        finally:
            for task in self.tasks + self.workers + list(self.partials):
                task.cancel()
            await asyncio.gather(*self.tasks, *self.workers, *self.partials, return_exceptions=True)
            problems += [b.problem for b in self.buffers.values() if b.problem]
            for proc in (self.capture, self.whisper):
                if proc and proc.returncode is None:
                    try:
                        proc.terminate()
                        await asyncio.wait_for(proc.wait(), 5)
                    except ProcessLookupError:
                        pass
                    except asyncio.TimeoutError:
                        proc.kill()
                        await proc.wait()
            for buf in self.buffers.values():
                if buf.recording:
                    try:
                        buf.recording.close()
                    except OSError as exc:
                        problems.append(f"Debug audio not saved: {exc}")
            await self.client.aclose()
        return problems


def permission_help(line: str) -> str:
    """Turn the helper's start failure into what the owner should do."""
    reason = line.split("could not start:")[-1].strip()
    if "is not running" in reason:
        return (f"Could not capture the call audio: {reason}. Open that app, or choose All system sound, "
                f"then press Start again.")
    host = os.environ.get("TERM_PROGRAM") or "the app that started Call Coach"
    names = {"Apple_Terminal": "Terminal", "iTerm.app": "iTerm", "vscode": "Visual Studio Code"}
    host = names.get(host, host)
    return (f"Could not start audio capture ({reason}). macOS may not "
            f"have given Screen & System Audio Recording permission to {host}. Open System Settings > Privacy "
            f"& Security > Screen & System Audio Recording, turn on {host}, quit and reopen it, then start "
            f"Call Coach again. Run `call-coach doctor` to check.")
