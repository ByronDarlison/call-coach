"""Call Coach service: the page, the live session, the Jev loop and the MCP tools."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse

from .audio import Pipeline
from .jev import JevClient, apply_scores
from .questions import Followup, Question, output_paths, parse_questions, render_results, render_transcript

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
SUPPORT = Path.home() / "Library" / "Application Support" / "call-coach"
CONFIG_PATH = Path(os.environ.get("CALL_COACH_CONFIG", Path.home() / ".config" / "call-coach" / "config.json"))
JEV_EVERY = 4.0
JEV_WINDOW_LINES = 40
FIT_MINIMUM = 0.3


class Session:
    def __init__(self) -> None:
        self.questions_path: Path | None = None
        self.title = ""
        self.questions: list[Question] = []
        self.loose: list[Followup] = []
        self.lines: list[dict] = []  # final transcript lines
        self.partial: dict[str, str] = {}
        self.running = False
        self.started: datetime | None = None
        self.saved: list[str] = []
        self.status = "Choose a questions file to begin."
        self.version = 0
        self.changed = asyncio.Condition()

    async def bump(self) -> None:
        async with self.changed:
            self.version += 1
            self.changed.notify_all()

    def next_question(self) -> Question | None:
        open_ = [q for q in self.questions if not q.answered]
        best = max(open_, key=lambda q: q.fit_score, default=None)
        return best if best and best.fit_score >= FIT_MINIMUM else None

    def snapshot(self) -> dict:
        nxt = self.next_question()
        return {
            "version": self.version,
            "title": self.title,
            "file": str(self.questions_path) if self.questions_path else None,
            "running": self.running,
            "status": self.status,
            "saved": self.saved,
            "next": nxt.number if nxt else None,
            "questions": [
                {
                    "number": q.number, "text": q.text, "section": q.section, "answered": q.answered,
                    "manual": q.manual, "answered_score": round(q.answered_score, 2), "fit_score": round(q.fit_score, 2),
                    "followups": [{"text": f.text, "source": f.source, "answered": f.answered} for f in q.followups],
                }
                for q in self.questions
            ],
            "loose_followups": [{"text": f.text, "source": f.source, "answered": f.answered} for f in self.loose],
            "transcript": self.lines,
            "partial": self.partial,
        }


session = Session()
config: dict = {}
pipeline: Pipeline | None = None
jev: JevClient | None = None
jev_task: asyncio.Task | None = None

mcp = FastMCP(
    "call-coach",
    instructions=(
        "Call Coach follows a live call. Call get_session first, then loop: wait_for_update with the "
        "returned cursor, read what the other person just said, and when a good follow-up question "
        "comes to mind, push it with suggest_followup. Keep follow-ups short and specific."
    ),
    port=int(os.environ.get("CALL_COACH_PORT", "4800")),
)


# ---------- transcript and Jev ----------

async def on_line(speaker: str, text: str, final: bool) -> None:
    if final:
        session.partial.pop(speaker, None)
        if text:
            session.lines.append({
                "index": len(session.lines), "speaker": speaker, "text": text,
                "clock": datetime.now().strftime("%H:%M:%S"),
            })
    else:
        session.partial[speaker] = text
    await session.bump()


async def jev_loop() -> None:
    seen = -1
    while session.running:
        await asyncio.sleep(JEV_EVERY)
        pending = any(q.high_passes and not q.answered and not q.manual for q in session.questions)
        if not jev or (len(session.lines) == seen and not pending):
            continue
        seen = len(session.lines)
        open_ = [q for q in session.questions if not q.answered and not q.manual]
        try:
            scores = await jev.score(session.lines[-JEV_WINDOW_LINES:], open_)
        except Exception as exc:  # keep the call going; show the problem on the page
            session.status = f"Jev error: {exc}"
            await session.bump()
            continue
        if apply_scores(session.questions, scores):
            await session.bump()


# ---------- session control ----------

def load_questions(path: Path) -> None:
    session.questions_path = path
    session.title = path.stem.replace("-", " ").replace("_", " ").strip().capitalize()
    session.questions = parse_questions(path.read_text())
    session.loose, session.lines, session.partial, session.saved = [], [], {}, []
    session.status = f"Loaded {len(session.questions)} questions. Put on headphones, then press Start."


async def start_session() -> None:
    global pipeline, jev, jev_task
    if session.running:
        return
    if not session.questions_path:
        raise RuntimeError("Load a questions file first.")
    if session.saved:
        load_questions(session.questions_path)  # a fresh call starts from a clean list
    try:
        jev = JevClient(config.get("jev", {}))
    except (RuntimeError, OSError) as exc:
        jev = None
        jev_note = f" Jev is off ({exc}); tick questions by hand."
    else:
        jev_note = ""
    pipeline = Pipeline(
        capture_bin=Path(config.get("capture_bin", ROOT.parent / "bin" / "cc-capture")),
        model=Path(os.path.expanduser(config.get("model", SUPPORT / "models" / "ggml-small.en.bin"))),
        emit=on_line,
        whisper_port=int(os.environ.get("CALL_COACH_WHISPER_PORT", "8178")),
    )
    await pipeline.start()
    session.running, session.started, session.saved = True, datetime.now(), []
    session.status = "Listening." + jev_note
    jev_task = asyncio.create_task(jev_loop())


async def stop_session() -> None:
    global pipeline, jev, jev_task
    if not session.running:
        return
    session.running = False
    if jev_task:
        jev_task.cancel()
    if pipeline:
        await pipeline.stop()
    if jev:
        await jev.aclose()
    pipeline = jev = jev_task = None
    assert session.questions_path and session.started
    transcript_path, results_path = output_paths(session.questions_path, session.started)
    transcript_path.write_text(render_transcript(session.title, session.started, session.lines))
    results_path.write_text(render_results(session.title, session.started, session.questions, session.loose))
    session.saved = [str(transcript_path), str(results_path)]
    session.status = "Stopped. Results saved."


def choose_file() -> Path | None:
    script = ('POSIX path of (choose file with prompt "Choose a questions file" '
              'of type {"net.daringfireball.markdown", "public.plain-text"} '
              'default location (path to desktop folder))')
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    path = result.stdout.strip()
    return Path(path) if result.returncode == 0 and path else None


# ---------- page routes ----------

@mcp.custom_route("/", methods=["GET"])
async def index(_: Request) -> Response:
    return FileResponse(STATIC / "index.html")


@mcp.custom_route("/static/{name}", methods=["GET"])
async def static(request: Request) -> Response:
    target = (STATIC / request.path_params["name"]).resolve()
    if target.parent != STATIC or not target.exists():
        return Response(status_code=404)
    return FileResponse(target)


@mcp.custom_route("/events", methods=["GET"])
async def events(request: Request) -> Response:
    async def stream():
        last = -1
        while not await request.is_disconnected():
            if session.version != last:
                last = session.version
                yield f"data: {json.dumps(session.snapshot())}\n\n"
            async with session.changed:
                try:
                    await asyncio.wait_for(session.changed.wait(), 15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")


async def _action(fn) -> Response:
    try:
        await fn()
    except Exception as exc:
        session.status = str(exc)
        await session.bump()
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    await session.bump()
    return JSONResponse({"ok": True})


@mcp.custom_route("/api/choose", methods=["POST"])
async def api_choose(_: Request) -> Response:
    async def run():
        if session.running:
            raise RuntimeError("Stop the call before loading another file.")
        path = await asyncio.to_thread(choose_file)
        if path:
            load_questions(path)
    return await _action(run)


@mcp.custom_route("/api/start", methods=["POST"])
async def api_start(_: Request) -> Response:
    return await _action(start_session)


@mcp.custom_route("/api/stop", methods=["POST"])
async def api_stop(_: Request) -> Response:
    return await _action(stop_session)


@mcp.custom_route("/api/toggle", methods=["POST"])
async def api_toggle(request: Request) -> Response:
    body = await request.json()

    async def run():
        q = session.questions[int(body["number"]) - 1]
        if "followup" in body:
            f = q.followups[int(body["followup"])]
            f.answered = not f.answered
        else:
            q.answered, q.manual = not q.answered, True
    return await _action(run)


@mcp.custom_route("/api/add", methods=["POST"])
async def api_add(request: Request) -> Response:
    body = await request.json()

    async def run():
        add_followup(str(body.get("text", "")), body.get("number"), "you")
    return await _action(run)


@mcp.custom_route("/api/dev-line", methods=["POST"])
async def api_dev_line(request: Request) -> Response:
    """Inject a transcript line without audio. Only enabled with --dev."""
    if not config.get("dev"):
        return Response(status_code=404)
    body = await request.json()
    await on_line(body.get("speaker", "Them"), body["text"], True)
    return JSONResponse({"ok": True})


@mcp.custom_route("/api/dev-load", methods=["POST"])
async def api_dev_load(request: Request) -> Response:
    """Load a questions file by path without the file picker. Only enabled with --dev."""
    if not config.get("dev"):
        return Response(status_code=404)
    body = await request.json()

    async def run():
        load_questions(Path(body["path"]).expanduser().resolve())
    return await _action(run)


def add_followup(text: str, number, source: str) -> str:
    text = " ".join(text.split())
    if not text:
        raise RuntimeError("A follow-up needs some text.")
    followup = Followup(text=text, source=source)
    if number:
        n = int(number)
        if not 1 <= n <= len(session.questions):
            raise RuntimeError(f"There is no question {n}.")
        session.questions[n - 1].followups.append(followup)
        return f"Added under question {n}."
    session.loose.append(followup)
    return "Added to Follow-ups."


# ---------- MCP tools ----------

@mcp.tool()
async def get_session() -> dict:
    """The questions (with which are answered), follow-ups, the transcript so far, and a cursor for wait_for_update."""
    snap = session.snapshot()
    snap["cursor"] = len(session.lines)
    snap.pop("partial", None)
    return snap


@mcp.tool()
async def wait_for_update(cursor: int, timeout_seconds: int = 60) -> dict:
    """Block until new transcript lines exist after `cursor` (or the call stops, or the timeout passes).
    Returns only the new lines, the questions still open, and the next cursor."""
    deadline = asyncio.get_running_loop().time() + max(1, min(timeout_seconds, 300))
    while len(session.lines) <= cursor and session.running:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            break
        async with session.changed:
            try:
                await asyncio.wait_for(session.changed.wait(), remaining)
            except asyncio.TimeoutError:
                break
    return {
        "running": session.running,
        "new_lines": session.lines[cursor:],
        "cursor": len(session.lines),
        "open_questions": [{"number": q.number, "text": q.text} for q in session.questions if not q.answered],
    }


@mcp.tool()
async def suggest_followup(text: str, question_number: int | None = None) -> str:
    """Put a follow-up question on the page, optionally under the numbered question it relates to."""
    message = add_followup(text, question_number, "mcp")
    await session.bump()
    return message


@mcp.tool()
async def get_transcript() -> str:
    """The full transcript so far, as Markdown."""
    started = session.started or datetime.now()
    return render_transcript(session.title or "Call", started, session.lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Call Coach")
    parser.add_argument("--questions", type=Path, help="Markdown questions file to load at start")
    parser.add_argument("--dev", action="store_true", help="enable /api/dev-line for testing without audio")
    args = parser.parse_args()
    if CONFIG_PATH.exists():
        config.update(json.loads(CONFIG_PATH.read_text()))
    config["dev"] = args.dev or config.get("dev", False)
    if args.questions:
        load_questions(args.questions.expanduser().resolve())
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
