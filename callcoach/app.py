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
CALLS_INDEX = Path(os.environ.get("CALL_COACH_CALLS_INDEX", SUPPORT / "calls.json"))  # saved calls, newest first
NEW_FILES = Path.home() / "Documents" / "Call Coach"  # where questions written by an AI are saved
CONFIG_PATH = Path(os.environ.get("CALL_COACH_CONFIG", Path.home() / ".config" / "call-coach" / "config.json"))
JEV_EVERY = 4.0
JEV_WINDOW_LINES = 40
BATCH_LINES = 12  # wait_for_update returns after this many new lines even with nothing flagged
JUMP_FIT = 0.85  # Jev may jump ahead of list order only when a later question fits this well...
JUMP_MARGIN = 0.3  # ...and fits this much better than the next question in order
OFF_LIST_THRESHOLD = 0.6  # tested: real questions 0.97 to 0.98, "Okay, great" 0.12
FOLLOW_UP_THRESHOLD = 0.5  # tested on 47 demo answers: flags the 9 most revealing, routine ones score 0.13 to 0.47
WRAP_UP_THRESHOLD = 0.8  # tested: "That is all my questions" 0.94, a polite "thanks" mid-call 0.37
FOLLOW_UP_COOLDOWN = 2  # no flag if one of the interviewee's previous two answers was already flagged


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
        self.next_override: str | None = None  # key of the question or follow-up the AI picked to ask next
        self.notices: list[dict] = []  # short on-page notices, e.g. "Claude added a follow-up"

    def notice(self, text: str) -> None:
        self.notices = (self.notices + [{"id": self.version + 1, "text": text}])[-5:]

    async def bump(self) -> None:
        async with self.changed:
            self.version += 1
            self.changed.notify_all()

    def items(self) -> list:
        """Every question and follow-up, in page order."""
        out = []
        for q in self.questions:
            out.append(q)
            out.extend(q.followups)
        return out + self.loose

    def find(self, key) -> Question | Followup:
        key = str(key).strip()
        for item in self.items():
            if item.key == key:
                return item
        raise RuntimeError(f"There is no question {key}.")

    def next_item(self):
        """What to ask next: the AI's pick (including a follow-up it just added), else the next question in
        list order, unless Jev has strong evidence the other person is already on a later question."""
        if self.next_override:
            try:
                picked = self.find(self.next_override)
                if not picked.answered:
                    return picked, "ai"
            except RuntimeError:
                pass
            self.next_override = None
        in_order = next((q for q in self.questions if not q.answered and not q.asked), None)
        if in_order is None:
            return None, None
        candidates = [i for i in self.items()
                      if not i.answered and not i.asked and not getattr(i, "asked_live", False)]
        best = max(candidates, key=lambda i: i.fit_score, default=None)
        if best and best is not in_order and best.fit_score >= JUMP_FIT \
                and best.fit_score - in_order.fit_score >= JUMP_MARGIN:
            return best, "jev"
        return in_order, "order"

    def describe(self, item) -> str:
        if isinstance(item, Question):
            return f"Question {item.number}"
        if "." in item.key:
            return f"Follow-up to question {item.key.split('.')[0]}"
        return "Added question"

    def snapshot(self) -> dict:
        nxt, picked_by = self.next_item()
        follow = lambda f: {"key": f.key, "text": f.text, "source": f.source, "answered": f.answered,
                            "asked_live": f.asked_live}
        return {
            "version": self.version,
            "title": self.title,
            "file": str(self.questions_path) if self.questions_path else None,
            "running": self.running,
            "status": self.status,
            "saved": self.saved,
            "next": {"key": nxt.key, "text": nxt.text, "kind": self.describe(nxt),
                     "by": picked_by} if nxt else None,
            "notices": self.notices,
            "questions": [
                {
                    "key": q.key, "number": q.number, "text": q.text, "section": q.section, "answered": q.answered,
                    "manual": q.manual, "answered_score": round(q.answered_score, 2), "fit_score": round(q.fit_score, 2),
                    "followups": [follow(f) for f in q.followups],
                }
                for q in self.questions
            ],
            "loose_followups": [follow(f) for f in self.loose],
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
        "Call Coach runs a live interview checklist; Jev ticks questions, keeps the list in order, records "
        "off-list questions and flags answers worth a follow-up. Your job is follow-ups, so keep calls few: "
        "load_questions_file and start_call if asked, then loop wait_for_update(cursor) with its default "
        "wake_on='follow_up'. When a returned line has worth_follow_up: true, call add_question once with one "
        "short follow-up (under 15 words) under the question it belongs to. When a line has wrapping_up: true, "
        "call stop_call, then read_saved_call and recap. Do not poll, do not call "
        "set_next_question unless the interviewer asks you to steer, and fetch get_open_questions only when "
        "needed. stop_call saves the transcript and results; list_saved_calls and read_saved_call read past "
        "calls back. Use a small, fast model if you can."
    ),
    port=int(os.environ.get("CALL_COACH_PORT", "4800")),
)


# ---------- transcript and Jev ----------

async def on_line(speaker: str, text: str, final: bool) -> None:
    if final:
        session.partial.pop(speaker, None)
        if text:
            line = {"index": len(session.lines), "speaker": speaker, "text": text,
                    "clock": datetime.now().strftime("%H:%M:%S")}
            await review_line(line)
            session.lines.append(line)
    else:
        session.partial[speaker] = text
    await session.bump()


async def review_line(line: dict) -> None:
    """Ask Jev about each finished line: is the interviewer asking something off the list, and does the
    interviewee's answer invite a follow-up? Off-list questions are recorded straight away."""
    if not jev or not session.running:
        return
    try:
        if line["speaker"] == "You":
            is_question, key, ending = await asyncio.wait_for(jev.which_question(line["text"], session.items()), 4)
            if ending >= WRAP_UP_THRESHOLD:
                line["wrapping_up"] = True
            elif is_question >= OFF_LIST_THRESHOLD:
                if key == "unclear":
                    pass
                elif key:
                    line["asked"] = key
                    try:
                        session.find(key).asked = True
                    except RuntimeError:
                        pass
                else:
                    add_followup(line["text"], None, "live", asked_live=True)
                    line["off_list"] = True
        else:
            recent = [l for l in session.lines if l["speaker"] == "Them"][-FOLLOW_UP_COOLDOWN:]
            if not any(l.get("worth_follow_up") for l in recent):
                score = await asyncio.wait_for(jev.worth_follow_up(line["text"]), 4)
                line["worth_follow_up"] = score >= FOLLOW_UP_THRESHOLD
    except Exception as exc:  # a missed check must never hold up the transcript
        session.status = f"Jev check skipped: {exc}"


async def jev_loop() -> None:
    seen = -1
    while session.running:
        await asyncio.sleep(JEV_EVERY)
        open_ = [i for i in session.items() if not i.answered and not i.manual]
        pending = any(i.high_passes for i in open_)
        if not jev or (len(session.lines) == seen and not pending):
            continue
        seen = len(session.lines)
        try:
            scores = await jev.score(session.lines[-JEV_WINDOW_LINES:], open_)
        except Exception as exc:  # keep the call going; show the problem on the page
            session.status = f"Jev error: {exc}"
            await session.bump()
            continue
        if apply_scores(open_, scores):
            await session.bump()


# ---------- session control ----------

def load_questions(path: Path) -> None:
    session.questions_path = path
    session.title = path.stem.replace("-", " ").replace("_", " ").strip().capitalize()
    session.questions = parse_questions(path.read_text())
    session.loose, session.lines, session.partial, session.saved = [], [], {}, []
    session.next_override, session.notices = None, []
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
        capture_bin=capture_bin(),
        model=Path(os.path.expanduser(config.get("model", SUPPORT / "models" / "ggml-small.en.bin"))),
        emit=on_line,
        whisper_port=int(os.environ.get("CALL_COACH_WHISPER_PORT", "8178")),
        mic_id=load_settings().get("mic") or None,
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
    calls = json.loads(CALLS_INDEX.read_text()) if CALLS_INDEX.exists() else []
    calls.insert(0, {"title": session.title, "started": session.started.isoformat(timespec="minutes"),
                     "questions": str(session.questions_path), "transcript": str(transcript_path),
                     "results": str(results_path)})
    CALLS_INDEX.parent.mkdir(parents=True, exist_ok=True)
    CALLS_INDEX.write_text(json.dumps(calls[:200], indent=1))


SETTINGS = SUPPORT / "settings.json"  # remembered choices, such as the microphone


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS.read_text())
    except (OSError, ValueError):
        return {}


def capture_bin() -> Path:
    return Path(config.get("capture_bin", ROOT.parent / "bin" / "cc-capture"))


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


@mcp.custom_route("/api/mics", methods=["GET"])
async def api_mics(_: Request) -> Response:
    """The microphones macOS knows about, and which one Call Coach will use."""
    proc = await asyncio.create_subprocess_exec(str(capture_bin()), "--list-mics",
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    out, _ = await proc.communicate()
    mics = json.loads(out or b"[]")
    chosen = load_settings().get("mic")
    if chosen not in {m["id"] for m in mics}:
        chosen = None  # the remembered mic is unplugged: fall back to the system default
    return JSONResponse({"mics": mics, "chosen": chosen})


@mcp.custom_route("/api/mic", methods=["POST"])
async def api_mic(request: Request) -> Response:
    body = await request.json()

    async def run():
        settings = load_settings()
        settings["mic"] = body.get("id") or None
        SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS.write_text(json.dumps(settings))
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
        item = session.find(body["key"])
        item.answered, item.manual = not item.answered, True
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


def add_followup(text: str, number, source: str, ask_next: bool = False, asked_live: bool = False) -> str:
    text = " ".join(text.split())
    if not text:
        raise RuntimeError("A question needs some text.")
    who = "AI" if source == "mcp" else "You"
    if number:
        n = int(number)
        if not 1 <= n <= len(session.questions):
            raise RuntimeError(f"There is no question {n}.")
        parent = session.questions[n - 1]
        item = Followup(key=f"{n}.{len(parent.followups) + 1}", text=text, source=source)
        parent.followups.append(item)
        where = f"under question {n}"
    else:
        item = Followup(key=f"f{len(session.loose) + 1}", text=text, source=source, asked_live=asked_live)
        session.loose.append(item)
        where = "as an off-list question you asked" if asked_live else "to Other questions"
    if ask_next and not asked_live:
        session.next_override = item.key
    if source == "live":
        session.notice("Recorded an off-list question you asked")
    elif source == "mcp":
        session.notice(f"{who} recorded an off-list question you asked" if asked_live else
                       f"{who} added a follow-up {where}" + (": ask it next" if ask_next else ""))
    return f"Added question {item.key} {where}."


# ---------- MCP tools ----------

@mcp.tool()
async def get_session() -> dict:
    """The questions (with which are answered), follow-ups, the transcript so far, and a cursor for wait_for_update."""
    snap = session.snapshot()
    snap["cursor"] = len(session.lines)
    snap.pop("partial", None)
    return snap


def _compact(line: dict) -> dict:
    out = {"speaker": line["speaker"], "text": line["text"]}
    for key in ("asked", "off_list", "worth_follow_up", "wrapping_up"):
        if line.get(key):
            out[key] = line[key]
    return out


@mcp.tool()
async def wait_for_update(cursor: int, timeout_seconds: int = 120, wake_on: str = "follow_up") -> dict:
    """Wait for the call to move on, then return only the transcript lines after `cursor` and the next cursor.
    wake_on="follow_up" (default, cheapest) returns as soon as the other person says something Jev marks
    worth_follow_up, when the interviewer is wrapping_up, after BATCH_LINES new lines, when the call stops,
    or at the timeout.
    wake_on="line" returns on every new line. Use get_open_questions only when you need the list."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(1, min(timeout_seconds, 300))

    def ready() -> bool:
        new = session.lines[cursor:]
        if not session.running:
            return True
        if wake_on == "line":
            return bool(new)
        return len(new) >= BATCH_LINES or any(l.get("worth_follow_up") or l.get("wrapping_up") for l in new)

    while not ready():
        remaining = deadline - loop.time()
        if remaining <= 0:
            break
        async with session.changed:
            try:
                await asyncio.wait_for(session.changed.wait(), remaining)
            except asyncio.TimeoutError:
                break
    return {"running": session.running, "cursor": len(session.lines),
            "new_lines": [_compact(l) for l in session.lines[cursor:]]}


@mcp.tool()
async def get_open_questions() -> list[dict]:
    """The questions and follow-ups not yet answered, with their keys ("7", "7.1", "f1")."""
    return [{"key": i.key, "text": i.text} for i in session.items() if not i.answered]


@mcp.tool()
async def load_questions_file(path: str | None = None, markdown: str | None = None, title: str | None = None) -> str:
    """Load the questions for a call: either the path of a Markdown file, or Markdown you write (every list
    item is a question, ## headings group them). Written questions are saved to ~/Documents/Call Coach."""
    if session.running:
        raise RuntimeError("Stop the call before loading other questions.")
    if markdown:
        NEW_FILES.mkdir(parents=True, exist_ok=True)
        name = "-".join((title or "questions").lower().split())[:60]
        target = NEW_FILES / f"{name}.md"
        target.write_text((f"# {title}\n\n" if title else "") + markdown.strip() + "\n")
    elif path:
        target = Path(path).expanduser().resolve()
    else:
        raise RuntimeError("Give a path or some markdown.")
    load_questions(target)
    await session.bump()
    return f"Loaded {len(session.questions)} questions from {target}."


@mcp.tool()
async def start_call() -> str:
    """Start listening: capture both sides, transcribe, and tick questions as they are answered."""
    await start_session()
    await session.bump()
    return session.status


@mcp.tool()
async def stop_call() -> dict:
    """Stop listening and save the transcript and results next to the questions file."""
    await stop_session()
    await session.bump()
    return {"status": session.status, "saved": session.saved}


@mcp.tool()
async def add_question(text: str, under_question: int | None = None, ask_next: bool = True,
                       already_asked: bool = False) -> str:
    """Add a question to the page. Give under_question to make it a follow-up to that numbered question.
    ask_next (default true) puts it in the Ask next bar straight away. Use already_asked=true to record a
    question the interviewer asked that was not on the list; Jev then ticks it when it is answered."""
    message = add_followup(text, under_question, "mcp", ask_next=ask_next, asked_live=already_asked)
    await session.bump()
    return message


@mcp.tool()
async def set_next_question(key: str) -> str:
    """Show this open item in the Ask next bar: a question number like "7", or a follow-up key like "7.1"."""
    item = session.find(key)
    if item.answered:
        raise RuntimeError(f"{key} is already answered.")
    session.next_override = item.key
    await session.bump()
    return f"Ask next: {item.key}."


@mcp.tool()
async def mark_question(key: str, answered: bool = True) -> str:
    """Tick (or untick) a question or follow-up by key ("7", "7.1", "f1"). Jev will not change it after."""
    item = session.find(key)
    item.answered, item.manual = answered, True
    await session.bump()
    return f"{item.key} {'ticked' if answered else 'unticked'}."


@mcp.tool()
async def list_saved_calls(limit: int = 20) -> list[dict]:
    """Past calls, newest first: title, start time, and the paths of their questions, transcript and results."""
    calls = json.loads(CALLS_INDEX.read_text()) if CALLS_INDEX.exists() else []
    return [dict(c, index=i) for i, c in enumerate(calls[: max(1, limit)])]


@mcp.tool()
async def read_saved_call(index: int = 0) -> dict:
    """Read a saved call from list_saved_calls (0 = the latest): its results checklist and its transcript."""
    calls = json.loads(CALLS_INDEX.read_text()) if CALLS_INDEX.exists() else []
    if not 0 <= index < len(calls):
        raise RuntimeError("No saved call at that index.")
    call = calls[index]
    return {"title": call["title"], "started": call["started"],
            "results": Path(call["results"]).read_text(), "transcript": Path(call["transcript"]).read_text()}


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
    import uvicorn
    # A short graceful timeout: open page connections (live updates) would otherwise hold up Ctrl-C.
    uvicorn.run(mcp.streamable_http_app(), host="127.0.0.1", port=mcp.settings.port,
                timeout_graceful_shutdown=2, log_level="warning")


if __name__ == "__main__":
    main()
