"""Private files: atomic writes, the per-call folder, the calls index and recovery after a crash.

Everything Call Coach keeps lives under ~/Library/Application Support/call-coach, readable only by you:

    calls/<call id>/state.json      the whole call, rewritten every few seconds while it runs
    calls/<call id>/transcript.md   written at Stop, with results.md and results.json
    calls.json                      the saved calls, newest first

At Stop the transcript and results are also written next to the questions file, under new names that
never replace an existing file.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

SUPPORT = Path(os.environ.get("CALL_COACH_HOME", Path.home() / "Library" / "Application Support" / "call-coach"))
CALLS = SUPPORT / "calls"
CALLS_INDEX = Path(os.environ.get("CALL_COACH_CALLS_INDEX", SUPPORT / "calls.json"))
INDEX_LIMIT = 200


def private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path


def write_atomic(path: Path, text: str, private: bool = True) -> None:
    """Replace path with text in one step, so a crash leaves either the old file or the new one.
    private: the folder is Call Coach's own and is kept readable only by you. Files elsewhere (copies
    beside a questions file) leave their folder's permissions alone."""
    if private:
        private_dir(path.parent)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_new(path: Path, text: str) -> Path:
    """Create a file that does not exist yet; if the name is taken, add -2, -3 and so on.
    Never follows or replaces an existing file or link."""
    stem, suffix = path.stem, path.suffix
    for n in range(1, 100):
        target = path if n == 1 else path.with_name(f"{stem}-{n}{suffix}")
        try:
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        except FileExistsError:
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        return target
    raise RuntimeError(f"Could not find a free file name for {path}")


def slug(text: str, default: str = "questions") -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or default


def new_call_id(started: datetime) -> str:
    return started.strftime("%Y%m%d-%H%M%S-") + os.urandom(3).hex()


def call_dir(call_id: str) -> Path:
    if not re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{6}", call_id):
        raise ValueError("Not a call id.")
    return CALLS / call_id


def read_index() -> list[dict]:
    """The saved calls, newest first. A damaged index is set aside rather than lost."""
    if not CALLS_INDEX.exists():
        return []
    try:
        calls = json.loads(CALLS_INDEX.read_text())
        if not isinstance(calls, list):
            raise ValueError("not a list")
        return [c for c in calls if isinstance(c, dict)]
    except (OSError, ValueError):
        damaged = CALLS_INDEX.with_name(f"calls.damaged-{datetime.now():%Y%m%d-%H%M%S}.json")
        try:
            os.replace(CALLS_INDEX, damaged)
        except OSError:
            pass
        return []


def add_to_index(entry: dict) -> None:
    calls = [c for c in read_index() if c.get("id") != entry.get("id")] + [entry]
    calls.sort(key=lambda c: str(c.get("started", "")), reverse=True)
    write_atomic(CALLS_INDEX, json.dumps(calls[:INDEX_LIMIT], indent=1))


def _call_folders() -> list[Path]:
    if not CALLS.exists():
        return []
    return [f for f in sorted(CALLS.iterdir()) if re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{6}", f.name)]


def interrupted_calls() -> list[Path]:
    """Calls whose state was checkpointed but never finished saving, for example after a crash.
    results.json is written last when a call is saved, so its absence means unfinished."""
    return [f for f in _call_folders() if (f / "state.json").exists() and not (f / "results.json").exists()]


def finished_calls() -> list[Path]:
    return [f for f in _call_folders() if (f / "results.json").exists()]
