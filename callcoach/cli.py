"""call-coach-cli: run any Call Coach MCP tool from a shell.

For agents that can run commands but cannot connect to MCP themselves.

    call-coach-cli tools
    call-coach-cli load_questions_file path=~/interviews/founder.md
    call-coach-cli start_call
    call-coach-cli wait_for_update cursor=<cursor> timeout_seconds=120
    call-coach-cli stop_call

When an argument holds text an AI wrote, or anything heard on a call, never put it inside the command
line: a shell expands $(...), backticks and quotes inside it. Send the arguments as JSON on stdin
through a quoted heredoc instead, which the shell passes through untouched:

    call-coach-cli add_question --json-stdin <<'CALL_COACH_JSON'
    {"text": "What made you pick March?", "under_question": 3, "from_line": 41}
    CALL_COACH_JSON

Other arguments are name=value; numbers and true/false are converted, anything else is text.
The result is printed as JSON: {"ok": true, "result": ...} or {"ok": false, "error": {"kind", "message"}}.
Exit codes: 0 ok, 1 the tool reported an error, 2 Call Coach is not reachable, 3 bad arguments.
The owner's MCP token is read from ~/Library/Application Support/call-coach/token; with
CALL_COACH_SCOPE=coach, the restricted coaching token (coach-token) is used instead.
Set CALL_COACH_URL to reach a copy on another port.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import timedelta

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from . import security

URL = os.environ.get("CALL_COACH_URL", f"http://127.0.0.1:{os.environ.get('CALL_COACH_PORT', '4800')}/mcp")
SCOPE = "coach" if os.environ.get("CALL_COACH_SCOPE") == "coach" else "owner"


class UsageError(Exception):
    pass


def parse_args(args: list[str], stdin=None) -> dict:
    if args == ["--json-stdin"]:
        try:
            value = json.loads((stdin or sys.stdin).read())
        except ValueError as exc:
            raise UsageError(f"stdin is not valid JSON: {exc}")
        if not isinstance(value, dict):
            raise UsageError("stdin must hold one JSON object.")
        return check(value)
    if len(args) == 1 and args[0].lstrip().startswith("{"):
        try:
            value = json.loads(args[0])
        except ValueError as exc:
            raise UsageError(f"not valid JSON: {exc}")
        if not isinstance(value, dict):
            raise UsageError("The JSON argument must be an object.")
        return check(value)
    out = {}
    for arg in args:
        name, sep, value = arg.partition("=")
        if not sep or not name:
            raise UsageError(f"Expected name=value, got: {arg}")
        if value.lower() in ("true", "false"):
            out[name] = value.lower() == "true"
        else:
            try:
                out[name] = int(value)
            except ValueError:
                out[name] = os.path.expanduser(value) if name == "path" else value
    return check(out)


def check(arguments: dict) -> dict:
    if "timeout_seconds" in arguments:
        value = arguments["timeout_seconds"]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 300:
            raise UsageError("timeout_seconds must be a whole number from 1 to 300.")
    return arguments


def emit(payload: dict) -> None:
    print(json.dumps(payload, indent=1))


async def run(tool: str, arguments: dict) -> int:
    headers = {"Authorization": f"Bearer {security.token(SCOPE)}"}
    # Stop can take a while: it drains transcription (30 s), reviews (30 s) and the last checks (60 s).
    wait = {"wait_for_update": arguments.get("timeout_seconds", 120), "stop_call": 150, "start_call": 90}.get(tool, 60)
    async with streamablehttp_client(URL, headers=headers, timeout=30,
                                     sse_read_timeout=wait + 60) as (read, write, _):
        async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=wait + 30)) as session:
            await session.initialize()
            if tool == "tools":
                listed = await session.list_tools()
                emit({"ok": True, "result": [{"name": t.name, "description": t.description,
                                              "input": t.inputSchema} for t in listed.tools]})
                return 0
            result = await session.call_tool(tool, arguments)
            texts = [c.text for c in result.content if getattr(c, "type", "") == "text"]
            if result.isError:
                emit({"ok": False, "error": {"kind": "tool", "message": "\n".join(texts)}})
                return 1
            payload = result.structuredContent if result.structuredContent is not None else (
                json.loads(texts[0]) if len(texts) == 1 and texts[0][:1] in "[{" else "\n".join(texts))
            if isinstance(payload, dict) and set(payload) == {"result"}:
                payload = payload["result"]
            emit({"ok": True, "result": payload})
            return 0


def _connection_error(exc: BaseException) -> bool:
    import httpx
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, ConnectionError)):
        return True
    return any(_connection_error(e) for e in getattr(exc, "exceptions", []))


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        raise SystemExit(0)
    try:
        arguments = parse_args(sys.argv[2:])
    except UsageError as exc:
        emit({"ok": False, "error": {"kind": "usage", "message": str(exc)}})
        raise SystemExit(3)
    try:
        code = asyncio.run(run(sys.argv[1], arguments))
    except BaseException as exc:  # the HTTP client wraps failures in exception groups
        if isinstance(exc, KeyboardInterrupt):
            raise
        if _connection_error(exc):
            emit({"ok": False, "error": {"kind": "unreachable",
                                         "message": f"Could not reach Call Coach at {URL}. Is it running?"}})
            code = 2
        else:
            message = str(exc) or type(exc).__name__
            for inner in getattr(exc, "exceptions", []):
                message = str(inner) or message
            kind = "auth" if "401" in message else "protocol"
            emit({"ok": False, "error": {"kind": kind, "message": message[:500]}})
            code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()
