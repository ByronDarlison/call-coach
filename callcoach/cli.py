"""call-coach-cli: run any Call Coach MCP tool from a shell.

For agents that can run commands but cannot connect to MCP themselves.

    call-coach-cli tools
    call-coach-cli load_questions_file path=~/interviews/founder.md
    call-coach-cli start_call
    call-coach-cli wait_for_update cursor=0 timeout_seconds=120
    call-coach-cli add_question text="Who were those six?" under_question=1
    call-coach-cli stop_call

Arguments are name=value; numbers and true/false are converted, anything else is text.
A single JSON object also works: call-coach-cli add_question '{"text": "...", "under_question": 1}'.
The result is printed as JSON. Set CALL_COACH_URL to reach a copy on another port.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

URL = os.environ.get("CALL_COACH_URL", "http://127.0.0.1:4800/mcp")


def parse_args(args: list[str]) -> dict:
    if len(args) == 1 and args[0].lstrip().startswith("{"):
        return json.loads(args[0])
    out = {}
    for arg in args:
        name, sep, value = arg.partition("=")
        if not sep:
            raise SystemExit(f"Expected name=value, got: {arg}")
        if value.lower() in ("true", "false"):
            out[name] = value.lower() == "true"
        else:
            try:
                out[name] = int(value)
            except ValueError:
                out[name] = os.path.expanduser(value) if name == "path" else value
    return out


async def run(tool: str, arguments: dict) -> int:
    async with streamablehttp_client(URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            if tool == "tools":
                listed = await session.list_tools()
                print(json.dumps([{"name": t.name, "description": t.description} for t in listed.tools], indent=1))
                return 0
            result = await session.call_tool(tool, arguments)
            texts = [c.text for c in result.content if getattr(c, "type", "") == "text"]
            payload = result.structuredContent if result.structuredContent is not None else (
                json.loads(texts[0]) if len(texts) == 1 and texts[0][:1] in "[{" else "\n".join(texts))
            print(json.dumps({"ok": not result.isError, "result": payload}, indent=1))
            return 1 if result.isError else 0


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        raise SystemExit(0)
    try:
        code = asyncio.run(run(sys.argv[1], parse_args(sys.argv[2:])))
    except Exception:  # connection refused surfaces as an ExceptionGroup from the HTTP client
        print(json.dumps({"ok": False, "error": f"Could not reach Call Coach at {URL}. Is it running?"}))
        code = 2
    raise SystemExit(code)


if __name__ == "__main__":
    main()
