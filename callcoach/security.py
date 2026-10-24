"""Who may talk to Call Coach.

Call Coach only listens on 127.0.0.1, but any web page open in your browser can also reach 127.0.0.1, and
so can any program on the Mac. So:

- Every request must name this Mac's loopback address in its Host header, which blocks DNS rebinding.
- MCP clients (and call-coach-cli) send `Authorization: Bearer <token>`. Tokens are random 256-bit values
  in ~/Library/Application Support/call-coach, readable only by you: `token` (the owner, everything) and
  `coach-token` (a live coaching assistant: follow-ups and the call it is following, no Stop, ticks,
  steering or history), so speech that misleads the assistant cannot stop or rewrite the call.
- The page signs in once through `call-coach open`, which writes a one-time code the page swaps for a
  cookie. The cookie is HttpOnly and SameSite=Strict, and every change the page makes must also come from
  this origin and carry an X-Call-Coach header, which another site cannot send without permission.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from http.cookies import SimpleCookie
from urllib.parse import parse_qs

from . import store

TOKEN_FILES = {"owner": store.SUPPORT / "token", "coach": store.SUPPORT / "coach-token"}
LOGIN_CODE = store.SUPPORT / "login-code"
COOKIE = "cc_session"
LOGIN_SECONDS = 120


def token(kind: str = "owner") -> str:
    """An MCP token, created on first use. "owner" can do everything; "coach" only what a live coaching
    assistant needs (see the tools marked owner only)."""
    path = TOKEN_FILES[kind]
    store.private_dir(path.parent)
    try:
        value = path.read_text().strip()
        if len(value) >= 64:
            return value
    except OSError:
        pass
    value = secrets.token_hex(32)
    store.write_atomic(path, value + "\n")
    return value


def page_cookie(tok: str) -> str:
    return hmac.new(tok.encode(), b"call-coach page", hashlib.sha256).hexdigest()


def write_login_code() -> str:
    code = secrets.token_urlsafe(24)
    store.write_atomic(LOGIN_CODE, json.dumps({"code": code, "at": time.time()}))
    return code


def take_login_code(code: str) -> bool:
    try:
        saved = json.loads(LOGIN_CODE.read_text())
        LOGIN_CODE.unlink()
    except (OSError, ValueError):
        return False
    fresh = time.time() - float(saved.get("at", 0)) < LOGIN_SECONDS
    return fresh and hmac.compare_digest(str(saved.get("code", "")), code)


class Guard:
    """ASGI middleware applying the rules above."""

    def __init__(self, app, port: int):
        self.app = app
        self.hosts = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}
        self.origins = {f"http://{h}" for h in self.hosts}
        self.tokens = {f"Bearer {token(kind)}": kind for kind in TOKEN_FILES}
        self.cookie = page_cookie(token())

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {k.decode().lower(): v.decode(errors="replace") for k, v in scope.get("headers", [])}
        path, method = scope.get("path", ""), scope.get("method", "GET")
        if headers.get("host") not in self.hosts:
            return await self._deny(send, 421, "Call Coach answers only on this Mac's loopback address.")
        origin = headers.get("origin")
        if origin and origin not in self.origins:
            return await self._deny(send, 403, "Requests from other sites are not allowed.")
        if path == "/login":
            return await self._login(scope, send)
        if path.startswith("/mcp"):
            auth = headers.get("authorization", "")
            kind = next((k for t, k in self.tokens.items() if hmac.compare_digest(auth, t)), None)
            if not kind:
                return await self._deny(send, 401, "Missing or wrong Call Coach token. See README: Connect an AI.")
            scope.setdefault("state", {})["cc_scope"] = kind
            return await self.app(scope, receive, send)
        if path == "/" or path.startswith("/static/"):
            return await self.app(scope, receive, send)
        cookies = SimpleCookie(headers.get("cookie", ""))
        signed_in = COOKIE in cookies and hmac.compare_digest(cookies[COOKIE].value, self.cookie)
        if not signed_in:
            return await self._deny(send, 401, "Open Call Coach with `call-coach open` to sign this page in.")
        if method not in ("GET", "HEAD") and headers.get("x-call-coach") != "1":
            return await self._deny(send, 403, "Missing X-Call-Coach header.")
        return await self.app(scope, receive, send)

    async def _login(self, scope, send):
        code = (parse_qs(scope.get("query_string", b"").decode()).get("code") or [""])[0]
        if not take_login_code(code):
            return await self._deny(send, 403, "That sign-in link has expired. Run `call-coach open` again.")
        cookie = f"{COOKIE}={self.cookie}; Path=/; Max-Age=31536000; HttpOnly; SameSite=Strict"
        await send({"type": "http.response.start", "status": 303,
                    "headers": [(b"location", b"/"), (b"set-cookie", cookie.encode()),
                                (b"cache-control", b"no-store")]})
        await send({"type": "http.response.body", "body": b""})

    @staticmethod
    async def _deny(send, status: int, message: str):
        body = json.dumps({"ok": False, "error": message}).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"), (b"cache-control", b"no-store")]})
        await send({"type": "http.response.body", "body": body})
