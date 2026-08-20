"""Web viewer for training streams: serves the page and mints the tokens it needs.

Why a backend exists at all: page source is public, so the LiveKit API secret cannot live in it.
This service holds the secret, hands out short-lived subscribe-only tokens, and lists the rooms
that are actually publishing -- the two things the browser is not allowed to do for itself
(the room list needs an admin token, and LiveKit's HTTP API is not CORS-open to pages anyway).

Standard library only, on purpose: this is the one component that holds a credential, so it
carries no dependency it does not need.

Configuration (environment):
  LIVEKIT_KEYS         LiveKit's own format, "api_key: api_secret" -- mount the same Secret the
                       server uses rather than a second copy of the same credential.
  LIVEKIT_API_KEY      Split form. Wins over LIVEKIT_KEYS when set.
  LIVEKIT_API_SECRET
  LIVEKIT_URL          Where THIS SERVICE reaches LiveKit (in-cluster Service).
  LIVEKIT_PUBLIC_URL   Where THE BROWSER reaches LiveKit (public CLB or domain). Defaults to
                       LIVEKIT_URL, which is right only when running outside the cluster.
  PORT                 Listen port, default 8080.
  TOKEN_TTL            Viewer token lifetime in seconds, default 3600.
  VIEWER_AUTH_USERNAME HTTP Basic Auth for the page and the API. Unset refuses to start, so a
  VIEWER_AUTH_PASSWORD missing Secret is a visible failure rather than a silently open page.
  VIEWER_AUTH_DISABLED "1" to run without authentication -- a decision someone has to write down.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

VERSION = "0.1.0"
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

PORT = int(os.environ.get("PORT", "8080"))
TOKEN_TTL = int(os.environ.get("TOKEN_TTL", "3600"))

_ROOM_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _credentials() -> tuple[str, str]:
    key = os.environ.get("LIVEKIT_API_KEY", "").strip()
    secret = os.environ.get("LIVEKIT_API_SECRET", "").strip()
    if not key:
        keys = os.environ.get("LIVEKIT_KEYS", "").strip()
        if keys:
            key, _, secret = keys.splitlines()[0].partition(":")
            key, secret = key.strip(), secret.strip()
    return key, secret


def _basic_auth() -> tuple[str, str]:
    """Viewer login. VIEWER_AUTH_USERNAME / VIEWER_AUTH_PASSWORD is the agreed contract with the
    chart; the other spellings are accepted as a courtesy so a mis-wired Secret still authenticates
    instead of silently starting an open page.
    """
    for user_key, pass_key in (("VIEWER_AUTH_USERNAME", "VIEWER_AUTH_PASSWORD"),
                               ("VIEWER_USERNAME", "VIEWER_PASSWORD"),
                               ("username", "password")):
        user = os.environ.get(user_key, "").strip()
        if user:
            return user, os.environ.get(pass_key, "")
    return "", ""


API_KEY, API_SECRET = _credentials()
AUTH_USER, AUTH_PASS = _basic_auth()
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "").strip()
PUBLIC_URL = os.environ.get("LIVEKIT_PUBLIC_URL", "").strip() or LIVEKIT_URL


def _http_base(url: str) -> str:
    """LiveKit's admin API is HTTP; the same endpoint is configured as a ws:// URL for clients."""
    return url.replace("wss://", "https://", 1).replace("ws://", "http://", 1).rstrip("/")


def _b64(raw: bytes) -> bytes:
    return base64.urlsafe_b64encode(raw).rstrip(b"=")


def sign_token(identity: str, ttl: int, video_grant: dict) -> str:
    now = int(time.time())
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload = _b64(json.dumps({
        "iss": API_KEY,
        "sub": identity,
        "name": identity,
        # A little slack on nbf: viewer clocks are not ours to trust.
        "nbf": now - 10,
        "exp": now + ttl,
        "video": video_grant,
    }, separators=(",", ":")).encode())
    body = header + b"." + payload
    sig = _b64(hmac.new(API_SECRET.encode(), body, hashlib.sha256).digest())
    return (body + b"." + sig).decode()


def _num(value) -> int:
    """Protobuf JSON sends int64 as a string, and this bit already cost us an empty room list once."""
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


def list_rooms() -> list[dict]:
    token = sign_token("viewer-service", 60, {"roomList": True})
    req = urllib.request.Request(
        _http_base(LIVEKIT_URL) + "/twirp/livekit.RoomService/ListRooms",
        data=b"{}",
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
    )
    # One retry: the hop to LiveKit drops connections often enough that a single reset would
    # otherwise surface as "cannot read the room list" on a page that is otherwise fine. An HTTP
    # status is a real answer and is never retried -- 401 twice is still 401.
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                payload = json.loads(resp.read().decode() or "{}")
            break
        except urllib.error.HTTPError:
            raise
        except OSError:
            if attempt:
                raise
            time.sleep(0.5)

    now = int(time.time())
    rooms = []
    for room in payload.get("rooms", []):
        # Field names arrive snake_case from the server and camelCase from some proxies; read both.
        created = _num(room.get("creation_time") or room.get("creationTime"))
        publishers = _num(room.get("num_publishers") or room.get("numPublishers"))
        rooms.append({
            "name": room.get("name", ""),
            "publishers": publishers,
            "participants": _num(room.get("num_participants") or room.get("numParticipants")),
            "ageSeconds": max(0, now - created) if created else 0,
            "live": publishers > 0,
        })
    rooms.sort(key=lambda r: (not r["live"], r["name"]))
    return rooms


class Handler(BaseHTTPRequestHandler):
    server_version = f"isaac-viewer/{VERSION}"

    def log_message(self, fmt, *args):  # noqa: A003 - quieter default logging
        if not self.path.startswith("/api/rooms"):  # the page polls this; do not flood the log
            super().log_message(fmt, *args)

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def _static(self, name: str) -> None:
        path = os.path.normpath(os.path.join(STATIC_DIR, name))
        if not path.startswith(STATIC_DIR) or not os.path.isfile(path):
            self._json(404, {"error": "not found"})
            return
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
        }.get(os.path.splitext(path)[1], "application/octet-stream")
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype)

    def _authorized(self) -> bool:
        if not AUTH_USER and not AUTH_PASS:  # no login configured: the page is open
            return True
        header = self.headers.get("Authorization", "")
        if not header.startswith("Basic "):
            return False
        try:
            user, _, password = base64.b64decode(header[6:]).decode("utf-8").partition(":")
        except Exception:  # noqa: BLE001 - a malformed header is simply not authorized
            return False
        # Constant time, so a wrong password cannot be narrowed down by how long the reply takes.
        return (hmac.compare_digest(user.encode(), AUTH_USER.encode())
                and hmac.compare_digest(password.encode(), AUTH_PASS.encode()))

    def _challenge(self) -> None:
        body = b'{"error": "unauthorized"}'
        self.send_response(401)
        # Sending the challenge is what makes the browser show its own login prompt, so the page
        # needs no login form of its own.
        self.send_header("WWW-Authenticate", 'Basic realm="Training viewer", charset="UTF-8"')
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        route = urlparse(self.path)
        # Everything is behind the login except the health check, which kubelet probes call
        # without credentials. Protecting only the page would leave /api/token open, and that is
        # the endpoint worth protecting.
        if route.path != "/healthz" and not self._authorized():
            self._challenge()
            return
        if route.path in ("/", "/index.html"):
            self._static("index.html")
        elif route.path.startswith("/static/"):
            self._static(route.path[len("/static/"):])
        elif route.path == "/healthz":
            self._json(200, {"ok": True, "version": VERSION, "configured": bool(API_KEY and LIVEKIT_URL)})
        elif route.path == "/api/rooms":
            self._rooms()
        elif route.path == "/api/token":
            self._token(parse_qs(route.query).get("room", [""])[0])
        else:
            self._json(404, {"error": "not found"})

    def _rooms(self) -> None:
        if not (API_KEY and LIVEKIT_URL):
            self._json(500, {"error": "服务端未配置 LiveKit 地址或凭据"})
            return
        try:
            self._json(200, {"rooms": list_rooms()})
        except urllib.error.HTTPError as e:
            self._json(502, {"error": f"LiveKit 拒绝了请求({e.code}),凭据可能不匹配"})
        except Exception as e:  # noqa: BLE001 - surface the cause to the page rather than a blank 500
            self._json(502, {"error": f"无法连接 LiveKit:{type(e).__name__}"})

    def _token(self, room: str) -> None:
        if not _ROOM_RE.match(room or ""):
            self._json(400, {"error": "房间名不合法"})
            return
        try:
            existing = {r["name"] for r in list_rooms()}
        except Exception:  # noqa: BLE001 - if the check cannot run, refuse rather than mint blindly
            self._json(502, {"error": "无法确认房间是否存在"})
            return
        # Only rooms that exist: a token for an unknown name would have LiveKit create an empty
        # room on join, which is a free way for anyone with the page to litter the server.
        if room not in existing:
            self._json(404, {"error": "房间不存在或已结束"})
            return

        identity = f"viewer-{base64.urlsafe_b64encode(os.urandom(6)).decode().rstrip('=')}"
        token = sign_token(identity, TOKEN_TTL, {
            "roomJoin": True,
            "room": room,
            "canSubscribe": True,
            # A viewer watches. Publishing rights would let anyone with the page put video on the
            # server, and this token is handed to every visitor.
            "canPublish": False,
            "canPublishData": False,
        })
        self._json(200, {"url": PUBLIC_URL, "room": room, "token": token, "expiresIn": TOKEN_TTL})


def main() -> None:
    # Refuse to start unconfigured rather than start open. This page lists every running training
    # and plays it, and it is published to the internet by default, so "forgot to set it" and
    # "chose not to set it" would otherwise look identical from outside -- and the first is far
    # more common. Failing at startup is visible; running unauthenticated is not.
    if not AUTH_USER and os.environ.get("VIEWER_AUTH_DISABLED") != "1":
        raise SystemExit(
            "[viewer] 未设置 VIEWER_AUTH_USERNAME / VIEWER_AUTH_PASSWORD,拒绝启动。\n"
            "         确实要在无鉴权状态下运行,请显式设置 VIEWER_AUTH_DISABLED=1。"
        )
    auth = f"已启用(Basic Auth,用户 {AUTH_USER})" if AUTH_USER else "⚠ 已显式关闭 —— 任何人可访问"
    lines = [f"{VERSION} 监听 :{PORT}", f"访问控制 {auth}",
             f"LiveKit(服务端用) {LIVEKIT_URL or '<未设置>'}",
             f"LiveKit(页面用)   {PUBLIC_URL or '<未设置>'}"]
    if not API_KEY or not API_SECRET:
        lines.insert(0, "未配置凭据:请设置 LIVEKIT_KEYS 或 LIVEKIT_API_KEY/LIVEKIT_API_SECRET")
    # Flushed: the container runs with -u, but a local run without it buffers these away.
    print("\n".join("[viewer] " + line for line in lines), flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
