"""Loopback desktop HTTP/WebSocket endpoint with a read-only results browser."""

import html
import io
import json
import os
import signal
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit

from websockify.websocketproxy import ProxyRequestHandler, WebSocketProxy

STATIC = Path("/opt/novnc")
RESULTS = Path("/results")
RUNTIME = Path("/tmp/organoidtracker")


def confined(root: Path, name: str) -> Path:
    """Reject traversal and escaping symlinks before opening a served file."""
    target = (root / name).resolve()
    target.relative_to(root.resolve())
    return target


class Handler(ProxyRequestHandler):
    def log_message(self, fmt, *args):
        # Avoid logging result filenames or full browser URLs.
        pass

    def allowed_host(self):
        try:
            host = urlsplit("http://" + self.headers.get("Host", "")).hostname
            return host in ("localhost", "127.0.0.1", "::1")
        except ValueError:
            return False

    def same_origin(self):
        return self.headers.get("Origin") == "http://" + self.headers.get("Host", "")

    def validate_connection(self):
        if not self.allowed_host() or not self.same_origin():
            raise self.server.EClose("Only the local application browser may connect")
        super().validate_connection()

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()

    def reply(self, data, content_type="application/json", status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def do_GET(self):
        if not self.allowed_host():
            self.send_error(403)
            return
        if urlsplit(self.path).path == "/api/status":
            state: dict[str, Any] = {"ready": (RUNTIME / "ready").is_file()}
            try:
                report = json.loads((RUNTIME / "report.json").read_text())
                state["report"] = {
                    "url": "/reports/" + quote(report["directory"]) + "/",
                    "sequence": report["sequence"],
                }
            except (OSError, ValueError, KeyError):
                pass
            self.reply(json.dumps(state).encode())
        else:
            super().do_GET()

    def do_POST(self):
        if (
            self.allowed_host()
            and self.same_origin()
            and self.headers.get("X-OrganoidTracker") == "local"
            and self.path == "/api/stop"
        ):
            self.reply(b'{"stopping":true}')
            os.kill(int(os.environ["OT_SUPERVISOR_PID"]), signal.SIGTERM)
        else:
            self.send_error(403)

    def send_head(self):
        if not self.allowed_host():
            self.send_error(403)
            return None
        route = unquote(urlsplit(self.path).path)
        root, relative = (RESULTS, route[9:]) if route.startswith("/reports/") else (STATIC, route.lstrip("/"))
        if route == "/":
            relative = "index.html"
        try:
            target = confined(root, relative)
            if root == RESULTS and target.is_dir():
                entries = []
                for item in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.casefold())):
                    try:
                        confined(root, str(item.relative_to(root)))
                    except ValueError:
                        continue
                    if not item.is_dir() and not item.is_file():
                        continue
                    name = item.name + ("/" if item.is_dir() else "")
                    link = "/reports/" + quote(str(item.relative_to(root))) + ("/" if item.is_dir() else "")
                    entries.append(f'<li><a href="{link}">{html.escape(name)}</a></li>')
                parent = "/reports/" + quote(str(target.parent.relative_to(root))) + "/" if target != root else "/"
                body = (
                    '<!doctype html><meta charset="utf-8"><title>Organoid Tracker results</title>'
                    "<style>body{font:18px system-ui;margin:2rem}li{margin:.6rem}</style>"
                    '<a href="/">Back to application</a> · '
                    f'<a href="{parent}">Parent folder</a><h1>Results</h1>'
                    "<p>These files are also in your selected host results folder.</p><ul>" + "".join(entries) + "</ul>"
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                return io.BytesIO(body)
            if not target.is_file():
                raise FileNotFoundError
            stream = target.open("rb")
            self.send_response(200)
            kind = self.guess_type(str(target))
            self.send_header("Content-Type", kind)
            if root == RESULTS:
                self.send_header("Content-Disposition", "attachment; filename*=UTF-8''" + quote(target.name))
            self.send_header("Content-Length", str(os.fstat(stream.fileno()).st_size))
            self.end_headers()
            return stream
        except (OSError, ValueError):
            self.send_error(404)
            return None


def main():
    server = WebSocketProxy(
        RequestHandlerClass=Handler,
        listen_host="0.0.0.0",
        listen_port=6080,
        target_host="127.0.0.1",
        target_port=5900,
        web=str(STATIC),
    )
    server.start_server()


if __name__ == "__main__":
    main()
