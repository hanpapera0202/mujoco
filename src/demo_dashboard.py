"""Local HTTP control plane for the MuJoCo dual-arm sorting demonstration."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from run_sorting_demo import SortingDemo


DASHBOARD_DIR = Path(__file__).resolve().parent / "dashboard"


@dataclass
class Dashboard:
    server: ThreadingHTTPServer
    thread: threading.Thread
    url: str

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def start_dashboard(demo: SortingDemo, host: str = "127.0.0.1", port: int = 8765) -> Dashboard:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/api/state":
                self._json(HTTPStatus.OK, demo.snapshot())
                return
            filename = "index.html" if self.path in ("/", "/index.html") else self.path.lstrip("/")
            path = (DASHBOARD_DIR / filename).resolve()
            if DASHBOARD_DIR not in path.parents or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type = "text/html; charset=utf-8" if path.suffix == ".html" else "text/css; charset=utf-8" if path.suffix == ".css" else "application/javascript; charset=utf-8"
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/control":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                action = payload.get("action")
                if action == "restart":
                    demo.request_reset()
                elif action == "pause":
                    demo.set_paused(True)
                elif action == "resume":
                    demo.set_paused(False)
                elif action == "settings":
                    demo.update_settings(payload.get("values", {}))
                else:
                    raise ValueError("unknown action")
                self._json(HTTPStatus.OK, demo.snapshot())
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

        def _json(self, status: HTTPStatus, payload: object) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    thread = threading.Thread(target=server.serve_forever, name="nova5-demo-dashboard", daemon=True)
    thread.start()
    return Dashboard(server, thread, f"http://{host}:{port}")
