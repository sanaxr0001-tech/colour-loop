"""The live view: a tiny local web server that the run feeds and the browser polls.

Standard library only. ``/`` serves one static page; ``/state`` returns the run's current state as
JSON. The PyLabRobot Visualizer (the deck simulation) runs its own server and is embedded in the
page as an iframe.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .loop import LiveState

PAGE = Path(__file__).parent / "static" / "index.html"


def _handler(live: LiveState):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server naming)
            if self.path == "/state":
                body = json.dumps(live.snapshot()).encode()
                kind = "application/json"
            elif self.path in ("/", "/index.html"):
                body = PAGE.read_bytes()
                kind = "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:
            pass

    return Handler


class LiveServer:
    def __init__(self, live: LiveState, host: str = "127.0.0.1", port: int = 8765):
        handler = _handler(live)
        for candidate in range(port, port + 20):
            try:
                self._httpd = ThreadingHTTPServer((host, candidate), handler)
                break
            except OSError:
                continue
        else:
            raise OSError(f"no free port in {port}-{port + 19}")
        self.url = f"http://{host}:{self._httpd.server_address[1]}/"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def start(self) -> "LiveServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
