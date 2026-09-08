"""
Static-file dev server for the product-tree visualizer.

Serves the viz/ HTML/JS, the corpus data files (product_info/,
document_info/), and exposes two narrow endpoints that operate ONLY on
file paths already listed in document_nodes.json (a whitelist -- arbitrary
paths are rejected). Binds to localhost only.

  /open?path=<absolute path>  open that file in the OS's default app
  /thumb?path=<absolute path> a tiny cached JPEG of the source image

The thumbnail endpoint exists because product photos in the corpus run
into tens of megabytes and a viewport full of originals will freeze a
browser tab; thumbs are resized and cached on disk under .thumb_cache/.

Usage:  python serve.py   (from this folder)
Open in browser: http://localhost:8000/
"""

import hashlib
import json
import os
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO

try:
    from PIL import Image
except ImportError:
    Image = None

VIZ_DIR = os.path.dirname(os.path.abspath(__file__))
# viz/ lives directly under tools/, but the data it serves
# (chatbot-corpus/) sits next to tools/ at the repo root, not inside it,
# so we walk up two levels to the repo root and then descend into the
# corpus.
_REPO_ROOT = os.path.dirname(os.path.dirname(VIZ_DIR))
ROOT_DIR = os.path.join(_REPO_ROOT, "chatbot-corpus")
DOCUMENT_NODES_PATH = os.path.join(ROOT_DIR, "document_info", "document_nodes.json")
THUMB_CACHE_DIR = os.path.join(VIZ_DIR, ".thumb_cache")
THUMB_MAX_SIZE = 120
PORT = 8000


def load_allowed_paths():
    with open(DOCUMENT_NODES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {doc["location"]["file_path"] for doc in data["documents"]}


def get_thumbnail_bytes(path):
    os.makedirs(THUMB_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(THUMB_CACHE_DIR, hashlib.sha1(path.encode("utf-8")).hexdigest() + ".jpg")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return f.read()

    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((THUMB_MAX_SIZE, THUMB_MAX_SIZE))
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=78)
        data = buf.getvalue()

    with open(cache_path, "wb") as f:
        f.write(data)
    return data


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT_DIR, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/open":
            self.handle_open(parsed)
            return
        if parsed.path == "/thumb":
            self.handle_thumb(parsed)
            return
        super().do_GET()

    def _whitelisted_path(self, parsed):
        params = urllib.parse.parse_qs(parsed.query)
        path = (params.get("path") or [None])[0]
        return path if path in ALLOWED_PATHS else None

    def handle_open(self, parsed):
        path = self._whitelisted_path(parsed)
        if not path:
            self.send_response(404)
            self.end_headers()
            return
        os.startfile(path)
        self.send_response(204)
        self.end_headers()

    def handle_thumb(self, parsed):
        path = self._whitelisted_path(parsed)
        if not path or Image is None:
            self.send_response(404)
            self.end_headers()
            return
        try:
            data = get_thumbnail_bytes(path)
        except Exception:  # noqa: BLE001 -- thumbnail generation failed: return HTTP 500 to the client and keep the dev server alive
            self.send_response(500)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        pass


def bind_server():
    """Try PORT first and fall forward to the first free port.

    If something else (e.g. VS Code Live Preview) is already on 8000, we
    slide to the next free port instead of crashing silently -- otherwise
    the user opens their browser to that other server's page and concludes
    the files just won't open.
    """
    for port in range(PORT, PORT + 10):
        try:
            return ThreadingHTTPServer(("127.0.0.1", port), Handler), port
        except OSError:
            print(f"Port {port} in use (another server is listening), trying next...")
    raise SystemExit(f"No free port found in {PORT}-{PORT + 9}.")


if __name__ == "__main__":
    ALLOWED_PATHS = load_allowed_paths()
    httpd, port = bind_server()
    with httpd:
        print(f"{len(ALLOWED_PATHS)} file(s) whitelisted for /open and /thumb.")
        print(f"Server: http://localhost:{port}/viz/")
        print("NOTE: Open the page from THIS address -- through VS Code Live")
        print("      Preview or a bare 'python -m http.server', clicking files")
        print("      will not open them.")
        httpd.serve_forever()
