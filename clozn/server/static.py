"""Serve the compiled Studio application and its static assets from ``studio/``.

Studio source lives in ``studio-frontend/`` and builds into ``studio/app/``. The server performs no
templating or frontend build work at runtime.
"""
import os

from clozn.server.config import DEMO

# ``clozn studio --open`` opens the bare server root, which redirects here. Vite's ``base: "./"``
# keeps every generated asset reference relative to this directory.
APP_INDEX = "/app/index.html"

def try_get(handler, path):
    """Serve Studio entry points and contained HTML, CSS, JavaScript, and font assets."""
    if path in ("/", "/index.html", "/app", "/app/"):
        handler._send(302, "", "text/plain; charset=utf-8", {"Location": APP_INDEX})
        return True
    text_types = {
        ".html": "text/html",
        ".css": "text/css",
        ".js": "application/javascript",
        ".mjs": "application/javascript",
    }
    binary_types = {
        ".woff2": "font/woff2",
    }
    extension = os.path.splitext(path)[1].lower()
    content_type = text_types.get(extension) or binary_types.get(extension)
    if content_type is not None:
        root = os.path.abspath(DEMO)
        fn = os.path.abspath(os.path.join(root, path.lstrip("/")))   # serve subdirs (pages/, heavn/) too
        try:
            contained = os.path.commonpath((root, fn)) == root
        except (OSError, ValueError):
            contained = False
        if contained and os.path.isfile(fn):
            binary = extension in binary_types
            with open(fn, "rb" if binary else "r", encoding=None if binary else "utf-8") as handle:
                content = handle.read()
            charset = "" if binary else "; charset=utf-8"
            handler._send(200, content, content_type + charset)
            return True
    return False
