"""Static-serving contract for the compiled Studio bundle (studio/next/).

Unlike every frontend that came before it, the served app is BUILT output: `studio-frontend/` React
source compiled by Vite into `studio/next/`. That introduces two failure modes no previous shell had,
and both produce a blank page rather than an error, so neither is loud without a test:

  1. The bundle filenames are content-hashed (index-BGckJFMk.js). A rebuild changes them. If
     index.html and the committed assets/ ever go out of sync -- one rebuilt, the other not -- the
     page loads and silently 404s its own script.
  2. Vite's `base` controls whether index.html references its bundle relatively (./assets/...) or
     absolutely (/assets/...). Only the relative form survives being served from /next/; the absolute
     form resolves to /assets/*, outside studio/, and 404s.

So these tests assert what the browser actually does: read the committed index.html, resolve every
asset it references the way a browser would, and require the static route to serve each one.
"""
from __future__ import annotations

import os
import re
import unittest

from clozn.server import static as static_routes
from clozn.server.config import DEMO


class FakeHandler:
    """Records what static.try_get would have sent."""

    def __init__(self):
        self.sent = None

    def _send(self, status, content, content_type="", extra_headers=None):
        self.sent = (status, content, content_type, extra_headers or {})


def _serve(path):
    h = FakeHandler()
    handled = static_routes.try_get(h, path)
    return handled, h.sent


def _index_path():
    return os.path.join(DEMO, static_routes.APP_INDEX.lstrip("/").replace("/", os.sep))


class StudioStaticTests(unittest.TestCase):
    def test_root_redirects_to_the_app_index(self):
        for path in ("/", "/index.html"):
            handled, sent = _serve(path)
            self.assertTrue(handled, path)
            status, _body, _ct, headers = sent
            self.assertEqual(status, 302, path)
            self.assertEqual(headers.get("Location"), static_routes.APP_INDEX, path)

    def test_the_app_index_named_by_the_redirect_actually_exists(self):
        """The redirect target must be a real file, or "/" sends every visitor to a 404."""
        self.assertTrue(os.path.isfile(_index_path()),
                        f"{static_routes.APP_INDEX} does not exist under {DEMO}")

    def test_the_app_index_is_served_not_just_present(self):
        handled, sent = _serve(static_routes.APP_INDEX)
        self.assertTrue(handled, "static.try_get refused to serve APP_INDEX")
        status, body, content_type, _headers = sent
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type)
        self.assertIn("<div id=\"root\">", body)      # the React mount point

    def test_every_asset_the_index_references_is_relative_and_servable(self):
        """Parse the committed index.html and resolve its refs like a browser would.

        This is the test that catches a stale/mismatched bundle and a Vite `base` flip. It reads the
        real file rather than a fixture precisely so a rebuild that forgets to commit assets/ fails
        here instead of in someone's browser.
        """
        with open(_index_path(), encoding="utf-8") as handle:
            html = handle.read()

        refs = re.findall(r'(?:src|href)="([^"]+)"', html)
        local = [r for r in refs if not r.startswith(("http://", "https://", "//", "data:"))]
        self.assertTrue(local, "index.html references no local assets -- did the build emit anything?")

        index_dir = os.path.dirname(static_routes.APP_INDEX)          # "/next"
        for ref in local:
            with self.subTest(ref=ref):
                self.assertFalse(
                    ref.startswith("/"),
                    f"{ref} is an ABSOLUTE reference; served from {index_dir}/ the browser resolves "
                    f"it to {ref}, outside studio/. Vite's base must stay './'.",
                )
                resolved = os.path.normpath(os.path.join(index_dir, ref)).replace(os.sep, "/")
                if not resolved.startswith("/"):
                    resolved = "/" + resolved
                handled, sent = _serve(resolved)
                self.assertTrue(handled, f"{resolved} (from {ref}) is not served by static.try_get")
                self.assertEqual(sent[0], 200, f"{resolved} did not return 200")

    def test_the_casting_is_served_and_self_contained(self):
        """The casting is the project's art half, and it has no callers -- which is exactly how it got
        deleted once: it lived inside the studio/app/ shell, and when that shell was retired the
        "does anything break if this goes?" check came back clean, because nothing imports art.

        This test IS its caller. It fails if the casting stops being served, so the next frontend
        retirement cannot quietly take it along. Kept deliberately strict about the entry point and
        its module graph; the demo page's one dangling href (casting-optics.css, never committed) is
        pre-existing and not asserted on.
        """
        handled, sent = _serve("/casting")
        self.assertTrue(handled, "/casting is not routed")
        self.assertEqual(sent[0], 302)
        self.assertEqual(sent[3].get("Location"), static_routes.CASTING_INDEX)

        handled, sent = _serve(static_routes.CASTING_INDEX)
        self.assertTrue(handled, f"{static_routes.CASTING_INDEX} is not served")
        self.assertEqual(sent[0], 200)

        # the module graph the entry point actually loads
        for module in ("/casting/casting.mjs", "/casting/casting-optics.mjs", "/casting/tokens.css"):
            with self.subTest(module=module):
                handled, sent = _serve(module)
                self.assertTrue(handled, f"{module} is not served")
                self.assertEqual(sent[0], 200, f"{module} did not return 200")

    def test_the_retired_shell_is_gone(self):
        """studio/app/ was the previous frontend. Its absence is the point of the cutover; if it
        reappears, two apps are shipping and "/" is ambiguous again."""
        self.assertFalse(os.path.isdir(os.path.join(DEMO, "app")),
                         "studio/app/ is back -- the retired shell should not be shipping")


if __name__ == "__main__":
    unittest.main()
