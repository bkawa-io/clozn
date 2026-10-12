"""Static-serving contract for the compiled Studio bundle in ``studio/app/``.

The bundle filenames are content-hashed and the app is served from a subdirectory, so two failures
would otherwise produce a blank page rather than a useful error:

  1. The bundle filenames are content-hashed (index-BGckJFMk.js). A rebuild changes them. If
     index.html and the committed assets/ ever go out of sync -- one rebuilt, the other not -- the
     page loads and silently 404s its own script.
  2. Vite's ``base`` controls whether ``index.html`` references its bundle relatively or absolutely.
     Asset URLs must stay relative to ``/app/``.

So these tests assert what the browser actually does: read the committed index.html, resolve every
asset it references the way a browser would, and require the static route to serve each one.
"""
from __future__ import annotations

import os
import re
import tempfile
import unittest
from unittest import mock

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
    def test_the_studio_tree_contains_only_current_product_surfaces(self):
        directories = {
            name for name in os.listdir(DEMO)
            if os.path.isdir(os.path.join(DEMO, name))
        }
        self.assertLessEqual(directories, {"app", "casting", "ext", "__pycache__"})

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

    def test_app_short_paths_redirect_to_the_app_index(self):
        for path in ("/app", "/app/"):
            with self.subTest(path=path):
                handled, sent = _serve(path)
                self.assertTrue(handled)
                self.assertEqual(sent[0], 302)
                self.assertEqual(sent[3].get("Location"), static_routes.APP_INDEX)

    def test_app_woff2_assets_are_served_as_binary_fonts(self):
        assets = os.path.join(DEMO, "app", "assets")
        fonts = sorted(name for name in os.listdir(assets) if name.endswith(".woff2"))
        self.assertTrue(fonts, "the committed Studio bundle has no .woff2 asset to exercise")

        filename = fonts[0]
        expected_path = os.path.join(assets, filename)
        with open(expected_path, "rb") as handle:
            expected = handle.read()

        handled, sent = _serve(f"/app/assets/{filename}")
        self.assertTrue(handled)
        status, body, content_type, _headers = sent
        self.assertEqual(status, 200)
        self.assertIsInstance(body, bytes)
        self.assertEqual(body, expected)
        self.assertEqual(content_type, "font/woff2")

    def test_binary_asset_serving_does_not_weaken_path_containment(self):
        with tempfile.TemporaryDirectory() as directory:
            demo = os.path.join(directory, "studio")
            os.mkdir(demo)
            outside = os.path.join(directory, "outside.woff2")
            with open(outside, "wb") as handle:
                handle.write(b"not a font")

            with mock.patch.object(static_routes, "DEMO", demo):
                handled, sent = _serve("/../outside.woff2")

            self.assertFalse(handled)
            self.assertIsNone(sent)

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

        index_dir = os.path.dirname(static_routes.APP_INDEX)
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
        """The casting is an independent project surface with its own entry point and module graph."""
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

if __name__ == "__main__":
    unittest.main()
