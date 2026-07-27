"""Studio static-file serving: the instrument's own HTML/CSS/JS, served straight off disk from
`studio/` (DEMO) -- no build step at serve time, no templating. Mechanical extraction of
clozn.server.app's `_html` helper + the do_GET literal-root / asset-suffix branches; "/" and
"/index.html" point at the current Studio app.

The served app is `studio/next/` -- the built output of the `studio-frontend/` React source. Unlike
the hand-written apps that came before it, this one is COMPILED: edit `studio-frontend/src/**`, run
its build, and commit the regenerated `studio/next/` bundle. Editing `studio/next/assets/*.js` by
hand edits minified output that the next build silently overwrites.
"""
import os

from clozn.server.config import DEMO

# The Studio app's canonical entry point. `clozn studio --open` targets the bare root
# (cli/commands/studio.py builds `http://127.0.0.1:<port>`), so it inherits this redirect for free --
# there is no second copy of this path to keep in sync.
#
# Vite is configured with `base: "./"` (studio-frontend/vite.config.ts), so index.html references its
# bundle RELATIVELY (./assets/index-*.js). That is what makes serving it from a subdirectory work at
# all: the browser resolves those to /next/assets/*, which the suffix branch below already serves. If
# that base is ever changed to "/", the asset requests become /assets/* -- outside studio/next/ -- and
# the app loads a blank page with two 404s. Keep them in sync.
APP_INDEX = "/next/index.html"

# The casting: the project's art half -- a live-fed, forkable, WebGL-shaded rendering, kept because
# clozn is an art project as much as a science one.
#
# It lives at its OWN url under studio/casting/, deliberately not inside any frontend. It was
# previously a set of files inside the studio/app/ shell, and when that shell was retired the casting
# went with it -- deleted as collateral, because the safety check asked only "does anything BREAK if
# this goes?" (no tests pinned it, no routes called it) and art has no callers. Standalone is what
# makes that mistake unrepeatable: retiring the next frontend cannot take it along.
CASTING_INDEX = "/casting/casting-demo.html"


def try_get(handler, path):
    """Studio static GETs: "/", "/index.html", "/next", "/next/" (and the legacy "/app", "/app/")
    302-redirect to APP_INDEX; any other .html/.css/.js/.mjs file under DEMO (including subdirs like
    next/, next/assets/) is served directly off disk, guarded against escaping DEMO. Returns True iff
    handled.

    Why a redirect and not serving app/index.html's bytes in place at "/": it loads itself entirely
    through RELATIVE references (./tokens.css, ./app.mjs, and the dynamic ./lens.mjs etc. the router
    imports). Those only resolve correctly if the browser's document URL is under /app/ -- serving the
    same bytes at "/" would break every one of them (they'd resolve to /tokens.css, /app.mjs, which
    don't exist) unless each reference were rewritten to an absolute /app/... URL. A redirect needs
    none of that bookkeeping and can never drift out of sync as the app grows new relative
    imports/assets; it also costs nothing extra here since APP_INDEX is itself served by the plain
    suffix branch below, a path already exercised by every /app/*.{html,css,mjs} request today.

    "/app" and "/app/" are handled because they are what a person actually types, and the suffix branch
    below only matches paths ending in a known asset extension -- without this they 404, which is
    exactly what they used to do.
    """
    if path in ("/", "/index.html", "/next", "/next/", "/app", "/app/"):
        handler._send(302, "", "text/plain; charset=utf-8", {"Location": APP_INDEX})
        return True
    if path in ("/casting", "/casting/"):
        handler._send(302, "", "text/plain; charset=utf-8", {"Location": CASTING_INDEX})
        return True
    if path.endswith((".html", ".css", ".js", ".mjs")):
        root = os.path.abspath(DEMO)
        fn = os.path.abspath(os.path.join(root, path.lstrip("/")))   # serve subdirs (pages/, heavn/) too
        try:
            contained = os.path.commonpath((root, fn)) == root
        except (OSError, ValueError):
            contained = False
        if contained and os.path.isfile(fn):
            ct = ("text/html" if path.endswith(".html") else
                  "text/css" if path.endswith(".css") else "application/javascript")   # .js and .mjs (ES modules)
            with open(fn, encoding="utf-8") as handle:
                content = handle.read()
            handler._send(200, content, ct + "; charset=utf-8")
            return True
    return False
