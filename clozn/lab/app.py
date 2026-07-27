"""HTTP workbench for PyTorch-only Qwen and Dream experiments.

This intentionally has no OpenAI or Clozn-native generation API.  It exists to keep
training/calibration and the research visualizations runnable without turning the lab
model into a second product-serving engine.
"""
from __future__ import annotations

import argparse
import os
import sys
from http.server import ThreadingHTTPServer

# A lab process must never inherit a handle to a product worker. Do this before the first app import;
# avoid mutating an already-running product module when tests merely import this module for its handler.
if "clozn.server.app" not in sys.modules:
    os.environ.pop("CLOZN_ENGINE_PORT", None)
    os.environ["CLOZN_RUNTIME_KIND"] = "lab"

# The lab -- and ONLY the lab -- needs the HF hub symlink workaround (Windows checkouts without symlink
# privilege). This used to load at PRODUCT import time (clozn/server/config.py); it moved here so a
# `clozn serve` process never pulls it in.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

from clozn.server import app as ctx


def make_lab_handler(sub=None, subname=None):
    # Lab builds its OWN handler by INJECTING its substrate/name/kind -- it never reaches into
    # clozn.server.app to mutate SUB/SUBNAME/ARGS/RUNTIME_KIND. Routes + active_subname(self) read the
    # injected values via the getattr fallback, so the product module's globals stay untouched.
    base = ctx.make_handler(sub=sub, subname=subname, runtime_kind="lab")

    class LabHandler(base):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/healthz", "/readyz"):
                self._json(200, {"status": "ok", "service": "clozn-lab", "active": ctx.active_subname(self)})
                return
            if path == "/substrate":
                nm = ctx.active_subname(self)
                self._json(200, {"active": nm, "available": [nm], "service": "clozn-lab"})
                return
            if path.startswith("/v1/") or path.startswith("/api/clozn/"):
                self._json(404, {"error": "the lab workbench does not expose a product generation API"})
                return
            super().do_GET()

        def do_POST(self):
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path == "/substrate":
                self._json(410, {"error": "restart the lab command to choose another workbench"})
                return
            if path.startswith("/v1/") or path.startswith("/api/clozn/"):
                self._json(404, {"error": "the lab workbench does not expose a product generation API"})
                return
            super().do_POST()

    return LabHandler


def main(argv=None):
    parser = argparse.ArgumentParser(description="Clozn's optional PyTorch workbench")
    parser.add_argument("substrate", choices=("qwen",))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args(argv)

    os.environ["CLOZN_RUNTIME_KIND"] = "lab"   # process env (memory_mode.set_mode reads it); NOT a product
    #                                            module global -- the substrate/kind are injected below.
    print(f"clozn lab: loading {args.substrate} ...", flush=True)
    from clozn.lab.substrates import QwenSubstrate
    sub = QwenSubstrate()
    server = ThreadingHTTPServer((args.host, args.port), make_lab_handler(sub, args.substrate))
    print(f"\n  Clozn lab -> http://{args.host}:{args.port}/ ({args.substrate})\n", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
