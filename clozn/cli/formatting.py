"""Small terminal formatting helpers shared by the retained CLI commands."""
from __future__ import annotations

import os
import sys


DIM = BOLD = RST = ""


def _setup_console() -> None:
    """Configure UTF-8 output and basic ANSI emphasis when the terminal supports it."""
    global DIM, BOLD, RST
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    use_ansi = sys.stderr.isatty() or bool(os.environ.get("CLICOLOR_FORCE"))
    if os.name == "nt" and sys.stderr.isatty():
        try:
            import ctypes

            kernel = ctypes.windll.kernel32
            for handle_id in (-11, -12):
                handle = kernel.GetStdHandle(handle_id)
                mode = ctypes.c_uint32()
                if kernel.GetConsoleMode(handle, ctypes.byref(mode)):
                    kernel.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:
            use_ansi = bool(os.environ.get("CLICOLOR_FORCE"))
    if use_ansi:
        DIM, BOLD, RST = "\033[2m", "\033[1m", "\033[0m"


def _confbar(value: float, width: int = 8) -> str:
    """Render the bounded progress bar used while downloading a model."""
    full = int(round(max(0.0, min(1.0, value)) * width))
    return "█" * full + "░" * (width - full)
