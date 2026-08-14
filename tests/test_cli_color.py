"""test_cli_color -- the confidence heatmap in the CLI (`clozn trace` / `clozn explain`): a terminal echo of
the denoise UI, where brightness = confidence and the palette is denoise.html's exact endpoints (#e07a96 /
#c3cde0 / #7aa7ff). Tests the PURE color helpers against canned trace/moments dicts -- no model, no GPU, no
tty.

The load-bearing safety property: painting is a STRICT no-op when color is off (non-tty, NO_COLOR, or no
ANSI), so piped output and every existing color-off test stay byte-for-byte unchanged; truecolor SGR escapes
appear ONLY when COLOR is on. That's what lets this feature land without touching any existing CLI test.
"""
from __future__ import annotations

import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))            # tests/
REPO = os.path.dirname(HERE)                                  # repo root (clozn/cli.py lives here)
sys.path.insert(0, REPO)

import clozn.cli.main as cli                                      # noqa: E402
import clozn.cli.formatting as fmt                                 # noqa: E402

_ESC = re.compile(r"\033\[[0-9;]*m")


def _visible(s: str) -> str:
    return _ESC.sub("", s)


@pytest.fixture
def color_on(monkeypatch):
    # The color globals live in clozn.cli.formatting (not clozn.cli.main): _paint/_conf_rgb/etc. are
    # defined there and read them live from their OWN module, so a patch has to land on the real owner to
    # be observed. cli.<name> re-exports below (e.g. cli._paint) are the SAME function objects either way.
    monkeypatch.setattr(fmt, "COLOR", True)
    monkeypatch.setattr(fmt, "RST", "\033[0m")
    monkeypatch.setattr(fmt, "DIM", "\033[2m")
    return True


@pytest.fixture
def color_off(monkeypatch):
    monkeypatch.setattr(fmt, "COLOR", False)
    monkeypatch.setattr(fmt, "RST", "")
    monkeypatch.setattr(fmt, "DIM", "")
    return False


# --- _conf_rgb: the denoise ramp -----------------------------------------------------------------------

def test_conf_rgb_endpoints_are_the_denoise_palette():
    assert cli._conf_rgb(0.0) == cli._C_HOT      # most uncertain
    assert cli._conf_rgb(0.5) == cli._C_PALE     # the 0.5 threshold starts the cool half
    assert cli._conf_rgb(1.0) == cli._C_BLUE     # fully sure


def test_conf_rgb_clamps_and_coerces():
    assert cli._conf_rgb(-3.0) == cli._C_HOT
    assert cli._conf_rgb(9.0) == cli._C_BLUE
    assert cli._conf_rgb("garbage") == cli._conf_rgb(0.0)    # _num -> 0.0, never raises


def test_conf_rgb_low_is_warm_high_is_cool():
    lr, lg, lb = cli._conf_rgb(0.15)
    hr, hg, hb = cli._conf_rgb(0.95)
    assert lr > lb          # low confidence: warm (more red than blue)
    assert hb > hr          # high confidence: cool (more blue than red)


def test_conf_rgb_threshold_is_a_deliberate_step():
    under, over = cli._conf_rgb(0.49), cli._conf_rgb(0.51)
    assert under[0] > under[2]      # warm just below the hesitation line
    assert over[2] > over[0]        # cool just above it -- the same '?' boundary cmd_trace marks


# --- _paint: the no-op-when-off guard (the whole safety story) -----------------------------------------

def test_paint_is_a_strict_noop_when_off(color_off):
    assert cli._paint("hello", 0.2) == "hello"
    assert cli._paint("hello", 0.99) == "hello"


def test_paint_wraps_truecolor_when_on(color_on):
    out = cli._paint("hi", 0.9)
    assert out.startswith("\033[38;2;")
    assert out.endswith("\033[0m")
    assert _visible(out) == "hi"


def test_paint_never_escapes_empty_text(color_on):
    assert cli._paint("", 0.9) == ""


# --- _heatmap_lines: reconstruct the reply, painted, wrapped -------------------------------------------

def test_heatmap_reconstructs_reply_plain_when_off(color_off):
    pairs = [("The", 0.99), (" cat", 0.8), (" sat", 0.4)]
    lines = cli._heatmap_lines(pairs, width=80)
    assert "".join(lines) == "The cat sat"
    assert all("\033[" not in ln for ln in lines)


def test_heatmap_paints_each_token_when_on(color_on):
    lines = cli._heatmap_lines([("The", 0.99), (" cat", 0.2)], width=80)
    joined = "\n".join(lines)
    assert "\033[38;2;" in joined
    assert _visible(joined) == "The cat"       # visible text still reconstructs with escapes stripped


def test_heatmap_honors_embedded_newlines(color_off):
    lines = cli._heatmap_lines([("line one", 0.9), ("\n", 0.9), ("line two", 0.9)], width=80)
    assert lines == ["line one", "line two"]


def test_heatmap_wraps_to_width(color_off):
    pairs = [("word ", 0.9)] * 10               # 50 visible chars
    lines = cli._heatmap_lines(pairs, width=20)
    assert len(lines) > 1
    assert all(len(_visible(ln)) <= 20 for ln in lines)


def test_heatmap_wrap_counts_visible_only_when_on(color_on):
    # with escapes present, wrapping must still be driven by VISIBLE length, not raw length
    pairs = [("word ", 0.9)] * 10
    lines = cli._heatmap_lines(pairs, width=20)
    assert all(len(_visible(ln)) <= 20 for ln in lines)
    assert any("\033[38;2;" in ln for ln in lines)


def test_heatmap_empty_is_empty():
    assert cli._heatmap_lines([]) == []


# --- _paint_sparkline: identical to _sparkline when off -----------------------------------------------

def test_paint_sparkline_identical_when_off(color_off):
    moments = [{"index": 2, "confidence": 0.2}, {"index": 5, "confidence": 0.4}]
    assert cli._paint_sparkline(8, moments) == cli._sparkline(8, moments)


def test_paint_sparkline_paints_one_glyph_per_token_when_on(color_on):
    out = cli._paint_sparkline(4, [{"index": 1, "confidence": 0.2}])
    assert "\033[38;2;" in out
    assert len(_visible(out)) == 4


# --- the regression guard: the explain panel is byte-clean with color off -----------------------------

def test_format_explain_has_no_truecolor_when_off(color_off):
    expl = {"confidence": {"available": True, "n_tokens": 4, "threshold": 0.5, "summary": "1 hesitation",
                           "uncertain_moments": [{"index": 2, "token": " sat", "confidence": 0.3,
                                                  "alternatives": [{"piece": " ran", "prob": 0.4}]}]},
            "influences_active": {"cards": [], "dials": []},
            "concepts": {"available": False, "note": "needs the engine"}}
    out = cli.format_explain(expl)
    assert "\033[38;2;" not in out
    assert "1 hesitation" in out


def test_format_explain_paints_the_panel_when_on(color_on):
    expl = {"confidence": {"available": True, "n_tokens": 4, "threshold": 0.5, "summary": "1 hesitation",
                           "uncertain_moments": [{"index": 2, "token": " sat", "confidence": 0.3}]},
            "influences_active": {"cards": [], "dials": []},
            "concepts": {"available": False, "note": "needs the engine"}}
    out = cli.format_explain(expl)
    assert "\033[38;2;" in out                 # the sparkline + hesitation token get painted
    assert "1 hesitation" in _visible(out)     # content intact under the escapes


# --- _stream_token: the live `clozn run --heat` paint, unit-testable without an engine -----------------

def test_stream_token_heat_off_is_the_raw_piece(color_on):
    # even with color available, heat=False must be byte-identical to the plain stream
    assert cli._stream_token(" hello", 0.2, heat=False) == " hello"
    assert cli._stream_token(" hello", 0.99, heat=False) == " hello"


def test_stream_token_paints_when_heat_and_color(color_on):
    out = cli._stream_token(" name", 0.98, heat=True)
    assert out.startswith("\033[38;2;")
    assert _visible(out) == " name"


def test_stream_token_heat_but_no_color_is_plain(color_off):
    # --heat while piped (COLOR off) still emits clean text -- the _paint no-op carries the guarantee
    assert cli._stream_token(" name", 0.98, heat=True) == " name"


def test_stream_token_reconstructs_a_reply_under_heat(color_on):
    stream = [(" I", 0.45), (" am", 0.7), (" sure", 0.99)]
    written = "".join(cli._stream_token(p, c, heat=True) for p, c in stream)
    assert "\033[38;2;" in written
    assert _visible(written) == " I am sure"


def test_run_parser_has_heat_flag_defaulting_off():
    p = cli.build_parser()
    assert p.parse_args(["run", "qwen", "hello"]).heat is False          # default: unchanged, plain stream
    assert p.parse_args(["run", "qwen", "hello", "--heat"]).heat is True  # opt in to the live heatmap


def test_run_and_serve_parsers_accept_only_positive_context_windows():
    p = cli.build_parser()
    assert p.parse_args(["run", "qwen", "hello"]).ctx is None
    assert p.parse_args(["serve", "qwen"]).ctx is None
    assert p.parse_args(["run", "qwen", "hello", "--ctx", "2048"]).ctx == 2048
    assert p.parse_args(["serve", "qwen", "--ctx", "3072"]).ctx == 3072
    assert p.parse_args(["run", "qwen", "hello", "--batch", "512", "--ubatch", "128"]).batch == 512
    assert p.parse_args(["serve", "qwen", "--batch", "768", "--ubatch", "256"]).ubatch == 256
    with pytest.raises(SystemExit):
        p.parse_args(["run", "qwen", "hello", "--ctx", "0"])
    with pytest.raises(SystemExit):
        p.parse_args(["serve", "qwen", "--ctx", "not-a-number"])
    with pytest.raises(SystemExit):
        p.parse_args(["run", "qwen", "hello", "--batch", "0"])
