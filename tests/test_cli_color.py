"""Retained CLI formatting and parser boundary tests."""
from __future__ import annotations

import pytest

import clozn.cli.main as cli
from clozn.cli.formatting import _confbar


def test_download_progress_bar_is_bounded():
    assert _confbar(-1.0, width=4) == "░░░░"
    assert _confbar(0.5, width=4) == "██░░"
    assert _confbar(2.0, width=4) == "████"


def test_run_and_serve_parsers_accept_only_positive_context_windows():
    parser = cli.build_parser()
    assert parser.parse_args(["run", "qwen", "hello"]).ctx is None
    assert parser.parse_args(["serve", "qwen"]).ctx is None
    assert parser.parse_args(["run", "qwen", "hello", "--ctx", "2048"]).ctx == 2048
    assert parser.parse_args(["serve", "qwen", "--ctx", "3072"]).ctx == 3072
    assert parser.parse_args([
        "run", "qwen", "hello", "--batch", "512", "--ubatch", "128",
    ]).batch == 512
    assert parser.parse_args([
        "serve", "qwen", "--batch", "768", "--ubatch", "256",
    ]).ubatch == 256
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "qwen", "hello", "--ctx", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "qwen", "--ctx", "not-a-number"])
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "qwen", "hello", "--batch", "0"])
