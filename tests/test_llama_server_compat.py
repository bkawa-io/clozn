from __future__ import annotations

import pytest

from clozn.cli.main import CloznError, build_parser
from clozn.cli.engine_process import _launch_args


def _serve(*args):
    return build_parser().parse_args(["serve", *args])


def test_llama_server_aliases_parse_into_one_serve_shape():
    args = _serve("-m", "model.gguf", "-c", "8192", "-ngl", "40",
                  "--host", "0.0.0.0", "-a", "local-model", "-np", "1")
    assert args.model is None
    assert args.model_flag == "model.gguf"
    assert args.ctx == 8192
    assert args.gpu_layers == 40
    assert args.host == "0.0.0.0"
    assert args.alias == "local-model"
    assert args.parallel == 1


def test_ctx_aliases_share_destination():
    assert _serve("model.gguf", "--ctx", "8").ctx == 8
    assert _serve("model.gguf", "-c", "8").ctx == 8
    assert _serve("model.gguf", "--ctx-size", "8").ctx == 8


@pytest.mark.parametrize("flag", ["-ngl", "--gpu-layers", "--n-gpu-layers"])
def test_gpu_layer_aliases_share_destination(flag):
    assert _serve("model.gguf", flag, "0").gpu_layers == 0
    assert _serve("model.gguf", flag, "40").gpu_layers == 40


def test_parallel_accepts_one_but_parser_keeps_invalid_value_visible():
    assert _serve("model.gguf", "--parallel", "1").parallel == 1
    assert _serve("model.gguf", "--parallel", "2").parallel == 2


def test_worker_launch_keeps_private_worker_loopback_and_honors_explicit_layers():
    argv = _launch_args("server", "model.gguf", 4321, {"gpu_layers": 40}, True)
    assert argv[:5] == ["server", "model.gguf", "--port", "4321", "--host"]
    assert argv[5] == "127.0.0.1"
    assert argv[argv.index("--gpu-layers") + 1] == "40"


def test_worker_launch_preserves_implicit_gpu_default():
    argv = _launch_args("server", "model.gguf", 4321, {}, True)
    assert argv[argv.index("--gpu-layers") + 1] == "99"


def test_worker_launch_explicit_zero_reaches_worker():
    argv = _launch_args("server", "model.gguf", 4321, {"gpu_layers": 0}, False)
    assert argv[argv.index("--gpu-layers") + 1] == "0"
