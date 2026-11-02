"""Build and install Clozn from this checkout into its own virtual environment."""
from __future__ import annotations

import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if sys.version_info < (3, 11):
        raise SystemExit("Use Python 3.11 or newer to run this installer.")
    if sys.platform not in ("darwin", "linux"):
        raise SystemExit("This source installer supports macOS and Linux.")
    required = ["git", "cmake", "ninja"]
    if sys.platform == "linux":
        required.append("c++")
    for program in required:
        if not shutil.which(program):
            raise SystemExit(f"Install {program} and rerun this script.")
    metal = sys.platform == "darwin" and platform.machine() == "arm64"
    build = ROOT / "engine/core" / ("build-gpu" if metal else "build-serve")
    env = ROOT / ".venv-clozn"
    subprocess.run([sys.executable, "engine/core/third_party/bootstrap_llama.py"], cwd=ROOT, check=True)
    subprocess.run(["cmake", "-S", str(ROOT / "engine/core"), "-B", str(build), "-G", "Ninja",
                    "-DCMAKE_BUILD_TYPE=Release", "-DCLOZN_BUILD_GGML=ON", "-DCLOZN_BUILD_SERVE=ON",
                    f"-DGGML_METAL={'ON' if metal else 'OFF'}", "-DGGML_CUDA=OFF"], check=True)
    subprocess.run(["cmake", "--build", str(build), "--target", "clozn-server", "--parallel",
                    os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "4")], check=True)
    subprocess.run([sys.executable, "-m", "venv", str(env)], check=True)
    subprocess.run([str(env / "bin/python"), "-m", "pip", "install", "-e", str(ROOT)], check=True)
    print(
        "\nClozn is installed in .venv-clozn.\n\n"
        "Download the recommended model:\n"
        "  .venv-clozn/bin/clozn pull qwen3.5\n\n"
        "Start Clozn:\n"
        "  .venv-clozn/bin/clozn serve qwen3.5 --port 8080\n\n"
        "Then open http://127.0.0.1:8080/app/\n"
        "Use qwen-0.5b instead for a smaller installation smoke test."
    )


if __name__ == "__main__":
    main()
