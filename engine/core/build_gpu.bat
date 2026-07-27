@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "PATH=%PATH%;C:\Program Files\CMake\bin;C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"
cd /d "%~dp0"
echo === configure (GPU ggml+serve+sae, GGML_CUDA=ON) ===
rem CMAKE_CUDA_ARCHITECTURES pinned to 120 (this box: RTX 5080) so the SAE targets match the
rem native arch ggml-cuda was already built for -- an unpinned reconfigure would flip ggml to the
rem multi-arch kernel default and rebuild the world.
cmake -S . -B build-gpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DCLOZN_BUILD_GGML=ON -DCLOZN_BUILD_SERVE=ON -DGGML_CUDA=ON -DCLOZN_BUILD_SAE=ON -DCMAKE_CUDA_ARCHITECTURES=120
if errorlevel 1 (echo CONFIGURE_FAILED & exit /b 2)
echo === build ===
cmake --build build-gpu
echo === BUILD_DONE exit=%errorlevel% ===
