@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "PATH=%PATH%;C:\Program Files\CMake\bin;C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"
cd /d "%~dp0"
echo === configure (GPU ggml+serve, GGML_CUDA=ON) ===
cmake -S . -B build-gpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DCLOZN_BUILD_GGML=ON -DCLOZN_BUILD_SERVE=ON -DGGML_CUDA=ON
if errorlevel 1 (echo CONFIGURE_FAILED & exit /b 2)
echo === build ===
cmake --build build-gpu
echo === BUILD_DONE exit=%errorlevel% ===
