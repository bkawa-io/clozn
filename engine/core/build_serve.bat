@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "PATH=%PATH%;C:\Program Files\CMake\bin;C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
cd /d "%~dp0"
echo === configure (CPU ggml+serve) ===
cmake -S . -B build-serve -G Ninja -DCMAKE_BUILD_TYPE=Release -DCLOZN_BUILD_GGML=ON -DCLOZN_BUILD_SERVE=ON -DGGML_CUDA=OFF
if errorlevel 1 (echo CONFIGURE_FAILED & exit /b 2)
echo === build ===
cmake --build build-serve --target clozn-server
echo === BUILD_DONE exit=%errorlevel% ===
