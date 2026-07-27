@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "PATH=%PATH%;C:\Program Files\CMake\bin;C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"
cd /d "%~dp0"
echo === configure (CLOZN_BUILD_CUDA=ON CLOZN_BUILD_SAE=ON) ===
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DCLOZN_BUILD_CUDA=ON -DCLOZN_BUILD_SAE=ON
if errorlevel 1 (echo CONFIGURE_FAILED & exit /b 2)
echo === build + run the sae parity tests (kernel vs CPU ref; encoder vs torch oracle) ===
cmake --build build-cuda --target test_sae_topk test_sae_encoder
if errorlevel 1 (echo BUILD_FAILED & exit /b 3)
ctest --test-dir build-cuda -R "sae_topk|sae_encoder" --output-on-failure
echo === DONE exit=%errorlevel% ===
