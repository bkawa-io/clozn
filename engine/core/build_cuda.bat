@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "PATH=%PATH%;C:\Program Files\CMake\bin;C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"
cd /d "%~dp0"
echo === configure (confidence-select kernel selector, CLOZN_BUILD_CUDA=ON) ===
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DCLOZN_BUILD_CUDA=ON
if errorlevel 1 (echo CONFIGURE_FAILED & exit /b 2)
echo === build + run the parity test (kernel vs CPU reference) ===
cmake --build build-cuda --target test_kernel_selector
if errorlevel 1 (echo BUILD_FAILED & exit /b 3)
ctest --test-dir build-cuda -R kernel_selector --output-on-failure
echo === DONE exit=%errorlevel% ===
