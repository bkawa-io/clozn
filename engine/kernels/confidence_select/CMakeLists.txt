# CMakeLists.txt — confidence-select kernel (Clozn DESIGN.md §4.3)
# =============================================================================
#  GREEDY PATH COMPILED + VALIDATED on RTX 5080 / CUDA 13.3 (sm_120).
#  reference.py is the tested oracle (run test_confidence_select_reference.py with pytest); the CUDA
#  kernel's deterministic paths are checked against it by validate.py. This build:
#    * with a CUDA compiler, builds confidence_select.cu into a static lib plus the
#      cs_test smoke exe and the cs_validate harness
#      (then: python validate.py <build>/cs_validate);
#    * without one, falls back to the host-only stub (#else of __CUDACC__) so the
#      C++ target still links on a CUDA-less CI box.
# =============================================================================

cmake_minimum_required(VERSION 3.18)  # 3.18+ for first-class CUDA language

project(clozn_confidence_select LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Detect CUDA but never hard-require it: this directory must configure on a
# CUDA-less CI box (where only the Python reference is tested).
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    message(STATUS "[confidence_select] CUDA found: building the GPU kernel (UNVERIFIED scaffold).")
    # Ampere / Ada / Blackwell for ALL CUDA targets — set BEFORE enable_language() so it
    # becomes the default architecture (CMake otherwise pre-seeds a default at enable
    # time). The lib, cs_test, and cs_validate each device-link, so each needs sm_120
    # code for the RTX 5080, else "no kernel image available for execution on the device"
    # at runtime. sm_70 was dropped in CUDA 13. Override with -DCMAKE_CUDA_ARCHITECTURES=native.
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "80;89;120")
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    add_library(confidence_select STATIC confidence_select.cu)
    set_target_properties(confidence_select PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # Validation harness: runs the kernel on file-fed logits so validate.py can diff
    # it against the numpy reference (the deterministic paths match exactly on sm_120).
    add_executable(cs_validate validate.cu)
    target_link_libraries(cs_validate PRIVATE confidence_select)

    # Microbenchmark: times the host-handoff the kernel replaces (full-logits D2H vs
    # on-device kernel + tiny D2H) at real model scales.
    add_executable(cs_bench bench.cu)
    target_link_libraries(cs_bench PRIVATE confidence_select)
else()
    message(WARNING
        "[confidence_select] No CUDA compiler found. Building the host-only stub "
        "of confidence_select.cu (the __CUDACC__ #else path). The GPU kernel is "
        "UNVERIFIED until a toolchain is available; correctness lives in reference.py.")
    # Compile the .cu as plain C++ so the stub links (the file guards CUDA-only
    # includes behind __CUDACC__).
    add_library(confidence_select STATIC confidence_select.cu)
    set_source_files_properties(confidence_select.cu PROPERTIES LANGUAGE CXX)
endif()

target_include_directories(confidence_select PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Tiny C++ smoke test. Exercises host_transfer_bytes and links the kernel symbol;
# the real correctness check remains the pytest suite.
add_executable(cs_test test_main.cpp)
target_link_libraries(cs_test PRIVATE confidence_select)

enable_testing()
add_test(NAME cs_smoke COMMAND cs_test)
