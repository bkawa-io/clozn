# CMakeLists.txt — SAE-sparsify top-k kernel (Clozn ROADMAP 3.3)
# =============================================================================
#  The on-device primitive for SAE/transcoder inference at scale: a per-row top-k
#  over the FEATURE dimension (the confidence-select top-k repointed at features).
#  reference.py is the tested oracle (run test_sae_topk_reference.py with pytest); the CUDA
#  kernel is checked against it by validate.py. This build mirrors
#  ../confidence_select/CMakeLists.txt:
#    * with a CUDA compiler, builds sae_topk.cu into a static lib plus the
#      sae_test smoke exe and the sae_validate harness
#      (then: python validate.py <build>/sae_validate);
#    * without one, falls back to the host-only stub (#else of __CUDACC__) so the
#      C++ target still links on a CUDA-less CI box.
# =============================================================================

cmake_minimum_required(VERSION 3.18)  # 3.18+ for first-class CUDA language

project(clozn_sae_topk LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Detect CUDA but never hard-require it: this directory must configure on a
# CUDA-less CI box (where only the Python reference is tested).
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    message(STATUS "[sae_topk] CUDA found: building the GPU kernel.")
    # Ampere / Ada / Blackwell for ALL CUDA targets — set BEFORE enable_language() so it
    # becomes the default architecture (matches ../confidence_select/CMakeLists.txt). sm_120
    # is the RTX 5080; sm_70 was dropped in CUDA 13. Override with -DCMAKE_CUDA_ARCHITECTURES=native.
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "80;89;120")
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    add_library(sae_topk STATIC sae_topk.cu)
    set_target_properties(sae_topk PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # Validation harness: runs the kernel on file-fed pre-activations so validate.py can diff
    # it against the numpy reference (indices exact on sm_120; values within f32-vs-f64 eps).
    add_executable(sae_validate validate.cu)
    target_link_libraries(sae_validate PRIVATE sae_topk)
else()
    message(WARNING
        "[sae_topk] No CUDA compiler found. Building the host-only stub of sae_topk.cu "
        "(the __CUDACC__ #else path). The GPU kernel is unavailable until a toolchain "
        "exists; correctness lives in reference.py.")
    # Compile the .cu as plain C++ so the stub links (CUDA-only includes are guarded).
    add_library(sae_topk STATIC sae_topk.cu)
    set_source_files_properties(sae_topk.cu PROPERTIES LANGUAGE CXX)
endif()

target_include_directories(sae_topk PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Tiny C++ smoke test. Exercises sparse_code_bytes and links the kernel symbol;
# the real correctness check remains the pytest suite + validate.py.
add_executable(sae_test test_main.cpp)
target_link_libraries(sae_test PRIVATE sae_topk)

enable_testing()
add_test(NAME sae_smoke COMMAND sae_test)
