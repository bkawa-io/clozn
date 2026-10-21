# core/ — the C++/ggml runtime (Phase 3). See README.md for the full port plan.
#
# The diffusion generator + scheduler (clozn_scheduler: policies/blocks/cache; the generate()/
# infill()/denoise() pass loop; the CommitSelector kernel seam) was removed with the diffusion
# program (THE_CUT, notes/THE_CUT.md) — Dream/LLaDA support never left the viz stage and the C++
# split confirmed it: generate_ar.cpp (the autoregressive white-box loop) already had zero runtime
# dependency on any of it. What's left is AR-only: the sampler + event spine (clozn_runtime) and the
# ggml adapter (clozn_ggml) that drives generate_ar.
cmake_minimum_required(VERSION 3.18)
project(clozn_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build identity is part of the install qualification contract. The Python package remains the release
# version source of truth; release automation may set a unique build ID, but it may not independently
# override the engine version or llama.cpp source revision.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/../../clozn/__init__.py" _clozn_version_line
     REGEX "^__version__[ \t]*=")
string(REGEX MATCH "\"([^\"]+)\"" _clozn_version_match "${_clozn_version_line}")
set(CLOZN_ENGINE_VERSION "${CMAKE_MATCH_1}")
if(NOT CLOZN_ENGINE_VERSION)
    message(FATAL_ERROR "could not derive CLOZN_ENGINE_VERSION from clozn/__init__.py")
endif()

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/bootstrap_llama.py" _clozn_llama_commit_line
     REGEX "^COMMIT[ \t]*=")
string(REGEX MATCH "\"([0-9a-f]+)\"" _clozn_llama_commit_match "${_clozn_llama_commit_line}")
set(CLOZN_LLAMA_CPP_COMMIT "${CMAKE_MATCH_1}")
string(LENGTH "${CLOZN_LLAMA_CPP_COMMIT}" _clozn_llama_commit_length)
if(NOT _clozn_llama_commit_length EQUAL 40)
    message(FATAL_ERROR "could not derive the pinned llama.cpp commit from bootstrap_llama.py")
endif()

set(CLOZN_ENGINE_BUILD_ID "development" CACHE STRING
    "Immutable release build identifier embedded in clozn-server")
if(NOT CLOZN_ENGINE_BUILD_ID MATCHES "^[A-Za-z0-9._+-]+$")
    message(FATAL_ERROR
        "CLOZN_ENGINE_BUILD_ID must contain only letters, digits, dot, underscore, plus, or hyphen")
endif()

# clozn_runtime: the sampler, event spine, and shared attention-mask type/builder. Backend-free;
# the ggml adapter below depends on this library, never the reverse.
add_library(clozn_runtime STATIC
    src/sample.cpp
    src/events.cpp
    src/blocks.cpp
)
target_include_directories(clozn_runtime PUBLIC include)

# Vendored, public-domain, dependency-free SHA-256. Exact-reference sessions use it to bind
# prompts, recorded references, and generation contracts to stable digests.
add_library(clozn_sha256 STATIC third_party/sha256/sha256.c)
set_source_files_properties(third_party/sha256/sha256.c PROPERTIES LANGUAGE CXX)
target_include_directories(clozn_sha256 PUBLIC third_party/sha256)

enable_testing()

# Request-wide reference-match radix planning is model-free.  The fake traversal callbacks in this
# test assert seq-0 depth restoration, duplicate-terminal collapse, and the no-remerge invariant
# without starting llama.cpp or requiring a GGUF.
add_executable(test_reference_match_plan tests/test_reference_match_plan.cpp)
target_include_directories(test_reference_match_plan PRIVATE include)
add_test(NAME reference_match_plan COMMAND test_reference_match_plan)

# Backend target-logprob reduction has a llama.cpp-specific decode constraint: no sequence may
# request two output rows in one decode. Keep the prefix-terminal chunk partition model-free.
add_executable(test_target_logprob_reduction tests/test_target_logprob_reduction.cpp)
target_include_directories(test_target_logprob_reduction PRIVATE include)
add_test(NAME target_logprob_reduction COMMAND test_target_logprob_reduction)

# Ordinary chat prompt-cache planning is exact-token-only and model-free. The full-prefix case must
# retain one bridge token to recover the next-token logits row.
add_executable(test_prompt_prefix_cache tests/test_prompt_prefix_cache.cpp)
target_include_directories(test_prompt_prefix_cache PRIVATE include)
add_test(NAME prompt_prefix_cache COMMAND test_prompt_prefix_cache)

# --- L0 ggml adapter (Phase 3, slice 3) ---
# Optional + heavy: this builds the vendored llama.cpp. Default OFF so the runtime-only
# build (and CI) stay fast and dependency-free — the runtime library above never links a
# model backend (the seam, enforced at the build level).
#
#   cmake -S core -B build-ggml -DCLOZN_BUILD_GGML=ON -DGGML_CUDA=OFF
option(CLOZN_BUILD_GGML "Build the ggml L0 adapter (builds llama.cpp)" OFF)
option(CLOZN_BUILD_SERVE "Build the HTTP serving layer (requires CLOZN_BUILD_GGML)" OFF)
if(CLOZN_BUILD_GGML)
    # The managed worker applies the GGUF's full Jinja chat template. llama.cpp keeps that
    # implementation in llama-common rather than the core llama target, so make the common
    # library part of server builds (but keep it out of backend-only/test builds).
    if(CLOZN_BUILD_SERVE)
        set(LLAMA_BUILD_COMMON ON CACHE BOOL "llama: build common utils library" FORCE)
    endif()
    add_subdirectory(third_party/llama.cpp)

    # GgmlAdapter: the ONE place a model backend lives (DESIGN invariant 1). It links
    # clozn_runtime (the sampler + event spine) and llama; the runtime library never links
    # llama, so the backend-free build stays backend-free.
    # generate_ar.cpp (the autoregressive white-box loop, the only generator left) lives here, not
    # in clozn_runtime: it drives GgmlAdapter::ar_forward directly (incremental causal decode), so
    # it's backend-coupled by nature.
    add_library(clozn_ggml STATIC src/model_ggml.cpp src/generate_ar.cpp)
    target_link_libraries(clozn_ggml PUBLIC clozn_runtime llama)
endif()

# --- L2 serving layer (HTTP, needs CLOZN_BUILD_GGML) ---
# An HTTP server over the runtime: /v1/completions (+ SSE streaming of the §5.1 events), using the
# cpp-httplib + nlohmann/json that llama.cpp already vendors (no new deps). The worker is private to
# the v1 gateway; research and diffusion product routes are deliberately not linked into it.
#   cmake -S core -B build -G Ninja -DCLOZN_BUILD_GGML=ON -DCLOZN_BUILD_SERVE=ON
if(CLOZN_BUILD_SERVE AND CLOZN_BUILD_GGML)
    set(_vendor ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor)
    # cpp-httplib is the split build: compile its httplib.cpp here for the out-of-line definitions
    # (silence its 3rd-party warnings; it needs /bigobj on MSVC).
    # Phase 12.4: the serve monolith split into server_main + route-family TUs that read shared state
    # through ServerContext (serve/server_context.hpp) + the shared helpers/structs (serve/server_shared.hpp).
    add_executable(clozn-server
        serve/server_main.cpp
        serve/chat_template_renderer.cpp
        serve/routes_whitebox.cpp
        serve/routes_reference_match.cpp
        ${_vendor}/cpp-httplib/httplib.cpp)
    if(GGML_CUDA)
        set(_clozn_engine_backend "cuda")
    elseif(GGML_METAL)
        set(_clozn_engine_backend "metal")
    else()
        set(_clozn_engine_backend "cpu")
    endif()
    target_compile_definitions(clozn-server PRIVATE
        CLOZN_ENGINE_VERSION="${CLOZN_ENGINE_VERSION}"
        CLOZN_ENGINE_BUILD_ID="${CLOZN_ENGINE_BUILD_ID}"
        CLOZN_ENGINE_BACKEND="${_clozn_engine_backend}"
        CLOZN_LLAMA_CPP_COMMIT="${CLOZN_LLAMA_CPP_COMMIT}")
    target_include_directories(clozn-server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/serve ${_vendor}/cpp-httplib ${_vendor})
    # Stable digests for exact-reference prompts, references, and generation contracts.
    target_link_libraries(clozn-server PRIVATE clozn_ggml llama-common clozn_sha256)
    find_package(Threads REQUIRED)
    target_link_libraries(clozn-server PRIVATE Threads::Threads)
    if(WIN32)
        target_link_libraries(clozn-server PRIVATE ws2_32)
    endif()
    if(MSVC)
        # /EHsc: enable C++ unwind semantics so a thrown exception (e.g. nlohmann json::value() on a
        # wrong-typed field in a malformed request body) UNWINDS into httplib's handler-exception
        # catch (-> 500) instead of crashing the process. ggml's global flags leave it off (the
        # C4530 warning), which turned a bad request into a server-killing DoS. /bigobj: httplib needs it.
        target_compile_options(clozn-server PRIVATE /bigobj /EHsc)
        set_source_files_properties(${_vendor}/cpp-httplib/httplib.cpp PROPERTIES COMPILE_FLAGS /w)
    endif()

    # Private Phase 2.8 native chat-I/O seam: model-free template preparation + PEG parsing.
    # This does not register a public HTTP route or qualify any model/template pair.
    add_executable(test_chat_template_renderer
        tests/test_chat_template_renderer.cpp
        serve/chat_template_renderer.cpp)
    target_include_directories(test_chat_template_renderer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/serve)
    target_link_libraries(test_chat_template_renderer PRIVATE llama-common)
    target_compile_definitions(test_chat_template_renderer PRIVATE
        CLOZN_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    if(MSVC)
        target_compile_options(test_chat_template_renderer PRIVATE /EHsc)
    endif()
    add_test(NAME chat_template_renderer COMMAND test_chat_template_renderer)
    set_tests_properties(chat_template_renderer PROPERTIES
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()
