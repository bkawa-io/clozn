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

# clozn_runtime: the sampler + the event spine + the attention-mask builder. Backend-free —
# clozn/sample.hpp speaks only to the abstract ForwardResult/Candidate/Mask types, so no model
# backend lives here either. blocks.cpp survives only for attention_mask() (still called, always
# bidirectional, by the concept-probe calibration's one-shot sentence read -- see clozn/blocks.hpp).
# The ggml adapter (below) depends on this; never the reverse.
add_library(clozn_runtime STATIC
    src/sample.cpp
    src/events.cpp
    src/blocks.cpp
)
target_include_directories(clozn_runtime PUBLIC include)

enable_testing()

# GAP #1 (task #43): the white-box state WRITE seam (ModelAdapter::write_state) — the
# read -> edit -> write -> observe loop, proven against a writable test double (backend-free, CPU).
# Links clozn_runtime for the include dir (Mask/ModelAdapter/ForwardResult all live there now).
# The ggml adapter implements write_state against the live llama context in a later slice below.
add_executable(test_state_write tests/test_state_write.cpp)
target_link_libraries(test_state_write PRIVATE clozn_runtime)
add_test(NAME state_write COMMAND test_state_write)

# --- On-device SAE readout (ROADMAP 3.3 wired into the engine) ---
# clozn_sae (src/sae_encoder.cu + clozn/sae.hpp): the JumpReLU SAE encoder GEMV + the vendored
# sae_topk kernel, behind a CUDA-free pimpl header so clozn-server links it untouched by nvcc.
# Weights come from tools/export_sae_weights.py; parity vs the torch oracle is gated by
# test_sae_encoder against tools/dump_sae_vectors.py artifacts (SKIPs where the export is absent).
#   cmake -S core -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DCLOZN_BUILD_SAE=ON
option(CLOZN_BUILD_SAE "Build the on-device SAE readout (JumpReLU encoder + sae_topk, CUDA)" OFF)
# Kept only to gate test_sae_topk below (the CUDA confidence-select kernel + its CommitSelector
# wrapper this option used to also build were diffusion-only and were removed with the scheduler).
#   cmake -S core -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DCLOZN_BUILD_CUDA=ON
option(CLOZN_BUILD_CUDA "Build the sae_topk CUDA parity test" OFF)

if(CLOZN_BUILD_CUDA OR CLOZN_BUILD_SAE)
    # Ampere/Ada/Blackwell for all CUDA targets — set BEFORE enable_language (CMake otherwise
    # pre-seeds a default at enable time), matching kernels/confidence_select/CMakeLists.txt.
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "80;89;120")
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # SAE-sparsify top-k kernel (ROADMAP 3.3: the confidence-select top-k repointed at the FEATURE
    # dim), vendored from kernels/ — shared by the CUDA parity test AND the SAE readout library.
    # Validated on sm_120: selected indices exact, values within f32-vs-f64 eps (also proven
    # standalone by kernels/sae_topk/validate.py vs the numpy reference).
    set(SAE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/sae_topk)  # monorepo: engine/core -> kernels/
    add_library(clozn_sae_topk STATIC ${SAE_DIR}/sae_topk.cu)
    set_target_properties(clozn_sae_topk PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(clozn_sae_topk PUBLIC ${SAE_DIR})
endif()

if(CLOZN_BUILD_CUDA)
    # Self-contained C++ parity test gating the sae_topk kernel (built above, shared with
    # CLOZN_BUILD_SAE) against an in-process CPU reference.
    add_executable(test_sae_topk tests/test_sae_topk.cu)
    set_target_properties(test_sae_topk PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_sae_topk PRIVATE clozn_sae_topk)
    add_test(NAME sae_topk COMMAND test_sae_topk)
endif()

if(CLOZN_BUILD_SAE)
    find_package(CUDAToolkit REQUIRED)
    # The SAE readout library: JumpReLU encoder GEMV + epilogue in front of the sae_topk kernel.
    # CUDA_RESOLVE_DEVICE_SYMBOLS so plain-CXX consumers (clozn-server, test_sae_encoder) link it
    # without a device-link step of their own; cudart_static rides the public interface for the same
    # reason. See clozn/sae.hpp for the pipeline + numerics contract.
    add_library(clozn_sae STATIC src/sae_encoder.cu)
    set_target_properties(clozn_sae PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_include_directories(clozn_sae PUBLIC include)
    target_link_libraries(clozn_sae PUBLIC clozn_sae_topk CUDA::cudart_static)

    # Engine-vs-torch parity gate (SKIPs cleanly when the exported weights are absent, so CI boxes
    # without the 0.94 GB export stay green — same convention as test_ggml_state_write).
    add_executable(test_sae_encoder tests/test_sae_encoder.cpp)
    target_link_libraries(test_sae_encoder PRIVATE clozn_sae)
    add_test(NAME sae_encoder COMMAND test_sae_encoder)
endif()

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

    # GAP #1 (task #43): the white-box state-WRITE on a REAL model — read -> edit -> write -> observe ->
    # clear through llama, via the patch-free eval-callback activation patch. SKIPS without a GGUF arg.
    add_executable(test_ggml_state_write tests/test_ggml_state_write.cpp)
    target_link_libraries(test_ggml_state_write PRIVATE clozn_ggml)
    if(WIN32)
        # llama/ggml DLLs are emitted under build/bin.  The product launcher prepends that directory,
        # but a bare CTest invocation otherwise stalls behind Windows' missing-DLL error dialog before
        # this test can reach its intentional no-model skip.
        add_test(
            NAME ggml_state_write
            COMMAND ${CMAKE_COMMAND} -E env
                "PATH=$<TARGET_FILE_DIR:llama>;$ENV{PATH}"
                $<TARGET_FILE:test_ggml_state_write>
        )
    else()
        add_test(NAME ggml_state_write COMMAND test_ggml_state_write)
    endif()

    # ffn_out-<il> hook (feed-forward/MLP contribution, additive analogue of kqv_out) + the
    # 2026-07-28 head_write/ffn_write honesty fix -- on a REAL model. SKIPS without a GGUF arg,
    # same convention as test_ggml_state_write above.
    add_executable(test_ffn_hook tests/test_ffn_hook.cpp)
    target_link_libraries(test_ffn_hook PRIVATE clozn_ggml)
    if(WIN32)
        add_test(
            NAME ffn_hook
            COMMAND ${CMAKE_COMMAND} -E env
                "PATH=$<TARGET_FILE_DIR:llama>;$ENV{PATH}"
                $<TARGET_FILE:test_ffn_hook>
        )
    else()
        add_test(NAME ffn_hook COMMAND test_ffn_hook)
    endif()

    # clozn-bench — the §9 quant-calibration sweep: quantize f16 -> Q8/Q5/Q4 in-process and
    # report size + token-agreement-vs-f16 + tok/s (the honest quality column for the quant axis).
    add_executable(clozn-bench tools/clozn_bench.cpp)
    target_link_libraries(clozn-bench PRIVATE clozn_ggml)

    # clozn-probe-sweep — sweep all layers to find the best tap for concept probes.
    add_executable(clozn-probe-sweep tools/clozn_probe_sweep.cpp)
    target_link_libraries(clozn-probe-sweep PRIVATE clozn_ggml)

    # clozn-ar — autoregressive white-box CLI: read tap + logit-lens per token on any AR GGUF.
    # The daily-driver shape behind the tests (clozn_cli.cpp / the "clozn" tool and clozn-arbench's
    # AR-vs-diffusion throughput comparison were diffusion-era tools and were removed with it).
    add_executable(clozn-ar tools/clozn_ar.cpp)
    target_link_libraries(clozn-ar PRIVATE clozn_ggml)

    # Resolving experiment for the zero-copy §4.3 path: does llama_get_logits_tensor (the CLOZN
    # PATCH accessor) return a DEVICE-resident tensor that byte-matches the host logits? Needs a
    # CUDA-capable llama (configure with -DGGML_CUDA=ON) and runs with all layers offloaded.
    add_executable(test_device_logits tests/test_device_logits.cpp)
    target_link_libraries(test_device_logits PRIVATE llama)
endif()

# --- L2 serving layer (HTTP, needs CLOZN_BUILD_GGML) ---
# An HTTP server over the runtime: /v1/completions (+ SSE streaming of the §5.1 events), using the
# cpp-httplib + nlohmann/json that llama.cpp already vendors (no new deps). /v1/infill and the
# diffusion-only /v1/revise + /v1/board routes are gone with the diffusion generator; /v1/completions
# always answers via the autoregressive generate_ar loop now.
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
        serve/routes_jlens.cpp
        serve/routes_whitebox.cpp
        serve/routes_state.cpp
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
    target_link_libraries(clozn-server PRIVATE clozn_ggml llama-common)
    if(CLOZN_BUILD_SAE)
        # --sae <dir>: the on-device SAE feature readout rides the server (clozn/sae.hpp is a
        # CUDA-free pimpl header, so this stays a plain-CXX target; clozn_sae brings cudart).
        target_link_libraries(clozn-server PRIVATE clozn_sae)
        target_compile_definitions(clozn-server PRIVATE CLOZN_SAE)
    endif()
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
