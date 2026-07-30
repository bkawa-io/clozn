"""Dependency-free contract tests for the post-beta runtime boundary."""
from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

from clozn.cli import engine_process, runtime_process, worker_handle
import clozn.settings as clozn_settings

from clozn.runs import store
from clozn.server import app
from clozn.server import generation_gateway
from clozn.server import http_policy
from clozn.server import sse
from clozn.server.request_gate import RequestGate
from clozn.server.routes import health


class FakeProcess:
    _next_pid = 2000

    def __init__(self, code=None):
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.code = code
        self.terminated = False

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated = True
        self.code = -15

    def kill(self):
        self.terminated = True
        self.code = -9

    def wait(self, timeout=None):
        return self.code


class SequencedProcess(FakeProcess):
    def __init__(self, codes):
        super().__init__()
        self.codes = list(codes)

    def poll(self):
        if len(self.codes) > 1:
            return self.codes.pop(0)
        return self.codes[0]


class FakeResponse:
    def __init__(self, lines=(), payload=b"{}", status=200):
        self.lines = list(lines)
        self.payload = payload
        self.status = status
        self.closed = False

    def __iter__(self):
        return iter(self.lines)

    def read(self):
        return self.payload

    def close(self):
        self.closed = True


class CaptureHandler:
    def __init__(self):
        self.code = None
        self.headers = {}
        self.sent_headers = []
        self.wfile = io.BytesIO()
        self.json = None

    def send_response(self, code):
        self.code = code

    def send_header(self, key, value):
        self.sent_headers.append((key, value))

    def end_headers(self):
        pass

    def _json(self, code, value, extra_headers=None):
        self.code, self.json = code, value

    def _send(self, code, body, ctype, extra_headers=None):
        self.code = code
        self.wfile.write(body if isinstance(body, bytes) else body.encode("utf-8"))

    def _log_run(self, *_args, **_kwargs):
        return "run_test"


def raw_gateway_request(method: str, *, path="/jlens", body=b"", headers=None):
    """Drive the real gateway handler without opening a socket."""
    handler_type = app.make_handler()
    handler = object.__new__(handler_type)
    handler.path = path
    handler.rfile = io.BytesIO(body)
    handler.wfile = io.BytesIO()
    handler.headers = {
        "Content-Length": str(len(body)),
        "User-Agent": "unittest",
        **(headers or {}),
    }
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.command = method
    handler.close_connection = False
    getattr(handler, f"do_{method}")()
    head, _, payload = handler.wfile.getvalue().partition(b"\r\n\r\n")
    return head.decode("latin-1"), payload, handler


class RuntimeBoundaryTests(unittest.TestCase):
    def test_worker_mode_is_explicit_not_inferred_from_vocabulary_tokens(self):
        ar = engine_process._launch_args("worker", "gemma.gguf", 9000, {"chat": True}, False)
        diffusion = engine_process._launch_args(
            "worker", "llada.gguf", 9000, {"chat": True, "mask": 126336}, False
        )
        self.assertNotIn("--diffusion", ar)
        self.assertIn("--diffusion", diffusion)
        self.assertIn("--mask-token", diffusion)

    def test_context_override_reaches_the_private_worker(self):
        args = engine_process._launch_args(
            "worker", "qwen35.gguf", 9000, {"chat": True, "ctx": 2048}, True
        )
        self.assertIn("--gpu-layers", args)
        self.assertEqual(args[args.index("--ctx") + 1], "2048")

    def test_explicit_context_does_not_reuse_a_mismatched_warm_runtime(self):
        registry = {"8080": {"model": "qwen35.gguf", "gpu": True, "mode": "autoregressive"}}
        health = {"status": "ok", "mode": "autoregressive", "worker": {"n_ctx": 4096}}
        with mock.patch.object(engine_process, "_reg_read", return_value=registry), \
             mock.patch.object(runtime_process, "gateway_health", return_value=health):
            self.assertIsNone(engine_process._find_warm("qwen35.gguf", 2048))
            self.assertEqual(engine_process._find_warm("qwen35.gguf", 4096),
                             (8080, True, "autoregressive"))

    def test_force_cpu_refuses_a_gpu_only_build(self):
        with tempfile.TemporaryDirectory(prefix="clozn-engine-select-") as root:
            gpu_root = os.path.join(root, "build-gpu")
            os.makedirs(gpu_root)
            open(os.path.join(gpu_root, "clozn-server.exe"), "wb").close()
            with mock.patch.object(engine_process, "ENGINE_CORE", root):
                with self.assertRaisesRegex(Exception, "no CPU engine build"):
                    engine_process.find_engine(prefer_gpu=False)

    def test_force_cpu_selects_the_cpu_build(self):
        with tempfile.TemporaryDirectory(prefix="clozn-engine-select-") as root:
            for build in ("build-gpu", "build-serve"):
                build_root = os.path.join(root, build)
                os.makedirs(build_root)
                open(os.path.join(build_root, "clozn-server.exe"), "wb").close()
            with mock.patch.object(engine_process, "ENGINE_CORE", root):
                executable, _dlls, gpu = engine_process.find_engine(prefer_gpu=False)
            self.assertFalse(gpu)
            self.assertIn("build-serve", executable)

    # ---- task #103: clozn-server.exe's own llama.dll/ggml-*.dll dir must land on a spawned child's PATH,
    # derived from the exe's own location -- never a hardcoded absolute path, never a mutation of the
    # parent's os.environ. See engine_process._dll_dirs_for / _env_with_dlls.

    def test_dll_dirs_for_derives_the_sibling_bin_dir_from_the_exes_own_path(self):
        with tempfile.TemporaryDirectory(prefix="clozn-dll-dirs-") as root:
            build_root = os.path.join(root, "build-gpu")
            bin_dir = os.path.join(build_root, "bin")
            os.makedirs(bin_dir)
            exe = os.path.join(build_root, "clozn-server.exe")
            open(exe, "wb").close()
            open(os.path.join(bin_dir, "llama.dll"), "wb").close()
            open(os.path.join(bin_dir, "ggml.dll"), "wb").close()

            dirs = engine_process._dll_dirs_for(exe)

        self.assertIn(build_root, dirs)
        self.assertIn(bin_dir, dirs)
        # No hardcoded absolute path leaked in -- every returned dir sits under this exe's own temp root.
        self.assertTrue(all(d == build_root or d.startswith(build_root) for d in dirs))

    def test_dll_dirs_for_ignores_a_bin_dir_with_no_engine_dll_in_it(self):
        """A directory NAMED `bin` that happens to exist but holds none of the engine's own DLLs must not
        be trusted just because the name matches -- this is the "check where they actually are, don't
        hardcode" contract _dll_dirs_for exists to enforce."""
        with tempfile.TemporaryDirectory(prefix="clozn-dll-dirs-") as root:
            build_root = os.path.join(root, "build-gpu")
            empty_bin = os.path.join(build_root, "bin")
            os.makedirs(empty_bin)
            exe = os.path.join(build_root, "clozn-server.exe")
            open(exe, "wb").close()
            open(os.path.join(empty_bin, "unrelated.txt"), "wb").close()

            dirs = engine_process._dll_dirs_for(exe)

        self.assertEqual(dirs, [build_root])   # only the exe's own dir -- the empty `bin` never qualifies

    def test_env_with_dlls_prepends_the_dll_dir_without_mutating_os_environ(self):
        before = dict(os.environ)
        fake_dir = os.path.join("Z:", "not-a-real-path", "build-gpu", "bin")

        env = engine_process._env_with_dlls([fake_dir], gpu=False)

        self.assertTrue(env["PATH"].startswith(fake_dir + os.pathsep))
        self.assertEqual(dict(os.environ), before, "must build a child env dict, never mutate os.environ")

    @unittest.skipUnless(os.name == "nt", "STATUS_DLL_NOT_FOUND / PATH-based DLL search is Windows-specific")
    def test_scrubbed_path_reproduces_dll_not_found_and_the_fix_resolves_it(self):
        """Live probe for task #103, run with NO model argument so clozn-server.exe prints its usage line
        and exits immediately (argc<2 in server_main.cpp) -- never loads a model, never touches the GPU.
        Skips cleanly when no engine build is present, mirroring test_engine_ctx_overflow.py's own
        find_engine-missing skip.
        """
        try:
            exe, dll_dirs, gpu = engine_process.find_engine(prefer_gpu=True)
        except Exception as exc:
            self.skipTest(f"no engine build available: {exc}")

        system_root = os.environ.get("SystemRoot", r"C:\Windows")
        scrubbed_path = os.pathsep.join([os.path.join(system_root, "System32"), system_root])

        def run(env, timeout=15):
            proc = subprocess.Popen([exe], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                out, err = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                self.fail("clozn-server.exe (no model arg) did not exit on its own -- killed the probe")
            return proc.returncode, out, err

        # Repro: a fresh-shell-like PATH with no build-gpu/bin (or CUDA toolkit) on it hits
        # STATUS_DLL_NOT_FOUND (0xC0000135). subprocess reports this as the raw unsigned NTSTATUS on some
        # Python builds and as its signed 32-bit twin on others -- mask to 32 bits either way.
        STATUS_DLL_NOT_FOUND = 0xC0000135
        bare_env = dict(os.environ)
        bare_env["PATH"] = scrubbed_path
        code, _out, _err = run(bare_env)
        self.assertEqual(code & 0xFFFFFFFF, STATUS_DLL_NOT_FOUND,
                         f"expected STATUS_DLL_NOT_FOUND (0xC0000135), got {code!r} -- "
                         "the exe's own DLL layout may have changed")

        # Fix: the SAME scrubbed PATH as the process's own os.environ, then _env_with_dlls (the actual
        # function spawn_engine calls) builds the child env from that -- proving the fix works even when
        # the parent shell's PATH is the scrubbed one, not just when the dev machine's real PATH leaks in.
        old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = scrubbed_path
        try:
            fixed_env = engine_process._env_with_dlls(dll_dirs, gpu)
        finally:
            os.environ["PATH"] = old_path
        code, out, err = run(fixed_env)
        self.assertEqual(code, 1, f"expected the no-model-arg usage exit (1), got {code!r}")
        self.assertIn(b"usage:", (out + err).lower())

    def test_runtime_config_copies_and_freezes_flags(self):
        flags = {"mask": 7}
        config = runtime_process.RuntimeConfig(model="m.gguf", public_port=8080, flags=flags)
        flags["mask"] = 8
        self.assertEqual(config.flags["mask"], 7)
        with self.assertRaises(TypeError):
            config.flags["mask"] = 9

    def test_spawn_runtime_starts_private_worker_before_gateway(self):
        originals = (runtime_process.spawn_engine, runtime_process.subprocess.Popen,
                     runtime_process.gateway_health, runtime_process.port_is_open)
        calls = []
        worker = FakeProcess()
        gateway = FakeProcess()

        def fake_spawn(model, port, flags, **kwargs):
            calls.append(("worker", port))
            return worker, {"status": "ok", "mode": "autoregressive"}, True

        def fake_popen(command, **kwargs):
            calls.append(("gateway", command, kwargs))
            return gateway

        runtime_process.spawn_engine = fake_spawn
        runtime_process.subprocess.Popen = fake_popen
        runtime_process.gateway_health = lambda port: {"status": "ok"}
        runtime_process.port_is_open = lambda port: False
        try:
            stack = runtime_process.spawn_runtime(runtime_process.RuntimeConfig(
                model="m.gguf", public_port=8123, worker_port=8456
            ))
        finally:
            (runtime_process.spawn_engine, runtime_process.subprocess.Popen,
             runtime_process.gateway_health, runtime_process.port_is_open) = originals

        self.assertEqual([call[0] for call in calls], ["worker", "gateway"])
        gateway_call = calls[1]
        self.assertEqual(gateway_call[2]["env"]["CLOZN_ENGINE_PORT"], "8456")
        self.assertEqual(gateway_call[2]["env"]["CLOZN_RUNTIME_KIND"], "product")
        self.assertNotIn("--substrate", gateway_call[1])
        self.assertEqual(stack.public_port, 8123)
        self.assertEqual(stack.worker_port, 8456)

    def test_interrupted_worker_boot_terminates_the_child(self):
        originals = (engine_process.find_engine, engine_process.subprocess.Popen, engine_process._health)
        worker = FakeProcess()
        engine_process.find_engine = lambda prefer_gpu=True: ("fake-engine", [], False)
        engine_process.subprocess.Popen = lambda *args, **kwargs: worker
        engine_process._health = lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt())
        try:
            with self.assertRaises(KeyboardInterrupt):
                engine_process.spawn_engine("m.gguf", 9001, {}, boot_timeout=1)
        finally:
            (engine_process.find_engine, engine_process.subprocess.Popen,
             engine_process._health) = originals
        self.assertTrue(worker.terminated)

    def test_interrupted_gateway_boot_terminates_the_worker(self):
        originals = (runtime_process.spawn_engine, runtime_process.subprocess.Popen,
                     runtime_process.port_is_open)
        worker = FakeProcess()
        runtime_process.spawn_engine = lambda *args, **kwargs: (
            worker, {"status": "ok", "mode": "autoregressive"}, False
        )
        runtime_process.subprocess.Popen = lambda *args, **kwargs: (
            (_ for _ in ()).throw(KeyboardInterrupt())
        )
        runtime_process.port_is_open = lambda port: False
        try:
            with self.assertRaises(KeyboardInterrupt):
                runtime_process.spawn_runtime(runtime_process.RuntimeConfig(
                    model="m.gguf", public_port=8123, worker_port=8456
                ))
        finally:
            (runtime_process.spawn_engine, runtime_process.subprocess.Popen,
             runtime_process.port_is_open) = originals
        self.assertTrue(worker.terminated)

    def test_readiness_requires_the_one_worker(self):
        class Worker:
            def health(self):
                return {"status": "ok", "model": "m.gguf", "mode": "autoregressive"}

        old_engine, old_sub = app.ENGINE, app.SUB
        handler = CaptureHandler()
        try:
            app.ENGINE, app.SUB = Worker(), object()
            self.assertTrue(health.try_get(handler, "/readyz"))
        finally:
            app.ENGINE, app.SUB = old_engine, old_sub
        self.assertEqual(handler.code, 200)
        self.assertEqual(handler.json["status"], "ok")
        self.assertEqual(handler.json["active"], "engine")

    def test_supervisor_restarts_an_unexpected_worker_exit(self):
        dead_worker = FakeProcess(code=1)
        replacement = FakeProcess()
        gateway = SequencedProcess([None, 0])
        stack = runtime_process.RuntimeStack(
            config=runtime_process.RuntimeConfig(model="m.gguf", public_port=8080, worker_port=9000),
            worker_port=9000,
            worker=dead_worker,
            gateway=gateway,
            worker_health={"status": "ok", "mode": "autoregressive"},
            gpu=False,
        )
        original = runtime_process.spawn_engine
        runtime_process.spawn_engine = lambda *a, **k: (
            replacement, {"status": "ok", "mode": "autoregressive"}, True
        )
        restarts = []
        try:
            code = stack.wait(on_worker_restart=lambda current: restarts.append(current.worker.pid),
                              poll_interval=0)
        finally:
            runtime_process.spawn_engine = original
        self.assertEqual(code, 0)
        self.assertIs(stack.worker, replacement)
        self.assertEqual(restarts, [replacement.pid])

    def test_worker_handle_owns_start_restart_registry_and_stop(self):
        processes = [FakeProcess(), FakeProcess()]
        calls = []

        def spawn(model, port, flags, **kwargs):
            calls.append((model, port, dict(flags), kwargs))
            return processes[len(calls) - 1], {
                "status": "ok", "mode": "autoregressive", "generation": len(calls),
            }, len(calls) == 2

        handle = worker_handle.WorkerHandle.start(
            model="m.gguf",
            port=9000,
            flags={"ctx": 4096},
            prefer_gpu=True,
            boot_timeout=17,
            restart_limit=3,
            restart_window=60,
            spawn=spawn,
        )
        self.assertIs(handle.process, processes[0])
        self.assertEqual(handle.registry_fields(), {
            "worker_pid": processes[0].pid, "worker_port": 9000,
        })

        handle.restart()
        self.assertIs(handle.process, processes[1])
        self.assertEqual(handle.health["generation"], 2)
        self.assertTrue(handle.gpu)
        self.assertEqual(handle.registry_fields()["worker_pid"], processes[1].pid)
        self.assertEqual(
            [call[:3] for call in calls],
            [("m.gguf", 9000, {"ctx": 4096}), ("m.gguf", 9000, {"ctx": 4096})],
        )
        self.assertTrue(all(call[3]["boot_timeout"] == 17 for call in calls))

        handle.stop()
        self.assertTrue(handle.stopping)
        self.assertTrue(processes[1].terminated)

    def test_worker_handle_restart_limit_uses_the_existing_sliding_window(self):
        processes = [FakeProcess(), FakeProcess(), FakeProcess()]
        calls = []

        def spawn(*_args, **_kwargs):
            process = processes[len(calls)]
            calls.append(process)
            return process, {"status": "ok"}, False

        handle = worker_handle.WorkerHandle.start(
            model="m.gguf",
            port=9000,
            flags={},
            prefer_gpu=False,
            boot_timeout=10,
            restart_limit=1,
            restart_window=10,
            spawn=spawn,
        )
        now = [100.0]
        handle.clock = lambda: now[0]
        handle.restart()
        with self.assertRaisesRegex(
            worker_handle.WorkerRestartLimitError,
            "model worker exited 1 times within 10s",
        ):
            handle.restart()
        self.assertEqual(len(calls), 2)

        now[0] = 111.0
        handle.restart()
        self.assertEqual(len(calls), 3)
        self.assertEqual(handle.restart_times, [111.0])

    def test_product_substrate_switch_is_gone(self):
        handler = CaptureHandler()
        self.assertTrue(health.try_post(handler, "/substrate", {"name": "qwen"}))
        self.assertEqual(handler.code, 410)



    def test_openai_completion_stream_uses_instrumented_substrate_and_standard_chunks(self):
        class CompletionSubstrate:
            def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
                return "hello"

            def chat_stream(self, messages, max_new=256, mem_out=None, sample=True):
                if mem_out is not None:
                    mem_out.update(mode="prompt", applied=[], assembled_messages=list(messages),
                                   final_prompt="<rendered>hi</rendered>")
                yield "hel"
                yield "lo"

            def last_finish_reason(self):
                return "stop"

            def last_stream_trace(self):
                return []

        original = app.SUB
        app.SUB = CompletionSubstrate()
        handler = CaptureHandler()
        try:
            generation_gateway.openai_completion(handler, {"prompt": "hi", "stream": True, "model": "m"})
        finally:
            app.SUB = original
        wire = handler.wfile.getvalue().decode("utf-8")
        self.assertIn('"text": "hel"', wire)
        self.assertIn('"text": "lo"', wire)
        self.assertNotIn("tokens_committed", wire)
        self.assertNotIn("step_lens", wire)
        self.assertTrue(wire.endswith("data: [DONE]\n\n"))

    def test_native_stream_preserves_clozn_events(self):
        frame = b'data: {"type":"tokens_committed","items":[{"piece":"x"}]}\n\n'
        response = FakeResponse([frame])
        original = generation_gateway._request
        generation_gateway._request = lambda body: response
        handler = CaptureHandler()
        try:
            generation_gateway.native_completion(handler, {"prompt": "hi", "stream": True})
        finally:
            generation_gateway._request = original
        self.assertEqual(handler.wfile.getvalue(), frame)
        self.assertTrue(response.closed)

    def test_openai_chat_lens_stays_inside_standard_chunk_envelopes(self):
        class LensSubstrate:
            def chat_stream(self, messages, max_new, mem_out=None, lens=None, on_frame=None):
                on_frame({"type": "jlens_live", "pieces": ["ready"]})
                yield "ready"

            def last_finish_reason(self):
                return "stop"

            def last_stream_trace(self):
                return []

        old_sub = app.SUB
        handler = CaptureHandler()
        app.SUB = LensSubstrate()
        try:
            sse.sse_chat(handler, [{"role": "user", "content": "hi"}], 8, "m",
                         lens={"layer": 8})
        finally:
            app.SUB = old_sub
        frames = []
        for line in handler.wfile.getvalue().decode("utf-8").splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                frames.append(json.loads(line[6:]))
        self.assertTrue(frames)
        self.assertTrue(any("clozn_lens" in frame for frame in frames))
        self.assertTrue(all(frame.get("object") == "chat.completion.chunk" for frame in frames))
        self.assertTrue(all(isinstance(frame.get("created"), int) for frame in frames))
        self.assertTrue(all(isinstance(frame.get("choices"), list) and frame["choices"] for frame in frames))


class GatewayHTTPPolicyTests(unittest.TestCase):
    def test_invalid_and_negative_content_lengths_are_rejected(self):
        for value in ("nope", "-1"):
            head, payload, handler = raw_gateway_request(
                "POST", body=b"{}", headers={"Content-Length": value}
            )
            self.assertIn(" 400 ", head)
            self.assertIn("invalid Content-Length", payload.decode("utf-8"))
            self.assertTrue(handler.close_connection)

    def test_request_body_limit_is_checked_before_reading(self):
        with mock.patch.dict(os.environ, {"CLOZN_MAX_REQUEST_BYTES": "4"}):
            head, payload, handler = raw_gateway_request(
                "POST", body=b"{}", headers={"Content-Length": "5"}
            )
        self.assertIn(" 413 ", head)
        self.assertIn("4-byte limit", payload.decode("utf-8"))
        self.assertEqual(handler.rfile.tell(), 0)
        self.assertTrue(handler.close_connection)

    def test_chunked_request_body_fails_closed(self):
        head, payload, handler = raw_gateway_request(
            "POST", body=b"{}", headers={"Transfer-Encoding": "chunked"}
        )
        self.assertIn(" 501 ", head)
        self.assertIn("chunked request bodies", payload.decode("utf-8"))
        self.assertTrue(handler.close_connection)

    def test_cors_preflight_allows_loopback_and_echoes_requested_headers(self):
        head, _, _ = raw_gateway_request("OPTIONS", headers={
            "Origin": "http://127.0.0.1:3000",
            "Access-Control-Request-Headers": "authorization, content-type",
        })
        self.assertIn(" 204 ", head)
        self.assertIn("Access-Control-Allow-Origin: http://127.0.0.1:3000", head)
        self.assertIn("Access-Control-Allow-Headers: authorization, content-type", head)

    def test_cors_preflight_rejects_untrusted_web_origins(self):
        head, _, _ = raw_gateway_request(
            "OPTIONS", headers={"Origin": "https://attacker.example"}
        )
        self.assertIn(" 403 ", head)
        self.assertNotIn("Access-Control-Allow-Origin", head)

    def test_cors_operator_can_add_an_exact_origin(self):
        with mock.patch.dict(os.environ, {"CLOZN_ORIGINS": "https://trusted.example"}):
            head, _, _ = raw_gateway_request(
                "OPTIONS", headers={"Origin": "https://trusted.example"}
            )
        self.assertIn(" 204 ", head)
        self.assertIn("Access-Control-Allow-Origin: https://trusted.example", head)

    def test_untrusted_origin_is_rejected_before_get_or_post_dispatch(self):
        for method, path in (("GET", "/healthz"), ("POST", "/capture/tier")):
            head, payload, _ = raw_gateway_request(
                method, path=path, body=b"{}", headers={"Origin": "https://attacker.example"}
            )
            self.assertIn(" 403 ", head)
            self.assertIn("browser origin is not allowed", payload.decode("utf-8"))

    def test_loopback_origin_reaches_normal_route_and_is_echoed(self):
        head, payload, _ = raw_gateway_request(
            "GET", path="/healthz", headers={"Origin": "http://localhost:3000"}
        )
        self.assertIn(" 200 ", head)
        self.assertIn("Access-Control-Allow-Origin: http://localhost:3000", head)
        self.assertEqual(json.loads(payload)["status"], "ok")


class PostGateScopeTests(unittest.TestCase):
    """backlog #2: POST_GATE stays serialized for anything substrate-shaped, but app._GATE_EXEMPT_POSTS
    carves out the two paths audited to touch neither generation state nor steer/memory (see the module
    comment above POST_GATE in app.py) -- and the gate's "cancelled" outcome (a client that vanished while
    queued, see RequestGateTests above) must surface as a distinct, documented status."""

    def test_capture_tier_bypasses_the_gate_entirely(self):
        calls = []
        original = app.POST_GATE.acquire
        app.POST_GATE.acquire = lambda *a, **kw: calls.append(1) or original()
        old_settings = clozn_settings.SETTINGS_PATH
        temp = tempfile.TemporaryDirectory(prefix="clozn-capture-tier-")
        try:
            clozn_settings.SETTINGS_PATH = os.path.join(temp.name, "studio_settings.json")
            head, payload, _ = raw_gateway_request("POST", path="/capture/tier", body=b'{"tier": "deep"}')
        finally:
            app.POST_GATE.acquire = original
            clozn_settings.SETTINGS_PATH = old_settings
            temp.cleanup()
        self.assertIn(" 200 ", head)
        self.assertEqual(json.loads(payload.decode("utf-8")), {"ok": True, "tier": "deep"})
        self.assertEqual(calls, [])                 # the gate was never even asked

    def test_substrate_post_bypasses_the_gate_entirely(self):
        calls = []
        original = app.POST_GATE.acquire
        app.POST_GATE.acquire = lambda *a, **kw: calls.append(1) or original()
        try:
            head, _, _ = raw_gateway_request("POST", path="/substrate", body=b'{}')
        finally:
            app.POST_GATE.acquire = original
        self.assertIn(" 410 ", head)                 # health.py's fixed response, reached either way
        self.assertEqual(calls, [])

    def test_a_non_exempt_post_still_goes_through_the_gate(self):
        calls = []
        original = app.POST_GATE.acquire
        app.POST_GATE.acquire = lambda *a, **kw: calls.append(1) or original(*a, **kw)
        try:
            raw_gateway_request("POST", path="/jlens", body=b'{}')
        finally:
            app.POST_GATE.acquire = original
        self.assertEqual(calls, [1])

    def test_a_client_gone_while_queued_surfaces_as_499_not_a_generic_503(self):
        original = app.POST_GATE.acquire
        app.POST_GATE.acquire = lambda *a, **kw: "cancelled"
        try:
            head, payload, _ = raw_gateway_request("POST", path="/jlens", body=b'{}')
        finally:
            app.POST_GATE.acquire = original
        self.assertIn(" 499 ", head)
        body = json.loads(payload.decode("utf-8"))
        self.assertEqual(body["error"]["message"], "client disconnected while queued")

    def test_full_and_timeout_keep_their_existing_status_codes(self):
        original = app.POST_GATE.acquire
        for outcome, expected_status in (("full", 429), ("timeout", 503)):
            def _fake_acquire(*a, _outcome=outcome, **kw):
                return _outcome
            app.POST_GATE.acquire = _fake_acquire
            try:
                head, _, _ = raw_gateway_request("POST", path="/jlens", body=b'{}')
            finally:
                app.POST_GATE.acquire = original
            self.assertIn(f" {expected_status} ", head)


class RequestGateTests(unittest.TestCase):
    def test_gate_serializes_and_bounds_admitted_requests(self):
        gate = RequestGate(capacity=2, wait_timeout=1)
        self.assertIsNone(gate.acquire())
        result = []

        def wait_for_turn():
            result.append(gate.acquire())

        thread = threading.Thread(target=wait_for_turn)
        thread.start()
        deadline = time.monotonic() + 1
        while gate.snapshot()["waiting"] != 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(gate.snapshot()["waiting"], 1)
        self.assertEqual(gate.acquire(), "full")
        gate.release()
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [None])
        self.assertEqual(gate.snapshot()["active"], 1)
        gate.release()
        self.assertEqual(gate.snapshot()["active"], 0)

    def test_gate_timeout_releases_its_admission_slot(self):
        gate = RequestGate(capacity=2, wait_timeout=0.01)
        self.assertIsNone(gate.acquire())
        self.assertEqual(gate.acquire(), "timeout")
        gate.release()
        self.assertIsNone(gate.acquire())
        gate.release()

    def test_gate_cancel_check_frees_a_queued_slot_before_the_timeout(self):
        """backlog #2: a cancel_check that fires (in production: the client's TCP connection already
        closed -- see http_policy.client_gone) must free a queued slot well before wait_timeout, not
        occupy it for the full bound. Poll fast so this stays quick without weakening the mechanism."""
        gate = RequestGate(capacity=2, wait_timeout=5)
        self.assertIsNone(gate.acquire())          # holds the only turn -- the next acquire() must queue
        calls = []

        def cancel_after_a_few_polls():
            calls.append(1)
            return len(calls) >= 3

        started = time.monotonic()
        result = gate.acquire(cancel_check=cancel_after_a_few_polls, poll_interval=0.02)
        elapsed = time.monotonic() - started
        self.assertEqual(result, "cancelled")
        self.assertLess(elapsed, 2.0)               # nowhere near the 5s wait_timeout
        self.assertGreaterEqual(len(calls), 3)
        self.assertEqual(gate.snapshot()["waiting"], 0)
        self.assertEqual(gate.snapshot()["active"], 1)   # the FIRST caller is still legitimately running
        gate.release()
        self.assertIsNone(gate.acquire())           # the slot cancellation freed is usable again
        gate.release()

    def test_gate_cancel_check_does_not_preempt_a_turn_that_becomes_available(self):
        """A cancel_check that never fires must not block a legitimate admission once the turn is free --
        cancellation is a bound on the WAIT, never a way to skip an otherwise-successful admission."""
        gate = RequestGate(capacity=2, wait_timeout=5)
        self.assertIsNone(gate.acquire())
        result = []

        def wait_for_turn():
            result.append(gate.acquire(cancel_check=lambda: False, poll_interval=0.02))

        thread = threading.Thread(target=wait_for_turn)
        thread.start()
        deadline = time.monotonic() + 1
        while gate.snapshot()["waiting"] != 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(gate.snapshot()["waiting"], 1)
        gate.release()                               # frees the turn for the queued acquire()
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [None])             # admitted normally, never "cancelled"
        gate.release()


class ClientGoneProbeTests(unittest.TestCase):
    """http_policy.client_gone: the disconnect probe RequestGate's cancel_check uses in production. A real
    socket pair (works on Windows too -- socket.socketpair() is emulated over a loopback TCP connection
    there since Python 3.5) proves the select()+MSG_PEEK mechanism against an actual closed connection,
    not just a mocked one."""

    def test_connected_socket_reads_as_not_gone(self):
        import socket
        a, b = socket.socketpair()
        try:
            handler = types.SimpleNamespace(connection=a)
            self.assertFalse(http_policy.client_gone(handler))
        finally:
            a.close(); b.close()

    def test_closed_peer_reads_as_gone(self):
        import socket
        a, b = socket.socketpair()
        try:
            b.close()                                  # the peer hangs up
            handler = types.SimpleNamespace(connection=a)
            deadline = time.monotonic() + 1
            gone = False
            while time.monotonic() < deadline:
                gone = http_policy.client_gone(handler)
                if gone:
                    break
                time.sleep(0.01)
            self.assertTrue(gone)
        finally:
            a.close()

    def test_a_handler_with_no_connection_attribute_reads_as_connected(self):
        """Every no-socket unit test (object.__new__(H) + io.BytesIO()) has no `.connection` at all -- the
        probe must fail closed toward "still connected" rather than raising or wrongly cancelling."""
        self.assertFalse(http_policy.client_gone(types.SimpleNamespace()))
        self.assertFalse(http_policy.client_gone(object()))


class SQLiteRunStoreTests(unittest.TestCase):
    def setUp(self):
        self.old_dir = store.RUNS_DIR
        self.temp = tempfile.TemporaryDirectory(prefix="clozn-sqlite-test-")
        store.RUNS_DIR = self.temp.name

    def tearDown(self):
        store.RUNS_DIR = self.old_dir
        self.temp.cleanup()

    def test_sqlite_is_authoritative_and_trace_is_content_addressed(self):
        rid = store.record(
            source="cli",
            model="m",
            substrate="engine",
            messages=[{"role": "user", "content": "hi"}],
            response="hello",
            trace={"tokens": ["hello"], "confidence": [0.9]},
        )
        self.assertIsNotNone(rid)
        self.assertTrue(os.path.isfile(os.path.join(self.temp.name, "runs.sqlite3")))
        self.assertEqual(store.get_run(rid)["trace"]["tokens"], ["hello"])
        blobs = []
        for root, _, files in os.walk(os.path.join(self.temp.name, "blobs")):
            blobs.extend(os.path.join(root, name) for name in files if name.endswith(".json"))
        self.assertEqual(len(blobs), 1)
        self.assertFalse(glob_run_json(self.temp.name))

    def test_legacy_json_requires_explicit_import(self):
        legacy = tempfile.TemporaryDirectory(prefix="clozn-json-test-")
        try:
            path = os.path.join(legacy.name, "run_legacy_one.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"id": "run_legacy_one", "source": "legacy", "response": "old"}, handle)
            self.assertIsNone(store.get_run("run_legacy_one"))
            result = store.import_json_dir(legacy.name)
            self.assertEqual(result["imported"], 1)
            self.assertEqual(store.get_run("run_legacy_one")["response"], "old")
        finally:
            legacy.cleanup()


def glob_run_json(root: str) -> bool:
    return any(name.startswith("run_") and name.endswith(".json") for name in os.listdir(root))


if __name__ == "__main__":
    unittest.main()
