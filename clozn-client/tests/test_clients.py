from __future__ import annotations

import base64
from copy import deepcopy
import json
import math
import struct
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from clozn_client import (
    AttentionKnockout,
    CloznClient,
    CloznHTTPError,
    CloznProtocolError,
    EngineClient,
    InterventionArm,
    InterventionManifest,
    MINIMAL_INTERVENTION_CONTRACT,
    PatchArm,
    PatchManifestArm,
    PatchSweepManifest,
    ScoreRequest,
    ScoreVariantsResult,
    run_manifest_batch,
)


class FakeCloznHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    requests: list[dict] = []

    def log_message(self, format, *args):  # noqa: A003 - BaseHTTPRequestHandler API
        pass

    def do_GET(self):
        parsed = urlsplit(self.path)
        self._record(None)
        if parsed.path == "/readyz":
            return self._json(200, {"status": "ready"})
        if parsed.path == "/health":
            return self._json(200, {
                "status": "ok",
                "model_sha256": "a" * 64,
                "mode": "ar",
                "capabilities": {"attn_knockout": True},
                "protocol": {
                    "schema": "clozn.engine_protocol.v1",
                    "version": "1.0",
                    "intervention_contract": {
                        "schema": MINIMAL_INTERVENTION_CONTRACT.SCHEMA,
                        "sha256": MINIMAL_INTERVENTION_CONTRACT.sha256,
                    },
                },
            })
        if parsed.path == "/runs":
            return self._json(200, {"runs": [self._run()]})
        if parsed.path == "/runs/latest":
            return self._json(200, {"available": True, "run": self._run(),
                                    "association": {"exact": True, "selector": "client_id"}})
        if parsed.path == "/runs/watch":
            return self._json(200, {"runs": [self._run()], "cursor": "cursor-2"})
        if parsed.path == "/runs/run-1":
            return self._json(200, self._run())
        if parsed.path == "/runs/missing":
            return self._json(404, {"error": "run not found"})
        if parsed.path == "/runs/run-1/timeline":
            return self._json(200, {"run_id": "run-1", "events": [{"kind": "generated"}]})
        if parsed.path == "/runs/run-1/diagnosis":
            return self._json(200, {"run_id": "run-1", "finding": "not-cut-off"})
        if parsed.path == "/runs/run-1/lineage":
            return self._json(200, {"run_id": "run-1", "ancestors": []})
        if parsed.path == "/runs/run-1/family":
            return self._json(200, {"runs": [self._run()]})
        if parsed.path == "/runs/run-1/spans":
            return self._json(200, {"run_id": "run-1", "spans": []})
        if parsed.path == "/runs/run-1/export" and parse_qs(parsed.query).get("format") == ["md"]:
            return self._text(200, "# Receipt\n")
        if parsed.path == "/runs/run-1/export":
            return self._json(200, {"schema": "clozn.receipt.bundle.v1", "run": self._run()})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        parsed = urlsplit(self.path)
        self._record(body)
        if parsed.path == "/harvest":
            raw = struct.pack("<6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
            return self._json(200, {
                "tokens": ["The", " capital", " is"],
                "layer": int(body.get("layer", 4)),
                "activations": {
                    "dtype": "float32",
                    "shape": [3, 2],
                    "data": base64.b64encode(raw).decode("ascii"),
                },
            })
        if parsed.path == "/state":
            values = body.get("values", [])
            moved = 0.0 if values == [1.0, 2.0] else round(sum(abs(float(v)) for v in values), 3)
            edited = " Paris" if moved == 0.0 else " Lyon"
            return self._json(200, {
                "applied": True,
                "layer": body["layer"],
                "moved_l2": moved,
                "baseline_top": [{"token": " Paris", "prob": 0.8}],
                "edited_top": [{"token": edited, "prob": 0.7}],
                "error": None,
            })
        if parsed.path == "/score":
            if body.get("attn_knockout"):
                logprob = -1.25
            elif body.get("steer") or body.get("steer_vec"):
                logprob = -0.75
            else:
                logprob = -0.25
            return self._json(200, {
                "n_prompt": 8,
                "n_cont": 1,
                "tokens": [{"id": 42, "piece": " Paris", "logprob": logprob,
                            "topk": [{"id": 42, "piece": " Paris", "logprob": logprob}]}],
                "sum_logprob": logprob,
                "boundary_approximate": "continuation" in body,
            })
        if parsed.path == "/v1/completions":
            return self._json(200, {"choices": [{"text": " Paris", "finish_reason": "stop"}]})
        if parsed.path == "/runs/run-1/explain":
            return self._json(200, {"run_id": "run-1", "signals": []})
        if parsed.path == "/runs/run-1/receipt":
            return self._json(200, {"run_id": "run-1", "causal_verified": True})
        return self._json(404, {"error": "not found"})

    def _record(self, body):
        self.__class__.requests.append({
            "method": self.command,
            "path": self.path,
            "headers": dict(self.headers.items()),
            "body": body,
        })

    @staticmethod
    def _run():
        return {"id": "run-1", "source": "openai_api", "model": "qwen",
                "response": "Paris", "created_at": "2026-07-21T00:00:00Z"}

    def _json(self, status, body):
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _text(self, status, body):
        payload = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/markdown")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class _StubJsonTransport:
    """No-socket transport for the private screening-only score endpoint."""

    base_url = "http://screening.invalid"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request_json(self, method, path, *, body=None):
        self.calls.append((method, path, deepcopy(body)))
        return deepcopy(self.responses.pop(0))


def _score_variants_response(ids=("baseline", "without-a"), *, status="completed"):
    return {
        "schema_version": "clozn.score-variants-worker.v1",
        "status": status,
        "continuation_ids": [10, 11],
        "results": [
            {
                "id": ids[0], "status": "completed", "prompt_token_count": 3,
                "target_logprobs": [-0.25, -0.5], "future_row_field": "retained",
            },
            {
                "id": ids[1], "status": "completed", "prompt_token_count": 3,
                "target_logprobs": [-0.75, -1.0],
            },
        ],
        "execution": {
            "strategy": "native_prefix_reuse_step_batch",
            "numerical_regime": "screening_unqualified",
            "evidence_use": "screening",
            "qualified": False,
            "execution_identity": {
                "algorithm_id": "clozn.score-variants.native-prefix-trie-step-batch.v1",
                "worker_generation_id": "worker-test",
                "model_sha256": "a" * 64,
                "tokenizer_sha256": "b" * 64,
                "future_identity_field": 1,
            },
            "waves": [{
                "wave_index": 0,
                "variant_ids": list(ids),
                "unique_prompt_count": 2,
                "logical_prompt_rows": 6,
                "physical_prompt_rows": 4,
                "continuation_kv_rows": 2,
                "required_kv_rows": 6,
                "future_wave_field": True,
            }],
            "future_execution_field": {"retained": True},
        },
        "future_additive_field": {"retained": True},
    }


class ScoreVariantsClientTests(unittest.TestCase):
    """Strict typed contract tests that never bind a local server socket."""

    @staticmethod
    def _client(*responses):
        client = EngineClient("http://screening.invalid")
        client._transport = _StubJsonTransport(responses)
        return client

    @staticmethod
    def _variants():
        return [
            {"id": "baseline", "prompt_ids": [1, 2, 3]},
            {"id": "without-a", "prompt_ids": [1, 2, 4]},
        ]

    def test_score_variants_sends_closed_body_and_returns_typed_screen(self):
        client = self._client(_score_variants_response())
        variants = self._variants()

        result = client.score_variants(variants, [10, 11])

        self.assertIsInstance(result, ScoreVariantsResult)
        self.assertEqual(client._transport.calls, [(
            "POST", "/v1/score-variants",
            {"variants": variants, "continuation_ids": [10, 11]},
        )])
        self.assertEqual([row.id for row in result.results], ["baseline", "without-a"])
        self.assertEqual(result.results[0].target_logprobs, (-0.25, -0.5))
        self.assertEqual(result.execution.numerical_regime, "screening_unqualified")
        self.assertEqual(result.execution.evidence_use, "screening")
        self.assertFalse(result.execution.qualified)
        self.assertIsNone(result.execution.full_vocab_host_reduction)
        self.assertIsNone(result.execution.target_logprob_reduction)
        self.assertIsNone(result.execution.generation_state_reuse)
        self.assertEqual(result.execution.waves[0].variant_ids, ("baseline", "without-a"))
        self.assertEqual(result.raw["future_additive_field"], {"retained": True})
        self.assertEqual(result.results[0].raw["future_row_field"], "retained")
        self.assertTrue(result.execution.waves[0].raw["future_wave_field"])

    def test_score_variants_sends_closed_generation_state_reference(self):
        response = _score_variants_response()
        response["execution"].update({
            "full_vocab_host_reduction": False,
            "target_logprob_reduction": "backend_softmax_target_gather",
            "generation_state_reuse": {
                "requested": True, "status": "applied", "applied_wave_count": 1,
            },
        })
        client = self._client(response)
        generation_state = {
            "checkpoint_id": "checkpoint-1",
            "worker_generation_id": "worker-test",
            "baseline_variant_id": "baseline",
        }

        result = client.score_variants(
            self._variants(), [10, 11], generation_state=generation_state)

        self.assertEqual(client._transport.calls[0], (
            "POST", "/v1/score-variants",
            {
                "variants": self._variants(),
                "continuation_ids": [10, 11],
                "generation_state": generation_state,
            },
        ))
        self.assertFalse(result.execution.full_vocab_host_reduction)
        self.assertEqual(
            result.execution.target_logprob_reduction,
            "backend_softmax_target_gather",
        )
        self.assertEqual(result.execution.generation_state_reuse["status"], "applied")

    def test_score_variants_rejects_malformed_generation_state_before_transport(self):
        invalid = [
            {},
            {"checkpoint_id": "c", "worker_generation_id": "w",
             "baseline_variant_id": "missing"},
            {"checkpoint_id": "c", "worker_generation_id": "w",
             "baseline_variant_id": "baseline", "extra": True},
        ]
        for generation_state in invalid:
            with self.subTest(generation_state=generation_state):
                client = self._client()
                with self.assertRaises(ValueError):
                    client.score_variants(
                        self._variants(), [10, 11], generation_state=generation_state)
                self.assertEqual(client._transport.calls, [])

    def test_score_variants_rejects_ambiguous_requests_before_transport(self):
        bad_requests = [
            ([], [3]),
            ([{"id": f"v{index}", "prompt_ids": [1]} for index in range(65)], [3]),
            ([{"id": "same", "prompt_ids": [1]},
              {"id": "same", "prompt_ids": [2]}], [3]),
            ([{"id": "", "prompt_ids": [1]}], [3]),
            ([{"id": "x" * 129, "prompt_ids": [1]}], [3]),
            ([{"id": "x", "prompt_ids": []}], [3]),
            ([{"id": "x", "prompt_ids": [True]}], [3]),
            ([{"id": "x", "prompt_ids": [1.0]}], [3]),
            ([{"id": "x", "prompt_ids": [1], "prompt": "not allowed"}], [3]),
            ([{"id": "contains space", "prompt_ids": [1]}], [3]),
            ([{"id": "x", "prompt_ids": [1]}], []),
            ([{"id": "x", "prompt_ids": [1]}], [False]),
        ]
        for variants, continuation in bad_requests:
            with self.subTest(variants=variants, continuation=continuation):
                client = self._client()
                with self.assertRaises(ValueError):
                    client.score_variants(variants, continuation)
                self.assertEqual(client._transport.calls, [])

    def test_score_variants_partial_rows_never_fabricate_scores(self):
        response = _score_variants_response(status="partial")
        response["results"][1] = {
            "id": "without-a", "status": "failed", "prompt_token_count": 3,
            "error": "decode failed",
        }
        result = self._client(response).score_variants(self._variants(), [10, 11])

        self.assertEqual(result.status, "partial")
        self.assertIsNone(result.results[1].target_logprobs)
        self.assertEqual(result.results[1].error, "decode failed")

    def test_score_variants_rejects_promoted_misaligned_or_incomplete_envelopes(self):
        promoted = _score_variants_response()
        promoted["execution"]["qualified"] = True
        short_scores = _score_variants_response()
        short_scores["results"][0]["target_logprobs"] = [-0.25]
        wrong_order = _score_variants_response()
        wrong_order["results"].reverse()
        missing_wave_member = _score_variants_response()
        missing_wave_member["execution"]["waves"][0]["variant_ids"] = ["baseline"]
        missing_identity = _score_variants_response()
        del missing_identity["execution"]["execution_identity"]

        for response in (
            promoted, short_scores, wrong_order, missing_wave_member, missing_identity,
        ):
            with self.subTest(response=response):
                client = self._client(response)
                with self.assertRaises(CloznProtocolError):
                    client.score_variants(self._variants(), [10, 11])


class ClientContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FakeCloznHandler.requests = []
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeCloznHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        FakeCloznHandler.requests.clear()

    @staticmethod
    def _manifest(*, expected_health=None) -> InterventionManifest:
        return InterventionManifest(
            name="capital-source-knockout",
            request=ScoreRequest(
                prompt="Context. Answer:",
                continuation_ids=(42,),
                topk=3,
            ),
            arms=(
                InterventionArm(
                    name="cut-source",
                    attention_knockout=(
                        AttentionKnockout(layer=3, queries=(7,), keys=(1, 2)),
                    ),
                    metadata={"role": "candidate"},
                ),
                InterventionArm(
                    name="steer-control",
                    steer={"coef": 0.25, "layer": 3},
                    steer_vec=(0.5, -0.5),
                    metadata={"role": "control"},
                ),
            ),
            expected_health=expected_health or {},
            metadata={"method": "teacher-forced"},
        )


    @staticmethod
    def _patch_manifest(*, expected_health=None) -> PatchSweepManifest:
        return PatchSweepManifest(
            name="capital-residual-patches",
            text="The capital is",
            layer=4,
            arms=(
                PatchManifestArm("identity", (0,), ((1.0, 2.0),), {"role": "control"}),
                PatchManifestArm("amplify", (0,), ((2.0, 4.0),), {"role": "candidate"}),
            ),
            expected_health=expected_health or {},
            metadata={"method": "residual-write"},
        )

    def test_gateway_returns_typed_run_and_has_no_engine_scoring_surface(self):
        client = CloznClient(self.url)
        run = client.run("run-1")
        self.assertEqual(run.id, "run-1")
        self.assertEqual(run.response, "Paris")
        self.assertFalse(hasattr(client, "score"))
        self.assertEqual(client.timeline(run.id).events[0]["kind"], "generated")
        self.assertEqual(client.export_receipt(run.id).raw["schema"], "clozn.receipt.bundle.v1")
        self.assertEqual(client.export_receipt_markdown(run.id), "# Receipt\n")

    def test_gateway_association_headers_and_query_are_explicit(self):
        client = CloznClient(self.url, client_id="research-tool", session_id="trial-7")
        latest = client.latest_run(include_derived=True)
        self.assertTrue(latest.available)
        request = FakeCloznHandler.requests[-1]
        self.assertEqual(request["headers"]["X-Clozn-Client-Id"], "research-tool")
        self.assertEqual(request["headers"]["X-Clozn-Session-Id"], "trial-7")
        self.assertEqual(parse_qs(urlsplit(request["path"]).query)["include_derived"], ["true"])

    def test_engine_score_and_knockout_are_typed_and_native_only(self):
        engine = EngineClient(self.url)
        self.assertTrue(engine.supports_attention_knockout())
        result = engine.knockout_score(
            prompt="Context. Answer:",
            continuation_ids=[42],
            knockouts=[AttentionKnockout(layer=3, queries=(7,), keys=(1, 2), renormalize=True)],
            topk=3,
        )
        self.assertEqual(result.tokens[0].piece, " Paris")
        self.assertAlmostEqual(result.sum_logprob, -1.25)
        request = FakeCloznHandler.requests[-1]
        self.assertEqual(request["path"], "/score")
        self.assertEqual(request["body"]["attn_knockout"], [{
            "layer": 3, "queries": [7], "keys": [1, 2], "renormalize": True,
        }])
        self.assertFalse(hasattr(engine, "run"))

    def test_harvest_decodes_exact_read_only_float32_tensor(self):
        engine = EngineClient(self.url)
        harvest = engine.harvest("The capital is", layer=4)
        self.assertEqual(harvest.tokens, ("The", " capital", " is"))
        self.assertEqual((harvest.n_tokens, harvest.n_embd, harvest.layer), (3, 2, 4))
        self.assertEqual(harvest.activations.dtype.str, "<f4")
        self.assertEqual(harvest.activations.tolist(), [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.assertFalse(harvest.activations.flags.writeable)

    def test_write_state_flattens_position_major_and_returns_typed_observation(self):
        import numpy as np

        engine = EngineClient(self.url)
        observation = engine.write_state(
            "The capital is", 4, (0,), np.asarray([[1.0, 2.0]], dtype=np.float64)
        )
        self.assertTrue(observation.applied)
        self.assertEqual(observation.moved_l2, 0.0)
        self.assertFalse(observation.shifted)
        request = FakeCloznHandler.requests[-1]
        self.assertEqual(request["body"]["values"], [1.0, 2.0])
        self.assertEqual(request["body"]["positions"], [0])

    def test_write_state_rejects_row_count_mismatch_before_http(self):
        import numpy as np

        engine = EngineClient(self.url)
        with self.assertRaisesRegex(ValueError, "row count"):
            engine.write_state("x", 0, (0, 1), np.asarray([[1.0, 2.0]], dtype=np.float32))
        self.assertEqual(FakeCloznHandler.requests, [])

    def test_patch_sweep_harvests_once_and_replays_named_arms_at_actual_layer(self):
        import numpy as np

        engine = EngineClient(self.url)
        sweep = engine.patch_sweep(
            "The capital is",
            (
                PatchArm("identity", (0,), np.asarray([[1.0, 2.0]], dtype=np.float32)),
                PatchArm("amplify", (0,), np.asarray([[2.0, 4.0]], dtype=np.float32),
                         metadata={"role": "candidate"}),
            ),
        )
        self.assertEqual(sweep.harvest.layer, 4)
        self.assertEqual([arm.name for arm in sweep.arms], ["identity", "amplify"])
        self.assertEqual(sweep.arms[0].observation.moved_l2, 0.0)
        self.assertTrue(sweep.arms[1].observation.shifted)
        self.assertEqual(sweep.arms[1].metadata, {"role": "candidate"})
        self.assertEqual(
            [(row["method"], urlsplit(row["path"]).path) for row in FakeCloznHandler.requests],
            [("POST", "/harvest"), ("POST", "/state"), ("POST", "/state")],
        )
        self.assertEqual(FakeCloznHandler.requests[1]["body"]["layer"], 4)
        self.assertEqual(FakeCloznHandler.requests[2]["body"]["layer"], 4)

    def test_patch_sweep_rejects_duplicate_names_and_out_of_range_positions(self):
        import numpy as np

        engine = EngineClient(self.url)
        values = np.asarray([[1.0, 2.0]], dtype=np.float32)
        arm = PatchArm("same", (0,), values)
        with self.assertRaisesRegex(ValueError, "unique"):
            engine.patch_sweep("x", (arm, arm))
        self.assertEqual(FakeCloznHandler.requests, [])
        with self.assertRaisesRegex(ValueError, "outside"):
            engine.patch_sweep("x", (PatchArm("bad", (9,), values),))
        self.assertEqual(len(FakeCloznHandler.requests), 1)
        self.assertEqual(urlsplit(FakeCloznHandler.requests[0]["path"]).path, "/harvest")

    def test_score_rejects_ambiguous_text_and_token_forms(self):
        engine = EngineClient(self.url)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            engine.score(prompt="x", prompt_ids=[1], continuation="y")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            engine.score(prompt="x", continuation=None, continuation_ids=None)

    def test_http_failures_preserve_status_and_body(self):
        client = CloznClient(self.url)
        with self.assertRaises(CloznHTTPError) as caught:
            client.run("missing")
        self.assertEqual(caught.exception.status, 404)
        self.assertEqual(caught.exception.body, {"error": "run not found"})

    def test_manifest_round_trip_and_hash_are_deterministic(self):
        manifest = self._manifest(expected_health={
            "mode": "ar",
            "capabilities": {"attn_knockout": True},
        })
        decoded = InterventionManifest.from_json(manifest.to_json())
        reordered = InterventionManifest.from_json({
            "metadata": {"method": "teacher-forced"},
            "expected_health": {"capabilities": {"attn_knockout": True}, "mode": "ar"},
            "arms": [arm.to_json_object() for arm in manifest.arms],
            "request": manifest.request.to_json_object(),
            "name": manifest.name,
            "schema": InterventionManifest.SCHEMA,
        })
        self.assertEqual(decoded, manifest)
        self.assertEqual(decoded.sha256, reordered.sha256)
        self.assertEqual(len(decoded.sha256), 64)

    def test_manifest_replay_runs_baseline_then_arms_and_reports_support_drop(self):
        manifest = self._manifest(expected_health={
            "model_sha256": "a" * 64,
            "capabilities": {"attn_knockout": True},
        })
        result = EngineClient(self.url).run_manifest(manifest)
        self.assertEqual(result.manifest_sha256, manifest.sha256)
        self.assertAlmostEqual(result.baseline.sum_logprob, -0.25)
        self.assertEqual([arm.name for arm in result.arms], ["cut-source", "steer-control"])
        self.assertAlmostEqual(result.arms[0].support_drop, 1.0)
        self.assertAlmostEqual(result.arms[1].support_drop, 0.5)
        self.assertEqual(result.engine_health["model_sha256"], "a" * 64)
        self.assertEqual(
            [(row["method"], urlsplit(row["path"]).path) for row in FakeCloznHandler.requests],
            [("GET", "/health"), ("POST", "/score"),
             ("POST", "/score"), ("POST", "/score")],
        )
        self.assertNotIn("attn_knockout", FakeCloznHandler.requests[1]["body"])
        self.assertIn("attn_knockout", FakeCloznHandler.requests[2]["body"])
        self.assertIn("steer_vec", FakeCloznHandler.requests[3]["body"])

    def test_manifest_expected_health_fails_closed_before_scoring(self):
        manifest = self._manifest(expected_health={"mode": "diffusion"})
        with self.assertRaisesRegex(CloznProtocolError, "health.mode mismatch"):
            EngineClient(self.url).run_manifest(manifest)
        self.assertEqual(len(FakeCloznHandler.requests), 1)
        self.assertEqual(FakeCloznHandler.requests[0]["method"], "GET")

    def test_manifest_rejects_duplicate_arms_and_unknown_schema(self):
        arm = InterventionArm(
            name="same",
            attention_knockout=(AttentionKnockout(layer=0, queries=(1,), keys=(0,)),),
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            InterventionManifest(
                name="bad",
                request=ScoreRequest(prompt="x", continuation="y"),
                arms=(arm, arm),
            )
        with self.assertRaisesRegex(CloznProtocolError, "unsupported manifest schema"):
            InterventionManifest.from_json({"schema": "future", "name": "x", "request": {}, "arms": []})

    def test_manifest_rejects_non_json_or_non_finite_metadata(self):
        with self.assertRaisesRegex(ValueError, "finite JSON values"):
            InterventionManifest(
                name="bad-metadata",
                request=ScoreRequest(prompt="x", continuation="y"),
                arms=(InterventionArm(
                    name="cut",
                    attention_knockout=(AttentionKnockout(layer=0, queries=(1,), keys=(0,)),),
                ),),
                metadata={"bad": math.nan},
            )

    def test_patch_manifest_round_trip_hash_and_float32_normalization(self):
        manifest = self._patch_manifest(expected_health={"mode": "ar"})
        decoded = PatchSweepManifest.from_json(manifest.to_json())
        self.assertEqual(decoded, manifest)
        self.assertEqual(decoded.sha256, manifest.sha256)
        self.assertEqual(decoded.arms[0].values, ((1.0, 2.0),))
        self.assertEqual(len(decoded.sha256), 64)

    def test_patch_manifest_replay_exports_portable_result(self):
        manifest = self._patch_manifest(expected_health={"model_sha256": "a" * 64})
        artifact = EngineClient(self.url).run_patch_manifest(manifest)
        payload = artifact.to_json_object()
        self.assertEqual(payload["schema"], "clozn.patch_sweep_result.v1")
        self.assertEqual(payload["manifest_sha256"], manifest.sha256)
        self.assertEqual(payload["layer"], 4)
        self.assertEqual(payload["n_embd"], 2)
        self.assertFalse(payload["arms"][0]["observation"]["shifted"])
        self.assertTrue(payload["arms"][1]["observation"]["shifted"])
        self.assertEqual(
            [(row["method"], urlsplit(row["path"]).path) for row in FakeCloznHandler.requests],
            [("GET", "/health"), ("POST", "/harvest"),
             ("POST", "/state"), ("POST", "/state")],
        )

    def test_patch_manifest_health_mismatch_fails_before_harvest(self):
        manifest = self._patch_manifest(expected_health={"mode": "diffusion"})
        with self.assertRaisesRegex(CloznProtocolError, "health.mode mismatch"):
            EngineClient(self.url).run_patch_manifest(manifest)
        self.assertEqual(len(FakeCloznHandler.requests), 1)
        self.assertEqual(urlsplit(FakeCloznHandler.requests[0]["path"]).path, "/health")

    def test_patch_manifest_rejects_duplicate_names_and_bad_dimensions(self):
        arm = PatchManifestArm("same", (0,), ((1.0, 2.0),))
        with self.assertRaisesRegex(ValueError, "unique"):
            PatchSweepManifest(name="bad", text="x", arms=(arm, arm))
        with self.assertRaisesRegex(ValueError, "row count"):
            PatchManifestArm("bad", (0, 1), ((1.0, 2.0),))

    def test_cli_validate_identifies_both_manifest_schemas(self):
        import contextlib
        import io
        import tempfile
        from clozn_client.__main__ import main

        for manifest in (self._manifest(), self._patch_manifest()):
            with self.subTest(schema=manifest.SCHEMA), tempfile.TemporaryDirectory() as tmp:
                path = f"{tmp}/manifest.json"
                manifest.write(path)
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    code = main(["validate", path])
                self.assertEqual(code, 0)
                payload = json.loads(stdout.getvalue())
                self.assertEqual(payload["schema"], manifest.SCHEMA)
                self.assertEqual(payload["sha256"], manifest.sha256)

    def test_cli_replay_patch_writes_result_file(self):
        import tempfile
        from clozn_client.__main__ import main

        manifest = self._patch_manifest()
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = f"{tmp}/manifest.json"
            output_path = f"{tmp}/result.json"
            manifest.write(manifest_path)
            code = main(["replay", manifest_path, "--engine-url", self.url, "-o", output_path])
            self.assertEqual(code, 0)
            with open(output_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["schema"], "clozn.patch_sweep_result.v1")
            self.assertEqual(payload["manifest_sha256"], manifest.sha256)

    def test_batch_runner_isolates_failures_and_hashes_result_names(self):
        import tempfile

        good = self._patch_manifest()
        bad = self._patch_manifest(expected_health={"mode": "diffusion"})
        with tempfile.TemporaryDirectory() as tmp:
            result = run_manifest_batch(
                EngineClient(self.url),
                (("good.json", good), ("bad.json", bad)),
                tmp,
            )
            self.assertEqual((result.succeeded, result.failed), (1, 1))
            self.assertEqual(result.items[0].status, "ok")
            self.assertTrue(result.items[0].result_path.endswith(f"{good.sha256}.result.json"))
            self.assertEqual(result.items[1].status, "error")
            self.assertIn("health.mode mismatch", result.items[1].error)

    def test_cli_batch_replays_mixed_manifests_and_writes_index(self):
        import contextlib
        import io
        import tempfile
        from pathlib import Path
        from clozn_client.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._manifest().write(root / "score.manifest.json")
            self._patch_manifest().write(root / "patch.manifest.json")
            output = root / "results"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main([
                    "batch", str(root / "score.manifest.json"), str(root / "patch.manifest.json"),
                    "--engine-url", self.url, "--output-dir", str(output),
                ])
            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["schema"], "clozn.batch_run.v1")
            self.assertEqual((payload["succeeded"], payload["failed"]), (2, 0))
            self.assertTrue((output / "index.json").exists())
            self.assertEqual(len(list(output.glob("*.result.json"))), 2)


if __name__ == "__main__":
    unittest.main()

class BatchComparisonTests(unittest.TestCase):
    def _write_run(self, root, name, digest, payload, *, status="ok", error=None):
        from pathlib import Path
        result = Path(root) / f"{name}.result.json"
        result.write_text(json.dumps(payload), encoding="utf-8")
        return {
            "path": f"{name}.manifest.json", "schema": "manifest", "name": name,
            "manifest_sha256": digest, "status": status,
            "result_path": str(result) if status == "ok" else None, "error": error,
        }

    def _write_index(self, root, filename, items):
        from pathlib import Path
        path = Path(root) / filename
        path.write_text(json.dumps({
            "schema": "clozn.batch_run.v1", "output_dir": str(root),
            "succeeded": sum(x["status"] == "ok" for x in items),
            "failed": sum(x["status"] != "ok" for x in items), "items": items, "metadata": {},
        }), encoding="utf-8")
        return path

    def test_compare_detects_numeric_and_boolean_regressions(self):
        import tempfile
        from clozn_client import compare_batch_runs
        digest = "a" * 64
        with tempfile.TemporaryDirectory() as tmp:
            baseline = self._write_run(tmp, "baseline", digest, {
                "schema": "clozn.patch_sweep_result.v1", "arms": [{"name": "edit", "observation": {
                    "moved_l2": 1.0, "shifted": False,
                }}],
            })
            candidate = self._write_run(tmp, "candidate", digest, {
                "schema": "clozn.patch_sweep_result.v1", "arms": [{"name": "edit", "observation": {
                    "moved_l2": 1.25, "shifted": True,
                }}],
            })
            left = self._write_index(tmp, "left.json", [baseline])
            right = self._write_index(tmp, "right.json", [candidate])
            result = compare_batch_runs(left, right, max_metric_delta=0.1)
            self.assertEqual(result.regressions, 2)
            self.assertEqual(result.experiments[0].status, "regressed")

    def test_compare_tolerance_and_cli_exit_codes(self):
        import contextlib
        import io
        import tempfile
        from clozn_client.__main__ import main
        digest = "b" * 64
        with tempfile.TemporaryDirectory() as tmp:
            base = self._write_run(tmp, "base", digest, {
                "schema": "clozn.intervention_run.v1", "arms": [{"name": "cut", "support_drop": 1.0}],
            })
            cand = self._write_run(tmp, "cand", digest, {
                "schema": "clozn.intervention_run.v1", "arms": [{"name": "cut", "support_drop": 1.05}],
            })
            left = self._write_index(tmp, "left.json", [base])
            right = self._write_index(tmp, "right.json", [cand])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["compare", str(left), str(right), "--max-metric-delta", "0.1"]), 0)
                self.assertEqual(main(["compare", str(left), str(right), "--max-metric-delta", "0.01"]), 1)

    def test_compare_marks_missing_manifest_and_changed_failure(self):
        import tempfile
        from clozn_client import compare_batch_runs
        a, b = "c" * 64, "d" * 64
        with tempfile.TemporaryDirectory() as tmp:
            left = self._write_index(tmp, "left.json", [
                self._write_run(tmp, "only-left", a, {}, status="error", error="old"),
                self._write_run(tmp, "changed", b, {}, status="error", error="old"),
            ])
            right = self._write_index(tmp, "right.json", [
                self._write_run(tmp, "changed2", b, {}, status="error", error="new"),
            ])
            result = compare_batch_runs(left, right)
            self.assertEqual(result.regressions, 2)

class CIReportingTests(unittest.TestCase):
    def _comparison(self):
        from clozn_client import BatchComparison, ExperimentComparison, MetricDelta
        return BatchComparison(
            "baseline/index.json",
            "candidate/index.json",
            (
                ExperimentComparison(
                    "a" * 64,
                    "capital|knockout",
                    "clozn.intervention_manifest.v1",
                    "regressed",
                    (MetricDelta("cut", "support_drop", 1.0, 1.25, 0.25, True),),
                ),
                ExperimentComparison(
                    "b" * 64,
                    "identity",
                    "clozn.patch_sweep_manifest.v1",
                    "ok",
                    (MetricDelta("same", "shifted", False, False, None, False),),
                ),
            ),
            0.1,
        )

    def test_junit_report_has_one_case_per_experiment(self):
        from xml.etree import ElementTree as ET
        from clozn_client import comparison_to_junit_xml
        root = ET.fromstring(comparison_to_junit_xml(self._comparison()))
        self.assertEqual(root.attrib["tests"], "2")
        self.assertEqual(root.attrib["failures"], "1")
        cases = root.findall("testcase")
        self.assertEqual(len(cases), 2)
        self.assertIsNotNone(cases[0].find("failure"))
        self.assertIsNone(cases[1].find("failure"))

    def test_markdown_and_annotations_include_regression_details(self):
        from clozn_client import comparison_to_github_annotations, comparison_to_markdown
        markdown = comparison_to_markdown(self._comparison())
        self.assertIn("❌ Regressed", markdown)
        self.assertIn("capital\\|knockout", markdown)
        self.assertIn("support_drop", markdown)
        annotations = comparison_to_github_annotations(self._comparison())
        self.assertIn("::error title=Clozn regression%3A capital|knockout::", annotations)
        self.assertIn("delta=0.25", annotations)

    def test_compare_cli_writes_all_ci_reports(self):
        import contextlib
        import io
        import tempfile
        from pathlib import Path
        from clozn_client.__main__ import main

        digest = "c" * 64
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_result = root / "baseline.result.json"
            candidate_result = root / "candidate.result.json"
            baseline_result.write_text(json.dumps({
                "schema": "clozn.intervention_run.v1",
                "arms": [{"name": "cut", "support_drop": 1.0}],
            }), encoding="utf-8")
            candidate_result.write_text(json.dumps({
                "schema": "clozn.intervention_run.v1",
                "arms": [{"name": "cut", "support_drop": 1.5}],
            }), encoding="utf-8")
            common = {"schema": "manifest", "name": "gate", "manifest_sha256": digest, "status": "ok", "error": None}
            left = root / "left.json"
            right = root / "right.json"
            left.write_text(json.dumps({
                "schema": "clozn.batch_run.v1", "items": [{**common, "result_path": str(baseline_result)}]
            }), encoding="utf-8")
            right.write_text(json.dumps({
                "schema": "clozn.batch_run.v1", "items": [{**common, "result_path": str(candidate_result)}]
            }), encoding="utf-8")
            junit = root / "report.xml"
            summary = root / "summary.md"
            annotations = root / "annotations.txt"
            with contextlib.redirect_stdout(io.StringIO()):
                code = main([
                    "compare", str(left), str(right), "--max-metric-delta", "0.1",
                    "--junit", str(junit), "--github-summary", str(summary),
                    "--github-annotations", str(annotations),
                ])
            self.assertEqual(code, 1)
            self.assertTrue(junit.read_text(encoding="utf-8").startswith("<?xml"))
            self.assertIn("Clozn experiment comparison", summary.read_text(encoding="utf-8"))
            self.assertIn("::error", annotations.read_text(encoding="utf-8"))

class ProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FakeCloznHandler.requests = []
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeCloznHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        FakeCloznHandler.requests.clear()

    def test_capture_round_trip_and_stable_identity(self):
        import tempfile
        from pathlib import Path
        from clozn_client import ReproducibilityRecord, capture_reproducibility

        record = capture_reproducibility(
            EngineClient(self.url),
            ("b" * 64, "a" * 64, "b" * 64),
            metadata={"git_sha": "deadbeef"},
        )
        self.assertEqual(record.manifest_sha256, ("a" * 64, "b" * 64))
        self.assertEqual(record.engine_health["model_sha256"], "a" * 64)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "provenance.json"
            record.write(path)
            loaded = ReproducibilityRecord.read(path)
        self.assertEqual(loaded.sha256, record.sha256)
        self.assertEqual(loaded.metadata["git_sha"], "deadbeef")

    def test_record_detects_tampering(self):
        from clozn_client import ReproducibilityRecord, capture_reproducibility
        record = capture_reproducibility(EngineClient(self.url), ("a" * 64,))
        payload = record.to_json_object()
        payload["engine_url"] = "http://changed"
        with self.assertRaisesRegex(ValueError, "sha256"):
            ReproducibilityRecord.from_json(payload)

    def test_cli_provenance_writes_content_addressed_record(self):
        import contextlib
        import io
        import tempfile
        from pathlib import Path
        from clozn_client.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = PatchSweepManifest(
                name="identity", text="The capital is", layer=4,
                arms=(PatchManifestArm("same", (0,), ((1.0, 2.0),)),),
            )
            path = root / "patch.manifest.json"
            manifest.write(path)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main([
                    "provenance", str(path), "--engine-url", self.url,
                    "--metadata", "git_sha=abc123",
                ])
            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["schema"], "clozn.reproducibility.v1")
            self.assertEqual(payload["manifest_sha256"], [manifest.sha256])
            self.assertEqual(payload["metadata"]["git_sha"], "abc123")
            self.assertEqual(len(payload["sha256"]), 64)

class InterventionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FakeCloznHandler.requests = []
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeCloznHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        FakeCloznHandler.requests.clear()
    def test_minimal_contract_is_small_and_deterministic(self):
        from clozn_client import MINIMAL_INTERVENTION_CONTRACT, InterventionOperation

        contract = MINIMAL_INTERVENTION_CONTRACT
        self.assertEqual(contract.to_json_object()["schema"], "clozn.intervention_contract.v1")
        self.assertEqual(len(contract.operations), 4)
        self.assertEqual(len(contract.sha256), 64)
        self.assertEqual(
            contract.spec(InterventionOperation.CAPTURE_RESIDUAL).tensor_contract,
            "little-endian float32 matrix shaped [n_tokens, n_embd]",
        )
        self.assertNotIn("steer", {spec.operation.value for spec in contract.operations})

    def test_required_operations_derive_from_manifest_without_promoting_steering(self):
        from clozn_client import InterventionOperation, required_operations

        manifest = ClientContractTests._manifest()
        self.assertEqual(
            required_operations(manifest),
            (
                InterventionOperation.ATTENTION_KNOCKOUT,
                InterventionOperation.SCORE_TEACHER_FORCED,
            ),
        )
        patch = ClientContractTests._patch_manifest()
        self.assertEqual(
            required_operations(patch),
            (
                InterventionOperation.CAPTURE_RESIDUAL,
                InterventionOperation.REPLACE_RESIDUAL,
            ),
        )

    def test_engine_contract_report_checks_advertised_capability(self):
        from clozn_client import InterventionOperation

        engine = EngineClient(self.url)
        report = engine.contract_report(
            InterventionOperation.SCORE_TEACHER_FORCED,
            InterventionOperation.ATTENTION_KNOCKOUT,
        )
        self.assertTrue(report.compatible)
        self.assertEqual(report.protocol_compatibility.value, "verified")
        self.assertEqual(report.to_json_object()["schema"], "clozn.intervention_contract_report.v1")
        report.require_compatible()

    def test_contract_fails_closed_when_protocol_is_unadvertised(self):
        from clozn_client import InterventionOperation, check_contract

        report = check_contract(
            {"status": "ok", "capabilities": {"attn_knockout": True}},
            [InterventionOperation.ATTENTION_KNOCKOUT],
        )
        self.assertFalse(report.compatible)
        self.assertEqual(report.protocol_compatibility.value, "unadvertised")
        with self.assertRaisesRegex(CloznProtocolError, "health.protocol is required"):
            report.require_compatible()

    def test_contract_rejects_mismatched_digest(self):
        from clozn_client import InterventionOperation, check_contract

        report = check_contract(
            {
                "status": "ok",
                "capabilities": {"attn_knockout": True},
                "protocol": {
                    "schema": "clozn.engine_protocol.v1",
                    "version": "1.0",
                    "intervention_contract": {
                        "schema": MINIMAL_INTERVENTION_CONTRACT.SCHEMA,
                        "sha256": "0" * 64,
                    },
                },
            },
            [InterventionOperation.ATTENTION_KNOCKOUT],
        )
        self.assertFalse(report.compatible)
        self.assertEqual(report.protocol_compatibility.value, "incompatible")
        with self.assertRaisesRegex(CloznProtocolError, "digest does not match"):
            report.require_compatible()

    def test_contract_fails_closed_when_knockout_capability_is_missing(self):
        from clozn_client import InterventionOperation, check_contract

        report = check_contract(
            {
                "status": "ok",
                "capabilities": {},
                "protocol": {
                    "schema": "clozn.engine_protocol.v1",
                    "version": "1.0",
                    "intervention_contract": {
                        "schema": MINIMAL_INTERVENTION_CONTRACT.SCHEMA,
                        "sha256": MINIMAL_INTERVENTION_CONTRACT.sha256,
                    },
                },
            },
            [InterventionOperation.ATTENTION_KNOCKOUT],
        )
        self.assertFalse(report.compatible)
        with self.assertRaisesRegex(CloznProtocolError, "attn_knockout"):
            report.require_compatible()

    def test_unknown_operation_is_rejected(self):
        from clozn_client import check_contract

        with self.assertRaisesRegex(ValueError, "not in clozn.intervention_contract.v1"):
            check_contract({"status": "ok"}, ["capture.everything"])

class CaptureContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FakeCloznHandler.requests = []
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeCloznHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_capture_budget_rejects_oversized_capture(self):
        from clozn_client import CaptureBudget
        with self.assertRaisesRegex(CloznProtocolError, "capture exceeds client budget"):
            EngineClient(self.url).harvest("The capital is", budget=CaptureBudget(max_tokens=1))

    def test_capture_statistics_and_binding_are_receipt_portable(self):
        from clozn_client import CaptureStatistics, bind_contract_evidence
        manifest = ClientContractTests._patch_manifest()
        harvest = EngineClient(self.url).harvest(manifest.text)
        stats = CaptureStatistics.from_harvest(harvest)
        self.assertEqual(stats.elements, stats.n_tokens * stats.n_embd)
        self.assertEqual(stats.n_bytes, stats.elements * 4)
        binding = bind_contract_evidence(manifest, capture=harvest)
        payload = binding.to_json_object()
        self.assertEqual(payload["schema"], "clozn.intervention_contract_binding.v1")
        self.assertEqual(len(payload["contract_sha256"]), 64)
        self.assertEqual(payload["replay_classes"], ["re_prefilled"])
        self.assertEqual(payload["capture"]["dtype"], "float32-le")

    def test_patch_artifact_embeds_contract_identity_and_capture_statistics(self):
        artifact = EngineClient(self.url).run_patch_manifest(ClientContractTests._patch_manifest())
        payload = artifact.to_json_object()
        self.assertEqual(payload["contract_binding"]["schema"], "clozn.intervention_contract_binding.v1")
        self.assertEqual(payload["capture_statistics"]["n_embd"], artifact.n_embd)
        self.assertEqual(payload["capture_statistics"]["n_tokens"], len(artifact.tokens))
