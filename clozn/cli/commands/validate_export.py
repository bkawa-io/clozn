"""Three-arm validation for a LoRA merged GGUF export.

``clozn validate-export BASE --adapter TUNE --merged MERGED --suite SUITE --out RECEIPT``
generates under base+adapter, then teacher-forces that exact continuation through the base,
base+adapter, and merged arms.  The adapted-vs-merged delta is the gated claim; base is retained as a
control showing what the adapter changed.  The pure ``evaluate_with_runner`` seam is model-free.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import uuid
from datetime import datetime, timezone

from clozn import __version__, schemas
from clozn.artifacts import contracts
from clozn.cli.commands.models import _flags_for, resolve_model
from clozn.cli.fit_planner import gguf_header_from_path
from clozn.cli.engine_process import _free_port, find_engine_ex, spawn_engine
from clozn.protocol import check_worker_protocol
from clozn.receipts.quant_receipts import diff_quant_scores
from clozn.runs.identity_providers import adapter as adapter_identity

CLOZN_AUTOLOAD = True

SCHEMA_VERSION = "clozn.adapter-export-receipt.v1"
SUITE_VERSION = "clozn.adapter-export-suite.v1"
DEFAULT_BUDGETS = {
    "max_argmax_flips": 0,
    "max_mean_abs_delta_nats": 0.02,
    "max_unknown_positions": 0,
}


class ExportIdentityError(RuntimeError):
    """A live arm did not load the identity requested by the validation."""


def _canonical_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _digest_value(value) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _clean(value):
    """Recursively omit unknown values; persisted receipts never null-pad."""
    if isinstance(value, dict):
        return {str(key): cleaned for key, item in value.items()
                if (cleaned := _clean(item)) is not None}
    if isinstance(value, list):
        return [cleaned for item in value if (cleaned := _clean(item)) is not None]
    return None if value is None else value


def _require_file(path: str, label: str) -> str:
    resolved = os.path.abspath(os.path.expanduser(os.fspath(path)))
    if not os.path.isfile(resolved):
        raise ValueError(f"{label} not found: {resolved}")
    return resolved


def inspect_model(path: str) -> dict:
    """Exact model/tokenizer/template/quant identity for one GGUF."""
    resolved = _require_file(path, "GGUF")
    identity = contracts.gguf_identity(resolved)
    metadata = (gguf_header_from_path(resolved).get("metadata") or {})
    out = {
        "path": resolved,
        "sha256": identity.get("sha256"),
        "file_size": identity.get("file_size"),
        "architecture": identity.get("architecture"),
        "hidden_size": identity.get("hidden_size"),
        "layer_count": identity.get("layer_count"),
        "vocab_size": identity.get("vocab_size"),
        "tokenizer_sha256": identity.get("tokenizer_sha256"),
        "chat_template_sha256": identity.get("chat_template_sha256"),
        "quantization": identity.get("quantization"),
        "chat_template_present": (
            isinstance(metadata.get("tokenizer.chat_template"), str)
            and bool(metadata["tokenizer.chat_template"].strip())
        ),
    }
    for source, target in (
        ("tokenizer.ggml.bos_token_id", "bos_token_id"),
        ("tokenizer.ggml.eos_token_id", "eos_token_id"),
    ):
        value = metadata.get(source)
        if isinstance(value, int) and not isinstance(value, bool):
            out[target] = value
    required = ("sha256", "file_size", "architecture", "vocab_size",
                "tokenizer_sha256", "chat_template_sha256", "quantization")
    missing = [name for name in required if out.get(name) in (None, "")]
    if missing:
        raise ValueError(f"GGUF identity is incomplete ({', '.join(missing)}): {resolved}")
    return _clean(out)


def inspect_adapter(path: str) -> dict:
    """Read a LoRA GGUF without importing Torch, Transformers, PEFT, or Safetensors."""
    resolved = _require_file(path, "adapter")
    header = gguf_header_from_path(resolved)
    metadata = header.get("metadata") or {}
    selected = {
        key: value for key, value in metadata.items()
        if key.startswith("adapter.") or key in {
            "general.type", "general.architecture", "general.name",
        }
    }
    return _clean({
        "path": resolved,
        "sha256": contracts.sha256_file(resolved),
        "file_size": os.path.getsize(resolved),
        "format": (
            "lora_gguf"
            if str(metadata.get("general.type", "")).lower() == "adapter"
            and str(metadata.get("adapter.type", "")).lower() == "lora"
            else "unsupported_gguf"
        ),
        "architecture": metadata.get("general.architecture"),
        "metadata": selected,
    })


def capture_engine_identity(*, prefer_gpu: bool) -> dict:
    """Pin the exact executable selected for all three live arms."""
    discovery = find_engine_ex(prefer_gpu=prefer_gpu)
    return _clean({
        "entrypoint": os.path.abspath(discovery.exe),
        "entrypoint_sha256": contracts.sha256_file(discovery.exe),
        "discovery_source": discovery.discovery_source,
        "backend": discovery.backend or ("gpu" if discovery.gpu else "cpu"),
        "engine_version": discovery.engine_version,
        "build_id": discovery.build_id,
        "llama_cpp_commit": discovery.llama_cpp_commit,
    })


def _positive_int(value, label: str, *, allow_zero: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < (0 if allow_zero else 1):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return value


def _budget_value(value, label: str, *, integer: bool):
    if integer:
        return _positive_int(value, label, allow_zero=True)
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value)) or float(value) < 0):
        raise ValueError(f"{label} must be a finite non-negative number")
    return float(value)


def normalize_suite(document: dict) -> dict:
    """Validate and normalize the small stdlib-only export suite format."""
    if not isinstance(document, dict):
        raise ValueError("suite must be a JSON object")
    version = document.get("schema_version", SUITE_VERSION)
    if version != SUITE_VERSION:
        raise ValueError(f"suite schema_version must be {SUITE_VERSION!r}")
    suite_id = document.get("suite_id")
    if not isinstance(suite_id, str) or not suite_id.strip():
        raise ValueError("suite_id must be a non-empty string")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("suite cases must be a non-empty array")
    default_seeds = document.get("seeds", [0])
    if (not isinstance(default_seeds, list) or not default_seeds
            or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in default_seeds)):
        raise ValueError("suite seeds must be a non-empty array of integers")
    if len(set(default_seeds)) != len(default_seeds):
        raise ValueError("suite seeds must not contain duplicates")

    raw_defaults = document.get("budgets", {})
    if not isinstance(raw_defaults, dict):
        raise ValueError("suite budgets must be an object")
    budgets = dict(DEFAULT_BUDGETS)
    budgets.update(raw_defaults)
    unknown_budgets = sorted(set(budgets) - set(DEFAULT_BUDGETS))
    if unknown_budgets:
        raise ValueError(f"unknown suite budget(s): {', '.join(unknown_budgets)}")
    budgets = {
        "max_argmax_flips": _budget_value(
            budgets["max_argmax_flips"], "max_argmax_flips", integer=True),
        "max_mean_abs_delta_nats": _budget_value(
            budgets["max_mean_abs_delta_nats"], "max_mean_abs_delta_nats", integer=False),
        "max_unknown_positions": _budget_value(
            budgets["max_unknown_positions"], "max_unknown_positions", integer=True),
    }
    default_max_tokens = _positive_int(document.get("max_tokens", 128), "max_tokens")
    default_topk = _positive_int(document.get("topk", 8), "topk")

    normalized = []
    seen_ids = set()
    for index, raw in enumerate(cases):
        if not isinstance(raw, dict):
            raise ValueError(f"cases[{index}] must be an object")
        case_id = raw.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"cases[{index}].id must be a non-empty string")
        if case_id in seen_ids:
            raise ValueError(f"duplicate case id: {case_id!r}")
        seen_ids.add(case_id)

        if "messages" in raw:
            messages = raw["messages"]
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"case {case_id!r} messages must be a non-empty array")
            normalized_messages = []
            for message_index, message in enumerate(messages):
                if (not isinstance(message, dict)
                        or not isinstance(message.get("role"), str)
                        or not message["role"]
                        or not isinstance(message.get("content"), str)):
                    raise ValueError(
                        f"case {case_id!r} messages[{message_index}] needs string role/content")
                normalized_messages.append({
                    "role": message["role"],
                    "content": message["content"],
                })
        elif isinstance(raw.get("prompt"), str) and raw["prompt"]:
            normalized_messages = [{"role": "user", "content": raw["prompt"]}]
        else:
            raise ValueError(f"case {case_id!r} needs messages or a non-empty prompt")

        seeds = raw.get("seeds", default_seeds)
        if (not isinstance(seeds, list) or not seeds
                or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds)
                or len(set(seeds)) != len(seeds)):
            raise ValueError(f"case {case_id!r} seeds must be unique integers")
        case_budgets = dict(budgets)
        overrides = raw.get("budgets", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"case {case_id!r} budgets must be an object")
        if set(overrides) - set(DEFAULT_BUDGETS):
            raise ValueError(f"case {case_id!r} has unknown budget names")
        case_budgets.update(overrides)
        case_budgets = {
            "max_argmax_flips": _budget_value(
                case_budgets["max_argmax_flips"], "max_argmax_flips", integer=True),
            "max_mean_abs_delta_nats": _budget_value(
                case_budgets["max_mean_abs_delta_nats"],
                "max_mean_abs_delta_nats", integer=False),
            "max_unknown_positions": _budget_value(
                case_budgets["max_unknown_positions"],
                "max_unknown_positions", integer=True),
        }
        normalized.append({
            "id": case_id,
            "messages": normalized_messages,
            "seeds": list(seeds),
            "max_tokens": _positive_int(
                raw.get("max_tokens", default_max_tokens), f"case {case_id!r} max_tokens"),
            "topk": _positive_int(raw.get("topk", default_topk), f"case {case_id!r} topk"),
            "budgets": case_budgets,
        })
    return {
        "schema_version": SUITE_VERSION,
        "suite_id": suite_id,
        "cases": normalized,
    }


def load_suite(path: str) -> tuple[dict, str]:
    resolved = _require_file(path, "suite")
    try:
        with open(resolved, encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read suite {resolved!r}: {error}") from None
    return normalize_suite(document), contracts.sha256_file(resolved)


def _check(name: str, passed: bool, detail: str) -> dict:
    return {"name": name, "status": "passed" if passed else "failed", "detail": detail}


def static_preflight(inputs: dict, engine_identity: dict) -> list[dict]:
    """Fail-fast comparisons performed before any model process starts."""
    base, adapter, merged = inputs["base"], inputs["adapter"], inputs["merged"]
    checks = []
    for field in ("architecture", "hidden_size", "layer_count", "vocab_size",
                  "tokenizer_sha256", "chat_template_sha256"):
        left, right = base.get(field), merged.get(field)
        checks.append(_check(
            field,
            left is not None and left == right,
            f"base={left!r}; merged={right!r}",
        ))
    checks.append(_check(
        "chat_template_present",
        base.get("chat_template_present") is True and merged.get("chat_template_present") is True,
        "both base and merged must carry a non-empty tokenizer.chat_template",
    ))
    checks.append(_check(
        "adapter_format",
        adapter.get("format") == "lora_gguf",
        f"detected {adapter.get('format', 'unknown')!r}; required 'lora_gguf'",
    ))
    checks.append(_check(
        "adapter_architecture",
        adapter.get("architecture") == base.get("architecture"),
        f"base={base.get('architecture')!r}; adapter={adapter.get('architecture')!r}",
    ))
    checks.append(_check(
        "quantization_declared",
        bool(base.get("quantization")) and bool(merged.get("quantization")),
        f"base={base.get('quantization')!r}; merged={merged.get('quantization')!r}",
    ))
    checks.append(_check(
        "input_hashes",
        all(isinstance(inputs[name].get("sha256"), str)
            and len(inputs[name]["sha256"]) == 64 for name in ("base", "adapter", "merged")),
        "base, adapter, and merged artifacts have exact SHA-256 identities",
    ))
    checks.append(_check(
        "engine_artifact",
        isinstance(engine_identity.get("entrypoint_sha256"), str)
        and len(engine_identity["entrypoint_sha256"]) == 64,
        f"selected {engine_identity.get('entrypoint', 'unknown')!r}",
    ))
    return checks


def _effective_arm(health: dict) -> dict:
    arm = {
        "model_sha256": health.get("model_sha256"),
        "protocol_version": health.get("protocol_version"),
        "architecture": health.get("architecture"),
        "device": health.get("device"),
    }
    attached = adapter_identity.identity({"engine_health": health})
    if attached:
        arm["adapter"] = _clean({
            "path": attached.get("path"),
            "scale": attached.get("scale"),
            "metadata": attached.get("meta"),
        })
    return _clean(arm)


def effective_preflight(inputs: dict, health_by_arm: dict) -> tuple[list[dict], dict]:
    """Verify requested identity against what the workers report as actually loaded."""
    base, adapter, merged = inputs["base"], inputs["adapter"], inputs["merged"]
    checks = []
    for arm in ("base", "adapted", "merged"):
        if not isinstance(health_by_arm.get(arm), dict):
            checks.append(_check(f"{arm}_health", False, "worker returned no health object"))
    if checks:
        return checks, {}

    h_base, h_adapted, h_merged = (
        health_by_arm["base"], health_by_arm["adapted"], health_by_arm["merged"])
    expected_hashes = {
        "base": base["sha256"],
        "adapted": base["sha256"],
        "merged": merged["sha256"],
    }
    for arm, health in health_by_arm.items():
        checks.append(_check(
            f"{arm}_model_sha256",
            health.get("model_sha256") == expected_hashes[arm],
            f"requested={expected_hashes[arm]}; loaded={health.get('model_sha256')!r}",
        ))
    protocols = [health.get("protocol_version") for health in health_by_arm.values()]
    protocol_ok = all(check_worker_protocol(version)[0] for version in protocols)
    checks.append(_check(
        "engine_protocol",
        protocol_ok and len(set(protocols)) == 1,
        f"worker protocol versions={protocols!r}",
    ))
    checks.append(_check(
        "base_unadapted",
        not isinstance(h_base.get("lora"), dict) or not h_base["lora"],
        "base control must not have an attached adapter",
    ))
    checks.append(_check(
        "merged_unadapted",
        not isinstance(h_merged.get("lora"), dict) or not h_merged["lora"],
        "merged arm must not have an attached adapter",
    ))
    loaded = adapter_identity.identity({"engine_health": h_adapted})
    requested_path = os.path.normcase(os.path.realpath(adapter["path"]))
    loaded_path = loaded.get("path")
    loaded_path_norm = (
        os.path.normcase(os.path.realpath(loaded_path))
        if isinstance(loaded_path, str) and loaded_path else "")
    checks.append(_check(
        "adapter_loaded",
        bool(loaded)
        and loaded_path_norm == requested_path
        and isinstance(loaded.get("scale"), (int, float))
        and not isinstance(loaded.get("scale"), bool)
        and math.isclose(float(loaded["scale"]), 1.0, rel_tol=0.0, abs_tol=1e-7),
        f"requested path={adapter['path']!r}, scale=1.0; "
        f"loaded path={loaded_path!r}, scale={loaded.get('scale')!r}",
    ))
    loaded_meta = loaded.get("meta") if isinstance(loaded.get("meta"), dict) else {}
    checks.append(_check(
        "adapter_loaded_metadata",
        str(loaded_meta.get("adapter.type", "")).lower() == "lora"
        and loaded_meta.get("general.architecture") == base.get("architecture"),
        f"loaded adapter.type={loaded_meta.get('adapter.type')!r}, "
        f"architecture={loaded_meta.get('general.architecture')!r}",
    ))
    effective = {arm: _effective_arm(health) for arm, health in health_by_arm.items()}
    return checks, effective


class LiveExportRunner:
    """One process per arm; injectable fakes implement the same two methods in unit tests."""

    def __init__(self, base: str, adapter: str, merged: str, *, prefer_gpu: bool):
        self.base = base
        self.adapter = adapter
        self.merged = merged
        self.prefer_gpu = prefer_gpu
        self.processes = []
        self.health_by_arm = {}
        self.engines = {}

    def __enter__(self):
        from clozn.cli.commands.quant_check import _import_engine_client
        EngineClient = _import_engine_client()
        definitions = [
            ("base", self.base, _flags_for(self.base)),
            ("adapted", self.base, {
                **_flags_for(self.base),
                "adapter": self.adapter,
                "adapter_scale": 1.0,
            }),
            ("merged", self.merged, _flags_for(self.merged)),
        ]
        try:
            for arm, model, flags in definitions:
                port = _free_port()
                process, health, _gpu = spawn_engine(
                    model, port, flags, prefer_gpu=self.prefer_gpu)
                self.processes.append(process)
                self.health_by_arm[arm] = health
                self.engines[arm] = EngineClient(port=port)
        except BaseException:
            self.close()
            raise
        return self

    def close(self):
        for process in reversed(self.processes):
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
        self.processes = []

    def __exit__(self, _exc_type, _exc, _tb):
        self.close()

    def effective_identity(self) -> dict:
        return self.health_by_arm

    def run_case(self, case: dict, seed: int) -> dict:
        """Generate on adapted; score its exact tokens on all arms."""
        prompts = {
            arm: engine.apply_template(case["messages"])
            for arm, engine in self.engines.items()
        }
        prompt_hashes = {arm: hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                         for arm, prompt in prompts.items()}
        if len(set(prompt_hashes.values())) != 1:
            raise ExportIdentityError(
                f"rendered chat template differs across live arms: {prompt_hashes}")

        generation = self.engines["adapted"].complete(
            prompts["adapted"], max_tokens=case["max_tokens"],
            temperature=0.0, seed=seed)
        try:
            continuation = generation["choices"][0]["text"]
        except (KeyError, IndexError, TypeError):
            continuation = ""
        if not isinstance(continuation, str) or not continuation:
            raise RuntimeError("adapted arm returned no continuation")

        adapted_score = self.engines["adapted"].score(
            prompt=prompts["adapted"], continuation=continuation, topk=case["topk"])
        adapted_tokens = adapted_score.get("tokens") if isinstance(adapted_score, dict) else None
        if not isinstance(adapted_tokens, list) or not adapted_tokens:
            raise RuntimeError("adapted arm could not fix continuation token ids")
        continuation_ids = [
            token.get("id") if isinstance(token, dict) else None
            for token in adapted_tokens
        ]
        if any(not isinstance(token_id, int) or isinstance(token_id, bool)
               for token_id in continuation_ids):
            raise RuntimeError("adapted arm returned malformed continuation token ids")

        scores = {"adapted": adapted_tokens}
        for arm in ("base", "merged"):
            response = self.engines[arm].score(
                prompt=prompts[arm], continuation_ids=continuation_ids, topk=case["topk"])
            scores[arm] = response.get("tokens") if isinstance(response, dict) else None
        return {
            "continuation": continuation,
            "continuation_ids": continuation_ids,
            "scores": scores,
            "prompt_sha256": prompt_hashes["adapted"],
        }


def _delta_receipt(answer_ids: list, scores_a: list, scores_b: list,
                   *, label_a: str, label_b: str) -> dict | None:
    raw = diff_quant_scores(
        answer_ids, scores_a, scores_b, label_a=label_a, label_b=label_b)
    if not isinstance(raw, dict) or raw.get("causal_verified") is not True:
        return None
    summary = raw.get("summary") or {}
    mean_delta = summary.get("mean_abs_delta_nats_all")
    if (not isinstance(mean_delta, (int, float))
            or not math.isfinite(float(mean_delta))):
        return None
    positions = []
    for position in raw.get("positions") or []:
        if not isinstance(position, dict):
            continue
        delta = position.get("delta_nats")
        if not isinstance(delta, (int, float)) or not math.isfinite(float(delta)):
            return None
        positions.append(_clean({
            "index": position.get("index"),
            "token_id": position.get("token_id"),
            "delta_nats": float(delta),
            "status": position.get("status"),
            "argmax_a_id": position.get("argmax_a_id"),
            "argmax_b_id": position.get("argmax_b_id"),
        }))
    if not positions:
        return None
    return {
        "causal_verified": True,
        "n_tokens": raw["n_tokens"],
        "n_flipped": int(summary.get("n_flipped", 0)),
        "n_preserved": int(summary.get("n_preserved", 0)),
        "n_unknown": int(summary.get("n_unknown", 0)),
        "mean_abs_delta_nats": float(mean_delta),
        "positions": positions,
    }


def _run_ids(receipt_id: str, case_id: str, seed: int) -> dict:
    return {
        arm: f"adapter-export-run-{_digest_value([receipt_id, case_id, seed, arm])[:20]}"
        for arm in ("base", "adapted", "merged")
    }


def _case_result(receipt_id: str, case: dict, seed: int, runner) -> dict:
    base = {
        "case_id": case["id"],
        "seed": seed,
        "messages_sha256": _digest_value(case["messages"]),
        "run_ids": _run_ids(receipt_id, case["id"], seed),
    }
    try:
        result = runner.run_case(case, seed)
    except ExportIdentityError as error:
        return {**base, "status": "identity_mismatch", "error": str(error)}
    except Exception as error:
        return {
            **base,
            "status": "execution_error",
            "error": f"{type(error).__name__}: {error}",
        }

    continuation = result.get("continuation")
    ids = result.get("continuation_ids")
    scores = result.get("scores")
    if (not isinstance(continuation, str) or not isinstance(ids, list)
            or not isinstance(scores, dict)):
        return {**base, "status": "execution_error",
                "error": "runner returned a malformed case result"}
    control = _delta_receipt(
        ids, scores.get("base"), scores.get("adapted"),
        label_a="base", label_b="adapted")
    primary = _delta_receipt(
        ids, scores.get("adapted"), scores.get("merged"),
        label_a="adapted", label_b="merged")
    if control is None or primary is None:
        return {
            **base,
            "status": "inconclusive",
            "continuation_sha256": hashlib.sha256(
                continuation.encode("utf-8")).hexdigest(),
            "error": "one or more arms could not be aligned for exact teacher-forced scoring",
        }

    budgets = case["budgets"]
    assertions = [
        {
            "name": "max_argmax_flips",
            "passed": primary["n_flipped"] <= budgets["max_argmax_flips"],
            "observed": primary["n_flipped"],
            "budget": budgets["max_argmax_flips"],
        },
        {
            "name": "max_mean_abs_delta_nats",
            "passed": (
                primary["mean_abs_delta_nats"]
                <= budgets["max_mean_abs_delta_nats"]
            ),
            "observed": primary["mean_abs_delta_nats"],
            "budget": budgets["max_mean_abs_delta_nats"],
        },
        {
            "name": "max_unknown_positions",
            "passed": primary["n_unknown"] <= budgets["max_unknown_positions"],
            "observed": primary["n_unknown"],
            "budget": budgets["max_unknown_positions"],
        },
    ]
    unknown_failed = not assertions[-1]["passed"]
    status = (
        "inconclusive" if unknown_failed
        else ("passed" if all(item["passed"] for item in assertions)
              else "behavioral_mismatch")
    )
    return {
        **base,
        "status": status,
        "continuation_sha256": hashlib.sha256(
            continuation.encode("utf-8")).hexdigest(),
        "teacher_forced": {
            "base_vs_adapted": control,
            "adapted_vs_merged": primary,
        },
        "assertions": assertions,
    }


def _summary(expanded_count: int, cases: list[dict]) -> dict:
    return {
        "expanded_case_count": expanded_count,
        "completed": sum(case["status"] not in ("execution_error", "identity_mismatch")
                         for case in cases),
        "passed": sum(case["status"] == "passed" for case in cases),
        "mismatched": sum(case["status"] == "behavioral_mismatch" for case in cases),
        "inconclusive": sum(case["status"] == "inconclusive" for case in cases),
        "execution_errors": sum(case["status"] == "execution_error" for case in cases),
        "teacher_forced_tokens": sum(
            ((case.get("teacher_forced") or {}).get("adapted_vs_merged") or {}).get(
                "n_tokens", 0)
            for case in cases
        ),
    }


def _verdict(preflight: dict, cases: list[dict], expanded_count: int,
             *, startup_error: str | None = None) -> dict:
    if startup_error:
        return {"status": "execution_error", "reason": startup_error}
    if preflight["status"] != "passed" or any(
            case["status"] == "identity_mismatch" for case in cases):
        failed = [check["name"] for check in preflight["checks"]
                  if check["status"] == "failed"]
        suffix = f": {', '.join(failed)}" if failed else ""
        return {"status": "identity_mismatch",
                "reason": f"static or effective identity preflight failed{suffix}"}
    if any(case["status"] == "execution_error" for case in cases):
        return {"status": "execution_error",
                "reason": "at least one case/seed arm failed to execute"}
    if any(case["status"] == "behavioral_mismatch" for case in cases):
        return {"status": "behavioral_mismatch",
                "reason": "adapted and merged arms exceeded a declared teacher-forced budget"}
    if len(cases) != expanded_count or any(
            case["status"] == "inconclusive" for case in cases):
        return {"status": "inconclusive",
                "reason": "one or more case/seed comparisons lacked complete causal scoring evidence"}
    return {
        "status": "equivalent_within_budget",
        "reason": (
            f"all {expanded_count} case/seed evaluations passed the declared "
            "teacher-forced budgets"
        ),
    }


def new_receipt(inputs: dict, suite: dict, suite_sha256: str,
                engine_identity: dict) -> dict:
    expanded = sum(len(case["seeds"]) for case in suite["cases"])
    same_quant = inputs["base"]["quantization"] == inputs["merged"]["quantization"]
    quant_note = (
        "same declared quantization; behavior was compared, byte equivalence was not claimed"
        if same_quant else
        f"different declared quantization "
        f"({inputs['base']['quantization']} vs {inputs['merged']['quantization']}); "
        "only behavior within budget can be claimed, never byte equivalence"
    )
    receipt_id = f"adapter-export-{uuid.uuid4().hex}"
    producer = {"clozn_version": __version__}
    if engine_identity:
        producer["engine"] = engine_identity
    return {
        "schema_version": SCHEMA_VERSION,
        "receipt_id": receipt_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "producer": producer,
        "suite": {
            "suite_id": suite["suite_id"],
            "sha256": suite_sha256,
            "case_count": len(suite["cases"]),
            "expanded_case_count": expanded,
        },
        "inputs": inputs,
        "preflight": {"status": "failed", "checks": []},
        "cases": [],
        "summary": _summary(expanded, []),
        "verdict": {"status": "inconclusive", "reason": "validation has not run"},
        "claims": {
            "primary_comparison": "base_plus_adapter_vs_merged",
            "base_role": "control",
            "byte_equivalent": False,
            "semantic_equivalence_proven": False,
            "quantization_note": quant_note,
        },
    }


def evaluate_with_runner(receipt: dict, suite: dict, runner) -> dict:
    """Pure orchestration over a duck-typed runner; no model imports or process creation."""
    static_checks = static_preflight(
        receipt["inputs"], receipt["producer"].get("engine", {}))
    preflight = {
        "status": (
            "passed" if all(check["status"] == "passed" for check in static_checks)
            else "failed"
        ),
        "checks": static_checks,
    }
    receipt["preflight"] = preflight
    expanded = receipt["suite"]["expanded_case_count"]
    if preflight["status"] == "passed":
        effective_checks, effective = effective_preflight(
            receipt["inputs"], runner.effective_identity())
        preflight["checks"].extend(effective_checks)
        if effective and all(
                {"model_sha256", "protocol_version"}.issubset(arm)
                for arm in effective.values()):
            preflight["effective_identity"] = effective
        preflight["status"] = (
            "passed" if all(check["status"] == "passed"
                            for check in preflight["checks"]) else "failed"
        )
    if preflight["status"] == "passed":
        for case in suite["cases"]:
            for seed in case["seeds"]:
                receipt["cases"].append(
                    _case_result(receipt["receipt_id"], case, seed, runner))
    receipt["summary"] = _summary(expanded, receipt["cases"])
    receipt["verdict"] = _verdict(preflight, receipt["cases"], expanded)
    schemas.validate(receipt, SCHEMA_VERSION)
    return receipt


def mark_execution_error(receipt: dict, error: BaseException) -> dict:
    message = f"{type(error).__name__}: {error}"
    expanded = receipt["suite"]["expanded_case_count"]
    receipt["summary"] = {
        **_summary(expanded, receipt["cases"]),
        "execution_errors": max(1, _summary(expanded, receipt["cases"])["execution_errors"]),
    }
    receipt["verdict"] = _verdict(
        receipt["preflight"], receipt["cases"], expanded, startup_error=message)
    receipt["startup_error"] = message
    schemas.validate(receipt, SCHEMA_VERSION)
    return receipt


def write_receipt(path: str, receipt: dict) -> str:
    """Validate, fsync, and atomically publish in the destination directory."""
    schemas.validate(receipt, SCHEMA_VERSION)
    resolved = os.path.abspath(os.path.expanduser(path))
    parent = os.path.dirname(resolved)
    os.makedirs(parent, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(resolved)}.", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(receipt, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
    except BaseException:
        try:
            os.remove(temporary)
        except OSError:
            pass
        raise
    return resolved


def format_report(receipt: dict, out_path: str) -> str:
    summary, verdict = receipt["summary"], receipt["verdict"]
    lines = [
        f"validate-export: {verdict['status']}",
        f"  {verdict['reason']}",
        f"  cases: {summary['passed']} passed, {summary['mismatched']} mismatched, "
        f"{summary['inconclusive']} inconclusive, "
        f"{summary['execution_errors']} execution error(s)",
        f"  teacher-forced tokens: {summary['teacher_forced_tokens']}",
        f"  receipt: {out_path}",
        "  claim: base+adapter vs merged behavior within declared budgets; "
        "not byte equivalence or proof of semantic equivalence",
    ]
    failed = [
        check["name"] for check in receipt["preflight"]["checks"]
        if check["status"] == "failed"
    ]
    if failed:
        lines.append(f"  failed preflight: {', '.join(failed)}")
    return "\n".join(lines)


def add_subparser(sub):
    parser = sub.add_parser(
        "validate-export",
        help="three-arm LoRA merged-export equivalence gate with a versioned receipt")
    parser.add_argument("base", help="base model GGUF")
    parser.add_argument("--adapter", required=True, help="LoRA GGUF applied to the base arm")
    parser.add_argument("--merged", required=True, help="merged model GGUF under validation")
    parser.add_argument("--suite", required=True, help="JSON export-validation suite")
    parser.add_argument("--out", required=True, help="atomic output receipt path")
    parser.add_argument("--cpu", action="store_true", help="force CPU workers")
    parser.add_argument("--json", action="store_true",
                        help="print the same machine-readable receipt written to --out")
    parser.set_defaults(fn=cmd_validate_export)
    return parser


def cmd_validate_export(args):
    from clozn.cli import main as ctx

    try:
        base_path = resolve_model(args.base)
        merged_path = resolve_model(args.merged)
        adapter_path = _require_file(args.adapter, "adapter")
        suite, suite_sha256 = load_suite(args.suite)
        inputs = {
            "base": inspect_model(base_path),
            "adapter": inspect_adapter(adapter_path),
            "merged": inspect_model(merged_path),
        }
    except Exception as error:
        raise ctx.CloznError(f"validate-export preflight could not be prepared: {error}") from None

    try:
        engine_identity = capture_engine_identity(prefer_gpu=not args.cpu)
    except Exception as error:
        receipt = new_receipt(inputs, suite, suite_sha256, {})
        receipt["preflight"] = {
            "status": "failed",
            "checks": static_preflight(inputs, {}),
        }
        mark_execution_error(receipt, error)
        try:
            out_path = write_receipt(args.out, receipt)
        except Exception as write_error:
            raise ctx.CloznError(
                f"engine unavailable ({error}); could not write export receipt: {write_error}"
            ) from None
        if args.json:
            print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(format_report(receipt, out_path))
        return 1

    receipt = new_receipt(inputs, suite, suite_sha256, engine_identity)
    static_checks = static_preflight(inputs, engine_identity)
    if any(check["status"] == "failed" for check in static_checks):
        receipt["preflight"] = {"status": "failed", "checks": static_checks}
        receipt["summary"] = _summary(receipt["suite"]["expanded_case_count"], [])
        receipt["verdict"] = _verdict(
            receipt["preflight"], [], receipt["suite"]["expanded_case_count"])
        schemas.validate(receipt, SCHEMA_VERSION)
    else:
        try:
            with LiveExportRunner(
                    base_path, adapter_path, merged_path,
                    prefer_gpu=not args.cpu) as runner:
                evaluate_with_runner(receipt, suite, runner)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            mark_execution_error(receipt, error)

    try:
        out_path = write_receipt(args.out, receipt)
    except Exception as error:
        raise ctx.CloznError(f"could not write export receipt: {error}") from None
    if args.json:
        print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(format_report(receipt, out_path))
    return 0 if receipt["verdict"]["status"] == "equivalent_within_budget" else 1
