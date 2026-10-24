"""Private HTTP client for the Clozn v1 native worker.

This package is an implementation seam between the Python gateway and one local
worker. It intentionally exposes only the worker operations required by the v1
product: generation, cancellation, model-native chat templating, recorded-answer
scoring, and exact-reference probes. It is not a public research SDK.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any


class EngineError(RuntimeError):
    """A typed error returned by the private native worker."""

    def __init__(self, message: str, *, response: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.response = dict(response) if isinstance(response, Mapping) else None


def _integer_ids(value: Sequence[int], label: str, *, nonempty: bool = False) -> list[int]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be an integer sequence")
    items = list(value)
    if nonempty and not items:
        raise ValueError(f"{label} must be a non-empty sequence")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in items):
        raise ValueError(f"{label} must contain only integers")
    return items


class EngineClient:
    """Synchronous client for one running Clozn worker."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8080, timeout: float = 120.0):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.base + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data is not None else {},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raw = error.read().decode("utf-8", "replace")
            try:
                document = json.loads(raw)
                message = document.get("error", raw) if isinstance(document, Mapping) else raw
            except json.JSONDecodeError:
                document = None
                message = raw
            raise EngineError(
                f"{method} {path} -> {error.code}: {message}",
                response=document if isinstance(document, Mapping) else None,
            ) from None
        if not isinstance(payload, dict):
            raise EngineError(f"{method} {path} returned a non-object JSON response")
        return payload

    def _get(self, path: str) -> dict:
        return self._request("GET", path)

    def _post(self, path: str, body: dict) -> dict:
        return self._request("POST", path, body)

    def health(self) -> dict:
        """Return the worker's model identity and retained capabilities."""
        return self._get("/health")

    def complete(self, prompt: str, **params: Any) -> dict:
        """Generate from one already-rendered prompt."""
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a non-empty string")
        body = dict(params)
        body["prompt"] = prompt
        return self._post("/v1/completions", body)

    def apply_template(self, messages: Sequence[Mapping[str, Any]],
                       add_assistant: bool = True,
                       enable_thinking: bool = False) -> str:
        """Render messages with the loaded model's embedded chat template."""
        return self.apply_template_info(
            messages,
            add_assistant=add_assistant,
            enable_thinking=enable_thinking,
        )["prompt"]

    def apply_template_info(
        self,
        messages: Sequence[Mapping[str, Any]],
        add_assistant: bool = True,
        enable_thinking: bool = False,
        include_token_ids: bool = False,
    ) -> dict[str, Any]:
        """Render messages and return the worker-authoritative prompt token count."""
        if isinstance(messages, (str, bytes, bytearray)) or not isinstance(messages, Sequence):
            raise ValueError("messages must be a sequence")
        body: dict[str, Any] = {
            "messages": [dict(message) for message in messages],
            "add_assistant": bool(add_assistant),
            "enable_thinking": bool(enable_thinking),
        }
        if include_token_ids:
            body["include_token_ids"] = True
        response = self._post("/apply_template", body)
        prompt = response.get("prompt")
        if not isinstance(prompt, str):
            raise EngineError("POST /apply_template returned invalid 'prompt' (expected string)")
        prompt_tokens = response.get("prompt_tokens")
        if (not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool)
                or prompt_tokens < 0):
            raise EngineError(
                "POST /apply_template returned invalid 'prompt_tokens' "
                "(expected non-negative int)"
            )
        result: dict[str, Any] = {"prompt": prompt, "prompt_tokens": prompt_tokens}
        if include_token_ids:
            token_ids = response.get("prompt_token_ids")
            if (not isinstance(token_ids, list)
                    or any(not isinstance(token, int) or isinstance(token, bool)
                           for token in token_ids)):
                raise EngineError(
                    "POST /apply_template returned invalid 'prompt_token_ids' "
                    "(expected an integer array)"
                )
            result["prompt_token_ids"] = token_ids
        return result

    def score(
        self,
        prompt: str | None = None,
        prompt_ids: Sequence[int] | None = None,
        continuation_ids: Sequence[int] | None = None,
        continuation: str | None = None,
        topk: int = 0,
        use_prefix_cache: bool = True,
    ) -> dict:
        """Teacher-force one continuation through the trusted scalar score route."""
        if not isinstance(topk, int) or isinstance(topk, bool) or topk < 0:
            raise ValueError("topk must be a non-negative integer")
        if not isinstance(use_prefix_cache, bool):
            raise ValueError("use_prefix_cache must be a boolean")
        body: dict[str, Any] = {"topk": topk, "use_prefix_cache": use_prefix_cache}
        if prompt_ids is not None:
            body["prompt_ids"] = _integer_ids(prompt_ids, "prompt_ids", nonempty=True)
        elif prompt is not None:
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("prompt must be a non-empty string")
            body["prompt"] = prompt
        else:
            raise ValueError("score requires prompt or prompt_ids")

        if continuation_ids is not None:
            body["continuation_ids"] = _integer_ids(
                continuation_ids, "continuation_ids", nonempty=True
            )
        elif continuation is not None:
            if not isinstance(continuation, str) or not continuation:
                raise ValueError("continuation must be a non-empty string")
            body["continuation"] = continuation
        else:
            raise ValueError("score requires continuation or continuation_ids")
        return self._post("/score", body)

    def reference_match_arms(
        self,
        arms: Sequence[Mapping[str, Any]],
        *,
        reference_token_ids: Sequence[int],
        generation_contract: Mapping[str, Any],
        parent_anchor_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Run the worker's bounded native exact-reference multi-arm probe."""
        if isinstance(arms, (str, bytes, bytearray)) or not isinstance(arms, Sequence) or not arms:
            raise ValueError("arms must be a non-empty sequence")
        body: dict[str, Any] = {
            "arms": [dict(arm) for arm in arms],
            "reference_token_ids": _integer_ids(
                reference_token_ids, "reference_token_ids", nonempty=True
            ),
            "generation_contract": dict(generation_contract),
        }
        if parent_anchor_prompt is not None:
            if not isinstance(parent_anchor_prompt, str) or not parent_anchor_prompt:
                raise ValueError("parent_anchor_prompt must be a non-empty string when provided")
            body["parent_anchor_prompt"] = parent_anchor_prompt
        response = self._post("/v1/reference-match/arms", body)
        if not isinstance(response.get("results"), list):
            raise EngineError("POST /v1/reference-match/arms returned an invalid response")
        if len(response["results"]) != len(arms):
            raise EngineError("POST /v1/reference-match/arms returned the wrong arm count")
        return response

    def reference_match_persistent_create(
        self,
        prompt: str,
        *,
        reference_token_ids: Sequence[int],
        generation_contract: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Create a worker-local persistent exact-reference parent session."""
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a non-empty canonical rendered prompt")
        response = self._post("/v1/reference-match/persistent/create", {
            "prompt": prompt,
            "reference_token_ids": _integer_ids(
                reference_token_ids, "reference_token_ids", nonempty=True
            ),
            "generation_contract": dict(generation_contract),
        })
        if not isinstance(response.get("session_id"), str):
            raise EngineError("persistent session create returned an invalid response")
        return response

    def reference_match_persistent_probe(
        self,
        session_id: str,
        *,
        expected_parent_version: int,
        children: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Probe one bounded child wave against a persistent parent."""
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(expected_parent_version, int) or isinstance(expected_parent_version, bool):
            raise ValueError("expected_parent_version must be an integer")
        if (isinstance(children, (str, bytes, bytearray))
                or not isinstance(children, Sequence) or not children):
            raise ValueError("children must be a non-empty sequence")
        response = self._post("/v1/reference-match/persistent/probe", {
            "session_id": session_id,
            "expected_parent_version": expected_parent_version,
            "children": [dict(child) for child in children],
        })
        if not isinstance(response.get("results"), list):
            raise EngineError("persistent session probe returned an invalid response")
        if len(response["results"]) != len(children):
            raise EngineError("persistent session probe returned the wrong child count")
        return response

    def reference_match_persistent_promote(
        self,
        session_id: str,
        *,
        expected_parent_version: int,
        candidate_id: str,
    ) -> dict[str, Any]:
        """Promote a retained child to the session parent without re-prefill."""
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(expected_parent_version, int) or isinstance(expected_parent_version, bool):
            raise ValueError("expected_parent_version must be an integer")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        response = self._post("/v1/reference-match/persistent/promote", {
            "session_id": session_id,
            "expected_parent_version": expected_parent_version,
            "candidate_id": candidate_id,
        })
        if (not isinstance(response.get("parent_version"), int)
                or isinstance(response.get("parent_version"), bool)):
            raise EngineError("persistent session promotion returned an invalid response")
        return response

    def reference_match_persistent_close(self, session_id: str) -> dict[str, Any]:
        """Close a persistent exact-reference session and release its KV lease."""
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        response = self._post(
            "/v1/reference-match/persistent/close", {"session_id": session_id}
        )
        if response.get("closed") is not True:
            raise EngineError("persistent session close returned an invalid response")
        return response

    def cancel(self, req_id: str) -> dict:
        """Cancel an active worker request by its worker request id."""
        return self._post("/cancel", {"req": req_id})
