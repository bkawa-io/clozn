"""The versioned, public hook/intervention vocabulary (docs/PRODUCT_ROADMAP.md §7 item 2, roadmap
Phase 4.2): every named native interception point clozn's Python server side can drive, with exact
semantics -- pure, static, model-free.  Nothing here calls the engine; ``GET /contracts/hooks``
(clozn/server/routes/contracts.py) just serves ``hook_vocabulary()`` verbatim.

Every claim is cited to the exact comment or validated error message it came from, read from:
  * engine/core/serve/routes_whitebox.cpp  -- the /score write/capture/attn_knockout/attn_capture
    request bodies, their validation, and their response shapes (primary source).
  * engine/core/include/clozn/model_ggml.hpp -- GgmlAdapter's WriteSpec, CaptureFrame, AttnKnockout,
    EngineCheckpoint, and the tap/capture-plane method contracts (primary source).
  * engine/core/serve/server_main.cpp -- ONLY for /v1/checkpoint, /v1/restore, /v1/branch, and
    GET /health's capabilities block. These routes were outside this task's originally-assigned
    reading list; they were read anyway (read-only) because describing checkpoint/branch/capability
    semantics from the header alone would have meant guessing at the actual wiring. Every claim
    sourced here says so explicitly.

Where the C++ leaves something ambiguous or this reading pass did not trace far enough to be sure,
the value is the literal string ``"UNSPECIFIED"`` (or an explanatory sentence containing it) rather
than a plausible-sounding guess -- see ``not_covered`` for the running list.

2026-07-28 addendum (branch feat/ffn-hook): added the previously-undocumented ``kqv_out-<il>``
(head_capture/head_write) hook -- fully implemented since the R5 head-units work but never entered
here -- and the new ``ffn_out-<il>`` (ffn_capture/ffn_write) hook, plus a top-level
``stream_semantics_summary`` making the OVERWRITES-vs-CONTRIBUTES distinction explicit across every
hook. This addendum's claims are additionally cited to live measurements (scripts/spike/
additive_writes_probe.py, scripts/spike/ffn_hook_probe.py, engine/core/tests/test_ffn_hook.cpp) where
the fact in question is a MEASURED behavior (composition, the head_write/ffn_write honesty fix) rather
than something a static code comment states outright -- each such citation says so.

Replay-class labels reuse ``clozn.experiments.stats.REPLAY_CLASSES`` verbatim (bit_identical_greedy /
re_prefilled / stochastic_sampled / unknown), per this roadmap item's explicit instruction, and are
kept distinct from clozn-client's OWN (coarser, 2-value) ``ReplayClass`` enum -- see
``replay_classes.distinct_from_client_replay_class`` below.
"""
from __future__ import annotations

from copy import deepcopy

from clozn.experiments.stats import REPLAY_CLASSES

SCHEMA = "clozn.hook_vocabulary.v1"

_L_OUT = {
    "name": "l_out-<il>",
    "tensor": (
        "the post-layer residual stream tensor at zero-based transformer layer <il> -- the ggml "
        "eval-callback tensor name GgmlAdapter prefix-matches during llama_decode (model_ggml.hpp: "
        "'the eval callback prefix-matching \"l_out-<il>\" across all layers in a single decode')."
    ),
    "read": {
        "endpoints": [
            "POST /harvest -- single tap, whole supplied text, one forward, ALL token rows at one layer",
            "POST /score with capture:{layers:[int], positions:[int]} -- multi-observer, rides one "
            "teacher-forced scoring forward (Phase 2.3 capture plane), specific token positions only",
        ],
        "mechanism": (
            "the eval callback intercepts the named tensor during llama_decode and copies it host-side; "
            "no source-graph patch, read-only."
        ),
        "layer_zero_sentinel": (
            "for the SINGLE-tap /harvest path only: tap_layer==0 is a SENTINEL meaning 'skip eval_cb "
            "capture; return the FINAL layer's hidden state via llama_get_embeddings_ith instead' -- it "
            "does NOT mean 'capture tensor l_out-0'. Quote (model_ggml.hpp): 'set_tap_layer(il): change "
            "which layer the cb_eval callback captures (0 = final via embeddings)'. /harvest's own comment: "
            "'0 / out-of-range => final-layer fallback'."
        ),
        "capture_plane_layer_range": (
            "(0, n_layer) -- open interval, BOTH endpoints excluded, i.e. valid layers are 1..n_layer-1. "
            "Quote (model_ggml.hpp): 'Layers outside (0, n_layer) are dropped (the l_out-<il> residual "
            "names exist only for mid layers).' Confirmed by the runtime 400 text (routes_whitebox.cpp): "
            "'capture layers all out of range (1..n_layer-1)'."
        ),
        "known_gap_last_layer": (
            "layer == n_layer-1 passes the range check but, inside a /score capture, commonly yields NO "
            "rows for an arbitrary whole-sequence position set: llama.cpp's inp_out_ids optimization at "
            "the last layer materializes ONLY the logit-producing rows (one row for a single-target "
            "/score). A request where NOTHING lands for ANY requested layer is an explicit 400, never a "
            "silently-empty object. Quote: 'the last layer (n_layer-1) materializes only the rows that "
            "produce logits ... so whole-sequence capture needs layer <= n_layer-2.' This also structurally "
            "bounds cross-position path patching: a source whose influence is re-imported only by "
            "final-layer attention cannot be captured by holding a destination column there."
        ),
        "pre_post_norm_position": (
            "UNSPECIFIED -- neither read file states whether l_out-<il> is captured before or after that "
            "layer's own final norm/residual-add relative to llama.cpp's internal graph node ordering; "
            "only the tensor's naming convention ('post-layer residual', 'the residual stream') and the "
            "callback's interception mechanics are stated."
        ),
        "write_capture_interaction": (
            "when a write and a capture target the SAME layer in one /score call, the captured row is the "
            "PRE-edit state, not the post-edit one. Quote: '(at the write layer itself the captured row is "
            "the PRE-edit state, same convention as the read tap)'."
        ),
        "response_shape": {
            "harvest": "{tokens:[piece...], layer:int, n_tokens:int, n_embd:int, activations:{dtype, shape, data}}",
            "score_capture": (
                "captured:{\"<layer>\":{\"<position>\":[n_embd floats], ...}, ...} + n_embd:int; "
                "capture_missing:[layer,...] lists layers that armed but yielded nothing (see known_gap_last_layer)"
            ),
        },
    },
    "write": {
        "endpoints": [
            "POST /score with write:{layer,positions,values} -- ONE spec, OR an array of specs = a JOINT "
            "multi-layer patch applied in a SINGLE forward (the circuit-tracer's 'all candidate nodes "
            "ablated simultaneously' arm), riding a teacher-forced scoring pass",
            "POST /state -- standalone baseline-then-write-then-observe loop, no continuation scoring. "
            "NOTE: /state's own route body was NOT read for this document (outside the assigned files); "
            "this entry is sourced only from engine/client/clozn_engine.py's EngineClient.write_state "
            "docstring and clozn-client's REPLACE_RESIDUAL OperationSpec, not verified against the C++ "
            "route directly -- treat its exact validation behavior as UNSPECIFIED here.",
        ],
        "layer_range": (
            "[1, n_layer) -- validated server-side error text (routes_whitebox.cpp): 'write rejected: "
            "layer must be in [1, n_layer) and values.size must equal positions.size * n_embd'. Layer 0 "
            "is REJECTED for writes (unlike attn_knockout's layer 0, which IS valid -- see kq_soft_max) "
            "because 0 is reserved as the read tap's final-layer-via-embeddings sentinel, not an "
            "addressable l_out-0 write target."
        ),
        "positions_range": "[0, n_total) where n_total = n_prompt + n_continuation for that /score call.",
        "values_shape": (
            "positions.size() * n_embd floats, POSITION-MAJOR (row i of the flattened buffer is "
            "positions[i]'s new residual row)."
        ),
        "mechanism": (
            "ggml_backend_tensor_set overwrite during llama_decode's eval callback -- no llama graph/"
            "source patch. Applied on EVERY subsequent forward until clear_write(). write_state() REPLACES "
            "the active write set (single-write semantics); add_write_state() APPENDS (the joint arm)."
        ),
        "propagation": (
            "the edit lands in the residual BEFORE later layers consume it, so it propagates forward "
            "through the remaining stack exactly like an unedited activation would -- an activation "
            "patch, not a final-logit hack."
        ),
    },
}

_KQV_OUT = {
    "name": "kqv_out-<il>",
    "tensor": (
        "the merged per-Q-head attention output at layer <il>, BEFORE the W_o projection (model_ggml.cpp: "
        "'kqv_out-<il>\" = merged per-Q-head attention output, PRE-W_o (verified: llama-graph.cpp names it "
        "before the wo matmul)'). Per-Q-head h occupies the contiguous rows [h*d_head, (h+1)*d_head) of ne0 "
        "-- GQA note (model_ggml.hpp): 'these are Q heads (kqv_out rows are per-Q-head even when KV heads "
        "are fewer)'. Emitted from ONE SHARED call site in llama-graph.cpp (not per-architecture like "
        "ffn_out below), so it is present for essentially every attention architecture this tree supports."
    ),
    "stream_semantics": (
        "CONTRIBUTES -- kqv_out is NOT the residual stream itself; it is later multiplied by W_o and ADDED "
        "into the residual (model_ggml.hpp: 'This tensor is NOT the residual stream -- it is later "
        "multiplied by W_o and ADDED into the residual'). Writes at TWO DIFFERENT layers COMPOSE (measured "
        "live, scripts/spike/additive_writes_probe.py, 2026-07-27: an L1-and-L2 head_write is bit-for-bit "
        "distinct from either single-layer arm) -- the opposite of l_out write's OVERWRITE semantics, where "
        "a write at L2 alone is bit-identical to writes at L1-and-L2 together (same spike, Assumption A). "
        "This is the single most important fact for anyone composing a multi-site intervention: l_out write "
        "silently DISCARDS an earlier-layer edit once a later-layer edit lands on the same position; "
        "kqv_out/ffn_out writes STACK. See hook_vocabulary()['stream_semantics_summary'] for the full table."
    ),
    "materialization_constraint": (
        "materialized at every position (a mid-graph tensor like l_out -- no inp_out_ids blocker) and "
        "exists with flash attention ON, UNLIKE kq_soft_max (model_ggml.hpp: 'exists with flash attention "
        "ON (unlike kq_soft_max)'). No --no-flash-attn requirement and no dedicated GET /health capability "
        "flag -- unlike kq_soft_max's capabilities.attn_knockout, this hook is unconditionally attempted "
        "and its success is reported per-request (head_dims/head_write_landed), not advertised statically."
    ),
    "head_capture": {
        "endpoint": "POST /score with head_capture: {layers:[int], positions:[int], rows?:bool}",
        "semantics": (
            "READ-ONLY: L2 norm of each head's [d_head] slice at the requested positions (the cheap "
            "screening signal, model_ggml.hpp: '32x smaller than the slices'); rows:true additionally "
            "returns the full [ne0] merged row per position (all heads' slices concatenated; 'the client "
            "slices/means them locally'). Capture runs BEFORE any same-layer write in the same forward, so "
            "a same-pass write never contaminates the recorded norms (model_ggml.cpp: 'Capture ... runs "
            "BEFORE writes, so a same-pass intervention never contaminates the recorded norms -- the "
            "convention every hook here follows')."
        ),
        "response_shape": (
            "head_norms: {\"<layer>\": {\"<position>\": [n_head floats]}}; head_rows (if rows:true): "
            "{\"<layer>\": {\"<position>\": [ne0 floats]}}; head_dims: {ne0, n_head, d_head} always present "
            "alongside either -- the shape-probe result (see d_head_probe below)."
        ),
        "position_range": "[0, n_total) -- validated route-side 400: 'head_capture position out of range'.",
    },
    "head_write": {
        "endpoint": (
            "POST /score with head_write: {layer, head, positions:[int], values:[float]} or an ARRAY of "
            "such specs (a joint multi-layer/multi-head patch in one forward, mirroring residual `write`)."
        ),
        "semantics": (
            "overwrite head `head`'s [d_head] slice at each requested position, BEFORE W_o consumes it "
            "(model_ggml.hpp: 'overwrite head h's slice at the requested positions before W_o consumes it'). "
            "values = positions.size()*d_head floats, position-major."
        ),
        "layer_range": "[0, n_layer) -- runtime-validated 400 (route, 2026-07-28): 'head_write layer out of range [0, n_layer)'.",
        "head_range": "[0, n_head) -- runtime-validated 400 (route, 2026-07-28): 'head_write head out of range [0, n_head)'.",
        "d_head_probe": (
            "d_head = ne0 / n_head when divisible, else 0, PROBED from the live tensor (not knowable "
            "statically -- unlike ffn_out's n_embd width below). model_ggml.cpp: 'd_head is derived "
            "ne0/n_head_ and PROBED (head_dims_): an architecture where that division fails records "
            "d_head=0 and applies nothing'. A caller must read head_dims (from a head_capture response, or "
            "any head hook's response) to learn the required values length before constructing a write."
        ),
        "honesty_fix_2026_07_28": (
            "PRIOR BEHAVIOR (measured live, scripts/spike/additive_writes_probe.py, now a documented "
            "regression case, engine/core/tests/test_ffn_hook.cpp): a head_write with values.size() != "
            "positions.size()*d_head (e.g. d_head-1 floats), or an out-of-range layer (n_layer+50), "
            "returned HTTP 200 with {\"head_write_applied\": true, \"n_head_writes\": 1} while writing "
            "NOTHING -- output was bit-identical to baseline. Root cause: the route validated only "
            "structural shape (layer>=0, head>=0, positions/values non-empty); the real bounds/size checks "
            "lived inside GgmlAdapter::set_head_writes/eval_cb, which silently dropped invalid specs with "
            "no record of what happened, and the route set head_write_applied from the REQUESTED vector, "
            "not from what actually landed. FIX: (1) layer/head bounds are now checked route-side, fail-"
            "closed, BEFORE any forward runs (both are known statically from ad.n_layer()/ad.n_head()); "
            "(2) GgmlAdapter now tracks per-spec landing at runtime (head_write_landed(), one bool per "
            "spec, true only if EVERY requested position was actually written) for the class of bug only "
            "detectable once d_head is known (the values-size mismatch); (3) the route refuses the WHOLE "
            "request (400, discarding the already-computed scores) if any requested spec did not land -- "
            "so head_write_applied:true in a response is now unconditionally true, never a residual claim "
            "left over from the request shape. Verified live: HTTP 400 for both original repro cases "
            "(scripts/spike/ffn_hook_probe.py, 2026-07-28) -- 'head_write layer out of range [0, n_layer)' "
            "and 'head_write[0] (layer=..., head=...) did not land: values.size() did not match the probed "
            "d_head ...'."
        ),
        "position_range": "[0, n_total) -- validated route-side 400: 'head_write position out of range'.",
    },
    "arms_exclusion": (
        "head_capture/head_write cannot be combined with `arms` (route 400: 'head_capture/head_write "
        "cannot be combined with arms (unvalidated under multi-seq batching)') -- the batched multi-arm "
        "tensor layout under kqv_out has not been proven correct, same rule as every other per-position hook."
    ),
}

_FFN_OUT = {
    "name": "ffn_out-<il>",
    "tensor": (
        "the feed-forward (MLP) block's output at layer <il>, BEFORE it is added into the residual stream "
        "(model_ggml.hpp: verified against models/qwen2.cpp: 'cb(cur, \"ffn_out\", il); cur = ggml_add(ctx0, "
        "cur, ffn_inp);' -- the residual add happens AFTER this callback fires). Full n_embd width like "
        "l_out (no per-head split -- the MLP has no head structure to slice)."
    ),
    "stream_semantics": (
        "CONTRIBUTES -- the exact additive-not-overwrite analogue of kqv_out, for the OTHER half of a "
        "transformer layer (attention vs feed-forward). Writes at two different layers COMPOSE, and an "
        "ffn_write composes with a head_write at a different layer too (both are additive into the SAME "
        "residual stream). MEASURED LIVE (2026-07-28): engine/core/tests/test_ffn_hook.cpp (adapter-level, "
        "Qwen2.5-0.5B) -- an L1-only ffn_write, an L2-only ffn_write, and BOTH together produced three "
        "MUTUALLY DIFFERENT logit vectors (both-vs-L1-only L2=270.7, both-vs-L2-only L2=79.9, both nonzero); "
        "an ffn_write(L1) + head_write(L2) combination differed from both single-surface arms (L2=65.4 and "
        "L2=106.8). Reproduced at the HTTP /score level in scripts/spike/ffn_hook_probe.py, same model: "
        "baseline logprob=-0.99621, L1-only=-0.93943, L2-only=-0.89557, BOTH=-0.79551 (both distinct from "
        "baseline and from either single-site arm); the ffn_write+head_write combo response carried "
        "ffn_write_applied=true AND head_write_applied=true, and its logprob (-0.89875) differed from both "
        "the ffn-only and head-only arms. See hook_vocabulary()['stream_semantics_summary'] for the table."
    ),
    "architecture_coverage": (
        "UNLIKE kqv_out (one shared llama-graph.cpp call site, present for essentially every attention "
        "architecture) ffn_out is emitted PER-ARCHITECTURE, directly inside each model's own .cpp file "
        "under engine/core/third_party/llama.cpp/src/models/ -- so its presence is NOT universal the way "
        "kqv_out/l_out are. VERIFIED PRESENT, by direct file read, for every architecture this repo's "
        "~/.clozn/models set actually uses: qwen2.cpp (Qwen2.5), llama.cpp (Llama-3.x), gemma4.cpp "
        "(gemma-4), qwen35.cpp (Qwen3.5), dream.cpp (dream-v0), and deepseek2.cpp (Ministral/mistral4 -- "
        "llama_model_mistral4 inherits llama_model_deepseek2's graph verbatim, models/models.h:1108; "
        "deepseek2's cb(cur,\"ffn_out\",il) fires on BOTH its dense-layer branch, models/deepseek2.cpp:384, "
        "and its MoE branch, :413, after moe_out+ffn_shexp are combined -- so a MoE-vs-dense layer split "
        "inside one model still names the tensor uniformly). VERIFIED ABSENT (repo-wide grep for the "
        "literal `cb(cur, \"ffn_out\", il)`, missing from 35 of this tree's 133 model files) for: "
        "state-space/no-conventional-FFN architectures (mamba*, rwkv*), several MoE variants that route "
        "through a differently-named combine step (qwen2moe, qwen3moe, olmoe, phimoe, openai-moe, "
        "smallthinker), and several encoder-only bert variants (bitnet, modern-bert, neo-bert, "
        "nomic-bert*, jina-bert*). UNSPECIFIED for the ~90 remaining llama.cpp-supported architectures in "
        "this tree neither individually confirmed present nor absent -- this is exactly why ffn_write/"
        "ffn_capture track LANDED state per request (below) rather than assuming success from a static "
        "architecture allowlist."
    ),
    "ffn_capture": {
        "endpoint": "POST /score with ffn_capture: {layers:[int], positions:[int]}",
        "semantics": (
            "READ-ONLY: the raw [n_embd] MLP-contribution row at each requested (layer, position) pair -- "
            "no norm summary (unlike head_capture there is no per-head split to compress away); same "
            "response shape as residual `capture`, not head_capture's norms+rows split."
        ),
        "response_shape": (
            "ffn_captured: {\"<layer>\": {\"<position>\": [n_embd floats]}} + n_embd:int; "
            "ffn_capture_missing:[layer,...] lists layers that armed (passed the [0,n_layer) range check) "
            "but yielded nothing -- the architecture-coverage gap above made real, per architecture that "
            "does not name the tensor \"ffn_out\". A request where NOTHING lands for ANY requested layer is "
            "an explicit 400 (route: 'ffn_capture produced no rows for any requested layer'), mirroring "
            "residual `capture`'s own known_gap_last_layer precedent -- never a silently-empty object."
        ),
        "layer_range": (
            "[0, n_layer) -- EVERY layer has its own FFN block (unlike l_out's write range, ffn_out has no "
            "layer-0-reserved sentinel; layer 0 IS a valid ffn_capture/ffn_write target)."
        ),
        "position_range": "[0, n_total) -- validated route-side 400: 'ffn_capture position out of range'.",
    },
    "ffn_write": {
        "endpoint": (
            "POST /score with ffn_write: {layer, positions:[int], values:[float]} or an ARRAY of such "
            "specs -- same shape convention as residual `write` (no `head` field; ffn_out has no head "
            "structure), values = positions.size()*n_embd floats, position-major."
        ),
        "semantics": (
            "overwrite the MLP contribution at requested positions BEFORE the residual add consumes it. "
            "COMPOSES with writes at other layers and with head_write (see stream_semantics above) -- never "
            "overwrites the whole downstream residual the way l_out write does."
        ),
        "layer_range": (
            "[0, n_layer) -- fully STATIC validation (model_ggml.hpp: 'unlike head_write's d_head ... ffn_out "
            "is always n_embd-wide ... so BOTH the layer range and the values shape are known STATICALLY'). "
            "add_ffn_write returns false -- and the route refuses the request with a 400 BEFORE any forward "
            "runs -- for layer out of [0, n_layer) or values.size() != positions.size()*n_embd. This is "
            "STRICTLY more fail-closed than head_write's d_head case, which can only be validated once the "
            "tensor has actually been probed at runtime."
        ),
        "landed_tracking": (
            "even with the shape statically valid, LANDING is still runtime-dependent: if this model's "
            "architecture never names the tensor \"ffn_out\" (see architecture_coverage above), eval_cb is "
            "simply never invoked for it. GgmlAdapter::ffn_write_landed() reports one bool per spec, true "
            "only if the tensor fired for that layer AND every requested position was written; the route "
            "refuses the WHOLE request (400, discarding the already-computed scores) if any requested "
            "ffn_write spec did not land -- so ffn_write_applied:true in a response is unconditionally true, "
            "the same honesty guarantee head_write_applied now carries (see head_write's "
            "honesty_fix_2026_07_28 above -- ffn_write was designed with this fix built in from the start, "
            "never shipped with the silent-drop bug)."
        ),
        "position_range": "[0, n_total) -- validated route-side 400: 'ffn_write position out of range'.",
    },
    "arms_exclusion": (
        "ffn_capture/ffn_write cannot be combined with `arms` (route 400: 'ffn_capture/ffn_write cannot be "
        "combined with arms (unvalidated under multi-seq batching)') -- same rule as every other "
        "per-position hook (head_capture/head_write, attn_knockout, attn_capture, residual capture)."
    ),
}

_STREAM_SEMANTICS_SUMMARY = {
    "why_this_exists": (
        "The single most important fact for anyone building a multi-site intervention: whether a hook's "
        "write OVERWRITES the residual stream (a later write on the same position DISCARDS an earlier one, "
        "even at a different layer) or CONTRIBUTES/ADDS into it (writes at different layers COMPOSE -- the "
        "combined effect is distinct from either alone). Getting this backwards silently produces either a "
        "'no effect' false negative (expecting composition from an overwrite hook) or a misattributed "
        "causal claim (expecting a clean intervention from an additive hook that actually stacked with "
        "something else already written). Every row below is measured, not inferred -- see each hook's own "
        "entry above for the exact measurement citation."
    ),
    "table": [
        {"hook": "l_out-<il>", "wire_fields": "write, capture", "semantics": "OVERWRITES",
         "composes_across_layers": False,
         "note": "a write at L2 alone is bit-identical to writes at L1-and-L2 together on the same "
                 "position (scripts/spike/additive_writes_probe.py, Assumption A, both_eq_L2_only=true)."},
        {"hook": "kqv_out-<il>", "wire_fields": "head_capture, head_write", "semantics": "CONTRIBUTES (additive)",
         "composes_across_layers": True,
         "note": "ADDED into the residual via W_o after this hook's write point; two-layer head_write is "
                 "bit-distinct from either single-layer arm (scripts/spike/additive_writes_probe.py, "
                 "Assumption B)."},
        {"hook": "ffn_out-<il>", "wire_fields": "ffn_capture, ffn_write", "semantics": "CONTRIBUTES (additive)",
         "composes_across_layers": True,
         "note": "ADDED into the residual right after this hook's write point; two-layer ffn_write, and an "
                 "ffn_write+head_write cross-hook combination, are both measured distinct from every "
                 "single-site arm (engine/core/tests/test_ffn_hook.cpp; scripts/spike/ffn_hook_probe.py)."},
        {"hook": "kq_soft_max-<il>", "wire_fields": "attn_knockout (write), attn_capture (read)",
         "semantics": "NOT A RESIDUAL WRITE",
         "composes_across_layers": "UNSPECIFIED",
         "note": "zeroes attention EDGES (query-key pairs), not a residual/contribution value -- "
                 "composition across layers was not measured for this hook; N/A in the OVERWRITES-vs-"
                 "CONTRIBUTES sense the other three rows use."},
    ],
    "flash_attn_requirement": {
        "l_out": "none",
        "kqv_out": "none (exists with flash attention ON -- model_ggml.hpp)",
        "ffn_out": "none",
        "kq_soft_max": "--no-flash-attn REQUIRED (flash attention fuses the softmax; the tensor never materializes)",
    },
}

_KQ_SOFT_MAX = {
    "name": "kq_soft_max-<il>",
    "tensor": (
        "post-softmax attention weights at layer <il>, shape [n_kv, n_tokens, n_head] (model_ggml.hpp: "
        "'kq_soft_max-<il> (shape [n_kv, n_tokens, n_head])')."
    ),
    "materialization_constraint": (
        "requires the engine PROCESS to have been started with --no-flash-attn (flash_attn=false at "
        "GgmlAdapter construction). With flash attention on, the softmax is fused inside the kernel and "
        "this tensor never materializes. This is a SERVER-STARTUP-TIME setting, not a per-request toggle "
        "-- knockout_available() (== !flash_attn_) is the one gate both operations below share. Quote: "
        "'knockout_available() reports this, so a caller gets a clean refusal instead of a "
        "silently-ignored intervention.'"
    ),
    "capability_flag": (
        "GET /health.capabilities.attn_knockout = !flash_attn (server_main.cpp). The SAME flag gates BOTH "
        "attn_knockout AND attn_capture below (both call the identical knockout_available() check "
        "server-side) -- there is no separate advertised capability name for attn_capture despite the "
        "different operation name. A caller must check this before either request or receive the "
        "explicit 400 refusal text quoted below."
    ),
    "knockout": {
        "endpoint": (
            "POST /score with attn_knockout: {layer, head?, queries:[int], keys:[int], renormalize?} "
            "or an array of such specs (a joint multi-edge cut in one forward)"
        ),
        "semantics": (
            "zero A[head, query, key] for every listed query x key pair at `layer`, BEFORE kqv consumes "
            "the weights -- stops `query` from reading `key`. The primitive residual patching could not "
            "provide: patching a destination site leaves the SOURCE free to re-supply the information "
            "downstream (measured 0.0% routed at every depth for cross-position edges); cutting the edge "
            "itself sidesteps that."
        ),
        "layer_range": (
            "[0, n_layer) -- runtime check: 'attn_knockout layer out of range [0, n_layer)'. Layer 0 IS "
            "valid here (unlike the l_out write range above) since kq_soft_max-0 is an ordinary "
            "first-layer attention tensor with no sentinel meaning attached."
        ),
        "head_range": (
            "head=-1 (the field's default) means every head at the layer; otherwise head must be in "
            "[0, n_head) -- runtime check: 'attn_knockout head out of range [0, n_head)'. Values below "
            "-1 are UNSPECIFIED (not explicitly validated in the read code)."
        ),
        "renormalize_default_discrepancy": (
            "the C++ struct's own default is FALSE (model_ggml.hpp: 'bool renormalize = false; // rescale "
            "the surviving row to sum 1 (else mass is dropped)'), and the route reproduces that default "
            "when the wire field is omitted ('renormalize', false). clozn-client's AttentionKnockout "
            "dataclass instead defaults renormalize to TRUE and always serializes it explicitly -- so "
            "this discrepancy only bites a hand-built request that omits the key entirely."
        ),
        "renormalized_vs_not_are_distinct": (
            "a renormalized cut (surviving weights rescaled to sum 1) and a non-renormalized cut (mass "
            "simply dropped) are DIFFERENT interventions and must never be conflated or averaged together."
        ),
        "client_head_gap": (
            "the C++ struct supports a per-layer `head` selector; clozn-client's v1 AttentionKnockout wire "
            "form (models.py) has NO `head` field at all -- every client-issued knockout implicitly "
            "targets every head at the layer (the server's own head=-1 default applies). Per-head "
            "knockout exists in the engine but has no v1 client surface yet."
        ),
    },
    "attn_capture": {
        "endpoint": "POST /score with attn_capture: {query: <board position, int >= 0>}",
        "semantics": (
            "READ-ONLY sibling of knockout, same materialization constraint: returns the requested query "
            "position's post-softmax attention row, HEAD-AVERAGED, at every layer the decode touches -- "
            "the correlational 'attention heatmap' the causal knockout ranking is compared against (the "
            "R1 attention-vs-causal head-to-head lane)."
        ),
        "response_shape": (
            "attn_rows: {\"<layer>\": [n_kv floats], ...} -- an object keyed by layer as a STRING; a "
            "layer is simply ABSENT (never a fabricated zero row) if the query position fell outside the "
            "decoded segment at every layer. Per-head rows are NOT exposed ('deliberately NOT exposed "
            "until a product question needs it (32x the payload for no current consumer)')."
        ),
        "query_validation": (
            "must be >= 0 (400 otherwise, 'attn_capture must be {query: <position >= 0>}'); no explicit "
            "upper-bound check against the sequence length in the read code -- an out-of-range query "
            "silently yields an empty attn_rows object rather than an error."
        ),
    },
}

_CHECKPOINT_BRANCH = {
    "checkpoint": {
        "endpoint": "POST /v1/checkpoint {tokens:[int], n_past?:int} -> {checkpoint_id, n_past, n_tokens, size_bytes}",
        "adapter_primitive": (
            "GgmlAdapter::save_checkpoint(tokens, n_past) serializes the KV cache for sequence 0 "
            "(llama_state_seq_get_data) plus the full token sequence, n_past, and the causal flag into "
            "an EngineCheckpoint. Held server-side in a bounded in-memory map keyed by a generated "
            "checkpoint_id (server_main.cpp)."
        ),
    },
    "restore": {
        "endpoint": "POST /v1/restore {checkpoint_id, max_tokens?, temperature?, ...} -> {text, tokens, finish_reason}",
        "mechanism_finding": (
            "this route does NOT call GgmlAdapter::load_checkpoint. Its own comment: 'First slice: "
            "re-prefill from the saved tokens (correct, no KV-blob restore yet). Phase 2 optimization: "
            "load_checkpoint + resume without re-prefill.' It resumes by re-running generation over the "
            "checkpoint's saved token list from a clean context -- correct, but it pays a full re-decode "
            "rather than the adapter's faster KV-blob restore path. Matches "
            "docs/PRODUCT_ROADMAP.md's Engine debt tail: 'KV-blob fast restore (restore currently "
            "re-prefills from saved tokens -- correct, just slower).'"
        ),
        "adapter_correctness_bar": (
            "GgmlAdapter's OWN load_checkpoint primitive (not currently exercised by this route) is "
            "documented as proven for 'greedy suffix after save->restore' -- i.e. bit-identical GREEDY "
            "continuation is the correctness bar, never claimed for a sampled/stochastic continuation, "
            "since EngineCheckpoint itself carries no sampler/RNG state (only kv_data, tokens, n_past, "
            "causal)."
        ),
    },
    "branch": {
        "endpoint": "POST /v1/branch {checkpoint_id, n<=16, max_tokens?, temperature?, seed?} -> {branches:[{index,text,finish_reason,generated_tokens}]}",
        "mechanism_finding": (
            "calls generate_ar_branched(...), a function this reading pass did not trace. Whether it "
            "internally uses GgmlAdapter::branch_kv/ar_forward_batch (the Phase 2.2 batched-decode "
            "primitives model_ggml.hpp documents: 'Decode one token per sequence in a single "
            "llama_decode call') or falls back to N independent re-prefills is UNSPECIFIED here."
        ),
    },
    "sampling_defaults": (
        "server_shared.hpp's sample_from(): temperature defaults to 0.0 (greedy), rep_penalty 1.0, "
        "top_k 0, top_p 1.0. Quote: 'Defaults (T=0, penalty=1) keep greedy decoding, so omitting them is "
        "byte-identical to before.' Both /v1/restore and /v1/branch build their sampler from this same "
        "helper, so an unmodified caller gets greedy resumption by default."
    ),
}

_REPLAY_CLASSES_SECTION = {
    "vocabulary": list(REPLAY_CLASSES),
    "source": (
        "clozn.experiments.stats.REPLAY_CLASSES / replay_class_for_meta -- reused verbatim here, never "
        "reinvented, per this roadmap item's explicit instruction."
    ),
    "rule_summary": (
        "a recorded run's meta.forced_rescore=True (no new tokens decoded, an existing continuation "
        "re-scored) -> 're_prefilled'; meta.sampler_mode=='greedy' (equivalently temperature==0.0) -> "
        "'bit_identical_greedy'; meta.sampler_mode=='sample' -> 'stochastic_sampled'; missing/malformed "
        "meta or an unrecognized sampler_mode -> 'unknown' (never defaults to the strongest claim)."
    ),
    "distinct_from_client_replay_class": (
        "clozn-client's OWN ReplayClass enum (request_replay / re_prefilled) is a DIFFERENT, coarser, "
        "2-value vocabulary describing whether an OPERATION replays a specific recorded request "
        "(request_replay: score.teacher_forced, intervention.attention_knockout) versus needs a fresh "
        "prefill of supplied text with no continuity to an original recorded run (re_prefilled: "
        "capture.residual.layer_output, intervention.residual.replace_rows). This document's "
        "operation_classes below use the 4-value clozn.experiments.stats vocabulary instead, per this "
        "task's explicit instruction -- the two vocabularies name different things and must not be "
        "conflated."
    ),
    "operation_classes": {
        "score.teacher_forced": (
            "re_prefilled -- a forced rescore of an already-fixed continuation; no new tokens are "
            "decoded at all (stats.py: 'a teacher-forced RE-SCORE of an already-fixed continuation ... "
            "-- marked by the caller setting meta[\"forced_rescore\"] = True')."
        ),
        "intervention.attention_knockout": (
            "re_prefilled -- same /score forced-rescore mechanics, with attention edges zeroed during "
            "the one forward."
        ),
        "capture.residual.layer_output": (
            "re_prefilled -- whether read via a standalone /harvest prefill or riding a /score capture, "
            "it is a fresh forward over supplied tokens, never a resumed generation."
        ),
        "intervention.residual.replace_rows": (
            "re_prefilled -- same reasoning as capture; /state's own baseline-then-write forwards are "
            "also fresh prefills of the supplied text, not resumed generation."
        ),
        "checkpoint_restore_or_branch_default_greedy": (
            "bit_identical_greedy -- /v1/restore and /v1/branch default to temperature=0.0 (see "
            "sampling_defaults); replaying the same checkpoint tokens under an unchanged build/quant "
            "reproduces the same greedy continuation deterministically. This route's actual mechanism "
            "(re-prefill, per checkpoint_branch.restore.mechanism_finding) reaches that SAME result more "
            "slowly than the adapter's not-yet-wired KV-blob restore would -- the replay-class label is "
            "about outcome determinism, not mechanism speed."
        ),
        "checkpoint_restore_or_branch_sampled": (
            "stochastic_sampled -- once a caller supplies temperature>0 (or narrows top_k/top_p), per "
            "stats.py: 'a recorded seed is provenance metadata only -- this codebase never demonstrates "
            "that replaying the same nominal seed reproduces the same sample.'"
        ),
    },
}

_NOT_COVERED = [
    "the exact pre-norm/post-norm position of l_out-<il> relative to llama.cpp's own graph node "
    "ordering (see l_out.read.pre_post_norm_position)",
    "POST /state's own C++ route body (not read for this document; sourced only from the Python "
    "engine client's docstring and clozn-client's REPLACE_RESIDUAL OperationSpec)",
    "whether /v1/branch's generate_ar_branched actually exercises GgmlAdapter's batched-KV branch "
    "primitives (branch_kv/ar_forward_batch) or falls back to independent re-prefills",
    "per-head attention knockout on the wire (the C++ struct supports it; clozn-client's v1 "
    "AttentionKnockout does not expose a head field)",
    "steering (POST /score's steer/steer_vec) -- a DIFFERENT mechanism (a llama control vector added "
    "over a layer range), not part of the l_out/kq_soft_max eval-callback tap/write/capture plane this "
    "vocabulary documents, and explicitly noted by clozn-client as having 'no stable v1 hook-contract "
    "identifier'",
    "ffn_out architecture coverage beyond the models actually present in ~/.clozn/models: verified "
    "PRESENT for qwen2/llama/gemma4/qwen35/deepseek2(mistral4)/dream and verified ABSENT for named "
    "families (mamba*, rwkv*, qwen2moe/qwen3moe/olmoe/phimoe/openai-moe/smallthinker, several bert "
    "variants) by a repo-wide grep; the other ~90 llama.cpp-supported architectures in this tree were "
    "NOT individually checked either way (see _FFN_OUT.architecture_coverage)",
]


def hook_vocabulary() -> dict:
    """The complete, versioned ``clozn.hook_vocabulary.v1`` document. Returns a fresh deep copy every
    call so a caller can never mutate the module's own constant."""
    return deepcopy({
        "schema": SCHEMA,
        "scope": (
            "Every native interception point the Python gateway can drive today via POST /score, "
            "POST /harvest(/layers), POST /state, and the /v1/checkpoint|restore|branch family -- named "
            "with exact semantics, not a general hook-authoring API. New entries belong here only once "
            "their native wire contract is qualified, mirroring clozn-client's own "
            "InterventionContract.scope discipline."
        ),
        "sources_read": [
            "engine/core/serve/routes_whitebox.cpp (primary -- /score, /harvest, /harvest/layers)",
            "engine/core/include/clozn/model_ggml.hpp (primary -- GgmlAdapter method contracts)",
            "engine/core/src/model_ggml.cpp (primary -- eval_cb dispatch, the kqv_out/ffn_out honesty "
            "fix, 2026-07-28)",
            "engine/core/serve/server_main.cpp (secondary, read-only, for accuracy only -- "
            "/v1/checkpoint, /v1/restore, /v1/branch, GET /health.capabilities)",
            "engine/core/third_party/llama.cpp/src/models/*.cpp + llama-graph.cpp (secondary -- "
            "confirming kqv_out's single shared call site vs ffn_out's per-architecture call sites, "
            "including the repo-wide presence/absence grep for \"ffn_out\")",
            "scripts/spike/additive_writes_probe.py + scripts/spike/ffn_hook_probe.py (secondary -- the "
            "LIVE measurements backing every composition/honesty-fix claim in this document; never "
            "guessed from reading code alone)",
            "engine/core/tests/test_ffn_hook.cpp (secondary -- the live regression coverage for the "
            "2026-07-28 honesty fix and ffn_out composition, run against a real GGUF)",
            "clozn/experiments/stats.py (replay-class vocabulary, reused not reinvented)",
        ],
        "gate": {
            "ar_mode_required": (
                "POST /score, /harvest, /harvest/layers, /v1/checkpoint, /v1/restore, and /v1/branch all "
                "require the loaded model to be autoregressive; GET /health.mode reports "
                "\"autoregressive\" or \"diffusion\". /score's own 400 text: 'score requires an "
                "autoregressive model'."
            ),
        },
        "hooks": [_L_OUT, _KQ_SOFT_MAX, _KQV_OUT, _FFN_OUT],
        # The single most load-bearing fact in this document for multi-site interventions -- see
        # _STREAM_SEMANTICS_SUMMARY's own why_this_exists. Kept as a TOP-LEVEL field (not buried inside
        # one hook's entry) precisely so it cannot be missed by a reader skimming the document shape.
        "stream_semantics_summary": _STREAM_SEMANTICS_SUMMARY,
        "checkpoint_branch": _CHECKPOINT_BRANCH,
        "replay_classes": _REPLAY_CLASSES_SECTION,
        "not_covered": _NOT_COVERED,
    })
