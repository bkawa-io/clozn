# Real-runtime v1 smoke

Start a gateway with a local GGUF, then run:

```sh
python3 scripts/smoke/v1_acceptance.py \
  --base-url http://127.0.0.1:8080 \
  --job-timeout 300
```

The harness proves one end-to-end product journey:

1. a fresh text completion is durably recorded;
2. Context Effects measures the canonical source universe;
3. Lean Context uses the same source identities and finds a tested smaller prompt;
4. the retained and removed sets form an exact partition; and
5. fresh full-context and lean-context answers are generated under the same deterministic contract.

The GitHub real-runtime workflow repeats the journey on the pinned reference model, restarts the
gateway, and verifies that the parent run and Context Effects measurement reopen from durable storage.
Lean Context job state is intentionally process-local.
