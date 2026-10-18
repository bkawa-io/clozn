# Real-runtime acceptance

The live gate attaches to an already running v1 gateway and exercises both retained analyses with a
real GGUF.

```sh
python -m clozn serve /absolute/path/to/model.gguf --port 8080
python scripts/smoke/v1_acceptance.py \
  --base-url http://127.0.0.1:8080 \
  --job-timeout 180
```

The harness verifies:

1. Chat Completions creates a durable parent run with canonical context sources.
2. Context Effects publishes measured evidence for the same source identities.
3. Minimal Context reaches a terminal result under its fixed budget.
4. The winner exactly preserves answer and termination while reducing rendered prompt tokens.
5. The result can be listed and reopened.
6. Saving creates an immutable child and a repeated save returns the same child.
7. Parent and child are readable and compare as a valid reduction pair.

Run the gateway against an isolated `CLOZN_DATA_ROOT`, stop it, restart it with the same root, and reopen
the parent, result, and child to complete the persistence check. Browser interaction and accessibility
remain separate release checks.
