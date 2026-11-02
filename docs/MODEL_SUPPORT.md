# Model support

v1 targets instruction-tuned, autoregressive text GGUFs with embedded chat templates. The recommended
first model is available through `clozn pull qwen3.5`; `qwen-0.5b` is useful for a smaller installation
smoke test.

Core support requires:

- the worker loads the GGUF and reports stable identity;
- template rendering and tokenization succeed;
- streaming generation records a durable run;
- scalar recorded-answer scoring works; and
- the complete Context Effects + Lean Context live smoke passes.

Other GGUFs may run but are not automatically qualified. Support does not imply multimodal input,
adapters, tools, structured output, J-lens, SAE, steering, or multiple resident models.
