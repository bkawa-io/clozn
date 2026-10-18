# Model support

The initial qualification target is one small instruct GGUF on macOS Apple Silicon, using the model's
own embedded chat template and one autoregressive worker configuration.

The current reference row is recorded in [V1_LIVE_QUALIFICATION.md](V1_LIVE_QUALIFICATION.md). Other
autoregressive GGUFs may run, but they are not covered by that reference evidence until the full live
journey passes with their exact model, tokenizer, template, engine, backend, and decoding identity.

Core support means:

- the worker loads the GGUF and reports stable identity;
- template rendering and tokenization succeed;
- streaming generation records a durable run;
- scalar recorded-answer scoring works; and
- exact-reference matching can verify a reduced prompt.

It does not imply support for multimodal input, adapters, tools, structured output, J-lens, SAE,
steering, or multiple resident models.
