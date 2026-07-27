# engine/ — the runtime ("clozn")

The performant local model runtime. Runs real models on ggml/llama.cpp (diffusion ·
autoregressive · later recurrent), **emits the state-stream**, **applies steers**, and hosts
the interp primitives that must scale. This is what used to be the `clozn` repo.

- `core/` — the C++/ggml runtime: scheduler, the L0 ggml adapter, the white-box server + viz,
  the CLI tools. The daily driver.
- `lab/` — the Python reference scheduler + model adapters + golden fixtures. The correctness
  oracle the C++ core is validated against, and the CPU/iteration path.

The engine never owns product opinions — it exposes the state-stream and the hooks; the
studio (the [`clozn/`](../clozn) package + [`studio/`](../studio)) owns all view/steer/memory
surface. They meet only at the [protocol](../protocol).
