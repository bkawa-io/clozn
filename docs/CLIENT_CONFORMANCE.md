# Released-client conformance

**Report date:** 2026-07-21
**Scope:** Phase 2.7 text-generation interoperability plus the Phase 2.8 structured-I/O gateway contract.
This report is intentionally narrower than "OpenAI compatible" or "Ollama compatible" as a general claim.

## Connecting a client

Clozn is the debugger; other applications own their own configuration. Clozn does not edit another
application's config file or `.env` -- it exposes an OpenAI-compatible endpoint and an
Ollama-compatible endpoint and documents the values to set. Start Clozn first:

```bash
clozn serve MODEL --port 8080
```

Then configure your application with these values (all default placeholders; any non-empty API key
is accepted since Clozn runs locally with no external auth):

```text
Base URL:  http://127.0.0.1:8080/v1
API key:   local-clozn
Model:     clozn-local
```

**Aider** -- environment variables, or `~/.aider.conf.yml`:

```bash
export OPENAI_API_BASE=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=local-clozn
aider --model openai/clozn-local
```

**Open WebUI** -- Admin Settings -> Connections -> OpenAI API, or the equivalent environment
variables on the Open WebUI server:

```text
OPENAI_API_BASE_URLS=http://127.0.0.1:8080/v1
OPENAI_API_KEYS=local-clozn
```

**A generic OpenAI SDK client** (Python, JS, ...):

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local-clozn")
```

**An Ollama SDK client** (Python, JS): point it at Clozn's Ollama-compatible base URL instead of a
real Ollama install:

```python
from ollama import Client
client = Client(host="http://127.0.0.1:8080")
```

None of the above writes anything to disk on Clozn's behalf -- the values are yours to set the same
way you would for any other OpenAI- or Ollama-compatible endpoint.

## Evidence labels

- **Released-client pass** means that the named, released client was actually executed against
  Clozn's real HTTP handler with a deterministic model-free substrate. It does not mean that a
  hand-written request resembling the client passed.
- **Executable test** means that a pinned released client is invoked by an automated test. A test
  that skips because its optional client is absent is not evidence of a pass.
- **Scheduled** means an executable job is committed but no successful job result is recorded in
  this report yet.
- **Contract pass** means only the HTTP shape was tested. It is useful regression evidence, but is
  not a released-client pass.
- **Expected rejection** means Clozn returns a typed 400 rather than pretending to support behavior
  it cannot honor.
- **Not qualified** means no compatibility claim is made.

## Matrix

| Client and exact target | Discovery | Non-stream text | Stream text | Cancellation | Run association | Tools / structured output | Evidence state |
|---|---|---|---|---|---|---|---|
| OpenAI Python `2.46.0` | `GET /v1/models` executable test | Chat Completions executable test | SSE executable test | Not exercised through this SDK | Body/terminal `clozn_run_id` parsed and resolved | **Executable native-boundary contract:** an exact-qualified fake atomic worker result covers one function call, assistant/tool-result continuation, buffered tool SSE, `json_object`, and typed failures. **No real model is qualified.** | `tests/test_openai_client_compat.py` and `tests/test_openai_structured_client_compat.py`; pinned in the CPU CI lane |
| Ollama Python `0.6.2` | `/api/tags` executable test | `/api/chat` and `/api/generate` executable tests | Both NDJSON paths executable tests | Client closes a live chat stream; partial run must record `client_disconnected` | Non-stream header hook; terminal extension checked if the SDK preserves it | **Expected rejection** for `format="json"` | `tests/test_ollama_python_client_compat.py`; pinned in the CPU CI lane |
| Ollama JavaScript `0.6.3` | `/api/tags` executable test | `/api/chat` and `/api/generate` executable probe | Both NDJSON paths executable probe | Not exercised | All four body/terminal run IDs resolved | Not qualified | `tests/test_ollama_js_client_compat.py`; pinned by `tests/clients/package-lock.json` |
| Aider `0.86.2` from PyPI | Aider does not use model discovery in the tested command | **Released-client pass** on 2026-07-20 | **Released-client pass** on 2026-07-20 | Not exercised | Exactly one journal run per command | Not applicable to the tested whole-edit prompt flow; no OpenAI tool call was requested | Actual `aider` subprocesses exited 0 locally; `tests/test_aider_client_compat.py` is the pinned executable regression test |
| Open WebUI `0.10.2`, server-side OpenAI provider | Source contract targets `GET /v1/models` | Scheduled released-server proxy probe | Scheduled released-server proxy probe | Not exercised | The probe currently checks response text, not Clozn run ID recovery | **Not qualified:** the full two-request native-tool loop has not run against an exact-qualified real model | `.github/workflows/client-conformance.yml` is scheduled/manual; **not executed in this environment** because the full released distribution was not installed locally and no successful workflow result is recorded here |
| Open WebUI `0.10.2`, browser chat UI | Source audit only | Not run | Not run | Not run | Not run | Not qualified | No browser-level claim. This remains Phase 2.10 work. |

The Aider run used the documented `openai/<model>` route with `OPENAI_API_BASE` pointing at Clozn.
The exact released-client payload captured at the gateway contained `messages`, `model`, `stream`, and
`temperature: 0`; both `stream: false` and `stream: true` completed, and each created one journal run.
This proves transport and response parsing for the one-shot workflow. It does **not** prove that an
arbitrary local model will reliably produce edits in Aider's requested format.

## Phase 2.8 structured-I/O boundary

The OpenAI 2.46.0 test executes the released SDK against Clozn's real HTTP handler, but uses a deterministic
fake of the atomic native worker result and a temporary qualification registry v2 entry for its exact
`model_sha256`, `template_fingerprint`, native worker pipeline, validator, schema subset, and passing evidence.
It proves request/response parsing for one `auto` function call, assistant tool-call plus tool-result
continuation, buffer-then-validate SSE tool deltas, `json_object`, and the restricted strict `json_schema`
contract. It also proves an unqualified identity fails before generation and native parse or validation failure
becomes one typed 502 with one errored journal run. The dependency-free contract suite additionally covers the
`tool_choice:"none"` text bypass, final-text continuation without redeclaring tools, strict history order,
non-finite JSON rejection, think-tag non-repair, schema-subset rejection, runtime-pipeline drift, and strict
validation of the native parser's assistant message.

That is executable transport/contract evidence, not real-model qualification. Clozn currently bundles no
qualified real-model entry; tools are never executed by Clozn, tools and schema output are mutually exclusive,
and structured streams are buffered rather than token-live. The native worker implementation now atomically
uses llama-common template rendering, AR grammar enforcement, and llama-common parsing; model-free C++ and
Python tests exercise those seams. A live exact-model qualification battery and the complete Open WebUI
first-call/tool-result/second-call loop remain open.

## Open WebUI audit

Open WebUI's official provider guide says connection verification calls the provider's `/models`
endpoint and its OpenAI-compatible path uses Chat Completions. Clozn implements both relevant routes.
The `0.10.2` source appends `/models` for discovery and `/chat/completions` for generation, matching a
Clozn base URL ending in `/v1`.

There are two important boundaries:

1. The scheduled probe installs released Open WebUI `0.10.2`, starts its server, calls Open WebUI's
   own `/api/models` and `/api/chat/completions` routes, and verifies that Open WebUI proxies both
   non-stream and stream traffic through Clozn. It is a real released-server test, but it has not yet
   produced a recorded green workflow result and it is not a browser test.
2. In `0.10.2`, an interactive UI session can inject native built-in tool definitions when a session
   ID is present. Clozn now has a fail-closed one-function path through its atomic native renderer, grammar,
   and parser, but it accepts active tools only for a registry v2 entry matching the loaded model SHA-256,
   template fingerprint, exact native pipeline, schema subset, and evidence. No real model is prequalified,
   and the released Open WebUI two-request tool loop has not run successfully, so native-tool conformance
   remains open. Usage-enabled models can also make Open WebUI request
   `stream_options.include_usage: true`, which Clozn rejects because it does not have honest usage
   counts to return.

Cancellation through the Open WebUI browser-to-server-to-provider chain remains untested. The Ollama
Python test is the current released-client cancellation evidence for Clozn itself.

## Reproducing the Aider target

The tested configuration follows Aider's documented OpenAI-compatible setup:

```text
OPENAI_API_BASE=http://127.0.0.1:8000/v1
OPENAI_API_KEY=local-test-key
aider --model openai/clozn-local --message "Reply briefly" --stream
```

The automated test additionally disables update checks, analytics, Git integration, and model
warnings so the conformance run is deterministic and makes no unrelated network request.

## Primary references

- [Open WebUI v0.10.2 release](https://github.com/open-webui/open-webui/releases/tag/v0.10.2)
- [Open WebUI OpenAI-compatible provider guide](https://docs.openwebui.com/getting-started/quick-start/connect-a-provider/starting-with-openai-compatible/)
- [Open WebUI v0.10.2 OpenAI provider routes](https://github.com/open-webui/open-webui/blob/v0.10.2/backend/open_webui/routers/openai.py)
- [Open WebUI v0.10.2 UI request construction](https://github.com/open-webui/open-webui/blob/v0.10.2/src/lib/components/chat/Chat.svelte)
- [Open WebUI v0.10.2 built-in tool injection](https://github.com/open-webui/open-webui/blob/v0.10.2/backend/open_webui/utils/tools.py)
- [Aider OpenAI-compatible API instructions](https://aider.chat/docs/llms/openai-compat.html)
- [Aider options reference](https://aider.chat/docs/config/options.html)
- [aider-chat 0.86.2 release on PyPI](https://pypi.org/project/aider-chat/0.86.2/)
- [OpenAI Python 2.46.0 on PyPI](https://pypi.org/project/openai/2.46.0/)
- [Ollama Python 0.6.2 on PyPI](https://pypi.org/project/ollama/0.6.2/)
- [Ollama JavaScript 0.6.3 on npm](https://www.npmjs.com/package/ollama/v/0.6.3)
