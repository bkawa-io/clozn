# bench_whitebox_tax.py — Phase 1 of the local-efficiency investigation: measure decode tok/s on a
# RUNNING clozn-server across white-box configurations, so the "white-box tax" is a number, not a
# vibe. Zero deps (stdlib urllib): each config streams one /v1/completions request and reads the
# engine's own GenFinished receipt (wall_ms, tok_per_s) off the SSE — the engine self-reports its
# throughput (generate_ar.cpp emits it), so the number excludes client/HTTP noise but INCLUDES the
# per-event serialization+write cost (on_event runs inline in the decode loop — that's the point:
# the wire format IS part of the tax).
#
#   python research/bench_whitebox_tax.py --port 8081 --label qwen7b-q4 [--sae] [--reps 5] [--new 128]
#
# Configs measured:
#   plain            stream, features off       (the serve floor; lens+confidence are unconditional in AR)
#   feat-legacy      features:true, legacy SSE  (adds tap D2H + probes + RAW-FLOAT-JSON activations on the wire)
#   feat-protocol    features:true, protocol:true, state:"light"  (same tap, activations held back)
#   state-full       protocol:true, state:"full"                  (heavy: base64 activation tensor per token)
#   (--sae adds nothing client-side: if the server was booted with --sae, every featureful request
#    pays the SAE encode — so run this script once against a plain boot and once against a --sae boot
#    and compare the featureful rows.)
import argparse
import json
import statistics
import sys
import time
import urllib.request

PROMPT = ("Write a long, meandering story about a lighthouse keeper who discovers a hidden "
          "staircase beneath the lamp room. Describe each step of the descent in detail, the "
          "smell of the salt, the sound of the sea, and what she finds at the bottom.")


def stream_once(port: int, body: dict):
    """POST a streaming completion; return (engine_wall_ms, engine_tok_per_s, new_tokens, client_wall_s, bytes_rx)."""
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    fin = None
    n_bytes = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            n_bytes += len(raw)
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            # legacy SSE: {"type":"gen_finished", "wall_ms":.., "tok_per_s":.., "new_tokens":..}
            if d.get("type") == "gen_finished":
                fin = d
            # protocol: control frame {"meta":{"kind":"end","wall_ms":..,"tok_per_s":..,"new_tokens":..}}
            meta = d.get("meta")
            if isinstance(meta, dict) and meta.get("kind") == "end":
                fin = meta
    client_wall = time.perf_counter() - t0
    if fin is None:
        raise RuntimeError("no gen_finished/end frame seen on the stream")
    return fin["wall_ms"], fin["tok_per_s"], fin["new_tokens"], client_wall, n_bytes


def bench(port: int, name: str, extra: dict, new: int, reps: int):
    body = {"prompt": PROMPT, "max_tokens": new, "stream": True}
    body.update(extra)
    stream_once(port, body)  # warmup (discarded): CUDA graphs, allocator, page-in
    walls, tps, toks, rx = [], [], [], []
    for _ in range(reps):
        w, t, n, _cw, b = stream_once(port, body)
        walls.append(w); tps.append(t); toks.append(n); rx.append(b)
    med_tps = statistics.median(tps)
    med_wall = statistics.median(walls)
    print(f"  {name:<14} {med_tps:8.1f} tok/s   wall {med_wall:8.1f} ms   "
          f"new_tokens {min(toks)}-{max(toks)}   sse {statistics.median(rx)/1024:8.1f} KiB")
    return {"config": name, "tok_per_s": med_tps, "wall_ms": med_wall,
            "new_tokens": [min(toks), max(toks)], "sse_kib": statistics.median(rx) / 1024,
            "all_tok_per_s": tps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=10) as r:
        health = json.loads(r.read())
    print(f"[bench] {args.label or 'server'} @ :{args.port}  model={health.get('model')}  "
          f"mode={health.get('mode')}  sae={'yes: ' + str(health['sae']) if 'sae' in health else 'no'}")
    print(f"[bench] max_tokens={args.new}, median of {args.reps} (1 warmup discarded), greedy, prompt fixed")

    rows = [
        bench(args.port, "plain", {}, args.new, args.reps),
        bench(args.port, "feat-legacy", {"features": True}, args.new, args.reps),
        bench(args.port, "feat-protocol", {"features": True, "protocol": True}, args.new, args.reps),
        bench(args.port, "state-full", {"protocol": True, "state": "full"}, args.new, args.reps),
    ]
    base = rows[0]["tok_per_s"]
    print(f"\n  {'config':<14} {'tok/s':>8} {'vs plain':>9}")
    for r in rows:
        print(f"  {r['config']:<14} {r['tok_per_s']:8.1f} {r['tok_per_s']/base*100:8.1f}%")

    if args.json_out:
        out = {"label": args.label, "port": args.port, "health": health,
               "new": args.new, "reps": args.reps, "rows": rows}
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"[bench] wrote {args.json_out}")


if __name__ == "__main__":
    sys.exit(main())
