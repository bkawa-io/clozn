#!/usr/bin/env python3
"""Exercise Clozn like a long-running OpenAI-compatible writing chat.

The script intentionally sends the complete message history on every turn,
streams every response, records latency/token measurements, and optionally
runs Context Effects plus Minimal Context against the final reply.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


SYSTEM = """You are a careful college writing coach. Work only from the assignment and source notes in the conversation. Never invent a quotation, statistic, source, or citation. Be concrete, preserve the student's chosen argument, and give polished prose when asked. Do not expose hidden reasoning."""


TURNS = [
    """I'm writing a 1,500-word first-year composition essay about public libraries as climate-resilience infrastructure. I do not want a generic "libraries are good" paper. I want to argue that their ordinary civic role makes them unusually useful during heat emergencies, while acknowledging that cities can overload librarians with unfunded social-service work. Help me identify the real tension and give me three possible research questions. Keep this exploratory, not a draft.""",
    """Here is the assignment rubric from my professor.

## Assignment
- Make one contestable claim, not a broad survey.
- Synthesize at least three sources instead of summarizing them one by one.
- Include a fair counterargument and a concrete policy implication.
- Use a clear line of reasoning that a skeptical city-budget reader could follow.
- Avoid inflated language, rhetorical questions, and a five-paragraph formula.

## My preferences
I like the question about whether a familiar public institution can outperform a purpose-built emergency site. I want the essay to focus on access and trust, not claim that libraries can solve climate change. Propose two thesis options and explain the tradeoff between them.""",
    """I'm pasting the first part of my course-packet notes. Treat these as the only facts available; the source names and figures are part of the classroom simulation.

## Source A — Redwood Municipal Heat Audit (2025)
During an 18-day heat emergency, six library branches extended hours until 9 p.m. The branches logged 2,400 cooling visits after 5 p.m. Three temporary cooling tents together logged 730 visits after 5 p.m. The audit says the counts are not unique visitors and cannot prove why residents chose one site. Library branches were reachable by 14 bus routes; the tents were reachable by four.

## Source B — Chen and Alvarez, "Familiar Doors" (2024)
Interviews with 42 residents found that participants associated libraries with routine errands, indoor bathrooms, water, seating, and staff who would not demand proof of emergency. Twenty-nine interviewees said they would hesitate to enter a site marked "disaster assistance." The authors caution that this small qualitative sample is not representative.

Build an evidence map: what each source can support, what it cannot support, and one synthesis claim that uses both without overstating either.""",
    """More course-packet notes:

## Source C — Redwood Library Workers Association memo (2025)
The union supports keeping branches open during dangerous heat but reports that 61% of surveyed staff felt unprepared to recognize heat illness. It asks for voluntary emergency shifts, two staff members per open floor, de-escalation training, security support, and a written rule that librarians are not medical responders. The survey had 118 respondents out of 190 represented workers.

## Source D — City Budget Office options sheet (2025)
Retrofitting the six branches with backup power and efficient cooling is estimated at $8.4 million in one-time capital cost. Operating the three tent sites for one 18-day event cost $410,000, excluding police overtime. The sheet warns that the alternatives serve different neighborhoods and should not be compared as if they were interchangeable.

Revise the evidence map. I especially need help handling the tempting but invalid cost comparison.""",
    """I had a detour that I now think is wrong: I wanted to call the essay "The Last Public Living Room" and make the opening a nostalgic scene about childhood story hour. My professor said that frames the library as sentimental refuge and delays the policy problem. So reject that opening and do not use the phrase "last public living room" in the essay.

Instead, sketch a six-part argument sequence. The counterargument should be serious: purpose-built centers can concentrate medical staff and supplies, while libraries may normalize an emergency so much that the city underinvests in specialized response. Show where the staffing memo changes—not merely qualifies—the thesis.""",
    """Here are two additional notes and a style constraint.

## Source E — Patel, transit-access analysis (2024)
In Redwood, 38% of households in the two hottest census tracts lack a private vehicle. All four library branches serving those tracts are within a quarter mile of a frequent bus stop. The analysis did not study whether buses remained reliable during the heat emergency.

## Source F — Emergency Management after-action review (2025)
The review praises tent sites for on-site nurses, electrolyte supplies, and rapid ambulance transfer. It also reports that two tents closed for four afternoon hours because their generators overheated. The review recommends a distributed network rather than naming one facility type as the universal solution.

Style: use plain, precise language. Avoid "crucial," "vital," "multifaceted," "underscores," "tapestry," and "in today's society." Give me a detailed outline with claims and evidence relationships, not source-by-source buckets.""",
    """Draft an opening paragraph of about 170 words. Start directly with the Redwood choice during the 18-day heat emergency, not with climate change in general. It should establish the apparent competition between libraries and tents, then turn that competition into the essay's distributed-network thesis. Do not put a statistic in the first sentence, and do not use a quotation.""",
    """That opening is close, but I want to preserve two choices for later revision. Choice one: the phrase "ordinary access can become emergency capacity." Choice two: the essay must distinguish availability from effectiveness; visit counts show use, not health outcomes. Please remember both.

Now draft the counterargument paragraph. Make the case for purpose-built sites strongly before answering it. The answer should not be "libraries are better." It should be that specialization and familiarity solve different access failures, which is why the network needs both.""",
    """Let's pressure-test the argument as a skeptical budget analyst. List the four strongest objections you would raise. For each, say whether my current evidence answers it, partly answers it, or leaves it unanswered. Pay special attention to recurring operating costs, duplicated coverage, worker liability, and the absence of health-outcome data. Do not quietly fill evidence gaps.""",
    """I checked with my professor. It is acceptable to leave the health-outcome question unresolved if I name it as a limit. The policy recommendation should be narrow: fund a two-year pilot at the six branches, with backup power, voluntary trained staffing, and a shared protocol with emergency management; collect unique-visitor, referral, and heat-illness outcome data. Do not recommend permanent citywide expansion yet.

Write a concise policy section that follows from the evidence rather than sounding like a wish list.""",
    """Before the final prose task, give me a compact continuity check. State the current thesis in one sentence; list the two claims I must not overstate; identify the strongest counterargument; and repeat the exact phrase I asked to preserve. Also identify the discarded title phrase so I can verify you do not accidentally revive it.""",
    """Now write only the bridge paragraph that moves from the evidence section into the counterargument. Use 90–110 words in exactly three sentences. It must mention the 18-day emergency and 2,400 after-5 p.m. library visits, immediately clarify that visits are not health outcomes, acknowledge the staffing and specialization concern, and end with the exact words: ordinary access can become emergency capacity. Do not use any banned style word or the discarded title phrase.""",
]


class HttpFailure(RuntimeError):
    pass


def _request(base_url: str, method: str, path: str, body: dict | None = None,
             *, timeout: float = 600.0) -> tuple[int, dict]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = int(exc.code)
    try:
        result = json.loads(raw.decode("utf-8", "replace") or "{}")
    except json.JSONDecodeError as exc:
        raise HttpFailure(f"{method} {path} returned non-JSON HTTP {status}") from exc
    if not isinstance(result, dict):
        raise HttpFailure(f"{method} {path} returned a non-object")
    return status, result


def _stream_turn(base_url: str, model: str, messages: list[dict], *, turn: int,
                 max_tokens: int, timeout: float) -> tuple[str, dict]:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.08,
        "seed": 7300 + turn,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_content_at: float | None = None
    pieces: list[str] = []
    usage: dict[str, int] = {}
    run_id: str | None = None
    finish_reason: str | None = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            if "error" in event:
                raise HttpFailure(f"stream turn {turn} failed: {event['error']}")
            if isinstance(event.get("clozn_run_id"), str):
                run_id = event["clozn_run_id"]
            if isinstance(event.get("usage"), dict):
                usage = dict(event["usage"])
            choices = event.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            if isinstance(choice.get("finish_reason"), str):
                finish_reason = choice["finish_reason"]
            content = (choice.get("delta") or {}).get("content")
            if isinstance(content, str) and content:
                if first_content_at is None:
                    first_content_at = time.perf_counter()
                pieces.append(content)
    ended = time.perf_counter()
    text = "".join(pieces)
    if not text:
        raise HttpFailure(f"turn {turn} returned no public answer")
    if not run_id:
        raise HttpFailure(f"turn {turn} did not expose clozn_run_id")
    completion_tokens = usage.get("completion_tokens")
    generation_seconds = ended - (first_content_at or started)
    metrics = {
        "turn": turn,
        "run_id": run_id,
        "finish_reason": finish_reason,
        "elapsed_seconds": round(ended - started, 3),
        "time_to_first_content_seconds": round((first_content_at or ended) - started, 3),
        "generation_seconds": round(generation_seconds, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "total_tokens": usage.get("total_tokens"),
        "completion_tokens_per_second": (
            round(completion_tokens / generation_seconds, 2)
            if isinstance(completion_tokens, int) and generation_seconds > 0 else None
        ),
    }
    return text, metrics


def _poll(base_url: str, path: str, job: dict, *, timeout: float) -> dict:
    job_id = job.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise HttpFailure(f"job start omitted job_id: {job}")
    deadline = time.monotonic() + timeout
    last_progress: Any = None
    while job.get("state") not in {"completed", "failed", "cancelled"}:
        if time.monotonic() >= deadline:
            raise HttpFailure(f"job {job_id} exceeded {timeout:g}s")
        time.sleep(0.25)
        status, job = _request(base_url, "GET", path + "/" + urllib.parse.quote(job_id, safe=""))
        if status != 200:
            raise HttpFailure(f"job poll returned HTTP {status}: {job}")
        progress = job.get("progress")
        if progress != last_progress:
            print(f"analysis progress: {progress}", file=sys.stderr, flush=True)
            last_progress = progress
    if job.get("state") != "completed":
        raise HttpFailure(f"job did not complete: {job}")
    return job


def _source_text(reader: dict, source_id: str) -> str:
    sources = reader.get("sources") if isinstance(reader.get("sources"), list) else []
    source = next((row for row in sources if row.get("source_id") == source_id), None)
    blocks = (reader.get("context") or {}).get("blocks") or []
    if not isinstance(source, dict):
        return ""
    message_index = source.get("message_index")
    block = next((row for row in blocks if row.get("message_index") == message_index), None)
    span = source.get("unicode_range")
    if not isinstance(block, dict) or not isinstance(block.get("text"), str):
        return ""
    if isinstance(span, list) and len(span) == 2 and all(isinstance(x, int) for x in span):
        return block["text"][span[0]:span[1]]
    return block["text"]


def _analyze(base_url: str, run_id: str, *, timeout: float) -> dict:
    quoted = urllib.parse.quote(run_id, safe="")
    reader_path = f"/runs/{quoted}/context-investigation/reader?floor_nats=0"
    status, reader_before = _request(base_url, "GET", reader_path)
    if status != 200:
        raise HttpFailure(f"Context Effects reader returned HTTP {status}: {reader_before}")
    source_ids = (reader_before.get("measurement") or {}).get("source_universe")
    if not isinstance(source_ids, list) or not source_ids:
        raise HttpFailure(f"reader exposed no source universe: {reader_before}")

    effects_root = f"/runs/{quoted}/context-investigation/effects/jobs"
    started = time.perf_counter()
    status, job = _request(base_url, "POST", effects_root, {"source_ids": source_ids})
    if status != 202:
        raise HttpFailure(f"Context Effects start returned HTTP {status}: {job}")
    effects_job = _poll(base_url, effects_root, job, timeout=timeout)
    effects_seconds = time.perf_counter() - started
    status, reader_after = _request(base_url, "GET", reader_path)
    if status != 200:
        raise HttpFailure(f"refreshed reader returned HTTP {status}: {reader_after}")

    source_rows = []
    for source in reader_after.get("sources") or []:
        measurement = source.get("measurement") if isinstance(source, dict) else None
        summary = source.get("summary") if isinstance(source, dict) else None
        source_rows.append({
            "source_id": source.get("source_id"),
            "message_index": source.get("message_index"),
            "text": _source_text(reader_after, source.get("source_id", "")),
            "measurement": measurement,
            "summary": summary,
        })

    minimal_root = f"/runs/{quoted}/minimal-context/jobs"
    started = time.perf_counter()
    status, job = _request(base_url, "POST", minimal_root, {})
    if status != 202:
        raise HttpFailure(f"Minimal Context start returned HTTP {status}: {job}")
    minimal_job = _poll(base_url, minimal_root, job, timeout=timeout)
    minimal_seconds = time.perf_counter() - started
    minimal = minimal_job.get("result") if isinstance(minimal_job.get("result"), dict) else {}
    best = minimal.get("best") if isinstance(minimal.get("best"), dict) else {}
    retained = set(best.get("retained_source_ids") or [])
    removed = set(best.get("removed_source_ids") or [])
    return {
        "source_count": len(source_ids),
        "context_effects_seconds": round(effects_seconds, 3),
        "context_effects_job": effects_job.get("result"),
        "context_effects_reader_status": reader_after.get("status"),
        "sources": source_rows,
        "minimal_context_seconds": round(minimal_seconds, 3),
        "minimal_context": minimal,
        "retained_sources": [row for row in source_rows if row["source_id"] in retained],
        "removed_sources": [row for row in source_rows if row["source_id"] in removed],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--replay-final-from",
        type=Path,
        help="Replay only the final turn using the earlier transcript in a qualification report.",
    )
    args = parser.parse_args()

    if args.model is None:
        status, models = _request(args.base_url, "GET", "/v1/models")
        rows = models.get("data") if isinstance(models.get("data"), list) else []
        if status != 200 or not rows:
            raise HttpFailure(f"could not resolve public model: {models}")
        args.model = rows[0]["id"]

    messages = [{"role": "system", "content": SYSTEM}]
    transcript: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    if args.replay_final_from is not None:
        source_report = json.loads(args.replay_final_from.read_text(encoding="utf-8"))
        source_transcript = source_report.get("transcript")
        if not isinstance(source_transcript, list) or len(source_transcript) != len(TURNS):
            raise HttpFailure("replay report does not contain the expected complete transcript")
        for source_turn in source_transcript[:-1]:
            user_text = source_turn.get("user")
            assistant_text = source_turn.get("assistant")
            if not isinstance(user_text, str) or not isinstance(assistant_text, str):
                raise HttpFailure("replay report contains a malformed transcript turn")
            messages.extend([
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ])
        turn_inputs = [(len(TURNS), source_transcript[-1]["user"])]
    else:
        turn_inputs = list(enumerate(TURNS, 1))

    for index, user_text in turn_inputs:
        messages.append({"role": "user", "content": user_text})
        max_tokens = 128 if index == len(TURNS) else 240
        assistant_text, metrics = _stream_turn(
            args.base_url, args.model, messages, turn=index,
            max_tokens=max_tokens, timeout=args.timeout,
        )
        messages.append({"role": "assistant", "content": assistant_text})
        transcript.append({"turn": index, "user": user_text, "assistant": assistant_text})
        turns.append(metrics)
        print(
            f"turn {index:02d}: prompt={metrics['prompt_tokens']} completion={metrics['completion_tokens']} "
            f"ttft={metrics['time_to_first_content_seconds']}s total={metrics['elapsed_seconds']}s",
            file=sys.stderr,
            flush=True,
        )

    final_run_id = turns[-1]["run_id"]
    report = {
        "schema": "clozn.long-essay-chat-qualification.v1",
        "model": args.model,
        "turn_count": len(TURNS),
        "final_run_id": final_run_id,
        "turns": turns,
        "transcript": transcript,
        "replay_final_from": str(args.replay_final_from) if args.replay_final_from else None,
        "analysis": _analyze(args.base_url, final_run_id, timeout=args.timeout) if args.analyze else None,
    }
    encoded = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
