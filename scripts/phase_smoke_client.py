#!/usr/bin/env python3
"""Minimal streaming client for the phase-timing smoke.

Stdlib only (the host python3 has no guarantees about third-party packages),
so the client can always run next to the launcher.  It sends N completion
requests, streams the SSE response, and records

  * TTFT            -- wall clock from "request sent" to the first content chunk
  * end-to-end wall -- from "request sent" to the terminating [DONE]
  * output tokens   -- number of streamed content deltas
  * per-chunk gaps  -- so an inter-token-latency figure is available too

Usage:
  phase_smoke_client.py --base-url URL --model NAME --outdir DIR
                        [--requests N] [--prompt-tokens N] [--max-tokens N]
                        [--concurrency C] [--seed S] [--tag T]
"""

from __future__ import annotations

import argparse
import http.client
import json
import pathlib
import random
import statistics
import sys
import threading
import time
import urllib.parse

VOCAB = (
    "the system reports a stable voltage across all measured rails while "
    "thermal margin remains wide and the control loop adjusts fan speed "
    "slowly without introducing audible ripple into the output"
).split()


def build_prompt(target_tokens: int, seed: int) -> str:
    """Deterministic filler prompt of roughly ``target_tokens`` tokens.

    ~1.3 tokens per whitespace word for English text; the exact count does not
    matter for a smoke, but it must be *reproducible*, so the word list is
    generated from a seeded RNG rather than sampled from a corpus.
    """
    rng = random.Random(seed)
    words = [rng.choice(VOCAB) for _ in range(max(1, int(target_tokens / 1.3)))]
    return " ".join(words)


def stream_one(host: str, port: int, path: str, payload: dict, timeout: float):
    """POST a streaming completion; return (ttft_s, total_s, tokens, gaps)."""
    body = json.dumps(payload).encode()
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    t0 = time.perf_counter()
    conn.request(
        "POST",
        path,
        body=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    resp = conn.getresponse()
    if resp.status != 200:
        detail = resp.read().decode(errors="replace")[:2000]
        conn.close()
        raise RuntimeError(f"HTTP {resp.status}: {detail}")

    ttft = None
    last = None
    gaps: list[float] = []
    tokens = 0
    while True:
        line = resp.readline()
        if not line:
            break
        stripped = line.strip()
        if not stripped.startswith(b"data:"):
            continue
        data = stripped[5:].strip()
        if data == b"[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        for choice in chunk.get("choices", []):
            text = choice.get("text") or ""
            if not text:
                continue
            now = time.perf_counter()
            if ttft is None:
                ttft = now - t0
            elif last is not None:
                gaps.append(now - last)
            last = now
            tokens += 1
    total = time.perf_counter() - t0
    conn.close()
    return ttft, total, tokens, gaps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--outdir", type=pathlib.Path, required=True)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--tag", default="smoke")
    args = parser.parse_args(argv)

    parsed = urllib.parse.urlparse(args.base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80

    args.outdir.mkdir(parents=True, exist_ok=True)
    # One distinct prompt per request so prefix caching cannot hide work; the
    # served config disables prefix caching anyway, this is belt and braces.
    for index in range(args.requests):
        prompt = build_prompt(args.prompt_tokens, args.seed + index)
        (args.outdir / f"req_{index:02d}.json").write_text(
            json.dumps(
                {
                    "model": args.model,
                    "prompt": prompt,
                    "max_tokens": args.max_tokens,
                    "temperature": 0.0,
                    "stream": True,
                },
                indent=2,
            )
            + "\n"
        )

    results: list[dict] = []
    errors: list[dict] = []
    lock = threading.Lock()
    slots = threading.Semaphore(max(1, args.concurrency))
    threads: list[threading.Thread] = []
    wall0 = time.perf_counter()

    def worker(index: int) -> None:
        with slots:
            payload = json.loads((args.outdir / f"req_{index:02d}.json").read_text())
            try:
                ttft, total, tokens, gaps = stream_one(
                    host, port, "/v1/completions", payload, args.timeout
                )
                record = {
                    "index": index,
                    "ok": True,
                    "ttft_s": ttft,
                    "total_s": total,
                    "output_tokens": tokens,
                    "output_tps": tokens / total if total > 0 else None,
                    "mean_itl_ms": (statistics.fmean(gaps) * 1e3) if gaps else None,
                }
            except Exception as exc:  # noqa: BLE001 - report, never crash the run
                record = {
                    "index": index,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            with lock:
                (results if record["ok"] else errors).append(record)

    for index in range(args.requests):
        thread = threading.Thread(target=worker, args=(index,), daemon=True)
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - wall0

    results.sort(key=lambda item: item["index"])
    ok = [item for item in results if item["ok"]]
    summary = {
        "tag": args.tag,
        "base_url": args.base_url,
        "model": args.model,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "prompt_tokens_target": args.prompt_tokens,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "wall_s": wall,
        "n_ok": len(ok),
        "n_error": len(errors),
        "errors": errors,
        "aggregate": {},
        "per_request": results,
    }
    if ok:
        total_tokens = sum(item["output_tokens"] for item in ok)
        itls = [item["mean_itl_ms"] for item in ok if item["mean_itl_ms"]]
        summary["aggregate"] = {
            "ttft_s_mean": statistics.fmean(item["ttft_s"] for item in ok),
            "ttft_s_min": min(item["ttft_s"] for item in ok),
            "ttft_s_max": max(item["ttft_s"] for item in ok),
            "e2e_s_mean": statistics.fmean(item["total_s"] for item in ok),
            "output_tokens_total": total_tokens,
            "output_tps_per_request_mean": statistics.fmean(
                item["output_tps"] for item in ok if item["output_tps"]
            ),
            "output_tps_aggregate": total_tokens / wall if wall > 0 else None,
            "mean_itl_ms": statistics.fmean(itls) if itls else None,
        }

    out = args.outdir / "client_summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))
    print(f"wrote {out}", file=sys.stderr)
    return 0 if ok and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
