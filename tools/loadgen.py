#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["tokenizers>=0.20", "huggingface_hub>=0.25"]
# ///
"""A small stand-in for `vllm bench serve --dataset-name random`, for machines vLLM doesn't install on (macOS).

Takes the same flags bench.py passes to vllm and writes the same result fields: random prompts of exactly
--random-input-len tokens (after an optional shared --random-prefix-len), --random-output-len tokens requested
with ignore_eos, --max-concurrency requests in flight, streamed over an OpenAI-compatible API. TTFT is the time
to the first streamed text; TPOT is (latency - TTFT) / (output tokens - 1) per request; output tokens come from
the server's usage report when it sends one, else from the streamed chunks.
"""

import argparse
import datetime as dt
import http.client
import json
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse


def load_tokenizer(names: list[str]):
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    for name in names:
        try:
            return Tokenizer.from_file(hf_hub_download(name, "tokenizer.json"))
        except Exception as e:  # a repo without tokenizer.json (or gated): try the next one
            print(f"tokenizer {name}: {e}", file=sys.stderr)
    sys.exit(f"no tokenizer.json in {names}")


def random_text(tok, rng, vocab, n: int) -> str:
    """Text that encodes to exactly n tokens (or as close as decode/re-encode allows)."""
    ids = [rng.choice(vocab) for _ in range(n)]
    for _ in range(8):  # decoding then re-encoding changes the count a little: settle on exactly n tokens
        text = tok.decode(ids)
        ids = tok.encode(text, add_special_tokens=False).ids
        if len(ids) == n:
            break
        ids = ids[:n] + [rng.choice(vocab) for _ in range(n - len(ids))]
    return tok.decode(ids)


def make_prompts(tok, n: int, isl: int, prefix_len: int = 0, seed: int = 0) -> list[tuple[str, int]]:
    """n prompts of prefix_len shared tokens then isl random ones. Each run (number of prompts, length) gets its
    own random text, so a server that caches prompts can't reuse an earlier run's; the shared prefix is the same in
    every run, like `vllm bench serve --random-prefix-len`."""
    special = set(getattr(tok, "get_added_tokens_decoder", dict)().keys())
    vocab = [i for i in range(tok.get_vocab_size()) if i not in special]
    prefix = random_text(tok, random.Random(f"{seed}-prefix-{prefix_len}"), vocab, prefix_len) if prefix_len else ""
    rng = random.Random(f"{seed}-{n}-{isl}-{prefix_len}")
    prompts = []
    for _ in range(n):
        text = prefix + random_text(tok, rng, vocab, isl)
        prompts.append((text, len(tok.encode(text, add_special_tokens=False).ids)))
    return prompts


def one_request(url, model, prompt, osl, temperature, chat) -> dict:
    u = urlparse(url)
    body = {"model": model, "max_tokens": osl, "stream": True, "ignore_eos": True,
            "stream_options": {"include_usage": True}}
    if temperature is not None:
        body["temperature"] = temperature
    body |= {"messages": [{"role": "user", "content": prompt}]} if chat else {"prompt": prompt}
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=3600)
    t0 = time.perf_counter()
    first, stamps, chunks, usage = None, [], 0, None
    try:
        conn.request("POST", u.path, json.dumps(body), {"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            return {"ok": False, "error": f"HTTP {resp.status}: {resp.read(300)!r}"}
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            msg = json.loads(data)
            usage = msg.get("usage") or usage
            for c in msg.get("choices", []):
                text = c.get("text") or (c.get("delta") or {}).get("content") or ""
                if text:
                    now = time.perf_counter()
                    first = first or now
                    stamps.append(now)
                    chunks += 1
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        conn.close()
    end = time.perf_counter()
    if first is None:
        return {"ok": False, "error": "no tokens streamed"}
    out = (usage or {}).get("completion_tokens") or chunks
    ttft, lat = first - t0, end - t0
    return {"ok": True, "ttft": ttft, "latency": lat, "output_tokens": out,
            "tpot": (lat - ttft) / (out - 1) if out > 1 else 0.0,
            "itl": [b - a for a, b in zip(stamps, stamps[1:])]}


def summarize(name: str, values: list[float], pcts: list[int]) -> dict:
    ms = [v * 1000 for v in values] or [0.0]
    s = sorted(ms)
    out = {f"mean_{name}_ms": statistics.fmean(ms), f"median_{name}_ms": statistics.median(ms),
           f"std_{name}_ms": statistics.pstdev(ms)}
    for p in pcts:
        out[f"p{p}_{name}_ms"] = s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", default="openai")
    p.add_argument("--base-url", required=True)
    p.add_argument("--endpoint")
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", action="append", required=True)
    p.add_argument("--dataset-name", default="random")
    p.add_argument("--random-input-len", type=int, required=True)
    p.add_argument("--random-output-len", type=int, required=True)
    p.add_argument("--random-range-ratio", default="0")
    p.add_argument("--random-prefix-len", type=int, default=0)
    p.add_argument("--ignore-eos", action="store_true")
    p.add_argument("--num-prompts", type=int, required=True)
    p.add_argument("--max-concurrency", type=int, required=True)
    p.add_argument("--request-rate", default="inf")
    p.add_argument("--percentile-metrics", default="ttft,tpot,itl,e2el")
    p.add_argument("--metric-percentiles", default="50,90,99")
    p.add_argument("--temperature", type=float)
    p.add_argument("--save-result", action="store_true")
    p.add_argument("--result-dir", default=".")
    p.add_argument("--result-filename", required=True)
    p.add_argument("--metadata", nargs="*", default=[])
    a = p.parse_args()

    chat = a.backend == "openai-chat"
    url = a.base_url.rstrip("/") + (a.endpoint or ("/v1/chat/completions" if chat else "/v1/completions"))
    tok = load_tokenizer(a.tokenizer)
    prompts = make_prompts(tok, a.num_prompts, a.random_input_len, a.random_prefix_len)
    pcts = [int(x) for x in a.metric_percentiles.split(",")]

    results, lock = [], threading.Lock()
    def work(pr):
        r = one_request(url, a.model, pr[0], a.random_output_len, a.temperature, chat)
        r["input_tokens"] = pr[1]
        with lock:
            results.append(r)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.max_concurrency) as pool:
        list(pool.map(work, prompts))
    duration = time.perf_counter() - t0

    ok = [r for r in results if r["ok"]]
    out_tokens = sum(r["output_tokens"] for r in ok)
    in_tokens = sum(r["input_tokens"] for r in ok)
    res = {
        "date": dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S"), "client": "bench loadgen.py",
        "backend": a.backend, "model_id": a.model, "tokenizer_id": a.tokenizer[0], "num_prompts": a.num_prompts,
        "request_rate": a.request_rate, "max_concurrency": a.max_concurrency, "duration": duration,
        "completed": len(ok), "failed": len(results) - len(ok),
        "total_input_tokens": in_tokens, "total_output_tokens": out_tokens,
        "request_throughput": len(ok) / duration, "output_throughput": out_tokens / duration,
        "total_token_throughput": (in_tokens + out_tokens) / duration,
        **summarize("ttft", [r["ttft"] for r in ok], pcts),
        **summarize("tpot", [r["tpot"] for r in ok if r["output_tokens"] > 1], pcts),
        **summarize("itl", [x for r in ok for x in r["itl"]], pcts),
        **summarize("e2el", [r["latency"] for r in ok], pcts),
        "errors": [r["error"] for r in results if not r["ok"]][:10],
    }
    for kv in a.metadata:
        k, _, v = kv.partition("=")
        res[k] = v
    print(f"completed {len(ok)}/{len(results)} in {duration:.1f}s: {res['output_throughput']:.1f} output tok/s, "
          f"mean TTFT {res['mean_ttft_ms']:.0f} ms, mean TPOT {res['mean_tpot_ms']:.1f} ms")
    if a.save_result:
        d = Path(a.result_dir).expanduser()
        d.mkdir(parents=True, exist_ok=True)
        (d / a.result_filename).write_text(json.dumps(res, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
