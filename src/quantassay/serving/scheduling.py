"""Replay concurrent traffic and retain native SGLang scheduling evidence.

This experiment deliberately uses /generate: token IDs, request IDs and cached
token counts remain observable. It does not change the scheduling algorithm.
"""

from __future__ import annotations

import json
import math
import random
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from quantassay.contracts import RequestRecord, RequestStatus, sha256_of
from quantassay.experiments.store import atomic_write_json
from quantassay.serving.benchmark import _open_stream
from quantassay.serving.evaluator import (
    iter_sse_payloads, percentile, record_from_stream, summarize_requests,
)


def arrival_offsets(count: int, *, mode: str, rate: float, seed: int) -> list[float]:
    """Open-loop arrivals: response times never determine the next arrival."""
    if count < 1 or not math.isfinite(rate) or rate <= 0:
        raise ValueError("request count and finite arrival rate must be positive")
    if mode not in ("burst", "fixed", "poisson"):
        raise ValueError("arrival mode must be burst, fixed, or poisson")
    rng = random.Random(seed)
    offsets = [0.0]
    for i in range(1, count):
        offsets.append(0.0 if mode == "burst" else (
            i / rate if mode == "fixed" else offsets[-1] + rng.expovariate(rate)
        ))
    return [round(value, 9) for value in offsets]


def build_trace(tokenizer: Any, *, count: int, mode: str, rate: float,
                seed: int, context_length: int) -> dict[str, Any]:
    """Materialize 60% shared-long, 20% unique-long and 20% unique-short."""
    offsets = arrival_offsets(count, mode=mode, rate=rate, seed=seed)
    rng = random.Random(seed)
    empty = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False,
    )
    requests = []
    for i, offset in enumerate(offsets):
        group = "unique-long" if i % 5 == 0 else (
            "unique-short" if i % 5 == 1 else "shared-long"
        )
        target = 64 if group == "unique-short" else 320
        cap = (32, 64, 96)[i % 3]
        identity = "shared handbook" if group == "shared-long" else (
            f"private case {rng.getrandbits(128):032x}"
        )
        prefix = (f"Context for {identity}. "
                  "A service accepts requests, caches intermediate states, "
                  "and must balance waiting time with efficient execution. ")
        suffix = f"\nTask {i}: summarize the context and explain one tradeoff."
        budget = target - len(empty) - len(tokenizer.encode(suffix, add_special_tokens=False))
        if budget < 1:
            raise ValueError("chat template does not fit the short prompt budget")
        prefix_ids = tokenizer.encode(prefix * 40, add_special_tokens=False)
        text = tokenizer.decode(prefix_ids[:budget]) + suffix
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
        )
        if len(ids) + cap > context_length:
            raise ValueError("prompt plus output budget exceeds context length")
        requests.append({
            "request_id": f"sched-{i:05d}", "group": group,
            "arrival_seconds": offset, "text": text, "input_ids": ids,
            "input_tokens": len(ids), "input_sha256": sha256_of(ids),
            "max_new_tokens": cap,
        })
    trace = {
        "schema_version": 1, "seed": seed, "arrival_mode": mode,
        "request_rate": rate, "thinking_disabled": True, "temperature": 0,
        "cache_policy": "cold_start_after_warmup_then_prefix_reuse",
        "requests": requests,
    }
    trace["fingerprint"] = sha256_of(trace)
    return trace


def validate_trace(trace: dict[str, Any], *, context_length: int) -> None:
    """Reject edited, ambiguous or oversized replay inputs before serving."""
    if trace.get("fingerprint") != sha256_of({k: v for k, v in trace.items() if k != "fingerprint"}):
        raise ValueError("replay trace fingerprint mismatch")
    if trace.get("schema_version") != 1 or not trace.get("thinking_disabled") or trace.get("temperature") != 0:
        raise ValueError("unsupported replay trace protocol")
    requests = trace.get("requests") or []
    seen = set()
    previous = -1.0
    for request in requests:
        rid = request["request_id"]
        offset = request["arrival_seconds"]
        ids = request["input_ids"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", rid) or rid in seen:
            raise ValueError("replay request IDs must be unique and safe for log correlation")
        if not math.isfinite(offset) or offset < 0 or offset < previous:
            raise ValueError("replay arrival offsets must be finite, nonnegative and ordered")
        if not ids or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("replay token IDs must be nonnegative integers")
        if request["input_tokens"] != len(ids) or request["input_sha256"] != sha256_of(ids):
            raise ValueError("replay input token identity mismatch")
        if type(request["max_new_tokens"]) is not int or request["max_new_tokens"] < 1:
            raise ValueError("replay output budgets must be positive integers")
        if len(ids) + request["max_new_tokens"] > context_length:
            raise ValueError("replay request exceeds server context length")
        if request["group"] not in ("shared-long", "unique-long", "unique-short"):
            raise ValueError("unsupported replay request group")
        seen.add(rid)
        previous = offset
    if len(requests) < 5:
        raise ValueError("replay trace must contain at least five requests")


def native_events(lines: Iterable[bytes], meta: dict[str, Any]) -> Iterable[bytes]:
    """Convert native cumulative-text SSE lazily, retaining actual usage."""
    previous = ""
    for event in iter_sse_payloads(lines):
        if event.get("error"):
            raise ValueError(f"native server error: {event['error']}")
        info = event.get("meta_info") or {}
        meta.update(info)
        finish = info.get("finish_reason") or {}
        if finish.get("type") == "abort":
            raise ValueError(f"native request aborted: {finish}")
        text = event.get("text", previous)
        if not text.startswith(previous):
            raise ValueError("native stream text is not cumulative")
        delta = text[len(previous):]
        previous = text
        # A generated special token can have empty decoded text. It is still
        # an output token arrival, observable via native output_ids.
        if not delta and event.get("output_ids"):
            delta = "<output-token>"
        usage = None
        if isinstance(info.get("completion_tokens"), int):
            usage = {"completion_tokens": info["completion_tokens"],
                     "prompt_tokens": info.get("prompt_tokens", 0),
                     "total_tokens": info.get("prompt_tokens", 0) + info["completion_tokens"]}
        converted = {"choices": [{"delta": {"content": delta},
                                  "finish_reason": finish.get("type")}], "usage": usage}
        yield b"data: " + json.dumps(converted).encode() + b"\n"


def send_request(base_url: str, request: dict[str, Any], *, origin: float,
                 timeout: float, opener: Any = _open_stream) -> dict[str, Any]:
    sent = time.monotonic()
    payload = {
        "rid": request["request_id"], "input_ids": request["input_ids"],
        "stream": True, "sampling_params": {
            "temperature": 0, "max_new_tokens": request["max_new_tokens"],
        },
    }
    meta: dict[str, Any] = {}
    lines = None
    try:
        lines = opener(f"{base_url}/generate", payload, timeout)
        record = record_from_stream(
            request["request_id"], native_events(lines, meta),
            sent_at=sent, input_tokens=request["input_tokens"],
        )
        if record.status is RequestStatus.OK and (
            meta.get("id") != request["request_id"]
            or meta.get("prompt_tokens") != request["input_tokens"]
            or not meta.get("finish_reason") or not record.usage_available
        ):
            raise ValueError("native response lacks matching ID, token usage, or finish reason")
    except Exception as exc:
        record = RequestRecord(
            request_id=request["request_id"],
            status=RequestStatus.TIMEOUT if isinstance(exc, TimeoutError) else RequestStatus.FAILED,
            failure_reason=f"{type(exc).__name__}: {str(exc)[:240]}",
            input_tokens=request["input_tokens"], usage_available=False,
        )
    finally:
        if lines is not None and hasattr(lines, "close"):
            lines.close()
    ended = time.monotonic()
    return {
        "request_id": request["request_id"], "group": request["group"],
        "input_sha256": request["input_sha256"],
        "max_new_tokens": request["max_new_tokens"],
        "scheduled_arrival_seconds": request["arrival_seconds"],
        "sent_seconds": sent - origin, "completed_seconds": ended - origin,
        "dispatch_lag_ms": max(0, (sent - origin - request["arrival_seconds"]) * 1000),
        "arrival_to_completion_ms": (ended - origin - request["arrival_seconds"]) * 1000,
        "server_meta": meta, "record": record.model_dump(mode="json"),
    }


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


_METRIC_PREFIXES = tuple("sglang:" + name for name in (
    "num_running_reqs", "num_queue_reqs", "num_used_tokens", "token_usage",
    "cache_hit_rate", "cached_tokens_total", "num_retracted_reqs", "queue_time",
))


def parse_metrics(text: str) -> dict[str, float]:
    """Keep Prometheus series labels; never silently sum different workers."""
    result = {}
    for line in text.splitlines():
        if not line.startswith(_METRIC_PREFIXES):
            continue
        match = re.match(r"^(\S+(?:\{.*\})?)\s+([-+.\deE]+)\s*$", line)
        if match:
            value = float(match[2])
            if math.isfinite(value):
                result[match[1]] = value
    return result


class MetricsSampler:
    """Native gauge snapshots, not an exact trace of every scheduler step."""

    def __init__(self, base_url: str, path: Path, *, interval: float = 0.2):
        self.url, self.path, self.interval = f"{base_url}/metrics", path, interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.started = time.monotonic()
        self.samples: list[dict[str, Any]] = []

    def capture(self, phase: str) -> None:
        sample: dict[str, Any] = {"elapsed_seconds": time.monotonic() - self.started,
                                  "phase": phase}
        try:
            with urllib.request.urlopen(self.url, timeout=3) as response:
                sample["values"] = parse_metrics(response.read().decode())
            sample["status"] = "ok" if sample["values"] else "unavailable"
        except Exception as exc:
            sample.update(status="unavailable", reason=f"{type(exc).__name__}: {exc}")
        self.samples.append(sample)
        append_jsonl(self.path, sample)

    def _loop(self) -> None:
        while not self.stop_event.wait(self.interval):
            self.capture("during")

    def start(self) -> None:
        self.capture("before")
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join()
        self.capture("after")

    def summary(self) -> dict[str, Any]:
        valid = [s["values"] for s in self.samples if s["status"] == "ok"]
        peaks = {}
        for values in valid:
            for key, value in values.items():
                if "_bucket" not in key and "_sum" not in key and "_count" not in key:
                    peaks[key] = max(peaks.get(key, value), value)
        return {"samples": len(self.samples), "valid_samples": len(valid),
                "sampled_peaks": peaks,
                "before": valid[0] if valid else {}, "after": valid[-1] if valid else {},
                "scope": "sampled native gauges; updates may lag actual batch steps",
                "eviction_count": {"status": "unavailable", "reason": "not exposed by collected native metrics"},
                "kv_admission_block_count": {"status": "unavailable", "reason": "requires direct scheduler instrumentation"}}


def run_trace(base_url: str, trace: dict[str, Any], output_dir: Path, *,
              concurrency: int, timeout: float = 120, opener: Any = _open_stream,
              sample_metrics: bool = True) -> dict[str, Any]:
    """Submit at planned absolute offsets, bounded by client worker count.

    Worker saturation causes recorded dispatch lag, never a hidden change to
    the planned arrival trace. Records are persisted by workers on completion.
    """
    path = output_dir / "requests.jsonl"
    if path.exists():
        raise ValueError(f"refusing to overwrite evidence: {path}")
    if concurrency < 1:
        raise ValueError("client concurrency must be positive")
    sampler = MetricsSampler(base_url, output_dir / "scheduler-metrics.jsonl")
    if sample_metrics:
        sampler.start()
    origin = time.monotonic()
    lock = threading.Lock()

    def worker(request):
        row = send_request(base_url, request, origin=origin, timeout=timeout, opener=opener)
        with lock:
            append_jsonl(path, row)
        return row

    try:
        with ThreadPoolExecutor(max_workers=min(concurrency, len(trace["requests"]))) as pool:
            futures = []
            for request in trace["requests"]:
                remaining = origin + request["arrival_seconds"] - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                futures.append(pool.submit(worker, request))
            rows = [future.result() for future in futures]
        wall = time.monotonic() - origin
    finally:
        if sample_metrics:
            sampler.stop()
    records = [RequestRecord.model_validate(row["record"]) for row in rows]
    serving = summarize_requests(records, side="bf16", wall_seconds=wall,
                                 workload_id="scheduling-mixed-prefix",
                                 workload_fingerprint=trace["fingerprint"],
                                 records_path=str(path), sglang_version="0.5.3")
    return {"status": "succeeded" if all(r.status is RequestStatus.OK for r in records) else "failed",
            "summary": serving.model_dump(mode="json"), "wall_seconds": wall,
            "metrics_origin_offset_seconds": sampler.started - origin,
            "client_concurrency": concurrency, "metrics": sampler.summary(), "rows": rows}


def parse_request_times(log_path: Path) -> dict[str, dict[str, Any]]:
    times = {}
    regex = re.compile(r"Req Time Stats\(rid=([^,]+),.*queue_duration=([\d.]+)(us|ms|s), "
                       r"forward_duration=([\d.]+)(us|ms|s)")
    factors = {"us": 0.001, "ms": 1, "s": 1000}
    for line_number, line in enumerate(log_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        match = regex.search(line)
        if match:
            times[match[1]] = {"server_queue_ms": float(match[2]) * factors[match[3]],
                               "server_forward_ms": float(match[4]) * factors[match[5]],
                               "server_log_line": line_number}
    return times


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"status": "unavailable", "sample_count": 0}
    return {"status": "ok", "sample_count": len(values),
            "max": round(max(values), 3),
            **{label: round(percentile(values, q), 3)
               for label, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))}}


def finalize_result(result: dict[str, Any], log_path: Path, output_dir: Path) -> dict[str, Any]:
    times = parse_request_times(log_path)
    rows = result.pop("rows")
    groups: dict[str, Any] = {}
    for row in rows:
        row.update(times.get(row["request_id"], {"server_queue_ms": None,
                                               "server_queue_reason": "request time not found in native log"}))
        append_jsonl(output_dir / "scheduler-requests.jsonl", row)
    for group in ("overall", "shared-long", "unique-long", "unique-short"):
        selected = rows if group == "overall" else [r for r in rows if r["group"] == group]
        ok = [r for r in selected if r["record"]["status"] == "ok"]
        fields = {
            "ttft_ms": [r["record"]["ttft_ms"] for r in ok],
            "e2e_ms": [r["record"]["e2e_latency_ms"] for r in ok],
            "tpot_ms": [RequestRecord.model_validate(r["record"]).tpot_ms for r in ok],
            "itl_ms": [gap for r in ok for gap in r["record"]["itl_ms"]],
            "server_queue_ms": [r["server_queue_ms"] for r in ok if r.get("server_queue_ms") is not None],
            "dispatch_lag_ms": [r["dispatch_lag_ms"] for r in selected],
            "arrival_to_completion_ms": [r["arrival_to_completion_ms"] for r in ok],
        }
        groups[group] = {"requests": len(selected), "requests_ok": len(ok),
                         "cached_tokens": sum(r["server_meta"].get("cached_tokens", 0) for r in ok),
                         "prompt_tokens": sum(r["record"]["input_tokens"] for r in ok),
                         "distributions": {key: distribution([v for v in vals if v is not None])
                                           for key, vals in fields.items()}}
    result["groups"] = groups
    result["queue_time_coverage"] = sum(r.get("server_queue_ms") is not None for r in rows)
    result["client_dispatch_delayed"] = any(r["dispatch_lag_ms"] > 25 for r in rows)
    result["limitations"] = [
        "ITL measures streamed output chunks, which may contain multiple tokens.",
        "Queue time is native scheduler wait-to-first-forward, excluding client/network time.",
        "Native queue-time logs are rounded; sub-millisecond values can be coarse.",
        "Gauge sampling is not a complete GPU batch trace; eviction/admission counts are unavailable.",
        "These are synthetic BF16 workloads; quality and quantization benefits are not evaluated.",
        "One run per policy does not establish a stable performance improvement.",
    ]
    if result.get("aging_enabled"):
        result["limitations"].append(
            "Aging changes priority after a wait threshold; it is not an admission or latency guarantee."
        )
    atomic_write_json(output_dir / "scheduling-result.json", result)
    lines = ["# Native SGLang scheduling baseline", "",
             f"Status: {result['status']}. Policy: {result['policy']}.", "",
             "| Request group | Success | TTFT p50 ms | Queue p50 ms | Cached / prompt tokens |",
             "|---|---:|---:|---:|---:|"]
    for group, data in groups.items():
        dist = data["distributions"]
        lines.append(f"| {group} | {data['requests_ok']}/{data['requests']} | "
                     f"{dist['ttft_ms'].get('p50', 'unavailable')} | "
                     f"{dist['server_queue_ms'].get('p50', 'unavailable')} | "
                     f"{data['cached_tokens']} / {data['prompt_tokens']} |")
    lines.extend(["", f"Client dispatch delayed >25 ms: {result['client_dispatch_delayed']}.", "",
                  "Evidence: workload.json, requests.jsonl, scheduler-requests.jsonl, "
                  "scheduler-metrics.jsonl, logs/server.log, manifest.json, scheduling-result.json.", ""])
    lines.extend("- " + limitation for limitation in result["limitations"])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return result
