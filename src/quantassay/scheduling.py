"""Independent BF16 scheduler baseline: python -m quantassay.scheduling."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from quantassay.contracts import MODEL_ID, sha256_of
from quantassay.engine import configure_engine_source
from quantassay.experiments.store import atomic_write_json, file_sha256
from quantassay.runtime import (
    _package_version, build_bf16_command, contention_blockers,
    resource_blockers, resource_warnings, server_session,
    source_file_hashes, wait_for_clean_gpu,
)
from quantassay.serving.scheduling import (
    append_jsonl, build_trace, finalize_result, run_trace, send_request,
    validate_trace,
)


def wait_native_ready(base_url: str, *, timeout: float = 60) -> None:
    """/v1/models can appear before SGLang's background warmup finishes."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=5) as response:
                if response.status == 200:
                    return
        except urllib.error.HTTPError as exc:
            if exc.code != 503:
                raise
        time.sleep(0.2)
    raise RuntimeError("SGLang native startup warmup did not finish")


def flush_idle_cache(base_url: str, *, timeout: float = 5) -> int:
    """A completed response can precede scheduler cleanup by a few steps."""
    deadline = time.monotonic() + timeout
    attempts = 0
    while time.monotonic() < deadline:
        attempts += 1
        try:
            with urllib.request.urlopen(base_url + "/flush_cache", timeout=3) as response:
                if not response.read().decode().startswith("Cache flushed."):
                    raise RuntimeError("cache flush returned an unrecognized response")
                return attempts
        except urllib.error.HTTPError as exc:
            if exc.code != 400:
                raise
        time.sleep(0.1)
    raise RuntimeError("scheduler remained busy; cache flush could not be confirmed")


def engine_source_hashes() -> dict[str, str]:
    """Version strings alone do not identify a locally modified engine."""
    package = Path(importlib.util.find_spec("sglang").origin).parent
    return {
        rel: file_sha256(package / "srt" / rel)
        for rel in ("managers/scheduler.py", "managers/schedule_policy.py",
                    "managers/schedule_batch.py", "mem_cache/radix_cache.py",
                    "mem_cache/allocator.py", "metrics/collector.py",
                    "managers/tokenizer_manager.py", "server_args.py")
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--policy", choices=("fcfs", "lpm", "lpm-aging"), required=True)
    parser.add_argument("--engine-source", type=Path,
                        help="directory containing an isolated sglang source package")
    parser.add_argument("--aging-threshold-ms", type=float, default=1000,
                        help="experimental lpm-aging queue-wait threshold, not a hard latency bound")
    parser.add_argument("--cache-start", choices=("cold", "warm-shared"), default="cold")
    parser.add_argument("--requests", type=int, default=60)
    parser.add_argument("--concurrency", type=int, default=64,
                        help="client worker cap; excess arrivals retain measured dispatch lag")
    parser.add_argument("--max-running-requests", type=int, default=4)
    parser.add_argument("--arrival-mode", choices=("burst", "fixed", "poisson"), default="fixed")
    parser.add_argument("--request-rate", type=float, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--cuda-graph-max-bs", type=int, default=4)
    parser.add_argument("--mem-fraction-static", type=float, default=0.8)
    parser.add_argument("--trace-file", type=Path,
                        help="replay a saved workload.json; overrides generation count/rate/mode/seed")
    args = parser.parse_args(argv)
    if not math.isfinite(args.aging_threshold_ms) or args.aging_threshold_ms <= 0:
        parser.error("aging threshold must be finite and positive")
    if args.policy == "lpm-aging" and not args.engine_source:
        parser.error("lpm-aging requires the patched isolated --engine-source")
    try:
        engine_identity = configure_engine_source(args.engine_source)
    except ValueError as exc:
        parser.error(str(exc))
    if args.engine_source:
        args.engine_source = args.engine_source.resolve()
    os.environ["SGLANG_LPM_MAX_WAIT_MS"] = str(args.aging_threshold_ms)
    if args.run_dir.exists():
        parser.error("run directory already exists; use a new directory to preserve evidence")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("revision must be a 40-character commit SHA")
    if args.requests < 5 or args.concurrency < 1:
        parser.error("at least 5 requests and positive client concurrency are required")
    if not args.model_dir.is_dir() or args.model_dir.name != args.revision:
        parser.error("model directory must be a cache snapshot for the requested revision")
    if _package_version("sglang") != "0.5.3":
        parser.error("the scheduling observation contract currently supports SGLang 0.5.3")

    args.run_dir.mkdir(parents=True)
    status_path = args.run_dir / "run-status.json"
    atomic_write_json(status_path, {"status": "preparing", "policy": args.policy})
    try:
        model_config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
        if (model_config.get("model_type"), model_config.get("hidden_size"),
                model_config.get("num_hidden_layers")) != ("qwen3", 1024, 28):
            raise ValueError("this baseline supports the Qwen3-0.6B source model only")
        if args.trace_file:
            trace = json.loads(args.trace_file.read_text(encoding="utf-8"))
            args.requests = len(trace.get("requests") or [])
            args.seed, args.arrival_mode, args.request_rate = trace["seed"], trace["arrival_mode"], trace["request_rate"]
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
            trace = build_trace(tokenizer, count=args.requests, mode=args.arrival_mode,
                                rate=args.request_rate, seed=args.seed,
                                context_length=args.context_length)
        validate_trace(trace, context_length=args.context_length)
        replayed_fingerprint = trace["fingerprint"]
        trace["cache_policy"] = ("warm_shared_prefix_after_flush" if args.cache_start == "warm-shared"
                                 else "cold_start_after_warmup_then_prefix_reuse")
        trace["fingerprint"] = sha256_of({k: v for k, v in trace.items() if k != "fingerprint"})
        atomic_write_json(args.run_dir / "workload.json", trace)
        command = build_bf16_command(
            args.model_dir, args.port, args.mem_fraction_static,
            sglang_version="0.5.3", context_length=args.context_length,
            max_running_requests=args.max_running_requests,
            cuda_graph_max_bs=args.cuda_graph_max_bs,
        )
        command += ["--schedule-policy", args.policy, "--enable-metrics",
                    "--enable-cache-report", "--enable-request-time-stats-logging",
                    "--decode-log-interval", "10"]
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                  if key not in ("run_dir", "model_dir", "trace_file")}
        manifest = {
            "schema_version": 1, "model_id": MODEL_ID,
            "model_dir": str(args.model_dir.resolve()), "config": config,
            "workload_fingerprint": trace["fingerprint"], "command": command,
            "replayed_trace_fingerprint": replayed_fingerprint,
            "traffic_fingerprint": sha256_of(trace["requests"]),
            "engine_package": str(Path(importlib.util.find_spec("sglang").origin).parent),
            "engine_identity": engine_identity,
            "environment_overrides": {"SGLANG_LPM_MAX_WAIT_MS": str(args.aging_threshold_ms)},
            "cache_prime_gap_seconds": 0.2 if args.cache_start == "warm-shared" else 0,
            "source_model_hashes": source_file_hashes(args.model_dir),
            "engine_source_hashes": engine_source_hashes(),
            "controller_source_hashes": {
                "scheduling.py": file_sha256(Path(__file__)),
                "serving/scheduling.py": file_sha256(Path(__file__).parent / "serving/scheduling.py"),
                "serving/evaluator.py": file_sha256(Path(__file__).parent / "serving/evaluator.py"),
                "runtime.py": file_sha256(Path(__file__).with_name("runtime.py")),
                "engine.py": file_sha256(Path(__file__).with_name("engine.py")),
            },
            "versions": {name: _package_version(name) for name in
                         ("sglang", "torch", "transformers", "triton")},
            "python": platform.python_version(),
            "metric_poll_interval_seconds": 0.2,
        }
        manifest["fingerprint"] = sha256_of(manifest)
        atomic_write_json(args.run_dir / "manifest.json", manifest)
        baseline, settled = wait_for_clean_gpu()
        if not settled or baseline is None or baseline > 2048:
            raise RuntimeError("GPU did not settle to an idle baseline; measurement refused")
        log_path = args.run_dir / "logs/server.log"
        atomic_write_json(status_path, {"status": "running", "policy": args.policy,
                                        "fingerprint": manifest["fingerprint"]})
        print(json.dumps({"status": "starting_server", "policy": args.policy,
                          "run_dir": str(args.run_dir)}), flush=True)
        with server_session(command, port=args.port, model_id=MODEL_ID,
                            log_path=log_path) as session:
            wait_native_ready(session["base_url"])
            for request in trace["requests"][:3]:
                warmup = {**request, "request_id": "warmup-" + request["request_id"],
                          "max_new_tokens": 16, "arrival_seconds": 0}
                row = send_request(session["base_url"], warmup, origin=time.monotonic(), timeout=120)
                append_jsonl(args.run_dir / "warmup.jsonl", row)
                if row["record"]["status"] != "ok":
                    raise RuntimeError("warmup failed; see warmup.jsonl")
            # Warmup initializes kernels, but must not warm the measured cache.
            flush_attempts = flush_idle_cache(session["base_url"])
            if args.cache_start == "warm-shared":
                source_request = next(r for r in trace["requests"] if r["group"] == "shared-long")
                prime = {**source_request, "request_id": "cache-prime-" + source_request["request_id"],
                         "max_new_tokens": 16, "arrival_seconds": 0}
                row = send_request(session["base_url"], prime, origin=time.monotonic(), timeout=120)
                append_jsonl(args.run_dir / "cache-prime.jsonl", row)
                if row["record"]["status"] != "ok":
                    raise RuntimeError("shared-prefix cache priming failed")
                time.sleep(0.2)
            print(json.dumps({"status": "measuring", "policy": args.policy,
                              "requests": args.requests,
                              "workload_fingerprint": trace["fingerprint"]}), flush=True)
            result = run_trace(session["base_url"], trace, args.run_dir, concurrency=args.concurrency)
            result["cache_flush_attempts"] = flush_attempts
            result["cache_start"] = args.cache_start
            result["warmup_requests"] = 3
            result["cache_prime_requests"] = int(args.cache_start == "warm-shared")
        session["gpu_settled_before_measure_mib"] = baseline
        session["gpu_settled"] = settled
        reasons = resource_blockers(session) + contention_blockers(session)
        if not session.get("gpu_release_confirmed"):
            reasons.append("GPU memory release could not be confirmed")
        result.update(policy=args.policy, fingerprint=manifest["fingerprint"],
                      resource_warnings=resource_warnings(session), server_session=session)
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        result["aging_enabled"] = "Experimental LPM aging enabled:" in log_text
        result["aging_threshold_reached"] = "LPM aging threshold reached:" in log_text
        if args.policy == "lpm-aging" and not result["aging_enabled"]:
            reasons.append("patched aging policy was not confirmed in the server log")
        if reasons:
            result.update(status="blocked", reason="; ".join(reasons))
        result = finalize_result(result, log_path, args.run_dir)
        observation_complete = (result["queue_time_coverage"] == args.requests
                                and result["metrics"]["valid_samples"] > 0)
        atomic_write_json(status_path, {"status": result["status"], "policy": args.policy,
                                        "fingerprint": manifest["fingerprint"],
                                        "observation_status": "complete" if observation_complete else "incomplete",
                                        "result_sha256": file_sha256(args.run_dir / "scheduling-result.json")})
        print(json.dumps({"status": result["status"], "run_dir": str(args.run_dir),
                          "requests_ok": result["summary"]["requests_ok"],
                          "queue_time_coverage": result["queue_time_coverage"],
                          "observation_complete": observation_complete}))
        return 0 if result["status"] == "succeeded" and observation_complete else 1
    except BaseException as exc:
        atomic_write_json(status_path, {"status": "failed", "policy": args.policy,
                                        "reason": f"{type(exc).__name__}: {exc}"})
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        print(json.dumps({"status": "failed", "reason": str(exc),
                          "run_dir": str(args.run_dir)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
