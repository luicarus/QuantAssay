"""Benchmark runner: drive a live SGLang server and persist per-request evidence.

Design constraints that come from the PRD (mvp-prd.md §4, §6):

* warmup requests are sent and discarded, so the measured window excludes
  server-side lazy initialization (kernel autotune, first-batch allocation);
* every request is appended to JSONL as it completes, so an interrupted run
  still leaves usable evidence rather than an empty file;
* the measured wall-clock window is the *timed* section only — including warmup
  would understate throughput;
* failures are recorded, never retried silently: ``retry_policy=none`` means a
  failed request stays failed and shows up in the coverage check.

Only :func:`benchmark` performs network I/O; the request loop is injectable so
CPU tests can run it against a fake opener.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable

from quantassay.contracts import (
    RequestRecord,
    RequestStatus,
    WorkloadSpec,
)
from quantassay.experiments.store import atomic_write_json
from quantassay.serving.evaluator import record_from_stream, summarize_requests
from quantassay.serving.workload import (
    Request,
    build_request_set,
    chat_payload,
    request_set_hashes,
)


class BenchmarkError(RuntimeError):
    """Raised when the benchmark cannot start or the server vanishes."""


def _open_stream(
    url: str,
    payload: dict[str, object],
    timeout: float,
) -> Iterable[bytes]:
    """POST a streaming request and yield SSE lines.

    Kept as a module-level function so tests can monkeypatch it without sockets.
    """
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    response = urllib.request.urlopen(request, timeout=timeout)
    return response  # iterating yields lines


def _run_one(
    base_url: str,
    request: Request,
    *,
    model_id: str,
    max_new_tokens: int,
    temperature: float,
    timeout_seconds: float,
    opener: Callable[[str, dict[str, object], float], Iterable[bytes]],
) -> RequestRecord:
    """Send one request and measure it. Never raises for a request-level failure."""
    url = f"{base_url}/v1/chat/completions"
    payload = chat_payload(
        request,
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    sent_at = time.monotonic()
    try:
        lines = opener(url, payload, timeout_seconds)
        return record_from_stream(request.request_id, lines, sent_at=sent_at)
    except TimeoutError:
        return RequestRecord(
            request_id=request.request_id,
            status=RequestStatus.TIMEOUT,
            failure_reason="timeout",
        )
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:200]
        except Exception:  # pragma: no cover - body may be unreadable
            pass
        code = int(getattr(exc, "code", 0) or 0)
        reason = "timeout" if code in (408, 504) else "http_error"
        return RequestRecord(
            request_id=request.request_id,
            status=RequestStatus.TIMEOUT if reason == "timeout" else RequestStatus.FAILED,
            failure_reason=f"{reason}: HTTP {code} {detail}".strip(),
        )
    except Exception as exc:  # connection reset, malformed SSE, ...
        return RequestRecord(
            request_id=request.request_id,
            status=RequestStatus.FAILED,
            failure_reason=f"{type(exc).__name__}: {str(exc)[:200]}",
        )


def _append_record(path: Path, record: RequestRecord) -> None:
    """Append one record as a JSON line. Flushed per request by the caller."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n")


def benchmark(
    *,
    base_url: str,
    workload: WorkloadSpec,
    model_id: str,
    output_dir: Path,
    side: str,
    sglang_version: str | None = None,
    quant_kernel: str | None = None,
    opener: Callable[[str, dict[str, object], float], Iterable[bytes]] = _open_stream,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, Any], list[RequestRecord]]:
    """Run the workload against a live server; return (summary dump, records).

    ``side`` names the evidence file (``bf16`` / ``gptq``) and the summary's
    side field, so the two runs cannot be confused.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / f"serving-{side}.jsonl"
    if records_path.exists():
        # Overwriting would destroy the evidence this file exists to hold.
        raise BenchmarkError(f"refusing to overwrite existing evidence: {records_path}")

    requests = build_request_set()
    hashes = request_set_hashes(requests)
    expected = workload.request_hashes
    if expected and expected != hashes:
        raise BenchmarkError(
            "workload.request_hashes does not match the built-in request set; "
            "the two sides would not be proven to share inputs"
        )
    if workload.timed_requests > len(requests):
        raise BenchmarkError(
            f"workload asks for {workload.timed_requests} timed requests "
            f"but the request set has {len(requests)}"
        )

    warmup = requests[: min(workload.warmup_requests, len(requests))]
    timed = requests[: workload.timed_requests]

    common = dict(
        model_id=model_id,
        max_new_tokens=workload.max_new_tokens,
        temperature=workload.temperature,
        timeout_seconds=workload.timeout_seconds,
    )

    # Warmup: measured but discarded. Recorded separately so the cost is visible.
    warmup_records: list[RequestRecord] = []
    for request in warmup:
        warmup_records.append(_run_one(base_url, request, opener=opener, **common))

    if workload.request_rate:
        interval = 1.0 / workload.request_rate
    else:
        interval = 0.0

    records: list[RequestRecord] = []
    started = time.monotonic()
    for request in timed:
        records.append(_run_one(base_url, request, opener=opener, **common))
        _append_record(records_path, records[-1])
        if interval:
            sleeper(interval)
    wall_seconds = time.monotonic() - started

    summary = summarize_requests(
        records,
        side=side,
        wall_seconds=wall_seconds,
        sglang_version=sglang_version,
        quant_kernel=quant_kernel,
        workload_id=workload.workload_id,
        workload_fingerprint=workload.workload_fingerprint(),
        records_path=str(records_path),
    )

    dump = {
        "status": "succeeded" if summary.requests_ok else "failed",
        "side": side,
        "records_path": str(records_path),
        "wall_seconds": wall_seconds,
        "warmup_requests": len(warmup),
        "warmup_failures": [r.failure_reason for r in warmup_records if r.failure_reason],
        "summary": summary.model_dump(mode="json"),
    }
    atomic_write_json(output_dir / f"benchmark-{side}.json", dump)
    return dump, records


def load_records(path: str | Path) -> list[RequestRecord]:
    """Read back a JSONL evidence file (used by analysis and tests)."""
    records: list[RequestRecord] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(RequestRecord.model_validate(json.loads(line)))
    return records
