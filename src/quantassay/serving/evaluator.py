"""Serving evaluator: turn raw SGLang streaming responses into measurable evidence.

Responsibilities (mvp-prd.md §4):

* parse the OpenAI-compatible SSE stream, recording when each output token
  chunk arrives (TTFT / ITL need per-token timestamps, not aggregates);
* derive TTFT, ITL list, TPOT and token counts per request;
* classify failures (a request that never produced a token is *failed*, not
  "zero-latency");
* aggregate per-request records into a
  :class:`~quantassay.contracts.ServingSummary` whose throughput numbers use
  the same measured wall-clock window.

The math is pure and CPU-testable; only network I/O lives outside this module.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

from quantassay.contracts import (
    SERVING_METRIC_UNITS,
    SideStatus,
    MetricDirection,
    MetricRecord,
    MetricStatus,
    RequestRecord,
    RequestStatus,
    ServingMetric,
    ServingSummary,
)


# --------------------------------------------------------------------------
# SSE parsing
# --------------------------------------------------------------------------


def parse_sse_payloads(lines: Iterable[bytes]) -> list[dict[str, Any]]:
    """Extract JSON payloads from an OpenAI-compatible SSE byte stream.

    NOTE: this collects the whole stream before returning, so it must not be
    used for latency measurement — by the time it returns, every token has
    arrived and per-token timestamps would be meaningless. Use
    :func:`iter_sse_payloads` for anything that measures time.
    """
    return list(iter_sse_payloads(lines))


def iter_sse_payloads(lines: Iterable[bytes]) -> Iterator[dict[str, Any]]:
    """Stream SSE payloads one at a time, as each line arrives.

    This is the measurement-safe variant: a caller that timestamps inside the
    loop records the moment the server delivered that chunk. Buffering first
    (as the list version does) collapses TTFT into total latency and turns ITL
    into JSON-parse cost.

    ``[DONE]`` markers carry no data and are skipped; malformed JSON raises,
    because silently dropping chunks would corrupt the measurement.
    """
    for line in lines:
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if not data or data == b"[DONE]":
            continue
        try:
            yield json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed SSE payload: {exc}") from exc


# --------------------------------------------------------------------------
# Per-request measurement
# --------------------------------------------------------------------------


@dataclass
class TokenClock:
    """Wall-clock timestamps of chunk arrivals, in seconds."""

    sent_at: float | None = None
    first_token_at: float | None = None
    last_token_at: float | None = None
    completed_at: float | None = None


def _finish_reason_of(event: dict[str, Any]) -> str | None:
    for choice in event.get("choices") or []:
        if choice.get("finish_reason") is not None:
            return str(choice["finish_reason"])
    return None


def _usage_of(event: dict[str, Any]) -> dict[str, Any] | None:
    usage = event.get("usage")
    return usage if isinstance(usage, dict) else None


def _delta_text_of(event: dict[str, Any]) -> str:
    pieces: list[str] = []
    for choice in event.get("choices") or []:
        delta = choice.get("delta") or {}
        if isinstance(delta.get("content"), str):
            pieces.append(delta["content"])
    return "".join(pieces)


def _record_from_stream(
    request_id: str,
    payload_count: int,
    any_payload: bool,
    finish_reason: str | None,
    usage: dict[str, Any] | None,
    clock: TokenClock,
    input_tokens: int | None,
) -> RequestRecord:
    """Derive the RequestRecord; every latency comes from measured timestamps."""
    if clock.first_token_at is None or clock.sent_at is None:
        # No output token ever arrived: a failure, never a fast success.
        # "connection_error" (server sent nothing at all) and "no_first_token"
        # (server answered but produced no content) lead to different
        # investigations, so they are kept distinct.
        reason = "connection_error" if not any_payload else "no_first_token"
        return RequestRecord(
            request_id=request_id,
            status=RequestStatus.FAILED,
            failure_reason=reason,
            output_tokens=0,
            input_tokens=input_tokens or 0,
            usage_available=usage is not None,
        )

    ttft_ms = (clock.first_token_at - clock.sent_at) * 1000.0
    # End-to-end is what the client waited: the stream closing, not the last
    # content chunk (a trailing usage/finish frame still costs time).
    ended_at = clock.completed_at if clock.completed_at is not None else clock.last_token_at
    e2e_ms = (ended_at - clock.sent_at) * 1000.0

    # Token accounting: prefer the server's usage block. Falling back to chunk
    # count is recorded explicitly, because a chunk may merge several tokens —
    # pretending chunk count equals token count would be a lie.
    if usage and isinstance(usage.get("completion_tokens"), int):
        output_tokens = int(usage["completion_tokens"])
        usage_available = True
    else:
        output_tokens = payload_count
        usage_available = False

    total_tokens: int | None = None
    if usage and isinstance(usage.get("total_tokens"), int):
        total_tokens = int(usage["total_tokens"])

    record = RequestRecord(
        request_id=request_id,
        status=RequestStatus.OK,
        ttft_ms=round(ttft_ms, 3),
        e2e_latency_ms=round(e2e_ms, 3),
        output_tokens=output_tokens,
        input_tokens=input_tokens or 0,
        total_tokens=total_tokens,
        finish_reason=finish_reason,
        stopped_on_eos=finish_reason == "stop",
        truncated=finish_reason == "length",
        usage_available=usage_available,
    )
    return record


def record_from_stream(
    request_id: str,
    lines: Iterable[bytes],
    *,
    sent_at: float | None = None,
    input_tokens: int | None = None,
    clock_fn: Any = time.monotonic,
) -> RequestRecord:
    """Measure one streaming response.

    ``clock_fn`` is injectable so tests can replay deterministic timings.
    Timestamps are taken as chunks arrive — TTFT/ITL cannot be reconstructed
    afterwards from an aggregated payload.
    """
    clock = TokenClock()
    clock.sent_at = clock_fn() if sent_at is None else sent_at

    payload_count = 0
    any_payload = False
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    gaps: list[float] = []
    previous_token_at: float | None = None

    for event in iter_sse_payloads(lines):
        any_payload = True
        text = _delta_text_of(event)
        if (u := _usage_of(event)) is not None:
            usage = u
        if (fr := _finish_reason_of(event)) is not None:
            finish_reason = fr
        if not text:
            continue

        now = clock_fn()
        payload_count += 1
        if clock.first_token_at is None:
            clock.first_token_at = now
        else:
            gaps.append(now - previous_token_at)
        previous_token_at = now
        clock.last_token_at = now

    clock.completed_at = clock_fn()

    record = _record_from_stream(
        request_id, payload_count, any_payload, finish_reason, usage, clock, input_tokens
    )
    record.itl_ms = [round(gap * 1000.0, 3) for gap in gaps]
    return record


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile over a list (sorted internally)."""
    if not values:
        raise ValueError("percentile of empty list")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def _latency_metric(
    name: str,
    values: list[float],
    unit: str,
    sample_count: int,
    unavailable_reason: str,
) -> MetricRecord:
    """p50 as the headline value with p95 kept in the reason field."""
    if not values:
        return MetricRecord.unavailable(name, unavailable_reason)
    return MetricRecord(
        name=name,
        value=round(percentile(values, 0.5), 3),
        unit=unit,
        direction=MetricDirection.LOWER_IS_BETTER,
        sample_count=sample_count,
        status=MetricStatus.OK,
        reason=f"p50 shown; p95={round(percentile(values, 0.95), 3)}; n={len(values)}",
    )


def summarize_requests(
    records: list[RequestRecord],
    *,
    side: str,
    wall_seconds: float | None,
    sglang_version: str | None = None,
    run_mode: Any = None,
    quant_kernel: str | None = None,
    workload_id: str | None = None,
    workload_fingerprint: str | None = None,
    records_path: str | None = None,
) -> ServingSummary:
    """Aggregate per-request records into a ServingSummary.

    Throughput uses the measured wall-clock window and only *successful*
    requests; latency percentiles come from per-request values, never from
    batch means.
    """
    ok_records = [r for r in records if r.status is RequestStatus.OK]
    failed = sum(1 for r in records if r.status is RequestStatus.FAILED)
    timed_out = sum(1 for r in records if r.status is RequestStatus.TIMEOUT)
    output_tokens_total = sum(r.output_tokens for r in ok_records)
    input_tokens_total = sum(r.input_tokens for r in ok_records)

    # Length/termination disclosure (mvp-prd.md §5): percentages are only
    # meaningful if both sides decoded comparable lengths for comparable reasons.
    output_lengths = [r.output_tokens for r in ok_records]
    finish_reasons: dict[str, int] = {}
    for record in ok_records:
        key = record.finish_reason or "unknown"
        finish_reasons[key] = finish_reasons.get(key, 0) + 1

    metrics: dict[str, MetricRecord] = {}

    ttft_values = [r.ttft_ms for r in ok_records if r.ttft_ms is not None]
    metrics[ServingMetric.TTFT.value] = _latency_metric(
        ServingMetric.TTFT.value,
        ttft_values,
        SERVING_METRIC_UNITS[ServingMetric.TTFT],
        len(ttft_values),
        "no successful requests to aggregate",
    )

    # TPOT: unavailable below 2 output tokens by definition. A request with
    # >= 2 tokens but missing E2E is a data problem and counts as excluded.
    tpot_values = [r.tpot_ms for r in ok_records if r.tpot_ms is not None]
    if ok_records and not tpot_values:
        metrics[ServingMetric.TPOT.value] = MetricRecord.unavailable(
            ServingMetric.TPOT.value,
            "no request produced >= 2 output tokens (or E2E latency missing)",
        )
    else:
        metrics[ServingMetric.TPOT.value] = _latency_metric(
            ServingMetric.TPOT.value,
            tpot_values,
            SERVING_METRIC_UNITS[ServingMetric.TPOT],
            len(tpot_values),
            "no successful requests to aggregate",
        )

    itl_values = [gap for r in ok_records for gap in r.itl_ms]
    if ok_records and not itl_values:
        metrics[ServingMetric.ITL.value] = MetricRecord.unavailable(
            ServingMetric.ITL.value,
            "no inter-token gaps recorded (stream chunks may merge tokens)",
        )
    else:
        metrics[ServingMetric.ITL.value] = _latency_metric(
            ServingMetric.ITL.value,
            itl_values,
            SERVING_METRIC_UNITS[ServingMetric.ITL],
            len(itl_values),
            "no successful requests to aggregate",
        )

    # Throughput: the measured window over successful output tokens only.
    if wall_seconds is not None and wall_seconds > 0:
        metrics[ServingMetric.TOKENS_PER_SEC.value] = MetricRecord(
            name=ServingMetric.TOKENS_PER_SEC.value,
            value=round(output_tokens_total / wall_seconds, 3),
            unit=SERVING_METRIC_UNITS[ServingMetric.TOKENS_PER_SEC],
            direction=MetricDirection.HIGHER_IS_BETTER,
            sample_count=len(ok_records),
            token_count=output_tokens_total,
            status=MetricStatus.OK,
        )
        metrics[ServingMetric.REQUESTS_PER_SEC.value] = MetricRecord(
            name=ServingMetric.REQUESTS_PER_SEC.value,
            value=round(len(ok_records) / wall_seconds, 4),
            unit=SERVING_METRIC_UNITS[ServingMetric.REQUESTS_PER_SEC],
            direction=MetricDirection.HIGHER_IS_BETTER,
            sample_count=len(ok_records),
            status=MetricStatus.OK,
        )
    else:
        metrics[ServingMetric.TOKENS_PER_SEC.value] = MetricRecord.unavailable(
            ServingMetric.TOKENS_PER_SEC.value, "no measured wall-clock window"
        )
        metrics[ServingMetric.REQUESTS_PER_SEC.value] = MetricRecord.unavailable(
            ServingMetric.REQUESTS_PER_SEC.value, "no measured wall-clock window"
        )

    return ServingSummary(
        side=side,
        status=SideStatus.OK if ok_records else SideStatus.FAILED,
        reason=None if ok_records else "no request produced an output token",
        workload_id=workload_id,
        workload_fingerprint=workload_fingerprint,
        sglang_version=sglang_version,
        run_mode=run_mode,
        quant_kernel=quant_kernel,
        requests_total=len(records),
        requests_ok=len(ok_records),
        requests_failed=failed,
        requests_timeout=timed_out,
        wall_seconds=wall_seconds,
        input_tokens_total=input_tokens_total,
        output_tokens_total=output_tokens_total,
        output_tokens_min=min(output_lengths) if output_lengths else None,
        output_tokens_p50=(
            round(percentile([float(n) for n in output_lengths], 0.5), 1)
            if output_lengths
            else None
        ),
        output_tokens_max=max(output_lengths) if output_lengths else None,
        finish_reason_counts=finish_reasons,
        truncated_requests=sum(1 for r in ok_records if r.truncated),
        stopped_on_eos=sum(1 for r in ok_records if r.stopped_on_eos),
        metrics=metrics,
        records_path=records_path,
    )
