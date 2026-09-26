"""M2/M3 acceptance: serving evaluator measurement and aggregation.

These are CPU contracts for the mvp-prd.md §4 metric definitions. The network
path is abstracted: tests feed canned SSE bytes and an injected clock, which is
what makes deterministic measurement contracts possible at all.

The failure cases matter as much as the success math: a request that never
produced a token must not become a fast success, and chunk-count must never be
passed off as token count.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from quantassay.contracts import (  # noqa: E402
    MetricStatus,
    RequestRecord,
    RequestStatus,
    ServingMetric,
    ServingSummary,
)
from quantassay.serving import (  # noqa: E402
    iter_sse_payloads,
    parse_sse_payloads,
    percentile,
    record_from_stream,
    summarize_requests,
)


def _sse(*payloads: str) -> list[bytes]:
    """Build SSE bytes from JSON payload strings."""
    lines = []
    for payload in payloads:
        lines.append(f"data: {payload}\n".encode("utf-8"))
    lines.append(b"data: [DONE]\n")
    return lines


def _chunk(text: str, **extra) -> str:
    import json

    return json.dumps({"choices": [{"delta": {"content": text}}]})


def _final(finish: str, completion_tokens: int, total: int) -> str:
    import json

    return json.dumps(
        {
            "choices": [{"delta": {}, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": completion_tokens,
                "total_tokens": total,
            },
        }
    )


class FakeClock:
    """Deterministic clock: each call returns the next scripted value.

    Values beyond the script repeat the last one: the completion stamp is an
    extra call, and tests that assert only TTFT/ITL should not have to budget
    a value for it.
    """

    def __init__(self, values: list[float]) -> None:
        self.values = list(values)
        self.index = 0

    def __call__(self) -> float:
        value = self.values[min(self.index, len(self.values) - 1)]
        self.index += 1
        return value


# --------------------------------------------------------------------------
# SSE parsing
# --------------------------------------------------------------------------


def test_parse_sse_skips_done_and_comments() -> None:
    lines = [
        b": keepalive comment\n",
        b'data: {"a": 1}\n',
        b"data: [DONE]\n",
        b'data: {"b": 2}\n',
    ]
    payloads = parse_sse_payloads(lines)
    assert len(payloads) == 2
    assert payloads[0] == {"a": 1}


def test_parse_sse_rejects_malformed_json() -> None:
    with pytest.raises(ValueError, match="malformed"):
        parse_sse_payloads([b"data: {broken\n"])


def test_iter_sse_payloads_yields_before_the_stream_ends() -> None:
    """Timestamps must be taken as chunks arrive, not after buffering the stream.

    A buffered implementation collapses TTFT into total latency and turns ITL
    into JSON-parse cost (measured once as a physically impossible
    0.002 ms/token on real traffic). This drives a generator that records how
    far it has advanced, so a buffering implementation cannot pass.
    """
    produced: list[str] = []

    def blocking_stream():
        yield b'data: {"choices":[{"delta":{"content":"a"}}]}\n'
        produced.append("first-consumed")
        yield b'data: {"choices":[{"delta":{"content":"b"}}]}\n'
        produced.append("second-consumed")
        yield b"data: [DONE]\n"

    seen = []
    for payload in iter_sse_payloads(blocking_stream()):
        seen.append((payload["choices"][0]["delta"]["content"], len(produced)))

    # Each payload was observed before the generator produced the next chunk.
    assert seen == [("a", 0), ("b", 1)]


def test_measurement_reflects_stream_timing_not_parse_speed() -> None:
    """A slow upstream must appear as latency, not as fast parsing."""
    timeline = {"now": 0.0}

    def clock() -> float:
        return timeline["now"]

    def slow_stream():
        for delay, text in ((0.5, "a"), (0.5, "b"), (0.5, "c")):
            timeline["now"] += delay
            yield b'data: {"choices":[{"delta":{"content":"' + text.encode() + b'"}}]}\n'
        timeline["now"] += 0.1
        yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n'

    record = record_from_stream("slow", slow_stream(), sent_at=0.0, clock_fn=clock)

    # First token 0.5 s in; whole response 1.6 s.
    assert record.ttft_ms == pytest.approx(500.0)
    assert record.e2e_latency_ms == pytest.approx(1600.0)
    assert record.itl_ms == [pytest.approx(500.0), pytest.approx(500.0)]
    assert record.tpot_ms == pytest.approx((1600.0 - 500.0) / 2)


# --------------------------------------------------------------------------
# TTFT / ITL / TPOT from scripted timings
# --------------------------------------------------------------------------


def test_ttft_itl_tpot_from_deterministic_stream() -> None:
    """sent=0; token chunks arrive at 0.1s then every 0.02s (last 0.16s)."""
    lines = _sse(
        _chunk("Hel"), _chunk("lo "), _chunk("wor"), _chunk("ld!"),
        _final("stop", completion_tokens=4, total=24),
    )
    clock = FakeClock([0.1, 0.12, 0.14, 0.16])
    record = record_from_stream("r1", lines, sent_at=0.0, clock_fn=clock, input_tokens=20)

    assert record.status is RequestStatus.OK
    assert record.ttft_ms == pytest.approx(100.0)
    assert record.e2e_latency_ms == pytest.approx(160.0)
    # TPOT = (E2E - TTFT) / (n - 1) = 60/3 = 20 ms — matching the 0.02s gaps.
    assert record.tpot_ms == pytest.approx(20.0)
    assert len(record.itl_ms) == 3
    assert record.itl_ms[0] == pytest.approx(20.0)
    assert record.stopped_on_eos is True
    assert record.truncated is False


def test_single_token_request_has_unavailable_tpot_but_valid_ttft() -> None:
    lines = _sse(_chunk("Hi"), _final("stop", completion_tokens=1, total=21))
    clock = FakeClock([0.05, 0.06])
    record = record_from_stream("r1", lines, sent_at=0.0, clock_fn=clock)

    assert record.status is RequestStatus.OK
    assert record.ttft_ms == pytest.approx(50.0)
    assert record.tpot_ms is None  # < 2 output tokens
    assert record.itl_ms == []


def test_zero_output_tokens_is_a_failure_not_fast_success() -> None:
    lines = _sse(_final("stop", completion_tokens=0, total=20))
    clock = FakeClock([0.02])
    record = record_from_stream("r1", lines, sent_at=0.0, clock_fn=clock)

    assert record.status is RequestStatus.FAILED
    assert record.failure_reason == "no_first_token"
    assert record.ttft_ms is None
    assert record.tpot_ms is None


def test_empty_stream_is_connection_error() -> None:
    clock = FakeClock([0.0])
    record = record_from_stream("r1", [], sent_at=0.0, clock_fn=clock)
    assert record.status is RequestStatus.FAILED
    assert record.failure_reason == "connection_error"


def test_usage_token_count_preferred_over_chunk_count() -> None:
    """A chunk may merge several tokens; the usage block is authoritative."""
    lines = _sse(
        _chunk("one two three"),  # 1 chunk, 3 tokens
        _final("stop", completion_tokens=3, total=23),
    )
    clock = FakeClock([0.1, 0.2])
    record = record_from_stream("r1", lines, sent_at=0.0, clock_fn=clock)

    assert record.output_tokens == 3
    assert record.usage_available is True
    # But ITL is still only known per chunk, and this is disclosed.
    assert record.itl_ms == []


def test_missing_usage_marks_chunk_fallback() -> None:
    import json

    lines = [
        b'data: {"choices":[{"delta":{"content":"a"}}]}\n',
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
        b"data: [DONE]\n",
    ]
    clock = FakeClock([0.1, 0.15])
    record = record_from_stream("r1", lines, sent_at=0.0, clock_fn=clock)
    assert record.output_tokens == 1  # chunk fallback
    assert record.usage_available is False


def test_truncated_by_length_is_recorded() -> None:
    lines = _sse(_chunk("abc"), _final("length", completion_tokens=64, total=84))
    clock = FakeClock([0.1, 0.2])
    record = record_from_stream("r1", lines, sent_at=0.0, clock_fn=clock)
    assert record.truncated is True
    assert record.stopped_on_eos is False


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def _ok(request_id: str, ttft: float, e2e: float, output: int) -> RequestRecord:
    return RequestRecord(
        request_id=request_id,
        ttft_ms=ttft,
        e2e_latency_ms=e2e,
        output_tokens=output,
        input_tokens=10,
        itl_ms=[ttft] * max(output - 1, 0),
    )


def test_summary_throughput_uses_wall_clock_and_ok_requests_only() -> None:
    records = [
        _ok("a", 100.0, 200.0, 10),
        _ok("b", 120.0, 260.0, 20),
        RequestRecord(
            request_id="c", status=RequestStatus.FAILED, failure_reason="no_first_token"
        ),
    ]
    summary = summarize_requests(records, side="candidate", wall_seconds=1.5)

    assert summary.requests_ok == 2
    assert summary.requests_failed == 1
    assert summary.output_tokens_total == 30
    tok = summary.metric(ServingMetric.TOKENS_PER_SEC)
    assert tok.value == pytest.approx(30 / 1.5)
    req = summary.metric(ServingMetric.REQUESTS_PER_SEC)
    # The summary rounds to 4 decimals; compare against the same rounding.
    assert req.value == pytest.approx(round(2 / 1.5, 4))


def test_summary_latency_p50_and_p95_in_reason() -> None:
    records = [
        _ok("a", 100.0, 200.0, 5),
        _ok("b", 200.0, 300.0, 5),
        _ok("c", 300.0, 400.0, 5),
        _ok("d", 400.0, 500.0, 5),
    ]
    summary = summarize_requests(records, side="base", wall_seconds=10.0)
    ttft = summary.metric(ServingMetric.TTFT)
    assert ttft.value == pytest.approx(300.0)  # p50 by nearest rank
    assert "p95=" in ttft.reason
    assert ttft.unit == "ms"


def test_summary_without_wall_clock_leaves_throughput_unavailable() -> None:
    records = [_ok("a", 100.0, 200.0, 10)]
    summary = summarize_requests(records, side="base", wall_seconds=None)
    tok = summary.metric(ServingMetric.TOKENS_PER_SEC)
    assert tok.value is None
    assert tok.status is MetricStatus.UNAVAILABLE
    # Latency metrics remain available.
    assert summary.metric(ServingMetric.TTFT).value == 100.0


def test_summary_all_failed_side_is_failed_with_reason() -> None:
    records = [
        RequestRecord(
            request_id="a", status=RequestStatus.FAILED, failure_reason="timeout"
        )
    ]
    summary = summarize_requests(records, side="candidate", wall_seconds=5.0)
    assert summary.status.value == "failed"
    assert summary.reason


def test_summary_tpot_unavailable_when_all_single_token() -> None:
    records = [_ok("a", 100.0, 110.0, 1), _ok("b", 120.0, 130.0, 1)]
    summary = summarize_requests(records, side="base", wall_seconds=1.0)
    tpot = summary.metric(ServingMetric.TPOT)
    assert tpot.value is None
    assert "2 output tokens" in tpot.reason


def test_percentile_nearest_rank() -> None:
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 0.0) == 10.0
    assert percentile(values, 0.5) == 30.0  # round(0.5*3)=2 -> 30
    assert percentile(values, 1.0) == 40.0
    with pytest.raises(ValueError):
        percentile([], 0.5)


# --------------------------------------------------------------------------
# Round-trip: records -> summary -> regression-ready contract objects
# --------------------------------------------------------------------------


def test_summary_survives_json_roundtrip() -> None:
    import json

    from quantassay.contracts import ServingSummary

    records = [_ok("a", 100.0, 200.0, 10)]
    summary = summarize_requests(records, side="base", wall_seconds=2.0)
    payload = json.loads(json.dumps(summary.model_dump(mode="json")))
    reloaded = ServingSummary.model_validate(payload)
    assert reloaded.metric(ServingMetric.TTFT).value == pytest.approx(100.0)
    assert reloaded.requests_ok == 1
