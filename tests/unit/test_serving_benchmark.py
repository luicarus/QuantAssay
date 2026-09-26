"""M2 acceptance: the benchmark runner against a fake SSE server.

The runner is the piece that turns a live server into comparable evidence, so
its failure modes matter more than its happy path:

* warmup must not pollute the measured window;
* a failed request must stay failed (no silent retry, no fabricated TTFT);
* evidence must be appended per request so an interrupted run is still usable;
* an existing evidence file must never be overwritten.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from quantassay.contracts import (  # noqa: E402
    RequestStatus,
    WorkloadSpec,
)
from quantassay.serving import (  # noqa: E402
    BenchmarkError,
    benchmark,
    build_request_set,
    load_records,
    request_set_hashes,
)
from quantassay.serving import benchmark as bench_mod  # noqa: E402


def _workload(**overrides) -> WorkloadSpec:
    requests = build_request_set()
    base = dict(
        workload_id="mvp-short-chat",
        request_set_id="builtin-short-chat",
        request_hashes=request_set_hashes(requests),
        warmup_requests=2,
        timed_requests=5,
        max_new_tokens=8,
        timeout_seconds=5.0,
    )
    base.update(overrides)
    return WorkloadSpec(**base)


def _sse_response(tokens: int = 4) -> list[bytes]:
    """A well-formed streaming response with `tokens` content chunks."""
    lines = []
    for i in range(tokens):
        payload = {"choices": [{"delta": {"content": f"t{i}"}}]}
        lines.append(f"data: {json.dumps(payload)}\n".encode())
    final = {
        "choices": [{"delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": tokens, "total_tokens": 10 + tokens},
    }
    lines.append(f"data: {json.dumps(final)}\n".encode())
    lines.append(b"data: [DONE]\n")
    return lines


class FakeOpener:
    """Records calls and optionally fails selected request ids."""

    def __init__(self, *, fail_on: set[int] | None = None, clock_step: float = 0.01) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.fail_on = fail_on or set()
        self.clock_step = clock_step

    def __call__(self, url, payload, timeout):
        index = len(self.calls)
        self.calls.append((url, payload))
        if index in self.fail_on:
            raise ConnectionResetError("simulated reset")
        return _sse_response()


def test_benchmark_writes_one_jsonl_line_per_timed_request(tmp_path: Path) -> None:
    opener = FakeOpener()
    workload = _workload(warmup_requests=2, timed_requests=5)
    dump, records = benchmark(
        base_url="http://127.0.0.1:30000",
        workload=workload,
        model_id="Qwen/Qwen3-0.6B",
        output_dir=tmp_path,
        side="bf16",
        opener=opener,
    )

    # 2 warmup + 5 timed requests reached the server.
    assert len(opener.calls) == 7
    assert len(records) == 5
    assert dump["status"] == "succeeded"
    assert dump["warmup_requests"] == 2

    lines = (tmp_path / "serving-bf16.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    reloaded = load_records(tmp_path / "serving-bf16.jsonl")
    assert [r.request_id for r in reloaded] == [r.request_id for r in records]


def test_warmup_is_excluded_from_the_measured_window(tmp_path: Path) -> None:
    """Throughput must be computed over timed requests only."""
    opener = FakeOpener()
    workload = _workload(warmup_requests=3, timed_requests=4)
    dump, records = benchmark(
        base_url="http://127.0.0.1:30000",
        workload=workload,
        model_id="Qwen/Qwen3-0.6B",
        output_dir=tmp_path,
        side="bf16",
        opener=opener,
    )
    summary = dump["summary"]
    assert summary["requests_total"] == 4
    assert summary["requests_ok"] == 4
    # Output tokens counted from the 4 timed requests only, not the 3 warmups.
    assert summary["output_tokens_total"] == 4 * 4


def test_failed_request_stays_failed_without_ttft(tmp_path: Path) -> None:
    opener = FakeOpener(fail_on={2, 3})  # first timed request fails (after 2 warmups)
    workload = _workload(warmup_requests=2, timed_requests=5)
    dump, records = benchmark(
        base_url="http://127.0.0.1:30000",
        workload=workload,
        model_id="Qwen/Qwen3-0.6B",
        output_dir=tmp_path,
        side="gptq",
        opener=opener,
    )
    failed = [r for r in records if r.status is not RequestStatus.OK]
    assert len(failed) == 2
    for record in failed:
        assert record.ttft_ms is None
        assert record.failure_reason
    assert dump["summary"]["requests_failed"] == 2
    # Coverage is below the comparability floor, and the summary must say so.
    assert dump["summary"]["requests_ok"] == 3


def test_benchmark_refuses_to_overwrite_existing_evidence(tmp_path: Path) -> None:
    (tmp_path / "serving-bf16.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(BenchmarkError, match="refusing to overwrite"):
        benchmark(
            base_url="http://127.0.0.1:30000",
            workload=_workload(),
            model_id="Qwen/Qwen3-0.6B",
            output_dir=tmp_path,
            side="bf16",
            opener=FakeOpener(),
        )


def test_workload_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    """A workload claiming different inputs must not be measured silently."""
    workload = _workload(request_hashes=["deadbeef"])
    with pytest.raises(BenchmarkError, match="does not match the built-in request set"):
        benchmark(
            base_url="http://127.0.0.1:30000",
            workload=workload,
            model_id="Qwen/Qwen3-0.6B",
            output_dir=tmp_path,
            side="bf16",
            opener=FakeOpener(),
        )


def test_more_timed_requests_than_the_set_is_rejected(tmp_path: Path) -> None:
    workload = _workload(timed_requests=999)
    with pytest.raises(BenchmarkError, match="request set has"):
        benchmark(
            base_url="http://127.0.0.1:30000",
            workload=workload,
            model_id="Qwen/Qwen3-0.6B",
            output_dir=tmp_path,
            side="bf16",
            opener=FakeOpener(),
        )


def test_request_payload_disables_thinking_and_streams(tmp_path: Path) -> None:
    opener = FakeOpener()
    benchmark(
        base_url="http://127.0.0.1:30000",
        workload=_workload(warmup_requests=0, timed_requests=1),
        model_id="Qwen/Qwen3-0.6B",
        output_dir=tmp_path,
        side="bf16",
        opener=opener,
    )
    _, payload = opener.calls[0]
    assert payload["stream"] is True
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["temperature"] == 0.0
    assert payload["model"] == "Qwen/Qwen3-0.6B"


def test_evidence_survives_a_crash_between_requests(tmp_path: Path) -> None:
    """Each record is flushed as it completes, so a partial run is still evidence."""

    class ExplodingOpener(FakeOpener):
        def __call__(self, url, payload, timeout):
            if len(self.calls) >= 3:
                raise KeyboardInterrupt("simulated interrupt")
            return super().__call__(url, payload, timeout)

    with pytest.raises(KeyboardInterrupt):
        benchmark(
            base_url="http://127.0.0.1:30000",
            workload=_workload(warmup_requests=0, timed_requests=5),
            model_id="Qwen/Qwen3-0.6B",
            output_dir=tmp_path,
            side="bf16",
            opener=ExplodingOpener(),
        )

    # Three completed requests were persisted before the interrupt.
    reloaded = load_records(tmp_path / "serving-bf16.jsonl")
    assert len(reloaded) == 3


def test_timeout_is_classified_as_timeout(tmp_path: Path) -> None:
    class TimeoutOpener(FakeOpener):
        def __call__(self, url, payload, timeout):
            self.calls.append((url, payload))
            raise TimeoutError("slow")

    dump, records = benchmark(
        base_url="http://127.0.0.1:30000",
        workload=_workload(warmup_requests=0, timed_requests=2),
        model_id="Qwen/Qwen3-0.6B",
        output_dir=tmp_path,
        side="gptq",
        opener=TimeoutOpener(),
    )
    assert all(r.status is RequestStatus.TIMEOUT for r in records)
    assert dump["summary"]["requests_timeout"] == 2
    assert dump["status"] == "failed"


def test_http_error_records_status_code(tmp_path: Path) -> None:
    import urllib.error

    class ErrorOpener(FakeOpener):
        def __call__(self, url, payload, timeout):
            self.calls.append((url, payload))
            raise urllib.error.HTTPError(url, 400, "Bad Request", {}, None)

    dump, records = benchmark(
        base_url="http://127.0.0.1:30000",
        workload=_workload(warmup_requests=0, timed_requests=1),
        model_id="Qwen/Qwen3-0.6B",
        output_dir=tmp_path,
        side="gptq",
        opener=ErrorOpener(),
    )
    assert "HTTP 400" in records[0].failure_reason
    assert records[0].status is RequestStatus.FAILED


def test_request_rate_paces_requests(tmp_path: Path) -> None:
    slept: list[float] = []
    benchmark(
        base_url="http://127.0.0.1:30000",
        workload=_workload(warmup_requests=0, timed_requests=3, request_rate=2.0),
        model_id="Qwen/Qwen3-0.6B",
        output_dir=tmp_path,
        side="bf16",
        opener=FakeOpener(),
        sleeper=slept.append,
    )
    # rate=2/s -> 0.5 s between requests, applied after each measured request.
    assert slept == [0.5, 0.5, 0.5]
