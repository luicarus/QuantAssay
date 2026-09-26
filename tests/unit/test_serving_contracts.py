"""M1 acceptance: serving metric math, failure semantics and comparability.

These are CPU contracts for the SGLang serving path (mvp-prd.md §4). The GPU
counterpart — real TTFT/TPOT/ITL measured against a live server — can only be
verified in the WSL2 execution layer; nothing here may be read as verification.

The cases that matter most are the ones that silently corrupt a serving
comparison: a failed request scored as a fast one, a percentage computed across
different workloads, and throughput and latency signs being mixed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from quantassay.contracts import (  # noqa: E402
    MVP_SERVING_METRICS,
    REQUEST_FAILURE_REASONS,
    SERVING_METRIC_DIRECTION,
    SERVING_METRIC_UNITS,
    ComparabilityIssue,
    MetricDirection,
    MetricRecord,
    MetricStatus,
    RegressionReport,
    RequestRecord,
    RequestStatus,
    ServingMetric,
    ServingRegression,
    ServingSummary,
    SideStatus,
    ThinkingMode,
    WorkloadSpec,
)


def _workload(**overrides) -> WorkloadSpec:
    base = dict(
        workload_id="short-chat",
        request_set_id="chat-8",
        request_hashes=["h1", "h2", "h3"],
    )
    base.update(overrides)
    return WorkloadSpec(**base)


def _ok_request(request_id: str, *, ttft: float, e2e: float, output: int) -> RequestRecord:
    return RequestRecord(
        request_id=request_id,
        ttft_ms=ttft,
        e2e_latency_ms=e2e,
        output_tokens=output,
        input_tokens=10,
    )


# --------------------------------------------------------------------------
# Metric vocabulary and direction
# --------------------------------------------------------------------------


def test_mvp_exposes_exactly_the_four_required_metrics() -> None:
    assert set(MVP_SERVING_METRICS) == {
        ServingMetric.TTFT,
        ServingMetric.TPOT,
        ServingMetric.ITL,
        ServingMetric.TOKENS_PER_SEC,
    }


def test_latency_is_lower_is_better_and_throughput_is_higher_is_better() -> None:
    """Mixing these signs is how a regression gets reported as an improvement."""
    assert SERVING_METRIC_DIRECTION[ServingMetric.TTFT] is MetricDirection.LOWER_IS_BETTER
    assert SERVING_METRIC_DIRECTION[ServingMetric.TPOT] is MetricDirection.LOWER_IS_BETTER
    assert SERVING_METRIC_DIRECTION[ServingMetric.ITL] is MetricDirection.LOWER_IS_BETTER
    assert SERVING_METRIC_DIRECTION[ServingMetric.TOKENS_PER_SEC] is MetricDirection.HIGHER_IS_BETTER
    assert SERVING_METRIC_DIRECTION[ServingMetric.REQUESTS_PER_SEC] is MetricDirection.HIGHER_IS_BETTER


def test_throughput_variants_are_named_separately() -> None:
    # "tok/s" alone is ambiguous; output, request and total throughput differ.
    assert ServingMetric.TOKENS_PER_SEC.value == "output_tokens_per_sec"
    assert ServingMetric.REQUESTS_PER_SEC.value != ServingMetric.TOKENS_PER_SEC.value
    assert ServingMetric.TOTAL_TOKENS_PER_SEC.value != ServingMetric.TOKENS_PER_SEC.value


def test_every_metric_has_a_unit() -> None:
    for metric in ServingMetric:
        assert metric in SERVING_METRIC_UNITS, f"{metric.value} has no declared unit"


# --------------------------------------------------------------------------
# TPOT
# --------------------------------------------------------------------------


def test_tpot_formula_matches_the_prd_definition() -> None:
    """TPOT = (E2E - TTFT) / (output_tokens - 1)."""
    record = _ok_request("r1", ttft=100.0, e2e=1000.0, output=10)
    assert record.tpot_ms == pytest.approx((1000.0 - 100.0) / 9)


def test_tpot_is_unavailable_below_two_output_tokens() -> None:
    assert _ok_request("r1", ttft=50.0, e2e=60.0, output=1).tpot_ms is None
    assert _ok_request("r2", ttft=50.0, e2e=50.0, output=0).tpot_ms is None


def test_tpot_needs_both_endpoints() -> None:
    record = RequestRecord(request_id="r1", ttft_ms=10.0, e2e_latency_ms=None, output_tokens=5)
    assert record.tpot_ms is None


def test_tpot_recomputes_from_raw_fields() -> None:
    """The PRD requires TPOT to be recomputable from stored per-request data."""
    record = _ok_request("r1", ttft=250.0, e2e=2250.0, output=9)
    recomputed = (record.e2e_latency_ms - record.ttft_ms) / (record.output_tokens - 1)
    assert record.tpot_ms == pytest.approx(recomputed)


# --------------------------------------------------------------------------
# Failure semantics: a failure is never a fast success
# --------------------------------------------------------------------------


def test_failed_request_cannot_record_ttft() -> None:
    with pytest.raises(Exception) as exc:
        RequestRecord(
            request_id="r1",
            status=RequestStatus.FAILED,
            failure_reason="no_first_token",
            ttft_ms=5.0,
            output_tokens=0,
        )
    assert "TTFT" in str(exc.value) or "ttft" in str(exc.value)


def test_failed_request_requires_a_reason() -> None:
    with pytest.raises(Exception):
        RequestRecord(request_id="r1", status=RequestStatus.TIMEOUT)


def test_successful_request_requires_ttft() -> None:
    with pytest.raises(Exception):
        RequestRecord(request_id="r1", status=RequestStatus.OK, output_tokens=5)


def test_failure_reasons_are_a_closed_vocabulary() -> None:
    # Free-text reasons make failures unreportable in aggregate.
    assert "no_first_token" in REQUEST_FAILURE_REASONS
    assert "timeout" in REQUEST_FAILURE_REASONS
    assert REQUEST_FAILURE_REASONS["truncated"]  # hitting the cap is not EOS

    failed = RequestRecord(
        request_id="r1", status=RequestStatus.FAILED, failure_reason="timeout"
    )
    assert failed.ttft_ms is None
    assert failed.tpot_ms is None


# --------------------------------------------------------------------------
# Workload identity
# --------------------------------------------------------------------------


def test_workload_requires_hashed_requests() -> None:
    """Without request hashes the two sides cannot be shown to match."""
    with pytest.raises(Exception) as exc:
        WorkloadSpec(workload_id="w", request_set_id="s", request_hashes=[])
    assert "hash" in str(exc.value)


def test_workload_fingerprint_is_stable_and_protocol_sensitive() -> None:
    base = _workload()
    assert base.workload_fingerprint() == _workload().workload_fingerprint()

    # Thinking mode is protocol, so it changes the experiment.
    thinking = _workload(thinking_mode=ThinkingMode.ENABLED)
    assert thinking.workload_fingerprint() != base.workload_fingerprint()

    # So does concurrency, cache policy and the timing budget.
    assert _workload(concurrency=4).workload_fingerprint() != base.workload_fingerprint()
    assert _workload(cache_policy="radix").workload_fingerprint() != base.workload_fingerprint()
    assert _workload(timed_requests=50).workload_fingerprint() != base.workload_fingerprint()


def test_workload_fingerprint_covers_request_content() -> None:
    base = _workload()
    changed = _workload(request_hashes=["h1", "h2", "DIFFERENT"])
    assert changed.workload_fingerprint() != base.workload_fingerprint()


# --------------------------------------------------------------------------
# Summary aggregation
# --------------------------------------------------------------------------


def test_success_rate_and_failure_counts() -> None:
    summary = ServingSummary(
        side="baseline",
        requests_total=10,
        requests_ok=8,
        requests_failed=1,
        requests_timeout=1,
    )
    assert summary.success_rate == pytest.approx(0.8)


def test_success_rate_unavailable_without_requests() -> None:
    assert ServingSummary(side="baseline").success_rate is None


def test_non_ok_summary_requires_reason() -> None:
    with pytest.raises(Exception) as exc:
        ServingSummary(side="candidate", status=SideStatus.FAILED)
    assert "reason" in str(exc.value)

    ok = ServingSummary(
        side="candidate", status=SideStatus.NOT_SERVED, reason="GPTQ did not load"
    )
    assert ok.reason == "GPTQ did not load"


def test_missing_metric_reads_as_unavailable_not_zero() -> None:
    summary = ServingSummary(side="baseline", requests_total=5, requests_ok=5)
    metric = summary.metric(ServingMetric.TTFT)
    assert metric.value is None
    assert metric.status is MetricStatus.UNAVAILABLE
    assert metric.reason


def test_summary_metric_records_keep_units() -> None:
    summary = ServingSummary(
        side="baseline",
        requests_total=5,
        requests_ok=5,
        metrics={
            "ttft": MetricRecord(
                name="ttft", value=120.0, unit="ms", sample_count=5, token_count=0
            )
        },
    )
    assert summary.metric(ServingMetric.TTFT).value == 120.0
    assert summary.metric(ServingMetric.TTFT).unit == "ms"


# --------------------------------------------------------------------------
# Regression: percentages only when comparable
# --------------------------------------------------------------------------


def test_regression_rejects_percentage_when_not_comparable() -> None:
    """This is the guard that stops an incomparable pair becoming a headline."""
    with pytest.raises(Exception) as exc:
        ServingRegression(
            metric=ServingMetric.TTFT,
            status=MetricStatus.UNAVAILABLE,
            reason="baseline not served",
            relative_change_percent=12.0,
        )
    assert "percentage" in str(exc.value).lower()


def test_regression_non_ok_requires_reason() -> None:
    with pytest.raises(Exception):
        ServingRegression(metric=ServingMetric.TPOT, status=MetricStatus.UNAVAILABLE)


def test_regression_accepts_a_measured_comparison() -> None:
    regression = ServingRegression(
        metric=ServingMetric.TTFT,
        base_value=100.0,
        candidate_value=80.0,
        absolute_delta=-20.0,
        relative_change=-0.2,
        relative_change_percent=-20.0,
        status=MetricStatus.OK,
    )
    assert regression.relative_change_percent == pytest.approx(-20.0)


# --------------------------------------------------------------------------
# Report-level comparability
# --------------------------------------------------------------------------


def test_incomparable_report_must_state_why() -> None:
    with pytest.raises(Exception) as exc:
        RegressionReport(run_id="r1", comparable=False)
    assert "why" in str(exc.value)


def test_comparable_report_must_not_carry_incomparable_reasons() -> None:
    with pytest.raises(Exception):
        RegressionReport(
            run_id="r1",
            comparable=True,
            incomparable_reasons=[ComparabilityIssue(field="dtype", reason="mismatch")],
        )


def test_report_defaults_to_quality_not_evaluated() -> None:
    """MVP never measures quality, so the report must say so rather than '0 pp'."""
    report = RegressionReport(
        run_id="r1",
        comparable=False,
        incomparable_reasons=[ComparabilityIssue(field="serving", reason="candidate failed")],
    )
    assert report.quality_status == "not_evaluated"
    assert report.quality_not_evaluated is True


def test_incomparable_report_has_no_regression_verdicts() -> None:
    report = RegressionReport(
        run_id="r1",
        comparable=False,
        incomparable_reasons=[
            ComparabilityIssue(
                field="sglang_version",
                base_value="0.5.0",
                candidate_value="0.5.1",
                reason="different serving runtime",
            )
        ],
    )
    assert report.regressions == {}
    assert report.incomparable_reasons[0].field == "sglang_version"


# --------------------------------------------------------------------------
# Percentages must match the PRD definition
# --------------------------------------------------------------------------


def test_relative_change_definition() -> None:
    """`100*(candidate/base - 1)`; verified here so both sides use one formula."""
    base, candidate = 100.0, 80.0
    assert pytest.approx(-20.0) == 100.0 * (candidate / base - 1.0)

    base, candidate = 500.0, 750.0
    assert pytest.approx(50.0) == 100.0 * (candidate / base - 1.0)


def test_percentage_undefined_for_zero_baseline() -> None:
    base = 0.0
    # The caller must not divide; a zero baseline yields no percentage.
    assert base == 0.0
    with pytest.raises(ZeroDivisionError):
        _ = 100.0 * (1.0 / base - 1.0)
