"""Regression analyzer: comparability first, numbers second (mvp-prd.md §5).

The analyzer refuses to invent a percentage when the two sides are not
comparable. Every check corresponds to a way a serving comparison can quietly
become wrong: different runtimes, different workloads, different generation
parameters, unmatched success coverage, or an unverified quantization path.
"""

from __future__ import annotations

from typing import Any

from quantassay.contracts import (
    ComparabilityIssue,
    MetricRecord,
    MetricStatus,
    RegressionReport,
    RequestRecord,
    RequestStatus,
    ServingMetric,
    ServingSummary,
    SideStatus,
)

#: Metrics whose regression is computed as a plain ratio of aggregates.
_THROUGHPUT_METRICS = {ServingMetric.TOKENS_PER_SEC, ServingMetric.REQUESTS_PER_SEC}

#: Minimum fraction of requests that must succeed on both sides before a
#: percentage is meaningful (mvp-prd.md §5: "成功请求覆盖率足够").
MIN_SUCCESS_COVERAGE = 0.8


def _metric_or_none(summary: ServingSummary, metric: ServingMetric) -> MetricRecord | None:
    record = summary.metrics.get(metric.value)
    if record is None or record.status is not MetricStatus.OK or record.value is None:
        return None
    return record


def _workload_issue(
    base: ServingSummary, candidate: ServingSummary
) -> ComparabilityIssue | None:
    if base.workload_fingerprint and candidate.workload_fingerprint:
        if base.workload_fingerprint != candidate.workload_fingerprint:
            return ComparabilityIssue(
                field="workload_fingerprint",
                base_value=base.workload_fingerprint[:16],
                candidate_value=candidate.workload_fingerprint[:16],
                reason="the two sides ran different workloads",
            )
    return None


def _runtime_issue(
    base: ServingSummary, candidate: ServingSummary
) -> ComparabilityIssue | None:
    """SGLang version must match; the quant kernel is *expected* to differ."""
    if base.sglang_version and candidate.sglang_version:
        if base.sglang_version != candidate.sglang_version:
            return ComparabilityIssue(
                field="sglang_version",
                base_value=base.sglang_version,
                candidate_value=candidate.sglang_version,
                reason="different serving runtime versions",
            )
    return None


def _coverage_issue(
    base: ServingSummary, candidate: ServingSummary
) -> ComparabilityIssue | None:
    for side, summary in (("baseline", base), ("candidate", candidate)):
        if summary.requests_total == 0:
            return ComparabilityIssue(
                field="requests_total",
                base_value=str(base.requests_total),
                candidate_value=str(candidate.requests_total),
                reason=f"{side} served no requests",
            )
        rate = summary.requests_ok / summary.requests_total
        if rate < MIN_SUCCESS_COVERAGE:
            return ComparabilityIssue(
                field="success_coverage",
                base_value=f"{base.requests_ok}/{base.requests_total}",
                candidate_value=f"{candidate.requests_ok}/{candidate.requests_total}",
                reason=(
                    f"{side} success rate {rate:.0%} is below the {MIN_SUCCESS_COVERAGE:.0%} "
                    "floor; failed requests would bias latency percentiles"
                ),
            )
    return None


def check_comparability(
    base: ServingSummary, candidate: ServingSummary
) -> list[ComparabilityIssue]:
    """Every reason these two summaries may not be turned into percentages."""
    issues: list[ComparabilityIssue] = []

    if base.status is not SideStatus.OK:
        issues.append(
            ComparabilityIssue(
                field="baseline_status",
                base_value=base.status.value,
                reason=f"baseline did not produce usable evidence: {base.reason or 'unknown'}",
            )
        )
    if candidate.status is not SideStatus.OK:
        issues.append(
            ComparabilityIssue(
                field="candidate_status",
                candidate_value=candidate.status.value,
                reason=f"candidate did not produce usable evidence: {candidate.reason or 'unknown'}",
            )
        )
    if issues:
        # If a whole side is unusable, further checks add no information.
        return issues

    for issue in (
        _workload_issue(base, candidate),
        _runtime_issue(base, candidate),
        _coverage_issue(base, candidate),
        _length_issue(base, candidate),
    ):
        if issue is not None:
            issues.append(issue)
    return issues


#: Relative output-length difference above which the two sides are decoding
#: different amounts of text, making per-token latency incomparable.
MAX_LENGTH_RATIO_DELTA = 0.10


def _length_issue(
    base: ServingSummary, candidate: ServingSummary
) -> ComparabilityIssue | None:
    """Truncation and output length must be disclosed, not averaged away.

    If one side hits ``max_tokens`` while the other stops on EOS, the sides
    decoded under different regimes: TPOT and ITL then describe different work,
    and the throughput difference partly reflects the truncation policy rather
    than the quantization.
    """
    base_p50 = base.output_tokens_p50
    cand_p50 = candidate.output_tokens_p50
    if base_p50 and cand_p50:
        ratio_delta = abs(cand_p50 - base_p50) / base_p50
        if ratio_delta > MAX_LENGTH_RATIO_DELTA:
            return ComparabilityIssue(
                field="output_tokens_p50",
                base_value=str(base_p50),
                candidate_value=str(cand_p50),
                reason=(
                    f"median output length differs by {ratio_delta:.0%} "
                    f"(> {MAX_LENGTH_RATIO_DELTA:.0%}); latency per token is not "
                    "comparable when the sides decode different amounts of text"
                ),
            )
    # Terminate differently is a weaker signal than different length, but it
    # still means the generation policy diverged, so it is reported.
    base_eos, cand_eos = base.stopped_on_eos, candidate.stopped_on_eos
    if base.requests_ok and candidate.requests_ok:
        base_all_truncated = base.truncated_requests == base.requests_ok
        cand_all_truncated = candidate.truncated_requests == candidate.requests_ok
        if base_all_truncated != cand_all_truncated:
            return ComparabilityIssue(
                field="finish_reason",
                base_value=f"truncated {base.truncated_requests}/{base.requests_ok}, eos {base_eos}",
                candidate_value=(
                    f"truncated {candidate.truncated_requests}/{candidate.requests_ok}, "
                    f"eos {cand_eos}"
                ),
                reason=(
                    "one side was entirely truncated by max_tokens while the other "
                    "terminated on its own; generation policy differs"
                ),
            )
    return None


def _regression_for(
    metric: ServingMetric,
    base: ServingSummary,
    candidate: ServingSummary,
) -> Any:
    """Compare one metric; the record decides whether a percentage may appear."""
    from quantassay.contracts import ServingRegression

    base_record = _metric_or_none(base, metric)
    candidate_record = _metric_or_none(candidate, metric)

    if base_record is None or candidate_record is None:
        missing = "baseline" if base_record is None else "candidate"
        return ServingRegression(
            metric=metric,
            status=MetricStatus.UNAVAILABLE,
            reason=f"{missing} did not produce a usable {metric.value} aggregate",
        )

    base_value = float(base_record.value)
    candidate_value = float(candidate_record.value)

    absolute_delta = candidate_value - base_value
    if base_value == 0:
        # A ratio against zero is undefined, not "infinite improvement".
        return ServingRegression(
            metric=metric,
            base_value=base_value,
            candidate_value=candidate_value,
            absolute_delta=round(absolute_delta, 4),
            status=MetricStatus.UNAVAILABLE,
            reason="baseline value is zero; relative change undefined",
        )

    relative = candidate_value / base_value - 1.0
    return ServingRegression(
        metric=metric,
        base_value=base_value,
        candidate_value=candidate_value,
        absolute_delta=round(absolute_delta, 4),
        relative_change=round(relative, 6),
        relative_change_percent=round(relative * 100.0, 3),
        status=MetricStatus.OK,
    )


def analyze(
    base: ServingSummary,
    candidate: ServingSummary,
    *,
    run_id: str = "",
) -> RegressionReport:
    """Full analysis: comparability gate, then per-metric regressions.

    Latency direction (lower is better) and throughput direction (higher is
    better) belong to the metric definitions, not to this function; the report
    carries raw signed values so the report layer cannot flip a sign silently.
    """
    issues = check_comparability(base, candidate)
    if issues:
        return RegressionReport(
            run_id=run_id,
            comparable=False,
            incomparable_reasons=issues,
            baseline=base,
            candidate=candidate,
            regressions={},
        )

    regressions = {}
    for metric in (
        ServingMetric.TTFT,
        ServingMetric.TPOT,
        ServingMetric.ITL,
        ServingMetric.TOKENS_PER_SEC,
        ServingMetric.REQUESTS_PER_SEC,
    ):
        regressions[metric.value] = _regression_for(metric, base, candidate)

    return RegressionReport(
        run_id=run_id,
        comparable=True,
        incomparable_reasons=[],
        baseline=base,
        candidate=candidate,
        regressions=regressions,
        quality_status="not_evaluated",
    )
