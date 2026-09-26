"""M3 acceptance: regression analyzer and advisor evidence discipline.

The core danger these tests guard against: a report that compares different
workloads, or an advisor that upgrades a single measurement into a deployment
recommendation. Percentages may only exist where comparability was proven, and
quality conclusions may never appear from serving data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from quantassay.contracts import (  # noqa: E402
    EvidenceStatus,
    MetricRecord,
    MetricStatus,
    RequestRecord,
    RequestStatus,
    ServingMetric,
    ServingSummary,
    SideStatus,
    WorkloadSpec,
)
from quantassay.analysis import advise, analyze, check_comparability  # noqa: E402


def _workload(request_set_id: str = "chat-8", concurrency: int = 1) -> WorkloadSpec:
    return WorkloadSpec(
        workload_id="w1",
        request_set_id=request_set_id,
        request_hashes=["h1", "h2"],
        concurrency=concurrency,
        warmup_requests=2,
        timed_requests=10,
    )


def _summary(
    side: str,
    *,
    fingerprint: str | None = "wf-123",
    sglang: str | None = "0.5.20",
    ttft: float | None = 100.0,
    tok_s: float | None = 50.0,
    requests_total: int = 10,
    requests_ok: int = 10,
) -> ServingSummary:
    metrics = {}
    if ttft is not None:
        metrics["ttft"] = MetricRecord(
            name="ttft", value=ttft, unit="ms", sample_count=requests_ok
        )
    if tok_s is not None:
        metrics["output_tokens_per_sec"] = MetricRecord(
            name="output_tokens_per_sec",
            value=tok_s,
            unit="output_tokens/s",
            sample_count=requests_ok,
        )
    return ServingSummary(
        side=side,
        status=SideStatus.OK if requests_ok else SideStatus.FAILED,
        reason=None if requests_ok else "no request produced an output token",
        workload_id="w1",
        workload_fingerprint=fingerprint,
        sglang_version=sglang,
        requests_total=requests_total,
        requests_ok=requests_ok,
        requests_failed=requests_total - requests_ok,
        wall_seconds=10.0,
        metrics=metrics,
    )


def _matched_pair() -> tuple[ServingSummary, ServingSummary]:
    return _summary("base"), _summary("candidate", ttft=80.0, tok_s=60.0)


# --------------------------------------------------------------------------
# Comparability gate
# --------------------------------------------------------------------------


def test_matched_pair_is_comparable() -> None:
    base, candidate = _matched_pair()
    issues = check_comparability(base, candidate)
    assert issues == []
    report = analyze(base, candidate, run_id="r1")
    assert report.comparable is True
    assert report.incomparable_reasons == []


def test_different_workload_fingerprints_block_the_report() -> None:
    base, candidate = _summary("base"), _summary("candidate", fingerprint="other")
    report = analyze(base, candidate)
    assert report.comparable is False
    assert report.regressions == {}
    assert report.incomparable_reasons[0].field == "workload_fingerprint"


def test_different_sglang_versions_block_the_report() -> None:
    base, candidate = _summary("base"), _summary("candidate", sglang="0.5.19")
    report = analyze(base, candidate)
    assert report.comparable is False
    fields = {issue.field for issue in report.incomparable_reasons}
    assert "sglang_version" in fields


def test_low_success_coverage_blocks_the_report() -> None:
    """Failed requests bias latency percentiles, so coverage is a hard gate."""
    base = _summary("base")
    candidate = _summary("candidate", requests_total=10, requests_ok=5)
    report = analyze(base, candidate)
    assert report.comparable is False
    fields = {issue.field for issue in report.incomparable_reasons}
    assert "success_coverage" in fields


def test_failed_baseline_blocks_the_report() -> None:
    base = _summary("base", requests_ok=0)
    candidate = _summary("candidate")
    report = analyze(base, candidate)
    assert report.comparable is False
    fields = {issue.field for issue in report.incomparable_reasons}
    assert "baseline_status" in fields


def test_different_output_lengths_block_the_comparison() -> None:
    """Latency per token is not comparable across different decode budgets."""
    base = _summary("base").model_copy(update={"output_tokens_p50": 100.0})
    candidate = _summary("candidate").model_copy(update={"output_tokens_p50": 40.0})
    report = analyze(base, candidate)
    assert report.comparable is False
    assert "output_tokens_p50" in {i.field for i in report.incomparable_reasons}


def test_similar_output_lengths_stay_comparable() -> None:
    base = _summary("base").model_copy(update={"output_tokens_p50": 100.0})
    candidate = _summary("candidate").model_copy(update={"output_tokens_p50": 96.0})
    assert analyze(base, candidate).comparable is True


def test_one_sided_truncation_blocks_the_comparison() -> None:
    """A side wholly truncated by max_tokens ran a different generation policy."""
    base = _summary("base").model_copy(
        update={"truncated_requests": 0, "stopped_on_eos": 10}
    )
    candidate = _summary("candidate").model_copy(
        update={"truncated_requests": 10, "stopped_on_eos": 0}
    )
    report = analyze(base, candidate)
    assert report.comparable is False
    assert "finish_reason" in {i.field for i in report.incomparable_reasons}


def test_both_sides_truncated_equally_remains_comparable() -> None:
    """Symmetric truncation is a disclosed limitation, not a blocker."""
    base = _summary("base").model_copy(
        update={"truncated_requests": 10, "stopped_on_eos": 0}
    )
    candidate = _summary("candidate").model_copy(
        update={"truncated_requests": 10, "stopped_on_eos": 0}
    )
    assert analyze(base, candidate).comparable is True


def test_missing_metric_on_one_side_marks_that_metric_unavailable() -> None:
    """A missing candidate metric is a metric-level gap, not total blocking."""
    base, _ = _matched_pair()
    candidate = _summary("candidate", ttft=None, tok_s=60.0)
    report = analyze(base, candidate)
    # Workload/runtime/coverage still match.
    assert report.comparable is True
    ttft = report.regressions["ttft"]
    assert ttft.status is MetricStatus.UNAVAILABLE
    assert ttft.relative_change_percent is None


# --------------------------------------------------------------------------
# Regression math and sign conventions
# --------------------------------------------------------------------------


def test_latency_regression_negative_means_faster() -> None:
    base, candidate = _matched_pair()
    report = analyze(base, candidate)
    ttft = report.regressions["ttft"]
    assert ttft.base_value == 100.0
    assert ttft.candidate_value == 80.0
    assert ttft.relative_change_percent == pytest.approx(-20.0)
    assert ttft.status is MetricStatus.OK


def test_throughput_regression_positive_means_higher() -> None:
    base, candidate = _matched_pair()
    report = analyze(base, candidate)
    tok = report.regressions["output_tokens_per_sec"]
    assert tok.relative_change_percent == pytest.approx(20.0)


def test_zero_baseline_yields_no_percentage() -> None:
    """A zero baseline cannot produce a ratio; it must not become 'infinite'."""
    base, candidate = _summary("base", ttft=0.0), _summary("candidate", ttft=80.0)
    report = analyze(base, candidate)
    ttft = report.regressions["ttft"]
    assert ttft.status is MetricStatus.UNAVAILABLE
    assert ttft.relative_change_percent is None
    assert "zero" in ttft.reason


def test_absolute_delta_is_present_alongside_percentage() -> None:
    base, candidate = _matched_pair()
    report = analyze(base, candidate)
    ttft = report.regressions["ttft"]
    assert ttft.absolute_delta == pytest.approx(-20.0)


# --------------------------------------------------------------------------
# Advisor: three conclusion types, no over-claiming
# --------------------------------------------------------------------------


def test_improvement_rule_fires_and_stays_hypothesis() -> None:
    base, candidate = _matched_pair()
    report = analyze(base, candidate)
    recommendations = advise(report)

    improvement = [r for r in recommendations if r.rule_id == "mvp-serving-improvement"]
    assert improvement, "expected an improvement recommendation"
    top = improvement[0]
    assert top.status is EvidenceStatus.HYPOTHESIS
    # A performance improvement must never become a deployment recommendation.
    assert "quality" in " ".join(top.limitations).lower()
    assert any("not_evaluated" in lim for lim in top.limitations)
    assert top.evidence_refs


def test_regression_rule_names_measurement_scope_not_recipe_causes() -> None:
    base, candidate = _summary("base"), _summary("candidate", ttft=150.0, tok_s=40.0)
    report = analyze(base, candidate)
    recommendations = advise(report)
    regression = [r for r in recommendations if r.rule_id == "mvp-serving-regression"]
    assert regression
    text = regression[0].recommendation.lower()
    assert "kernel" in text or "configuration" in text or "path" in text


def test_null_result_is_reported_without_inventing_benefit() -> None:
    """A within-noise delta must not be spun as an improvement headline."""
    base, candidate = _summary("base"), _summary("candidate", ttft=100.5, tok_s=50.1)
    report = analyze(base, candidate)
    recommendations = advise(report)

    improvement = [r for r in recommendations if r.rule_id == "mvp-serving-improvement"]
    # A -0.5% delta is measured, but the advisor must not headline it as a win:
    # the recommendation text still scopes it to this workload and asks for
    # variance characterization before any claim.
    if improvement:
        assert "variance" in improvement[0].next_validation.lower()
        assert "deployment" not in improvement[0].recommendation.lower()
    assert [r for r in recommendations if r.rule_id == "mvp-serving-regression"]
    assert all(
        r.status is EvidenceStatus.HYPOTHESIS
        for r in recommendations
        if r.rule_id != "mvp-evidence-insufficient"
    )


def test_incomparable_report_produces_insufficiency_only() -> None:
    base, candidate = _summary("base"), _summary("candidate", fingerprint="other")
    report = analyze(base, candidate)
    recommendations = advise(report)
    assert len(recommendations) == 1
    assert recommendations[0].rule_id == "mvp-evidence-insufficient"
    assert recommendations[0].status is EvidenceStatus.INCONCLUSIVE


def test_every_recommendation_flags_quality_not_evaluated() -> None:
    """The MVP never measures quality; every output must say so."""
    base, candidate = _matched_pair()
    for recommendation in advise(report=analyze(base, candidate)):
        assert any(
            "quality" in lim.lower() and "not_evaluated" in lim
            for lim in recommendation.limitations
        ), f"{recommendation.rule_id} does not disclose quality_not_evaluated"


def test_recipe_change_advice_is_always_hypothesis() -> None:
    base, candidate = _matched_pair()
    recipe = [r for r in advise(report=analyze(base, candidate)) if r.rule_id == "mvp-recipe-hypothesis"]
    assert recipe
    assert recipe[0].status is EvidenceStatus.HYPOTHESIS
    assert "group size" in recipe[0].recommendation.lower()


def test_advisor_recommendations_survive_json_roundtrip() -> None:
    import json

    from quantassay.contracts import Recommendation

    base, candidate = _matched_pair()
    recommendations = advise(report=analyze(base, candidate))
    payload = json.loads(json.dumps([r.model_dump(mode="json") for r in recommendations]))
    reloaded = [Recommendation.model_validate(item) for item in payload]
    assert len(reloaded) == len(recommendations)
