"""Advisor and report behaviour once quality IS measured.

The MVP's advisor hardcoded "quality was not evaluated". Now that the quality
layer exists, that disclaimer must become conditional — otherwise a report with
a measured perplexity would still claim nothing was measured, which is its own
kind of false statement.

What these tests lock out:

* a measured run still printing the "not evaluated" disclaimer;
* a quality number appearing without the caveat that it is perplexity only;
* an incomparable quality comparison being folded into a recommendation.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from quantassay.analysis.advisor import (  # noqa: E402
    QUALITY_NOT_EVALUATED_NOTE,
    advise,
)
from quantassay.analysis.quality import compare_quality  # noqa: E402
from quantassay.contracts import (  # noqa: E402
    DataSplit,
    MetricRecord,
    QualityStatus,
    QualitySummary,
    RegressionReport,
    ServingSummary,
    SideStatus,
)
from quantassay.reporting.render import render_html, render_markdown  # noqa: E402


def _serving(side: str, *, ttft: float, tok_s: float) -> ServingSummary:
    return ServingSummary(
        side=side,
        status=SideStatus.OK,
        workload_id="mvp-short-chat",
        workload_fingerprint="wf-1",
        sglang_version="0.5.3",
        requests_total=20,
        requests_ok=20,
        wall_seconds=10.0,
        metrics={
            "ttft": MetricRecord(name="ttft", value=ttft, unit="ms", sample_count=20),
            "output_tokens_per_sec": MetricRecord(
                name="output_tokens_per_sec", value=tok_s,
                unit="output_tokens/s", sample_count=20,
            ),
        },
    )


def _quality(side: str, *, nll_per_token: float, documents: int = 10) -> QualitySummary:
    per_doc_tokens = 10
    per_document = [(nll_per_token * per_doc_tokens, per_doc_tokens)] * documents
    nll = sum(n for n, _ in per_document)
    tokens = sum(t for _, t in per_document)
    return QualitySummary(
        side=side,
        status=QualityStatus.MEASURED,
        split=DataSplit.DEV,
        documents_total=documents,
        documents_scored=documents,
        valid_tokens=tokens,
        nll_sum=nll,
        ppl=math.exp(nll / tokens),
        per_document=per_document,
        prompt_mode="raw_completion",
        context_length=512,
    )


def _report_without_quality() -> RegressionReport:
    """A comparable report built through the real analyzer, not by hand.

    Hand-assembling the regressions dict is how a fixture drifts away from what
    the analyzer actually produces — an earlier version of this helper left
    ``regressions`` empty and silently exercised the "insufficient" path.
    """
    from quantassay.analysis.regression import analyze

    return analyze(
        _serving("bf16", ttft=100.0, tok_s=50.0),
        _serving("gptq", ttft=80.0, tok_s=60.0),
        run_id="r",
    )


def _report_with_quality(comparison) -> RegressionReport:
    report = _report_without_quality()
    return report.model_copy(
        update={
            "quality": comparison,
            "quality_status": (
                "measured" if comparison.comparable else "measured_not_comparable"
            ),
        }
    )


# --------------------------------------------------------------------------
# Advisor — the disclaimer must be conditional
# --------------------------------------------------------------------------


def test_unmeasured_quality_still_says_not_evaluated() -> None:
    recommendations = advise(_report_without_quality())
    assert recommendations
    for rec in recommendations:
        assert any("quality" in lim.lower() for lim in rec.limitations), rec.rule_id
    assert any(
        QUALITY_NOT_EVALUATED_NOTE in lim
        for rec in recommendations
        for lim in rec.limitations
    )


def test_measured_quality_drops_the_not_evaluated_disclaimer() -> None:
    """A run that measured quality must not claim quality was not evaluated."""
    comparison = compare_quality(
        _quality("bf16", nll_per_token=1.0),
        _quality("gptq", nll_per_token=1.1),
    )
    report = _report_with_quality(comparison)
    assert report.quality_not_evaluated is False

    recommendations = advise(report)
    assert not any(
        QUALITY_NOT_EVALUATED_NOTE in lim
        for rec in recommendations
        for lim in rec.limitations
    ), "measured run still printed the not-evaluated disclaimer"


def test_measured_quality_states_its_scope() -> None:
    """The caveat becomes 'perplexity only', not 'nothing measured'."""
    comparison = compare_quality(
        _quality("bf16", nll_per_token=1.0),
        _quality("gptq", nll_per_token=1.2),
    )
    recommendations = advise(_report_with_quality(comparison))
    joined = " ".join(lim for rec in recommendations for lim in rec.limitations).lower()
    assert "perplexity" in joined
    assert "task accuracy" in joined


def test_quality_rule_fires_only_when_quality_is_comparable() -> None:
    comparison = compare_quality(
        _quality("bf16", nll_per_token=1.0),
        _quality("gptq", nll_per_token=1.2),
    )
    rules = {r.rule_id for r in advise(_report_with_quality(comparison))}
    assert "quality-constraint" in rules

    # Not comparable (different splits) -> no quality rule.
    base = _quality("bf16", nll_per_token=1.0)
    cand = _quality("gptq", nll_per_token=1.2)
    cand = cand.model_copy(update={"split": DataSplit.FINAL})
    not_comparable = compare_quality(base, cand)
    rules = {r.rule_id for r in advise(_report_with_quality(not_comparable))}
    assert "quality-constraint" not in rules


def test_quality_observation_reports_the_direction() -> None:
    comparison = compare_quality(
        _quality("bf16", nll_per_token=1.0),
        _quality("gptq", nll_per_token=1.3),
    )
    recommendations = advise(_report_with_quality(comparison))
    quality_rule = next(r for r in recommendations if r.rule_id == "quality-constraint")
    # Positive change means the candidate perplexity is worse.
    assert "+" in quality_rule.observation
    assert "perplexity" in quality_rule.observation.lower()


def test_ci_crossing_zero_yields_an_inconclusive_quality_rule() -> None:
    """Identical sides cannot support a quality verdict."""
    comparison = compare_quality(
        _quality("bf16", nll_per_token=1.0),
        _quality("gptq", nll_per_token=1.0),
    )
    assert comparison.ci_crosses_zero is True
    quality_rule = next(
        r for r in advise(_report_with_quality(comparison)) if r.rule_id == "quality-constraint"
    )
    assert any("inconclusive" in lim for lim in quality_rule.limitations)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def test_markdown_reports_unmeasured_quality_plainly() -> None:
    text = render_markdown(_report_without_quality())
    assert "not evaluated" in text
    assert "Perplexity change" not in text


def test_markdown_renders_the_quality_change() -> None:
    comparison = compare_quality(
        _quality("bf16", nll_per_token=1.0),
        _quality("gptq", nll_per_token=1.2),
    )
    text = render_markdown(_report_with_quality(comparison))
    assert "## Quality" in text
    assert "Perplexity change" in text
    assert "positive means the candidate is *worse*" in text
    assert "per-document perplexities are never averaged" in text
    assert "not evaluated" not in text.split("## Limits")[0].split("## Quality")[1]


def test_html_and_markdown_agree_on_the_quality_number() -> None:
    comparison = compare_quality(
        _quality("bf16", nll_per_token=1.0),
        _quality("gptq", nll_per_token=1.2),
    )
    report = _report_with_quality(comparison)
    md = render_markdown(report)
    html = render_html(report)
    formatted = f"{comparison.ppl_relative_change * 100:+.2f}%"
    assert formatted in md
    assert formatted in html


def test_markdown_shows_incomparable_quality_reasons_without_numbers() -> None:
    base = _quality("bf16", nll_per_token=1.0)
    cand = _quality("gptq", nll_per_token=1.2).model_copy(
        update={"split": DataSplit.FINAL}
    )
    comparison = compare_quality(base, cand)
    text = render_markdown(_report_with_quality(comparison))
    assert "not comparable" in text
    assert "No quality difference may be quoted" in text
    assert "Perplexity change" not in text


def test_html_escapes_quality_reasons() -> None:
    from quantassay.contracts import ComparabilityIssue, QualityComparison

    comparison = QualityComparison(
        comparable=False,
        incomparable_reasons=[
            ComparabilityIssue(field="split", reason="<script>alert(1)</script>")
        ],
    )
    html = render_html(_report_with_quality(comparison))
    assert "<script>" not in html
    assert "&lt;script&gt;" in html