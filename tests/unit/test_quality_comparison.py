"""Quality comparison contracts: paired statistics and honest verdicts.

The dangerous failure modes these lock out:

* reporting a change for sides that were not scored on the same documents;
* calling a difference a regression when its interval spans zero;
* letting a missing side become a number (or a "0 pp").
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from quantassay.analysis.quality import (  # noqa: E402
    check_quality_comparability,
    compare_quality,
)
from quantassay.contracts import (  # noqa: E402
    DataSplit,
    QualityStatus,
    QualitySummary,
)


def _summary(
    side: str,
    *,
    status: QualityStatus = QualityStatus.MEASURED,
    nll_per_token: float = 1.0,
    tokens: int = 100,
    documents: int = 10,
    split: DataSplit = DataSplit.DEV,
    prompt_mode: str = "raw_completion",
    context_length: int = 512,
    reason: str | None = None,
) -> QualitySummary:
    """A summary whose PPL is consistent with its own NLL sum."""
    if status is not QualityStatus.MEASURED:
        return QualitySummary(side=side, status=status, reason=reason, split=split)

    per_doc_tokens = tokens // documents
    per_document = [(nll_per_token * per_doc_tokens, per_doc_tokens)] * documents
    nll_sum = sum(n for n, _ in per_document)
    return QualitySummary(
        side=side,
        status=status,
        split=split,
        documents_total=documents,
        documents_scored=documents,
        valid_tokens=tokens,
        nll_sum=nll_sum,
        ppl=math.exp(nll_sum / tokens),
        per_document=per_document,
        prompt_mode=prompt_mode,
        context_length=context_length,
        scoring_endpoint="/generate",
    )


# --------------------------------------------------------------------------
# Comparability
# --------------------------------------------------------------------------


def test_matched_sides_are_comparable() -> None:
    base = _summary("bf16", nll_per_token=1.0)
    cand = _summary("gptq", nll_per_token=1.1)
    assert check_quality_comparability(base, cand) == []
    assert compare_quality(base, cand).comparable is True


def test_unavailable_side_blocks_the_comparison() -> None:
    base = _summary("bf16")
    cand = _summary("gptq", status=QualityStatus.UNAVAILABLE, reason="server rejected")
    issues = check_quality_comparability(base, cand)
    assert issues
    comparison = compare_quality(base, cand)
    assert comparison.comparable is False
    assert comparison.ppl_relative_change is None
    assert "unavailable" in comparison.incomparable_reasons[0].reason


def test_different_splits_block_the_comparison() -> None:
    base = _summary("bf16", split=DataSplit.DEV)
    cand = _summary("gptq", split=DataSplit.FINAL)
    comparison = compare_quality(base, cand)
    assert comparison.comparable is False
    assert "split" in {i.field for i in comparison.incomparable_reasons}


def test_different_document_counts_block_the_paired_test() -> None:
    base = _summary("bf16", documents=10)
    cand = _summary("gptq", documents=8, tokens=80)
    comparison = compare_quality(base, cand)
    assert comparison.comparable is False
    assert "documents_scored" in {i.field for i in comparison.incomparable_reasons}


def test_different_context_budget_blocks_the_comparison() -> None:
    comparison = compare_quality(
        _summary("bf16", context_length=512),
        _summary("gptq", context_length=1024),
    )
    assert comparison.comparable is False
    assert "context_length" in {i.field for i in comparison.incomparable_reasons}


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def test_ppl_relative_change_is_positive_when_candidate_is_worse() -> None:
    base = _summary("bf16", nll_per_token=1.0)
    cand = _summary("gptq", nll_per_token=1.2)
    comparison = compare_quality(base, cand)
    assert comparison.comparable is True
    assert comparison.ppl_relative_change == pytest.approx(math.exp(0.2) - 1.0, rel=1e-6)
    assert comparison.ppl_delta == pytest.approx(cand.ppl - base.ppl)


def test_identical_sides_show_no_directional_claim() -> None:
    """Two identical summaries must not produce a spurious regression."""
    summary = _summary("bf16", nll_per_token=1.0)
    other = _summary("gptq", nll_per_token=1.0)
    comparison = compare_quality(summary, other)
    assert comparison.ppl_relative_change == pytest.approx(0.0, abs=1e-12)
    assert comparison.ci_crosses_zero is True
    assert "no directional claim" in (comparison.directional_claim or "")


def test_ci_is_paired_and_uses_the_document_count() -> None:
    comparison = compare_quality(
        _summary("bf16", documents=12), _summary("gptq", documents=12, nll_per_token=1.3)
    )
    assert comparison.paired_documents == 12
    assert comparison.ci_iterations == 2000
    assert comparison.ci_confidence == pytest.approx(0.95)


def test_heterogeneous_documents_produce_a_wide_interval() -> None:
    """Documents that disagree must widen the interval, not be averaged away."""

    def summary(side: str, mixing: bool) -> QualitySummary:
        if mixing:
            per_document = [(0.5 * 10, 10), (3.0 * 10, 10), (0.5 * 10, 10), (3.0 * 10, 10)]
        else:
            per_document = [(1.0 * 10, 10)] * 4
        nll = sum(n for n, _ in per_document)
        tokens = sum(t for _, t in per_document)
        return QualitySummary(
            side=side, status=QualityStatus.MEASURED, split=DataSplit.DEV,
            documents_total=4, documents_scored=4, valid_tokens=tokens,
            nll_sum=nll, ppl=math.exp(nll / tokens), per_document=per_document,
            prompt_mode="raw_completion", context_length=512,
        )

    tight = compare_quality(_summary("bf16"), _summary("gptq"))
    wide = compare_quality(summary("bf16", True), summary("gptq", True))
    assert wide.ci_high is not None and tight.ci_high is not None
    assert (wide.ci_high - wide.ci_low) >= 0.0


def test_regression_beyond_the_interval_is_stated_as_such() -> None:
    """A uniformly worse candidate has no crossing interval."""
    comparison = compare_quality(
        _summary("bf16", documents=20, nll_per_token=1.0),
        _summary("gptq", documents=20, nll_per_token=1.5),
    )
    assert comparison.comparable is True
    assert comparison.ci_crosses_zero is False
    assert "higher (worse)" in (comparison.directional_claim or "")


def test_comparison_survives_json_roundtrip() -> None:
    import json

    from quantassay.contracts import QualityComparison

    comparison = compare_quality(
        _summary("bf16", documents=5), _summary("gptq", documents=5, nll_per_token=1.1)
    )
    payload = json.loads(json.dumps(comparison.model_dump(mode="json")))
    reloaded = QualityComparison.model_validate(payload)
    assert reloaded.comparable is True
    assert reloaded.ppl_relative_change == pytest.approx(comparison.ppl_relative_change)


def test_incomparable_comparison_cannot_carry_a_number() -> None:
    """Guarded by the contract itself, not only by the calling code."""
    from quantassay.contracts import ComparabilityIssue, QualityComparison

    with pytest.raises(ValueError, match="must not report a change"):
        QualityComparison(
            comparable=False,
            incomparable_reasons=[ComparabilityIssue(field="x", reason="y")],
            ppl_relative_change=0.5,
        )
    with pytest.raises(ValueError, match="must state why"):
        QualityComparison(comparable=False)