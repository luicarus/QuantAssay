"""Paired quality comparison (full-prd.md §6).

The comparison is *paired*: both sides score the same documents in the same
order, so the uncertainty is estimated by resampling documents in pairs. The
rules that keep it honest:

* PPL is pooled from NLL sums, never averaged from per-document PPLs;
* a confidence interval spanning zero cannot support a directional claim;
* sides scored on different documents or different splits are not comparable;
* an incomparable comparison reports no change at all, rather than a number
  nobody should act on.
"""

from __future__ import annotations

from quantassay.contracts import (
    ComparabilityIssue,
    QualityComparison,
    QualityStatus,
    QualitySummary,
)
from quantassay.evaluation.ppl import paired_bootstrap_delta

#: How much of a PPL increase is worth calling a regression rather than noise
#: on a first measurement. Deliberately coarse; the CI is the real evidence.
MIN_MEANINGFUL_PPL_CHANGE = 0.01


def check_quality_comparability(
    base: QualitySummary, candidate: QualitySummary
) -> list[ComparabilityIssue]:
    """Reasons the two quality summaries may not be turned into a delta."""
    issues: list[ComparabilityIssue] = []

    for name, summary in (("baseline", base), ("candidate", candidate)):
        if summary.status is not QualityStatus.MEASURED:
            issues.append(
                ComparabilityIssue(
                    field=f"{name}_status",
                    base_value=base.status.value,
                    candidate_value=candidate.status.value,
                    reason=(
                        f"{name} quality is {summary.status.value}: "
                        f"{summary.reason or 'no usable measurement'}"
                    ),
                )
            )
    if issues:
        return issues

    if base.split != candidate.split:
        issues.append(
            ComparabilityIssue(
                field="split",
                base_value=base.split.value,
                candidate_value=candidate.split.value,
                reason="the two sides were scored on different data splits",
            )
        )
    if base.prompt_mode != candidate.prompt_mode:
        issues.append(
            ComparabilityIssue(
                field="prompt_mode",
                base_value=str(base.prompt_mode),
                candidate_value=str(candidate.prompt_mode),
                reason="the two sides were scored with different prompt protocols",
            )
        )
    if base.context_length != candidate.context_length:
        issues.append(
            ComparabilityIssue(
                field="context_length",
                base_value=str(base.context_length),
                candidate_value=str(candidate.context_length),
                reason="different context budgets make the windowing differ",
            )
        )
    # A paired test needs the same documents in the same order.
    if base.documents_scored != candidate.documents_scored:
        issues.append(
            ComparabilityIssue(
                field="documents_scored",
                base_value=str(base.documents_scored),
                candidate_value=str(candidate.documents_scored),
                reason=(
                    "the sides scored different numbers of documents; the "
                    "comparison would not be paired"
                ),
            )
        )
    return issues


def compare_quality(
    base: QualitySummary,
    candidate: QualitySummary,
    *,
    iterations: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> QualityComparison:
    """Compare two quality summaries, with a paired bootstrap interval."""
    issues = check_quality_comparability(base, candidate)
    if issues:
        return QualityComparison(
            comparable=False,
            incomparable_reasons=issues,
            baseline=base,
            candidate=candidate,
        )

    if base.ppl is None or candidate.ppl is None:
        return QualityComparison(
            comparable=False,
            incomparable_reasons=[
                ComparabilityIssue(
                    field="ppl",
                    reason="one side has no pooled perplexity",
                )
            ],
            baseline=base,
            candidate=candidate,
        )

    stats = paired_bootstrap_delta(
        base.per_document,
        candidate.per_document,
        iterations=iterations,
        seed=seed,
        confidence=confidence,
    )
    relative = stats["delta_ppl_relative"]
    ci_low = stats["ci_low"]
    ci_high = stats["ci_high"]
    crosses = (ci_low is None or ci_high is None) or (ci_low <= 0.0 <= ci_high)

    if relative is None:
        directional = "no measurable change"
    elif crosses:
        directional = "interval spans zero; no directional claim"
    elif relative > 0:
        directional = "candidate perplexity is higher (worse) beyond the interval"
    else:
        directional = "candidate perplexity is lower (better) beyond the interval"

    notes = [
        f"paired bootstrap over {stats['n_documents']} documents, "
        f"{iterations} resamples, {confidence:.0%} interval",
        "perplexity is pooled from NLL sums; per-document PPLs are never averaged",
    ]
    if abs(relative or 0.0) < MIN_MEANINGFUL_PPL_CHANGE and not crosses:
        notes.append(
            f"change is below the {MIN_MEANINGFUL_PPL_CHANGE:.0%} practical threshold"
        )

    return QualityComparison(
        comparable=True,
        baseline=base,
        candidate=candidate,
        ppl_relative_change=relative,
        ppl_delta=(candidate.ppl - base.ppl),
        ci_low=ci_low,
        ci_high=ci_high,
        ci_confidence=confidence,
        ci_iterations=iterations,
        paired_documents=int(stats["n_documents"]),
        ci_crosses_zero=crosses,
        directional_claim=directional,
        notes=notes,
    )