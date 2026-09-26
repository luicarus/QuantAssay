"""Quantization Advisor: condition recommendations on measured evidence only.

The MVP's evidence base is *serving performance on one workload*. Quality was
never measured, so the advisor must say ``quality_not_evaluated`` and cannot
recommend deployment. Three conclusion types are allowed (mvp-prd.md §5):

1. performance improved and comparable -> recommend further quality/resource
   evaluation, never deployment;
2. performance regressed or no clear benefit -> point at measurement scope and
   suggest re-measurement, without inventing quantization-recipe causes;
3. evidence insufficient / blocked -> state exactly what is missing.

A single GPTQ candidate can never validate a *recipe change*; such advice is
always ``hypothesis`` until a new checkpoint is built and re-measured.
"""

from __future__ import annotations

from typing import Any

from quantassay.contracts import (
    EvidenceStatus,
    MetricStatus,
    Recommendation,
    RegressionReport,
    ServingMetric,
)

RULE_VERSION = "1"

#: Minimum percent change before a delta is treated as directional. Measured
#: deltas below this are "no clear effect" on a single workload: with no
#: variance characterization, a -0.5% TTFT move is noise, not an improvement.
#: The threshold is deliberately conservative and per-metric configurable later.
MIN_MEANINGFUL_PERCENT = 1.0

_IMPROVEMENT_RULE = "mvp-serving-improvement"
_REGRESSION_RULE = "mvp-serving-regression"
_INSUFFICIENT_RULE = "mvp-evidence-insufficient"
_HYPOTHESIS_RULE = "mvp-recipe-hypothesis"
_QUALITY_RULE = "quality-constraint"

#: Text used while quality has not been measured at all. Kept as one constant so
#: the wording cannot drift between rules.
QUALITY_NOT_EVALUATED_NOTE = "quality was not evaluated (quality_not_evaluated)"


def _quality_limitations(report: RegressionReport) -> list[str]:
    """What the quality evidence does and does not support.

    The MVP measures serving performance only; the full PRD's quality layer adds
    a measured perplexity comparison. The advisor must reflect whichever is true
    rather than repeating a fixed disclaimer.
    """
    comparison = report.quality
    if comparison is None or report.quality_not_evaluated:
        return [QUALITY_NOT_EVALUATED_NOTE]

    if not comparison.comparable:
        why = "; ".join(i.reason for i in comparison.incomparable_reasons) or "unknown"
        return [
            "quality was measured but is not comparable across sides: " + why,
            "no quality difference may be quoted from this run",
        ]

    notes = [
        "quality here means held-out perplexity only; no task accuracy, "
        "EM/F1 or output-consistency claim is supported",
        "a perplexity difference is not a statement about task quality",
    ]
    if comparison.ci_crosses_zero:
        notes.append(
            "the perplexity interval spans zero, so no directional quality claim "
            "is supported"
        )
    else:
        notes.append(
            "perplexity moved beyond its interval, but the practical tolerance "
            "for the target task is still a user decision"
        )
    return notes


def _quality_observation(report: RegressionReport) -> str | None:
    """One-line quality finding, or None when there is nothing to quote."""
    comparison = report.quality
    if comparison is None or not comparison.comparable:
        return None
    if comparison.ppl_relative_change is None:
        return None
    return (
        f"held-out perplexity {comparison.ppl_relative_change * 100:+.2f}% "
        f"(baseline {comparison.baseline.ppl:.3f} -> candidate "
        f"{comparison.candidate.ppl:.3f})"
    )


def _metric(report: RegressionReport, name: ServingMetric) -> Any:
    return report.regressions.get(name.value)


def _direction_note(metric: ServingMetric) -> str:
    """Sign explanation so readers cannot misread an improvement."""
    if metric in (ServingMetric.TTFT, ServingMetric.TPOT, ServingMetric.ITL):
        return "negative percent means faster (improvement)"
    if metric in (ServingMetric.TOKENS_PER_SEC, ServingMetric.REQUESTS_PER_SEC):
        return "positive percent means higher throughput (improvement)"
    return ""


def _direction(record: Any) -> str | None:
    """Classify a measured delta: 'improved', 'regressed', or None (within noise)."""
    if record is None or record.status is not MetricStatus.OK:
        return None
    if record.relative_change_percent is None:
        return None
    if abs(record.relative_change_percent) < MIN_MEANINGFUL_PERCENT:
        return None
    if record.metric in (ServingMetric.TTFT, ServingMetric.TPOT, ServingMetric.ITL):
        return "improved" if record.relative_change_percent < 0 else "regressed"
    return "improved" if record.relative_change_percent > 0 else "regressed"


def _primary_latency_result(report: RegressionReport) -> str:
    """Summarize latency movement across TTFT/TPOT/ITL for the headline."""
    improved: list[str] = []
    regressed: list[str] = []
    for metric in (ServingMetric.TTFT, ServingMetric.TPOT, ServingMetric.ITL):
        record = _metric(report, metric)
        direction = _direction(record)
        if direction is None or record.relative_change_percent is None:
            continue
        label = metric.value.upper()
        if direction == "improved":
            improved.append(f"{label} {record.relative_change_percent:+.1f}%")
        else:
            regressed.append(f"{label} {record.relative_change_percent:+.1f}%")
    if improved:
        return "latency improved: " + ", ".join(improved)
    if regressed:
        return "latency regressed: " + ", ".join(regressed)
    return "no directional latency change beyond the ±" f"{MIN_MEANINGFUL_PERCENT:g}% noise floor"


def advise(
    report: RegressionReport,
    *,
    workload_scope: str | None = None,
) -> list[Recommendation]:
    """Derive recommendations strictly from the report's evidence."""
    recommendations: list[Recommendation] = []
    scope = workload_scope or (report.baseline.workload_id if report.baseline else "unknown")

    # Gate 1: an incomparable or empty report yields only the insufficiency rule.
    if not report.comparable or not report.baseline or not report.candidate:
        reasons = "; ".join(
            f"{issue.field}: {issue.reason}" for issue in report.incomparable_reasons
        ) or "no comparable measurements"
        recommendations.append(
            Recommendation(
                rule_id=_INSUFFICIENT_RULE,
                rule_version=RULE_VERSION,
                status=EvidenceStatus.INCONCLUSIVE,
                evidence_refs=[issue.field for issue in report.incomparable_reasons] or ["no_data"],
                workload_scope=scope,
                observation=f"comparison is not possible: {reasons}",
                recommendation=(
                    "fix the blocking conditions and re-run both sides under an "
                    "identical workload before drawing any performance conclusion"
                ),
                limitations=[
                    "no serving-performance claim is supported by the current evidence",
                    *_quality_limitations(report),
                ],
                next_validation="re-run baseline and candidate with matched workload/protocol",
            )
        )
        return recommendations

    # Gate 2: comparable. Look at the four MVP metrics.
    latency = _metric(report, ServingMetric.TTFT)
    throughput = _metric(report, ServingMetric.TOKENS_PER_SEC)
    usable = [
        record
        for record in (latency, throughput)
        if record is not None and record.status is MetricStatus.OK
    ]

    if not usable:
        recommendations.append(
            Recommendation(
                rule_id=_INSUFFICIENT_RULE,
                rule_version=RULE_VERSION,
                status=EvidenceStatus.INCONCLUSIVE,
                evidence_refs=[],
                workload_scope=scope,
                observation="comparable sides but no usable headline metric",
                recommendation="increase the measured request count and re-run",
                limitations=_quality_limitations(report),
                next_validation="re-measure with a larger fixed workload",
            )
        )
        return recommendations

    evidence_refs = [
        f"regressions.{metric.value}"
        for metric in (ServingMetric.TTFT, ServingMetric.TPOT, ServingMetric.ITL, ServingMetric.TOKENS_PER_SEC)
        if _metric(report, metric) is not None
    ]

    # Directional classification goes through _direction(), which applies the
    # noise floor: tiny deltas on a single uncharacterized workload are not wins.
    latency_direction = _direction(latency)
    throughput_direction = _direction(throughput)
    latency_improved = latency_direction == "improved"
    throughput_improved = throughput_direction == "improved"
    latency_regressed = latency_direction == "regressed"

    observation = _primary_latency_result(report)
    if throughput is not None and throughput.status is MetricStatus.OK:
        if throughput_direction is None and throughput.relative_change_percent is not None:
            observation += (
                f"; output throughput {throughput.relative_change_percent:+.1f}% "
                f"(within ±{MIN_MEANINGFUL_PERCENT:g}% noise floor)"
            )
        elif throughput.relative_change_percent is not None:
            observation += f"; output throughput {throughput.relative_change_percent:+.1f}%"
        else:
            observation += "; output throughput measured"
    # Fold in the quality finding when one exists, so a reader cannot read the
    # speed number without the accuracy context sitting next to it.
    quality_note = _quality_observation(report)
    if quality_note is not None:
        observation += f"; {quality_note}"

    if latency_improved or throughput_improved:
        recommendations.append(
            Recommendation(
                rule_id=_IMPROVEMENT_RULE,
                rule_version=RULE_VERSION,
                status=EvidenceStatus.HYPOTHESIS,
                evidence_refs=evidence_refs,
                workload_scope=scope,
                observation=observation,
                recommendation=(
                    "the quantized candidate served measurably faster on this "
                    "workload; proceed to quality evaluation and a resource "
                    "comparison before any deployment consideration"
                ),
                limitations=[
                    "single workload, single run: variance is not characterized",
                    "deployment decisions require a user-defined quality tolerance",
                    f"sign convention: {_direction_note(ServingMetric.TTFT)}; "
                    f"{_direction_note(ServingMetric.TOKENS_PER_SEC)}",
                    *_quality_limitations(report),
                ],
                next_validation=(
                    "run quality evaluation and a VRAM/resource comparison on the "
                    "same checkpoint; repeat the workload to characterize variance"
                ),
            )
        )
    elif latency_regressed:
        recommendations.append(
            Recommendation(
                rule_id=_REGRESSION_RULE,
                rule_version=RULE_VERSION,
                status=EvidenceStatus.HYPOTHESIS,
                evidence_refs=evidence_refs,
                workload_scope=scope,
                observation=observation,
                recommendation=(
                    "the quantized candidate served measurably slower on this "
                    "workload; check the actual quantized execution path "
                    "(kernel/backend) and service configuration before attributing "
                    "the regression to quantization itself"
                ),
                limitations=[
                    "the serving path was not proven to use a quantized kernel",
                    f"sign convention: {_direction_note(ServingMetric.TTFT)}",
                    *_quality_limitations(report),
                ],
                next_validation=(
                    "confirm the executed kernel from service logs; re-run with the "
                    "same workload to exclude measurement noise"
                ),
            )
        )
    else:
        recommendations.append(
            Recommendation(
                rule_id=_REGRESSION_RULE,
                rule_version=RULE_VERSION,
                status=EvidenceStatus.HYPOTHESIS,
                evidence_refs=evidence_refs,
                workload_scope=scope,
                observation=observation,
                recommendation=(
                    "no clear serving-performance benefit on this workload; the "
                    "quantized checkpoint adds a format dependency without a "
                    "measured latency/throughput gain here"
                ),
                limitations=[
                    "a null result on one workload does not generalize to others",
                    *_quality_limitations(report),
                ],
                next_validation=(
                    "re-run with a longer/different workload before concluding; "
                    "consider VRAM savings as a separate axis (they are not latency)"
                ),
            )
        )

    # The recipe-change rule is always a hypothesis in the MVP: one candidate
    # cannot prove that a *different* recipe would be better.
    recommendations.append(
        Recommendation(
            rule_id=_HYPOTHESIS_RULE,
            rule_version=RULE_VERSION,
            status=EvidenceStatus.HYPOTHESIS,
            evidence_refs=evidence_refs or ["no_data"],
            workload_scope=scope,
            observation=(
                "only one quantization recipe has been measured in this MVP "
                "(W4A16, group_size=128, 4-sample calibration)"
            ),
            recommendation=(
                "treat any recipe change (group size, calibration set size/distribution, "
                "ignored modules) as an untested hypothesis requiring a new checkpoint "
                "and a full re-measurement"
            ),
            limitations=[
                "single-recipe evidence cannot rank recipes",
                *_quality_limitations(report),
            ],
            next_validation=(
                "build a new checkpoint from the changed recipe and repeat the "
                "identical workload before claiming any recipe-level benefit"
            ),
        )
    )

    # Quality-constraint rule: only when a comparable quality comparison exists.
    # This is the one place the advisor may talk about quality, and even then it
    # stays conditional on a tolerance the user has not supplied.
    comparison = report.quality
    if comparison is not None and comparison.comparable:
        quality_observation = _quality_observation(report) or "quality was measured"
        if comparison.ci_crosses_zero:
            recommendation = (
                "the perplexity difference is not distinguishable from zero on this "
                "holdout, so no quality constraint can be applied yet; increase the "
                "holdout or accept a coarser tolerance before deciding"
            )
            status_note = "inconclusive"
        elif (comparison.ppl_relative_change or 0.0) > 0:
            recommendation = (
                "perplexity worsened beyond its interval while serving got faster: "
                "this is the tradeoff to decide on, and it needs a task-specific "
                "quality tolerance rather than a default"
            )
            status_note = "reported"
        else:
            recommendation = (
                "perplexity did not worsen beyond its interval on this holdout; the "
                "candidate is a reasonable subject for task-level evaluation before "
                "any deployment decision"
            )
            status_note = "reported"

        recommendations.append(
            Recommendation(
                rule_id=_QUALITY_RULE,
                rule_version=RULE_VERSION,
                status=EvidenceStatus.HYPOTHESIS,
                evidence_refs=["quality.ppl_relative_change", "quality.ci_low", "quality.ci_high"],
                workload_scope=scope,
                observation=quality_observation,
                recommendation=recommendation,
                limitations=[
                    f"quality status: {status_note}",
                    "perplexity is measured on wikitext-2 held-out text only; "
                    "task accuracy, EM/F1 and output consistency were not measured",
                    "a perplexity difference does not by itself establish that a "
                    "task-level quality bar is met",
                ],
                next_validation=(
                    "run a task-level quality set the target use case cares about, "
                    "and state the tolerance the deployment actually requires"
                ),
            )
        )

    return recommendations
