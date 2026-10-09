"""Build and persist the same comparison report online and offline."""

import json

from pathlib import Path
from typing import Any

from quantassay.analysis.advisor import advise
from quantassay.analysis.quality import compare_quality
from quantassay.analysis.regression import analyze, compare_serving_parameters
from quantassay.contracts import (
    ComparabilityIssue,
    QualitySummary,
    RegressionReport,
    ServingSummary,
)
from quantassay.experiments.store import StoreError, atomic_write_json, file_sha256, read_json
from quantassay.reporting.render import render_report


def build_comparison_report(
    baseline: dict[str, Any], candidate: dict[str, Any], *, run_id: str
) -> RegressionReport:
    report = analyze(
        ServingSummary.model_validate(baseline["summary"]),
        ServingSummary.model_validate(candidate["summary"]),
        run_id=run_id,
    )
    mismatches = compare_serving_parameters(baseline, candidate)
    if mismatches:
        issue = ComparabilityIssue(
            field="serving_parameters",
            base_value=json.dumps(baseline.get("serving_parameters") or {}, sort_keys=True),
            candidate_value=json.dumps(candidate.get("serving_parameters") or {}, sort_keys=True),
            reason="; ".join(mismatches),
        )
        report = report.model_copy(update={
            "comparable": False,
            "incomparable_reasons": [*report.incomparable_reasons, issue],
            "regressions": {},
        })
    return report


def attach_quality(
    report: RegressionReport, run_dir: Path, *, method: str, required: bool = False
) -> RegressionReport:
    paths = [run_dir / f"quality-{side}.json" for side in ("bf16", method)]
    if not all(path.is_file() for path in paths):
        if required:
            missing = next(path for path in paths if not path.is_file())
            raise StoreError(f"cannot fold quality into the report: {missing.name} is missing")
        return report
    try:
        summaries = [read_json(path) for path in paths]
        # Summaries are written before resource checks finish. A file alone must
        # not promote a blocked or interrupted quality stage to measured evidence.
        journal_path = next((run_dir / name for name in ("run-status.json", "m0-status.json")
                             if (run_dir / name).is_file()), None)
        if journal_path is not None:
            stages = read_json(journal_path)["stages"]
            for side, summary in zip(("bf16", method), summaries):
                stage = f"quality_{side}"
                record = stages.get(stage) or {}
                result_path = run_dir / f"{stage}.json"
                if (record.get("status") != "succeeded" or not result_path.is_file()
                        or record.get("sha256") != file_sha256(result_path)):
                    raise StoreError(f"{stage} is not verified as succeeded")
                if read_json(result_path).get("summary") != summary:
                    raise StoreError(f"{stage} summary differs from its accepted result")
        # Legacy summary-only runs have no journal to consult.
        baseline, candidate = [QualitySummary.model_validate(summary) for summary in summaries]
    except (StoreError, OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        if required:
            raise StoreError(f"cannot fold quality into the report: {exc}") from exc
        return report.model_copy(update={
            "notes": [*report.notes, f"Quality evidence was not attached: {exc}"],
        })
    comparison = compare_quality(baseline, candidate)
    return report.model_copy(update={
        "quality": comparison,
        "quality_status": "measured" if comparison.comparable else "measured_not_comparable",
    })


def save_report(
    report: RegressionReport, run_dir: Path, *, rebuilt_from: str | None = None
) -> dict[str, Any]:
    workload_id = report.baseline.workload_id if report.baseline else None
    recommendations = advise(report, workload_scope=workload_id)
    payload: dict[str, Any] = {"recommendations": [r.model_dump(mode="json") for r in recommendations]}
    if rebuilt_from is not None:
        payload["rebuilt_from"] = rebuilt_from
    atomic_write_json(run_dir / "regressions.json", report.model_dump(mode="json"))
    atomic_write_json(run_dir / "recommendations.json", payload)
    paths = render_report(report, run_dir)
    return {
        "comparable": report.comparable,
        "incomparable_reasons": [issue.field for issue in report.incomparable_reasons],
        "report": paths.markdown_path,
        "recommendations": [r.rule_id for r in recommendations],
    }
