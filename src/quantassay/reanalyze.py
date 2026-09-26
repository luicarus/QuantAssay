"""Rebuild report, regressions and advice from a finished run directory.

This is the offline path required by mvp-prd.md §7: no model, no server, no GPU
— just the persisted evidence. It exists so a report can be regenerated after
the rendering or analysis code changes, without re-measuring (which would
produce different numbers and destroy the original evidence).

Usage:
    python -m quantassay.reanalyze runs/<run-id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from quantassay.analysis.advisor import advise
from quantassay.analysis.regression import analyze
from quantassay.contracts import (
    ComparabilityIssue,
    RegressionReport,
    ServingSummary,
)
from quantassay.experiments.store import atomic_write_json, read_json
from quantassay.reporting.render import render_report
from quantassay.serving.benchmark import load_records
from quantassay.serving.evaluator import summarize_requests


class ReanalyzeError(RuntimeError):
    """Raised when the run directory lacks the evidence needed to re-derive."""


def _require(path: Path) -> dict:
    if not path.is_file():
        raise ReanalyzeError(f"missing evidence: {path}")
    try:
        return read_json(path)
    except Exception as exc:
        raise ReanalyzeError(f"unreadable evidence {path}: {exc}") from exc


def _serving_parameters(run_dir: Path, side: str) -> dict:
    return _require(run_dir / f"benchmark_{side}.json").get("serving_parameters") or {}


def _summary_from_records(run_dir: Path, side: str) -> ServingSummary:
    """Rebuild the summary from the per-request JSONL evidence."""
    records_path = run_dir / f"serving-{side}.jsonl"
    if not records_path.is_file():
        raise ReanalyzeError(f"missing evidence: {records_path}")
    try:
        records = load_records(records_path)
    except Exception as exc:
        raise ReanalyzeError(f"unreadable evidence {records_path}: {exc}") from exc
    if not records:
        raise ReanalyzeError(f"no request records in {records_path}")

    meta = _require(run_dir / f"benchmark_{side}.json")
    return summarize_requests(
        records,
        side=side,
        # The wall-clock window is a property of the measurement run, not of the
        # records, so it is carried over rather than recomputed.
        wall_seconds=meta.get("wall_seconds"),
        sglang_version=(meta.get("serving_parameters") or {}).get("sglang_version"),
        quant_kernel=meta.get("quant_kernel"),
        workload_id=(meta.get("summary") or {}).get("workload_id"),
        workload_fingerprint=(meta.get("summary") or {}).get("workload_fingerprint"),
        records_path=str(records_path),
    )


def build_report(run_dir: Path) -> RegressionReport:
    """Re-derive the RegressionReport from files already on disk.

    Per-request statistics are recomputed from the JSONL evidence — the primary
    record — rather than trusting the derived summary. A run produced before a
    metric definition changed would otherwise keep reporting the old numbers
    (or blanks) forever. The measured wall-clock window cannot be recovered
    from the records, so it is carried over from the run.
    """
    run_dir = Path(run_dir)
    bf16 = _summary_from_records(run_dir, "bf16")
    gptq = _summary_from_records(run_dir, "gptq")
    report = analyze(bf16, gptq, run_id=run_dir.name)

    # Same serving-parameter gate the live path applies: a report rebuilt from
    # disk must not be more permissive than the one produced during the run.
    left, right = _serving_parameters(run_dir, "bf16"), _serving_parameters(run_dir, "gptq")
    mismatches = [
        f"{key}: {left.get(key)!r} != {right.get(key)!r}"
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    ]
    if not left or not right:
        mismatches = ["serving parameters were not recorded for both sides"]
    if mismatches:
        report = report.model_copy(
            update={
                "comparable": False,
                "incomparable_reasons": [
                    ComparabilityIssue(
                        field="serving_parameters",
                        base_value=json.dumps(left, sort_keys=True),
                        candidate_value=json.dumps(right, sort_keys=True),
                        reason="; ".join(mismatches),
                    )
                ],
                "regressions": {},
            }
        )
    return report


def reanalyze(run_dir: Path) -> dict:
    """Rewrite regressions.json, recommendations.json and the report."""
    run_dir = Path(run_dir)
    report = build_report(run_dir)
    atomic_write_json(run_dir / "regressions.json", report.model_dump(mode="json"))

    workload_id = report.baseline.workload_id if report.baseline else None
    recommendations = advise(report, workload_scope=workload_id)
    atomic_write_json(
        run_dir / "recommendations.json",
        {
            "recommendations": [r.model_dump(mode="json") for r in recommendations],
            "rebuilt_from": str(run_dir),
        },
    )
    paths = render_report(report, run_dir)
    return {
        "comparable": report.comparable,
        "incomparable_reasons": [i.field for i in report.incomparable_reasons],
        "report": paths.markdown_path,
        "recommendations": [r.rule_id for r in recommendations],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild report/analysis from a run directory")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        result = reanalyze(args.run_dir)
    except ReanalyzeError as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"status": "succeeded", **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
