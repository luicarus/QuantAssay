"""Rebuild GPTQ/AWQ serving and quality reports from saved evidence, without a GPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from quantassay.contracts import RegressionReport
from quantassay.experiments.store import read_json
from quantassay.reporting.pipeline import attach_quality, build_comparison_report, save_report
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


def _benchmark_from_records(run_dir: Path, side: str) -> dict:
    """Recompute request statistics and retain the original measurement metadata."""
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
    summary = summarize_requests(
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
    return {**meta, "summary": summary.model_dump(mode="json")}


def build_report(run_dir: Path, *, method: str | None = None) -> RegressionReport:
    """Build the same report as the live pipeline, recomputing serving aggregates."""
    run_dir = Path(run_dir)
    if method is None:
        methods = [m for m in ("gptq", "awq") if (run_dir / f"benchmark_{m}.json").is_file()]
        if len(methods) != 1:
            raise ReanalyzeError("expected one GPTQ/AWQ benchmark; select --quant-method explicitly")
        method = methods[0]
    if method not in ("gptq", "awq"):
        raise ReanalyzeError(f"unsupported quantization method: {method}")
    baseline = _benchmark_from_records(run_dir, "bf16")
    candidate = _benchmark_from_records(run_dir, method)
    try:
        report = build_comparison_report(baseline, candidate, run_id=run_dir.name)
        return attach_quality(report, run_dir, method=method)
    except Exception as exc:
        raise ReanalyzeError(f"cannot rebuild comparison: {exc}") from exc


def reanalyze(run_dir: Path, *, method: str | None = None) -> dict:
    """Rewrite regressions.json, recommendations.json and the report."""
    run_dir = Path(run_dir)
    report = build_report(run_dir, method=method)
    return save_report(report, run_dir, rebuilt_from=str(run_dir))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild report/analysis from a run directory")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--quant-method", choices=("gptq", "awq"),
                        help="candidate method; inferred when exactly one benchmark exists")
    args = parser.parse_args(argv)
    try:
        result = reanalyze(args.run_dir, method=args.quant_method)
    except ReanalyzeError as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"status": "succeeded", **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
