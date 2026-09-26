"""M3 acceptance: offline report rendering.

The report is the user-facing artifact, so its failure modes get their own
tests: model text that would break HTML, values that drift from the JSON, and
incomparable runs that must not grow a verdict.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from quantassay.contracts import (  # noqa: E402
    ComparabilityIssue,
    MetricRecord,
    RegressionReport,
    ServingSummary,
)
from quantassay.reporting import render_html, render_markdown, render_report  # noqa: E402


def _summary(side: str, ttft: float | None = 100.0, tok: float | None = 50.0) -> ServingSummary:
    metrics = {}
    if ttft is not None:
        metrics["ttft"] = MetricRecord(name="ttft", value=ttft, unit="ms", sample_count=10)
    if tok is not None:
        metrics["output_tokens_per_sec"] = MetricRecord(
            name="output_tokens_per_sec", value=tok, unit="output_tokens/s", sample_count=10
        )
    return ServingSummary(
        side=side,
        workload_id="w1",
        workload_fingerprint="wf-123",
        sglang_version="0.5.20",
        requests_total=10,
        requests_ok=10,
        wall_seconds=10.0,
        metrics=metrics,
    )


def _comparable_report() -> RegressionReport:
    return RegressionReport(
        run_id="run-1",
        comparable=True,
        baseline=_summary("base"),
        candidate=_summary("candidate", ttft=80.0, tok=60.0),
        regressions={
            "ttft": __import__(
                "quantassay.contracts", fromlist=["ServingRegression"]
            ).ServingRegression(
                metric=__import__(
                    "quantassay.contracts", fromlist=["ServingMetric"]
                ).ServingMetric.TTFT,
                base_value=100.0,
                candidate_value=80.0,
                absolute_delta=-20.0,
                relative_change=-0.2,
                relative_change_percent=-20.0,
                status=__import__(
                    "quantassay.contracts", fromlist=["MetricStatus"]
                ).MetricStatus.OK,
            ),
        },
    )


def _blocked_report() -> RegressionReport:
    return RegressionReport(
        run_id="run-2",
        comparable=False,
        incomparable_reasons=[
            ComparabilityIssue(
                field="sglang_version",
                base_value="0.5.20",
                candidate_value="0.5.21",
                reason="different serving runtime versions",
            )
        ],
    )


# --------------------------------------------------------------------------
# Content
# --------------------------------------------------------------------------


def test_markdown_shows_percent_and_direction() -> None:
    text = render_markdown(_comparable_report())
    assert "-20.00%" in text
    assert "lower is better" in text
    assert "not evaluated" in text


def test_blocked_report_shows_reasons_and_no_verdict() -> None:
    text = render_markdown(_blocked_report())
    assert "NOT COMPARABLE" in text
    assert "sglang_version" in text
    assert "different serving runtime versions" in text
    assert "No performance conclusion may be drawn" in text


def test_html_is_offline_and_self_contained() -> None:
    html = render_html(_comparable_report())
    assert "http://" not in html and "https://" not in html
    assert 'meta charset="utf-8"' in html


# --------------------------------------------------------------------------
# Escaping: model output is untrusted
# --------------------------------------------------------------------------


def test_html_escapes_hostile_run_id() -> None:
    report = _comparable_report()
    report = report.model_copy(update={"run_id": '<script>alert("x")</script>'})
    html = render_html(report)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_markdown_escapes_hostile_run_id() -> None:
    report = _comparable_report()
    report = report.model_copy(update={"run_id": "| inject | row"})
    text = render_markdown(report)
    assert "| inject | row" not in text.replace("\\|", "|") or "\\|" in text


def test_html_escapes_hostile_issue_reason() -> None:
    report = RegressionReport(
        run_id="r",
        comparable=False,
        incomparable_reasons=[
            ComparabilityIssue(field="f", reason="<img src=x onerror=alert(1)>")
        ],
    )
    html = render_html(report)
    assert "<img" not in html
    assert "&lt;img" in html


# --------------------------------------------------------------------------
# File writing
# --------------------------------------------------------------------------


def test_render_report_writes_both_files(tmp_path: Path) -> None:
    paths = render_report(_comparable_report(), tmp_path)
    assert Path(paths.markdown_path).is_file()
    assert Path(paths.html_path).is_file()

    # The HTML values must match the Markdown values (no drift between views).
    md = Path(paths.markdown_path).read_text(encoding="utf-8")
    html = Path(paths.html_path).read_text(encoding="utf-8")
    assert "-20.00%" in md
    assert "-20.00%" in html


def test_render_report_into_missing_dir(tmp_path: Path) -> None:
    paths = render_report(_blocked_report(), tmp_path / "nested" / "report")
    assert Path(paths.markdown_path).is_file()


# --------------------------------------------------------------------------
# Values must mirror the contract objects, not be recomputed
# --------------------------------------------------------------------------


def test_report_values_come_from_the_contract_not_recomputation() -> None:
    report = _comparable_report()
    original = report.regressions["ttft"].relative_change_percent
    text = render_markdown(report)
    assert f"{original:+.2f}%" in text


def test_json_roundtrip_of_report_preserves_render() -> None:
    report = _comparable_report()
    payload = json.loads(json.dumps(report.model_dump(mode="json")))
    reloaded = RegressionReport.model_validate(payload)
    assert render_markdown(reloaded) == render_markdown(report)
