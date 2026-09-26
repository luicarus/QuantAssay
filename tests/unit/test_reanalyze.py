"""Offline re-analysis from a run directory (mvp-prd.md §7)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from quantassay.reanalyze import ReanalyzeError, build_report, reanalyze  # noqa: E402


def _summary_dump(side: str, *, ttft: float, tok_s: float, p50: float) -> dict:
    return {
        "side": side,
        "status": "ok",
        "workload_id": "mvp-short-chat",
        "workload_fingerprint": "wf-1",
        "sglang_version": "0.5.3",
        "requests_total": 20,
        "requests_ok": 20,
        "wall_seconds": 10.0,
    }


def _write_jsonl(path: Path, *, ttft: float, output_tokens: int, count: int = 4) -> None:
    """Write per-request evidence like a real benchmark run would."""
    lines = []
    for i in range(count):
        itl = [(ttft + 10) / max(output_tokens - 1, 1)] * max(output_tokens - 1, 0)
        lines.append(
            json.dumps(
                {
                    "request_id": f"req-{i:02d}",
                    "status": "ok",
                    "ttft_ms": ttft,
                    "e2e_latency_ms": ttft + 10 * len(itl),
                    "output_tokens": output_tokens,
                    "input_tokens": 12,
                    "itl_ms": itl,
                    "finish_reason": "length",
                    "truncated": True,
                    "stopped_on_eos": False,
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_run(tmp_path: Path, *, same_params: bool = True) -> Path:
    params = {
        "attention_backend": "triton",
        "disable_cuda_graph": False,
        "sglang_version": "0.5.3",
    }
    other = dict(params)
    if not same_params:
        other["disable_cuda_graph"] = True
    (tmp_path / "benchmark_bf16.json").write_text(
        json.dumps({"summary": _summary_dump("bf16", ttft=100.0, tok_s=50.0, p50=64.0),
                    "serving_parameters": params, "wall_seconds": 10.0}),
        encoding="utf-8",
    )
    (tmp_path / "benchmark_gptq.json").write_text(
        json.dumps({"summary": _summary_dump("gptq", ttft=80.0, tok_s=60.0, p50=64.0),
                    "serving_parameters": other, "wall_seconds": 10.0}),
        encoding="utf-8",
    )
    _write_jsonl(tmp_path / "serving-bf16.jsonl", ttft=100.0, output_tokens=8)
    _write_jsonl(tmp_path / "serving-gptq.jsonl", ttft=80.0, output_tokens=8)
    return tmp_path


def test_rebuild_derives_a_comparable_report(tmp_path: Path) -> None:
    report = build_report(_write_run(tmp_path))
    assert report.comparable is True
    assert report.regressions["ttft"].relative_change_percent == pytest.approx(-20.0)


def test_rebuild_applies_the_serving_parameter_gate(tmp_path: Path) -> None:
    """The offline path must not be more permissive than the live one."""
    report = build_report(_write_run(tmp_path, same_params=False))
    assert report.comparable is False
    assert report.regressions == {}
    assert "serving_parameters" in {i.field for i in report.incomparable_reasons}


def test_reanalyze_writes_report_and_recommendations(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    result = reanalyze(run_dir)
    assert result["comparable"] is True
    assert Path(result["report"]).is_file()
    assert (run_dir / "regressions.json").is_file()
    assert (run_dir / "recommendations.json").is_file()

    payload = json.loads((run_dir / "recommendations.json").read_text(encoding="utf-8"))
    for rec in payload["recommendations"]:
        assert any("not_evaluated" in lim for lim in rec["limitations"])


def test_rebuild_is_stable_across_repeated_runs(tmp_path: Path) -> None:
    """Re-deriving twice must give the same numbers (no hidden state)."""
    run_dir = _write_run(tmp_path)
    first = reanalyze(run_dir)
    second = reanalyze(run_dir)
    assert first["comparable"] == second["comparable"]
    md = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "-20.00%" in md


def test_missing_evidence_is_reported_not_guessed(tmp_path: Path) -> None:
    with pytest.raises(ReanalyzeError, match="missing evidence"):
        build_report(tmp_path)


def test_unreadable_evidence_is_reported(tmp_path: Path) -> None:
    (tmp_path / "benchmark_bf16.json").write_text("{broken", encoding="utf-8")
    (tmp_path / "benchmark_gptq.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ReanalyzeError):
        build_report(tmp_path)


def test_report_discloses_truncation(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    reanalyze(run_dir)
    md = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "Output length and termination" in md
    assert "Truncation notice" in md
