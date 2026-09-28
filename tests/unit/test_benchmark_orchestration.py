"""M2 orchestration: benchmark stages, workload loading and resume semantics."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from quantassay.contracts import WorkloadSpec  # noqa: E402
from quantassay.gating import (  # noqa: E402
    ProbeError,
    compare_serving_parameters,
    load_workload,
    verify_benchmark_evidence,
)
from quantassay.serving.workload import build_request_set, request_set_hashes  # noqa: E402


def test_load_workload_defaults_to_the_builtin_set() -> None:
    workload = load_workload(None)
    assert workload.request_set_id == "builtin-short-chat"
    assert workload.request_hashes == request_set_hashes(build_request_set())
    assert workload.warmup_requests >= 0
    assert workload.timed_requests >= 1


def test_load_workload_reads_the_config_block(tmp_path: Path) -> None:
    config = tmp_path / "spec.yaml"
    config.write_text(
        """
workload:
  workload_id: mvp-short-chat
  request_set_id: builtin-short-chat
  request_hashes: [ignored-on-purpose]
  warmup_requests: 2
  timed_requests: 6
  max_new_tokens: 32
""",
        encoding="utf-8",
    )
    workload = load_workload(config)
    assert workload.timed_requests == 6
    assert workload.max_new_tokens == 32
    # The request set is code-owned: a config cannot substitute different inputs.
    assert workload.request_hashes == request_set_hashes(build_request_set())


def test_load_workload_rejects_a_config_without_a_workload_block(tmp_path: Path) -> None:
    config = tmp_path / "spec.yaml"
    config.write_text("model_id: Qwen/Qwen3-0.6B\n", encoding="utf-8")
    with pytest.raises(ProbeError, match="no workload block"):
        load_workload(config)


def test_load_workload_rejects_an_invalid_workload(tmp_path: Path) -> None:
    config = tmp_path / "spec.yaml"
    config.write_text(
        "workload:\n  workload_id: x\n  timed_requests: 0\n  request_hashes: [a]\n",
        encoding="utf-8",
    )
    with pytest.raises(ProbeError, match="invalid workload"):
        load_workload(config)


def _evidence(tmp_path: Path, *, records: int = 3, summary_total: int | None = None) -> dict:
    records_path = tmp_path / "serving-bf16.jsonl"
    lines = []
    for i in range(records):
        lines.append(
            json.dumps(
                {
                    "request_id": f"req-{i:02d}",
                    "status": "ok",
                    "ttft_ms": 10.0,
                    "e2e_latency_ms": 100.0,
                    "output_tokens": 4,
                    "input_tokens": 8,
                    "itl_ms": [20.0, 20.0, 20.0],
                }
            )
        )
    records_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log_path = tmp_path / "benchmark-bf16-server-1.log"
    log_path.write_text("served", encoding="utf-8")
    return {
        "records_path": str(records_path),
        "summary": {"requests_total": summary_total if summary_total is not None else records},
        "log_path": str(log_path),
        "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
    }


def test_verify_benchmark_evidence_accepts_intact_evidence(tmp_path: Path) -> None:
    assert verify_benchmark_evidence(_evidence(tmp_path)) is True


def test_verify_benchmark_evidence_rejects_summary_record_mismatch(tmp_path: Path) -> None:
    """If the JSONL no longer matches the summary, the stage must re-run."""
    result = _evidence(tmp_path, records=3, summary_total=9)
    assert verify_benchmark_evidence(result) is False


def test_verify_benchmark_evidence_rejects_a_tampered_log(tmp_path: Path) -> None:
    result = _evidence(tmp_path)
    Path(result["log_path"]).write_text("tampered", encoding="utf-8")
    assert verify_benchmark_evidence(result) is False


def test_verify_benchmark_evidence_rejects_missing_evidence(tmp_path: Path) -> None:
    result = _evidence(tmp_path)
    Path(result["records_path"]).unlink()
    assert verify_benchmark_evidence(result) is False


def test_verify_benchmark_evidence_rejects_corrupt_jsonl(tmp_path: Path) -> None:
    result = _evidence(tmp_path)
    Path(result["records_path"]).write_text("{not json\n", encoding="utf-8")
    assert verify_benchmark_evidence(result) is False


def test_workload_fingerprint_is_stable_and_input_sensitive() -> None:
    base = load_workload(None)
    assert base.workload_fingerprint() == load_workload(None).workload_fingerprint()
    changed = base.model_copy(update={"timed_requests": base.timed_requests + 1})
    assert changed.workload_fingerprint() != base.workload_fingerprint()


# --------------------------------------------------------------------------
# Serving-parameter parity between the two sides
# --------------------------------------------------------------------------


def _side(**params) -> dict:
    base = {
        "attention_backend": "triton",
        "operator_backend": "sglang",
        "mem_fraction_static": 0.8,
        "cuda_graph_max_bs": 2,
        "disable_cuda_graph": False,
        "context_length": 512,
        "max_running_requests": 1,
        "dtype": "bfloat16",
        "sglang_version": "0.5.3",
    }
    base.update(params)
    return {"serving_parameters": base}


def test_identical_serving_parameters_compare_equal() -> None:
    assert compare_serving_parameters(_side(), _side()) == []


def test_cuda_graph_mismatch_blocks_the_comparison() -> None:
    """This exact mismatch once produced a bogus 45 ms/token TPOT.

    BF16 ran with CUDA graphs and GPTQ with --disable-cuda-graph, so the
    measured difference was mostly the graph setting, not quantization.
    """
    base, candidate = _side(), _side(disable_cuda_graph=True)
    mismatches = compare_serving_parameters(base, candidate)
    assert len(mismatches) == 1
    assert "disable_cuda_graph" in mismatches[0]


def test_every_parameter_difference_is_reported() -> None:
    mismatches = compare_serving_parameters(
        _side(),
        _side(mem_fraction_static=0.7, attention_backend="torch_native", operator_backend="triton"),
    )
    joined = " ".join(mismatches)
    assert "mem_fraction_static" in joined
    assert "attention_backend" in joined
    assert "operator_backend" in joined
    assert len(mismatches) == 3


def test_missing_serving_parameters_are_reported_not_assumed_equal() -> None:
    assert compare_serving_parameters({}, _side())
    assert compare_serving_parameters(_side(), {})
