"""Existing paired-bootstrap interval checks."""

from quantassay.evaluation.ppl import paired_bootstrap_delta


def test_paired_bootstrap_reports_interval() -> None:
    base = [(1.0, 10) for _ in range(20)]
    quant = [(1.2, 10) for _ in range(20)]
    result = paired_bootstrap_delta(base, quant, iterations=300, seed=0)

    assert result["n_documents"] == 20
    assert result["delta_ppl_relative"] is not None
    assert result["delta_ppl_relative"] > 0
    assert result["ci_low"] is not None and result["ci_high"] is not None
    assert result["ci_low"] <= result["delta_ppl_relative"] <= result["ci_high"]


def test_paired_bootstrap_is_deterministic_for_a_seed() -> None:
    base = [(1.0, 10), (2.0, 5), (0.5, 8)]
    quant = [(1.1, 10), (2.1, 5), (0.6, 8)]
    first = paired_bootstrap_delta(base, quant, iterations=200, seed=42)
    second = paired_bootstrap_delta(base, quant, iterations=200, seed=42)
    assert first == second


def test_paired_bootstrap_without_documents() -> None:
    result = paired_bootstrap_delta([], [], iterations=10, seed=0)
    assert result["n_documents"] == 0
    assert result["delta_ppl_relative"] is None
