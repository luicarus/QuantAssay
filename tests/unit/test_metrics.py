"""Quality-metric math: PPL definitions and output-regression comparison.

**Out of MVP scope.** The MVP measures serving performance only and reports
quality as ``not_evaluated`` (mvp-prd.md §1). These formulas belong to the full
PRD's quality expansion (full-prd.md §4, §8 layer 2); they are kept because the
contracts exist and must not be confused with MVP results.

Every formula here is fixed by full-prd.md §4. The tests encode the stated
acceptance cases (uniform binary distribution => PPL 2, 0.70 -> 0.68 is -2 pp,
all-padding yields no valid sample) plus the edge cases most likely to corrupt a
real comparison.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from quantassay.contracts import MetricStatus  # noqa: E402
from quantassay.diagnostics import (  # noqa: E402
    GenerationSample,
    TopKComparison,
    compare_token_sequences,
    first_divergence,
    summarize_generations,
    top_k_from_logits,
)
from quantassay.evaluation import (  # noqa: E402
    NllAccumulator,
    accuracy_delta_metric,
    accuracy_delta_pp,
    accuracy_metric,
    has_discriminating_power,
    interval_crosses_zero,
    is_guessing_chance,
    nll_of_uniform,
    paired_bootstrap_delta,
    perplexity_from_nll,
    ppl_change_metric,
    relative_ppl_change,
    uniform_ppl,
)


# --------------------------------------------------------------------------
# PPL: definition, aggregation and the mandatory edge cases
# --------------------------------------------------------------------------


def test_uniform_binary_distribution_gives_ppl_two() -> None:
    """full-prd.md §4 acceptance case: uniform over 2 classes => PPL = 2."""
    assert uniform_ppl(2) == 2.0
    acc = NllAccumulator()
    # A uniform binary model has NLL = ln(2) at every token.
    acc.add_document([nll_of_uniform(2)] * 10, 10)
    assert acc.ppl == pytest.approx(2.0)


def test_ppl_is_exp_of_mean_nll() -> None:
    acc = NllAccumulator()
    acc.add_document([1.0, 2.0, 3.0], 3)
    assert acc.mean_nll == pytest.approx(2.0)
    assert acc.ppl == pytest.approx(math.exp(2.0))


def test_all_padding_yields_no_valid_sample() -> None:
    """full-prd.md §4 acceptance case: all-padding must not produce a value."""
    acc = NllAccumulator()
    acc.add_document([], 0)
    acc.add_document([1.0], 0)

    assert acc.token_count == 0
    assert acc.ppl is None
    metric = acc.to_metric()
    assert metric.value is None
    assert metric.status is MetricStatus.UNAVAILABLE
    assert "no valid target tokens" in (metric.reason or "")


def test_empty_accumulator_reports_unavailable_not_zero() -> None:
    metric = NllAccumulator().to_metric()
    assert metric.value is None
    assert metric.status is MetricStatus.UNAVAILABLE
    assert metric.reason


def test_nll_sum_not_average_across_documents() -> None:
    """Aggregating must weight by token count, not average per-document PPL."""
    acc = NllAccumulator()
    # Document A: 1 token with NLL 0 (PPL 1); document B: 9 tokens with NLL 1.
    acc.add_document([0.0], 1)
    acc.add_document([1.0] * 9, 9)

    # Correct: exp((0 + 9) / 10).
    assert acc.ppl == pytest.approx(math.exp(0.9))
    # The wrong "average the PPLs" answer would be (1 + e) / 2.
    assert acc.ppl != pytest.approx((1.0 + math.e) / 2.0)


def test_shard_merge_matches_single_pass() -> None:
    whole = NllAccumulator()
    whole.add_document([1.0, 2.0], 2)
    whole.add_document([3.0], 1)

    shard_a = NllAccumulator()
    shard_a.add_document([1.0, 2.0], 2)
    shard_b = NllAccumulator()
    shard_b.add_document([3.0], 1)
    shard_a.merge(shard_b)

    assert shard_a.nll_sum == whole.nll_sum
    assert shard_a.token_count == whole.token_count
    assert shard_a.ppl == pytest.approx(whole.ppl)  # type: ignore[arg-type]


def test_non_finite_nll_values_are_dropped_and_counted() -> None:
    acc = NllAccumulator()
    acc.add_document([1.0, float("nan"), float("inf"), 1.0], 4)
    assert acc.non_finite_values == 2
    assert acc.token_count == 2
    assert acc.ppl == pytest.approx(math.e)


def test_document_with_only_non_finite_values_is_skipped() -> None:
    acc = NllAccumulator()
    acc.add_document([float("nan")], 1)
    assert acc.skipped_documents == 1
    assert acc.token_count == 0


def test_negative_valid_tokens_rejected() -> None:
    acc = NllAccumulator()
    with pytest.raises(ValueError):
        acc.add_document([1.0], -1)


def test_zero_token_ppl_is_none_not_one() -> None:
    # exp(0/0) is undefined; a missing result must not silently become 1.0.
    assert perplexity_from_nll(0.0, 0) is None
    assert perplexity_from_nll(1.0, 0) is None


def test_ppl_overflow_reports_inf_status() -> None:
    acc = NllAccumulator()
    acc.add_document([1e5], 1)
    metric = acc.to_metric()
    assert metric.value is None
    assert metric.status is MetricStatus.INF
    assert metric.reason


# --------------------------------------------------------------------------
# Relative change
# --------------------------------------------------------------------------


def test_relative_ppl_change_sign_convention() -> None:
    # Larger PPL is worse, so degradation is positive.
    assert relative_ppl_change(7.0, 7.7) == pytest.approx(0.1, abs=1e-9)
    assert relative_ppl_change(7.0, 6.3) == pytest.approx(-0.1, abs=1e-9)
    assert relative_ppl_change(7.0, 7.0) == pytest.approx(0.0)


def test_relative_change_undefined_for_zero_baseline() -> None:
    assert relative_ppl_change(0.0, 7.0) is None


def test_ppl_change_metric_refuses_missing_inputs() -> None:
    base = NllAccumulator()
    base.add_document([1.0], 1)
    quant_missing = NllAccumulator()

    metric = ppl_change_metric(base.to_metric("ppl"), quant_missing.to_metric("ppl"))
    assert metric.value is None
    assert metric.status is MetricStatus.UNAVAILABLE
    assert "quantized PPL unavailable" in (metric.reason or "")


def test_ppl_change_metric_computes_when_both_present() -> None:
    base = NllAccumulator()
    base.add_document([0.0], 1)
    quant = NllAccumulator()
    quant.add_document([math.log(2.0)], 1)

    metric = ppl_change_metric(base.to_metric("ppl"), quant.to_metric("ppl"))
    assert metric.value == pytest.approx(1.0)
    assert metric.status is MetricStatus.OK


# --------------------------------------------------------------------------
# Accuracy: percentage points
# --------------------------------------------------------------------------


def test_accuracy_delta_070_to_068_is_minus_two_pp() -> None:
    """full-prd.md §4 acceptance case: accuracy differences are in pp."""
    assert accuracy_delta_pp(0.70, 0.68) == pytest.approx(-2.0)


def test_accuracy_delta_metric_uses_pp_units() -> None:
    base = accuracy_metric(70, 100)
    quant = accuracy_metric(68, 100)
    delta = accuracy_delta_metric(base, quant)
    assert delta.value == pytest.approx(-2.0)
    assert delta.unit == "pp"
    assert delta.sample_count == 100


def test_accuracy_with_no_questions_is_unavailable() -> None:
    metric = accuracy_metric(0, 0)
    assert metric.value is None
    assert metric.status is MetricStatus.UNAVAILABLE


def test_accuracy_delta_requires_both_sides() -> None:
    delta = accuracy_delta_metric(accuracy_metric(0, 0), accuracy_metric(1, 2))
    assert delta.value is None
    assert delta.status is MetricStatus.UNAVAILABLE


def test_guessing_chance_detection() -> None:
    # 4-choice task: chance is 0.25.
    assert is_guessing_chance(0.25, 4) is True
    assert is_guessing_chance(0.24, 4) is True
    assert is_guessing_chance(0.60, 4) is False
    assert has_discriminating_power(0.60, 4) is True
    assert has_discriminating_power(0.25, 4) is False


# --------------------------------------------------------------------------
# Paired bootstrap
# --------------------------------------------------------------------------


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


def test_interval_crosses_zero_treats_unknown_as_inconclusive() -> None:
    assert interval_crosses_zero(None, None) is True
    assert interval_crosses_zero(-0.01, 0.02) is True
    assert interval_crosses_zero(0.01, 0.05) is False
    assert interval_crosses_zero(-0.05, -0.01) is False


# --------------------------------------------------------------------------
# Output regression: divergence and agreement
# --------------------------------------------------------------------------


def test_identical_sequences_have_no_divergence() -> None:
    assert first_divergence([1, 2, 3], [1, 2, 3]) is None
    result = compare_token_sequences([1, 2, 3], [1, 2, 3])
    assert result.first_divergence_index is None
    assert result.agreement_rate == 1.0
    assert result.compared_positions == 3


def test_first_divergence_position_is_detected() -> None:
    result = compare_token_sequences([1, 2, 3, 4], [1, 2, 9, 4])
    assert result.first_divergence_index == 2
    # The diverging position itself is still a shared context.
    assert result.prefix_length == 3
    assert result.compared_positions == 3
    assert result.matching_positions == 2


def test_positions_after_divergence_are_excluded() -> None:
    """After divergence the contexts differ, so agreement is not comparable."""
    result = compare_token_sequences([1, 9, 9, 9, 9], [1, 8, 9, 9, 9])
    assert result.total_positions == 5
    assert result.compared_positions == 2  # only prefix up to divergence
    assert result.agreement_rate == pytest.approx(0.5)


def test_different_lengths_are_handled() -> None:
    assert first_divergence([1, 2], [1, 2, 3]) == 2
    result = compare_token_sequences([1, 2], [1, 2, 3])
    assert result.total_positions == 2
    assert result.first_divergence_index == 2


def test_agreement_metric_status_for_empty_comparison() -> None:
    result = compare_token_sequences([], [])
    metric = result.to_metric()
    assert metric.value is None
    assert metric.status is MetricStatus.UNAVAILABLE


def test_top1_agreement_is_fraction_not_pp() -> None:
    result = compare_token_sequences([1, 2, 3], [1, 2, 4])
    metric = result.to_metric()
    assert metric.unit == "fraction"
    assert metric.value == pytest.approx(2 / 3)


# --------------------------------------------------------------------------
# Top-k comparison
# --------------------------------------------------------------------------


def test_top_k_from_logits_returns_normalised_probabilities() -> None:
    pairs = top_k_from_logits([1.0, 3.0, 2.0], k=2)
    assert [token for token, _ in pairs] == [1, 2]
    assert sum(prob for _, prob in pairs) <= 1.0 + 1e-9
    assert pairs[0][1] > pairs[1][1]


def test_top_k_is_numerically_stable_for_large_logits() -> None:
    pairs = top_k_from_logits([1000.0, 999.0], k=2)
    assert all(math.isfinite(prob) for _, prob in pairs)
    assert sum(prob for _, prob in pairs) == pytest.approx(1.0)


def test_top_k_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError):
        top_k_from_logits([1.0], k=0)


def test_top_k_empty_logits() -> None:
    assert top_k_from_logits([], k=5) == []


def test_topk_comparison_describes_probability_shift() -> None:
    """The documented example: the chosen token changed at the divergence."""
    comparison = TopKComparison(
        position=26,
        prefix_token=42,
        base_top_k=[(42, 0.73), (41, 0.10)],
        quant_top_k=[(41, 0.46), (42, 0.31)],
        k=2,
        base_choice=42,
        quant_choice=41,
    )
    assert comparison.base_top1_probability == pytest.approx(0.73)
    assert comparison.quant_top1_probability == pytest.approx(0.46)
    assert comparison.top_k_overlap == pytest.approx(1.0)

    text = comparison.describe()
    assert "position 26" in text
    assert "0.73" in text
    assert "0.46" in text
    # A changed choice is described as a change, not hidden.
    assert "choice changed" in text


def test_topk_comparison_describes_confidence_drop_for_same_choice() -> None:
    """The plan's other documented case: same token, lower confidence."""
    comparison = TopKComparison(
        position=26,
        prefix_token=42,
        base_top_k=[(42, 0.73)],
        quant_top_k=[(42, 0.31)],
        k=1,
        base_choice=42,
        quant_choice=42,
    )
    text = comparison.describe()
    assert "0.73 -> 0.31" in text
    assert "choice changed" not in text


def test_topk_overlap_computation() -> None:
    comparison = TopKComparison(
        position=0,
        prefix_token=0,
        base_top_k=[(1, 0.5), (2, 0.3), (3, 0.2)],
        quant_top_k=[(1, 0.4), (4, 0.3), (5, 0.3)],
        k=3,
    )
    assert comparison.top_k_overlap == pytest.approx(1 / 3)


def test_topk_metrics_use_unavailable_for_missing_values() -> None:
    comparison = TopKComparison(position=0, prefix_token=0, base_top_k=[], quant_top_k=[])
    metrics = comparison.to_metrics("s1")
    assert metrics["s1_topk_overlap"].status is MetricStatus.UNAVAILABLE
    assert metrics["s1_base_top1_probability"].status is MetricStatus.UNAVAILABLE
    # Never encoded as zero.
    assert metrics["s1_topk_overlap"].value is None


# --------------------------------------------------------------------------
# Generation summaries
# --------------------------------------------------------------------------


def test_generation_summary_counts_divergence() -> None:
    same = GenerationSample(prompt_id="p1", prompt="a", base_token_ids=[1, 2], quant_token_ids=[1, 2])
    diff = GenerationSample(prompt_id="p2", prompt="b", base_token_ids=[1, 2], quant_token_ids=[1, 9])

    metrics = summarize_generations([same, diff])
    assert metrics["generation_divergence_rate"].value == pytest.approx(0.5)
    assert metrics["generation_token_agreement"].status is MetricStatus.OK


def test_generation_summary_without_samples() -> None:
    metrics = summarize_generations([])
    assert metrics["generation_divergence_rate"].value is None
    assert metrics["generation_divergence_rate"].status is MetricStatus.UNAVAILABLE
    assert metrics["generation_token_agreement"].status is MetricStatus.UNAVAILABLE


def test_eos_and_truncation_are_recorded_separately() -> None:
    sample = GenerationSample(
        prompt_id="p1",
        prompt="x",
        base_token_ids=[1, 2],
        quant_token_ids=[1, 2],
        base_stopped_on_eos=True,
        quant_truncated=True,
    )
    assert sample.base_stopped_on_eos is True
    assert sample.quant_truncated is True
    assert sample.quant_stopped_on_eos is False
