"""Output-level regression: token agreement, first divergence, top-k comparison.

This is where the project answers "how did quantization change the output?"
rather than only "accuracy moved by X pp" (full-prd.md §4).

NOT part of the MVP: the MVP measures serving performance only. Output
consistency belongs to the full PRD's quality expansion (full-prd.md §8 layer 2)
and must not be presented as an MVP result.

Rules that keep the comparison honest:

* Only tokens under the *same prefix* are comparable. After the first divergence
  the two models are no longer reading the same context, so later logits are not
  treated as a same-context comparison.
* Top-k agreement is reported with its ``k``; it must never stand in for a full
  KL divergence.
* A missing probability is recorded with a status, not as 0.0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from quantassay.contracts import MetricDirection, MetricRecord, MetricStatus


@dataclass
class TokenAgreementResult:
    """Teacher-forced agreement over a shared prefix."""

    compared_positions: int
    matching_positions: int
    first_divergence_index: int | None
    prefix_length: int
    total_positions: int

    @property
    def agreement_rate(self) -> float | None:
        if self.compared_positions == 0:
            return None
        return self.matching_positions / self.compared_positions

    def to_metric(self, name: str = "top1_agreement") -> MetricRecord:
        rate = self.agreement_rate
        if rate is None:
            return MetricRecord.unavailable(name, "no comparable token positions", unit="fraction")
        return MetricRecord(
            name=name,
            value=rate,
            unit="fraction",
            direction=MetricDirection.HIGHER_IS_BETTER,
            sample_count=self.compared_positions,
            token_count=self.compared_positions,
            status=MetricStatus.OK,
        )


def first_divergence(base_ids: Sequence[int], quant_ids: Sequence[int]) -> int | None:
    """Index of the first differing position, or None if the prefixes match."""
    for index, (base, quant) in enumerate(zip(base_ids, quant_ids)):
        if base != quant:
            return index
    if len(base_ids) != len(quant_ids):
        return min(len(base_ids), len(quant_ids))
    return None


def compare_token_sequences(
    base_ids: Sequence[int], quant_ids: Sequence[int]
) -> TokenAgreementResult:
    """Compare teacher-forced argmax sequences under a shared input prefix.

    Positions after the first divergence are excluded from the agreement rate:
    there the models see different contexts, so agreement is not a meaningful
    measure of quantization error.
    """
    total = min(len(base_ids), len(quant_ids))
    divergence = first_divergence(base_ids, quant_ids)

    if divergence is None:
        compared = total
        matching = total
        prefix = total
    else:
        # Include the diverging position itself: it is still a shared context.
        prefix = divergence + 1
        compared = prefix
        matching = divergence

    return TokenAgreementResult(
        compared_positions=compared,
        matching_positions=matching,
        first_divergence_index=divergence,
        prefix_length=prefix,
        total_positions=total,
    )


@dataclass
class TopKComparison:
    """How the top-k distribution differs at one divergent position."""

    position: int
    prefix_token: int
    base_top_k: list[tuple[int, float]] = field(default_factory=list)
    quant_top_k: list[tuple[int, float]] = field(default_factory=list)
    k: int = 5
    base_choice: int | None = None
    quant_choice: int | None = None

    @property
    def base_top1_probability(self) -> float | None:
        for token, prob in self.base_top_k:
            if token == self.base_choice:
                return prob
        return None

    @property
    def quant_top1_probability(self) -> float | None:
        for token, prob in self.quant_top_k:
            if token == self.quant_choice:
                return prob
        return None

    @property
    def top_k_overlap(self) -> float | None:
        if not self.base_top_k or not self.quant_top_k:
            return None
        base_set = {token for token, _ in self.base_top_k}
        quant_set = {token for token, _ in self.quant_top_k}
        denom = min(len(base_set), len(quant_set))
        if denom == 0:
            return None
        return len(base_set & quant_set) / denom

    def describe(self, decode: object | None = None) -> str:
        """Human-readable summary, e.g. for the report's divergence cases."""

        def render(pairs: list[tuple[int, float]], decode_fn: object) -> str:
            parts = []
            for token, prob in pairs:
                label = str(token)
                if callable(decode_fn):
                    try:
                        label = decode_fn(token)  # type: ignore[misc]
                    except Exception:
                        label = str(token)
                parts.append(f"{label!r} p={prob:.2f}")
            return ", ".join(parts)

        lines = [
            f"position {self.position} (prefix token {self.prefix_token})",
            f"  base  top-{self.k}: {render(self.base_top_k, decode)}",
            f"  quant top-{self.k}: {render(self.quant_top_k, decode)}",
        ]
        base_p = self.base_top1_probability
        quant_p = self.quant_top1_probability
        if base_p is not None and quant_p is not None:
            lines.append(
                f"  same token, probability {base_p:.2f} -> {quant_p:.2f}"
                if self.base_choice == self.quant_choice
                else f"  choice changed: {self.base_choice} (p={base_p:.2f}) -> "
                f"{self.quant_choice} (p={quant_p:.2f})"
            )
        return "\n".join(lines)

    def to_metrics(self, prefix: str = "divergence") -> dict[str, MetricRecord]:
        records: dict[str, MetricRecord] = {}
        overlap = self.top_k_overlap
        records[f"{prefix}_topk_overlap"] = (
            MetricRecord(
                name=f"{prefix}_topk_overlap",
                value=overlap,
                unit="fraction",
                direction=MetricDirection.HIGHER_IS_BETTER,
                status=MetricStatus.OK,
            )
            if overlap is not None
            else MetricRecord.unavailable(
                f"{prefix}_topk_overlap", "top-k lists unavailable for comparison"
            )
        )

        for label, value in (
            ("base_top1_probability", self.base_top1_probability),
            ("quant_top1_probability", self.quant_top1_probability),
        ):
            name = f"{prefix}_{label}"
            records[name] = (
                MetricRecord(
                    name=name,
                    value=value,
                    unit="probability",
                    direction=MetricDirection.HIGHER_IS_BETTER,
                    status=MetricStatus.OK,
                )
                if value is not None
                else MetricRecord.unavailable(name, "token not present in the top-k list")
            )
        return records


def top_k_from_logits(logits: Sequence[float], k: int) -> list[tuple[int, float]]:
    """Top-k (token_id, probability) pairs from a logit vector.

    Softmax is computed in a numerically stable way; a uniform input is handled
    without special-casing.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not logits:
        return []

    indexed = sorted(enumerate(logits), key=lambda pair: pair[1], reverse=True)
    top = indexed[: min(k, len(indexed))]

    maximum = top[0][1]
    exps = [(token, math.exp(value - maximum)) for token, value in indexed]
    total = sum(value for _, value in exps)
    if total == 0 or not math.isfinite(total):
        # Degenerate distribution: report as unavailable rather than guessing.
        return []
    return [(token, value / total) for token, value in exps[: len(top)]]


def softmax_entropy(logits: Sequence[float]) -> float | None:
    """Entropy in nats, or None when the distribution is degenerate."""
    if not logits:
        return None
    maximum = max(logits)
    exps = [math.exp(value - maximum) for value in logits]
    total = sum(exps)
    if total <= 0 or not math.isfinite(total):
        return None
    probs = [value / total for value in exps]
    return -sum(p * math.log(p) for p in probs if p > 0)


@dataclass
class GenerationSample:
    """One greedy generation from each model, compared under the same prompt."""

    prompt_id: str
    prompt: str
    base_token_ids: list[int] = field(default_factory=list)
    quant_token_ids: list[int] = field(default_factory=list)
    base_text: str = ""
    quant_text: str = ""
    base_stopped_on_eos: bool = False
    quant_stopped_on_eos: bool = False
    base_truncated: bool = False
    quant_truncated: bool = False
    topk_at_divergence: TopKComparison | None = None

    def agreement(self) -> TokenAgreementResult:
        return compare_token_sequences(self.base_token_ids, self.quant_token_ids)

    def to_metrics(self) -> dict[str, MetricRecord]:
        agreement = self.agreement()
        records = {f"{self.prompt_id}_top1_agreement": agreement.to_metric(
            f"{self.prompt_id}_top1_agreement"
        )}
        if self.topk_at_divergence is not None:
            records.update(self.topk_at_divergence.to_metrics(self.prompt_id))
        return records


def summarize_generations(samples: Sequence[GenerationSample]) -> dict[str, MetricRecord]:
    """Aggregate divergence statistics across prompts."""
    records: dict[str, MetricRecord] = {}

    diverged = [s for s in samples if s.agreement().first_divergence_index is not None]
    if samples:
        records["generation_divergence_rate"] = MetricRecord(
            name="generation_divergence_rate",
            value=len(diverged) / len(samples),
            unit="fraction",
            direction=MetricDirection.LOWER_IS_BETTER,
            sample_count=len(samples),
            status=MetricStatus.OK,
        )
    else:
        records["generation_divergence_rate"] = MetricRecord.unavailable(
            "generation_divergence_rate", "no generation samples"
        )

    total_compared = sum(s.agreement().compared_positions for s in samples)
    total_matching = sum(s.agreement().matching_positions for s in samples)
    records["generation_token_agreement"] = (
        MetricRecord(
            name="generation_token_agreement",
            value=total_matching / total_compared,
            unit="fraction",
            direction=MetricDirection.HIGHER_IS_BETTER,
            sample_count=len(samples),
            token_count=total_compared,
            status=MetricStatus.OK,
        )
        if total_compared
        else MetricRecord.unavailable("generation_token_agreement", "no comparable tokens")
    )
    return records
