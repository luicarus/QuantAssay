"""NLL/PPL metric helpers and paired bootstrap statistics.

Quality comparisons use paired document resampling and pool NLL sums over valid
target tokens: PPL = exp(sum_nll / token_count). Per-document PPLs are not averaged.
These functions do not load a model; serving logprobs are collected in quality.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from quantassay.contracts import (
    MetricDirection,
    MetricRecord,
    MetricStatus,
)

# Tolerance for guessing-chance perplexity (uniform over N classes => PPL = N).
DEFAULT_REL_TOL = 1e-9


@dataclass
class NllAccumulator:
    """Accumulates NLL totals and valid-token counts.

    Aggregating the *sum* rather than an average is what makes the final PPL
    correct across shards of differing length.
    """

    nll_sum: float = 0.0
    token_count: int = 0
    document_count: int = 0
    skipped_documents: int = 0
    non_finite_values: int = 0
    _per_document: list[tuple[float, int]] = field(default_factory=list)

    def add_document(self, nll_values: Sequence[float], valid_tokens: int) -> None:
        """Add one document's per-token NLLs and its valid-token count."""
        if valid_tokens < 0:
            raise ValueError("valid_tokens must not be negative")
        if valid_tokens == 0:
            # A document with no valid target tokens contributes nothing and must
            # not be counted as an evaluation of zero loss.
            self.skipped_documents += 1
            return

        total = 0.0
        counted = 0
        for value in nll_values:
            if not math.isfinite(value):
                self.non_finite_values += 1
                continue
            total += float(value)
            counted += 1

        if counted == 0:
            self.skipped_documents += 1
            return

        self.nll_sum += total
        self.token_count += counted
        self.document_count += 1
        self._per_document.append((total, counted))

    def merge(self, other: "NllAccumulator") -> None:
        """Combine accumulators (e.g. from separate shards)."""
        self.nll_sum += other.nll_sum
        self.token_count += other.token_count
        self.document_count += other.document_count
        self.skipped_documents += other.skipped_documents
        self.non_finite_values += other.non_finite_values
        self._per_document.extend(other._per_document)

    @property
    def mean_nll(self) -> float | None:
        if self.token_count == 0:
            return None
        return self.nll_sum / self.token_count

    @property
    def ppl(self) -> float | None:
        """Perplexity, or None when there were no valid tokens."""
        mean = self.mean_nll
        if mean is None:
            return None
        try:
            return math.exp(mean)
        except OverflowError:
            return math.inf

    @property
    def documents(self) -> list[tuple[float, int]]:
        """Per-document (nll_sum, token_count), for resampling-based intervals."""
        return list(self._per_document)

    def to_metric(self, name: str = "ppl", **kw: object) -> MetricRecord:
        """Convert to a MetricRecord, using explicit status instead of 0."""
        if self.token_count == 0:
            reason = "no valid target tokens"
            if self.skipped_documents:
                reason += f" ({self.skipped_documents} document(s) skipped)"
            if self.non_finite_values:
                reason += f"; {self.non_finite_values} non-finite NLL value(s) dropped"
            return MetricRecord.unavailable(
                name,
                reason,
                unit="",
                direction=MetricDirection.LOWER_IS_BETTER,
                sample_count=self.document_count,
                token_count=0,
                **kw,  # type: ignore[arg-type]
            )

        ppl = self.ppl
        if ppl is None or not math.isfinite(ppl):
            return MetricRecord(
                name=name,
                value=None,
                status=MetricStatus.INF,
                reason="perplexity overflowed to infinity",
                unit="",
                direction=MetricDirection.LOWER_IS_BETTER,
                sample_count=self.document_count,
                token_count=self.token_count,
                **kw,  # type: ignore[arg-type]
            )

        return MetricRecord(
            name=name,
            value=ppl,
            unit="",
            direction=MetricDirection.LOWER_IS_BETTER,
            sample_count=self.document_count,
            token_count=self.token_count,
            status=MetricStatus.OK,
            **kw,  # type: ignore[arg-type]
        )


def perplexity_from_nll(nll_sum: float, token_count: int) -> float | None:
    """Direct PPL from an NLL sum; None when there are no valid tokens."""
    if token_count <= 0:
        return None
    return math.exp(nll_sum / token_count)


def relative_ppl_change(ppl_base: float, ppl_quant: float) -> float | None:
    """``(PPL_quant / PPL_base) - 1``; positive means worse."""
    if ppl_base == 0:
        return None
    if not math.isfinite(ppl_base) or not math.isfinite(ppl_quant):
        return None
    return (ppl_quant / ppl_base) - 1.0


def ppl_change_metric(
    base: MetricRecord, quant: MetricRecord, name: str = "ppl_relative_change"
) -> MetricRecord:
    """Relative PPL degradation as a MetricRecord.

    Refuses to produce a number when either side is unusable, rather than
    inventing one from a missing value.
    """
    if base.status is not MetricStatus.OK or base.value is None:
        return MetricRecord.unavailable(
            name, f"baseline PPL unavailable: {base.reason or base.status.value}"
        )
    if quant.status is not MetricStatus.OK or quant.value is None:
        return MetricRecord.unavailable(
            name, f"quantized PPL unavailable: {quant.reason or quant.status.value}"
        )

    change = relative_ppl_change(base.value, quant.value)
    if change is None:
        return MetricRecord.unavailable(name, "relative change undefined or non-finite")

    return MetricRecord(
        name=name,
        value=change,
        unit="fraction",
        direction=MetricDirection.LOWER_IS_BETTER,
        sample_count=min(base.sample_count, quant.sample_count),
        token_count=0,
        status=MetricStatus.OK,
    )


# --------------------------------------------------------------------------
# Accuracy
# --------------------------------------------------------------------------


def accuracy_from_correct(correct: int, total: int) -> float | None:
    if total <= 0:
        return None
    return correct / total


def accuracy_delta_pp(acc_base: float, acc_quant: float) -> float:
    """Accuracy change in percentage points: ``100 * (acc_quant - acc_base)``."""
    return 100.0 * (acc_quant - acc_base)


def accuracy_metric(
    correct: int, total: int, name: str = "accuracy"
) -> MetricRecord:
    if total <= 0:
        return MetricRecord.unavailable(name, "no evaluated questions", unit="fraction")
    return MetricRecord(
        name=name,
        value=correct / total,
        unit="fraction",
        direction=MetricDirection.HIGHER_IS_BETTER,
        sample_count=total,
        token_count=0,
        status=MetricStatus.OK,
    )


def accuracy_delta_metric(
    base: MetricRecord, quant: MetricRecord, name: str = "accuracy_delta_pp"
) -> MetricRecord:
    """Accuracy difference expressed in percentage points (pp)."""
    if base.status is not MetricStatus.OK or base.value is None:
        return MetricRecord.unavailable(
            name, f"baseline accuracy unavailable: {base.reason or base.status.value}"
        )
    if quant.status is not MetricStatus.OK or quant.value is None:
        return MetricRecord.unavailable(
            name, f"quantized accuracy unavailable: {quant.reason or quant.status.value}"
        )
    return MetricRecord(
        name=name,
        value=accuracy_delta_pp(base.value, quant.value),
        unit="pp",
        direction=MetricDirection.HIGHER_IS_BETTER,
        sample_count=min(base.sample_count, quant.sample_count),
        token_count=0,
        status=MetricStatus.OK,
    )


def has_discriminating_power(accuracy: float, num_choices: int, *, rel_tol: float = 0.02) -> bool:
    """Whether a task can distinguish models at all.

    Accuracy at (or below) guessing chance means the task cannot support a
    conclusion; the report must say so rather than treat it as a measurement
    (full-prd.md §4).
    """
    if num_choices < 2:
        return True
    chance = 1.0 / num_choices
    return accuracy > chance * (1.0 + rel_tol)


def is_guessing_chance(accuracy: float, num_choices: int, *, rel_tol: float = 0.02) -> bool:
    chance = 1.0 / max(num_choices, 1)
    return accuracy <= chance * (1.0 + rel_tol)


# --------------------------------------------------------------------------
# Binary classification reference case (uniform distribution over 2 => PPL 2)
# --------------------------------------------------------------------------


def uniform_ppl(num_classes: int) -> float:
    """Perplexity of a uniform distribution over ``num_classes``; N=2 gives 2."""
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    return float(num_classes)


def nll_of_uniform(num_classes: int) -> float:
    return math.log(num_classes)


# --------------------------------------------------------------------------
# Paired bootstrap for small samples
# --------------------------------------------------------------------------


def paired_bootstrap_delta(
    base_documents: Sequence[tuple[float, int]],
    quant_documents: Sequence[tuple[float, int]],
    *,
    iterations: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, float | int | None]:
    """Bootstrap CI for the paired relative PPL change.

    Documents are resampled in pairs (same index in both conditions), which is
    required because the comparison is paired. NLL sums and token counts are
    re-aggregated per resample; per-document PPLs are never averaged.
    """
    import random

    n = min(len(base_documents), len(quant_documents))
    if n == 0:
        return {
            "n_documents": 0,
            "delta_ppl_relative": None,
            "ci_low": None,
            "ci_high": None,
            "iterations": iterations,
        }

    def _ppl(pairs: Iterable[tuple[float, int]]) -> float | None:
        nll = 0.0
        tokens = 0
        for nll_sum, count in pairs:
            nll += nll_sum
            tokens += count
        if tokens == 0:
            return None
        return math.exp(nll / tokens)

    base_ppl = _ppl(base_documents[:n])
    quant_ppl = _ppl(quant_documents[:n])
    if base_ppl is None or quant_ppl is None or base_ppl == 0:
        return {
            "n_documents": n,
            "delta_ppl_relative": None,
            "ci_low": None,
            "ci_high": None,
            "iterations": iterations,
        }

    observed = (quant_ppl / base_ppl) - 1.0

    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(iterations):
        idx = [rng.randrange(n) for _ in range(n)]
        b = _ppl([base_documents[i] for i in idx])
        q = _ppl([quant_documents[i] for i in idx])
        if b is None or q is None or b == 0:
            continue
        deltas.append((q / b) - 1.0)

    if not deltas:
        return {
            "n_documents": n,
            "delta_ppl_relative": observed,
            "ci_low": None,
            "ci_high": None,
            "iterations": iterations,
        }

    deltas.sort()
    alpha = (1.0 - confidence) / 2.0
    low_index = max(0, int(math.floor(alpha * len(deltas))))
    high_index = min(len(deltas) - 1, int(math.ceil((1.0 - alpha) * len(deltas))) - 1)

    return {
        "n_documents": n,
        "delta_ppl_relative": observed,
        "ci_low": deltas[low_index],
        "ci_high": deltas[high_index],
        "iterations": iterations,
    }


def interval_crosses_zero(ci_low: float | None, ci_high: float | None) -> bool:
    """A CI spanning zero cannot support a directional claim."""
    if ci_low is None or ci_high is None:
        return True
    return ci_low <= 0.0 <= ci_high
