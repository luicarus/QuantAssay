"""Paired bootstrap intervals for pooled perplexity comparisons."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


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
