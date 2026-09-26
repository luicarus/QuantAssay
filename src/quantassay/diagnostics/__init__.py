"""Output comparison: token agreement, divergence cases and generation differences."""

from quantassay.diagnostics.token_diff import (
    GenerationSample,
    TokenAgreementResult,
    TopKComparison,
    compare_token_sequences,
    first_divergence,
    softmax_entropy,
    summarize_generations,
    top_k_from_logits,
)

__all__ = [
    "GenerationSample",
    "TokenAgreementResult",
    "TopKComparison",
    "compare_token_sequences",
    "first_divergence",
    "softmax_entropy",
    "summarize_generations",
    "top_k_from_logits",
]
