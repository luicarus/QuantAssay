"""Regression analysis and advisor (mvp-prd.md §5, full-prd.md §6)."""

from quantassay.analysis.advisor import advise
from quantassay.analysis.quality import (
    MIN_MEANINGFUL_PPL_CHANGE,
    check_quality_comparability,
    compare_quality,
)
from quantassay.analysis.regression import analyze, check_comparability

__all__ = [
    "MIN_MEANINGFUL_PPL_CHANGE",
    "advise",
    "analyze",
    "check_comparability",
    "check_quality_comparability",
    "compare_quality",
]