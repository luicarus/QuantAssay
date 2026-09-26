"""The AWQ claims in the public docs must match the measured artifacts.

Two published documents assert specific numbers (perplexity for BF16/GPTQ/AWQ,
their confidence intervals, and "no meaningful speed difference"). Those numbers
came from run directories that are gitignored and may later be cleaned up, so
drift is easy: a doc could keep quoting a figure the evidence no longer supports.

This test does double duty:
  * it reads the numbers from the docs and re-derives them from the artifacts;
  * it skips cleanly when the artifacts are absent (a clone has none), while
    still asserting the doc's internal consistency.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from quantassay.gating import SUPPORTED_QUANT_METHODS  # noqa: E402

RUNS = REPO_ROOT / "runs"
AWQ_RUN = RUNS / "awq-final"
GPTQ_RUN = RUNS / "gptq-final"
GUIDE = (REPO_ROOT / "docs/guide.md").read_text(encoding="utf-8")
README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

#: The reference measurements quoted in docs/guide.md §5.5.
#: These must come from the pair (awq-final, gptq-final), which the guide cites.
DOC_PPL_BF16 = "30.1311"
DOC_PPL_GPTQ = "42.1247"
DOC_PPL_AWQ = "37.9930"
DOC_CI_GPTQ = ("+36.67%", "+43.14%")
DOC_CI_AWQ = ("+23.88%", "+28.52%")


def _has_artifacts() -> bool:
    return all(
        (run / name).is_file()
        for run, name in (
            (AWQ_RUN, "quality-bf16.json"),
            (AWQ_RUN, "quality-awq.json"),
            (GPTQ_RUN, "quality-gptq.json"),
        )
    )


requires_artifacts = pytest.mark.skipif(
    not _has_artifacts(),
    reason="reference run artifacts are local-only and absent in a clone",
)


def test_guide_quotes_the_reference_numbers() -> None:
    """The comparison table must carry every exact figure.

    Strength chosen to match the document: the table holds exact values, while
    the prose deliberately rounds (``+26%``, ``差 13.7 个百分点``). An earlier
    version used a bare `in` check, so removing an occurrence still passed;
    requiring the exact value at least once plus the rounded claim keeps the
    check meaningful without demanding wording the doc never had.
    """
    for value in (DOC_PPL_BF16, DOC_PPL_GPTQ, DOC_PPL_AWQ):
        assert GUIDE.count(value) >= 1, f"guide should quote perplexity {value}"
    for bound in (*DOC_CI_GPTQ, *DOC_CI_AWQ):
        assert GUIDE.count(bound) >= 1, f"guide should quote interval bound {bound}"

    # Both percentages must appear exactly as measured.
    assert "+39.80%" in GUIDE, "the GPTQ degeneration figure must be quoted"
    assert "+26.09%" in GUIDE, "the AWQ degeneration figure must be quoted"

    # And the prose must state the comparison in readable terms.
    assert "13.7 个百分点" in GUIDE, "the prose must state the size of the gap"
    assert "区间不重叠" in GUIDE, "the prose must state that the intervals don't overlap"

    # The superseded figure must not linger anywhere.
    assert "42.1240" not in GUIDE, (
        "guide still carries the superseded GPTQ figure 42.1240; the cited run "
        "measured 42.1247"
    )


def test_guide_states_both_published_intervals() -> None:
    for lo, hi in (DOC_CI_GPTQ, DOC_CI_AWQ):
        assert GUIDE.count(lo) >= 1, f"guide should quote interval bound {lo}"
        assert GUIDE.count(hi) >= 1, f"guide should quote interval bound {hi}"


def test_guide_warns_that_ttft_is_not_conclusive() -> None:
    """Both the metric table and the example must warn about TTFT.

    Presence alone was not enough: the two warnings live in different sections
    for different readers, so each is asserted specifically.
    """
    assert "不要**根据单次运行的 TTFT 差异下结论" in GUIDE, (
        "the serving-metrics section must warn that single-run TTFT differences "
        "are not conclusive"
    )
    assert "TTFT 不参与结论" in GUIDE, (
        "the comparison example must state that TTFT is excluded from the finding"
    )


def test_guide_discloses_the_small_calibration_set_in_both_places() -> None:
    """The 4-prompt caveat must survive in the example AND the FAQ."""
    assert "**仅 4 条校准样本**" in GUIDE, "the example must disclose the calibration size"
    assert "只有 4 条校准样本" in GUIDE, "the caveat must be repeated in the notes"
    assert "幅度不可外推" in GUIDE, "the guide must bound how far the result generalises"


@requires_artifacts
def test_quoted_perplexities_match_the_artifacts() -> None:
    """Each quoted perplexity must equal what the run actually measured."""
    measured = {
        "bf16": json.loads((AWQ_RUN / "quality-bf16.json").read_text())["ppl"],
        "gptq": json.loads((GPTQ_RUN / "quality-gptq.json").read_text())["ppl"],
        "awq": json.loads((AWQ_RUN / "quality-awq.json").read_text())["ppl"],
    }
    for side, quoted in (("bf16", DOC_PPL_BF16), ("gptq", DOC_PPL_GPTQ), ("awq", DOC_PPL_AWQ)):
        # The docs round to 4 decimals; the artifacts carry full precision.
        assert f"{measured[side]:.4f}" == quoted, (
            f"{side}: docs quote {quoted} but the artifact says {measured[side]:.4f}"
        )
        assert f"{measured[side]:.4f}" in GUIDE


@requires_artifacts
def test_quoted_intervals_match_the_measured_comparisons() -> None:
    """Intervals must be the measured ones, not softened for readability."""
    for run, method, (lo, hi) in (
        (GPTQ_RUN, "gptq", DOC_CI_GPTQ),
        (AWQ_RUN, "awq", DOC_CI_AWQ),
    ):
        reg = json.loads((run / "regressions.json").read_text())
        quality = reg.get("quality") or {}
        assert quality.get("comparable") is True, f"{run.name}: comparison not comparable"
        measured_lo = f"{quality['ci_low'] * 100:+.2f}%"
        measured_hi = f"{quality['ci_high'] * 100:+.2f}%"
        assert measured_lo == lo, f"{run.name}: ci_low {measured_lo} != documented {lo}"
        assert measured_hi == hi, f"{run.name}: ci_high {measured_hi} != documented {hi}"


@requires_artifacts
def test_the_awq_quality_advantage_is_a_real_gap() -> None:
    """The headline claim: AWQ preserves perplexity better, with no overlap.

    If a future re-measurement makes the intervals overlap, the guide must be
    corrected rather than left claiming a difference that is no longer supported.
    """
    comparisons = {}
    for run, method in ((GPTQ_RUN, "gptq"), (AWQ_RUN, "awq")):
        reg = json.loads((run / "regressions.json").read_text())
        comparisons[method] = reg["quality"]
    gptq, awq = comparisons["gptq"], comparisons["awq"]
    assert awq["ppl_relative_change"] < gptq["ppl_relative_change"], (
        "the guide claims AWQ degrades perplexity less than GPTQ"
    )
    assert awq["ci_high"] < gptq["ci_low"], (
        "the guide claims non-overlapping intervals; they now overlap, so the "
        "claim of a real difference is no longer supported"
    )


@requires_artifacts
def test_the_documented_speed_equivalence_holds() -> None:
    """The guide claims no meaningful speed difference; TPOT must back it up.

    Also asserts the guide does NOT make a TTFT claim, because same-side TTFT
    varies too much across runs for a single pair of measurements to support one.
    """
    def metrics(run: Path, side: str) -> dict:
        return json.loads((run / f"benchmark_{side}.json").read_text())["summary"]["metrics"]

    gptq = metrics(GPTQ_RUN, "gptq")
    awq = metrics(AWQ_RUN, "awq")
    delta = (awq["tpot"]["value"] - gptq["tpot"]["value"]) / gptq["tpot"]["value"]
    assert abs(delta) <= 0.05, (
        f"docs claim no meaningful speed difference but TPOT differs by {delta:+.1%}"
    )

    # The guide must not present a TTFT-based conclusion.
    assert "TTFT 不参与结论" in GUIDE or "不要**根据单次运行的 TTFT" in GUIDE, (
        "the guide must warn that single-run TTFT differences are not conclusive"
    )


def test_guide_documents_both_methods_and_their_shared_kernel() -> None:
    """The comparison only means something if the shared path is stated.

    Requires the specific claim (same marlin kernel) rather than a loose `or`
    between phrasings: an earlier version accepted "同一个 kernel" too, which let
    a weakened sentence pass.
    """
    for method in SUPPORTED_QUANT_METHODS:
        # Case-insensitive: the docs write the method names in bold caps.
        assert method.lower() in README.lower(), f"README should mention {method}"
        assert method.lower() in GUIDE.lower(), f"guide should mention {method}"
    assert "同一个 marlin kernel" in GUIDE, (
        "the guide must state that both methods execute the same marlin kernel; "
        "without it the comparison's attribution is unsupported"
    )
    assert "对称" in GUIDE, "the guide must state the shared symmetric scheme"


def test_guide_discloses_the_small_calibration_set() -> None:
    """The headline comparison rests on 4 prompts; that must stay visible."""
    assert "4 条校准样本" in GUIDE
    assert "不代表" in GUIDE