"""`--stage full` must run the whole pipeline in a valid order.

The chain is input model -> quantize -> evaluate (serving + quality) -> score.
Two ordering invariants make it valid, and both are easy to break by reordering
branch checks:

1. a stage that *requires* another must come after it (gptq needs bf16;
   quality_gptq needs gptq_service and quality_bf16);
2. the report/advice emission must come after both evaluation phases, or the
   report would be rendered before the quality comparison exists.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

GATING = REPO_ROOT / "src" / "quantassay" / "gating.py"


def _full_includes_every_stage() -> None:
    source = GATING.read_text(encoding="utf-8")
    # Every individual stage condition must also accept "full".
    for stage in (
        "bf16",
        "gptq",
        "gptq_service",
        "benchmark_bf16",
        "benchmark_gptq",
        "quality_bf16",
        "quality_gptq",
    ):
        assert f'"{stage}"' in source, f"stage {stage} disappeared from gating.py"


def test_full_is_accepted_by_every_stage_branch() -> None:
    """A stage reachable alone must also be reachable under --stage full."""
    source = GATING.read_text(encoding="utf-8")
    tree = ast.parse(source)

    checks: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        segment = ast.get_source_segment(source, node) or ""
        if "args.stage" not in segment:
            continue
        checks.append((node.lineno, segment))

    # Every stage condition must mention "full" as an accepted literal.
    stage_conditions = [
        (line, seg) for line, seg in checks if any(
            s in seg for s in ("bf16", "gptq", "quality", "benchmark")
        )
    ]
    assert stage_conditions, "no stage conditions found"

    missing = [
        (line, seg) for line, seg in stage_conditions if '"full"' not in seg
    ]
    assert missing == [], (
        "these stage conditions do not accept --stage full, so a full run would "
        "silently skip that stage:\n  "
        + "\n  ".join(f"line {line}: {seg}" for line, seg in missing)
    )


def test_quality_runs_after_the_serving_benchmarks() -> None:
    """Scoring consumes both evaluation phases, so it must be emitted last."""
    source = GATING.read_text(encoding="utf-8")
    benchmark_block = source.index('if args.stage in ("benchmark_bf16", f"benchmark_{quant_method}", "full"):')
    quality_block = source.index('if args.stage in ("quality_bf16", f"quality_{quant_method}", "full"):')
    assert benchmark_block < quality_block, (
        "the quality branch must come after the benchmark branch: the report is "
        "rendered from serving results and then re-rendered with quality folded in"
    )


def test_quality_attach_happens_after_the_quality_stages() -> None:
    """The report must be re-rendered only once a quality comparison exists."""
    source = GATING.read_text(encoding="utf-8")
    # Search from the end: `source.index` would find the *definition* first.
    call = source.rindex("_attach_quality_to_report(")
    quality_branch = source.index('if args.stage in (f"quality_{quant_method}", "full"):')
    assert quality_branch < call, (
        "_attach_quality_to_report must be called inside the quantized-quality "
        "branch, after that stage has produced its summary"
    )
    # And it must be told which method's summary to compare, not assume GPTQ.
    assert "method=quant_method" in source[call : call + 220], (
        "the attach call must pass the selected method; a hardcoded candidate "
        "filename made an AWQ run report 'quality not evaluated' after measuring it"
    )


def test_cli_help_documents_both_modes() -> None:
    """`all` and `full` must not be confusable; the help text says which is which."""
    source = GATING.read_text(encoding="utf-8")
    assert '"full",' in source
    assert "'all' = capability gates only" in source
    assert "the whole pipeline" in source