"""Stage independence: no stage may depend on another having run in the same
invocation.

A real failure this locks out: running `--stage quality_gptq` after the
benchmarks had already been recorded in a *previous* invocation crashed with
`UnboundLocalError: cannot access local variable 'workload'`, because the
workload was only bound inside the benchmark branch. The stage had already
succeeded, so the journal looked clean while the report silently claimed quality
was not evaluated — success message, missing evidence.

The check is deliberately narrow: it looks for names bound *inside a branch* and
read *at the same indentation level as the branch* (i.e. outside it), which is
precisely the mechanism. A broader heuristic produced 28 false positives on
correct closures and would simply be ignored.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

GATING = REPO_ROOT / "src" / "quantassay" / "gating.py"


def _main_function() -> ast.FunctionDef:
    tree = ast.parse(GATING.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    raise AssertionError("main() not found in gating.py")


def _names_bound_in(node: ast.AST) -> set[str]:
    """Names assigned anywhere inside this subtree (not nested functions)."""
    bound: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            bound.add(child.id)
    return bound


def test_no_branch_local_binding_escapes_its_branch() -> None:
    """A name assigned only under an `if` and read outside it may be unbound."""
    main = _main_function()

    # Collect names bound inside a top-level `if` body of main together with
    # that body's indentation, then look for reads at the enclosing level.
    escaping: list[str] = []
    for stmt in main.body:
        if not isinstance(stmt, ast.If):
            continue
        for branch_body, branch_label in (
            (stmt.body, "if"),
            (stmt.orelse, "else"),
        ):
            if not branch_body:
                continue
            bound_inside = set()
            for inner in branch_body:
                bound_inside |= _names_bound_in(inner)
            # Reads after the if-statement, at the same level (main.body scope).
            index = main.body.index(stmt)
            for later in main.body[index + 1 :]:
                for node in ast.walk(later):
                    if (
                        isinstance(node, ast.Name)
                        and isinstance(node.ctx, ast.Load)
                        and node.id in bound_inside
                    ):
                        escaping.append(
                            f"{node.id} bound only in the `{branch_label}` branch "
                            f"(line {stmt.lineno}) but read at line {node.lineno}"
                        )

    assert escaping == [], (
        "branch-local bindings escape their branch; a stage run on its own would "
        "raise UnboundLocalError after already recording success:\n  "
        + "\n  ".join(sorted(set(escaping)))
    )


def test_quality_branch_binds_what_it_needs() -> None:
    """Direct check of the specific regression."""
    source = GATING.read_text(encoding="utf-8")
    start = source.index('if args.stage in ("quality_bf16", f"quality_{quant_method}", "full"):')
    branch = source[start : start + 900]
    assert "load_workload(args.workload_config)" in branch, (
        "the quality branch must bind its own workload: relying on the benchmark "
        "branch is what caused the UnboundLocalError"
    )


def test_measurement_waits_for_a_clean_gpu_before_starting_the_server() -> None:
    """The settle-wait must precede the server, not follow it.

    `mem-fraction-static=0.8` scales the KV pool with whatever the card already
    carries, so starting a server while a previous CUDA context is still tearing
    down shrinks the pool and slows decode. Waiting *after* the server started
    (the first version of this fix) only measures the damage it was meant to
    prevent — it reported the elevated baseline and blocked a healthy run.
    """
    source = GATING.read_text(encoding="utf-8")
    tree = ast.parse(source)

    checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        wait_lines = [
            n.lineno
            for n in ast.walk(node)
            if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "wait_for_clean_gpu"
        ]
        session_lines = [
            n.lineno
            for n in ast.walk(node)
            if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "server_session"
        ]
        if not wait_lines or not session_lines:
            continue
        checked += 1
        assert min(wait_lines) < min(session_lines), (
            f"{node.name}: wait_for_clean_gpu (line {min(wait_lines)}) must run "
            f"before server_session (line {min(session_lines)})"
        )

    # Both benchmark operations and the quality operation are covered.
    assert checked >= 3, f"expected at least 3 guarded measurement paths, found {checked}"