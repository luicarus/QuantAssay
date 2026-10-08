"""Select external SGLang sources and retain their real revision identity."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from quantassay.experiments.store import file_sha256


ENGINE_FILES = (
    "srt/managers/scheduler.py", "srt/managers/schedule_policy.py",
    "srt/managers/schedule_batch.py", "srt/mem_cache/radix_cache.py",
    "srt/mem_cache/allocator.py", "srt/metrics/collector.py",
    "srt/managers/tokenizer_manager.py", "srt/server_args.py", "version.py",
)


def _git_identity(path: Path) -> dict[str, Any] | None:
    def git(*args):
        result = subprocess.run(["git", "-C", str(path), *args],
                                capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None

    try:
        root = git("rev-parse", "--show-toplevel")
        if root is None:
            return None
        status = git("status", "--porcelain")
        return {"root": root, "commit": git("rev-parse", "HEAD"),
                "branch": git("branch", "--show-current"),
                "dirty": bool(status) if status is not None else None}
    except (OSError, subprocess.TimeoutExpired):
        return None


def configure_engine_source(source: Path | None) -> dict[str, Any] | None:
    """Expose one source tree to both this controller and its server children.

    Dependencies remain in the existing execution environment. The Git commit
    and source hashes identify the code actually selected via PYTHONPATH.
    """
    if source is None:
        return None
    source = source.resolve()
    package = source / "sglang"
    if not all((package / rel).is_file() for rel in ENGINE_FILES):
        raise ValueError("engine source must contain the complete SGLang Python package")
    version = None
    for node in ast.parse((package / "version.py").read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets
        ):
            version = ast.literal_eval(node.value)
    if version != "0.5.3":
        raise ValueError("external engine sources currently require the locked SGLang 0.5.3 version")
    loaded = sys.modules.get("sglang")
    if loaded is not None and Path(loaded.__file__).resolve().parent != package:
        raise ValueError("another SGLang source is already imported; start a fresh controller")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    parts = [part for part in os.environ.get("PYTHONPATH", "").split(os.pathsep)
             if part and part != str(source)]
    os.environ["PYTHONPATH"] = os.pathsep.join([str(source), *parts])
    tree_hashes = {
        path.relative_to(package).as_posix(): file_sha256(path)
        for path in sorted(package.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    return {"source_dir": str(source), "package_dir": str(package),
            "source_version": version, "git": _git_identity(source),
            "source_file_count": len(tree_hashes),
            "source_tree_sha256": hashlib.sha256(json.dumps(tree_hashes, sort_keys=True).encode()).hexdigest(),
            "source_hashes": {rel: file_sha256(package / rel) for rel in ENGINE_FILES}}
