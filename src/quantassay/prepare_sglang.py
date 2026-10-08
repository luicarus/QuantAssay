"""Copy installed SGLang 0.5.3 into an isolated source tree and apply a patch."""

import argparse
import importlib.metadata
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

from quantassay.experiments.store import atomic_write_json, file_sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="new directory containing the copied sglang package")
    parser.add_argument("--patch", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output directory already exists; use a new source directory")
    if importlib.metadata.version("sglang") != "0.5.3":
        parser.error("this source patch supports SGLang 0.5.3 only")
    if not args.patch.is_file() or not shutil.which("patch"):
        parser.error("an existing patch file and GNU patch executable are required")
    spec = importlib.util.find_spec("sglang")
    source = Path(spec.origin).parent
    args.output.mkdir(parents=True)
    shutil.copytree(source, args.output / "sglang",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    files = ("srt/managers/schedule_policy.py", "srt/server_args.py")
    original = {rel: file_sha256(source / rel) for rel in files}
    subprocess.run(["patch", "--batch", "--forward", "-p1", "-i", str(args.patch.resolve())],
                   cwd=args.output, check=True)
    atomic_write_json(args.output / "engine-manifest.json", {
        "sglang_version": "0.5.3", "source_package": str(source),
        "patch_sha256": file_sha256(args.patch), "original_source_hashes": original,
        "patched_source_hashes": {rel: file_sha256(args.output / "sglang" / rel) for rel in files},
        "scope": "isolated installed-source copy; original environment is not modified",
    })
    print(json.dumps({"engine_source": str(args.output), "status": "prepared"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
