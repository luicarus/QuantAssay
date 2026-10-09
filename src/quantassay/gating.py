"""Capability gating and sequential run orchestration for the WSL2 execution layer.

This module does not install packages or download weights. The caller supplies a
local Qwen3-0.6B snapshot and its immutable revision. GPU work is never started
until the preflight gate passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from quantassay.analysis.regression import compare_serving_parameters as compare_serving_parameters
from quantassay.contracts import (
    MODEL_ID,
    DataSplit,
    RegressionReport,
    WorkloadSpec,
)
from quantassay.evaluation.corpus import (
    DEFAULT_CORPUS as DEFAULT_QUALITY_CORPUS,
    CorpusSpec,
    holdout_records,
)
from quantassay.evaluation.data import PreparedSample, PrepareResult
from quantassay.evaluation.quality import score_documents
from quantassay.experiments.store import StoreError, atomic_write_json, file_sha256, read_json
from quantassay.reporting.pipeline import attach_quality, build_comparison_report, save_report
from quantassay.serving.benchmark import benchmark, load_records
from quantassay.serving.workload import build_request_set, request_set_hashes
from quantassay.runtime import (
    DEFAULT_ATTENTION_BACKEND,
    DEFAULT_CUDA_GRAPH_MAX_BS,
    DEFAULT_OPERATOR_BACKEND,
    KNOWN_ATTENTION_BACKENDS,
    KNOWN_OPERATOR_BACKENDS,
    CLEAN_GPU_MIB as CLEAN_GPU_MIB,
    GPU_CONCERN_MIB as GPU_CONCERN_MIB,
    MIN_FREE_MIB_FOR_MEASUREMENT as MIN_FREE_MIB_FOR_MEASUREMENT,
    PEAK_ANOMALY_MIB as PEAK_ANOMALY_MIB,
    ProbeError,
    _GpuPeakSampler,
    _LIVE_CHILDREN as _LIVE_CHILDREN,
    _REVISION_RE,
    _install_child_reaper as _install_child_reaper,
    _package_version,
    _reap_live_children as _reap_live_children,
    _sample_gpu_used_mib,
    _stop_process_group,
    _tracked_process,
    _wait_gpu_release,
    anomaly_warnings,
    build_bf16_command,
    build_smoke_payload as build_smoke_payload,
    contention_blockers,
    parse_sse_stream as parse_sse_stream,
    resource_blockers,
    resource_warnings,
    run_server_smoke,
    server_session,
    source_file_hashes,
    verify_snapshot_layout,
    wait_for_clean_gpu,
)


def _gpu_info() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
        row = completed.stdout.strip().splitlines()[0].split(",")
        if len(row) != 4:
            raise ValueError("unexpected nvidia-smi column count")
        return {
            "gpu_name": row[0].strip(),
            "gpu_total_mib": int(row[1].strip()),
            "gpu_free_mib": int(row[2].strip()),
            "driver": row[3].strip(),
            "gpu_error": None,
        }
    except (OSError, subprocess.SubprocessError, IndexError, ValueError) as exc:
        return {
            "gpu_name": None,
            "gpu_total_mib": None,
            "gpu_free_mib": None,
            "driver": None,
            "gpu_error": str(exc),
        }


def _available_ram_gib() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return round(int(line.split()[1]) / 1024**2, 2)
    return None


def _cpu_check(command: list[str], *, timeout: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "output": (completed.stdout + completed.stderr)[-3000:],
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "returncode": None, "output": str(exc)}


def _index_weight_shard(model_dir: Path) -> str | None:
    """Return the weight index file content hash, when a sharded model uses one."""
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        candidate = model_dir.resolve() / name
        if candidate.is_file():
            return file_sha256(candidate)
    return None


def collect_preflight(model_dir: Path, revision: str) -> dict[str, Any]:
    """Capture facts only; validation is separate from collection."""
    model_dir = model_dir.resolve()
    config_path = model_dir / "config.json"
    try:
        model_type = json.loads(config_path.read_text(encoding="utf-8")).get("model_type")
        model_config_error = None
    except (OSError, ValueError, AttributeError) as exc:
        model_type = None
        model_config_error = str(exc)
    try:
        import torch

        torch_version = torch.__version__
        torch_cuda = torch.version.cuda
        cuda_available = bool(torch.cuda.is_available())
        torch_error = None
    except Exception as exc:
        torch_version = None
        torch_cuda = None
        cuda_available = False
        torch_error = repr(exc)
    pip_check = _cpu_check([sys.executable, "-m", "pip", "check"], timeout=60)
    # Required: without these the M0 pipeline cannot run at all.
    required_import = _cpu_check(
        [
            sys.executable,
            "-c",
            "import sglang.launch_server; "
            "from compressed_tensors.quantization import preset_name_to_scheme; "
            "s=preset_name_to_scheme('W4A16', ['Linear']); "
            "assert s.weights.num_bits == 4 and s.weights.group_size == 128; "
            "from datasets import Dataset; "
            "from transformers import AutoModelForCausalLM, AutoTokenizer; "
            "import psutil",
        ],
        timeout=90,
    )
    # Diagnostic only: GPTQModifier construction pulls in optional acceleration
    # backends (e.g. transformer_engine, which needs cuDNN). A failure here is
    # recorded, not treated as a blocker, because the real quantization runs in
    # an isolated worker that will surface its own error. Blocking on it would
    # stop runs that would otherwise succeed.
    gptq_import = _cpu_check(
        [
            sys.executable,
            "-c",
            # llmcompressor moved GPTQModifier between versions; try both paths.
            "try:\n"
            "    from llmcompressor.modifiers.gptq import GPTQModifier\n"
            "except ImportError:\n"
            "    from llmcompressor.modifiers.quantization import GPTQModifier\n"
            "from compressed_tensors.quantization import preset_name_to_scheme; "
            "s=preset_name_to_scheme('W4A16', ['Linear']); "
            # 0.9-era GPTQModifier takes scheme args positionally rather than
            # config_groups; construction alone is the check.
            "print('GPTQModifier importable')",
        ],
        timeout=90,
    )
    disk = shutil.disk_usage(model_dir if model_dir.exists() else model_dir.parent)
    return {
        "model_id": MODEL_ID,
        "model_dir": str(model_dir),
        "model_config_exists": config_path.is_file(),
        "model_type": model_type,
        "model_config_error": model_config_error,
        "revision": revision,
        "revision_valid": bool(_REVISION_RE.fullmatch(revision)),
        "snapshot_revision_verified": verify_snapshot_layout(model_dir, revision),
        "system": platform.system(),
        "python": platform.python_version(),
        "torch": torch_version,
        "torch_cuda": torch_cuda,
        "torch_error": torch_error,
        "cuda_available": cuda_available,
        "disk_free_gib": round(disk.free / 1024**3, 2),
        "ram_available_gib": _available_ram_gib(),
        "packages": {
            name: _package_version(name)
            for name in ("sglang", "llmcompressor", "compressed-tensors", "transformers", "datasets", "psutil")
        },
        "required_import_ok": required_import["ok"],
        "required_import_output": required_import["output"],
        "gptq_import_ok": gptq_import["ok"],
        "gptq_import_output": gptq_import["output"],
        # Kept for the record only. The image ships unrelated packages (vllm,
        # ms-swift, litellm) whose internal version pins cannot all be satisfied
        # at once; that is not a reason to block this experiment.
        "pip_check_ok": pip_check["ok"],
        "pip_check_output": pip_check["output"],
        **_gpu_info(),
    }


def validate_preflight(report: dict[str, Any]) -> list[str]:
    """Return blockers; never silently change model, dtype or dependencies."""
    reasons: list[str] = []
    if report.get("system") not in (None, "Linux"):
        reasons.append("the execution layer requires Linux (WSL2)")
    if not str(report.get("python", "")).startswith("3.12."):
        reasons.append("Python 3.12 is required in the execution layer")
    # Record installed versions; the gate requires usable torch/CUDA.
    torch_version = str(report.get("torch") or "")
    if not torch_version:
        reasons.append("torch is not installed in the execution layer")
    torch_cuda = str(report.get("torch_cuda") or "")
    if not report.get("cuda_available"):
        reasons.append("CUDA must be visible from the project environment")
    elif not torch_cuda:
        reasons.append("torch reports no CUDA runtime version")
    gpu_name = str(report.get("gpu_name") or "")
    if "RTX 3050" not in gpu_name:
        reasons.append("RTX 3050 Ti Laptop GPU was not detected")
    if report.get("gpu_total_mib") is None:
        reasons.append("GPU memory could not be read from nvidia-smi")
    # Free-memory floor for the supported 0.6B model on the 4 GB card.
    if report.get("gpu_free_mib") is not None and report["gpu_free_mib"] < 2000:
        reasons.append("less than 2000 MiB free GPU memory on the 4 GB card")
    if report.get("ram_available_gib") is not None and report["ram_available_gib"] < 5:
        reasons.append("less than 5 GiB available RAM for 0.6B CPU loading")
    if report.get("model_config_exists") is False:
        reasons.append("local model snapshot is missing config.json")
    elif report.get("model_config_exists") is True and report.get("model_type") != "qwen3":
        reasons.append("model config must declare model_type=qwen3")
    if report.get("revision_valid") is False:
        reasons.append("model revision must be a 40-character commit SHA")
    if report.get("snapshot_revision_verified") is False:
        reasons.append("model directory must be the Qwen3-0.6B cache snapshot for that revision")
    packages = report.get("packages") or {}
    for name in ("sglang", "llmcompressor", "compressed-tensors", "transformers"):
        if not packages.get(name):
            reasons.append(f"{name} is not installed in the active environment")
    # Only imports the pipeline genuinely needs are blocking. GPTQModifier is
    # checked for the record: constructing it can fail on optional acceleration
    # backends (transformer_engine needs cuDNN) while quantization itself may
    # still work, and the isolated worker reports its own failure if not.
    if report.get("required_import_ok") is False:
        reasons.append("required SGLang/quantization runtime import failed")
    if report.get("disk_free_gib") is not None and report["disk_free_gib"] < 60:
        reasons.append("less than 60 GiB free disk for model and experiment artifacts")
    return reasons


def preflight_warnings(report: dict[str, Any]) -> list[str]:
    """Non-blocking observations worth recording before a GPU session starts.

    These do not stop the run, but they must be visible: a silently ignored
    warning is how an unexplained failure later becomes unexplainable.
    """
    warnings: list[str] = []
    if report.get("pip_check_ok") is False:
        warnings.append(
            "pip check reports conflicts (the image ships unrelated packages such as "
            "vllm/ms-swift whose pins cannot all be satisfied); not blocking"
        )
    if report.get("gptq_import_ok") is False:
        warnings.append(
            "GPTQModifier could not be constructed in the controller environment; "
            "see gptq_import_output. The quantization worker will report its own error"
        )
    return warnings


def build_quant_command(
    model_dir: Path, revision: str, output_dir: Path, *, method: str = "gptq"
) -> list[str]:
    if method not in SUPPORTED_QUANT_METHODS:
        raise ProbeError(
            f"unknown quantization method {method!r}; supported: {list(SUPPORTED_QUANT_METHODS)}"
        )
    return [
        sys.executable,
        "-m",
        "quantassay.quantize_worker",
        "--model-dir",
        str(model_dir.resolve()),
        "--revision",
        revision,
        "--output-dir",
        str(output_dir.resolve()),
        "--method",
        method,
    ]


def read_quant_artifact(checkpoint: Path, *, method: str | None = None) -> dict[str, Any]:
    """Validate the committed artifact and every recorded file checksum.

    The expected recipe is read from the artifact's own recorded method rather
    than assumed to be GPTQ: both supported methods share the W4A16 shape and
    differ only in the algorithm field. ``method``, when given, additionally
    asserts which method produced it, so a caller cannot accept the wrong
    algorithm's checkpoint.
    """
    manifest_path = checkpoint / "artifact-manifest.json"
    if not manifest_path.is_file():
        raise ProbeError(f"{method or 'quantization'} artifact manifest is missing")
    try:
        manifest = read_json(manifest_path)
    except Exception as exc:
        raise ProbeError(f"invalid quantization artifact manifest: {exc}") from exc
    if manifest.get("model_id") != MODEL_ID or not _REVISION_RE.fullmatch(
        str(manifest.get("parent_revision", ""))
    ):
        raise ProbeError("quantization artifact has the wrong model identity")
    # Legacy artifacts predate the `method` field; they were all GPTQ.
    recorded_method = manifest.get("method") or "gptq"
    if recorded_method not in SUPPORTED_QUANT_METHODS:
        raise ProbeError(
            f"artifact records unknown quantization method {recorded_method!r}"
        )
    if method is not None and recorded_method != method:
        raise ProbeError(
            f"expected a {method!r} artifact but {checkpoint} was produced by "
            f"{recorded_method!r}"
        )
    recipe = manifest.get("recipe") or {}
    expected_algorithm = recorded_method.upper()
    if (
        str(recipe.get("algorithm", "")).upper() != expected_algorithm
        or recipe.get("scheme") != "W4A16"
        or recipe.get("group_size") != 128
    ):
        raise ProbeError(
            f"{expected_algorithm} artifact has the wrong recipe: {recipe!r}"
        )
    files = manifest.get("files") or {}
    if not isinstance(files, dict) or not files:
        raise ProbeError("quantization artifact file checksums are missing")
    if not {"config.json", "tokenizer.json"}.issubset(files) or not any(
        name.endswith(".safetensors") for name in files
    ):
        raise ProbeError("artifact lacks packed config, tokenizer or weights")
    root = checkpoint.resolve()
    for relative, expected in files.items():
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ProbeError(f"artifact file is missing or escaped: {relative}")
        if file_sha256(path) != expected:
            raise ProbeError(f"artifact checksum mismatch: {relative}")
    from quantassay.quantize_worker import QuantizeError, inspect_packed_checkpoint

    try:
        inspected = inspect_packed_checkpoint(checkpoint)
    except QuantizeError as exc:
        raise ProbeError(f"artifact is not a verified packed W4A16 checkpoint: {exc}") from exc
    if manifest.get("format") != inspected:
        raise ProbeError("artifact packed format does not match manifest")
    return manifest


def classify_quant_kernel(log_path: Path, *, method: str | None = None) -> str | None:
    """Require an explicit runtime kernel marker, not just checkpoint metadata.

    Both supported methods converge on the same kernel here, because both emit
    symmetric compressed-tensors W4A16. The parameter is kept so a future method
    with a distinct kernel cannot silently inherit this one's answer.
    """
    if not log_path.is_file():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if re.search(r"Using\s+MarlinLinearKernel\s+for\s+CompressedTensorsWNA16", text):
        return "compressed_tensors_wna16_marlin"
    if re.search(r"Using\s+CompressedTensorsWNA16MarlinMethod", text):
        return "compressed_tensors_wna16_marlin"
    return None


#: Quantization methods this orchestrator can run. Kept in sync with the worker's
#: ``QUANT_METHODS``; a mismatch is caught by tests rather than at runtime.
SUPPORTED_QUANT_METHODS = ("gptq", "awq")


def quant_artifact_name(method: str) -> str:
    """Directory name for a method's checkpoint.

    Method-specific so two methods can live in one run directory without
    overwriting each other — the whole point of adding a second method is to
    compare them side by side.
    """
    if method not in SUPPORTED_QUANT_METHODS:
        raise ProbeError(
            f"unknown quantization method {method!r}; supported: {list(SUPPORTED_QUANT_METHODS)}"
        )
    return f"{method}-w4a16"


def quant_checkpoint_dir(run_dir: Path, method: str) -> Path:
    """Checkpoint path for ``method`` inside a run directory."""
    return run_dir / "artifacts" / quant_artifact_name(method)


def method_stage_names(method: str) -> dict[str, str]:
    """Stage names for a method: quantize / service / benchmark / quality.

    Names are derived, not duplicated per method, so adding a third method does
    not mean copying a block of stage wiring.
    """
    return {
        "quantize": method,
        "service": f"{method}_service",
        "benchmark": f"benchmark_{method}",
        "quality": f"quality_{method}",
    }


def selected_quant_methods(args: Any) -> list[str]:
    """Return the one selected method; each run directory owns one recipe."""
    requested = getattr(args, "quant_method", None) or "gptq"
    if requested not in SUPPORTED_QUANT_METHODS:
        raise ProbeError(
            f"unknown quantization method {requested!r}; "
            f"supported: {list(SUPPORTED_QUANT_METHODS)}"
        )
    return [requested]


def stage_is_known(stage: str) -> bool:
    """Whether ``--stage`` names a real stage (single stage or mode)."""
    if stage in ("preflight", "bf16", "all", "full", "benchmark_bf16"):
        return True
    for method in SUPPORTED_QUANT_METHODS:
        if stage in method_stage_names(method).values() or stage == f"benchmark_{method}":
            return True
    return False


def verify_service_log(result: dict[str, Any]) -> bool:
    path = Path(result.get("log_path", ""))
    expected = result.get("log_sha256")
    return bool(expected and path.is_file() and file_sha256(path) == expected)


def verify_quant_service_evidence(result: dict[str, Any]) -> bool:
    if not verify_service_log(result):
        return False
    checkpoint = Path(result.get("checkpoint_path", ""))
    manifest_path = checkpoint / "artifact-manifest.json"
    if not manifest_path.is_file() or not result.get("checkpoint_manifest_sha256"):
        return False
    if file_sha256(manifest_path) != result["checkpoint_manifest_sha256"]:
        return False
    try:
        read_quant_artifact(checkpoint, method=result.get("method"))
    except ProbeError:
        return False
    return result.get("quant_kernel") == classify_quant_kernel(
        Path(result["log_path"]), method=result.get("method")
    )


def serving_parameters(args: Any) -> dict[str, Any]:
    """The serving configuration the two sides must share exactly."""
    parameters = {
        "attention_backend": args.attention_backend,
        "operator_backend": args.operator_backend,
        "mem_fraction_static": args.mem_fraction_static,
        "cuda_graph_max_bs": args.cuda_graph_max_bs,
        "disable_cuda_graph": bool(args.disable_cuda_graph),
        "context_length": 512,
        "max_running_requests": 1,
        "dtype": "bfloat16",
        "sglang_version": _package_version("sglang"),
    }
    if getattr(args, "engine_identity", None) is not None:
        parameters["engine_identity"] = args.engine_identity
    return parameters


def verify_benchmark_evidence(result: dict[str, Any]) -> bool:
    """A cached benchmark is reusable only if its JSONL evidence is intact.

    Re-running the measurement silently would be worse than re-running it
    loudly: the summary would no longer describe the file on disk.
    """
    records_path = Path(result.get("records_path", ""))
    summary = result.get("summary") or {}
    if not records_path.is_file() or not summary:
        return False
    try:
        records = load_records(records_path)
    except Exception:
        return False
    # The file must still hold exactly the requests the summary counted.
    if len(records) != summary.get("requests_total"):
        return False
    log_path = Path(result.get("log_path", ""))
    expected = result.get("log_sha256")
    if not expected or not log_path.is_file():
        return False
    return file_sha256(log_path) == expected


# --------------------------------------------------------------------------
# Quality evaluation helpers (full-prd.md §4 layer 2)
# --------------------------------------------------------------------------


def verify_quality_evidence(result: dict[str, Any]) -> bool:
    """A cached quality stage is reusable only if its JSONL is intact.

    Also re-checks the dataset fingerprint: reusing a measurement taken on a
    different corpus slice would silently compare different data.
    """
    records_path = Path(result.get("records_path", ""))
    summary = result.get("summary") or {}
    if not records_path.is_file() or not summary:
        return False
    try:
        lines = [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return False
    if len(lines) != summary.get("documents_total"):
        return False
    if result.get("dataset_fingerprint") != summary.get("dataset_fingerprint"):
        return False
    log_path = Path(result.get("log_path", ""))
    expected = result.get("log_sha256")
    if not expected or not log_path.is_file():
        return False
    return file_sha256(log_path) == expected


def corpus_spec_from_args(args: Any) -> "CorpusSpec":
    """Resolve corpus options; only wikitext inherits its default subset name."""
    from quantassay.evaluation.corpus import (
        DEFAULT_CORPUS,
        DEFAULT_CORPUS_CONFIG,
        DEFAULT_CORPUS_ENDPOINT,
        DEFAULT_CORPUS_SPLIT,
        DEFAULT_TEXT_COLUMN,
        CorpusSpec,
    )

    corpus = getattr(args, "corpus", None)
    if not corpus:
        raise ProbeError("--corpus must name a HuggingFace dataset id")
    default_config = DEFAULT_CORPUS_CONFIG if corpus == DEFAULT_CORPUS else ""
    spec = CorpusSpec(
        corpus=corpus,
        config=getattr(args, "corpus_config", None)
        if getattr(args, "corpus_config", None) is not None
        else default_config,
        source_split=getattr(args, "corpus_split", None) or DEFAULT_CORPUS_SPLIT,
        text_column=getattr(args, "corpus_text_column", None) or DEFAULT_TEXT_COLUMN,
        endpoint=getattr(args, "corpus_endpoint", None) or DEFAULT_CORPUS_ENDPOINT,
        revision=getattr(args, "corpus_revision", None) or None,
        min_chars=getattr(args, "corpus_min_chars", 400),
    )
    if spec.min_chars <= 0:
        raise ProbeError("--corpus-min-chars must be positive")
    return spec


def prepare_quality_data(
    run_dir: Path,
    *,
    spec: "CorpusSpec",
    calibration_samples: int,
    documents: int,
    max_tokens: int,
) -> "PrepareResult":
    """Materialize the quality corpus described by ``spec`` and persist its manifest.

    The manifest is written once and reused, so a rerun scores the same text.
    If the stored manifest came from a *different* corpus, the run directory is
    refused rather than silently measuring two datasets under one run id.
    """
    manifest_path = run_dir / "data-manifest.json"
    if manifest_path.is_file():
        stored = read_json(manifest_path)
        if stored.get("corpus_fingerprint") and stored.get("dataset"):
            from quantassay.contracts import DatasetManifest
            from quantassay.evaluation.corpus import (
                DEFAULT_CORPUS_SPLIT,
                DEFAULT_TEXT_COLUMN,
            )

            # A manifest written before these fields existed records None for
            # them. Treat that as "the default that was in force then" rather
            # than as a mismatch, or every pre-existing run would be refused on
            # resume — breaking exactly the recoverability the journal exists to
            # provide.
            stored_identity = (
                stored.get("corpus"),
                stored.get("corpus_config"),
                stored.get("corpus_split") or DEFAULT_CORPUS_SPLIT,
                stored.get("corpus_text_column") or DEFAULT_TEXT_COLUMN,
            )
            wanted_identity = (spec.corpus, spec.config, spec.source_split, spec.text_column)
            if stored_identity != wanted_identity:
                raise ProbeError(
                    f"run directory already holds quality data for "
                    f"{stored_identity[0]!r} (split {stored_identity[2]!r}, column "
                    f"{stored_identity[3]!r}), but {wanted_identity[0]!r} (split "
                    f"{wanted_identity[2]!r}, column {wanted_identity[3]!r}) was "
                    "requested. Use a new run directory: mixing corpora under one "
                    "run id would make the comparison meaningless."
                )
            dataset = DatasetManifest.model_validate(stored["dataset"])
            texts = stored.get("texts") or {}
            samples = [
                PreparedSample(
                    record=record,
                    text=texts.get(record.sample_id, ""),
                )
                for record in dataset.samples
            ]
            return PrepareResult(manifest=dataset, samples=samples)

    from quantassay.contracts import DataConfig
    from quantassay.evaluation.corpus import (
        build_quality_dataset,
        dataset_fingerprint,
        load_corpus,
    )

    data_config = DataConfig(
        source=spec.corpus,
        revision=spec.revision or spec.config,
        language=["en"],
        calibration_samples=calibration_samples,
        dev_ppl_documents=documents,
        final_ppl_documents=0,
        max_tokens=max_tokens,
        smoke_max_tokens=128,
    )
    corpus = load_corpus(spec)
    result = build_quality_dataset(
        corpus=corpus,
        data_config=data_config,
        tokenizer_id=MODEL_ID,
        min_chars=spec.min_chars,
    )
    fingerprint = dataset_fingerprint(result.manifest, corpus)
    atomic_write_json(
        manifest_path,
        {
            "corpus": corpus.corpus,
            "corpus_config": corpus.config,
            "corpus_split": corpus.source_split,
            "corpus_text_column": corpus.text_column,
            "corpus_endpoint": corpus.endpoint,
            "corpus_revision": corpus.revision,
            "corpus_fingerprint": fingerprint,
            "dataset": result.manifest.model_dump(mode="json"),
            "texts": {s.record.sample_id: s.text for s in result.samples},
        },
    )
    return result


def _benchmark_operation(
    *, journal: "StageJournal", args: Any, workload: WorkloadSpec,
    side: str, model_dir: Path,
) -> dict[str, Any]:
    """Measure either side through the same server lifecycle and resource checks."""
    stage = f"benchmark_{side}"
    attempt = journal.state["stages"][stage]["attempts"]
    log_path = args.run_dir / "logs" / f"benchmark-{side}-server-{attempt}.log"
    quantized = side != "bf16"
    version = _package_version("sglang")
    settled_mib, settled = wait_for_clean_gpu()
    with server_session(
        build_bf16_command(
            model_dir, args.port, args.mem_fraction_static,
            attention_backend=args.attention_backend,
            operator_backend=args.operator_backend,
            cuda_graph_max_bs=args.cuda_graph_max_bs,
            sglang_version=version,
            disable_cuda_graph=args.disable_cuda_graph,
        ),
        port=args.port, model_id=MODEL_ID, log_path=log_path,
    ) as session:
        dump, _ = benchmark(
            base_url=session["base_url"], workload=workload,
            model_id=MODEL_ID, output_dir=args.run_dir, side=side,
            sglang_version=version,
            quant_kernel=classify_quant_kernel(log_path, method=side) if quantized else None,
        )
        session.update(dump)
        if quantized:
            session["quant_kernel"] = classify_quant_kernel(log_path, method=side)
        session["serving_parameters"] = serving_parameters(args)
    session["gpu_settled_before_measure_mib"] = settled_mib
    session["gpu_settled"] = settled
    reasons = resource_blockers(session) + contention_blockers(session)
    session["resource_warnings"] = resource_warnings(session) + anomaly_warnings(session, side=side)
    if quantized and session.get("quant_kernel") is None:
        reasons.append("quantized SGLang kernel could not be confirmed")
    if session["gpu_release_confirmed"] is not True:
        reasons.append("GPU memory release could not be confirmed")
    if reasons:
        session.update(status="blocked", reason="; ".join(reasons))
    return session


def _quality_operation(
    *,
    journal: "StageJournal",
    args: Any,
    stage: str,
    side: str,
    checkpoint: Path | None,
    model_dir: Path,
    dataset: "PrepareResult",
    holdout: list,
    holdout_texts: list[str],
):
    """Build the stage operation closure for one quality side."""

    def operation() -> dict[str, Any]:
        attempt = journal.state["stages"][stage]["attempts"]
        log_path = args.run_dir / "logs" / f"quality-{side}-server-{attempt}.log"
        target = checkpoint if checkpoint is not None else model_dir
        # Settle before the server starts (see wait_for_clean_gpu): the KV pool
        # scales with the pre-existing baseline, so starting during teardown
        # would shrink it and change decode speed.
        settled_mib, settled = wait_for_clean_gpu()
        with server_session(
            build_bf16_command(
                target,
                args.port,
                args.mem_fraction_static,
                attention_backend=args.attention_backend,
                operator_backend=args.operator_backend,
                cuda_graph_max_bs=args.cuda_graph_max_bs,
                sglang_version=_package_version("sglang"),
                disable_cuda_graph=args.disable_cuda_graph,
            ),
            port=args.port,
            model_id=MODEL_ID,
            log_path=log_path,
        ) as session:
            summary, _ = score_documents(
                holdout,
                holdout_texts,
                base_url=session["base_url"],
                model_id=MODEL_ID,
                output_dir=args.run_dir,
                side=side,
                context_length=args.quality_context_length,
                dataset_fingerprint=_quality_dataset_fingerprint(args.run_dir),
            )
            session["summary"] = summary.model_dump(mode="json")
            session["records_path"] = summary.records_path
            session["dataset_fingerprint"] = _quality_dataset_fingerprint(args.run_dir)
            session["quant_kernel"] = (
                classify_quant_kernel(log_path, method=side)
                if checkpoint is not None
                else None
            )
        session["gpu_settled_before_measure_mib"] = settled_mib
        session["gpu_settled"] = settled
        resource_reasons = resource_blockers(session)
        resource_reasons.extend(contention_blockers(session))
        session["resource_warnings"] = resource_warnings(session)
        if session["gpu_release_confirmed"] is not True:
            resource_reasons.append("GPU memory release could not be confirmed")
        if checkpoint is not None and session.get("quant_kernel") is None:
            resource_reasons.append("quantized SGLang kernel could not be confirmed")
        if resource_reasons:
            session["status"] = "blocked"
            session["reason"] = "; ".join(resource_reasons)
        return session

    return operation


def _quality_dataset_fingerprint(run_dir: Path) -> str | None:
    manifest_path = run_dir / "data-manifest.json"
    if not manifest_path.is_file():
        return None
    return read_json(manifest_path).get("corpus_fingerprint")


def _attach_quality_to_report(
    run_dir: Path, workload_id: str | None, *, method: str = "gptq"
) -> None:
    """Attach quality evidence and save through the shared report pipeline."""
    report_path = run_dir / "regressions.json"
    if not report_path.is_file():
        return
    report = RegressionReport.model_validate(read_json(report_path))
    try:
        report = attach_quality(report, run_dir, method=method, required=True)
    except StoreError as exc:
        raise ProbeError(str(exc)) from exc
    save_report(report, run_dir)


def load_workload(config_path: Path | None) -> WorkloadSpec:
    """Load the fixed workload, defaulting to the built-in request set.

    The workload is part of the experiment fingerprint: a different request set
    or timing window produces a different fingerprint and cannot be compared.
    """
    import yaml

    requests = build_request_set()
    hashes = request_set_hashes(requests)
    if config_path is None:
        return WorkloadSpec(
            workload_id="mvp-short-chat",
            request_set_id="builtin-short-chat",
            request_hashes=hashes,
            warmup_requests=3,
            timed_requests=20,
            max_new_tokens=64,
        )
    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProbeError(f"cannot read workload config {config_path}: {exc}") from exc
    raw = (document or {}).get("workload")
    if not isinstance(raw, dict):
        raise ProbeError(f"{config_path} has no workload block")
    raw = dict(raw)
    # The request set is code-owned: a config cannot smuggle in different inputs.
    raw["request_hashes"] = hashes
    raw.setdefault("request_set_id", "builtin-short-chat")
    try:
        return WorkloadSpec(**raw)
    except Exception as exc:
        raise ProbeError(f"invalid workload in {config_path}: {exc}") from exc


def run_quantization_process(
    model_dir: Path,
    revision: str,
    checkpoint: Path,
    *,
    log_path: Path,
    timeout_seconds: float,
    method: str = "gptq",
) -> dict[str, Any]:
    """Run one quantization method and adopt only a complete, checksum-valid checkpoint.

    ``method`` is threaded through to the worker so GPTQ and AWQ share this
    orchestration instead of each getting a near-identical copy.
    """
    label = method.upper()
    if checkpoint.exists():
        manifest = read_quant_artifact(checkpoint, method=method)
        if manifest["parent_revision"] != revision:
            raise ProbeError(f"existing {label} checkpoint comes from a different revision")
        recorded = manifest.get("method")
        if recorded is not None and recorded != method:
            raise ProbeError(
                f"existing checkpoint at {checkpoint} was produced by {recorded!r}, "
                f"not {method!r}"
            )
        return {"checkpoint_path": str(checkpoint.resolve()), "artifact": manifest, "recovered": True}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        raise ProbeError(f"log file already exists: {log_path}")
    baseline_gpu_mib = _sample_gpu_used_mib()
    sampler = _GpuPeakSampler()
    command = build_quant_command(model_dir, revision, checkpoint, method=method)
    started = time.monotonic()
    worker_error: Exception | None = None
    returncode: int | None = None
    with log_path.open("xb") as log:
        with _tracked_process(command, log) as process:
            sampler.root_pid = process.pid
            sampler.start()
            try:
                try:
                    returncode = process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired as exc:
                    worker_error = ProbeError(f"{label} worker exceeded {timeout_seconds}s; see {log_path}")
            finally:
                _stop_process_group(process)
                sampler.stop()
    after_gpu_mib, release_confirmed = _wait_gpu_release(baseline_gpu_mib)
    if worker_error is not None:
        raise ProbeError(
            f"{worker_error}; release_confirmed={release_confirmed}; "
            f"gpu_used_after_mib={after_gpu_mib}"
        ) from worker_error
    if returncode != 0:
        raise ProbeError(
            f"{label} worker exited with code {returncode}; see {log_path}; "
            f"release_confirmed={release_confirmed}; gpu_used_after_mib={after_gpu_mib}"
        )
    manifest = read_quant_artifact(checkpoint, method=method)
    if manifest["parent_revision"] != revision:
        raise ProbeError(f"{label} output revision does not match input")
    recorded = manifest.get("method")
    if recorded is not None and recorded != method:
        raise ProbeError(
            f"{label} worker produced an artifact labelled {recorded!r}"
        )
    return {
        "checkpoint_path": str(checkpoint.resolve()),
        "artifact": manifest,
        "recovered": False,
        "method": method,
        "worker_command": command,
        "worker_log": str(log_path),
        "wall_seconds": round(time.monotonic() - started, 3),
        "gpu_used_before_mib": baseline_gpu_mib,
        "gpu_peak_sampled_mib": sampler.peak_mib,
        "peak_rss_bytes": sampler.peak_rss_bytes,
        "gpu_used_after_mib": after_gpu_mib,
        "gpu_release_confirmed": release_confirmed,
    }


class StageJournal:
    """Two-phase local record: result checksum first, succeeded state second.

    The journal file is ``run-status.json``. Runs created before the rename
    stored it as ``m0-status.json``; those are read for backward compatibility
    but any new write goes to the current name.
    """

    LEGACY_STATE_NAME = "m0-status.json"
    STATE_NAME = "run-status.json"

    def __init__(self, run_dir: Path, *, fingerprint: str) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / self.STATE_NAME
        legacy_path = self.run_dir / self.LEGACY_STATE_NAME
        if not self.state_path.is_file() and legacy_path.is_file():
            # Adopt the legacy journal under its new name so resume keeps working.
            self.state_path = legacy_path
        if self.state_path.is_file():
            try:
                self.state = read_json(self.state_path)
            except StoreError as exc:
                raise ProbeError(f"invalid run journal: {exc}") from exc
            if not isinstance(self.state, dict) or not isinstance(self.state.get("stages"), dict):
                raise ProbeError("invalid run journal structure")
            if self.state.get("fingerprint") != fingerprint:
                raise ProbeError("run fingerprint changed; choose a new run directory")
        else:
            self.state = {"fingerprint": fingerprint, "stages": {}}
            atomic_write_json(self.run_dir / self.STATE_NAME, self.state)

    def run(
        self,
        stage: str,
        operation: Callable[[], dict[str, Any]],
        *,
        verify_reuse: Callable[[dict[str, Any]], bool] | None = None,
    ) -> dict[str, Any]:
        path = self.run_dir / f"{stage}.json"
        previous = self.state["stages"].get(stage) or {}
        attempts = previous.get("attempts", 0)
        if (
            previous.get("status") == "succeeded"
            and path.is_file()
            and previous.get("sha256") == file_sha256(path)
        ):
            cached = read_json(path)
            try:
                reusable = verify_reuse is None or verify_reuse(cached)
            except Exception as exc:
                self.state["stages"][stage] = {
                    "status": "invalid", "attempts": attempts, "reason": str(exc)
                }
                atomic_write_json(self.state_path, self.state)
                raise
            if reusable:
                return cached
        attempts += 1
        self.state["stages"][stage] = {"status": "running", "attempts": attempts}
        atomic_write_json(self.state_path, self.state)
        try:
            result = operation()
            atomic_write_json(path, result)
        except Exception as exc:
            self.state["stages"][stage] = {
                "status": "failed", "attempts": attempts, "reason": str(exc)
            }
            atomic_write_json(self.state_path, self.state)
            raise
        self.state["stages"][stage] = {
            "status": "blocked" if result.get("status") == "blocked" else "succeeded",
            "attempts": attempts,
            "sha256": file_sha256(path),
        }
        atomic_write_json(self.state_path, self.state)
        return result

    def require_succeeded(
        self,
        stage: str,
        *,
        verify_result: Callable[[dict[str, Any]], bool] | None = None,
    ) -> dict[str, Any]:
        """A downstream GPU stage may consume only a checksum-valid success."""
        record = self.state["stages"].get(stage) or {}
        path = self.run_dir / f"{stage}.json"
        if (
            record.get("status") != "succeeded"
            or not path.is_file()
            or record.get("sha256") != file_sha256(path)
        ):
            raise ProbeError(f"{stage.upper()} stage is not verified as succeeded")
        result = read_json(path)
        if verify_result is not None and not verify_result(result):
            self.state["stages"][stage] = {
                "status": "invalid",
                "attempts": record.get("attempts", 0),
                "reason": "required evidence failed checksum validation",
            }
            atomic_write_json(self.state_path, self.state)
            raise ProbeError(f"{stage.upper()} evidence failed checksum validation")
        return result


def _fingerprint(
    model_dir: Path,
    revision: str,
    port: int,
    mem_fraction: float,
    source_hashes: dict[str, str] | None = None,
    attention_backend: str = DEFAULT_ATTENTION_BACKEND,
    cuda_graph_max_bs: int = DEFAULT_CUDA_GRAPH_MAX_BS,
    quant_method: str = "gptq",
    operator_backend: str = DEFAULT_OPERATOR_BACKEND,
    engine_identity: dict[str, Any] | None = None,
) -> str:
    """Bind a run to model content, runtime sources, dependencies and launch settings."""
    resolved = model_dir.resolve()
    hashes = source_file_hashes(resolved) if source_hashes is None else dict(source_hashes)
    payload = {
        "model_id": MODEL_ID,
        "model_dir": str(resolved),
        "revision": revision,
        "port": port,
        "mem_fraction_static": mem_fraction,
        "attention_backend": attention_backend,
        "operator_backend": operator_backend,
        "engine_identity": engine_identity,
        "cuda_graph_max_bs": cuda_graph_max_bs,
        "quant_method": quant_method,
        "torch": _package_version("torch"),
        "sglang": _package_version("sglang"),
        "llmcompressor": _package_version("llmcompressor"),
        "compressed_tensors": _package_version("compressed-tensors"),
        "transformers": _package_version("transformers"),
        "datasets": _package_version("datasets"),
        "psutil": _package_version("psutil"),
        "python": platform.python_version(),
        "controller_sha256": file_sha256(Path(__file__)),
        "engine_selector_sha256": file_sha256(Path(__file__).with_name("engine.py")),
        "runtime_sha256": file_sha256(Path(__file__).with_name("runtime.py")),
        "serving_evaluator_sha256": file_sha256(Path(__file__).parent / "serving/evaluator.py"),
        "quant_worker_sha256": file_sha256(Path(__file__).with_name("quantize_worker.py")),
        "source_files": hashes,
        "source_files_digest": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "weight_index_sha256": _index_weight_shard(resolved),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Qwen3-0.6B BF16/GPTQ/AWQ SGLang evaluation (WSL2, 4 GB GPU)"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True, help="immutable 40-character model commit SHA")
    parser.add_argument(
        "--stage",
        choices=(
            "preflight",
            "bf16",
            *(name for m in SUPPORTED_QUANT_METHODS
              for name in (m, f"{m}_service", f"benchmark_{m}", f"quality_{m}")),
            "benchmark_bf16",
            "quality_bf16",
            "all",
            "full",
        ),
        default="all",
        help=(
            "'all' = capability gates only (bf16 + the selected method's "
            "quantize/service); 'full' = the whole pipeline: gates + serving "
            "benchmarks + quality + report, in one invocation. Quantized stages "
            "are named after --quant-method (e.g. 'awq', 'benchmark_awq')"
        ),
    )
    parser.add_argument(
        "--quality-documents",
        type=int,
        default=64,
        help=(
            "held-out documents scored for perplexity (full-prd.md §4 layer 2). "
            "Kept modest because each document costs a prefill per window."
        ),
    )
    # ------------------------------------------------------------------
    # Quality corpus: user-supplied by design. The purpose of this tool is
    # comparing quantization options on the data a user actually cares about,
    # so the corpus is configuration, not a built-in constant. Every field
    # below lands in the run fingerprint, so changing any of them produces a
    # new run rather than silently re-using older measurements.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--corpus",
        default=DEFAULT_QUALITY_CORPUS,
        help=(
            "HuggingFace dataset id to measure quality on "
            f"(default {DEFAULT_QUALITY_CORPUS!r}); pass your own domain data here"
        ),
    )
    parser.add_argument(
        "--corpus-config",
        default=None,
        help="dataset configuration/subset name (required by some datasets)",
    )
    parser.add_argument(
        "--corpus-split",
        default=None,
        help="dataset split to draw holdout text from (default 'test')",
    )
    parser.add_argument(
        "--corpus-text-column",
        default=None,
        help=(
            "column holding the document text (default 'text'); if it is wrong the "
            "error lists the columns the dataset actually has"
        ),
    )
    parser.add_argument(
        "--corpus-revision",
        default=None,
        help=(
            "optional dataset commit/tag to pin; unset means the corpus can move "
            "underneath the run (the fingerprint still detects a change)"
        ),
    )
    parser.add_argument(
        "--corpus-endpoint",
        default=None,
        help="dataset mirror endpoint (default the configured HF mirror)",
    )
    parser.add_argument(
        "--corpus-min-chars",
        type=int,
        default=400,
        help="skip documents shorter than this many characters when selecting text",
    )
    parser.add_argument(
        "--quality-calibration-samples",
        type=int,
        default=4,
        help="corpus documents reserved for the calibration split (kept disjoint)",
    )
    parser.add_argument(
        "--quality-max-tokens",
        type=int,
        default=512,
        help="token budget used when slicing holdout documents",
    )
    parser.add_argument(
        "--quality-context-length",
        type=int,
        default=512,
        help=(
            "server context length during scoring; documents are windowed to fit "
            "it, and a rejected window is recorded rather than imputed"
        ),
    )
    parser.add_argument(
        "--workload-config",
        type=Path,
        default=None,
        help=(
            "experiment YAML whose workload block defines the measured load; "
            "the request set itself always comes from code so both sides share inputs"
        ),
    )
    parser.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        help=(
            "disable CUDA graphs. Needed on SGLang 0.5.20 where graph capture "
            "hung on the quantized model; on 0.5.3 + marlin capture works, so "
            "both sides can (and must) run with graphs enabled"
        ),
    )
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--mem-fraction-static", type=float, default=0.8)
    parser.add_argument(
        "--attention-backend",
        choices=KNOWN_ATTENTION_BACKENDS,
        default=DEFAULT_ATTENTION_BACKEND,
        help=(
            "SGLang attention backend; triton avoids the flashinfer JIT path that "
            "needs a newer nvcc than this machine provides"
        ),
    )
    parser.add_argument(
        "--operator-backend",
        choices=KNOWN_OPERATOR_BACKENDS,
        default=DEFAULT_OPERATOR_BACKEND,
        help=(
            "RMSNorm backend: SGLang's built-in kernels, Kernscope PyTorch, or "
            "Kernscope Triton (requires the Kernscope package in this environment)"
        ),
    )
    parser.add_argument(
        "--cuda-graph-max-bs",
        type=int,
        default=DEFAULT_CUDA_GRAPH_MAX_BS,
        help=(
            "largest batch size captured by the prefill CUDA graph; smaller "
            "values save VRAM that a 4 GB card needs for KV cache"
        ),
    )
    parser.add_argument("--quant-timeout-minutes", type=int, default=90)
    parser.add_argument(
        "--quant-method",
        choices=SUPPORTED_QUANT_METHODS,
        default="gptq",
        help=(
            "quantization algorithm. Both emit the same compressed-tensors "
            "symmetric W4A16 shape and therefore serve through the same kernel, "
            "so a comparison between them isolates the algorithm rather than the "
            "serving path. Comparing two methods means two run directories: "
            "report and evidence names stay stable, and each run keeps one "
            "unambiguous serving configuration"
        ),
    )
    parser.add_argument("--engine-source", type=Path,
                        help="SGLang fork Python directory; dependencies stay in the execution environment")
    args = parser.parse_args(argv)
    try:
        from quantassay.engine import configure_engine_source
        try:
            args.engine_identity = configure_engine_source(args.engine_source)
        except ValueError as exc:
            raise ProbeError(str(exc)) from exc
        if not _REVISION_RE.fullmatch(args.revision):
            raise ProbeError("revision must be a 40-character commit SHA")
        if args.quant_timeout_minutes <= 0:
            raise ProbeError("quant-timeout-minutes must be positive")
        if args.operator_backend != "sglang":
            if _package_version("sglang") != "0.5.3":
                raise ProbeError("QuantAssay's Kernscope adapter requires SGLang 0.5.3")
            import_command = [
                sys.executable,
                "-c",
                "import quantassay.integrations.sglang.v0_5_3",
            ]
            if args.operator_backend == "triton":
                import_command[2] += "; import triton"
            adapter_check = _cpu_check(import_command, timeout=15)
            if not adapter_check["ok"]:
                raise ProbeError("Kernscope adapter is unavailable: " + adapter_check["output"])
        journal = StageJournal(
            args.run_dir,
            fingerprint=_fingerprint(
                args.model_dir,
                args.revision,
                args.port,
                args.mem_fraction_static,
                attention_backend=args.attention_backend,
                operator_backend=args.operator_backend,
                cuda_graph_max_bs=args.cuda_graph_max_bs,
                quant_method=args.quant_method,
                engine_identity=args.engine_identity,
            ),
        )
        # Bound once, used by every stage condition below so the method is never
        # re-derived (and cannot drift between the gate, benchmark and quality
        # branches).
        quant_method = args.quant_method
        def preflight_operation() -> dict[str, Any]:
            report = collect_preflight(args.model_dir, args.revision)
            blockers = validate_preflight(report)
            warnings = preflight_warnings(report)
            return {
                **report,
                "status": "blocked" if blockers else "succeeded",
                "reasons": blockers,
                "warnings": warnings,
            }

        report = journal.run("preflight", preflight_operation, verify_reuse=lambda _: False)
        for warning in report.get("warnings") or []:
            # Non-blocking, but never silent.
            print(json.dumps({"preflight_warning": warning}, ensure_ascii=False))
        blockers = report["reasons"]
        if blockers:
            raise ProbeError("preflight blocked: " + "; ".join(blockers))
        if args.stage in ("bf16", "all", "full"):
            def bf16_operation() -> dict[str, Any]:
                attempt = journal.state["stages"]["bf16"]["attempts"]
                result = run_server_smoke(
                    build_bf16_command(
                        args.model_dir,
                        args.port,
                        args.mem_fraction_static,
                        attention_backend=args.attention_backend,
                        operator_backend=args.operator_backend,
                        cuda_graph_max_bs=args.cuda_graph_max_bs,
                        sglang_version=_package_version("sglang"),
                        disable_cuda_graph=args.disable_cuda_graph,
                    ),
                    port=args.port,
                    model_id=MODEL_ID,
                    log_path=args.run_dir / "logs" / f"bf16-server-{attempt}.log",
                )
                resource_reasons = resource_blockers(result)
                if result["gpu_release_confirmed"] is not True:
                    resource_reasons.append("GPU memory release could not be confirmed after BF16 shutdown")
                if resource_reasons:
                    result["status"] = "blocked"
                    result["reason"] = "; ".join(resource_reasons)
                return result

            result = journal.run(
                "bf16",
                bf16_operation,
                verify_reuse=verify_service_log,
            )
            if result.get("status") == "blocked":
                raise ProbeError(result["reason"])
        # ------------------------------------------------------------------
        # Quantization gates, one per selected method.
        #
        # Written as a loop over methods rather than a block per method: the
        # quantize + serve-and-confirm-kernel logic is identical for every
        # method, and only the method name and checkpoint path differ. A copy
        # per method would drift (this repo already had to clean up parallel
        # implementations once).
        #
        # BF16 is measured once and shared: both quantized sides are compared
        # against the same baseline, which is what makes their numbers
        # comparable to each other as well as to the baseline.
        # ------------------------------------------------------------------
        gate_methods = [
            m
            for m in selected_quant_methods(args)
            if args.stage in (m, f"{m}_service", "all", "full")
        ]
        if gate_methods:
            bf16 = journal.require_succeeded("bf16", verify_result=verify_service_log)
            if bf16.get("gpu_release_confirmed") is not True:
                raise ProbeError("BF16 GPU memory release was not verified")
        for method in gate_methods:
            stages = method_stage_names(method)
            label = method.upper()
            checkpoint = quant_checkpoint_dir(args.run_dir, method)
            if args.stage in (method, "all", "full"):
                def quantize_operation(
                    method: str = method, checkpoint: Path = checkpoint, label: str = label
                ) -> dict[str, Any]:
                    stage = method_stage_names(method)["quantize"]
                    attempt = journal.state["stages"][stage]["attempts"]
                    result = run_quantization_process(
                        args.model_dir,
                        args.revision,
                        checkpoint,
                        log_path=args.run_dir / "logs" / f"{method}-worker-{attempt}.log",
                        timeout_seconds=args.quant_timeout_minutes * 60,
                        method=method,
                    )
                    if result["recovered"]:
                        result["status"] = "blocked"
                        result["reason"] = (
                            f"recovered {label} checkpoint lacks verified GPU/RSS peak and "
                            "release evidence; the previous attempt cannot be promoted to success"
                        )
                        return result
                    resource_reasons = resource_blockers(result)
                    if result.get("gpu_release_confirmed") is not True:
                        resource_reasons.append(
                            f"GPU memory release could not be confirmed after {label}"
                        )
                    if resource_reasons:
                        result["status"] = "blocked"
                        result["reason"] = "; ".join(resource_reasons)
                    return result

                quant_result = journal.run(
                    stages["quantize"],
                    quantize_operation,
                    verify_reuse=lambda cached, _rev=args.revision, _m=method: (
                        read_quant_artifact(Path(cached["checkpoint_path"]), method=_m)
                        == cached.get("artifact")
                        and cached["artifact"]["parent_revision"] == _rev
                    ),
                )
                if quant_result.get("status") == "blocked":
                    raise ProbeError(quant_result["reason"])
            else:
                quant_result = journal.require_succeeded(stages["quantize"])
            if (
                read_quant_artifact(Path(quant_result["checkpoint_path"]), method=method)[
                    "parent_revision"
                ]
                != args.revision
            ):
                raise ProbeError(f"{label} checkpoint revision mismatch")
            if args.stage in (stages["service"], "all", "full"):
                def quant_service_operation(
                    method: str = method, checkpoint: Path = checkpoint, label: str = label
                ) -> dict[str, Any]:
                    stage = method_stage_names(method)["service"]
                    attempt = journal.state["stages"][stage]["attempts"]
                    log_path = args.run_dir / "logs" / f"{method}-server-{attempt}.log"
                    result = run_server_smoke(
                        build_bf16_command(
                            checkpoint,
                            args.port,
                            args.mem_fraction_static,
                            attention_backend=args.attention_backend,
                            operator_backend=args.operator_backend,
                            cuda_graph_max_bs=args.cuda_graph_max_bs,
                            sglang_version=_package_version("sglang"),
                            disable_cuda_graph=args.disable_cuda_graph,
                        ),
                        port=args.port,
                        model_id=MODEL_ID,
                        log_path=log_path,
                    )
                    result["checkpoint_path"] = str(checkpoint.resolve())
                    result["method"] = method
                    result["checkpoint_manifest_sha256"] = file_sha256(
                        checkpoint / "artifact-manifest.json"
                    )
                    result["quant_kernel"] = classify_quant_kernel(log_path, method=method)
                    resource_reasons = resource_blockers(result)
                    if result["quant_kernel"] is None:
                        resource_reasons.append(
                            f"{label} SGLang kernel could not be confirmed"
                        )
                    if result["gpu_release_confirmed"] is not True:
                        resource_reasons.append("GPU memory release could not be confirmed")
                    if resource_reasons:
                        result["status"] = "blocked"
                        result["reason"] = "; ".join(resource_reasons)
                    return result

                service_result = journal.run(
                    stages["service"],
                    quant_service_operation,
                    verify_reuse=verify_quant_service_evidence,
                )
                if service_result.get("status") == "blocked":
                    raise ProbeError(service_result["reason"])
        # `all` runs capability gates; `full` also measures serving and quality.
        if args.stage in ("benchmark_bf16", f"benchmark_{quant_method}", "full"):
            workload = load_workload(args.workload_config)
            sides = ("bf16", quant_method) if args.stage == "full" else (
                "bf16" if args.stage == "benchmark_bf16" else quant_method,
            )
            for side in sides:
                if side == "bf16":
                    evidence = journal.require_succeeded("bf16", verify_result=verify_service_log)
                    if evidence.get("gpu_release_confirmed") is not True:
                        raise ProbeError("BF16 GPU memory release was not verified")
                    model_dir = args.model_dir
                else:
                    stages = method_stage_names(side)
                    for required in (stages["quantize"], stages["service"]):
                        journal.require_succeeded(required)
                    journal.require_succeeded(
                        "benchmark_bf16", verify_result=verify_benchmark_evidence
                    )
                    model_dir = quant_checkpoint_dir(args.run_dir, side)
                result = journal.run(
                    f"benchmark_{side}",
                    lambda: _benchmark_operation(
                        journal=journal, args=args, workload=workload,
                        side=side, model_dir=model_dir,
                    ),
                    verify_reuse=verify_benchmark_evidence,
                )
                if result.get("status") == "blocked":
                    raise ProbeError(result["reason"])
            if quant_method in sides:
                baseline = journal.require_succeeded("benchmark_bf16")
                candidate = journal.require_succeeded(f"benchmark_{quant_method}")
                report = build_comparison_report(baseline, candidate, run_id=args.run_dir.name)
                report = attach_quality(report, args.run_dir, method=quant_method)
                saved = save_report(report, args.run_dir)
                print(json.dumps({"report": saved["report"], "comparable": report.comparable},
                                 ensure_ascii=False))
        # Quality can also run independently of the serving benchmark invocation.
        if args.stage in ("quality_bf16", f"quality_{quant_method}", "full"):
            # Bound here as well as in the benchmark branch: the quality stage
            # must not depend on a benchmark having run in the same invocation
            # (an UnboundLocalError was how that coupling first surfaced).
            workload = load_workload(args.workload_config)
            corpus_spec = corpus_spec_from_args(args)
            print(
                json.dumps(
                    {"quality_corpus": corpus_spec.identity,
                     "documents": args.quality_documents,
                     "text_column": corpus_spec.text_column},
                    ensure_ascii=False,
                )
            )
            dataset = prepare_quality_data(
                args.run_dir,
                spec=corpus_spec,
                calibration_samples=args.quality_calibration_samples,
                documents=args.quality_documents,
                max_tokens=args.quality_max_tokens,
            )
            holdout = holdout_records(dataset.manifest, DataSplit.DEV)
            texts_by_id = {s.record.sample_id: s.text for s in dataset.samples}
            holdout_texts = [texts_by_id[r.sample_id] for r in holdout]
            if not holdout:
                raise ProbeError("quality holdout is empty; cannot evaluate quality")

            if args.stage in ("quality_bf16", "full"):
                quality_bf16_operation = _quality_operation(
                    journal=journal,
                    args=args,
                    stage="quality_bf16",
                    side="bf16",
                    checkpoint=None,
                    model_dir=args.model_dir,
                    dataset=dataset,
                    holdout=holdout,
                    holdout_texts=holdout_texts,
                )
                result = journal.run(
                    "quality_bf16", quality_bf16_operation, verify_reuse=verify_quality_evidence
                )
                if result.get("status") == "blocked":
                    raise ProbeError(result["reason"])
            if args.stage in (f"quality_{quant_method}", "full"):
                method_stages = method_stage_names(quant_method)
                for required in (method_stages["quantize"], method_stages["service"]):
                    journal.require_succeeded(required)
                journal.require_succeeded("quality_bf16", verify_result=verify_quality_evidence)
                checkpoint = quant_checkpoint_dir(args.run_dir, quant_method)
                quality_gptq_operation = _quality_operation(
                    journal=journal,
                    args=args,
                    stage=f"quality_{quant_method}",
                    side=quant_method,
                    checkpoint=checkpoint,
                    model_dir=args.model_dir,
                    dataset=dataset,
                    holdout=holdout,
                    holdout_texts=holdout_texts,
                )
                result = journal.run(
                    f"quality_{quant_method}",
                    quality_gptq_operation,
                    verify_reuse=verify_quality_evidence,
                )
                if result.get("status") == "blocked":
                    raise ProbeError(result["reason"])
                # Fold quality into the existing report so one artifact carries
                # both the serving and the quality verdict.
                _attach_quality_to_report(
                    args.run_dir, workload.workload_id, method=quant_method
                )
        print(json.dumps({"status": "succeeded", "run_dir": str(args.run_dir)}, ensure_ascii=False))
        return 0
    except (ProbeError, OSError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
