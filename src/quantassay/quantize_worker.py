"""Isolated W4A16 quantization worker (GPTQ and AWQ) for the fixed Qwen3-0.6B snapshot.

This is an offline quantizer, never an evaluator. Heavy imports happen only in
the worker process so the gating controller can inspect and resume without a GPU.

Both methods produce the *same* artifact shape: compressed-tensors,
pack-quantized, 4-bit, group_size=128, **symmetric** weights with unquantized
activations. That is deliberate:

* serving runs through the one proven path (``CompressedTensorsWNA16`` → the
  marlin kernel), whose dispatch requires symmetric weights;
* the two methods differ only in *how weights are chosen*, so a serving
  comparison between them is not confounded by a scheme difference.

AWQ's published example recipe uses asymmetric weights, which sglang 0.5.3
cannot serve here (see ``compatibility.md``). Symmetric AWQ is therefore a
documented trade-off, not the upstream default.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from quantassay.experiments.store import atomic_write_json, file_sha256
from quantassay.contracts import MODEL_ID
from quantassay.runtime import verify_snapshot_layout

CALIBRATION_PROMPTS = (
    "Explain why a sorted list permits binary search in one sentence.",
    "Write a short description of a quiet library at night.",
    "Translate 'the sky is clear' into Chinese and explain the tone.",
    "List two reasons to validate a software checkpoint before deployment.",
)
CALIBRATION_MAX_TOKENS = 128

#: Supported quantization algorithms. Each maps to a one-shot modifier.
QUANT_METHODS = ("gptq", "awq")

RECIPES: dict[str, dict[str, Any]] = {
    "gptq": {
        "algorithm": "GPTQ",
        "scheme": "W4A16",
        "group_size": 128,
        "targets": ["Linear"],
        "ignore": ["lm_head"],
    },
    "awq": {
        "algorithm": "AWQ",
        "scheme": "W4A16",
        "group_size": 128,
        "targets": ["Linear"],
        "ignore": ["lm_head"],
        # Recorded because it deviates from the upstream example (which uses
        # symmetric: false). Raising the symmetry choice into the recipe keeps
        # the artifact self-describing.
        "symmetric": True,
        "note": (
            "symmetric weights are required by the sglang CompressedTensorsWNA16 "
            "path; upstream AWQ examples use asymmetric weights"
        ),
        "duo_scaling": True,
        "n_grid": 20,
    },
}

#: AWQ smooths activations at specific layer norms before quantizing. The
#: mapping decides *where* that happens, so a wrong mapping reduces AWQ to
#: plain quantization while still reporting success.
#:
#: The AWQModifier docstring's example uses OPT-style names
#: (``self_attn_layer_norm`` / ``final_layer_norm``). On Qwen3 those match
#: **nothing** (verified against the real weight map: 0 hits), so copying the
#: example verbatim would silently perform no smoothing at all. These are the
#: Qwen3 names, which match 28 smooth layers and 84/56 balance layers.
QWEN3_AWQ_MAPPINGS: tuple[dict[str, Any], ...] = (
    {
        "smooth_layer": "re:.*input_layernorm",
        "balance_layers": ["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
    },
    {
        "smooth_layer": "re:.*post_attention_layernorm",
        "balance_layers": ["re:.*gate_proj", "re:.*up_proj"],
    },
)


class QuantizeError(RuntimeError):
    """The worker could not produce a verified, packed checkpoint."""


def validate_w4a16_scheme(scheme: Any, *, require_symmetric: bool = True) -> None:
    """Reject a scheme that this machine cannot serve.

    Symmetry is not cosmetic: sglang's ``CompressedTensorsWNA16`` dispatch is

        is_channel_group and input_quant is None and is_symmetric and is_static

    so an asymmetric checkpoint does not reach the marlin kernel at all. Checking
    it here fails fast in the quantizer instead of producing a checkpoint that
    only breaks later, at serving time.
    """
    weights = getattr(scheme, "weights", None)
    weight_type = getattr(getattr(weights, "type", None), "value", getattr(weights, "type", None))
    if getattr(weights, "num_bits", None) != 4:
        raise QuantizeError("W4A16 preset must have 4-bit weights")
    if getattr(weights, "group_size", None) != 128:
        raise QuantizeError("W4A16 preset must use group_size=128")
    if str(weight_type).lower() != "int" or getattr(scheme, "input_activations", None) is not None:
        raise QuantizeError("W4A16 preset must use integer weights and unquantized activations")
    if require_symmetric and getattr(weights, "symmetric", None) is not True:
        raise QuantizeError(
            "W4A16 weights must be symmetric: sglang's CompressedTensorsWNA16 path "
            "(the marlin kernel) requires it, so an asymmetric checkpoint would not "
            "be servable on this stack"
        )


def build_symmetric_w4a16_scheme() -> Any:
    """The one scheme both methods use, built explicitly rather than by preset.

    ``preset_name_to_scheme("W4A16", ...)`` happens to be symmetric already, but
    AWQ's own examples are asymmetric, so the symmetric variant is constructed
    field-by-field and asserted rather than assumed.
    """
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            type="int",
            symmetric=True,
            strategy="group",
            group_size=128,
            dynamic=False,
        ),
    )
    validate_w4a16_scheme(scheme)
    return scheme


def import_modifier(method: str) -> Any:
    """Import the one-shot modifier for ``method``.

    Both are imported by full path so a renamed/removed class fails loudly here
    rather than during a long GPU run. GPTQ's location moved between
    llmcompressor versions, so that one keeps a fallback.
    """
    if method == "awq":
        from llmcompressor.modifiers.awq import AWQModifier  # type: ignore

        return AWQModifier
    if method == "gptq":
        try:
            from llmcompressor.modifiers.gptq import GPTQModifier  # type: ignore

            return GPTQModifier
        except ImportError:
            from llmcompressor.modifiers.quantization import GPTQModifier  # type: ignore

            return GPTQModifier
    raise QuantizeError(
        f"unknown quantization method {method!r}; supported: {list(QUANT_METHODS)}"
    )


def build_recipe(method: str, scheme: Any) -> Any:
    """Construct the modifier for ``method`` around the shared scheme."""
    modifier_cls = import_modifier(method)
    if method == "awq":
        from llmcompressor.modifiers.awq.mappings import AWQMapping  # type: ignore

        mappings = [
            AWQMapping(smooth_layer=m["smooth_layer"], balance_layers=list(m["balance_layers"]))
            for m in QWEN3_AWQ_MAPPINGS
        ]
        return modifier_cls(
            config_groups={"group_0": scheme},
            mappings=mappings,
            ignore=list(RECIPES["awq"]["ignore"]),
            duo_scaling=RECIPES["awq"]["duo_scaling"],
            n_grid=RECIPES["awq"]["n_grid"],
        )
    return modifier_cls(config_groups={"group_0": scheme}, ignore=list(RECIPES[method]["ignore"]))


def render_calibration_texts(tokenizer: Any) -> list[str]:
    """Apply the same explicit non-thinking Qwen3 template to every sample."""
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in CALIBRATION_PROMPTS
    ]
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise QuantizeError("calibration template produced an empty or non-text sample")
    return texts


def build_calibration_records(
    tokenizer: Any, texts: list[str] | None = None
) -> list[dict[str, list[int]]]:
    """Tokenize templated prompts once, without adding a second BOS/template."""
    records: list[dict[str, list[int]]] = []
    for text in texts if texts is not None else render_calibration_texts(tokenizer):
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=CALIBRATION_MAX_TOKENS,
            padding=False,
        )
        ids = encoded["input_ids"]
        mask = encoded["attention_mask"]
        if not ids or len(ids) != len(mask):
            raise QuantizeError("invalid tokenized calibration sample")
        records.append({"input_ids": list(ids), "attention_mask": list(mask)})
    return records


def inspect_packed_checkpoint(checkpoint: Path) -> dict[str, Any]:
    """Reject an uncompressed or differently quantized output before serving."""
    try:
        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise QuantizeError(f"invalid saved config.json: {exc}") from exc
    if config.get("model_type") != "qwen3":
        raise QuantizeError("saved model_type must be qwen3")
    quant = config.get("quantization_config") or {}
    if str(quant.get("quant_method", "")).replace("_", "-") != "compressed-tensors":
        raise QuantizeError("saved checkpoint is not compressed-tensors")
    format_name = str(quant.get("format", "")).replace("_", "-")
    if format_name != "pack-quantized":
        raise QuantizeError("saved checkpoint is not pack-quantized")
    groups = quant.get("config_groups") or {}
    if not isinstance(groups, dict) or not groups:
        raise QuantizeError("saved checkpoint has no quantization config groups")
    for name, group in groups.items():
        if not isinstance(group, dict):
            raise QuantizeError(f"invalid quantization group {name}")
        weights = group.get("weights") or {}
        if weights.get("num_bits") != 4:
            raise QuantizeError(f"group {name} is not 4-bit")
        if weights.get("group_size") != 128:
            raise QuantizeError(f"group {name} does not use group_size=128")
        if weights.get("type") != "int" or group.get("input_activations") is not None:
            raise QuantizeError(f"group {name} is not W4A16 integer weight-only")
    safetensors = sorted(checkpoint.glob("*.safetensors"))
    if not safetensors or any(path.stat().st_size == 0 for path in safetensors):
        raise QuantizeError("saved checkpoint has no nonempty safetensors weights")
    if not (checkpoint / "tokenizer.json").is_file():
        raise QuantizeError("saved checkpoint has no tokenizer.json")
    return {
        "quant_method": "compressed-tensors",
        "format": format_name,
        "num_bits": 4,
        "group_size": 128,
        "safetensors_files": [path.name for path in safetensors],
    }


def _calibration_hashes(texts: list[str]) -> list[str]:
    return [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]


def _runtime_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("torch", "transformers", "llmcompressor", "compressed-tensors", "datasets"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def quantize_checkpoint(
    model_dir: Path,
    revision: str,
    output_dir: Path,
    *,
    method: str = "gptq",
) -> dict[str, Any]:
    if method not in QUANT_METHODS:
        raise QuantizeError(
            f"unknown quantization method {method!r}; supported: {list(QUANT_METHODS)}"
        )
    if not verify_snapshot_layout(model_dir, revision):
        raise QuantizeError("input must be the Qwen3-0.6B cache snapshot at the claimed revision")
    if output_dir.exists():
        raise QuantizeError(f"output path already exists: {output_dir}")
    # Heavy imports stay inside this process. The sequential pipeline onloads
    # decoder layers to GPU while most BF16 weights remain on CPU.
    import torch
    from datasets import Dataset
    from llmcompressor import oneshot
    from transformers import AutoModelForCausalLM, AutoTokenizer

    recipe_meta = RECIPES[method]

    # Local WSL2 has no pinned torch; the gate only requires CUDA to be usable.
    # The execution layer must report CUDA availability; the actual version is
    # recorded in the artifact manifest rather than enforced here.
    if not torch.cuda.is_available():
        raise QuantizeError(f"CUDA is not available in the execution layer; refusing {method.upper()}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False
    )
    texts = render_calibration_texts(tokenizer)
    dataset = Dataset.from_list(build_calibration_records(tokenizer, texts))
    torch.set_grad_enabled(False)
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=torch.bfloat16,
        device_map="cpu",
        local_files_only=True,
        trust_remote_code=False,
    )
    input_dtypes: dict[str, int] = {}
    for parameter in model.parameters():
        if parameter.is_floating_point():
            dtype = str(parameter.dtype)
            input_dtypes[dtype] = input_dtypes.get(dtype, 0) + parameter.numel()
    if not input_dtypes or set(input_dtypes) != {"torch.bfloat16"}:
        raise QuantizeError(f"input floating weights are not uniformly BF16: {input_dtypes}")
    torch.cuda.reset_peak_memory_stats()
    scheme = build_symmetric_w4a16_scheme()
    recipe = build_recipe(method, scheme)
    oneshot_kwargs: dict[str, Any] = dict(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=CALIBRATION_MAX_TOKENS,
        num_calibration_samples=len(texts),
        batch_size=1,
        shuffle_calibration_samples=False,
    )
    # `pipeline` (sequential onloading) is 0.13+ only; 0.9.x runs sequentially
    # on CPU without it. Both paths keep weights on CPU and calibration short.
    import inspect

    if "pipeline" in inspect.signature(oneshot).parameters:
        oneshot_kwargs["pipeline"] = "sequential"
    oneshot(**oneshot_kwargs)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{method}-incomplete-", dir=output_dir.parent)
    )
    model.save_pretrained(str(temporary), save_compressed=True, safe_serialization=True)
    tokenizer.save_pretrained(str(temporary))
    format_info = inspect_packed_checkpoint(temporary)
    recipe_record = {
        **recipe_meta,
        "method": method,
        "calibration_max_tokens": CALIBRATION_MAX_TOKENS,
        "calibration_hashes": _calibration_hashes(texts),
        "parent_model_id": MODEL_ID,
        "parent_revision": revision,
        "input_weight_dtypes": input_dtypes,
        "resolved_scheme": scheme.model_dump(mode="json"),
    }
    if method == "awq":
        recipe_record["awq_mappings"] = [dict(m) for m in QWEN3_AWQ_MAPPINGS]
    atomic_write_json(temporary / "recipe.json", recipe_record)
    files = {
        str(path.relative_to(temporary)).replace("\\", "/"): file_sha256(path)
        for path in sorted(temporary.rglob("*"))
        if path.is_file() and path.name != "artifact-manifest.json"
    }
    try:
        import resource

        worker_peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except ImportError:
        worker_peak_rss_kib = None
    manifest = {
        "model_id": MODEL_ID,
        "parent_revision": revision,
        # The method is recorded explicitly: two artifacts of the same shape can
        # come from different algorithms, and a report must never guess which.
        "method": method,
        "recipe": recipe_meta,
        "calibration_hashes": _calibration_hashes(texts),
        "calibration_max_tokens": CALIBRATION_MAX_TOKENS,
        "input_weight_dtypes": input_dtypes,
        "resolved_scheme": scheme.model_dump(mode="json"),
        "format": format_info,
        "versions": _runtime_versions(),
        "quantize_seconds": round(time.monotonic() - started, 3),
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "worker_peak_rss_kib": worker_peak_rss_kib,
        "files": files,
    }
    if method == "awq":
        manifest["awq_mappings"] = [dict(m) for m in QWEN3_AWQ_MAPPINGS]
    atomic_write_json(temporary / "artifact-manifest.json", manifest)
    os.replace(temporary, output_dir)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Isolated Qwen3-0.6B W4A16 quantization worker (GPTQ / AWQ)"
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=QUANT_METHODS,
        default="gptq",
        help="quantization algorithm; both emit the same compressed-tensors W4A16 shape",
    )
    args = parser.parse_args(argv)
    try:
        result = quantize_checkpoint(
            args.model_dir, args.revision, args.output_dir, method=args.method
        )
        print(json.dumps({"status": "succeeded", "checkpoint": str(args.output_dir), "manifest": result}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
