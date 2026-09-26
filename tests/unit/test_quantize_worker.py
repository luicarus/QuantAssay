"""CPU validation for the quantization artifact boundary; no backend imports."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from quantassay.quantize_worker import (
    QUANT_METHODS,
    QWEN3_AWQ_MAPPINGS,
    RECIPES,
    QuantizeError,
    build_calibration_records,
    build_recipe,
    build_symmetric_w4a16_scheme,
    import_modifier,
    inspect_packed_checkpoint,
    render_calibration_texts,
    validate_w4a16_scheme,
)


def test_inspect_checkpoint_requires_packed_w4a16_group_128(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"nonempty")
    (checkpoint / "tokenizer.json").write_text("{}", encoding="utf-8")
    config = {
        "model_type": "qwen3",
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {"num_bits": 4, "group_size": 128, "type": "int"},
                    "input_activations": None,
                }
            },
        },
    }
    (checkpoint / "config.json").write_text(json.dumps(config), encoding="utf-8")
    assert inspect_packed_checkpoint(checkpoint)["format"] == "pack-quantized"

    config["quantization_config"]["config_groups"]["group_0"]["weights"]["num_bits"] = 8
    (checkpoint / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(QuantizeError, match="4-bit"):
        inspect_packed_checkpoint(checkpoint)


def test_calibration_uses_non_thinking_template() -> None:
    calls: list[dict[str, object]] = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append(kwargs)
            return f"formatted: {messages[0]['content']}"

    texts = render_calibration_texts(Tokenizer())
    assert len(texts) == 4
    assert all(call["enable_thinking"] is False for call in calls)
    assert all(call["add_generation_prompt"] is True for call in calls)
    assert all("Reply with one short sentence in English." not in text for text in texts)


def test_calibration_tokenization_does_not_add_special_tokens_twice() -> None:
    calls: list[dict[str, object]] = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return f"<chat>{messages[0]['content']}</chat>"

        def __call__(self, text, **kwargs):
            calls.append(kwargs)
            return {"input_ids": [1, 2], "attention_mask": [1, 1]}

    records = build_calibration_records(Tokenizer())
    assert len(records) == 4
    assert records[0] == {"input_ids": [1, 2], "attention_mask": [1, 1]}
    assert all(call["add_special_tokens"] is False for call in calls)
    assert all(call["max_length"] == 128 for call in calls)


def test_calibration_records_use_the_exact_texts_that_are_hashed() -> None:
    class Tokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            raise AssertionError("must not render calibration twice")

        def __call__(self, text, **kwargs):
            assert text == "already-rendered"
            return {"input_ids": [5], "attention_mask": [1]}

    assert build_calibration_records(Tokenizer(), ["already-rendered"]) == [
        {"input_ids": [5], "attention_mask": [1]}
    ]


def test_recipe_preset_is_validated_before_quantization() -> None:
    scheme = SimpleNamespace(
        weights=SimpleNamespace(num_bits=4, group_size=128, type="int", symmetric=True),
        input_activations=None,
    )
    validate_w4a16_scheme(scheme)
    scheme.weights.group_size = 64
    with pytest.raises(QuantizeError, match="group_size=128"):
        validate_w4a16_scheme(scheme)


# --------------------------------------------------------------------------
# AWQ support: the servability constraint and the mapping trap
# --------------------------------------------------------------------------


def test_asymmetric_scheme_is_rejected_because_it_cannot_be_served() -> None:
    """AWQ's upstream examples are asymmetric; this stack cannot serve that.

    sglang's CompressedTensorsWNA16 dispatch requires is_symmetric, so an
    asymmetric checkpoint never reaches the marlin kernel. Catching it in the
    quantizer avoids producing a checkpoint that only fails later, at serving
    time, after a long GPU run.
    """
    asymmetric = SimpleNamespace(
        weights=SimpleNamespace(num_bits=4, group_size=128, type="int", symmetric=False),
        input_activations=None,
    )
    with pytest.raises(QuantizeError, match="symmetric"):
        validate_w4a16_scheme(asymmetric)

    # A scheme that does not state symmetry at all is also refused: assuming
    # symmetry would be exactly the silent failure this guard exists for.
    silent = SimpleNamespace(
        weights=SimpleNamespace(num_bits=4, group_size=128, type="int"),
        input_activations=None,
    )
    with pytest.raises(QuantizeError, match="symmetric"):
        validate_w4a16_scheme(silent)


def test_both_methods_share_one_symmetric_scheme() -> None:
    """The scheme is shared so a method comparison isolates the algorithm.

    If AWQ ran asymmetric while GPTQ ran symmetric, a latency/quality difference
    would be confounded by the scheme instead of attributable to the algorithm.
    """
    # The scheme builder needs compressed-tensors (execution layer only); the
    # recipe metadata below is plain data and is checked everywhere.
    for method in QUANT_METHODS:
        recipe = RECIPES[method]
        assert recipe["scheme"] == "W4A16"
        assert recipe["group_size"] == 128
        assert recipe["ignore"] == ["lm_head"]

    pytest.importorskip("compressed_tensors", reason="execution-layer dependency")
    first = build_symmetric_w4a16_scheme()
    second = build_symmetric_w4a16_scheme()
    assert first.weights.symmetric is True
    assert first.model_dump() == second.model_dump()


def test_awq_mappings_target_qwen3_module_names() -> None:
    """The AWQ docstring example uses OPT names, which match nothing on Qwen3.

    A mapping that matches no module would make AWQ perform no smoothing at all
    while still reporting success — the silent no-op this constant prevents.
    """
    smooth_names = [m["smooth_layer"] for m in QWEN3_AWQ_MAPPINGS]
    assert any("input_layernorm" in n for n in smooth_names)
    assert any("post_attention_layernorm" in n for n in smooth_names)
    for forbidden in ("self_attn_layer_norm", "final_layer_norm"):
        assert not any(forbidden in n for n in smooth_names), (
            f"{forbidden!r} is the OPT-style name from the upstream example and "
            "matches no Qwen3 module"
        )
    for mapping in QWEN3_AWQ_MAPPINGS:
        assert mapping["balance_layers"], "a mapping needs balance layers"
        for layer in mapping["balance_layers"]:
            assert layer.startswith("re:")


def test_awq_recipe_records_the_symmetry_trade_off() -> None:
    """The deviation from upstream must be recorded, not silent."""
    recipe = RECIPES["awq"]
    assert recipe["symmetric"] is True
    assert "symmetric" in recipe["note"]
    assert "upstream" in recipe["note"]


def test_unknown_method_is_rejected() -> None:
    """Routing fails loudly before any backend import is attempted."""
    with pytest.raises(QuantizeError, match="unknown quantization method"):
        import_modifier("int8")


def test_each_method_resolves_to_a_distinct_modifier() -> None:
    """Catching a renamed/removed class here beats failing mid-GPU-run."""
    pytest.importorskip("llmcompressor", reason="execution-layer dependency")
    resolved = {method: import_modifier(method).__name__ for method in QUANT_METHODS}
    assert resolved["gptq"] == "GPTQModifier"
    assert resolved["awq"] == "AWQModifier"
    assert len(set(resolved.values())) == 2


def test_awq_mapping_builder_is_only_reachable_with_the_backend() -> None:
    """A mapping is constructed only for AWQ, and only where the backend exists."""
    pytest.importorskip("llmcompressor", reason="execution-layer dependency")
    scheme = build_symmetric_w4a16_scheme()
    recipe = build_recipe("awq", scheme)
    assert recipe.__class__.__name__ == "AWQModifier"
    assert len(recipe.mappings) == len(QWEN3_AWQ_MAPPINGS)
