"""The shipped MVP config must load and keep its documented meaning.

The config is the contract between the plan and any run. If it drifts — a
different model, a mutable workload, thinking mode quietly on — the resulting
numbers stop meaning what the report claims, so it is asserted here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from quantassay.config import ConfigError, load_spec  # noqa: E402
from quantassay.contracts import (  # noqa: E402
    MODEL_ID,
    ExperimentSpec,
    ServingMetric,
    ThinkingMode,
)

CONFIG_PATH = REPO_ROOT / "configs" / "mvp-qwen3-0p6b.yaml"


def test_formal_config_exists() -> None:
    assert CONFIG_PATH.is_file(), f"missing formal config: {CONFIG_PATH}"


def test_formal_config_loads() -> None:
    assert isinstance(load_spec(CONFIG_PATH), ExperimentSpec)


def test_formal_config_uses_the_fixed_model() -> None:
    assert load_spec(CONFIG_PATH).model_id == MODEL_ID


def test_formal_config_is_bf16() -> None:
    assert load_spec(CONFIG_PATH).dtype == "bf16"


def test_formal_config_disables_thinking() -> None:
    spec = load_spec(CONFIG_PATH)
    assert spec.thinking_mode is ThinkingMode.DISABLED
    assert spec.workload.thinking_mode is ThinkingMode.DISABLED


def test_formal_config_serves_one_model_at_a_time() -> None:
    assert load_spec(CONFIG_PATH).batch_size == 1


def test_formal_config_declares_a_single_reproducible_workload() -> None:
    spec = load_spec(CONFIG_PATH)
    workload = spec.workload
    assert workload.timed_requests >= 1
    assert workload.warmup_requests >= 0
    # Cache and arrival settings must be stated, not left implicit.
    assert workload.cache_policy
    assert workload.retry_policy


def test_formal_config_workload_hashes_its_requests() -> None:
    spec = load_spec(CONFIG_PATH)
    assert spec.workload.request_hashes
    assert spec.workload.workload_fingerprint()


def test_formal_config_uses_w4a16_group128() -> None:
    spec = load_spec(CONFIG_PATH)
    assert spec.quant.algorithm == "gptq"
    assert spec.quant.scheme == "W4A16"
    assert spec.quant.weight_bits == 4
    assert spec.quant.activation_bits == 16
    assert spec.quant.group_size == 128


def test_formal_config_ignores_lm_head() -> None:
    assert "lm_head" in load_spec(CONFIG_PATH).quant.ignore


def test_formal_config_leaves_revision_unpinned_until_first_run() -> None:
    """A fabricated revision hash would be worse than an explicit null."""
    spec = load_spec(CONFIG_PATH)
    assert spec.model_revision is None


def test_formal_config_does_not_pretend_to_evaluate_quality() -> None:
    spec = load_spec(CONFIG_PATH)
    # The MVP does not run quality evaluation; document counts stay zero.
    assert spec.data.dev_ppl_documents == 0
    assert spec.data.final_ppl_documents == 0
    # The schema has no layer-trace or intervention blocks at all: those belong
    # to the abandoned precision-regression route, not this serving MVP.
    assert not hasattr(spec, "trace")
    assert not hasattr(spec, "intervention")


def test_formal_config_keeps_calibration_separate_from_serving() -> None:
    spec = load_spec(CONFIG_PATH)
    assert spec.quant.calibration_samples == spec.data.calibration_samples
    assert spec.workload.request_set_id
    # Calibration source must not be the serving request set.
    assert spec.data.source != spec.workload.request_set_id


def test_formal_config_fingerprint_is_stable() -> None:
    assert load_spec(CONFIG_PATH).config_fingerprint() == load_spec(CONFIG_PATH).config_fingerprint()


def test_four_mvp_metrics_are_representable() -> None:
    """The workload/contract layer must be able to carry all four metrics."""
    from quantassay.contracts import MVP_SERVING_METRICS

    for metric in MVP_SERVING_METRICS:
        assert isinstance(metric, ServingMetric)


def test_config_with_a_wrong_model_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("model_id: Qwen/Qwen2.5-3B-Instruct\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_spec(bad)
    assert MODEL_ID in str(exc.value)
