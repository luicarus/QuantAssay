"""M1 acceptance: experiment manifests, fingerprints, store and resume semantics."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from quantassay.config import ConfigError, dump_spec, load_spec, validate_spec
from quantassay.contracts import (
    EvidenceStatus,
    ExperimentSpec,
    MetricRecord,
    MetricStatus,
    QuantConfig,
    Recommendation,
    RunMode,
    ModelArtifact,
    StageName,
    StageStatus,
    sha256_of,
)
from quantassay.experiments import ExperimentStore, StoreError


# --------------------------------------------------------------------------
# Config validation: invalid specs must fail immediately
# --------------------------------------------------------------------------


def test_invalid_group_size_fails() -> None:
    with pytest.raises(ConfigError) as exc:
        validate_spec({"quant": {"group_size": 0}})
    assert "group_size" in str(exc.value)

    with pytest.raises(ConfigError):
        validate_spec({"quant": {"group_size": -128}})


def test_negative_sample_counts_fail() -> None:
    with pytest.raises(ConfigError) as exc:
        validate_spec({"data": {"calibration_samples": -1}})
    assert "calibration_samples" in str(exc.value)

    with pytest.raises(ConfigError):
        validate_spec({"data": {"dev_ppl_documents": -5}})

    with pytest.raises(ConfigError):
        validate_spec({"intervention": {"top_k_layers": -2}})


def test_empty_root_and_non_mapping_fail(tmp_path: Path) -> None:
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_spec(empty)
    assert "empty" in str(exc.value)

    listed = tmp_path / "list.yaml"
    listed.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_spec(listed)
    assert "mapping" in str(exc.value)


def test_missing_config_file_fails(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as exc:
        load_spec(tmp_path / "nope.yaml")
    assert "not found" in str(exc.value)


def test_unknown_field_is_rejected() -> None:
    # A silently ignored key would mean the run does not match its stated config.
    with pytest.raises(ConfigError) as exc:
        validate_spec({"bogus_key": 1})
    assert "bogus_key" in str(exc.value)


def test_batch_size_other_than_one_fails() -> None:
    with pytest.raises(ConfigError) as exc:
        validate_spec({"batch_size": 2})
    assert "batch_size" in str(exc.value)


def test_targets_ignore_and_exempt_overlap_fail() -> None:
    with pytest.raises(ValidationError):
        QuantConfig(targets=["q_proj"], ignore=["q_proj"])

    with pytest.raises(ValidationError):
        QuantConfig(high_precision_layers=["model.layers.18"], ignore=["model.layers.18"])

    # The same rule must also be reachable through the config-file path.
    with pytest.raises(ConfigError):
        validate_spec({"quant": {"targets": ["q_proj"], "ignore": ["q_proj"]}})


def test_loads_valid_yaml(tmp_path: Path) -> None:
    from quantassay.contracts import MODEL_ID

    config = tmp_path / "spec.yaml"
    config.write_text(
        f"name: smoke\nmodel_id: {MODEL_ID}\nseed: 7\n"
        "quant:\n  group_size: 128\n  symmetric: false\n",
        encoding="utf-8",
    )
    spec = load_spec(config)
    assert spec.model_id == MODEL_ID
    assert spec.seed == 7
    assert spec.quant.group_size == 128


def test_model_identity_is_fixed_by_the_mvp(tmp_path: Path) -> None:
    """A different model is a different experiment, not a config option."""
    from quantassay.contracts import MODEL_ID

    with pytest.raises(ConfigError) as exc:
        validate_spec({"model_id": "Qwen/Qwen2.5-3B-Instruct"})
    assert MODEL_ID in str(exc.value)

    with pytest.raises(ConfigError):
        validate_spec({"model_id": "meta-llama/Llama-3.1-8B-Instruct"})

    # Omitting it entirely must land on the fixed identity, never on a stale one.
    assert ExperimentSpec().model_id == MODEL_ID


def test_dtype_cannot_silently_switch_to_fp16() -> None:
    with pytest.raises(ConfigError) as exc:
        validate_spec({"dtype": "fp16"})
    assert "BF16" in str(exc.value) or "bf16" in str(exc.value)


def test_thinking_mode_is_part_of_the_protocol() -> None:
    from quantassay.contracts import ThinkingMode

    default = ExperimentSpec()
    # Default is explicitly non-thinking, matching the calibration path.
    assert default.thinking_mode is ThinkingMode.DISABLED
    assert default.enable_thinking is False

    enabled = ExperimentSpec(thinking_mode=ThinkingMode.ENABLED)
    assert enabled.enable_thinking is True
    # The two modes are different experiments and must not share a fingerprint.
    assert enabled.config_fingerprint() != default.config_fingerprint()


# --------------------------------------------------------------------------
# Fingerprints: a config change must yield a new fingerprint
# --------------------------------------------------------------------------


def test_config_fingerprint_is_stable_and_change_sensitive() -> None:
    base = ExperimentSpec()
    same = ExperimentSpec()
    assert base.config_fingerprint() == same.config_fingerprint()

    changed = ExperimentSpec(quant=QuantConfig(group_size=64))
    assert changed.config_fingerprint() != base.config_fingerprint()


def test_config_fingerprint_ignores_name_but_not_meaning() -> None:
    # `name` is a label, not an experimental condition.
    a = ExperimentSpec(name="alpha")
    b = ExperimentSpec(name="beta")
    assert a.config_fingerprint() == b.config_fingerprint()

    # Model revision changes the experiment.
    c = ExperimentSpec(model_revision="abc123")
    assert c.config_fingerprint() != a.config_fingerprint()


def test_fingerprint_is_deterministic_across_processes() -> None:
    spec = ExperimentSpec()
    assert sha256_of(spec.model_dump(mode="json")) == sha256_of(spec.model_dump(mode="json"))


# --------------------------------------------------------------------------
# Stores, atomic writes and run directories
# --------------------------------------------------------------------------


def test_create_run_writes_manifest_and_stage_status(tmp_path: Path) -> None:
    store = ExperimentStore.create(tmp_path / "runs", ExperimentSpec(), run_id="run-001")
    assert store.manifest_path.is_file()
    assert store.stage_status_path.is_file()

    manifest = store.load_manifest()
    assert manifest.run_id == "run-001"
    assert manifest.config_fingerprint == ExperimentSpec().config_fingerprint()
    # Every stage is present and starts pending.
    assert set(manifest.stages) == set(StageName)
    assert all(r.status is StageStatus.PENDING for r in manifest.stages.values())


def test_existing_run_dir_is_never_overwritten(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    ExperimentStore.create(root, ExperimentSpec(), run_id="dup")
    with pytest.raises(StoreError) as exc:
        ExperimentStore.create(root, ExperimentSpec(), run_id="dup")
    assert "refusing to overwrite" in str(exc.value)


def test_invalid_run_id_is_rejected(tmp_path: Path) -> None:
    for bad in ("../escape", "a/b", "..", "", "with space"):
        with pytest.raises(StoreError):
            ExperimentStore.create(tmp_path / "runs", ExperimentSpec(), run_id=bad)


def test_path_escape_is_rejected(tmp_path: Path) -> None:
    store = ExperimentStore.create(tmp_path / "runs", ExperimentSpec(), run_id="r1")
    with pytest.raises(StoreError) as exc:
        store.path("../../etc/passwd")
    assert "escapes" in str(exc.value)


def test_atomic_write_leaves_no_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    store = ExperimentStore.create(tmp_path / "runs", ExperimentSpec(), run_id="r1")
    store.save_manifest(store.load_manifest())
    leftovers = [p.name for p in tmp_path.glob(".out.json.*.tmp")]
    assert leftovers == []
    assert (tmp_path / "runs" / "r1" / "manifest.json").is_file()
    assert not target.exists()


def test_corrupt_manifest_is_detected(tmp_path: Path) -> None:
    store = ExperimentStore.create(tmp_path / "runs", ExperimentSpec(), run_id="r1")
    store.manifest_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(StoreError) as exc:
        store.load_manifest()
    assert "corrupt" in str(exc.value)


# --------------------------------------------------------------------------
# State machine, failure recording and resume
# --------------------------------------------------------------------------


def test_stage_transitions_and_resume_plan(tmp_path: Path) -> None:
    store = ExperimentStore.create(tmp_path / "runs", ExperimentSpec(), run_id="r1")

    store.begin_stage(StageName.DATA)
    store.complete_stage(StageName.DATA, artifacts=["data-manifest.json"])
    store.begin_stage(StageName.BASELINE)
    store.complete_stage(StageName.BASELINE, artifacts=["baseline.json"])

    plan = store.resume_plan()
    assert plan["reusable_stages"] == ["data", "baseline"]
    assert plan["last_succeeded_stage"] == "baseline"
    assert plan["next_stage"] == "quantize"
    assert "quantize" in plan["stages_to_run"]
    assert not store.is_complete()


def test_interrupted_stage_is_not_treated_as_success(tmp_path: Path) -> None:
    # Simulates a session dying mid-stage: status stays `running`.
    store = ExperimentStore.create(tmp_path / "runs", ExperimentSpec(), run_id="r1")
    store.begin_stage(StageName.DATA)
    store.begin_stage(StageName.BASELINE)
    store.begin_stage(StageName.QUANTIZE)

    plan = store.resume_plan()
    assert plan["reusable_stages"] == []
    assert "quantize" in plan["stages_to_run"]
    assert plan["next_stage"] == "data"


def test_failed_stage_records_error_and_is_rerun(tmp_path: Path) -> None:
    store = ExperimentStore.create(tmp_path / "runs", ExperimentSpec(), run_id="r1")
    store.begin_stage(StageName.DATA)
    store.fail_stage(StageName.DATA, "OOM during calibration")

    manifest = store.load_manifest()
    record = manifest.stage(StageName.DATA)
    assert record.status is StageStatus.FAILED
    assert record.error == "OOM during calibration"
    assert record.finished_at is not None

    plan = store.resume_plan()
    assert "data" in plan["stages_to_run"]
    assert plan["reusable_stages"] == []


def test_stage_status_artifact_matches_manifest(tmp_path: Path) -> None:
    store = ExperimentStore.create(tmp_path / "runs", ExperimentSpec(), run_id="r1")
    store.begin_stage(StageName.DATA)
    store.complete_stage(StageName.DATA, artifacts=["data-manifest.json"])

    payload = json.loads(store.stage_status_path.read_text(encoding="utf-8"))
    assert payload["data"]["status"] == "succeeded"
    assert payload["data"]["artifacts"] == ["data-manifest.json"]
    assert payload["quantize"]["status"] == "pending"


def test_terminal_status_requires_finished_at() -> None:
    from quantassay.contracts import StageRecord

    with pytest.raises(Exception):
        StageRecord(stage=StageName.DATA, status=StageStatus.SUCCEEDED)


def test_failed_status_requires_error() -> None:
    from quantassay.contracts import StageRecord

    with pytest.raises(Exception):
        StageRecord(
            stage=StageName.DATA,
            status=StageStatus.FAILED,
            finished_at=datetime.now(timezone.utc),
        )


# --------------------------------------------------------------------------
# Cache: config changes must not reuse stale results
# --------------------------------------------------------------------------


def test_cache_roundtrip(tmp_path: Path) -> None:
    store = ExperimentStore.create(tmp_path / "runs", ExperimentSpec(), run_id="r1")
    key = store.cache_key(StageName.EVALUATE, {"dataset": "dev"})
    store.store_cache(key, "cache/dev-metrics.json", {"ppl": 7.1})

    entry = store.lookup_cache(key)
    assert entry is not None
    assert entry.checksum


def test_cache_key_changes_when_config_changes(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    base_store = ExperimentStore.create(root, ExperimentSpec(), run_id="base")
    changed_store = ExperimentStore.create(
        root, ExperimentSpec(quant=QuantConfig(group_size=64)), run_id="changed"
    )

    payload = {"dataset": "dev"}
    assert base_store.cache_key(StageName.EVALUATE, payload) != changed_store.cache_key(
        StageName.EVALUATE, payload
    )


def test_cache_key_changes_when_inputs_change(tmp_path: Path) -> None:
    store = ExperimentStore.create(tmp_path / "runs", ExperimentSpec(), run_id="r1")
    first = store.cache_key(StageName.EVALUATE, {"dataset": "dev"})
    second = store.cache_key(StageName.EVALUATE, {"dataset": "final"})
    assert first != second


def test_tampered_cache_artifact_is_not_reused(tmp_path: Path) -> None:
    store = ExperimentStore.create(tmp_path / "runs", ExperimentSpec(), run_id="r1")
    key = store.cache_key(StageName.EVALUATE, {"dataset": "dev"})
    store.store_cache(key, "cache/dev.json", {"ppl": 7.1})

    store.path("cache/dev.json").write_text('{"ppl": 1.0}\n', encoding="utf-8")
    assert store.lookup_cache(key) is None


def test_missing_cache_artifact_is_not_reused(tmp_path: Path) -> None:
    store = ExperimentStore.create(tmp_path / "runs", ExperimentSpec(), run_id="r1")
    key = store.cache_key(StageName.EVALUATE, {"dataset": "dev"})
    store.store_cache(key, "cache/dev.json", {"ppl": 7.1})
    store.path("cache/dev.json").unlink()
    assert store.lookup_cache(key) is None


def test_invalidate_and_prune_cache(tmp_path: Path) -> None:
    store = ExperimentStore.create(tmp_path / "runs", ExperimentSpec(), run_id="r1")
    key_a = store.cache_key(StageName.EVALUATE, {"dataset": "a"})
    key_b = store.cache_key(StageName.EVALUATE, {"dataset": "b"})
    store.store_cache(key_a, "cache/a.json", {"v": 1})
    store.store_cache(key_b, "cache/b.json", {"v": 2})

    assert store.invalidate_cache(key_a) is True
    assert store.lookup_cache(key_a) is None
    assert store.lookup_cache(key_b) is not None

    removed = store.prune_cache(keep_keys=set())
    assert key_b in removed
    assert not store.path("cache/b.json").exists()


# --------------------------------------------------------------------------
# Metrics must never encode "missing" as zero
# --------------------------------------------------------------------------


def test_unavailable_metric_cannot_carry_a_value() -> None:
    with pytest.raises(Exception):
        MetricRecord(name="ppl", value=0.0, status=MetricStatus.UNAVAILABLE, reason="n/a")
    with pytest.raises(Exception):
        MetricRecord(name="ppl", value=0.0, status=MetricStatus.NOT_SUPPORTED, reason="n/a")


def test_unavailable_metric_requires_reason() -> None:
    with pytest.raises(Exception):
        MetricRecord(name="ppl", status=MetricStatus.UNAVAILABLE)

    record = MetricRecord.unavailable("ppl", "zero valid tokens")
    assert record.value is None
    assert record.status is MetricStatus.UNAVAILABLE
    assert record.reason == "zero valid tokens"


def test_ok_metric_requires_value() -> None:
    with pytest.raises(Exception):
        MetricRecord(name="ppl", status=MetricStatus.OK)
    assert MetricRecord(name="ppl", value=7.1).value == 7.1


# --------------------------------------------------------------------------
# Evidence discipline
# --------------------------------------------------------------------------


def test_quantized_artifact_requires_recipe() -> None:
    with pytest.raises(Exception):
        ModelArtifact(
            artifact_id="a1",
            checkpoint_path="/tmp/a1",
            parent_model_id="Qwen/Qwen2.5-3B-Instruct",
            format="gptq",
            run_mode=RunMode.PACKED,
        )

    ok = ModelArtifact(
        artifact_id="a1",
        checkpoint_path="/tmp/a1",
        parent_model_id="Qwen/Qwen2.5-3B-Instruct",
        format="gptq",
        run_mode=RunMode.PACKED,
        quant=QuantConfig(),
    )
    assert ok.quant is not None


def test_unsupported_and_not_tested_are_distinct() -> None:
    from quantassay.contracts import CapabilityReport

    report = CapabilityReport(
        unsupported={"fp8_w8a8": "no native FP8 on this GPU"},
        not_tested={"kv_cache_int8": "deferred to F4"},
    )
    assert "fp8_w8a8" in report.unsupported
    assert "kv_cache_int8" not in report.unsupported

    with pytest.raises(Exception):
        CapabilityReport(unsupported={"x": "a"}, not_tested={"x": "b"})


# --------------------------------------------------------------------------
# Snapshot export (needed before a cloud session ends)
# --------------------------------------------------------------------------


def test_export_snapshot_copies_records(tmp_path: Path) -> None:
    store = ExperimentStore.create(tmp_path / "runs", ExperimentSpec(), run_id="r1")
    store.complete_stage(StageName.DATA, artifacts=["data-manifest.json"])
    dest = store.export_snapshot(tmp_path / "backup")
    assert (dest / "manifest.json").is_file()
    assert (dest / "stage-status.json").is_file()


def test_resolved_config_dump_roundtrips(tmp_path: Path) -> None:
    spec = ExperimentSpec()
    out = dump_spec(spec, tmp_path / "resolved-config.yaml")
    reloaded = load_spec(out)
    assert reloaded.config_fingerprint() == spec.config_fingerprint()
