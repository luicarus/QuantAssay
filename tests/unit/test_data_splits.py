"""M1 acceptance: data splits, fixed IDs, hashing and leakage detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from quantassay.contracts import DataConfig, DataSplit
from quantassay.evaluation.data import (
    TextDocument,
    detect_leakage,
    normalize_text,
    prepare_data,
    stable_sample_id,
    text_hash,
)


def _docs(prefix: str, count: int, *, words: int = 8) -> list[TextDocument]:
    return [
        TextDocument(document_id=f"{prefix}-{i}", text=f"{prefix} document number {i} " + "lorem ipsum " * words)
        for i in range(count)
    ]


# --------------------------------------------------------------------------
# Normalization and hashing
# --------------------------------------------------------------------------


def test_normalize_collapses_formatting_only_differences() -> None:
    a = "Hello   world\r\n\r\nsecond   line"
    b = "Hello world second line"
    assert normalize_text(a) == normalize_text(b)
    assert text_hash(a) == text_hash(b)


def test_normalize_applies_nfkc() -> None:
    # Full-width and compatibility forms must not disguise a duplicate.
    assert normalize_text("ＡＢＣ１２３") == normalize_text("ABC123")


def test_text_hash_is_stable() -> None:
    assert text_hash("abc") == text_hash("abc")
    assert text_hash("abc") != text_hash("abd")


def test_stable_sample_id_is_deterministic_and_split_scoped() -> None:
    first = stable_sample_id(DataSplit.DEV, "doc-1", 0)
    assert first == stable_sample_id(DataSplit.DEV, "doc-1", 0)
    assert first != stable_sample_id(DataSplit.DEV, "doc-1", 1)
    assert first != stable_sample_id(DataSplit.FINAL, "doc-1", 0)


# --------------------------------------------------------------------------
# Basic split preparation
# --------------------------------------------------------------------------


def test_prepare_data_respects_requested_counts() -> None:
    config = DataConfig(
        calibration_samples=2, dev_ppl_documents=3, final_ppl_documents=4, max_tokens=512
    )
    result = prepare_data(
        config,
        "Qwen/Qwen3-0.6B",
        calibration=_docs("cal", 5),
        dev=_docs("dev", 5),
        final=_docs("fin", 5),
    )

    assert len(result.manifest.ids_for(DataSplit.CALIBRATION)) == 2
    assert len(result.manifest.ids_for(DataSplit.DEV)) == 3
    assert len(result.manifest.ids_for(DataSplit.FINAL)) == 4
    assert result.is_usable


def test_prepare_data_records_shortfall_instead_of_backfilling() -> None:
    config = DataConfig(calibration_samples=10, dev_ppl_documents=1, final_ppl_documents=1)
    result = prepare_data(
        config,
        "tok",
        calibration=_docs("cal", 2),
        dev=_docs("dev", 1),
        final=_docs("fin", 1),
    )

    assert len(result.manifest.ids_for(DataSplit.CALIBRATION)) == 2
    # The shortfall is recorded, not silently padded from another split.
    assert any("short by 8" in note for note in result.manifest.notes)
    assert len(result.manifest.ids_for(DataSplit.DEV)) == 1


def test_empty_documents_are_skipped() -> None:
    config = DataConfig(calibration_samples=3, dev_ppl_documents=1, final_ppl_documents=1)
    calibration = [
        TextDocument(document_id="empty", text="   \n\t  "),
        TextDocument(document_id="real-1", text="real content one"),
        TextDocument(document_id="real-2", text="real content two"),
    ]
    result = prepare_data(config, "tok", calibration=calibration, dev=_docs("dev", 1), final=_docs("fin", 1))

    ids_used = result.manifest.ids_for(DataSplit.CALIBRATION)
    assert len(ids_used) == 2
    assert any("skipped 1 empty document" in note for note in result.manifest.notes)


def test_token_budget_truncation_is_recorded() -> None:
    config = DataConfig(calibration_samples=1, dev_ppl_documents=1, final_ppl_documents=1, max_tokens=4)
    long_doc = [TextDocument(document_id="long", text="word " * 500)]
    result = prepare_data(config, "tok", calibration=long_doc, dev=_docs("dev", 1), final=_docs("fin", 1))

    record = result.manifest.samples[0]
    assert record.truncated is True
    assert record.token_count <= 4


# --------------------------------------------------------------------------
# Leakage detection: the blocking condition
# --------------------------------------------------------------------------


def test_cross_split_duplicate_is_detected_and_blocks_use() -> None:
    shared = "this exact text appears in two splits"
    config = DataConfig(calibration_samples=1, dev_ppl_documents=1, final_ppl_documents=1)

    result = prepare_data(
        config,
        "tok",
        calibration=[TextDocument(document_id="c", text=shared)],
        dev=[TextDocument(document_id="d", text="unique dev text")],
        final=[TextDocument(document_id="f", text=shared)],
    )

    # The duplicate is kept out of the later split entirely...
    assert len(result.manifest.ids_for(DataSplit.FINAL)) == 0
    # ...and the shortfall is recorded rather than backfilled.
    assert any("short by 1" in note for note in result.manifest.notes)
    assert result.is_usable  # no actual overlap remains in the prepared set


def test_detect_leakage_reports_cross_split_hashes() -> None:
    from quantassay.contracts import SampleRecord

    digest = text_hash("shared")
    records = [
        SampleRecord(
            sample_id="a", text_hash=digest, split=DataSplit.CALIBRATION, token_count=2
        ),
        SampleRecord(sample_id="b", text_hash=digest, split=DataSplit.FINAL, token_count=2),
    ]
    dup_ids, dup_hashes, cross = detect_leakage(records, ["shared", "shared"])
    assert digest in dup_hashes
    assert cross == [digest]


def test_duplicate_sample_ids_are_reported() -> None:
    from quantassay.contracts import SampleRecord

    records = [
        SampleRecord(sample_id="same", text_hash="h1", split=DataSplit.DEV, token_count=1),
        SampleRecord(sample_id="same", text_hash="h2", split=DataSplit.DEV, token_count=1),
    ]
    dup_ids, _, _ = detect_leakage(records, ["x", "y"])
    assert dup_ids == ["same"]


def test_manifest_has_leakage_flag() -> None:
    from quantassay.contracts import DatasetManifest

    clean = DatasetManifest(dataset=DataConfig(), tokenizer_id="tok")
    assert clean.has_leakage() is False

    leaked = DatasetManifest(
        dataset=DataConfig(),
        tokenizer_id="tok",
        cross_split_text_hashes=["deadbeef"],
    )
    assert leaked.has_leakage() is True


def test_formatting_variant_cannot_smuggle_a_duplicate_into_holdout() -> None:
    """Whitespace-only differences must not defeat duplicate detection."""
    config = DataConfig(calibration_samples=1, dev_ppl_documents=1, final_ppl_documents=1)
    result = prepare_data(
        config,
        "tok",
        calibration=[TextDocument(document_id="c", text="Alpha beta gamma")],
        dev=[TextDocument(document_id="d", text="unrelated dev content")],
        final=[TextDocument(document_id="f", text="Alpha   beta\n\n gamma")],
    )
    assert len(result.manifest.ids_for(DataSplit.FINAL)) == 0


# --------------------------------------------------------------------------
# Manifest fingerprints and reproducibility
# --------------------------------------------------------------------------


def test_same_inputs_produce_same_split_hashes() -> None:
    config = DataConfig(calibration_samples=2, dev_ppl_documents=2, final_ppl_documents=2)
    kwargs = dict(calibration=_docs("cal", 2), dev=_docs("dev", 2), final=_docs("fin", 2))

    first = prepare_data(config, "tok", **kwargs)
    second = prepare_data(config, "tok", **kwargs)

    assert first.manifest.split_hashes == second.manifest.split_hashes
    assert first.manifest.manifest_fingerprint() == second.manifest.manifest_fingerprint()


def test_changed_data_changes_split_hash() -> None:
    config = DataConfig(calibration_samples=1, dev_ppl_documents=1, final_ppl_documents=1)
    base = prepare_data(
        config, "tok", calibration=_docs("cal", 1), dev=_docs("dev", 1), final=_docs("fin", 1)
    )
    changed = prepare_data(
        config,
        "tok",
        calibration=[TextDocument(document_id="cal-0", text="completely different text")],
        dev=_docs("dev", 1),
        final=_docs("fin", 1),
    )
    assert base.manifest.split_hashes != changed.manifest.split_hashes


def test_manifest_records_inputs_fingerprint() -> None:
    config = DataConfig(calibration_samples=1, dev_ppl_documents=1, final_ppl_documents=1)
    result = prepare_data(
        config, "tok", calibration=_docs("cal", 1), dev=_docs("dev", 1), final=_docs("fin", 1)
    )
    assert any(note.startswith("inputs_fingerprint=") for note in result.manifest.notes)


def test_holdout_text_is_not_used_for_selection() -> None:
    """The prepared result must let callers address dev and final separately."""
    config = DataConfig(calibration_samples=1, dev_ppl_documents=1, final_ppl_documents=1)
    result = prepare_data(
        config, "tok", calibration=_docs("cal", 1), dev=_docs("dev", 1), final=_docs("fin", 1)
    )
    dev_texts = result.texts_for(DataSplit.DEV)
    final_texts = result.texts_for(DataSplit.FINAL)
    assert dev_texts and final_texts
    assert not set(dev_texts) & set(final_texts)


def test_calibration_texts_are_available_for_quantization() -> None:
    config = DataConfig(calibration_samples=2, dev_ppl_documents=1, final_ppl_documents=1)
    result = prepare_data(
        config, "tok", calibration=_docs("cal", 3), dev=_docs("dev", 1), final=_docs("fin", 1)
    )
    assert len(result.calibration_texts()) == 2
