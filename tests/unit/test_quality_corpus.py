"""Corpus slicing contracts: reproducible, disjoint, and honest about shortfall.

The measured numbers depend on *which* documents were scored, so the selection
must be deterministic and its identity recorded. The dangerous outcomes:

* dev and final overlapping (the holdout stops being held out);
* a selection that shifts when the loader's iteration order changes;
* silently backfilling a split that could not be filled.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from quantassay.contracts import DataConfig, DataSplit  # noqa: E402
from quantassay.evaluation.corpus import (  # noqa: E402
    CorpusDocuments,
    corpus_id_prefix,
    build_quality_dataset,
    dataset_fingerprint,
    holdout_records,
    select_documents,
)


def _corpus(n: int, *, chars: int = 500) -> CorpusDocuments:
    """A synthetic corpus whose documents are identifiable by index."""
    return CorpusDocuments(
        documents=[f"doc{i:03d}-" + ("word " * (chars // 5)) for i in range(n)],
        corpus="synthetic/test",
        config="v1",
        source_split="test",
        endpoint="https://example.invalid",
        revision="rev-1",
    )


def _config(calibration: int, dev: int, final: int) -> DataConfig:
    return DataConfig(
        source="synthetic/test",
        revision="v1",
        language=["en"],
        calibration_samples=calibration,
        dev_ppl_documents=dev,
        final_ppl_documents=final,
        max_tokens=512,
        smoke_max_tokens=128,
    )


# --------------------------------------------------------------------------
# Deterministic selection
# --------------------------------------------------------------------------


def test_selection_takes_corpus_order_and_skips_short_documents() -> None:
    docs = ["too short", "a" * 500, "b" * 500]
    selected = select_documents(docs, limit=5, min_chars=400)
    # The short row is skipped; ids keep their original corpus position.
    assert [d.document_id for d in selected] == ["doc-000001", "doc-000002"]


def test_selection_id_prefix_is_configurable() -> None:
    """Sample ids must name the corpus that produced them.

    A hardcoded ``wiki-`` prefix would mislabel runs that measured a user's own
    dataset — and the tool exists to be pointed at such datasets.
    """
    docs = ["a" * 500]
    selected = select_documents(docs, limit=1, min_chars=400, id_prefix="c4")
    assert selected[0].document_id == "c4-000000"


def test_corpus_id_prefix_is_derived_from_the_corpus_name() -> None:
    def corpus(name: str) -> CorpusDocuments:
        return CorpusDocuments(
            documents=[], corpus=name, config="", source_split="test", endpoint="x"
        )

    assert corpus_id_prefix(corpus("Salesforce/wikitext")) == "wikitext"
    assert corpus_id_prefix(corpus("allenai/c4")) == "c4"
    # Unsafe characters are normalised, so the id is report/filesystem friendly.
    assert corpus_id_prefix(corpus("my_org/My.Data.v2")) == "my-data-v2"
    assert corpus_id_prefix(corpus("plain")) == "plain"


def test_selection_is_stable_across_repeated_calls() -> None:
    corpus = _corpus(50)
    first = select_documents(corpus.documents, limit=10)
    second = select_documents(corpus.documents, limit=10)
    assert [d.document_id for d in first] == [d.document_id for d in second]


def test_selection_stops_at_the_limit() -> None:
    selected = select_documents(_corpus(100).documents, limit=7)
    assert len(selected) == 7


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------


def test_splits_are_disjoint_by_text_hash() -> None:
    """dev and final must not share content, or the holdout is not held out."""
    result = build_quality_dataset(
        corpus=_corpus(60),
        data_config=_config(calibration=4, dev=10, final=10),
        tokenizer_id="Qwen/Qwen3-0.6B",
    )
    manifest = result.manifest
    assert manifest.cross_split_text_hashes == []
    assert result.is_usable is True

    dev = {r.text_hash for r in holdout_records(manifest, DataSplit.DEV)}
    final = {r.text_hash for r in holdout_records(manifest, DataSplit.FINAL)}
    assert dev and final
    assert dev.isdisjoint(final)


def test_split_sizes_match_the_requested_counts() -> None:
    result = build_quality_dataset(
        corpus=_corpus(60),
        data_config=_config(calibration=4, dev=10, final=6),
        tokenizer_id="Qwen/Qwen3-0.6B",
    )
    manifest = result.manifest
    assert len(holdout_records(manifest, DataSplit.CALIBRATION)) == 4
    assert len(holdout_records(manifest, DataSplit.DEV)) == 10
    assert len(holdout_records(manifest, DataSplit.FINAL)) == 6


def test_shortfall_is_recorded_and_never_backfilled() -> None:
    """Asking for more than the corpus supplies must not borrow from elsewhere."""
    result = build_quality_dataset(
        corpus=_corpus(5),
        data_config=_config(calibration=2, dev=8, final=8),
        tokenizer_id="Qwen/Qwen3-0.6B",
    )
    manifest = result.manifest
    total = sum(
        len(holdout_records(manifest, split))
        for split in (DataSplit.CALIBRATION, DataSplit.DEV, DataSplit.FINAL)
    )
    assert total == 5  # exactly what the corpus had
    assert any("short by" in note for note in manifest.notes)


def test_duplicate_documents_do_not_create_overlapping_splits() -> None:
    """A corpus that repeats a document must not leak it into two splits."""
    docs = [f"same-{i:03d}-" + ("word " * 100) for i in range(3)]
    repeated = docs * 4  # identical text repeated at different positions
    corpus = CorpusDocuments(
        documents=repeated, corpus="synthetic/dup", config="v1",
        source_split="test", endpoint="x", revision=None,
    )
    result = build_quality_dataset(
        corpus=corpus,
        data_config=_config(calibration=1, dev=2, final=2),
        tokenizer_id="Qwen/Qwen3-0.6B",
    )
    assert result.manifest.duplicate_text_hashes == []
    assert result.manifest.cross_split_text_hashes == []
    # Only the three distinct documents can be used, however often they repeat.
    assert len(result.manifest.samples) == 3


# --------------------------------------------------------------------------
# Fingerprint
# --------------------------------------------------------------------------


def test_fingerprint_changes_when_the_corpus_identity_changes() -> None:
    config = _config(2, 4, 4)
    base = build_quality_dataset(
        corpus=_corpus(30), data_config=config, tokenizer_id="tok"
    )
    other_corpus = _corpus(30)
    other_corpus.revision = "rev-2"
    other = build_quality_dataset(
        corpus=other_corpus, data_config=config, tokenizer_id="tok"
    )

    assert dataset_fingerprint(base.manifest, _corpus(30)) != dataset_fingerprint(
        other.manifest, other_corpus
    )


def test_fingerprint_is_stable_for_identical_inputs() -> None:
    config = _config(2, 4, 4)
    first = build_quality_dataset(corpus=_corpus(30), data_config=config, tokenizer_id="tok")
    second = build_quality_dataset(corpus=_corpus(30), data_config=config, tokenizer_id="tok")
    assert dataset_fingerprint(first.manifest, _corpus(30)) == dataset_fingerprint(
        second.manifest, _corpus(30)
    )


def test_fingerprint_changes_when_the_split_sizes_change() -> None:
    """A different dev size means different documents were measured."""
    small = build_quality_dataset(
        corpus=_corpus(30), data_config=_config(1, 3, 3), tokenizer_id="tok"
    )
    large = build_quality_dataset(
        corpus=_corpus(30), data_config=_config(1, 6, 3), tokenizer_id="tok"
    )
    corpus = _corpus(30)
    assert dataset_fingerprint(small.manifest, corpus) != dataset_fingerprint(
        large.manifest, corpus
    )