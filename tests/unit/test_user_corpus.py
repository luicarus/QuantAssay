"""User-supplied corpora: the tool must work on data the user chooses.

The project's purpose is letting someone compare quantization options on *their*
data, so the corpus is configuration. These contracts pin the parts that fail
silently if wrong:

* flags must map onto corpus identity (and identity lands in the fingerprint);
* a wrong text column must say which columns exist, not raise a bare KeyError;
* an empty holdout must fail loudly rather than produce a meaningless PPL;
* switching corpora inside one run directory must be refused.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from quantassay.contracts import DataConfig  # noqa: E402
from quantassay.evaluation.corpus import (  # noqa: E402
    DEFAULT_TEXT_COLUMN,
    CorpusDocuments,
    CorpusError,
    CorpusSpec,
    build_quality_dataset,
    dataset_fingerprint,
)
from quantassay.gating import ProbeError, corpus_spec_from_args  # noqa: E402


def _args(**overrides) -> SimpleNamespace:
    base = dict(
        corpus="Salesforce/wikitext",
        corpus_config=None,
        corpus_split=None,
        corpus_text_column=None,
        corpus_revision=None,
        corpus_endpoint=None,
        corpus_min_chars=400,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _corpus(name: str = "my-org/my-data", n: int = 30, chars: int = 500) -> CorpusDocuments:
    return CorpusDocuments(
        documents=[f"doc{i:03d}-" + ("word " * (chars // 5)) for i in range(n)],
        corpus=name,
        config="default",
        source_split="train",
        endpoint="https://hf-mirror.com",
        text_column="content",
    )


def _config(calibration: int = 2, dev: int = 5, final: int = 0) -> DataConfig:
    return DataConfig(
        source="my-org/my-data",
        revision=None,
        language=["en"],
        calibration_samples=calibration,
        dev_ppl_documents=dev,
        final_ppl_documents=final,
        max_tokens=512,
        smoke_max_tokens=128,
    )


# --------------------------------------------------------------------------
# Flags -> corpus identity
# --------------------------------------------------------------------------


def test_flags_default_to_the_general_corpus() -> None:
    spec = corpus_spec_from_args(_args())
    assert spec.corpus == "Salesforce/wikitext"
    assert spec.text_column == DEFAULT_TEXT_COLUMN
    assert spec.revision is None  # unpinned unless the user asks


def test_flags_carry_a_user_supplied_corpus() -> None:
    """The whole point: point the tool at your own data."""
    spec = corpus_spec_from_args(
        _args(
            corpus="acme/domain-corpus",
            corpus_config="legal",
            corpus_split="validation",
            corpus_text_column="content",
            corpus_revision="abc1234",
        )
    )
    assert spec.corpus == "acme/domain-corpus"
    assert spec.config == "legal"
    assert spec.source_split == "validation"
    assert spec.text_column == "content"
    assert spec.revision == "abc1234"
    assert "acme/domain-corpus" in spec.identity
    assert "validation" in spec.identity


def test_missing_corpus_is_rejected() -> None:
    with pytest.raises(ProbeError, match="--corpus"):
        corpus_spec_from_args(_args(corpus=""))


def test_config_default_follows_the_corpus() -> None:
    """`wikitext-2-raw-v1` must not leak into a non-wikitext dataset.

    That config name only exists for Salesforce/wikitext, so carrying it to
    another dataset fails inside `load_dataset` with a confusing error. Omitted
    config therefore means "the dataset's own default" for anything else.
    """
    wikitext = corpus_spec_from_args(_args())
    assert wikitext.config == "wikitext-2-raw-v1"

    for other in ("allenai/c4", "roneneldan/TinyStories", "my-org/my-data"):
        spec = corpus_spec_from_args(_args(corpus=other))
        assert spec.config == "", (
            f"{other} inherited config {spec.config!r}, which is wikitext-specific"
        )

    # An explicit config is always honoured.
    explicit = corpus_spec_from_args(_args(corpus="allenai/c4", corpus_config="en"))
    assert explicit.config == "en"

    # And an explicitly empty config stays empty even for wikitext.
    blank = corpus_spec_from_args(_args(corpus_config=""))
    assert blank.config == ""


def test_non_positive_min_chars_is_rejected() -> None:
    with pytest.raises(ProbeError, match="min-chars"):
        corpus_spec_from_args(_args(corpus_min_chars=0))


def test_corpus_identity_reaches_the_fingerprint() -> None:
    """Switching corpora must change the fingerprint, or old runs get reused.

    Each of these is a different measurement; sharing a fingerprint would let a
    later run compare numbers produced from different text.
    """
    base_manifest = build_quality_dataset(
        corpus=_corpus(), data_config=_config(), tokenizer_id="tok"
    ).manifest
    base = dataset_fingerprint(base_manifest, _corpus())

    other_text_column = _corpus()
    other_text_column.text_column = "body"
    other_split = _corpus()
    other_split.source_split = "test"
    other_name = _corpus("other-org/other-data")

    assert dataset_fingerprint(base_manifest, _corpus()) == base  # stable
    assert dataset_fingerprint(base_manifest, other_text_column) != base
    assert dataset_fingerprint(base_manifest, other_split) != base
    assert dataset_fingerprint(base_manifest, other_name) != base


# --------------------------------------------------------------------------
# Failure modes that would otherwise be silent
# --------------------------------------------------------------------------


def test_empty_holdout_raises_instead_of_measuring_nothing() -> None:
    """Too-short documents must not silently yield an empty evaluation set."""
    short = CorpusDocuments(
        documents=["tiny"] * 10,
        corpus="my-org/tiny-docs",
        config="default",
        source_split="train",
        endpoint="x",
        text_column="text",
    )
    with pytest.raises(CorpusError, match="no usable documents"):
        build_quality_dataset(
            corpus=short, data_config=_config(), tokenizer_id="tok", min_chars=400
        )


def test_error_message_names_the_corpus_and_the_threshold() -> None:
    short = CorpusDocuments(
        documents=["tiny"], corpus="acme/reports", config="", source_split="train",
        endpoint="x", text_column="content",
    )
    with pytest.raises(CorpusError) as excinfo:
        build_quality_dataset(
            corpus=short, data_config=_config(), tokenizer_id="tok", min_chars=999
        )
    message = str(excinfo.value)
    assert "acme/reports" in message
    assert "999" in message
    assert "content" in message


# --------------------------------------------------------------------------
# Cross-corpus guard inside one run directory
# --------------------------------------------------------------------------


def _write_manifest(
    run_dir: Path,
    *,
    corpus: str = "Salesforce/wikitext",
    split: str = "test",
    column: str = "text",
) -> None:
    """Persist a manifest as a completed quality run would."""
    result = build_quality_dataset(
        corpus=_corpus(corpus),
        data_config=_config(),
        tokenizer_id="tok",
    )
    spec = CorpusSpec(corpus=corpus, config="default", source_split=split,
                      text_column=column)
    (run_dir / "data-manifest.json").write_text(
        json.dumps(
            {
                "corpus": spec.corpus,
                "corpus_config": spec.config,
                "corpus_split": spec.source_split,
                "corpus_text_column": spec.text_column,
                "corpus_fingerprint": "fp",
                "dataset": result.manifest.model_dump(mode="json"),
                "texts": {s.record.sample_id: s.text for s in result.samples},
            }
        ),
        encoding="utf-8",
    )


def test_same_corpus_reuses_the_stored_manifest(tmp_path: Path) -> None:
    from quantassay.gating import prepare_quality_data

    _write_manifest(tmp_path, corpus="my-org/my-data")
    spec = CorpusSpec(corpus="my-org/my-data", config="default",
                      source_split="test", text_column="text")
    result = prepare_quality_data(
        tmp_path, spec=spec, calibration_samples=2, documents=5, max_tokens=512
    )
    assert result.manifest.samples  # reused, not re-fetched


def test_switching_corpus_in_one_run_dir_is_refused(tmp_path: Path) -> None:
    """Mixing corpora under one run id would make the comparison meaningless."""
    from quantassay.gating import prepare_quality_data

    _write_manifest(tmp_path, corpus="my-org/my-data")
    other = CorpusSpec(corpus="other-org/other-data", config="default",
                       source_split="test", text_column="text")
    with pytest.raises(ProbeError, match="already holds quality data"):
        prepare_quality_data(
            tmp_path, spec=other, calibration_samples=2, documents=5, max_tokens=512
        )


def test_changing_only_the_text_column_is_also_refused(tmp_path: Path) -> None:
    """A different column is different content, so it is a different corpus."""
    from quantassay.gating import prepare_quality_data

    _write_manifest(tmp_path, corpus="my-org/my-data")
    other = CorpusSpec(corpus="my-org/my-data", config="default",
                       source_split="test", text_column="body")
    with pytest.raises(ProbeError, match="already holds quality data"):
        prepare_quality_data(
            tmp_path, spec=other, calibration_samples=2, documents=5, max_tokens=512
        )


def _strip_new_identity_fields(run_dir: Path) -> None:
    """Simulate a manifest written before these fields existed."""
    path = run_dir / "data-manifest.json"
    legacy = json.loads(path.read_text(encoding="utf-8"))
    legacy.pop("corpus_split", None)
    legacy.pop("corpus_text_column", None)
    path.write_text(json.dumps(legacy), encoding="utf-8")


def test_quality_is_attached_for_the_selected_method(tmp_path: Path) -> None:
    """The attach step must read the *selected* method's quality summary.

    Real bug this pins: the candidate filename was hardcoded to
    ``quality-gptq.json``. An AWQ run measured perplexity successfully, then
    reported ``quality not evaluated`` because the attach step looked for a file
    that did not exist — success message, missing evidence, from a hardcoded
    name rather than a missing measurement.
    """
    import json

    from quantassay.gating import ProbeError as _ProbeError
    from quantassay.gating import _attach_quality_to_report

    run = tmp_path
    gptq_baseline = {
        "side": "bf16", "status": "measured", "documents_total": 2,
        "documents_scored": 2, "valid_tokens": 10, "ppl": 30.0,
        "per_document": [[10.0, 5], [11.0, 5]],
    }
    awq_candidate = {
        "side": "awq", "status": "measured", "documents_total": 2,
        "documents_scored": 2, "valid_tokens": 10, "ppl": 38.0,
        "per_document": [[13.0, 5], [12.0, 5]],
    }
    (run / "quality-bf16.json").write_text(json.dumps(gptq_baseline), encoding="utf-8")
    (run / "quality-awq.json").write_text(json.dumps(awq_candidate), encoding="utf-8")

    # No regressions.json: the function returns early without touching anything.
    _attach_quality_to_report(run, "workload", method="awq")

    # With a report present, asking for a method that has no summary must fail
    # loudly rather than silently reporting "not evaluated".
    (run / "regressions.json").write_text(
        json.dumps({"run_id": "x", "comparable": True, "regressions": {},
                    "quality_status": "not_evaluated"}),
        encoding="utf-8",
    )
    with pytest.raises(_ProbeError, match="quality-missing.json"):
        _attach_quality_to_report(run, "workload", method="missing")

    # And the real path works: AWQ's summary is found and folded in.
    _attach_quality_to_report(run, "workload", method="awq")
    folded = json.loads((run / "regressions.json").read_text(encoding="utf-8"))
    assert folded["quality_status"] == "measured"
    assert folded["quality"]["comparable"] is True


def test_manifest_without_the_new_fields_still_resumes(tmp_path: Path) -> None:
    """A run created before corpus_split/text_column existed must still resume.

    Those fields were absent, so such a manifest records None for them.
    Comparing None against the modern default would refuse every pre-existing
    run — silently breaking resume, which is the behaviour the journal exists to
    provide. A missing field must mean "the default in force then".
    """
    from quantassay.gating import prepare_quality_data

    _write_manifest(tmp_path, corpus="my-org/my-data", split="test", column="text")
    _strip_new_identity_fields(tmp_path)

    # The defaults match what an unconfigured legacy run actually used.
    spec = CorpusSpec(corpus="my-org/my-data", config="default",
                      source_split="test", text_column="text")
    result = prepare_quality_data(
        tmp_path, spec=spec, calibration_samples=2, documents=5, max_tokens=512
    )
    assert result.manifest.samples  # resumed, not refused


def test_legacy_manifest_still_refuses_a_genuinely_different_corpus(
    tmp_path: Path,
) -> None:
    """Backward compatibility must not disarm the guard."""
    from quantassay.gating import prepare_quality_data

    _write_manifest(tmp_path, corpus="my-org/my-data", split="test", column="text")
    _strip_new_identity_fields(tmp_path)

    other = CorpusSpec(corpus="other-org/other-data", config="default",
                       source_split="test", text_column="text")
    with pytest.raises(ProbeError, match="already holds quality data"):
        prepare_quality_data(
            tmp_path, spec=other, calibration_samples=2, documents=5, max_tokens=512
        )


def test_legacy_manifest_refuses_a_different_split_or_column(tmp_path: Path) -> None:
    """A non-default request against a field-less manifest is still a mismatch."""
    from quantassay.gating import prepare_quality_data

    _write_manifest(tmp_path, corpus="my-org/my-data", split="test", column="text")
    _strip_new_identity_fields(tmp_path)

    different_split = CorpusSpec(corpus="my-org/my-data", config="default",
                                 source_split="train", text_column="text")
    with pytest.raises(ProbeError, match="already holds quality data"):
        prepare_quality_data(
            tmp_path, spec=different_split,
            calibration_samples=2, documents=5, max_tokens=512,
        )