"""Fixed quality corpus: fetch once, slice deterministically, hash everything.

The corpus is part of the experiment fingerprint, so it must be reproducible:
the same document selection and the same split assignment on every run. The
PRD requires calibration / dev / final to be mutually disjoint and checked by
normalized-text hash (full-prd.md §7); the checks live in
:func:`quantassay.evaluation.data.prepare_data`.

Design notes:

* Documents are taken in corpus order, so the selection does not depend on
  randomness or on the order a remote loader happens to return.
* Only documents long enough to be worth scoring are candidates; a one-line
  document contributes almost no target tokens.
* The loader is injectable so the slicing logic is CPU-testable without
  downloading anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from quantassay.contracts import (
    DataConfig,
    DatasetManifest,
    DataSplit,
    sha256_of,
)
from quantassay.evaluation.data import (
    PrepareResult,
    TextDocument,
    prepare_data,
)

#: Default corpus for the quality holdout. Users are expected to substitute
#: their own domain data — see ``CorpusSpec``. The default is a general-purpose
#: English corpus so an unconfigured run still works.
DEFAULT_CORPUS = "Salesforce/wikitext"
DEFAULT_CORPUS_CONFIG = "wikitext-2-raw-v1"
DEFAULT_CORPUS_SPLIT = "test"
#: Recorded in the manifest; the mirror is what is actually reachable here.
DEFAULT_CORPUS_ENDPOINT = "https://hf-mirror.com"
#: Column holding the document text. Datasets differ (``text``, ``content``,
#: ``document``, ...), so it is configurable rather than assumed.
DEFAULT_TEXT_COLUMN = "text"


class CorpusError(RuntimeError):
    """The requested corpus could not be loaded or sliced."""


@dataclass
class CorpusSpec:
    """Which dataset to measure quality on.

    Deliberately user-supplied: the point of this tool is comparing
    quantization options on the data a user actually cares about, so the corpus
    is configuration rather than a built-in constant.

    Every field lands in the run fingerprint (via
    :func:`dataset_fingerprint`), so switching corpora can never be compared
    against an older run by accident.
    """

    corpus: str = DEFAULT_CORPUS
    config: str = DEFAULT_CORPUS_CONFIG
    source_split: str = DEFAULT_CORPUS_SPLIT
    text_column: str = DEFAULT_TEXT_COLUMN
    endpoint: str = DEFAULT_CORPUS_ENDPOINT
    revision: str | None = None
    #: Documents shorter than this are skipped when selecting holdout text.
    min_chars: int = 400

    @property
    def identity(self) -> str:
        """Human-readable identity for logs and reports."""
        parts = [self.corpus]
        if self.config and self.config != DEFAULT_CORPUS_CONFIG:
            parts.append(self.config)
        parts.append(self.source_split)
        if self.revision:
            parts.append(self.revision[:8])
        return "/".join(parts)


@dataclass
class CorpusDocuments:
    """Loaded corpus rows plus the identity of the source."""

    documents: list[str]
    corpus: str
    config: str
    source_split: str
    endpoint: str
    revision: str | None = None
    text_column: str = DEFAULT_TEXT_COLUMN


def text_loader_hf(
    corpus: str = DEFAULT_CORPUS,
    config: str = DEFAULT_CORPUS_CONFIG,
    source_split: str = DEFAULT_CORPUS_SPLIT,
    *,
    endpoint: str = DEFAULT_CORPUS_ENDPOINT,
    revision: str | None = None,
    text_column: str = DEFAULT_TEXT_COLUMN,
) -> CorpusDocuments:
    """Load corpus rows through the HF mirror.

    Imported lazily: the module must stay importable (and its slicing logic
    testable) on the Windows side, where ``datasets`` is not installed.

    ``revision`` pins the dataset commit when one is known; it is recorded in
    the fingerprint either way so an unpinned corpus is visible in the report.

    A wrong ``text_column`` is the most likely user error when pointing this at
    their own dataset, so it raises a message naming the columns that *do*
    exist instead of a bare ``KeyError``.
    """
    import os

    os.environ.setdefault("HF_ENDPOINT", endpoint)
    from datasets import load_dataset

    try:
        dataset = load_dataset(corpus, config, split=source_split, revision=revision)
    except Exception as exc:
        raise CorpusError(
            f"could not load corpus {corpus!r} (config={config!r}, "
            f"split={source_split!r}) via {endpoint}: {type(exc).__name__}: {exc}"
        ) from exc

    available = list(getattr(dataset, "column_names", []) or [])
    if text_column not in available:
        raise CorpusError(
            f"corpus {corpus!r} has no column {text_column!r}; available columns: "
            f"{available}. Pass --corpus-text-column to pick the right one."
        )

    documents: list[str] = []
    for row in dataset:
        value = row.get(text_column)
        documents.append(value if isinstance(value, str) else "")

    return CorpusDocuments(
        documents=documents,
        corpus=corpus,
        config=config,
        source_split=source_split,
        endpoint=endpoint,
        revision=revision,
        text_column=text_column,
    )


def load_corpus(spec: CorpusSpec) -> CorpusDocuments:
    """Load the corpus described by ``spec`` (single entry point for callers)."""
    return text_loader_hf(
        spec.corpus,
        spec.config,
        spec.source_split,
        endpoint=spec.endpoint,
        revision=spec.revision,
        text_column=spec.text_column,
    )


def select_documents(
    documents: Iterable[str],
    *,
    limit: int,
    min_chars: int = 400,
    id_prefix: str = "doc",
) -> list[TextDocument]:
    """Deterministically take the first ``limit`` usable documents.

    Order is the corpus order, so the selection is stable across runs and does
    not depend on a loader's iteration order changing.

    ``id_prefix`` is derived from the corpus name, not hardcoded: a sample id
    like ``wiki-000123`` would be actively misleading in a run that measured a
    user's own dataset.
    """
    selected: list[TextDocument] = []
    for index, raw in enumerate(documents):
        text = (raw or "").strip()
        if len(text) < min_chars:
            continue
        selected.append(TextDocument(document_id=f"{id_prefix}-{index:06d}", text=text))
        if len(selected) >= limit:
            break
    return selected


def corpus_id_prefix(corpus: CorpusDocuments) -> str:
    """Short, filesystem- and report-safe prefix identifying the corpus."""
    name = corpus.corpus.rsplit("/", 1)[-1]
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()
    return cleaned or "doc"


def build_quality_dataset(
    *,
    corpus: CorpusDocuments,
    data_config: DataConfig,
    tokenizer_id: str,
    min_chars: int = 400,
) -> PrepareResult:
    """Slice the corpus into calibration / dev / final splits.

    The three splits are drawn from disjoint regions of the corpus (not the
    first N rows each) so that dev and final cannot accidentally share text;
    ``prepare_data`` then verifies disjointness by normalized-text hash.
    """
    needed = (
        data_config.calibration_samples
        + data_config.dev_ppl_documents
        + data_config.final_ppl_documents
    )
    pool = select_documents(
        corpus.documents,
        limit=needed,
        min_chars=min_chars,
        id_prefix=corpus_id_prefix(corpus),
    )
    if not pool:
        raise CorpusError(
            f"no usable documents in {corpus.corpus!r} split {corpus.source_split!r}: "
            f"every row was shorter than {min_chars} characters (or the column "
            f"{corpus.text_column!r} is empty). Lower the minimum or choose another "
            "corpus — an empty holdout cannot measure quality."
        )

    calibration_count = data_config.calibration_samples
    dev_count = data_config.dev_ppl_documents
    calibration = pool[:calibration_count]
    dev = pool[calibration_count : calibration_count + dev_count]
    final = pool[calibration_count + dev_count :]

    return prepare_data(
        data_config,
        tokenizer_id,
        calibration=calibration,
        dev=dev,
        final=final,
    )


def dataset_fingerprint(manifest: DatasetManifest, corpus: CorpusDocuments) -> str:
    """Identity of the measured data: corpus, pin, slicing and split hashes.

    ``revision`` is part of the identity on purpose: a dataset re-published
    under the same name would otherwise keep the same fingerprint and the
    change in measured text could not be detected.
    """
    return sha256_of(
        {
            "corpus": corpus.corpus,
            "config": corpus.config,
            "source_split": corpus.source_split,
            "endpoint": corpus.endpoint,
            "revision": corpus.revision,
            # The text column decides *which content* is measured, so swapping it
            # is a data change like any other.
            "text_column": corpus.text_column,
            "split_hashes": manifest.split_hashes,
            "sample_count": len(manifest.samples),
            "duplicate_text_hashes": manifest.duplicate_text_hashes,
            "cross_split_text_hashes": manifest.cross_split_text_hashes,
        }
    )


def holdout_records(manifest: DatasetManifest, split: DataSplit) -> list:
    """The records of one split, in the order they were selected."""
    return [record for record in manifest.samples if record.split is split]
