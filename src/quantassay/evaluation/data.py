"""Dataset preparation: fixed splits, stable IDs, hashing and leakage detection.

The blocking conditions (full-prd.md §7):

* Calibration, dev and final holdout must not overlap in content.
* Duplicate IDs and duplicate normalized text are detected, not tolerated.
* If a split cannot supply the requested count, we take fewer samples and record
  that in the manifest — never backfill from another split.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from quantassay.contracts import (
    SCHEMA_VERSION,
    DataConfig,
    DatasetManifest,
    DataSplit,
    SampleRecord,
    sha256_of,
    sha256_text,
)

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Canonical form used for duplicate detection.

    NFKC-normalizes, unifies line endings and collapses whitespace so that
    formatting-only differences cannot disguise a reused sample.
    """
    unified = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = unicodedata.normalize("NFKC", unified)
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def text_hash(text: str) -> str:
    """Hash of the normalized text — the duplicate-detection key."""
    return sha256_text(normalize_text(text))


def stable_sample_id(split: DataSplit, source_document_id: str, index: int) -> str:
    """Deterministic ID so the same sample keeps its identity across runs."""
    payload = f"{split.value}|{source_document_id}|{index}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class TextDocument:
    """One source document before slicing."""

    document_id: str
    text: str


@dataclass
class PreparedSample:
    """A fixed sample plus the text it came from (kept out of the manifest)."""

    record: SampleRecord
    text: str
    calibration_text: str | None = None


@dataclass
class PrepareResult:
    manifest: DatasetManifest
    samples: list[PreparedSample] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        """Leakage or duplicates block any further comparison."""
        return not self.manifest.has_leakage()

    def texts_for(self, split: DataSplit) -> list[str]:
        return [s.text for s in self.samples if s.record.split is split]

    def calibration_texts(self) -> list[str]:
        return [
            s.calibration_text if s.calibration_text is not None else s.text
            for s in self.samples
            if s.record.split is DataSplit.CALIBRATION
        ]


def _truncate_to_tokens(text: str, max_tokens: int, *, chars_per_token: int = 4) -> tuple[str, bool]:
    """Approximate token-budget truncation for data preparation.

    A real tokenizer is applied by the GPU-side path; this keeps local split
    preparation and its tests free of heavy dependencies.
    """
    budget = max_tokens * chars_per_token
    if len(text) <= budget:
        return text, False
    return text[:budget], True


def detect_leakage(records: Sequence[SampleRecord], texts: Sequence[str]) -> tuple[
    list[str], list[str], list[str]
]:
    """Return (duplicate_ids, duplicate_hashes, cross_split_hashes)."""
    duplicate_ids: list[str] = []
    seen_ids: set[str] = set()
    for record in records:
        if record.sample_id in seen_ids:
            duplicate_ids.append(record.sample_id)
        seen_ids.add(record.sample_id)

    split_by_hash: dict[str, set[DataSplit]] = {}
    duplicate_hashes: list[str] = []
    for record in records:
        bucket = split_by_hash.setdefault(record.text_hash, set())
        if bucket:
            duplicate_hashes.append(record.text_hash)
        bucket.add(record.split)

    cross_split = sorted(h for h, splits in split_by_hash.items() if len(splits) > 1)
    return sorted(set(duplicate_ids)), sorted(set(duplicate_hashes)), cross_split


def prepare_data(
    spec_data: DataConfig,
    tokenizer_id: str,
    *,
    calibration: Iterable[TextDocument],
    dev: Iterable[TextDocument],
    final: Iterable[TextDocument],
) -> PrepareResult:
    """Build fixed, non-overlapping splits and their manifest.

    Content is deduplicated *globally and in split order* (calibration, then dev,
    then final) so that a text appearing in an earlier split can never reappear
    in a later one — the holdout stays genuinely held out.
    """
    manifest_inputs = {
        "dataset": spec_data.model_dump(mode="json"),
        "tokenizer_id": tokenizer_id,
    }

    samples: list[PreparedSample] = []
    records: list[SampleRecord] = []
    notes: list[str] = []
    seen_text_hashes: dict[str, DataSplit] = {}
    split_hashes: dict[str, str] = {}

    plan: list[tuple[DataSplit, Iterable[TextDocument], int, int]] = [
        (DataSplit.CALIBRATION, calibration, spec_data.calibration_samples, spec_data.max_tokens),
        (DataSplit.DEV, dev, spec_data.dev_ppl_documents, spec_data.max_tokens),
        (DataSplit.FINAL, final, spec_data.final_ppl_documents, spec_data.max_tokens),
    ]

    for split, documents, wanted, max_tokens in plan:
        taken = 0
        skipped_duplicate = 0
        skipped_empty = 0
        split_digest = hashlib.sha256()

        for doc in documents:
            if taken >= wanted:
                break
            normalized = normalize_text(doc.text)
            if not normalized:
                skipped_empty += 1
                continue
            digest = sha256_text(normalized)
            if digest in seen_text_hashes:
                # Already used by this or an earlier split: never reuse for holdout.
                skipped_duplicate += 1
                continue

            sliced, truncated = _truncate_to_tokens(normalized, max_tokens)
            sample_id = stable_sample_id(split, doc.document_id, taken)
            record = SampleRecord(
                sample_id=sample_id,
                text_hash=digest,
                split=split,
                token_count=len(sliced) // 4,
                source_document_id=doc.document_id,
                truncated=truncated,
            )
            records.append(record)
            samples.append(PreparedSample(record=record, text=sliced))
            seen_text_hashes[digest] = split
            split_digest.update(f"{sample_id}:{digest}\n".encode("utf-8"))
            taken += 1

        split_hashes[split.value] = split_digest.hexdigest()

        if taken < wanted:
            # Record the shortfall; never backfill from another split.
            notes.append(
                f"split {split.value}: requested {wanted}, used {taken} "
                f"(short by {wanted - taken})"
            )
        if skipped_duplicate:
            notes.append(
                f"split {split.value}: skipped {skipped_duplicate} text(s) already used "
                "by this or an earlier split"
            )
        if skipped_empty:
            notes.append(f"split {split.value}: skipped {skipped_empty} empty document(s)")

    duplicate_ids, duplicate_hashes, cross_split = detect_leakage(records, [s.text for s in samples])

    manifest = DatasetManifest(
        schema_version=SCHEMA_VERSION,
        dataset=spec_data,
        tokenizer_id=tokenizer_id,
        samples=records,
        split_hashes=split_hashes,
        duplicate_sample_ids=duplicate_ids,
        duplicate_text_hashes=duplicate_hashes,
        cross_split_text_hashes=cross_split,
        notes=notes,
    )
    # Bind the manifest to its inputs so a data change is detectable.
    manifest.notes.append(f"inputs_fingerprint={sha256_of(manifest_inputs)}")

    return PrepareResult(manifest=manifest, samples=samples)
