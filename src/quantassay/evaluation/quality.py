"""Score PPL from native SGLang prompt logprobs in non-overlapping windows.

The first token of each window has no target logprob. Rejected windows remain
visible in the evidence. PPL is pooled from NLL sums and valid-token counts.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Sequence

from quantassay.contracts import (
    DocumentScore,
    QualityStatus,
    QualitySummary,
    SampleRecord,
)
from quantassay.experiments.store import atomic_write_json

#: Endpoint used for scoring. Native SGLang API: it is the only one that
#: returns prompt-side logprobs (verified by probe on the locked version).
SCORING_ENDPOINT = "/generate"


class QualityError(RuntimeError):
    """Raised when scoring cannot proceed at all (e.g. server unreachable)."""


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def score_text(
    text: str,
    *,
    base_url: str,
    model_id: str,
    timeout_seconds: float = 120.0,
    poster: Callable[[str, dict[str, Any], float], dict[str, Any]] = _post_json,
) -> dict[str, Any]:
    """Score one window: return (nll_sum, valid_tokens, prompt_tokens).

    ``max_new_tokens=0`` so no decoding is paid for; scoring is prefill-only.
    """
    payload = {
        "text": text,
        "sampling_params": {"max_new_tokens": 0, "temperature": 0.0},
        "return_logprob": True,
        # 0 means "start logprobs from the beginning of the prompt".
        "logprob_start_len": 0,
        "top_logprobs_num": 0,
    }
    result = poster(f"{base_url}{SCORING_ENDPOINT}", payload, timeout_seconds)
    meta = result.get("meta_info") or {}
    entries = meta.get("input_token_logprobs")
    if entries is None:
        raise QualityError(
            "server returned no input_token_logprobs; PPL is unavailable on this path"
        )

    nll_sum = 0.0
    valid = 0
    for entry in entries:
        # entry = [logprob, token_id, ...]; the first token's logprob is null
        # because it has no context — correctly not a target.
        if not entry:
            continue
        value = entry[0]
        if value is None:
            continue
        value = float(value)
        if not math.isfinite(value):
            continue
        nll_sum += -value
        valid += 1

    return {
        "nll_sum": nll_sum,
        "valid_tokens": valid,
        "prompt_tokens": int(meta.get("prompt_tokens") or len(entries)),
    }


def window_text(text: str, *, max_window_chars: int) -> list[str]:
    """Split a document into non-overlapping windows by character budget.

    The server enforces token limits, but tokenizing locally would need the
    tokenizer; a conservative character budget is checked against the real
    token count server-side, and a rejected window is reported rather than
    silently dropped.
    """
    if max_window_chars <= 0:
        raise ValueError("max_window_chars must be positive")
    text = text.strip()
    if not text:
        return []
    return [text[i : i + max_window_chars] for i in range(0, len(text), max_window_chars)]


def score_document(
    record: SampleRecord,
    text: str,
    *,
    base_url: str,
    model_id: str,
    max_window_chars: int,
    timeout_seconds: float = 120.0,
    poster: Callable[[str, dict[str, Any], float], dict[str, Any]] = _post_json,
) -> DocumentScore:
    """Score one document, window by window.

    A window the server rejects is recorded and skipped; the document keeps the
    windows that did work. Nothing is imputed.
    """
    windows = window_text(text, max_window_chars=max_window_chars)
    if not windows:
        return DocumentScore(
            sample_id=record.sample_id,
            text_hash=record.text_hash,
            split=record.split,
            failure_reason="empty document",
        )

    nll_sum = 0.0
    valid_tokens = 0
    prompt_tokens = 0
    done = 0
    skipped = 0
    truncated = False

    for window in windows:
        try:
            scored = score_text(
                window,
                base_url=base_url,
                model_id=model_id,
                timeout_seconds=timeout_seconds,
                poster=poster,
            )
        except (urllib.error.HTTPError, urllib.error.URLError, QualityError, TimeoutError, OSError) as exc:
            skipped += 1
            # HTTP 400 on a window is the context-length limit: mark it so the
            # report can show the budget was the binding constraint.
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 400:
                truncated = True
            continue
        nll_sum += scored["nll_sum"]
        valid_tokens += scored["valid_tokens"]
        prompt_tokens += scored["prompt_tokens"]
        done += 1

    if done == 0:
        return DocumentScore(
            sample_id=record.sample_id,
            text_hash=record.text_hash,
            split=record.split,
            # `windows` counts windows that actually produced a score; attempts
            # are visible as windows + skipped_windows. Keeping one meaning for
            # the field is what makes the two branches comparable.
            windows=0,
            skipped_windows=skipped,
            truncated_by_context=truncated,
            failure_reason="every window was rejected by the server",
        )

    return DocumentScore(
        sample_id=record.sample_id,
        text_hash=record.text_hash,
        split=record.split,
        prompt_tokens=prompt_tokens,
        valid_tokens=valid_tokens,
        nll_sum=nll_sum,
        windows=done,
        skipped_windows=skipped,
        truncated_by_context=truncated,
    )


def summarize_scores(
    scores: Sequence[DocumentScore],
    *,
    side: str,
    context_length: int | None = None,
    prompt_mode: str | None = None,
    records_path: str | None = None,
    split: Any = None,
    dataset_fingerprint: str | None = None,
) -> QualitySummary:
    """Aggregate document scores into a QualitySummary.

    Only documents that produced at least one valid target token contribute;
    PPL comes from the pooled NLL sum, never from averaging per-document PPLs.
    """
    from quantassay.contracts import DataSplit

    usable = [s for s in scores if s.valid_tokens > 0 and s.nll_sum is not None]
    skipped = [s for s in scores if s not in usable]

    if not usable:
        reasons = [s.failure_reason for s in scores if s.failure_reason]
        reason = reasons[0] if reasons else "no document produced a usable logprob"
        return QualitySummary(
            side=side,
            status=QualityStatus.UNAVAILABLE,
            reason=reason,
            split=split or (scores[0].split if scores else DataSplit.DEV),
            documents_total=len(scores),
            documents_skipped=len(skipped),
            context_length=context_length,
            prompt_mode=prompt_mode,
            scoring_endpoint=SCORING_ENDPOINT,
            records_path=records_path,
            dataset_fingerprint=dataset_fingerprint,
        )

    nll_sum = sum(s.nll_sum or 0.0 for s in usable)
    valid_tokens = sum(s.valid_tokens for s in usable)
    ppl = math.exp(nll_sum / valid_tokens) if valid_tokens else None

    return QualitySummary(
        side=side,
        status=QualityStatus.MEASURED,
        split=split or usable[0].split,
        documents_total=len(scores),
        documents_scored=len(usable),
        documents_skipped=len(skipped),
        valid_tokens=valid_tokens,
        nll_sum=nll_sum,
        ppl=ppl,
        per_document=[(s.nll_sum or 0.0, s.valid_tokens) for s in usable],
        context_length=context_length,
        prompt_mode=prompt_mode,
        scoring_endpoint=SCORING_ENDPOINT,
        records_path=records_path,
        dataset_fingerprint=dataset_fingerprint,
    )


def score_documents(
    records: Sequence[SampleRecord],
    texts: Sequence[str],
    *,
    base_url: str,
    model_id: str,
    output_dir: Path,
    side: str,
    context_length: int,
    max_window_chars: int | None = None,
    timeout_seconds: float = 120.0,
    dataset_fingerprint: str | None = None,
    poster: Callable[[str, dict[str, Any], float], dict[str, Any]] = _post_json,
) -> tuple[QualitySummary, list[DocumentScore]]:
    """Score a fixed document set and persist per-document evidence as JSONL.

    The JSONL is written incrementally so an interrupted scoring run still
    leaves usable evidence, mirroring the serving benchmark's behaviour.
    """
    if len(records) != len(texts):
        raise QualityError(
            f"records ({len(records)}) and texts ({len(texts)}) must align"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / f"quality-{side}.jsonl"

    # Budget: assume <=3 chars/token to stay clear of the context limit even for
    # text-heavy documents. The server is the final authority (a rejected window
    # is recorded, and truncation is flagged).
    budget = max_window_chars or max(1, context_length * 3)

    scores: list[DocumentScore] = []
    with records_path.open("w", encoding="utf-8") as handle:
        for record, text in zip(records, texts):
            score = score_document(
                record,
                text,
                base_url=base_url,
                model_id=model_id,
                max_window_chars=budget,
                timeout_seconds=timeout_seconds,
                poster=poster,
            )
            scores.append(score)
            handle.write(json.dumps(score.model_dump(mode="json"), sort_keys=True) + "\n")
            handle.flush()

    summary = summarize_scores(
        scores,
        side=side,
        context_length=context_length,
        prompt_mode="raw_completion",
        records_path=str(records_path),
        split=records[0].split if records else None,
        dataset_fingerprint=dataset_fingerprint,
    )
    atomic_write_json(output_dir / f"quality-{side}.json", summary.model_dump(mode="json"))
    return summary, scores
