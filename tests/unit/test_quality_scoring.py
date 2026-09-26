"""Quality (PPL) scoring contracts — the full PRD's quality layer.

These are CPU contracts for the serving-path PPL scorer. The rules they lock in
are the ones the PRD calls non-negotiable (full-prd.md §4):

* PPL is ``exp(sum_nll / valid_tokens)`` pooled over all valid target tokens;
  per-window or per-document PPLs are never averaged;
* the first token of a window has no context and therefore no target — it must
  be excluded, not counted as zero loss;
* a window the server rejects is recorded and skipped, never imputed;
* a Transformers-side PPL is never substituted for a serving result.
"""

from __future__ import annotations

import json
import math
import sys
import urllib.error
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from quantassay.contracts import (  # noqa: E402
    DataSplit,
    QualityStatus,
    SampleRecord,
)
from quantassay.evaluation.quality import (  # noqa: E402
    QualityError,
    score_document,
    score_documents,
    score_text,
    summarize_scores,
    window_text,
)


def _record(sample_id: str = "s1", split: DataSplit = DataSplit.DEV) -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        text_hash="hash-" + sample_id,
        split=split,
        token_count=10,
    )


def _entry(logprob: float | None) -> list:
    """One ``input_token_logprobs`` entry: [logprob, token_id, extra]."""
    return [logprob, 1234, None]


class FakePoster:
    """Serves canned ``/generate`` responses and records the requests."""

    def __init__(self, *, logprob_per_token: float = -1.0, n_tokens: int = 5,
                 fail_windows: set[int] | None = None, http_400: bool = False) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.logprob = logprob_per_token
        self.n_tokens = n_tokens
        self.fail_windows = fail_windows or set()
        self.http_400 = http_400

    def __call__(self, url: str, payload: dict, timeout: float) -> dict:
        index = len(self.calls)
        self.calls.append((url, payload))
        if self.http_400 or index in self.fail_windows:
            raise urllib.error.HTTPError(url, 400, "Bad Request", {}, None)
        # First token has no context: real servers return null there.
        entries = [_entry(None)] + [_entry(self.logprob)] * (self.n_tokens - 1)
        return {
            "text": payload["text"],
            "meta_info": {
                "prompt_tokens": self.n_tokens,
                "input_token_logprobs": entries,
            },
        }


# --------------------------------------------------------------------------
# Scoring mechanics
# --------------------------------------------------------------------------


def test_score_text_excludes_the_contextless_first_token() -> None:
    """n prompt tokens yield n-1 targets, not n."""
    poster = FakePoster(logprob_per_token=-2.0, n_tokens=5)
    scored = score_text("hello world", base_url="http://x", model_id="m", poster=poster)

    assert scored["prompt_tokens"] == 5
    assert scored["valid_tokens"] == 4  # the null first entry is not a target
    assert scored["nll_sum"] == pytest.approx(2.0 * 4)


def test_score_text_requests_prefill_only_scoring() -> None:
    """max_new_tokens=0 and logprob_start_len=0 are what make this a scorer."""
    poster = FakePoster()
    score_text("t", base_url="http://x", model_id="m", poster=poster)
    _, payload = poster.calls[0]

    assert payload["sampling_params"]["max_new_tokens"] == 0
    assert payload["return_logprob"] is True
    assert payload["logprob_start_len"] == 0
    assert poster.calls[0][0].endswith("/generate")


def test_score_text_rejects_a_response_without_logprobs() -> None:
    def empty_poster(url, payload, timeout):
        return {"meta_info": {"prompt_tokens": 3}}

    with pytest.raises(QualityError, match="no input_token_logprobs"):
        score_text("t", base_url="http://x", model_id="m", poster=empty_poster)


def test_non_finite_logprobs_are_dropped_not_counted() -> None:
    def nan_poster(url, payload, timeout):
        return {
            "meta_info": {
                "prompt_tokens": 4,
                "input_token_logprobs": [
                    _entry(None), _entry(float("nan")), _entry(-1.0), _entry(-1.0),
                ],
            }
        }

    scored = score_text("t", base_url="http://x", model_id="m", poster=nan_poster)
    assert scored["valid_tokens"] == 2
    assert scored["nll_sum"] == pytest.approx(2.0)


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------


def test_window_text_splits_without_overlap_and_covers_everything() -> None:
    text = "abcdefghij"  # 10 chars
    windows = window_text(text, max_window_chars=4)
    assert windows == ["abcd", "efgh", "ij"]
    assert "".join(windows) == text  # nothing dropped, nothing duplicated


def test_window_text_handles_empty_and_whitespace() -> None:
    assert window_text("", max_window_chars=10) == []
    assert window_text("   \n  ", max_window_chars=10) == []
    with pytest.raises(ValueError):
        window_text("x", max_window_chars=0)


def test_long_document_is_scored_window_by_window() -> None:
    poster = FakePoster(logprob_per_token=-1.0, n_tokens=3)
    score = score_document(
        _record(), "x" * 100,
        base_url="http://x", model_id="m", max_window_chars=25, poster=poster,
    )
    assert score.windows == 4
    assert len(poster.calls) == 4
    # 4 windows x (3 prompt tokens - 1 contextless) = 8 valid targets
    assert score.valid_tokens == 8
    assert score.nll_sum == pytest.approx(8.0)
    assert score.failure_reason is None


def test_empty_document_is_recorded_as_a_failure_not_a_zero() -> None:
    score = score_document(
        _record(), "   ",
        base_url="http://x", model_id="m", max_window_chars=10, poster=FakePoster(),
    )
    assert score.valid_tokens == 0
    assert score.nll_sum is None
    assert score.failure_reason == "empty document"


def test_rejected_window_is_skipped_and_flagged_as_context_truncation() -> None:
    """HTTP 400 means the context budget bound us: disclose, don't impute."""
    poster = FakePoster(n_tokens=3, fail_windows={1})
    score = score_document(
        _record(), "x" * 50,
        base_url="http://x", model_id="m", max_window_chars=25, poster=poster,
    )
    assert score.windows == 1
    assert score.skipped_windows == 1
    assert score.truncated_by_context is True
    # Only the surviving window's targets count.
    assert score.valid_tokens == 2


def test_all_windows_rejected_marks_the_document_failed() -> None:
    poster = FakePoster(n_tokens=3, http_400=True)
    score = score_document(
        _record(), "x" * 50,
        base_url="http://x", model_id="m", max_window_chars=25, poster=poster,
    )
    assert score.windows == 0
    assert score.valid_tokens == 0
    assert score.nll_sum is None
    assert "every window was rejected" in score.failure_reason


# --------------------------------------------------------------------------
# Aggregation: pooled NLL only
# --------------------------------------------------------------------------


def test_ppl_is_pooled_over_tokens_not_averaged_over_documents() -> None:
    """The wrong quantity (mean of per-document PPL) must differ from ours.

    Documents with different token counts and different loss make the two
    definitions diverge; the pooled one is the correct one.
    """
    from quantassay.contracts import DocumentScore

    short = DocumentScore(
        sample_id="a", text_hash="h", split=DataSplit.DEV,
        valid_tokens=2, nll_sum=1.0,   # per-doc ppl = e^0.5
    )
    long = DocumentScore(
        sample_id="b", text_hash="h", split=DataSplit.DEV,
        valid_tokens=100, nll_sum=500.0,  # per-doc ppl = e^5
    )
    summary = summarize_scores([short, long], side="bf16")

    pooled = math.exp(501.0 / 102)
    mean_of_ppls = (math.exp(0.5) + math.exp(5.0)) / 2
    assert summary.ppl == pytest.approx(pooled)
    assert summary.ppl != pytest.approx(mean_of_ppls)
    assert summary.valid_tokens == 102


def test_summary_keeps_per_document_pairs_for_paired_resampling() -> None:
    from quantassay.contracts import DocumentScore

    scores = [
        DocumentScore(sample_id="a", text_hash="h", split=DataSplit.DEV,
                      valid_tokens=3, nll_sum=1.5),
        DocumentScore(sample_id="b", text_hash="h", split=DataSplit.DEV,
                      valid_tokens=7, nll_sum=3.5),
    ]
    summary = summarize_scores(scores, side="gptq")
    assert summary.per_document == [(1.5, 3), (3.5, 7)]
    assert summary.status is QualityStatus.MEASURED


def test_summary_with_no_usable_document_is_unavailable_with_a_reason() -> None:
    from quantassay.contracts import DocumentScore

    scores = [
        DocumentScore(sample_id="a", text_hash="h", split=DataSplit.DEV,
                      failure_reason="every window was rejected by the server"),
    ]
    summary = summarize_scores(scores, side="gptq")
    assert summary.status is QualityStatus.UNAVAILABLE
    assert summary.ppl is None
    assert "rejected" in summary.reason


def test_partial_failures_do_not_become_zero_loss_documents() -> None:
    """A failed document must be skipped, not treated as a perfect score."""
    from quantassay.contracts import DocumentScore

    ok = DocumentScore(sample_id="a", text_hash="h", split=DataSplit.DEV,
                       valid_tokens=4, nll_sum=4.0)
    bad = DocumentScore(sample_id="b", text_hash="h", split=DataSplit.DEV,
                        failure_reason="boom")
    summary = summarize_scores([ok, bad], side="bf16")

    assert summary.documents_scored == 1
    assert summary.documents_skipped == 1
    assert summary.documents_total == 2
    # PPL reflects only the scored document; the failure contributed nothing.
    assert summary.ppl == pytest.approx(math.e)


# --------------------------------------------------------------------------
# Persistence and orchestration
# --------------------------------------------------------------------------


def test_score_documents_writes_incremental_jsonl(tmp_path: Path) -> None:
    records = [_record("s1"), _record("s2")]
    texts = ["aaa", "bbb"]
    summary, scores = score_documents(
        records, texts,
        base_url="http://x", model_id="m", output_dir=tmp_path,
        side="bf16", context_length=512, max_window_chars=10,
        poster=FakePoster(n_tokens=3),
    )

    jsonl = tmp_path / "quality-bf16.jsonl"
    assert jsonl.is_file()
    lines = [json.loads(x) for x in jsonl.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(lines) == 2
    assert lines[0]["sample_id"] == "s1"
    assert summary.status is QualityStatus.MEASURED
    assert (tmp_path / "quality-bf16.json").is_file()


def test_score_documents_requires_aligned_inputs(tmp_path: Path) -> None:
    with pytest.raises(QualityError, match="must align"):
        score_documents(
            [_record()], ["a", "b"],
            base_url="http://x", model_id="m", output_dir=tmp_path,
            side="bf16", context_length=512, poster=FakePoster(),
        )


def test_summary_survives_json_roundtrip(tmp_path: Path) -> None:
    from quantassay.contracts import QualitySummary

    summary, _ = score_documents(
        [_record()], ["aaa"],
        base_url="http://x", model_id="m", output_dir=tmp_path,
        side="bf16", context_length=512, max_window_chars=10,
        poster=FakePoster(n_tokens=3),
    )
    payload = json.loads(json.dumps(summary.model_dump(mode="json")))
    reloaded = QualitySummary.model_validate(payload)
    assert reloaded.ppl == pytest.approx(summary.ppl)
    assert reloaded.per_document == summary.per_document