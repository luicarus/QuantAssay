"""Persisted contracts for serving, quality, datasets and reports.

Unavailable metrics carry a status and reason. Evidence formats and comparison
fingerprints are shared by native and custom SGLang execution paths.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


SCHEMA_VERSION = "0.1.0"
MODEL_ID = "Qwen/Qwen3-0.6B"


class ThinkingMode(str, Enum):
    """Qwen3 thinking mode is part of the protocol, not a preference.

    Results from the two modes are separate experiments and must never be merged
    into one comparison (full-prd.md §2).
    """

    DISABLED = "disabled"
    ENABLED = "enabled"


class MetricStatus(str, Enum):
    """Why a metric may have no usable value.

    ``UNAVAILABLE`` must never be reported as ``0``.
    """

    OK = "ok"
    UNAVAILABLE = "unavailable"
    NOT_EVALUATED = "not_evaluated"
    NOT_SUPPORTED = "not_supported"
    ZERO_DENOMINATOR = "zero_denominator"
    ZERO_ERROR = "zero_error"
    ZERO_SIGNAL = "zero_signal"
    NAN = "nan"
    INF = "inf"


class EvidenceStatus(str, Enum):
    """Recommendation evidence ladder.

    An error ranking alone may only ever produce ``HYPOTHESIS``; only a completed
    intervention retest may produce ``VALIDATED`` (full-prd.md §6).
    """

    HYPOTHESIS = "hypothesis"
    VALIDATED = "validated"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


class RunMode(str, Enum):
    """How a model actually ran. Diagnostic modes do not prove deployment gains."""

    PACKED = "packed"
    DEQUANTIZED = "dequantized"
    FAKE_QUANT = "fake_quant"
    FP16 = "fp16"
    BF16 = "bf16"


class DataSplit(str, Enum):
    """Calibration / development / final holdout must never overlap."""

    CALIBRATION = "calibration"
    DEV = "dev"
    FINAL = "final"


class MetricDirection(str, Enum):
    LOWER_IS_BETTER = "lower_is_better"
    HIGHER_IS_BETTER = "higher_is_better"
    NEUTRAL = "neutral"


def canonical_json(value: Any) -> str:
    """Serialize deterministically so fingerprints are stable across runs."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_of(value: Any) -> str:
    """Hash any JSON-serializable value; used for config and data fingerprints."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    """Hash a single text payload (used for normalized-text duplicate detection)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ReportPaths(BaseModel):
    """Rendered report locations for a finished run."""

    model_config = ConfigDict(extra="forbid")

    markdown_path: str | None = None
    html_path: str | None = None
    generated_at: datetime = Field(default_factory=_utcnow)


class ServingMetric(str, Enum):
    """The four MVP serving metrics, plus explicitly-named throughput variants.

    ``TOKENS_PER_SEC`` is *output* token throughput. Request throughput and
    total-token throughput are separate names because conflating them is the
    most common way a serving comparison becomes wrong.
    """

    TTFT = "ttft"
    ITL = "itl"
    TPOT = "tpot"
    TOKENS_PER_SEC = "output_tokens_per_sec"
    REQUESTS_PER_SEC = "requests_per_sec"
    TOTAL_TOKENS_PER_SEC = "total_tokens_per_sec"
    E2E_LATENCY = "e2e_latency_ms"


SERVING_METRIC_DIRECTION: dict[ServingMetric, MetricDirection] = {
    ServingMetric.TTFT: MetricDirection.LOWER_IS_BETTER,
    ServingMetric.ITL: MetricDirection.LOWER_IS_BETTER,
    ServingMetric.TPOT: MetricDirection.LOWER_IS_BETTER,
    ServingMetric.E2E_LATENCY: MetricDirection.LOWER_IS_BETTER,
    ServingMetric.TOKENS_PER_SEC: MetricDirection.HIGHER_IS_BETTER,
    ServingMetric.REQUESTS_PER_SEC: MetricDirection.HIGHER_IS_BETTER,
    ServingMetric.TOTAL_TOKENS_PER_SEC: MetricDirection.HIGHER_IS_BETTER,
}


SERVING_METRIC_UNITS: dict[ServingMetric, str] = {
    ServingMetric.TTFT: "ms",
    ServingMetric.ITL: "ms",
    ServingMetric.TPOT: "ms/token",
    ServingMetric.E2E_LATENCY: "ms",
    ServingMetric.TOKENS_PER_SEC: "output_tokens/s",
    ServingMetric.REQUESTS_PER_SEC: "requests/s",
    ServingMetric.TOTAL_TOKENS_PER_SEC: "total_tokens/s",
}


MVP_SERVING_METRICS: tuple[ServingMetric, ...] = (
    ServingMetric.TTFT,
    ServingMetric.TPOT,
    ServingMetric.ITL,
    ServingMetric.TOKENS_PER_SEC,
)


REQUEST_FAILURE_REASONS: dict[str, str] = {
    "no_first_token": "stream ended before any output token arrived",
    "timeout": "request exceeded the configured timeout",
    "http_error": "server returned an error status",
    "connection_error": "connection failed or was reset",
    "malformed_stream": "stream could not be parsed",
    "truncated": "generation stopped at the token cap, not at EOS",
    "empty_output": "server returned an empty completion",
}


class RequestStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"
    TIMEOUT = "timeout"


class SideStatus(str, Enum):
    """Whether a served side produced usable evidence at all."""

    OK = "ok"
    FAILED = "failed"
    NOT_SERVED = "not_served"
    NOT_EVALUATED = "not_evaluated"


class WorkloadSpec(BaseModel):
    """One immutable serving workload. Both sides must share it exactly.

    Timing window, cache policy and warmup are part of the workload because
    changing any of them changes what the numbers mean (full-prd.md §5).
    """

    model_config = ConfigDict(extra="forbid")

    workload_id: str
    request_set_id: str
    request_hashes: list[str] = Field(default_factory=list)
    thinking_mode: ThinkingMode = ThinkingMode.DISABLED
    input_tokens_target: int = Field(default=128, gt=0)
    max_new_tokens: int = Field(default=64, gt=0)
    temperature: float = Field(default=0.0, ge=0.0)
    concurrency: int = Field(default=1, ge=1)
    request_rate: float | None = Field(default=None, gt=0)
    warmup_requests: int = Field(default=3, ge=0)
    timed_requests: int = Field(default=20, ge=1)
    cache_policy: str = "disabled"
    timeout_seconds: float = Field(default=120.0, gt=0)
    retry_policy: str = "none"

    @model_validator(mode="after")
    def _workload_is_measurable(self) -> "WorkloadSpec":
        if self.timed_requests < 1:
            raise ValueError("a workload must measure at least one request")
        if not self.request_hashes:
            raise ValueError(
                "workload must identify its requests by hash, otherwise the two "
                "sides cannot be proven to have seen the same input"
            )
        return self

    def workload_fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        payload.pop("request_hashes", None)
        payload["request_set_digest"] = sha256_of(self.request_hashes)
        return sha256_of(payload)


class RequestRecord(BaseModel):
    """Per-request evidence. Latency fields are absent when the request failed."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    status: RequestStatus = RequestStatus.OK
    failure_reason: str | None = None
    ttft_ms: float | None = Field(default=None, ge=0)
    e2e_latency_ms: float | None = Field(default=None, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    itl_ms: list[float] = Field(default_factory=list)
    finish_reason: str | None = None
    stopped_on_eos: bool | None = None
    truncated: bool | None = None
    usage_available: bool = True
    started_at: str | None = None

    @model_validator(mode="after")
    def _failures_carry_no_latency(self) -> "RequestRecord":
        """A failed request must not masquerade as a very fast success."""
        if self.status is not RequestStatus.OK and self.ttft_ms is not None:
            raise ValueError(
                f"request {self.request_id!r}: failed requests must not record TTFT "
                "(a failure is not a fast response)"
            )
        if self.status is not RequestStatus.OK and not self.failure_reason:
            raise ValueError(f"request {self.request_id!r}: failure requires a reason")
        if self.status is RequestStatus.OK and self.ttft_ms is None:
            raise ValueError(
                f"request {self.request_id!r}: a successful request requires a TTFT"
            )
        return self

    @property
    def tpot_ms(self) -> float | None:
        """``(E2E - TTFT) / (output_tokens - 1)``; unavailable below 2 tokens."""
        if self.ttft_ms is None or self.e2e_latency_ms is None:
            return None
        if self.output_tokens < 2:
            return None
        return (self.e2e_latency_ms - self.ttft_ms) / (self.output_tokens - 1)


class ServingSummary(BaseModel):
    """Aggregated serving result for one side on one workload."""

    model_config = ConfigDict(extra="forbid")

    side: str
    status: SideStatus = SideStatus.OK
    reason: str | None = None
    workload_id: str | None = None
    workload_fingerprint: str | None = None
    sglang_version: str | None = None
    run_mode: RunMode | None = None
    quant_kernel: str | None = None
    requests_total: int = Field(default=0, ge=0)
    requests_ok: int = Field(default=0, ge=0)
    requests_failed: int = Field(default=0, ge=0)
    requests_timeout: int = Field(default=0, ge=0)
    warmup_requests: int = Field(default=0, ge=0)
    wall_seconds: float | None = Field(default=None, ge=0)
    input_tokens_total: int = Field(default=0, ge=0)
    output_tokens_total: int = Field(default=0, ge=0)
    # Output-length and termination disclosure (mvp-prd.md §5). A comparison in
    # which one side was truncated by max_tokens while the other stopped on EOS
    # is measuring two different decoding regimes, and the report must say so
    # rather than present the ratio as a quantization effect.
    output_tokens_min: int | None = Field(default=None, ge=0)
    output_tokens_p50: float | None = Field(default=None, ge=0)
    output_tokens_max: int | None = Field(default=None, ge=0)
    finish_reason_counts: dict[str, int] = Field(default_factory=dict)
    truncated_requests: int = Field(default=0, ge=0)
    stopped_on_eos: int = Field(default=0, ge=0)
    metrics: dict[str, MetricRecord] = Field(default_factory=dict)
    records_path: str | None = None

    @model_validator(mode="after")
    def _status_requires_reason_when_not_ok(self) -> "ServingSummary":
        if self.status is not SideStatus.OK and not self.reason:
            raise ValueError(f"side {self.side!r}: non-ok status requires a reason")
        return self

    @property
    def success_rate(self) -> float | None:
        if self.requests_total == 0:
            return None
        return self.requests_ok / self.requests_total

    def metric(self, name: ServingMetric) -> MetricRecord:
        return self.metrics.get(
            name.value,
            MetricRecord.unavailable(name.value, "metric not produced by this side"),
        )


class ComparabilityIssue(BaseModel):
    """Why two sides may not be turned into a percentage."""

    model_config = ConfigDict(extra="forbid")

    field: str
    base_value: str | None = None
    candidate_value: str | None = None
    reason: str


class ServingRegression(BaseModel):
    """Per-metric comparison. Percentages appear only when genuinely comparable."""

    model_config = ConfigDict(extra="forbid")

    metric: ServingMetric
    base_value: float | None = None
    candidate_value: float | None = None
    absolute_delta: float | None = None
    relative_change: float | None = None
    relative_change_percent: float | None = None
    status: MetricStatus = MetricStatus.OK
    reason: str | None = None

    @model_validator(mode="after")
    def _percentage_requires_comparability(self) -> "ServingRegression":
        if self.relative_change_percent is not None and self.status is not MetricStatus.OK:
            raise ValueError(
                f"{self.metric.value}: a percentage must not be reported while "
                f"status is {self.status.value}"
            )
        if self.status is not MetricStatus.OK and not self.reason:
            raise ValueError(f"{self.metric.value}: non-ok status requires a reason")
        return self


class RegressionReport(BaseModel):
    """Regression analysis output, with comparability decided before numbers."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    run_id: str = ""
    comparable: bool = False
    incomparable_reasons: list[ComparabilityIssue] = Field(default_factory=list)
    baseline: ServingSummary | None = None
    candidate: ServingSummary | None = None
    regressions: dict[str, ServingRegression] = Field(default_factory=dict)
    quality_status: str = "not_evaluated"
    quality: "QualityComparison | None" = None
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _incomparable_reports_carry_no_verdict(self) -> "RegressionReport":
        if not self.comparable and not self.incomparable_reasons:
            raise ValueError(
                "an incomparable report must state why, otherwise the absence of "
                "a verdict looks like a finding"
            )
        if self.comparable and self.incomparable_reasons:
            raise ValueError("a comparable report must not carry incomparable reasons")
        return self

    @property
    def quality_not_evaluated(self) -> bool:
        """True while quality has not been measured.

        The serving MVP reports ``not_evaluated``; the full PRD's quality layer
        replaces it with a measured status. Either way the report must never
        render an unmeasured quality difference as "0 pp".
        """
        return self.quality_status == "not_evaluated"


class QualityStatus(str, Enum):
    """How much the quality comparison can actually claim."""

    NOT_EVALUATED = "not_evaluated"
    MEASURED = "measured"
    MEASURED_NOT_COMPARABLE = "measured_not_comparable"
    UNAVAILABLE = "unavailable"


class DocumentScore(BaseModel):
    """Per-document scoring evidence. NLL sums are kept, never per-doc PPLs."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    text_hash: str
    split: DataSplit
    #: Number of prompt tokens the server actually tokenized.
    prompt_tokens: int = Field(default=0, ge=0)
    #: Target tokens that had a usable logprob (prompt_tokens - 1 per window).
    valid_tokens: int = Field(default=0, ge=0)
    #: Sum of per-token negative log-likelihood; PPL is derived from this + count.
    nll_sum: float | None = None
    windows: int = Field(default=0, ge=0)
    skipped_windows: int = Field(default=0, ge=0)
    truncated_by_context: bool = False
    failure_reason: str | None = None


class QualitySummary(BaseModel):
    """One side's quality result. PPL is derived from aggregated NLL only."""

    model_config = ConfigDict(extra="forbid")

    side: str
    status: QualityStatus = QualityStatus.UNAVAILABLE
    reason: str | None = None
    split: DataSplit = DataSplit.DEV
    documents_total: int = Field(default=0, ge=0)
    documents_scored: int = Field(default=0, ge=0)
    documents_skipped: int = Field(default=0, ge=0)
    valid_tokens: int = Field(default=0, ge=0)
    nll_sum: float | None = None
    #: exp(nll_sum / valid_tokens) over all valid target tokens.
    ppl: float | None = None
    #: Per-document (nll_sum, valid_tokens) for paired resampling.
    per_document: list[tuple[float, int]] = Field(default_factory=list)
    context_length: int | None = None
    prompt_mode: str | None = None
    scoring_endpoint: str | None = None
    records_path: str | None = None
    #: Identity of the scored text (corpus pin + slicing + split hashes).
    #: Recorded so reuse cannot silently compare different documents.
    dataset_fingerprint: str | None = None

    @model_validator(mode="after")
    def _ppl_matches_its_inputs(self) -> "QualitySummary":
        if self.ppl is not None and self.valid_tokens <= 0:
            raise ValueError("PPL requires a positive valid-token count")
        return self


class QualityComparison(BaseModel):
    """Paired quality comparison. A CI spanning zero cannot claim a direction."""

    model_config = ConfigDict(extra="forbid")

    comparable: bool = False
    incomparable_reasons: list[ComparabilityIssue] = Field(default_factory=list)
    baseline: QualitySummary | None = None
    candidate: QualitySummary | None = None
    #: (ppl_candidate / ppl_baseline) - 1; positive means the candidate is worse.
    ppl_relative_change: float | None = None
    ppl_delta: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    ci_confidence: float | None = None
    ci_iterations: int = 0
    paired_documents: int = Field(default=0, ge=0)
    ci_crosses_zero: bool | None = None
    directional_claim: str | None = None
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _incomparable_carries_no_numbers(self) -> "QualityComparison":
        if not self.comparable:
            if self.ppl_relative_change is not None:
                raise ValueError(
                    "an incomparable quality comparison must not report a change"
                )
            if not self.incomparable_reasons:
                raise ValueError(
                    "an incomparable quality comparison must state why"
                )
        return self


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = "wikitext-2"
    revision: str | None = None
    language: list[str] = Field(default_factory=lambda: ["en"])
    calibration_samples: int = Field(default=64, ge=0)
    dev_ppl_documents: int = Field(default=128, ge=0)
    final_ppl_documents: int = Field(default=128, ge=0)
    max_tokens: int = Field(default=512, gt=0)
    smoke_max_tokens: int = Field(default=128, gt=0)


class SampleRecord(BaseModel):
    """One fixed sample. `sample_id` and `text_hash` are both checked for duplicates."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    text_hash: str
    split: DataSplit
    token_count: int = Field(ge=0)
    source_document_id: str | None = None
    truncated: bool = False


class DatasetManifest(BaseModel):
    """Fixed splits, their fingerprints and duplicate-check results."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    dataset: DataConfig
    tokenizer_id: str
    samples: list[SampleRecord] = Field(default_factory=list)
    split_hashes: dict[str, str] = Field(default_factory=dict)
    duplicate_sample_ids: list[str] = Field(default_factory=list)
    duplicate_text_hashes: list[str] = Field(default_factory=list)
    cross_split_text_hashes: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def ids_for(self, split: DataSplit) -> list[str]:
        return [s.sample_id for s in self.samples if s.split is split]

    def has_leakage(self) -> bool:
        """Cross-split text reuse is a blocking condition, not a warning."""
        return bool(self.cross_split_text_hashes or self.duplicate_sample_ids)

    def manifest_fingerprint(self) -> str:
        return sha256_of(self.model_dump(mode="json"))


class MetricRecord(BaseModel):
    """A single measured value. No value -> a status explaining why."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float | None = None
    unit: str = ""
    direction: MetricDirection = MetricDirection.LOWER_IS_BETTER
    scope: str = "overall"
    sample_count: int = Field(default=0, ge=0)
    token_count: int = Field(default=0, ge=0)
    status: MetricStatus = MetricStatus.OK
    reason: str | None = None
    run_id: str | None = None

    @model_validator(mode="after")
    def _value_matches_status(self) -> "MetricRecord":
        if self.status is MetricStatus.OK:
            if self.value is None:
                raise ValueError(f"metric {self.name!r}: status=ok requires a value")
        elif self.value is not None:
            raise ValueError(
                f"metric {self.name!r}: status={self.status.value} must not carry a value "
                "(missing metrics must never be encoded as 0)"
            )
        if self.status is not MetricStatus.OK and not self.reason:
            raise ValueError(f"metric {self.name!r}: non-ok status requires a reason")
        return self

    @classmethod
    def unavailable(cls, name: str, reason: str, **kw: Any) -> "MetricRecord":
        return cls(name=name, value=None, status=MetricStatus.UNAVAILABLE, reason=reason, **kw)


class Recommendation(BaseModel):
    """Advisory output. May only cite evidence that already exists.

    Field set follows mvp-prd.md §5: rule_id, status, evidence_refs,
    workload_scope, observation, recommendation, limitations, next_validation.
    """

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    rule_version: str = "1"
    status: EvidenceStatus = EvidenceStatus.HYPOTHESIS
    evidence_refs: list[str] = Field(default_factory=list)
    workload_scope: str | None = None
    observation: str = ""
    recommendation: str = ""
    limitations: list[str] = Field(default_factory=list)
    next_validation: str | None = None
    config_patch: dict[str, Any] = Field(default_factory=dict)
    validation_run_id: str | None = None
    text: str = ""

    @model_validator(mode="after")
    def _validated_requires_run(self) -> "Recommendation":
        if self.status is EvidenceStatus.VALIDATED and not self.validation_run_id:
            raise ValueError(
                f"rule {self.rule_id!r}: VALIDATED requires validation_run_id"
            )
        if self.status is EvidenceStatus.VALIDATED and not self.evidence_refs:
            raise ValueError(f"rule {self.rule_id!r}: VALIDATED requires evidence_refs")
        return self
