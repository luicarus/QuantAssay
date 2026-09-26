"""Core data contracts shared by all quantassay modules.

Every type that is persisted or passed across a module boundary lives here.
Modules must not define competing versions of these fields (see full-prd.md §3
and mvp-prd.md §3).

Design rules enforced by this module:

* Missing metrics carry an explicit ``status``/``reason``; they are never
  encoded as ``0``.
* Anything that changes an experiment's meaning changes its fingerprint.
* Status enums are closed sets so evidence levels cannot be silently upgraded.
* The MVP model identity is fixed and validated, never defaulted into drift.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "0.1.0"

#: The MVP fixes this model. Serving regression is only meaningful when both
#: sides descend from the same original checkpoint (mvp-prd.md §2).
#: 0.6B is the minimum scale that completes the full closed loop on this
#: machine's 4 GB GPU; it is a verification vehicle, not a production scale.
MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION_PATTERN = r"^[0-9a-f]{40}$"


class ThinkingMode(str, Enum):
    """Qwen3 thinking mode is part of the protocol, not a preference.

    Results from the two modes are separate experiments and must never be merged
    into one comparison (full-prd.md §2).
    """

    DISABLED = "disabled"
    ENABLED = "enabled"


# --------------------------------------------------------------------------
# Enums: closed sets that encode the project's evidence discipline
# --------------------------------------------------------------------------


class StageName(str, Enum):
    """Ordered pipeline stages of a `run` (mvp-prd.md §3)."""

    DATA = "data"
    BASELINE = "baseline"
    QUANTIZE = "quantize"
    EVALUATE = "evaluate"
    TRACE = "trace"
    INTERVENTION = "intervention"
    ADVISE = "advise"
    FINAL_EVAL = "final_eval"
    REPORT = "report"


STAGE_ORDER: tuple[StageName, ...] = (
    StageName.DATA,
    StageName.BASELINE,
    StageName.QUANTIZE,
    StageName.EVALUATE,
    StageName.TRACE,
    StageName.INTERVENTION,
    StageName.ADVISE,
    StageName.FINAL_EVAL,
    StageName.REPORT,
)


class StageStatus(str, Enum):
    """`exists on disk` is not `succeeded`; recovery reads this, not the filesystem."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


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


# --------------------------------------------------------------------------
# Fingerprinting
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Serving evaluation (SGLang path) - mvp-prd.md §4
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Serving evaluation (SGLang path) — mvp-prd.md §4
# --------------------------------------------------------------------------




# --------------------------------------------------------------------------
# Report paths
# --------------------------------------------------------------------------


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


#: Direction of improvement. Lower latency is better; higher throughput is better.
#: Reporting a "throughput improvement" with a latency sign is a defect.
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

#: Only these four are MVP deliverables.
MVP_SERVING_METRICS: tuple[ServingMetric, ...] = (
    ServingMetric.TTFT,
    ServingMetric.TPOT,
    ServingMetric.ITL,
    ServingMetric.TOKENS_PER_SEC,
)

#: Reason codes for a request that did not produce a usable measurement.
#: A failed request is never recorded as a zero-latency success.
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


def default_mvp_workload() -> "WorkloadSpec":
    """A minimal, explicitly-declared MVP workload.

    Deliberately small and non-thinking: the first local GPU session is a feasibility
    gate, not a performance campaign. Real request hashes are supplied by the
    evaluator once the request set is fixed; the placeholder keeps the object
    valid so a run cannot silently proceed with no declared load.
    """
    return WorkloadSpec(
        workload_id="mvp-short-chat",
        request_set_id="builtin-short-chat",
        request_hashes=["unset:workload-not-yet-materialised"],
        thinking_mode=ThinkingMode.DISABLED,
        input_tokens_target=128,
        max_new_tokens=64,
        temperature=0.0,
        concurrency=1,
        warmup_requests=3,
        timed_requests=20,
        cache_policy="disabled",
    )


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


# --------------------------------------------------------------------------
# Quality evaluation (full-prd.md §4 layer 2 — NOT an MVP result)
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


class QuantConfig(BaseModel):
    """How to quantize. Mirrors the recipe the backend actually applied."""

    model_config = ConfigDict(extra="forbid")

    algorithm: str = "gptq"
    scheme: str = "W4A16"
    weight_bits: int = Field(default=4, ge=2, le=16)
    activation_bits: int = Field(default=16, ge=2, le=16)
    group_size: int = Field(default=128, gt=0)
    symmetric: bool = False
    targets: list[str] = Field(default_factory=list)
    ignore: list[str] = Field(default_factory=list)
    calibration_samples: int = Field(default=64, ge=0)
    calibration_max_tokens: int = Field(default=512, gt=0)
    calibration_split_hash: str | None = None
    backend_options: dict[str, Any] = Field(default_factory=dict)
    # Layers kept at higher precision, e.g. ["model.layers.18"].
    high_precision_layers: list[str] = Field(default_factory=list)

    @field_validator("targets", "ignore", "high_precision_layers")
    @classmethod
    def _no_blank_entries(cls, v: list[str]) -> list[str]:
        if any(not item.strip() for item in v):
            raise ValueError("module patterns must not be blank")
        return v

    @model_validator(mode="after")
    def _no_target_ignore_overlap(self) -> "QuantConfig":
        overlap = set(self.targets) & set(self.ignore)
        if overlap:
            raise ValueError(f"targets and ignore overlap: {sorted(overlap)}")
        exempt = set(self.high_precision_layers) & set(self.ignore)
        if exempt:
            raise ValueError(
                f"a layer cannot be both high-precision and ignored: {sorted(exempt)}"
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


class BudgetConfig(BaseModel):
    """Hard caps. Reaching a cap means "record what we have", never "retry forever"."""

    model_config = ConfigDict(extra="forbid")

    max_trials: int = Field(default=8, ge=0)
    max_gpu_hours: float = Field(default=20.0, gt=0)
    max_wall_time_minutes: int = Field(default=480, gt=0)
    max_artifact_bytes: int = Field(default=30 * 1024**3, gt=0)


class PerformanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(default=1, ge=1)
    input_tokens: int = Field(default=128, gt=0)
    output_tokens: int = Field(default=32, gt=0)
    warmup_runs: int = Field(default=3, ge=0)
    measured_runs: int = Field(default=10, ge=1)


class ExperimentSpec(BaseModel):
    """Validated experiment configuration. Any change here yields a new fingerprint."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    name: str = "mvp-qwen3-0.6b"
    model_id: str = MODEL_ID
    model_revision: str | None = None
    tokenizer_id: str | None = None
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    # Qwen3 thinking mode is part of the protocol: a run must state which mode it
    # used, and the two modes are never one comparison (full-prd.md §2).
    thinking_mode: ThinkingMode = ThinkingMode.DISABLED
    seed: int = 0
    batch_size: int = Field(default=1, ge=1)
    smoke_max_tokens: int = Field(default=128, gt=0)
    max_tokens: int = Field(default=512, gt=0)
    quant: QuantConfig = Field(default_factory=QuantConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    #: The MVP's measured object. Required: an experiment with no declared
    #: workload has nothing to compare, and the two sides must share it exactly.
    workload: WorkloadSpec = Field(default_factory=default_mvp_workload)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)

    @field_validator("model_id")
    @classmethod
    def _model_identity_is_fixed(cls, value: str) -> str:
        """The MVP model is fixed by the PRD; a different id must fail, not drift."""
        if value != MODEL_ID:
            raise ValueError(
                f"the MVP fixes the model to {MODEL_ID!r}; got {value!r}. "
                "Changing the model changes the whole comparison and is not a "
                "config-level decision (mvp-prd.md §2)."
            )
        return value

    @model_validator(mode="after")
    def _enforce_single_model_batch_one(self) -> "ExperimentSpec":
        # full-prd.md §3: one service at a time on the single 4 GB card.
        if self.batch_size != 1:
            raise ValueError("MVP requires batch_size=1 (one model served at a time)")
        if self.max_tokens < self.smoke_max_tokens:
            raise ValueError("max_tokens must be >= smoke_max_tokens")
        if self.dtype != "bf16":
            raise ValueError(
                "the baseline is BF16 by protocol; FP16 would require a matched "
                "baseline rebuild and must not be switched silently (AGENTS.md §3)"
            )
        return self

    @property
    def tokenizer(self) -> str:
        return self.tokenizer_id or self.model_id

    @property
    def enable_thinking(self) -> bool:
        return self.thinking_mode is ThinkingMode.ENABLED

    def config_fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        payload.pop("name", None)
        return sha256_of(payload)


# --------------------------------------------------------------------------
# Data manifests
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


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


class MetricBundle(BaseModel):
    """Overall + grouped metrics for one artifact on one dataset."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    run_id: str = ""
    overall: dict[str, MetricRecord] = Field(default_factory=dict)
    slices: dict[str, dict[str, MetricRecord]] = Field(default_factory=dict)
    samples_ref: str | None = None
    notes: list[str] = Field(default_factory=list)

    def require(self, name: str) -> MetricRecord:
        if name not in self.overall:
            raise KeyError(f"metric {name!r} missing from bundle")
        return self.overall[name]


# --------------------------------------------------------------------------
# Artifacts, traces, interventions, recommendations
# --------------------------------------------------------------------------


class ModelArtifact(BaseModel):
    """A loadable model product and the provenance needed to reproduce it."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    checkpoint_path: str
    parent_model_id: str
    parent_model_revision: str | None = None
    parent_fingerprint: str | None = None
    quant: QuantConfig | None = None
    dtype: str = "bf16"
    format: str = "hf"
    run_mode: RunMode = RunMode.BF16
    checksum: str | None = None
    size_bytes: int | None = None
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _quantized_requires_recipe(self) -> "ModelArtifact":
        # A quantized artifact without its recipe cannot be reproduced or explained.
        if self.format not in {"hf", "bf16"} and self.quant is None:
            raise ValueError(
                f"artifact {self.artifact_id!r}: format={self.format!r} requires a QuantConfig"
            )
        return self


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


# --------------------------------------------------------------------------
# Environment capability reporting
# --------------------------------------------------------------------------


class CapabilityReport(BaseModel):
    """What a model x format x backend x GPU combination can actually do."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    gpu_name: str | None = None
    gpu_total_memory_bytes: int | None = None
    compute_capability: str | None = None
    has_native_fp8: bool | None = None
    image_tag: str | None = None
    python_version: str | None = None
    torch_version: str | None = None
    cuda_version: str | None = None
    driver_version: str | None = None
    package_versions: dict[str, str] = Field(default_factory=dict)
    capabilities: dict[str, str] = Field(default_factory=dict)
    unsupported: dict[str, str] = Field(default_factory=dict)
    not_tested: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unsupported_and_not_tested_are_distinct(self) -> "CapabilityReport":
        overlap = set(self.unsupported) & set(self.not_tested)
        if overlap:
            raise ValueError(
                f"entries marked both unsupported and not_tested: {sorted(overlap)}"
            )
        return self


# --------------------------------------------------------------------------
# Run manifest and stage state
# --------------------------------------------------------------------------


class StageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: StageName
    status: StageStatus = StageStatus.PENDING
    started_at: datetime | None = None
    finished_at: datetime | None = None
    attempts: int = Field(default=0, ge=0)
    error: str | None = None
    artifacts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _terminal_states_are_complete(self) -> "StageRecord":
        if self.status in {StageStatus.SUCCEEDED, StageStatus.FAILED} and self.finished_at is None:
            raise ValueError(f"stage {self.stage.value}: terminal status requires finished_at")
        if self.status is StageStatus.FAILED and not self.error:
            raise ValueError(f"stage {self.stage.value}: failed status requires an error")
        return self


class RunManifest(BaseModel):
    """Environment, fingerprints, seeds, parent/child relations and stage state."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    run_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    parent_run_id: str | None = None
    spec: ExperimentSpec
    config_fingerprint: str
    model_fingerprint: str | None = None
    data_fingerprint: str | None = None
    environment: CapabilityReport | None = None
    seed: int = 0
    stages: dict[StageName, StageRecord] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _all_stages_present(self) -> "RunManifest":
        for stage in STAGE_ORDER:
            self.stages.setdefault(stage, StageRecord(stage=stage))
        return self

    def stage(self, stage: StageName) -> StageRecord:
        return self.stages[stage]

    def mark_running(self, stage: StageName) -> StageRecord:
        record = self.stages[stage]
        record.status = StageStatus.RUNNING
        record.started_at = _utcnow()
        record.finished_at = None
        record.error = None
        record.attempts += 1
        self.updated_at = _utcnow()
        return record

    def mark_succeeded(self, stage: StageName, artifacts: list[str] | None = None) -> StageRecord:
        record = self.stages[stage]
        record.status = StageStatus.SUCCEEDED
        record.finished_at = _utcnow()
        record.error = None
        if artifacts:
            record.artifacts = list(artifacts)
        self.updated_at = _utcnow()
        return record

    def mark_failed(self, stage: StageName, error: str) -> StageRecord:
        record = self.stages[stage]
        record.status = StageStatus.FAILED
        record.finished_at = _utcnow()
        record.error = error
        self.updated_at = _utcnow()
        return record

    def last_succeeded_stage(self) -> StageName | None:
        """Highest stage that genuinely succeeded; drives `resume`."""
        last: StageName | None = None
        for stage in STAGE_ORDER:
            if self.stages[stage].status is StageStatus.SUCCEEDED:
                last = stage
            else:
                break
        return last

    def next_pending_stage(self) -> StageName | None:
        for stage in STAGE_ORDER:
            if self.stages[stage].status is not StageStatus.SUCCEEDED:
                return stage
        return None
