"""Load the serving workload shared by both sides of a comparison."""

from pathlib import Path

import yaml

from quantassay.contracts import WorkloadSpec
from quantassay.runtime import ProbeError
from quantassay.serving.workload import build_request_set, request_set_hashes


def load_workload(config_path: Path | None) -> WorkloadSpec:
    """Load the fixed workload, defaulting to the built-in request set.

    The workload is part of the experiment fingerprint: a different request set
    or timing window produces a different fingerprint and cannot be compared.
    """
    requests = build_request_set()
    hashes = request_set_hashes(requests)
    if config_path is None:
        return WorkloadSpec(
            workload_id="mvp-short-chat",
            request_set_id="builtin-short-chat",
            request_hashes=hashes,
            warmup_requests=3,
            timed_requests=20,
            max_new_tokens=64,
        )
    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProbeError(f"cannot read workload config {config_path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ProbeError(f"{config_path} has no workload block")
    raw = document.get("workload")
    if not isinstance(raw, dict):
        raise ProbeError(f"{config_path} has no workload block")
    raw = dict(raw)
    # The request set is code-owned: a config cannot smuggle in different inputs.
    raw["request_hashes"] = hashes
    raw.setdefault("request_set_id", "builtin-short-chat")
    try:
        return WorkloadSpec(**raw)
    except Exception as exc:
        raise ProbeError(f"invalid workload in {config_path}: {exc}") from exc
