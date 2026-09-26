"""Config loading and validation.

Invalid configurations must fail immediately and loudly — a spec that silently
drifts produces experiments that cannot be compared (mvp-prd.md §3).

The model identity is validated here rather than merely defaulted: a config that
names a different model is a different experiment, not a configurable option.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from quantassay.contracts import ExperimentSpec


class ConfigError(ValueError):
    """Raised when a config file is missing, unreadable or semantically invalid."""


def load_spec(path: str | Path) -> ExperimentSpec:
    """Load and validate an ``ExperimentSpec`` from a YAML file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc

    if raw is None:
        raise ConfigError(f"config file is empty: {config_path}")
    if not isinstance(raw, dict):
        raise ConfigError(
            f"config root must be a mapping, got {type(raw).__name__}: {config_path}"
        )

    return validate_spec(raw)


def validate_spec(raw: dict[str, Any]) -> ExperimentSpec:
    """Validate a raw mapping into an ``ExperimentSpec`` with readable errors."""
    try:
        return ExperimentSpec.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc


def _format_validation_error(exc: ValidationError) -> str:
    lines = ["invalid experiment config:"]
    for err in exc.errors():
        location = ".".join(str(part) for part in err["loc"]) or "<root>"
        lines.append(f"  - {location}: {err['msg']}")
    return "\n".join(lines)


def dump_spec(spec: ExperimentSpec, path: str | Path) -> Path:
    """Write the fully resolved spec, so a run records what it actually used."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = spec.model_dump(mode="json")
    out.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return out
