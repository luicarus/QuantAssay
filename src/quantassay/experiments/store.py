"""Experiment directory, state machine, atomic writes and cache fingerprints.

Two rules drive this module (mvp-prd.md §6):

1. **A file existing is not a stage succeeding.** Recovery reads the manifest and
   verification values, never the filesystem alone.
2. **Partial writes must not look like success.** Every write goes to a temp file
   and is atomically replaced, after which the manifest is updated.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quantassay.contracts import (
    SCHEMA_VERSION,
    STAGE_ORDER,
    ExperimentSpec,
    RunManifest,
    StageName,
    StageStatus,
    canonical_json,
    sha256_of,
)

MANIFEST_NAME = "manifest.json"
STAGE_STATUS_NAME = "stage-status.json"
RESOLVED_CONFIG_NAME = "resolved-config.yaml"
DATA_MANIFEST_NAME = "data-manifest.json"
METRICS_NAME = "metrics.json"
CACHE_DIR_NAME = "cache"
LOGS_DIR_NAME = "logs"

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class StoreError(RuntimeError):
    """Raised for invalid run directories or broken state."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def generate_run_id(spec: ExperimentSpec, *, now: datetime | None = None) -> str:
    """Human-sortable run id containing the short config fingerprint."""
    stamp = (now or _utcnow()).strftime("%Y%m%d-%H%M%S")
    short = spec.config_fingerprint()[:8]
    return f"{stamp}-{short}-{uuid.uuid4().hex[:6]}"


def validate_run_id(run_id: str) -> str:
    """Reject anything that could escape the runs directory."""
    if not _RUN_ID_RE.match(run_id):
        raise StoreError(f"invalid run id: {run_id!r}")
    if ".." in run_id:
        raise StoreError(f"run id must not contain '..': {run_id!r}")
    return run_id


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write via a temp file in the same directory, then atomically replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return target


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    """Deterministic JSON write, so checksums are stable."""
    return atomic_write_text(path, canonical_json(payload) + "\n")


def read_json(path: str | Path) -> Any:
    file_path = Path(path)
    if not file_path.is_file():
        raise StoreError(f"expected file not found: {file_path}")
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StoreError(f"corrupt JSON in {file_path}: {exc}") from exc


def file_sha256(path: str | Path) -> str:
    """Checksum used to confirm a cached artifact is the one we measured."""
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CacheEntry:
    """A reusable artifact keyed by the inputs that produced it."""

    key: str
    relative_path: str
    checksum: str
    created_at: str


class ExperimentStore:
    """Owns one run directory and the state transitions within it."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)

    # -- construction ------------------------------------------------------

    @classmethod
    def create(cls, runs_root: str | Path, spec: ExperimentSpec, *, run_id: str | None = None,
               parent_run_id: str | None = None) -> "ExperimentStore":
        """Create a fresh run directory. Never reuses or overwrites an existing one."""
        root = Path(runs_root)
        root.mkdir(parents=True, exist_ok=True)
        # Distinguish "no id supplied" from "an id supplied that is invalid":
        # an empty string must be rejected, not silently auto-generated.
        rid = validate_run_id(run_id) if run_id is not None else generate_run_id(spec)
        run_dir = root / rid
        if run_dir.exists():
            raise StoreError(
                f"run directory already exists, refusing to overwrite: {run_dir}"
            )
        run_dir.mkdir(parents=True)
        store = cls(run_dir)
        store._ensure_subdirs()

        manifest = RunManifest(
            schema_version=SCHEMA_VERSION,
            run_id=rid,
            parent_run_id=parent_run_id,
            spec=spec,
            config_fingerprint=spec.config_fingerprint(),
            seed=spec.seed,
        )
        store.save_manifest(manifest)
        store._write_stage_status(manifest)
        return store

    @classmethod
    def open(cls, run_dir: str | Path) -> "ExperimentStore":
        store = cls(run_dir)
        store.load_manifest()  # validates presence and shape
        return store

    def _ensure_subdirs(self) -> None:
        for name in (CACHE_DIR_NAME, LOGS_DIR_NAME):
            (self.run_dir / name).mkdir(parents=True, exist_ok=True)

    # -- paths -------------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / MANIFEST_NAME

    @property
    def stage_status_path(self) -> Path:
        return self.run_dir / STAGE_STATUS_NAME

    @property
    def resolved_config_path(self) -> Path:
        return self.run_dir / RESOLVED_CONFIG_NAME

    @property
    def data_manifest_path(self) -> Path:
        return self.run_dir / DATA_MANIFEST_NAME

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / METRICS_NAME

    @property
    def cache_dir(self) -> Path:
        return self.run_dir / CACHE_DIR_NAME

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / LOGS_DIR_NAME

    def path(self, relative: str | Path) -> Path:
        """Resolve a path inside the run directory, rejecting escapes."""
        candidate = (self.run_dir / relative).resolve()
        root = self.run_dir.resolve()
        if candidate != root and root not in candidate.parents:
            raise StoreError(f"path escapes run directory: {relative}")
        return candidate

    # -- manifest ----------------------------------------------------------

    def load_manifest(self) -> RunManifest:
        payload = read_json(self.manifest_path)
        try:
            return RunManifest.model_validate(payload)
        except Exception as exc:  # pydantic ValidationError
            raise StoreError(f"invalid manifest in {self.run_dir}: {exc}") from exc

    def save_manifest(self, manifest: RunManifest) -> Path:
        manifest.updated_at = _utcnow()
        return atomic_write_json(self.manifest_path, manifest.model_dump(mode="json"))

    def _write_stage_status(self, manifest: RunManifest) -> Path:
        payload = {
            stage.value: {
                "status": record.status.value,
                "attempts": record.attempts,
                "started_at": record.started_at.isoformat() if record.started_at else None,
                "finished_at": record.finished_at.isoformat() if record.finished_at else None,
                "error": record.error,
                "artifacts": record.artifacts,
            }
            for stage, record in manifest.stages.items()
        }
        return atomic_write_json(self.stage_status_path, payload)

    # -- state transitions -------------------------------------------------

    def begin_stage(self, stage: StageName) -> RunManifest:
        manifest = self.load_manifest()
        manifest.mark_running(stage)
        self.save_manifest(manifest)
        self._write_stage_status(manifest)
        return manifest

    def complete_stage(self, stage: StageName, artifacts: list[str] | None = None) -> RunManifest:
        manifest = self.load_manifest()
        manifest.mark_succeeded(stage, artifacts)
        self.save_manifest(manifest)
        self._write_stage_status(manifest)
        return manifest

    def fail_stage(self, stage: StageName, error: str) -> RunManifest:
        manifest = self.load_manifest()
        manifest.mark_failed(stage, error)
        self.save_manifest(manifest)
        self._write_stage_status(manifest)
        return manifest

    # -- cache -------------------------------------------------------------

    def cache_key(self, stage: StageName, payload: Any) -> str:
        """Cache key binds a stage's outputs to its inputs *and* the run config.

        A config change therefore produces a different key, so stale results are
        never silently reused.
        """
        manifest = self.load_manifest()
        return sha256_of(
            {
                "schema_version": SCHEMA_VERSION,
                "stage": stage.value,
                "config_fingerprint": manifest.config_fingerprint,
                "inputs": payload,
            }
        )

    def _cache_index_path(self) -> Path:
        return self.cache_dir / "index.json"

    def load_cache_index(self) -> dict[str, dict[str, str]]:
        path = self._cache_index_path()
        if not path.is_file():
            return {}
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise StoreError(f"cache index must be a mapping: {path}")
        return payload

    def lookup_cache(self, key: str) -> CacheEntry | None:
        """Return a cached entry only if the artifact still matches its checksum."""
        raw = self.load_cache_index().get(key)
        if raw is None:
            return None
        entry = CacheEntry(**raw)
        artifact_path = self.path(entry.relative_path)
        if not artifact_path.is_file():
            return None
        if file_sha256(artifact_path) != entry.checksum:
            return None
        return entry

    def store_cache(self, key: str, relative_path: str, payload: Any) -> CacheEntry:
        """Persist a payload under the cache dir and index it by key."""
        artifact_path = self.path(relative_path)
        atomic_write_json(artifact_path, payload)
        entry = CacheEntry(
            key=key,
            relative_path=relative_path.replace("\\", "/"),
            checksum=file_sha256(artifact_path),
            created_at=_utcnow().isoformat(),
        )
        index = self.load_cache_index()
        index[key] = {
            "key": entry.key,
            "relative_path": entry.relative_path,
            "checksum": entry.checksum,
            "created_at": entry.created_at,
        }
        atomic_write_json(self._cache_index_path(), index)
        return entry

    def invalidate_cache(self, key: str) -> bool:
        index = self.load_cache_index()
        entry = index.pop(key, None)
        if entry is None:
            return False
        atomic_write_json(self._cache_index_path(), index)
        return True

    def prune_cache(self, *, keep_keys: set[str] | None = None) -> list[str]:
        """Drop cache files that are not in ``keep_keys``; returns removed keys."""
        index = self.load_cache_index()
        removed: list[str] = []
        for key, entry in list(index.items()):
            if keep_keys is not None and key in keep_keys:
                continue
            try:
                self.path(entry["relative_path"]).unlink(missing_ok=True)
            except StoreError:
                pass
            index.pop(key, None)
            removed.append(key)
        atomic_write_json(self._cache_index_path(), index)
        return removed

    # -- resume ------------------------------------------------------------

    def resume_plan(self) -> dict[str, Any]:
        """Decide what may be reused and what must be re-run.

        Succeeded stages are reused. A stage left ``running`` (interrupted mid-way)
        is re-run, because its outputs may be partially written.
        """
        manifest = self.load_manifest()
        reusable: list[str] = []
        rerun: list[str] = []
        # Iterate STAGE_ORDER, not the manifest's dict order: pipeline order is a
        # property of the pipeline, and resume decisions depend on it.
        for stage in STAGE_ORDER:
            record = manifest.stages[stage]
            if record.status is StageStatus.SUCCEEDED:
                reusable.append(stage.value)
            elif record.status in {StageStatus.RUNNING, StageStatus.FAILED, StageStatus.PENDING}:
                rerun.append(stage.value)
        return {
            "run_id": manifest.run_id,
            "reusable_stages": reusable,
            "stages_to_run": rerun,
            "last_succeeded_stage": (
                manifest.last_succeeded_stage().value
                if manifest.last_succeeded_stage()
                else None
            ),
            "next_stage": (
                manifest.next_pending_stage().value if manifest.next_pending_stage() else None
            ),
        }

    def is_complete(self) -> bool:
        return self.load_manifest().next_pending_stage() is None

    # -- housekeeping ------------------------------------------------------

    def export_snapshot(self, destination: str | Path, *, include_cache: bool = False) -> Path:
        """Copy records to a durable location before the cloud session ends."""
        dest = Path(destination)
        dest.mkdir(parents=True, exist_ok=True)
        for item in self.run_dir.iterdir():
            if item.name == CACHE_DIR_NAME and not include_cache:
                continue
            if item.name == LOGS_DIR_NAME and not include_cache:
                continue
            target = dest / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
        return dest
