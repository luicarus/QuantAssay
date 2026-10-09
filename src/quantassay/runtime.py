"""Shared SGLang process lifecycle, GPU observations and launch parameters."""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from quantassay.contracts import MODEL_ID
from quantassay.experiments.store import file_sha256
from quantassay.serving.evaluator import iter_sse_payloads


_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


# Triton avoids FlashInfer JIT compilation on the supported WSL stack.
DEFAULT_ATTENTION_BACKEND = "triton"


KNOWN_ATTENTION_BACKENDS = (
    "triton",
    "torch_native",
    "flashinfer",
    "fa3",
    "flashmla",
    "cutlass_mla",
)


DEFAULT_OPERATOR_BACKEND = "sglang"
KNOWN_OPERATOR_BACKENDS = ("sglang", "torch", "triton")


# Limit CUDA graph capture to preserve KV headroom on the 4 GB GPU.
DEFAULT_CUDA_GRAPH_MAX_BS = 2


# File contents, rather than revision names alone, identify the input model.
_SNAPSHOT_FILE_SUFFIXES = (".safetensors", ".bin", ".json", ".model", ".txt")


# Track GPU children so interrupted controllers can reap their process groups.
_LIVE_CHILDREN: set[subprocess.Popen[bytes]] = set()
_CHILD_REAPER_INSTALLED = False


# A sampled peak above this is a warning, not a failed measurement.
GPU_CONCERN_MIB = 3584  # 3.5 GiB


# Weak empirical hint: a healthy run reached 3474 MiB; the cause remains unknown.
PEAK_ANOMALY_MIB = {"bf16": 3500, "gptq": 3500}


# Accept the stable desktop baseline visible through WSL (about 669 MiB).
CLEAN_GPU_MIB = 768


# Minimum free memory before measurement, independent of sampled peaks.
MIN_FREE_MIB_FOR_MEASUREMENT = 2048


class ProbeError(RuntimeError):
    """A capability gate failed without changing the requested experiment."""


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def verify_snapshot_layout(model_dir: Path, revision: str) -> bool:
    """Accept only a cached Qwen3-0.6B snapshot at the requested commit.

    Accepts the HuggingFace layout (``models--Qwen--Qwen3-0.6B``) or the
    ModelScope layout (``Qwen/Qwen3-0.6B``); both resolve ``repo_root`` two
    levels above the snapshot.
    """
    path = model_dir.resolve()
    repo_root = path.parent.parent
    matching_repo = (
        repo_root.name == "models--Qwen--Qwen3-0.6B"
        or (repo_root.name == "Qwen3-0.6B" and repo_root.parent.name == "Qwen")
    )
    return (
        path.is_dir()
        and bool(_REVISION_RE.fullmatch(revision))
        and path.name == revision
        and path.parent.name == "snapshots"
        and matching_repo
    )


def source_file_hashes(model_dir: Path, *, max_files: int = 64) -> dict[str, str]:
    """Hash model content once before measurement, following snapshot symlinks."""
    root = model_dir.resolve()
    if not root.is_dir():
        return {}
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        # Follow links to the real blob; only require that the target is a file.
        try:
            if not path.is_file():
                continue
        except OSError:
            continue
        if path.suffix not in _SNAPSHOT_FILE_SUFFIXES:
            continue
        try:
            hashes[str(path.relative_to(root)).replace("\\", "/")] = file_sha256(path)
        except OSError:
            # An unreadable or dangling link is recorded as absent rather than
            # silently hashed as empty content.
            continue
        if len(hashes) >= max_files:
            break
    return hashes


def build_smoke_payload(model_id: str = MODEL_ID) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": "Reply with one short sentence in English."}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 16,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def parse_sse_stream(lines: Iterable[bytes]) -> dict[str, Any]:
    pieces: list[str] = []
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    try:
        for event in iter_sse_payloads(lines):
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                if isinstance(delta.get("content"), str):
                    pieces.append(delta["content"])
                if choice.get("finish_reason") is not None:
                    finish_reason = str(choice["finish_reason"])
    except ValueError as exc:
        raise ProbeError(f"invalid SSE JSON: {exc}") from exc
    content = "".join(pieces)
    if not content.strip():
        raise ProbeError("stream returned no output text")
    return {"content": content, "usage": usage, "finish_reason": finish_reason}


def _get_models(port: int) -> list[str]:
    with urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
        payload = json.load(response)
    return [item["id"] for item in payload.get("data", []) if isinstance(item, dict) and "id" in item]


def _sample_gpu_used_mib() -> int | None:
    """Used GPU memory in MiB, or None when nvidia-smi cannot be read."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        return int(out.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def _wait_gpu_release(
    baseline_mib: int | None, *, timeout_seconds: float = 30
) -> tuple[int | None, bool | None]:
    """Wait for two release samples, allowing 1 GiB of WSL desktop baseline drift."""
    if baseline_mib is None:
        return _sample_gpu_used_mib(), None
    tolerance = 1024
    deadline = time.monotonic() + timeout_seconds
    after: int | None = None
    stable_samples = 0
    while True:
        after = _sample_gpu_used_mib()
        stable_samples = stable_samples + 1 if after is not None and after <= baseline_mib + tolerance else 0
        if stable_samples >= 2:
            return after, True
        if time.monotonic() >= deadline:
            return after, False if baseline_mib is not None and after is not None else None
        time.sleep(0.5)


class _GpuPeakSampler:
    def __init__(self) -> None:
        self.peak_mib: int | None = None
        self.peak_rss_bytes: int | None = None
        self.root_pid: int | None = None
        try:
            import psutil

            self._psutil = psutil
        except ImportError:
            self._psutil = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            sample = _sample_gpu_used_mib()
            if sample is not None:
                self.peak_mib = sample if self.peak_mib is None else max(self.peak_mib, sample)
            if self.root_pid is not None and self._psutil is not None:
                try:
                    root = self._psutil.Process(self.root_pid)
                    processes = [root, *root.children(recursive=True)]
                    rss = sum(process.memory_info().rss for process in processes)
                    self.peak_rss_bytes = (
                        rss if self.peak_rss_bytes is None else max(self.peak_rss_bytes, rss)
                    )
                except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
                    pass
            self._stop.wait(0.5)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=6)


def _port_in_use(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _stop_process_group(process: subprocess.Popen[bytes]) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        # The launcher may have exited while worker children still own GPU
        # memory. Signal the whole session even when poll() saw an exit.
        try:
            os.killpg(process.pid, getattr(signal, "SIGKILL", 9))
        except ProcessLookupError:
            pass
        if process.poll() is None:
            process.wait(timeout=5)
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _reap_live_children(signum: int, _frame: Any) -> None:
    """Stop every live child, then re-raise the signal with default handling."""
    for process in list(_LIVE_CHILDREN):
        try:
            _stop_process_group(process)
        except Exception:
            pass
    _LIVE_CHILDREN.clear()
    # Restore default disposition and re-send, so the exit status still reports
    # "killed by signal" rather than a clean exit.
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def _install_child_reaper() -> None:
    """Install once per process; only meaningful on POSIX."""
    global _CHILD_REAPER_INSTALLED
    if _CHILD_REAPER_INSTALLED or os.name != "posix":
        return
    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            previous = signal.getsignal(signum)
            if previous in (signal.SIG_DFL, None):
                signal.signal(signum, _reap_live_children)
        except (ValueError, OSError):
            # Not on the main thread: skip rather than fail the run.
            return
    _CHILD_REAPER_INSTALLED = True


@contextlib.contextmanager
def _tracked_process(command: list[str], log_handle: Any):
    """Popen + register in the reap set, removing it on exit."""
    _install_child_reaper()
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=(os.name == "posix"),
    )
    _LIVE_CHILDREN.add(process)
    try:
        yield process
    finally:
        _LIVE_CHILDREN.discard(process)


@contextlib.contextmanager
def server_session(
    command: list[str],
    *,
    port: int,
    model_id: str,
    log_path: Path,
    startup_timeout: float = 240,
):
    """Launch one server and retain startup, process cleanup and GPU release evidence."""
    if _port_in_use(port):
        raise ProbeError(f"port {port} already in use; refusing to probe a different server")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        raise ProbeError(f"log file already exists: {log_path}")

    baseline_gpu_mib = _sample_gpu_used_mib()
    sampler = _GpuPeakSampler()
    start = time.monotonic()
    server_error: Exception | None = None
    handle: dict[str, Any] = {"model_id": model_id, "command": command, "log_path": str(log_path)}

    with log_path.open("xb") as log:
        with _tracked_process(command, log) as process:
            sampler.root_pid = process.pid
            sampler.start()
            try:
                deadline = time.monotonic() + startup_timeout
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise ProbeError(
                            f"server exited during startup with code {process.returncode}"
                        )
                    try:
                        models = _get_models(port)
                        if model_id in models:
                            break
                    except (URLError, HTTPError, TimeoutError, OSError, ValueError, KeyError):
                        pass
                    time.sleep(0.2)
                else:
                    raise ProbeError(
                        f"server did not expose {model_id!r} within {startup_timeout}s"
                    )
                handle["startup_seconds"] = round(time.monotonic() - start, 3)
                handle["base_url"] = f"http://127.0.0.1:{port}"
                yield handle
            except Exception as exc:
                server_error = exc
            finally:
                _stop_process_group(process)
                sampler.stop()

    after_gpu_mib, gpu_release_confirmed = _wait_gpu_release(baseline_gpu_mib)
    handle.update(
        {
            "log_sha256": file_sha256(log_path),
            "gpu_used_before_mib": baseline_gpu_mib,
            "gpu_peak_sampled_mib": sampler.peak_mib,
            "peak_rss_bytes": sampler.peak_rss_bytes,
            "gpu_used_after_mib": after_gpu_mib,
            "gpu_release_confirmed": gpu_release_confirmed,
        }
    )
    if server_error is not None:
        raise ProbeError(
            f"{server_error}; release_confirmed={gpu_release_confirmed}; "
            f"gpu_used_after_mib={after_gpu_mib}; log={log_path}"
        ) from server_error
    if _port_in_use(port):
        raise ProbeError(f"server port {port} remained open after shutdown")


def run_server_smoke(
    command: list[str],
    *,
    port: int,
    model_id: str,
    log_path: Path,
    startup_timeout: float = 240,
    request_timeout: float = 30,
) -> dict[str, Any]:
    """Start one server, prove a stream works, and always release it."""
    with server_session(
        command,
        port=port,
        model_id=model_id,
        log_path=log_path,
        startup_timeout=startup_timeout,
    ) as handle:
        payload = json.dumps(build_smoke_payload(model_id)).encode("utf-8")
        request = Request(
            f"{handle['base_url']}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=request_timeout) as response:
                parsed = parse_sse_stream(response)
        except (URLError, HTTPError, TimeoutError, OSError) as exc:
            raise ProbeError(f"streaming request failed: {exc}") from exc
        handle["content"] = parsed["content"]
        handle["usage"] = parsed["usage"]
        handle["finish_reason"] = parsed["finish_reason"]
    return handle


def anomaly_warnings(result: dict[str, Any], *, side: str) -> list[str]:
    """Flag high sampled peaks as a weak anomaly hint, not proof of slowdown."""
    warnings: list[str] = []
    peak = result.get("gpu_peak_sampled_mib")
    threshold = PEAK_ANOMALY_MIB.get(side)
    if peak is not None and threshold is not None and peak > threshold:
        baseline = "~7.9 ms" if side == "bf16" else "~4.3 ms"
        warnings.append(
            f"{side} peak {peak} MiB exceeds the {threshold} MiB line for this side: "
            "possible recurrence of the run-to-run speed anomaly "
            f"(compatibility.md §5.8). Check whether this side's TPOT is ~3x its "
            f"healthy baseline ({baseline}); if so, re-running that side is enough. "
            "This flag is weak (one healthy sample reached 3474 MiB) — treat it as a "
            "hint, not evidence."
        )
    return warnings


def wait_for_clean_gpu(
    *,
    timeout_seconds: float = 30.0,
    sampler: Callable[[], int | None] = _sample_gpu_used_mib,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[int | None, bool]:
    """Return (used_mib, settled), allowing the observed WSL desktop baseline."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        used = sampler()
        if used is None or used <= CLEAN_GPU_MIB:
            return used, True
        if time.monotonic() >= deadline:
            return used, False
        sleeper(1.0)


def resource_blockers(result: dict[str, Any]) -> list[str]:
    """Require a GPU peak reading; RSS sampling may double-count shared pages."""
    reasons: list[str] = []
    if result.get("gpu_peak_sampled_mib") is None:
        reasons.append("GPU peak measurement unavailable")
    return reasons


def resource_warnings(result: dict[str, Any]) -> list[str]:
    """Recorded concerns that do not invalidate the measurement."""
    warnings: list[str] = []
    gpu_peak = result.get("gpu_peak_sampled_mib")
    if gpu_peak is not None and gpu_peak > GPU_CONCERN_MIB:
        warnings.append(
            f"GPU peak {gpu_peak} MiB exceeded the {GPU_CONCERN_MIB} MiB concern "
            "line on the 4 GB card; the service stayed healthy and the "
            "measurement completed, but headroom is thin"
        )
    return warnings


def contention_blockers(result: dict[str, Any], *, total_mib: int = 4096) -> list[str]:
    """Reject measurements with insufficient free GPU memory before startup."""
    before = result.get("gpu_used_before_mib")
    if before is None:
        return ["GPU baseline was not sampled before the measurement"]
    free = total_mib - before
    if free < MIN_FREE_MIB_FOR_MEASUREMENT:
        return [
            f"only {free} MiB of GPU memory was free before the measurement "
            f"(>= {MIN_FREE_MIB_FOR_MEASUREMENT} MiB required); another process or "
            "the Windows desktop held the rest, which changes KV headroom and "
            "makes latency/throughput incomparable to other runs"
        ]
    return []


def build_bf16_command(
    model_dir: Path,
    port: int,
    mem_fraction: float,
    *,
    attention_backend: str = DEFAULT_ATTENTION_BACKEND,
    operator_backend: str = DEFAULT_OPERATOR_BACKEND,
    cuda_graph_max_bs: int = DEFAULT_CUDA_GRAPH_MAX_BS,
    sglang_version: str | None = None,
    disable_cuda_graph: bool = False,
    context_length: int = 512,
    max_running_requests: int = 1,
) -> list[str]:
    if not 0.5 <= mem_fraction <= 0.9:
        raise ProbeError("mem_fraction_static must be between 0.5 and 0.9")
    if attention_backend not in KNOWN_ATTENTION_BACKENDS:
        raise ProbeError(
            f"unknown attention backend {attention_backend!r}; "
            f"known: {sorted(KNOWN_ATTENTION_BACKENDS)}"
        )
    if operator_backend not in KNOWN_OPERATOR_BACKENDS:
        raise ProbeError(
            f"unknown operator backend {operator_backend!r}; "
            f"known: {sorted(KNOWN_OPERATOR_BACKENDS)}"
        )
    if operator_backend != "sglang" and sglang_version is not None:
        if _sglang_version_tuple(sglang_version) != (0, 5, 3):
            raise ProbeError("QuantAssay's Kernscope adapter requires SGLang 0.5.3")
    if not 1 <= cuda_graph_max_bs <= 8:
        raise ProbeError("cuda-graph-max-bs must be between 1 and 8")
    if context_length < 1 or max_running_requests < 1:
        raise ProbeError("context length and max running requests must be positive")
    module = (
        "sglang.launch_server"
        if operator_backend == "sglang"
        else "quantassay.integrations.sglang.v0_5_3"
    )
    command = [
        sys.executable,
        "-m",
        module,
    ]
    if operator_backend != "sglang":
        command += ["--kernscope-backend", operator_backend]
    command += [
        "--model-path",
        str(model_dir.resolve()),
        "--served-model-name",
        MODEL_ID,
        "--dtype",
        "bfloat16",
        "--attention-backend",
        attention_backend,
        "--context-length",
        str(context_length),
        "--mem-fraction-static",
        str(mem_fraction),
        "--max-running-requests",
        str(max_running_requests),
    ]
    # The CUDA-graph cap flag differs by version: 0.5.20 split it per phase,
    # 0.5.3 exposes a combined --cuda-graph-max-bs. Version-gate the choice so
    # the command is valid against the runtime that will actually parse it.
    split_graph_flags = _sglang_has_split_graph_flags(sglang_version)
    if split_graph_flags:
        command += [
            "--cuda-graph-max-bs-decode",
            str(cuda_graph_max_bs),
            "--cuda-graph-max-bs-prefill",
            str(cuda_graph_max_bs),
        ]
    else:
        command += ["--cuda-graph-max-bs", str(cuda_graph_max_bs)]
    # Graphs affect execution; use the same setting on both comparison sides.
    if disable_cuda_graph:
        command += ["--disable-cuda-graph"]
    command += [
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    return command


def _sglang_version_tuple(version: str | None) -> tuple[int, int, int]:
    parts = (version or "0").split(".")
    numbers: list[int] = []
    for part in parts[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        numbers.append(int(digits) if digits else 0)
    while len(numbers) < 3:
        numbers.append(0)
    return (numbers[0], numbers[1], numbers[2])


def _sglang_has_split_graph_flags(version: str | None) -> bool:
    """0.5.20 introduced --cuda-graph-max-bs-{decode,prefill}; 0.5.3 has the
    combined flag only."""
    return _sglang_version_tuple(version) >= (0, 5, 20)
