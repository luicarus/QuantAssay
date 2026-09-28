"""CPU contract tests for the Qwen3/SGLang M0 gate."""

from __future__ import annotations

import os
import socket
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import quantassay.gating as gating
from quantassay.gating import (
    contention_blockers,
    wait_for_clean_gpu,
    resource_warnings,
    ProbeError,
    StageJournal,
    build_bf16_command,
    build_quant_command,
    build_smoke_payload,
    classify_quant_kernel,
    read_quant_artifact,
    resource_blockers,
    run_quantization_process,
    source_file_hashes,
    parse_sse_stream,
    run_server_smoke,
    validate_preflight,
    verify_snapshot_layout,
    verify_quant_service_evidence,
)


def _write_packed_artifact(
    checkpoint: Path, revision: str, *, method: str = "gptq"
) -> dict[str, object]:
    import hashlib
    import json

    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "model.safetensors").write_bytes(b"packed")
    (checkpoint / "tokenizer.json").write_text("{}", encoding="utf-8")
    config = {
        "model_type": "qwen3",
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {"num_bits": 4, "group_size": 128, "type": "int"},
                    "input_activations": None,
                }
            },
        },
    }
    (checkpoint / "config.json").write_text(json.dumps(config), encoding="utf-8")
    files = {
        name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
        for name in ("model.safetensors", "tokenizer.json", "config.json")
    }
    manifest = {
        "model_id": "Qwen/Qwen3-0.6B",
        "parent_revision": revision,
        "method": method,
        "recipe": {"algorithm": method.upper(), "scheme": "W4A16", "group_size": 128},
        "format": {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "num_bits": 4,
            "group_size": 128,
            "safetensors_files": ["model.safetensors"],
        },
        "files": files,
    }
    (checkpoint / "artifact-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_preflight_rejects_missing_torch_and_missing_runtime() -> None:
    """Local WSL2 has no pinned torch, but torch and the stack must be present."""
    report = {
        "python": "3.12.5",
        "torch": "2.6.0+cu124",
        "torch_cuda": "13.0",
        "cuda_available": True,
        "gpu_name": "NVIDIA GeForce RTX 3050 Ti Laptop GPU",
        "gpu_total_mib": 4096,
        "disk_free_gib": 90.0,
        "packages": {
            "sglang": "0.5.0",
            "llmcompressor": "0.13.0",
            "compressed-tensors": "0.18.0",
            "transformers": "5.14.1",
        },
    }
    assert validate_preflight(report) == []

    # A different torch version is acceptable locally; absence is not.
    report["torch"] = "2.12.0+cu128"
    assert validate_preflight(report) == []

    report["torch"] = None
    report["packages"]["sglang"] = None
    reasons = validate_preflight(report)
    assert any("torch" in reason for reason in reasons)
    assert any("sglang" in reason for reason in reasons)


def test_preflight_rejects_a_different_model_architecture() -> None:
    report = {
        "python": "3.12.5",
        "torch": "2.6.0+cu124",
        "torch_cuda": "13.0",
        "cuda_available": True,
        "gpu_name": "NVIDIA GeForce RTX 3050 Ti Laptop GPU",
        "gpu_total_mib": 4096,
        "disk_free_gib": 90.0,
        "model_config_exists": True,
        "model_type": "qwen2",
        "revision_valid": True,
        "packages": {
            "sglang": "0.5.0",
            "llmcompressor": "0.13.0",
            "compressed-tensors": "0.18.0",
            "transformers": "5.14.1",
        },
    }
    assert any("qwen3" in reason for reason in validate_preflight(report))


def test_preflight_blocks_on_required_import_failure() -> None:
    """A genuinely unusable runtime must block before any GPU work starts."""
    report = {
        "python": "3.12.5",
        "torch": "2.6.0+cu124",
        "torch_cuda": "13.0",
        "cuda_available": True,
        "gpu_name": "NVIDIA GeForce RTX 3050 Ti Laptop GPU",
        "gpu_total_mib": 4096,
        "disk_free_gib": 90.0,
        "packages": {
            "sglang": "0.5.0",
            "llmcompressor": "0.13.0",
            "compressed-tensors": "0.18.0",
            "transformers": "5.14.1",
        },
        "required_import_ok": False,
    }
    reasons = validate_preflight(report)
    assert any("import" in reason for reason in reasons)


def test_pip_conflicts_and_gptq_import_do_not_block() -> None:
    """Image-shipped packages conflict by construction; that must not stop a run.

    `pip check` scans unrelated preinstalled packages (vllm, ms-swift, litellm)
    whose pins cannot all be satisfied, and GPTQModifier construction can fail on
    an optional backend such as transformer_engine. Neither proves this
    experiment cannot run, so both are warnings rather than blockers.
    """
    report = {
        "python": "3.12.5",
        "torch": "2.6.0+cu124",
        "torch_cuda": "13.0",
        "cuda_available": True,
        "gpu_name": "NVIDIA GeForce RTX 3050 Ti Laptop GPU",
        "gpu_total_mib": 4096,
        "disk_free_gib": 90.0,
        "packages": {
            "sglang": "0.5.0",
            "llmcompressor": "0.13.0",
            "compressed-tensors": "0.18.0",
            "transformers": "5.14.1",
        },
        "required_import_ok": True,
        "pip_check_ok": False,
        "gptq_import_ok": False,
    }
    assert validate_preflight(report) == []

    warnings = gating.preflight_warnings(report)
    assert any("pip check" in w for w in warnings)
    assert any("GPTQModifier" in w for w in warnings)


def test_missing_quantization_packages_block() -> None:
    """Packages the pipeline actually imports must be present."""
    report = {
        "python": "3.12.5",
        "torch": "2.6.0+cu124",
        "torch_cuda": "13.0",
        "cuda_available": True,
        "gpu_name": "NVIDIA GeForce RTX 3050 Ti Laptop GPU",
        "gpu_total_mib": 4096,
        "disk_free_gib": 90.0,
        "packages": {"sglang": "0.5.0", "llmcompressor": "0.13.0"},
        "required_import_ok": True,
    }
    reasons = validate_preflight(report)
    assert any("compressed-tensors" in r for r in reasons)
    assert any("transformers" in r for r in reasons)


def test_model_snapshot_must_match_the_claimed_commit(tmp_path: Path) -> None:
    revision = "a" * 40
    snapshot = tmp_path / "models--Qwen--Qwen3-0.6B" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    assert verify_snapshot_layout(snapshot, revision) is True
    modelscope_snapshot = tmp_path / "models" / "Qwen" / "Qwen3-0.6B" / "snapshots" / revision
    modelscope_snapshot.mkdir(parents=True)
    assert verify_snapshot_layout(modelscope_snapshot, revision) is True
    assert verify_snapshot_layout(snapshot, "b" * 40) is False
    assert verify_snapshot_layout(tmp_path / "Qwen3-0.6B", revision) is False


def test_source_snapshot_hash_changes_when_weights_change(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    weights = snapshot / "model.safetensors"
    weights.write_bytes(b"first")
    before = source_file_hashes(snapshot)
    before_fingerprint = gating._fingerprint(snapshot, "a" * 40, 30000, 0.8, before)
    weights.write_bytes(b"second")
    after = source_file_hashes(snapshot)
    after_fingerprint = gating._fingerprint(snapshot, "a" * 40, 30000, 0.8, after)
    assert before["model.safetensors"] != after["model.safetensors"]
    assert before_fingerprint != after_fingerprint


@pytest.mark.skipif(
    os.name == "nt",
    reason="creating symlinks on Windows needs elevation; the WSL execution layer is the real target",
)
def test_source_file_hashes_follows_symlinks(tmp_path: Path) -> None:
    """HuggingFace snapshots are symlinks into ../../blobs/<sha>.

    Skipping symlinks would yield an empty fingerprint, silently disabling the
    content check that the revision string alone cannot provide.
    """
    repo = tmp_path / "models--Qwen--Qwen3-0.6B"
    blobs = repo / "blobs"
    blobs.mkdir(parents=True)
    snapshot = repo / "snapshots" / ("b" * 40)
    snapshot.mkdir(parents=True)

    blob = blobs / "deadbeef"
    blob.write_bytes(b"weights-v1")
    (snapshot / "model.safetensors").symlink_to(blob)
    (snapshot / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")

    hashes = source_file_hashes(snapshot)
    assert "model.safetensors" in hashes, "symlinked weight shard must be hashed"
    assert hashes["model.safetensors"]

    before = gating._fingerprint(snapshot, "b" * 40, 30000, 0.8)

    # Mutating the blob behind the link must change the fingerprint.
    blob.write_bytes(b"weights-v2")
    after = gating._fingerprint(snapshot, "b" * 40, 30000, 0.8)
    assert before != after


@pytest.mark.skipif(
    os.name == "nt",
    reason="creating symlinks on Windows needs elevation; the WSL execution layer is the real target",
)
def test_source_file_hashes_ignores_dangling_symlinks(tmp_path: Path) -> None:
    """A broken link must not crash preflight; it is simply not hashed."""
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "missing.safetensors").symlink_to(tmp_path / "does-not-exist")

    hashes = source_file_hashes(snapshot)
    assert "config.json" in hashes
    assert "missing.safetensors" not in hashes


def test_smoke_payload_explicitly_disables_thinking() -> None:
    payload = build_smoke_payload("Qwen/Qwen3-0.6B")
    assert payload["model"] == "Qwen/Qwen3-0.6B"
    assert payload["stream"] is True
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["max_tokens"] == 16


def test_stream_parser_requires_output_and_keeps_usage() -> None:
    events = [
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
        b'data: {"choices":[{"delta":{"content":" world"}}]}\n',
        b'data: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":2}}\n',
        b'data: [DONE]\n',
    ]
    parsed = parse_sse_stream(events)
    assert parsed == {
        "content": "Hello world",
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        "finish_reason": None,
    }
    with pytest.raises(ProbeError, match="no output"):
        parse_sse_stream([b'data: {"choices":[]}\n', b'data: [DONE]\n'])


def test_journal_reuses_only_verified_result_and_rejects_changed_inputs(tmp_path: Path) -> None:
    journal = StageJournal(tmp_path, fingerprint="same-model-and-settings")
    calls: list[int] = []

    def operation() -> dict[str, int]:
        calls.append(1)
        return {"attempt": len(calls)}

    assert journal.run("preflight", operation) == {"attempt": 1}
    assert journal.run("preflight", operation) == {"attempt": 1}
    assert len(calls) == 1
    assert journal.state["stages"]["preflight"]["attempts"] == 1

    (tmp_path / "preflight.json").write_text('{"attempt":999}', encoding="utf-8")
    assert journal.run("preflight", operation) == {"attempt": 2}
    assert len(calls) == 2
    assert journal.state["stages"]["preflight"]["attempts"] == 2

    with pytest.raises(ProbeError, match="fingerprint"):
        StageJournal(tmp_path, fingerprint="different-model")


def test_blocked_preflight_is_recorded_and_rechecked(tmp_path: Path) -> None:
    journal = StageJournal(tmp_path, fingerprint="same-settings")
    calls: list[int] = []

    def operation() -> dict[str, object]:
        calls.append(1)
        return {"status": "blocked", "reasons": ["sglang is missing"]}

    journal.run("preflight", operation)
    assert journal.state["stages"]["preflight"]["status"] == "blocked"
    journal.run("preflight", operation)
    assert len(calls) == 2


def test_server_smoke_uses_real_http_and_releases_port(tmp_path: Path) -> None:
    server = tmp_path / "fake_server.py"
    server.write_text(
        """
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != '/v1/models':
            self.send_error(404)
            return
        body = json.dumps({'data': [{'id': 'Qwen/Qwen3-0.6B'}]}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        size = int(self.headers['Content-Length'])
        payload = json.loads(self.rfile.read(size))
        if payload.get('chat_template_kwargs') != {'enable_thinking': False}:
            self.send_error(400)
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        self.wfile.write(b'data: {"choices":[{"delta":{"content":"ready"}}]}\\n\\n')
        self.wfile.write(b'data: [DONE]\\n\\n')
        self.wfile.flush()

    def log_message(self, *args):
        pass

HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
""".strip(),
        encoding="utf-8",
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    result = run_server_smoke(
        [sys.executable, str(server), str(port)],
        port=port,
        model_id="Qwen/Qwen3-0.6B",
        log_path=tmp_path / "server.log",
        startup_timeout=5,
        request_timeout=3,
    )
    assert result["content"] == "ready"
    assert result["model_id"] == "Qwen/Qwen3-0.6B"
    with socket.socket() as sock:
        assert sock.connect_ex(("127.0.0.1", port)) != 0


def test_server_smoke_does_not_talk_to_existing_port(tmp_path: Path) -> None:
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        port = occupied.getsockname()[1]
        with pytest.raises(ProbeError, match="port .* already in use"):
            run_server_smoke(
                [sys.executable, "-c", "print('should not start')"],
                port=port,
                model_id="Qwen/Qwen3-0.6B",
                log_path=tmp_path / "server.log",
                startup_timeout=1,
                request_timeout=1,
            )


def test_posix_cleanup_signals_group_even_if_launcher_already_exited(monkeypatch) -> None:
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        gating,
        "os",
        SimpleNamespace(name="posix", killpg=lambda pid, sig: signals.append((pid, sig))),
    )

    class ExitedLauncher:
        pid = 4242

        def poll(self) -> int:
            return 1

        def wait(self, timeout: int) -> int:
            return 1

    gating._stop_process_group(ExitedLauncher())
    assert signals and signals[0][0] == 4242


def test_quant_command_uses_same_snapshot_and_separate_worker(tmp_path: Path) -> None:
    command = build_quant_command(
        tmp_path / "snapshot", "a" * 40, tmp_path / "checkpoint"
    )
    assert command[:3] == [sys.executable, "-m", "quantassay.quantize_worker"]
    assert command[command.index("--model-dir") + 1] == str((tmp_path / "snapshot").resolve())
    assert command[command.index("--revision") + 1] == "a" * 40
    assert command[command.index("--output-dir") + 1] == str((tmp_path / "checkpoint").resolve())


def test_bf16_command_defaults_to_triton_attention_backend(tmp_path: Path) -> None:
    """flashinfer JIT needs nvcc >= 12.4; this machine ships 12.0.

    The default backend must therefore avoid the flashinfer compile path
    entirely, and the choice must be visible in the command.
    """
    command = build_bf16_command(tmp_path, 30000, 0.8)
    assert "--attention-backend" in command
    assert command[command.index("--attention-backend") + 1] == gating.DEFAULT_ATTENTION_BACKEND
    assert gating.DEFAULT_ATTENTION_BACKEND == "triton"
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert command[command.index("--context-length") + 1] == "512"
    assert command[command.index("--mem-fraction-static") + 1] == "0.8"
    assert command[command.index("--max-running-requests") + 1] == "1"


@pytest.mark.parametrize("backend", ["torch", "triton"])
def test_bf16_command_uses_versioned_kernscope_adapter(tmp_path: Path, backend: str) -> None:
    command = build_bf16_command(
        tmp_path,
        30000,
        0.8,
        operator_backend=backend,
        sglang_version="0.5.3",
    )
    assert command[:3] == [
        sys.executable,
        "-m",
        "quantassay.integrations.sglang.v0_5_3",
    ]
    assert command[3:5] == ["--kernscope-backend", backend]
    assert command[command.index("--attention-backend") + 1] == gating.DEFAULT_ATTENTION_BACKEND


def test_kernscope_adapter_rejects_other_sglang_versions(tmp_path: Path) -> None:
    with pytest.raises(ProbeError, match="requires SGLang 0.5.3"):
        build_bf16_command(
            tmp_path,
            30000,
            0.8,
            operator_backend="triton",
            sglang_version="0.5.20",
        )


def test_bf16_command_rejects_unknown_backend(tmp_path: Path) -> None:
    with pytest.raises(ProbeError, match="attention backend"):
        build_bf16_command(tmp_path, 30000, 0.8, attention_backend="nope")


def test_bf16_command_caps_cuda_graph_batch_size(tmp_path: Path) -> None:
    """The prefill CUDA graph captures one graph per batch-size bucket.

    With max-running-requests=1 the larger buckets are never used, but each
    still costs VRAM at startup — the GPTQ service OOMed exactly there on the
    4 GB card, so the command must pass an explicit cap.
    """
    command = build_bf16_command(tmp_path, 30000, 0.8, sglang_version='0.5.20')
    assert '--cuda-graph-max-bs-decode' in command
    assert '--cuda-graph-max-bs-prefill' in command
    assert command[command.index('--cuda-graph-max-bs-decode') + 1] == str(
        gating.DEFAULT_CUDA_GRAPH_MAX_BS
    )
    assert command[command.index('--cuda-graph-max-bs-prefill') + 1] == str(
        gating.DEFAULT_CUDA_GRAPH_MAX_BS
    )
    assert 1 <= gating.DEFAULT_CUDA_GRAPH_MAX_BS <= 8

    capped = build_bf16_command(tmp_path, 30000, 0.8, sglang_version='0.5.20', cuda_graph_max_bs=4)
    assert capped[capped.index('--cuda-graph-max-bs-decode') + 1] == '4'

    # The 0.5.3 runtime only knows the combined flag.
    old_style = build_bf16_command(tmp_path, 30000, 0.8, sglang_version='0.5.3')
    assert '--cuda-graph-max-bs' in old_style
    assert '--cuda-graph-max-bs-decode' not in old_style
    with pytest.raises(ProbeError, match="cuda-graph-max-bs"):
        build_bf16_command(tmp_path, 30000, 0.8, cuda_graph_max_bs=64)


def test_fingerprint_changes_with_cuda_graph_cap(tmp_path: Path) -> None:
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")

    base = gating._fingerprint(snapshot, "a" * 40, 30000, 0.8, cuda_graph_max_bs=2)
    other = gating._fingerprint(snapshot, "a" * 40, 30000, 0.8, cuda_graph_max_bs=4)
    assert base != other


def test_fingerprint_changes_with_attention_backend(tmp_path: Path) -> None:
    """The backend changes the execution path, so it changes the run identity."""
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")

    base = gating._fingerprint(snapshot, "a" * 40, 30000, 0.8, attention_backend="triton")
    other = gating._fingerprint(snapshot, "a" * 40, 30000, 0.8, attention_backend="flashinfer")
    assert base != other
    assert base == gating._fingerprint(
        snapshot, "a" * 40, 30000, 0.8, attention_backend="triton"
    )


def test_fingerprint_changes_with_operator_backend(tmp_path: Path) -> None:
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")

    builtin = gating._fingerprint(snapshot, "a" * 40, 30000, 0.8, operator_backend="sglang")
    triton = gating._fingerprint(snapshot, "a" * 40, 30000, 0.8, operator_backend="triton")
    assert builtin != triton


def test_checkpoint_manifest_detects_tampering(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_packed_artifact(checkpoint, "a" * 40)

    assert read_quant_artifact(checkpoint)["parent_revision"] == "a" * 40
    (checkpoint / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ProbeError, match="checksum"):
        read_quant_artifact(checkpoint)


def test_artifact_rejects_unpacked_config_even_with_matching_new_hashes(tmp_path: Path) -> None:
    import hashlib
    import json

    checkpoint = tmp_path / "checkpoint"
    manifest = _write_packed_artifact(checkpoint, "a" * 40)
    config_path = checkpoint / "config.json"
    config_path.write_text(json.dumps({"model_type": "qwen3"}), encoding="utf-8")
    manifest["files"]["config.json"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    (checkpoint / "artifact-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ProbeError, match="packed W4A16"):
        read_quant_artifact(checkpoint)


def test_quant_service_evidence_is_bound_to_the_checkpoint_manifest(tmp_path: Path) -> None:
    import hashlib
    import json

    checkpoint = tmp_path / "checkpoint"
    manifest = _write_packed_artifact(checkpoint, "a" * 40)
    log = tmp_path / "server.log"
    log.write_text("Using MarlinLinearKernel for CompressedTensorsWNA16", encoding="utf-8")
    result = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_manifest_sha256": hashlib.sha256(
            (checkpoint / "artifact-manifest.json").read_bytes()
        ).hexdigest(),
        "log_path": str(log),
        "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        "quant_kernel": "compressed_tensors_wna16_marlin",
    }
    assert verify_quant_service_evidence(result) is True
    (checkpoint / "model.safetensors").write_bytes(b"new-packed")
    manifest["files"]["model.safetensors"] = hashlib.sha256(b"new-packed").hexdigest()
    (checkpoint / "artifact-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_quant_service_evidence(result) is False


def test_journal_rechecks_quant_artifact_before_reuse(tmp_path: Path) -> None:
    journal = StageJournal(tmp_path, fingerprint="fixed")
    calls = 0
    artifact = tmp_path / "weights.safetensors"

    def operation() -> dict[str, str]:
        nonlocal calls
        calls += 1
        artifact.write_text(str(calls), encoding="utf-8")
        return {"path": str(artifact)}

    def valid(result: dict[str, str]) -> bool:
        return Path(result["path"]).read_text(encoding="utf-8") == str(calls)

    journal.run("gptq", operation, verify_reuse=valid)
    journal.run("gptq", operation, verify_reuse=valid)
    assert calls == 1
    artifact.write_text("corrupt", encoding="utf-8")
    journal.run("gptq", operation, verify_reuse=valid)
    assert calls == 2


def test_corrupt_artifact_invalidates_recorded_success(tmp_path: Path) -> None:
    journal = StageJournal(tmp_path, fingerprint="fixed")
    journal.run("gptq", lambda: {"checkpoint": "present"})

    def reject(_):
        raise ProbeError("checkpoint checksum mismatch")

    with pytest.raises(ProbeError, match="checksum"):
        journal.run("gptq", lambda: {"checkpoint": "new"}, verify_reuse=reject)
    assert journal.state["stages"]["gptq"]["status"] == "invalid"


def test_corrupt_stage_journal_blocks_resume_with_a_readable_error(tmp_path: Path) -> None:
    (tmp_path / "run-status.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(ProbeError, match="invalid run journal"):
        StageJournal(tmp_path, fingerprint="fixed")


def test_server_probe_refuses_to_append_to_an_existing_log(tmp_path: Path) -> None:
    log = tmp_path / "server.log"
    log.write_text("Using MarlinLinearKernel for CompressedTensorsWNA16", encoding="utf-8")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with pytest.raises(ProbeError, match="log file already exists"):
        run_server_smoke(
            [sys.executable, "-c", "print('should not start')"],
            port=port,
            model_id="Qwen/Qwen3-0.6B",
            log_path=log,
            startup_timeout=1,
            request_timeout=1,
        )


def test_all_stages_run_in_order_and_resume_without_repeating_gpu_work(tmp_path: Path, monkeypatch) -> None:
    import hashlib
    import json

    revision = "a" * 40
    run_dir = tmp_path / "run"
    snapshot = tmp_path / "models--Qwen--Qwen3-0.6B" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    calls: list[str] = []
    report = {
        "python": "3.12.5",
        "torch": "2.6.0+cu124",
        "torch_cuda": "13.0",
        "cuda_available": True,
        "gpu_name": "NVIDIA GeForce RTX 3050 Ti Laptop GPU",
        "gpu_total_mib": 4096,
        "gpu_free_mib": 22000,
        "disk_free_gib": 90.0,
        "packages": {
            "sglang": "0.5.0",
            "llmcompressor": "0.13.0",
            "compressed-tensors": "0.18.0",
            "transformers": "5.14.1",
        },
    }
    monkeypatch.setattr(gating, "collect_preflight", lambda *_: report)
    monkeypatch.setattr(gating, "_fingerprint", lambda *_, **__: "fixed")

    def fake_server(command, *, log_path, **_):
        checkpoint = "gptq-w4a16" in command[command.index("--model-path") + 1]
        calls.append("gptq_service" if checkpoint else "bf16")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "Using MarlinLinearKernel for CompressedTensorsWNA16" if checkpoint else "BF16",
            encoding="utf-8",
        )
        return {
            "gpu_release_confirmed": True,
            "gpu_peak_sampled_mib": 3400,
            "peak_rss_bytes": 5 * 1024**3,
            "log_path": str(log_path),
            "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
            "content": "ready",
        }

    def fake_quant(
        model_dir, parent_revision, checkpoint, *, log_path, timeout_seconds, method="gptq"
    ):
        calls.append(method)
        assert model_dir == snapshot
        assert parent_revision == revision
        assert method in gating.SUPPORTED_QUANT_METHODS
        if checkpoint.exists():
            manifest = json.loads((checkpoint / "artifact-manifest.json").read_text(encoding="utf-8"))
            return {"checkpoint_path": str(checkpoint.resolve()), "artifact": manifest, "recovered": True}
        manifest = _write_packed_artifact(checkpoint, revision, method=method)
        return {
            "checkpoint_path": str(checkpoint.resolve()),
            "artifact": manifest,
            "recovered": False,
            "method": method,
            "gpu_release_confirmed": True,
            "gpu_peak_sampled_mib": 3400,
            "peak_rss_bytes": 5 * 1024**3,
        }

    monkeypatch.setattr(gating, "run_server_smoke", fake_server)
    monkeypatch.setattr(gating, "run_quantization_process", fake_quant)
    args = [
        "--run-dir", str(run_dir),
        "--model-dir", str(snapshot),
        "--revision", revision,
        "--stage", "all",
    ]
    assert gating.main(args) == 0
    assert calls == ["bf16", "gptq", "gptq_service"]
    assert gating.main(args) == 0
    assert calls == ["bf16", "gptq", "gptq_service"]
    (run_dir / "logs" / "gptq-server-1.log").write_text("tampered", encoding="utf-8")
    assert gating.main(args) == 0
    assert calls == ["bf16", "gptq", "gptq_service", "gptq_service"]
    state_path = run_dir / "run-status.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["stages"]["gptq"]["status"] = "blocked"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    assert gating.main(args) == 2
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["stages"]["gptq"]["status"] == "blocked"


def test_resource_gate_only_blocks_unmeasurable_or_unreleased_runs() -> None:
    """A transient high-water mark is a warning, not a block.

    Rationale measured in the quality run: the BF16 quality stage scored all its
    documents (PPL 30.13, 63/64 windows) with a healthy server, yet the sampled
    peak touched 3941 MiB during long-prompt prefill. Blocking there discarded a
    valid measurement because of *when the poller looked*, which is not a
    property of the run.
    """
    # No peak measured -> cannot tell what happened -> block.
    assert any(
        "measurement unavailable" in reason
        for reason in resource_blockers({"peak_rss_bytes": 1024**3})
    )
    # A healthy run above the concern line is not blocked...
    assert resource_blockers({"gpu_peak_sampled_mib": 3941}) == []
    # ...but it is recorded, so thin headroom stays visible.
    assert resource_warnings({"gpu_peak_sampled_mib": 3941})
    assert resource_warnings({"gpu_peak_sampled_mib": 3400}) == []
    # RSS never gates: WSL double-counts shared pages (17.9 GiB on a 7 GiB box).
    assert resource_blockers({"gpu_peak_sampled_mib": 3400, "peak_rss_bytes": 17 * 1024**3}) == []


def test_measurement_only_blocks_on_a_genuinely_occupied_card() -> None:
    """The block threshold rules out a real competing process, nothing subtler.

    Measured on this machine: used memory reads 0-21 MiB or a *persistent*
    669 MiB with zero compute processes attached (it did not clear over 60 s of
    observation). That 669 baseline appears in both a healthy run (TPOT
    4.42 ms) and a slow one (12.92 ms), so it is not by itself the cause of the
    slowdown. What must never happen is measuring while a real process holds the
    card, which costs far more than 669 MiB.
    """
    # The host's own ~669 MiB baseline is allowed through.
    assert contention_blockers({"gpu_used_before_mib": 669}) == []
    assert contention_blockers({"gpu_used_before_mib": 0}) == []
    assert contention_blockers({"gpu_used_before_mib": 21}) == []

    # A real occupier (e.g. a stale server holding 2.5 GiB) is blocked.
    blocked = contention_blockers({"gpu_used_before_mib": 2500})
    assert blocked and "only 1596 MiB" in blocked[0]

    # An unmeasured baseline is itself a problem: comparability is unknown.
    assert contention_blockers({})


def test_settle_wait_proceeds_at_the_host_baseline() -> None:
    """The wait tolerates the persistent ~669 MiB baseline.

    It is a short courtesy pause for a previous stage's context, not a remedy:
    669 MiB was measured to persist for minutes, so blocking on it would block
    every run on this machine.
    """
    used, settled = wait_for_clean_gpu(
        timeout_seconds=30, sampler=lambda: 669, sleeper=lambda _: None
    )
    assert settled is True
    assert used == 669

    # A genuinely busy card is waited on, then reported rather than hung on.
    used2, settled2 = wait_for_clean_gpu(
        timeout_seconds=0, sampler=lambda: 3000, sleeper=lambda _: None
    )
    assert settled2 is False
    assert used2 == 3000


def test_settle_wait_sleeps_while_the_card_is_busy() -> None:
    samples = iter([3000, 3000, 300])
    slept: list[float] = []
    used, settled = wait_for_clean_gpu(
        timeout_seconds=30, sampler=lambda: next(samples), sleeper=slept.append
    )
    assert settled is True
    assert used == 300
    assert len(slept) == 2  # waited while busy, returned as soon as it cleared


def test_settle_wait_treats_an_unreadable_gpu_as_settled() -> None:
    """No nvidia-smi reading is not evidence of contention."""
    used, settled = wait_for_clean_gpu(sampler=lambda: None, sleeper=lambda _: None)
    assert settled is True
    assert used is None


def test_gpu_release_waits_for_memory_to_return(monkeypatch) -> None:
    samples = iter([3000, 2100, 1000, 1000])
    monkeypatch.setattr(gating, "_sample_gpu_used_mib", lambda: next(samples))
    monkeypatch.setattr(gating.time, "sleep", lambda _: None)
    after, confirmed = gating._wait_gpu_release(1000, timeout_seconds=5)
    assert after == 1000
    assert confirmed is True


def test_gpu_release_rejects_one_transient_low_sample(monkeypatch) -> None:
    """A model still resident (baseline + 2 GiB) must not count as released.

    The tolerance must absorb baseline drift from the Windows host (measured
    ~700 MiB), so the not-released case is simulated at +2 GiB.
    """
    samples = iter([1000, 3000, 3000, 3000])
    monkeypatch.setattr(gating, "_sample_gpu_used_mib", lambda: next(samples))
    monotonic_values = iter([0, 1, 2, 3, 4, 5])
    monkeypatch.setattr(gating.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(gating.time, "sleep", lambda _: None)
    after, confirmed = gating._wait_gpu_release(1000, timeout_seconds=3)
    assert after == 3000
    assert confirmed is False


def test_failed_server_probe_records_post_shutdown_release(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(gating, "_sample_gpu_used_mib", lambda: 1000)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with pytest.raises(ProbeError, match="release_confirmed=True"):
        run_server_smoke(
            [sys.executable, "-c", "raise SystemExit(3)"],
            port=port,
            model_id="Qwen/Qwen3-0.6B",
            log_path=tmp_path / "failed-server.log",
            startup_timeout=2,
            request_timeout=1,
        )


def test_quant_worker_process_commits_checkpoint_and_recovers_it(tmp_path: Path, monkeypatch) -> None:
    worker = tmp_path / "fake_quant_worker.py"
    worker.write_text(
        """
import hashlib
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1])
checkpoint.mkdir(parents=True)
weights = checkpoint / 'model.safetensors'
weights.write_bytes(b'packed')
config = {
    'model_type': 'qwen3',
    'quantization_config': {
        'quant_method': 'compressed-tensors',
        'format': 'pack-quantized',
        'config_groups': {
            'group_0': {
                'targets': ['Linear'],
                'weights': {'num_bits': 4, 'group_size': 128, 'type': 'int'},
                'input_activations': None,
            }
        },
    },
}
(checkpoint / 'config.json').write_text(json.dumps(config), encoding='utf-8')
(checkpoint / 'tokenizer.json').write_text('{}', encoding='utf-8')
files = {
    name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
    for name in ('model.safetensors', 'config.json', 'tokenizer.json')
}
manifest = {
    'model_id': 'Qwen/Qwen3-0.6B',
    'parent_revision': 'a' * 40,
    'recipe': {'algorithm': 'GPTQ', 'scheme': 'W4A16', 'group_size': 128},
    'format': {
        'quant_method': 'compressed-tensors',
        'format': 'pack-quantized',
        'num_bits': 4,
        'group_size': 128,
        'safetensors_files': ['model.safetensors'],
    },
    'files': files,
}
(checkpoint / 'artifact-manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
""".strip(),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint"
    monkeypatch.setattr(
        gating,
        "build_quant_command",
        lambda *_, **__: [sys.executable, str(worker), str(checkpoint)],
    )
    monkeypatch.setattr(gating, "_sample_gpu_used_mib", lambda: 1000)
    first = run_quantization_process(
        tmp_path / "snapshot", "a" * 40, checkpoint,
        log_path=tmp_path / "quant-1.log", timeout_seconds=5,
    )
    assert first["recovered"] is False
    assert first["artifact"]["recipe"]["scheme"] == "W4A16"
    second = run_quantization_process(
        tmp_path / "snapshot", "a" * 40, checkpoint,
        log_path=tmp_path / "quant-2.log", timeout_seconds=5,
    )
    assert second["recovered"] is True
    assert not (tmp_path / "quant-2.log").exists()


def test_kernel_evidence_requires_specific_runtime_marker(tmp_path: Path) -> None:
    log = tmp_path / "server.log"
    log.write_text("Using MarlinLinearKernel for CompressedTensorsWNA16\n", encoding="utf-8")
    assert classify_quant_kernel(log) == "compressed_tensors_wna16_marlin"
    log.write_text("Loaded compressed-tensors checkpoint\n", encoding="utf-8")
    assert classify_quant_kernel(log) is None


def test_preflight_blocks_when_the_gpu_is_not_free_for_the_baseline() -> None:
    """The 4 GB card must have room before a 0.6B service is started."""
    report = {
        "python": "3.12.5",
        "torch": "2.6.0+cu124",
        "torch_cuda": "13.0",
        "cuda_available": True,
        "gpu_name": "NVIDIA GeForce RTX 3050 Ti Laptop GPU",
        "gpu_total_mib": 4096,
        "gpu_free_mib": 900,
        "disk_free_gib": 90.0,
        "packages": {
            "sglang": "0.5.0",
            "llmcompressor": "0.13.0",
            "compressed-tensors": "0.18.0",
            "transformers": "5.14.1",
        },
    }
    assert any("free GPU" in reason for reason in validate_preflight(report))
    report["gpu_free_mib"] = 3200
    report["ram_available_gib"] = 3.0
    assert any("available RAM" in reason for reason in validate_preflight(report))
    report["ram_available_gib"] = 6.0
    assert validate_preflight(report) == []


def test_gptq_stage_requires_a_verified_bf16_stage(tmp_path: Path, monkeypatch, capsys) -> None:
    report = {
        "python": "3.12.5",
        "torch": "2.6.0+cu124",
        "torch_cuda": "13.0",
        "cuda_available": True,
        "gpu_name": "NVIDIA GeForce RTX 3050 Ti Laptop GPU",
        "gpu_total_mib": 4096,
        "gpu_free_mib": 22000,
        "disk_free_gib": 90.0,
        "packages": {
            "sglang": "0.5.0",
            "llmcompressor": "0.13.0",
            "compressed-tensors": "0.18.0",
            "transformers": "5.14.1",
        },
    }
    monkeypatch.setattr(gating, "collect_preflight", lambda *_: report)
    monkeypatch.setattr(gating, "_fingerprint", lambda *_, **__: "fixed")
    code = gating.main(
        [
            "--run-dir", str(tmp_path / "run"),
            "--model-dir", str(tmp_path / "snapshot"),
            "--revision", "a" * 40,
            "--stage", "gptq",
        ]
    )
    assert code == 2
    assert "BF16" in capsys.readouterr().err
