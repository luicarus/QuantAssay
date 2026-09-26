"""Interrupt safety: an interrupted controller must not orphan its GPU child.

Discovered by the real recovery demo: SIGTERM killed the controller instantly
(Python's default disposition), so the `finally` that stops the worker never
ran. The orphaned worker kept 624 MiB of GPU memory and kept writing into the
run directory, which then made the resumed preflight fail on available RAM.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from quantassay import gating  # noqa: E402


class FakeProcess:
    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid
        self.waited = False
        self.killed = False

    def wait(self, timeout=None):
        self.waited = True
        return 0

    def poll(self):
        return None


def test_stop_process_group_signals_the_whole_session(monkeypatch) -> None:
    """The launcher may exit while children still hold GPU memory, so the
    session — not just the direct child — must be signalled."""
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        gating,
        "os",
        SimpleNamespace(name="posix", killpg=lambda pid, sig: signals.append((pid, sig))),
    )
    process = FakeProcess(pid=777)
    gating._stop_process_group(process)

    assert (777, signal.SIGTERM) in signals
    # Windows has no SIGKILL; the code falls back to signal 9.
    assert (777, getattr(signal, "SIGKILL", 9)) in signals


def test_reap_live_children_stops_every_tracked_child(monkeypatch) -> None:
    """A SIGTERM must reap all live children before the controller exits."""
    stopped: list[int] = []
    resent: list[int] = []
    monkeypatch.setattr(gating, "_stop_process_group", lambda p: stopped.append(p.pid))
    monkeypatch.setattr(gating.signal, "signal", lambda *_: None)
    monkeypatch.setattr(gating.os, "kill", lambda _pid, signum: resent.append(signum))

    gating._LIVE_CHILDREN.clear()
    gating._LIVE_CHILDREN.update({FakeProcess(1), FakeProcess(2)})
    try:
        gating._reap_live_children(signal.SIGTERM, None)
    finally:
        gating._LIVE_CHILDREN.clear()

    assert sorted(stopped) == [1, 2]
    # The signal is re-raised so the exit status still reports the interruption.
    assert resent == [signal.SIGTERM]


def test_tracked_process_registers_and_deregisters(monkeypatch, tmp_path: Path) -> None:
    process = FakeProcess()
    monkeypatch.setattr(gating.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(gating, "_install_child_reaper", lambda: None)

    log = tmp_path / "out.log"
    with log.open("wb") as handle:
        with gating._tracked_process(["true"], handle) as tracked:
            assert tracked is process
            assert process in gating._LIVE_CHILDREN

    assert process not in gating._LIVE_CHILDREN


def test_reaper_reaps_a_real_child_on_sigterm(tmp_path: Path) -> None:
    """End-to-end with real signals: the grandchild must not survive.

    Skipped on Windows, where the process-group semantics differ.
    """
    if os.name != "posix":
        pytest.skip("POSIX process groups required")

    script = tmp_path / "controller.py"
    script.write_text(
        """
import os, signal, subprocess, sys, time
sys.path.insert(0, {src!r})
from quantassay import gating

marker = {marker!r}
child = subprocess.Popen(
    [sys.executable, "-c", "import time; open(%r,'w').write('up'); time.sleep(120)" % marker],
    start_new_session=True,
)
gating._install_child_reaper()
gating._LIVE_CHILDREN.add(child)
print("READY", flush=True)
time.sleep(120)
""".format(src=str(REPO_ROOT / "src"), marker=str(tmp_path / "child-alive")),
        encoding="utf-8",
    )

    controller = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        # Wait until the child has actually started.
        assert controller.stdout is not None
        for _ in range(100):
            line = controller.stdout.readline()
            if "READY" in line:
                break
        marker = tmp_path / "child-alive"
        for _ in range(100):
            if marker.exists():
                break
            import time

            time.sleep(0.05)
        assert marker.exists(), "child never started"

        controller.send_signal(signal.SIGTERM)
        controller.wait(timeout=30)
    finally:
        if controller.poll() is None:
            controller.kill()
        controller.wait(timeout=10)

    # The controller died from the signal; the tracked child must be gone too.
    import time

    time.sleep(1.0)
    probe = subprocess.run(
        ["pgrep", "-f", "child-alive"],
        capture_output=True,
        text=True,
    )
    survivors = [line for line in probe.stdout.split() if line and line != str(os.getpid())]
    assert not survivors, f"orphaned child survived: {survivors}"
