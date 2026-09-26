"""The §5.8 anomaly flag must not cry wolf, and must not be oversold.

Historical peaks (compatibility.md §5.8), with each side judged against its own
healthy baseline (bf16 ~7.9 ms, gptq ~4.3 ms):

    healthy   bf16 2877, 2898, 3474    (3474 came with a normal 7.845 ms)
    healthy   gptq 2891, 2912
    anomalous gptq 3560                (12.894 ms = 3.0x healthy)

The 3474 sample is why the line cannot be tight: a threshold low enough to catch
milder anomalies would have flagged a healthy run. The flag is therefore weak by
construction, and the tests pin both properties.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from quantassay.gating import PEAK_ANOMALY_MIB, anomaly_warnings  # noqa: E402

#: (side, peak MiB, measured TPOT ms, was it anomalous?)
HISTORICAL = [
    ("bf16", 2877, 7.742, False),
    ("bf16", 2898, 8.105, False),
    ("bf16", 3474, 7.845, False),   # high peak, healthy speed: the hard case
    ("gptq", 2891, 4.277, False),
    ("gptq", 2912, 4.364, False),
    ("gptq", 3560, 12.894, True),
]

HEALTHY_TPOT = {"bf16": 7.9, "gptq": 4.3}


def test_flag_never_fires_on_a_healthy_sample() -> None:
    """A false alarm would train readers to ignore the flag."""
    false_alarms = []
    for side, peak, tpot, anomalous in HISTORICAL:
        if anomalous:
            continue
        if anomaly_warnings({"gpu_peak_sampled_mib": peak}, side=side):
            false_alarms.append(f"{side} peak={peak} tpot={tpot} (healthy)")
    assert false_alarms == [], (
        "the anomaly flag fired on healthy runs:\n  " + "\n  ".join(false_alarms)
    )


def test_flag_fires_on_the_observed_anomaly() -> None:
    """It must at least catch the case it was built for."""
    warnings = anomaly_warnings({"gpu_peak_sampled_mib": 3560}, side="gptq")
    assert warnings
    assert "5.8" in warnings[0]


def test_flag_states_that_it_is_weak() -> None:
    """Over-selling the detector is worse than not having one."""
    message = anomaly_warnings({"gpu_peak_sampled_mib": 3560}, side="gptq")[0]
    assert "weak" in message
    assert "not evidence" in message


def test_threshold_sits_above_every_healthy_peak() -> None:
    """Calibration invariant: above the healthy band, or healthy runs get flagged."""
    healthy_peaks = {"bf16": [], "gptq": []}
    for side, peak, _tpot, anomalous in HISTORICAL:
        if not anomalous:
            healthy_peaks[side].append(peak)
    for side, peaks in healthy_peaks.items():
        assert PEAK_ANOMALY_MIB[side] > max(peaks), (
            f"{side}: threshold {PEAK_ANOMALY_MIB[side]} is not above the observed "
            f"healthy peaks {peaks}; healthy runs would be flagged"
        )


def test_flag_is_advisory_not_a_gate() -> None:
    """An anomalous run is still valid evidence and must not be discarded.

    The flag returns warnings only; nothing in the gate path consumes it as a
    block. This pins that separation so a future change cannot start throwing
    away measurements.
    """
    import inspect

    from quantassay import gating

    source = inspect.getsource(gating.anomaly_warnings)
    assert "raise" not in source
    # And it must not appear in the hard-block helper.
    blocker_source = inspect.getsource(gating.resource_blockers)
    assert "anomaly" not in blocker_source


def test_missing_or_unknown_side_is_not_flagged() -> None:
    """No measurement, or an uncalibrated side, yields no claim."""
    assert anomaly_warnings({}, side="gptq") == []
    assert anomaly_warnings({"gpu_peak_sampled_mib": 9999}, side="unknown") == []