"""Persisted metric status/value contract checks."""

import pytest

from quantassay.contracts import MetricRecord, MetricStatus


def test_unavailable_metric_cannot_carry_a_value() -> None:
    with pytest.raises(Exception):
        MetricRecord(name="ppl", value=0.0, status=MetricStatus.UNAVAILABLE, reason="n/a")
    with pytest.raises(Exception):
        MetricRecord(name="ppl", value=0.0, status=MetricStatus.NOT_SUPPORTED, reason="n/a")


def test_unavailable_metric_requires_reason() -> None:
    with pytest.raises(Exception):
        MetricRecord(name="ppl", status=MetricStatus.UNAVAILABLE)

    record = MetricRecord.unavailable("ppl", "zero valid tokens")
    assert record.value is None
    assert record.status is MetricStatus.UNAVAILABLE
    assert record.reason == "zero valid tokens"


def test_ok_metric_requires_value() -> None:
    with pytest.raises(Exception):
        MetricRecord(name="ppl", status=MetricStatus.OK)
    assert MetricRecord(name="ppl", value=7.1).value == 7.1
