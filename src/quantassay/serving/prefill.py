"""Join source-side Prefill preparation events to the timed client workload."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

MARKER = "QUANTASSAY_PREFILL_EVENT "


def collect_prefill_evidence(result: dict[str, Any], rows: list[dict[str, Any]],
                             log_path: Path, output_dir: Path) -> dict[str, Any]:
    from quantassay.serving.scheduling import append_jsonl, distribution

    ids = {r["request_id"] for r in rows}
    events, invalid = [], 0
    for number, line in enumerate(log_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if MARKER not in line:
            continue
        try:
            event = json.loads(line.split(MARKER, 1)[1])
        except (ValueError, TypeError):
            invalid += 1
            continue
        linked = set(event.get("request_ids", [])) | set(event.get("candidate_request_ids", []))
        linked.add(event.get("chunk_request_id"))
        aligned = (result.get("clock_id") == event.get("clock_id")
                   and result.get("origin_monotonic_seconds") is not None)
        elapsed = (event["server_monotonic_seconds"] - result["origin_monotonic_seconds"]
                   if aligned else None)
        event.update(server_log_line=number, elapsed_seconds=elapsed,
                     timed_workload=bool(ids & linked), clock_alignment_available=aligned)
        append_jsonl(output_dir / "prefill-events.jsonl", event)
        if event["timed_workload"]:
            events.append(event)
    prepared = [e for e in events if e["status"] == "prepared"]
    blocked = [e for e in events if e["status"] == "blocked"]
    if not prepared:
        return {"status": "unavailable", "reason": "no timed prepared-prefill source events",
                "invalid_log_events": invalid, "prepared_events": 0}
    budget_counts = Counter(str(e["gross_chunk_budget_tokens"]) for e in prepared)
    pressure_counts = Counter(str(e["active_decode_reqs"]) for e in prepared)
    pressure_budgets = {}
    kv = {}
    for event in prepared:
        active = str(event["active_decode_reqs"])
        pressure_budgets.setdefault(active, Counter())[str(event["gross_chunk_budget_tokens"])] += 1
    for event in events:
        for moment in ("kv_before", "kv_after"):
            snapshot = event.get(moment) or {}
            for pool, values in snapshot.get("pools", {}).items():
                for field, value in values.items():
                    kv.setdefault(f"{pool}.{field}", []).append(value)
    phases = {}
    for phase in result.get("arrival_phases", []):
        selected = [e for e in prepared if e["elapsed_seconds"] is not None
                    and phase["start_seconds"] <= e["elapsed_seconds"] < phase["end_seconds"]]
        phases[phase["name"]] = {
            "prepared_events": len(selected),
            "budget_histogram": dict(Counter(str(e["gross_chunk_budget_tokens"]) for e in selected)),
            "active_decode_histogram": dict(Counter(str(e["active_decode_reqs"]) for e in selected)),
        }
    return {
        "status": "ok" if not invalid else "incomplete", "invalid_log_events": invalid,
        "prepared_events": len(prepared), "blocked_events": len(blocked),
        "prepared_budget_histogram": dict(budget_counts),
        "prepared_active_decode_histogram": dict(pressure_counts),
        "prepared_budgets_by_active_decode": {k: dict(v) for k, v in pressure_budgets.items()},
        "adaptive_budget_varied": any(e.get("adaptive") for e in prepared) and len(budget_counts) > 1,
        "logical_prefill_tokens": distribution([e["logical_prefill_tokens"] for e in prepared]),
        "mixed_decode_reqs": distribution([e["mixed_decode_reqs"] for e in prepared]),
        "prepared_stop_reasons": dict(Counter(e["stop_reason"] or "queue_exhausted" for e in prepared)),
        "blocked_stop_reasons": dict(Counter(e["stop_reason"] for e in blocked)),
        "kv_token_ranges": {k: {"min": min(v), "max": max(v)} for k, v in kv.items()},
        "phases": phases,
        "scope": "host prepared batches; repeated identical block states suppressed; event counts are not request counts",
        "eviction_count": {"status": "unavailable", "reason": "allocator eviction is not instrumented"},
    }
