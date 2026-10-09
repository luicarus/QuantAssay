"""Offline report rendering from saved run records (mvp-prd.md §7).

The renderer never loads a model and never touches the network: it reads the
persisted contract objects and emits Markdown/HTML. Escaping is mandatory —
model output is untrusted text that must not break the report's structure.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from quantassay.contracts import ReportPaths, RegressionReport

def _esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _esc_md(text: Any) -> str:
    """Escape for Markdown, including pipe characters that would break tables."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_markdown(report: RegressionReport) -> str:
    """Human-readable report. Values mirror the JSON, never re-computed."""
    lines: list[str] = []
    add = lines.append

    add(f"# Serving regression report — run `{_esc_md(report.run_id)}`")
    add("")
    add(f"- comparability: **{'COMPARABLE' if report.comparable else 'NOT COMPARABLE'}**")
    add(f"- quality status: **{_esc(report.quality_status)}**")
    add("")

    if not report.comparable:
        add("## Why this comparison is not possible")
        add("")
        add("| field | baseline | candidate | reason |")
        add("|---|---|---|---|")
        for issue in report.incomparable_reasons:
            add(
                f"| {_esc_md(issue.field)} | {_esc_md(issue.base_value or '—')} | "
                f"{_esc_md(issue.candidate_value or '—')} | {_esc_md(issue.reason)} |"
            )
        add("")
        add("No performance conclusion may be drawn from this run.")
        return "\n".join(lines) + "\n"

    add("## Per-metric comparison")
    add("")
    add("| metric | baseline | candidate | delta | change | direction |")
    add("|---|---:|---:|---:|---:|---|")
    directions = {
        "ttft": "lower is better",
        "tpot": "lower is better",
        "itl": "lower is better",
        "output_tokens_per_sec": "higher is better",
        "requests_per_sec": "higher is better",
    }
    for metric, regression in report.regressions.items():
        base_v = "unavailable" if regression.base_value is None else f"{regression.base_value:g}"
        cand_v = (
            "unavailable" if regression.candidate_value is None else f"{regression.candidate_value:g}"
        )
        delta = "—" if regression.absolute_delta is None else f"{regression.absolute_delta:+g}"
        change = (
            "—"
            if regression.relative_change_percent is None
            else f"{regression.relative_change_percent:+.2f}%"
        )
        add(
            f"| {_esc(metric)} | {base_v} | {cand_v} | {delta} | {change} | "
            f"{_esc(directions.get(metric, ''))} |"
        )
    add("")

    if report.baseline and report.candidate:
        add("## Request coverage")
        add("")
        add("| side | total | ok | failed | timeout | success rate |")
        add("|---|---:|---:|---:|---:|---:|")
        for summary in (report.baseline, report.candidate):
            rate = summary.success_rate
            rate_s = "—" if rate is None else f"{rate:.0%}"
            add(
                f"| {_esc(summary.side)} | {summary.requests_total} | {summary.requests_ok} | "
                f"{summary.requests_failed} | {summary.requests_timeout} | {rate_s} |"
            )
        add("")

        # Output length and termination (mvp-prd.md §5): a truncated side decoded
        # under a different regime, so this is disclosed next to the percentages.
        add("## Output length and termination")
        add("")
        add("| side | out tokens min/p50/max | finish reasons | truncated | stopped on EOS |")
        add("|---|---|---|---:|---:|")
        for summary in (report.baseline, report.candidate):
            lengths = (
                f"{summary.output_tokens_min} / {summary.output_tokens_p50:g} / "
                f"{summary.output_tokens_max}"
                if summary.output_tokens_p50 is not None
                else "—"
            )
            reasons = (
                ", ".join(f"{k}×{v}" for k, v in sorted(summary.finish_reason_counts.items()))
                or "—"
            )
            add(
                f"| {_esc(summary.side)} | {lengths} | {_esc(reasons)} | "
                f"{summary.truncated_requests} | {summary.stopped_on_eos} |"
            )
        add("")
        if max(report.baseline.truncated_requests, report.candidate.truncated_requests) > 0:
            add(
                f"> **Truncation notice:** {report.baseline.truncated_requests} baseline and "
                f"{report.candidate.truncated_requests} candidate requests hit `max_tokens` "
                "rather than stopping on EOS. Per-token latency still compares like for "
                "like within those requests, but a larger `max_new_tokens` would show "
                "whether the gap persists at full length."
            )
            add("")

    add("## Quality")
    add("")
    if report.quality_not_evaluated or report.quality is None:
        add("quality was **not evaluated** in this run: no accuracy, perplexity or")
        add("output-consistency claim is supported.")
        add("")
    else:
        comparison = report.quality
        if not comparison.comparable:
            add("quality was measured but the two sides are **not comparable**:")
            add("")
            add("| field | baseline | candidate | reason |")
            add("|---|---|---|---|")
            for issue in comparison.incomparable_reasons:
                add(
                    f"| {_esc_md(issue.field)} | {_esc_md(issue.base_value or '—')} | "
                    f"{_esc_md(issue.candidate_value or '—')} | {_esc_md(issue.reason)} |"
                )
            add("")
            add("No quality difference may be quoted from this run.")
            add("")
        else:
            add("| side | documents | valid tokens | perplexity |")
            add("|---|---:|---:|---:|")
            for summary in (comparison.baseline, comparison.candidate):
                if summary is None:
                    continue
                ppl = "—" if summary.ppl is None else f"{summary.ppl:.4f}"
                add(
                    f"| {_esc(summary.side)} | {summary.documents_scored}/"
                    f"{summary.documents_total} | {summary.valid_tokens} | {ppl} |"
                )
            add("")
            if comparison.ppl_relative_change is not None:
                ci = ""
                if comparison.ci_low is not None and comparison.ci_high is not None:
                    ci = (
                        f" (paired {comparison.ci_confidence:.0%} interval "
                        f"{comparison.ci_low * 100:+.2f}% … "
                        f"{comparison.ci_high * 100:+.2f}%, "
                        f"n={comparison.paired_documents})"
                    )
                add(
                    f"**Perplexity change:** {comparison.ppl_relative_change * 100:+.2f}%"
                    f"{ci} — positive means the candidate is *worse*."
                )
                add("")
                if comparison.directional_claim:
                    add(f"> {_esc(comparison.directional_claim)}")
                    add("")
            add("Perplexity is pooled from NLL sums over valid target tokens;")
            add("per-document perplexities are never averaged. Task accuracy, EM/F1 and")
            add("output consistency were **not** measured.")
            add("")

    add("## Limits of this evidence")
    add("")
    if report.quality_not_evaluated or report.quality is None:
        add("- quality was **not evaluated**; no accuracy/quality claim is supported")
    else:
        add("- quality here is held-out perplexity only; it is not task accuracy")
    add("- conclusions are scoped to this workload on this GPU and SGLang version")
    add("- negative percent on latency metrics means faster; positive percent on")
    add("  throughput metrics means higher throughput — the signs are not comparable")
    add("")

    if report.notes:
        add("## Notes")
        add("")
        for note in report.notes:
            add(f"- {_esc(note)}")
        add("")

    return "\n".join(lines) + "\n"


def render_html(report: RegressionReport) -> str:
    """Offline HTML: no CDN, model text escaped, mirrors the Markdown values."""
    body: list[str] = []
    add = body.append

    add("<!DOCTYPE html>")
    add('<html lang="en"><head><meta charset="utf-8">')
    add("<title>Serving regression report</title>")
    add("<style>body{font-family:system-ui,sans-serif;max-width:60rem;margin:2rem auto;padding:0 1rem;}"
        "table{border-collapse:collapse;width:100%;margin:1rem 0;}"
        "th,td{border:1px solid #ccc;padding:.4rem .6rem;text-align:left;}"
        "th{background:#f5f5f5;}td.num{text-align:right;}"
        ".blocked{background:#fff3f3;border-left:4px solid #c0392b;padding:.75rem 1rem;}"
        ".ok{background:#f3fbf3;border-left:4px solid #2e7d32;padding:.75rem 1rem;}"
        "</style></head><body>")

    add(f"<h1>Serving regression report — run <code>{_esc(report.run_id)}</code></h1>")
    verdict = "COMPARABLE" if report.comparable else "NOT COMPARABLE"
    css = "ok" if report.comparable else "blocked"
    add(f'<p class="{css}">comparability: <strong>{verdict}</strong> — '
        f"quality status: <strong>{_esc(report.quality_status)}</strong></p>")

    if not report.comparable:
        add("<h2>Why this comparison is not possible</h2><table>"
            "<tr><th>field</th><th>baseline</th><th>candidate</th><th>reason</th></tr>")
        for issue in report.incomparable_reasons:
            add(
                f"<tr><td>{_esc(issue.field)}</td><td>{_esc(issue.base_value or '—')}</td>"
                f"<td>{_esc(issue.candidate_value or '—')}</td><td>{_esc(issue.reason)}</td></tr>"
            )
        add("</table><p>No performance conclusion may be drawn from this run.</p>")
        add("</body></html>")
        return "\n".join(body)

    add("<h2>Per-metric comparison</h2><table>")
    add("<tr><th>metric</th><th>baseline</th><th>candidate</th><th>delta</th>"
        "<th>change</th><th>direction</th></tr>")
    directions = {
        "ttft": "lower is better",
        "tpot": "lower is better",
        "itl": "lower is better",
        "output_tokens_per_sec": "higher is better",
        "requests_per_sec": "higher is better",
    }
    for metric, regression in report.regressions.items():
        base_v = "unavailable" if regression.base_value is None else f"{regression.base_value:g}"
        cand_v = (
            "unavailable" if regression.candidate_value is None else f"{regression.candidate_value:g}"
        )
        delta = "—" if regression.absolute_delta is None else f"{regression.absolute_delta:+g}"
        change = (
            "—" if regression.relative_change_percent is None
            else f"{regression.relative_change_percent:+.2f}%"
        )
        add(
            f"<tr><td>{_esc(metric)}</td><td class='num'>{base_v}</td>"
            f"<td class='num'>{cand_v}</td><td class='num'>{delta}</td>"
            f"<td class='num'>{change}</td><td>{_esc(directions.get(metric, ''))}</td></tr>"
        )
    add("</table>")

    if report.baseline and report.candidate:
        add("<h2>Request coverage</h2><table>")
        add("<tr><th>side</th><th>total</th><th>ok</th><th>failed</th>"
            "<th>timeout</th><th>success rate</th></tr>")
        for summary in (report.baseline, report.candidate):
            rate = summary.success_rate
            rate_s = "—" if rate is None else f"{rate:.0%}"
            add(
                f"<tr><td>{_esc(summary.side)}</td><td class='num'>{summary.requests_total}</td>"
                f"<td class='num'>{summary.requests_ok}</td>"
                f"<td class='num'>{summary.requests_failed}</td>"
                f"<td class='num'>{summary.requests_timeout}</td>"
                f"<td class='num'>{rate_s}</td></tr>"
            )
        add("</table>")

    add("<h2>Quality</h2>")
    if report.quality_not_evaluated or report.quality is None:
        add("<p>quality was <strong>not evaluated</strong>: no accuracy, perplexity "
            "or output-consistency claim is supported.</p>")
    else:
        comparison = report.quality
        if not comparison.comparable:
            add("<p>quality was measured but the two sides are "
                "<strong>not comparable</strong>:</p><table>"
                "<tr><th>field</th><th>baseline</th><th>candidate</th>"
                "<th>reason</th></tr>")
            for issue in comparison.incomparable_reasons:
                add(
                    f"<tr><td>{_esc(issue.field)}</td>"
                    f"<td>{_esc(issue.base_value or '—')}</td>"
                    f"<td>{_esc(issue.candidate_value or '—')}</td>"
                    f"<td>{_esc(issue.reason)}</td></tr>"
                )
            add("</table><p>No quality difference may be quoted from this run.</p>")
        else:
            add("<table><tr><th>side</th><th>documents</th><th>valid tokens</th>"
                "<th>perplexity</th></tr>")
            for summary in (comparison.baseline, comparison.candidate):
                if summary is None:
                    continue
                ppl = "—" if summary.ppl is None else f"{summary.ppl:.4f}"
                add(
                    f"<tr><td>{_esc(summary.side)}</td>"
                    f"<td class='num'>{summary.documents_scored}/"
                    f"{summary.documents_total}</td>"
                    f"<td class='num'>{summary.valid_tokens}</td>"
                    f"<td class='num'>{ppl}</td></tr>"
                )
            add("</table>")
            if comparison.ppl_relative_change is not None:
                ci = ""
                if comparison.ci_low is not None and comparison.ci_high is not None:
                    ci = (
                        f" (paired {comparison.ci_confidence:.0%} interval "
                        f"{comparison.ci_low * 100:+.2f}% … "
                        f"{comparison.ci_high * 100:+.2f}%, "
                        f"n={comparison.paired_documents})"
                    )
                add(
                    f"<p><strong>Perplexity change:</strong> "
                    f"{comparison.ppl_relative_change * 100:+.2f}%{ci} — positive "
                    "means the candidate is <em>worse</em>.</p>"
                )
            add("<p>Perplexity is pooled from NLL sums over valid target tokens; "
                "per-document perplexities are never averaged. Task accuracy, EM/F1 "
                "and output consistency were <strong>not</strong> measured.</p>")

    add("<h2>Limits of this evidence</h2><ul>")
    if report.quality_not_evaluated or report.quality is None:
        add("<li>quality was <strong>not evaluated</strong></li>")
    else:
        add("<li>quality here is held-out perplexity only; it is not task accuracy</li>")
    add("<li>conclusions are scoped to this workload, GPU and SGLang version</li>")
    add("<li>latency and throughput signs are not comparable</li>")
    add("</ul>")

    add("</body></html>")
    return "\n".join(body)


def render_report(
    report: RegressionReport,
    output_dir: str | Path,
) -> ReportPaths:
    """Write Markdown + HTML next to the run's other artifacts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    md_path = out / "report.md"
    md_path.write_text(render_markdown(report), encoding="utf-8")

    html_path = out / "report.html"
    html_path.write_text(render_html(report), encoding="utf-8")

    return ReportPaths(markdown_path=str(md_path), html_path=str(html_path))
