from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a Markdown report from D455 stream probe results.")
    parser.add_argument("--results_path", required=True)
    parser.add_argument("--summary_path", required=True)
    parser.add_argument("--doc_json_path")
    parser.add_argument("--doc_md_path")
    return parser.parse_args()


def _fmt_bool(value: bool) -> str:
    return "pass" if value else "fail"


def _result_note(result: dict[str, Any]) -> str:
    if result["success"]:
        return result["recommendation_note"] if "recommendation_note" in result else "stable"
    if result["error_message"]:
        return result["error_message"]
    return "unstable"


def _markdown_table(results: list[dict[str, Any]], include_serial: bool) -> list[str]:
    lines = []
    headers = ["Stream Set", "Resolution", "Emitter", "Status", "Note"]
    if include_serial:
        headers.insert(0, "Serials")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for result in results:
        row = [
            result["stream_set"],
            f"{result['width']}x{result['height']}@{result['fps']}",
            result["emitter_request"],
            _fmt_bool(bool(result["success"])),
            _result_note(result),
        ]
        if include_serial:
            row.insert(0, ", ".join(result["serials"]))
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _group_results(data: dict[str, Any], topology_type: str) -> list[dict[str, Any]]:
    return [
        result
        for result in data["cases"]
        if result["topology_type"] == topology_type
    ]


def render_probe_report(
    *,
    results_path: Path,
    summary_path: Path,
    doc_json_path: Path | None = None,
    doc_md_path: Path | None = None,
) -> None:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    single_results = _group_results(data, "single")
    triple_results = _group_results(data, "three_camera")
    stable_results = [result for result in data["cases"] if result["success"]]
    failed_results = [result for result in data["cases"] if not result["success"]]
    key_errors = []
    seen_errors = set()
    for result in failed_results:
        error_key = (result.get("error_type"), result.get("error_message"))
        if error_key in seen_errors:
            continue
        seen_errors.add(error_key)
        key_errors.append(result)

    lines = [
        "# D455 Stream Probe Results",
        "",
        "Observed results are authoritative for this machine. Official docs define expectations, but support is claimed only when the probe passed.",
        "",
        "## Expected From Docs",
        "",
    ]
    for source in data["expectation_sources"]:
        lines.append(f"- `{source['name']}`: {source['expectation']} ({source['url']})")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Run id: `{data['run_id']}`",
            f"- Stable serial order: `{', '.join(data['stable_serial_order'])}`",
            f"- Total cases: `{len(data['cases'])}`",
            f"- Passed: `{len(stable_results)}`",
            f"- Failed: `{len(failed_results)}`",
            f"- Primary recommendation: `{data['recommendation']['primary_case']}` - {data['recommendation']['primary_statement']}",
            f"- Same-take comparison recommendation: `{data['recommendation']['comparison_case']}` - {data['recommendation']['comparison_statement']}",
            "",
            "## Single-Camera Results By Serial",
            "",
        ]
    )

    grouped_single: dict[str, list[dict[str, Any]]] = {}
    for result in single_results:
        grouped_single.setdefault(result["serials"][0], []).append(result)
    for serial, results in grouped_single.items():
        lines.append(f"### `{serial}`")
        lines.append("")
        lines.extend(_markdown_table(results, include_serial=False))
        lines.append("")

    lines.extend(
        [
            "## Three-Camera Results",
            "",
        ]
    )
    lines.extend(_markdown_table(triple_results, include_serial=True))
    lines.append("")

    lines.extend(["## Stable Stream Sets", ""])
    if stable_results:
        for result in stable_results:
            lines.append(
                f"- `{result['case_id']}`: `{result['stream_set']}` on `{', '.join(result['serials'])}` "
                f"at `{result['width']}x{result['height']}@{result['fps']}` with emitter `{result['emitter_request']}`"
            )
    else:
        lines.append("- none")
    lines.append("")

    lines.extend(["## Unstable / Failed Stream Sets", ""])
    if failed_results:
        for result in failed_results:
            lines.append(
                f"- `{result['case_id']}`: `{result['error_type']}` - {result['error_message']}"
            )
    else:
        lines.append("- none")
    lines.append("")

    lines.extend(["## Key Errors", ""])
    if key_errors:
        for result in key_errors:
            lines.append(f"- `{result['error_type']}`: {result['error_message']}")
    else:
        lines.append("- none")
    lines.append("")

    lines.extend(
        [
            "## Recommended Next Move",
            "",
            f"- Primary: `{data['recommendation']['primary_case']}` - {data['recommendation']['primary_statement']}",
            f"- Comparison feasibility: `{data['recommendation']['comparison_case']}` - {data['recommendation']['comparison_statement']}",
            "",
        ]
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    if doc_json_path is not None:
        doc_json_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(results_path, doc_json_path)
    if doc_md_path is not None:
        doc_md_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(summary_path, doc_md_path)


def main() -> int:
    args = parse_args()
    render_probe_report(
        results_path=Path(args.results_path).resolve(),
        summary_path=Path(args.summary_path).resolve(),
        doc_json_path=Path(args.doc_json_path).resolve() if args.doc_json_path else None,
        doc_md_path=Path(args.doc_md_path).resolve() if args.doc_md_path else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
