from __future__ import annotations

import argparse
import json
from pathlib import Path

from japca.config import ensure_directory, load_paths_config


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _leaderboard_rows(paths_cfg: dict, suite: str) -> list[dict]:
    rows = []
    baseline_metrics = Path(paths_cfg["paths"]["baselines_dir"]) / suite / "metrics.json"
    if baseline_metrics.exists():
        payload = _load_json(baseline_metrics)
        for name, metrics in payload.get("models", {}).items():
            rows.append({"family": name, "composite_score": metrics.get("composite_score", 0.0), "source": str(baseline_metrics)})
    models_dir = Path(paths_cfg["paths"]["models_dir"])
    for candidate in models_dir.glob("*/metrics.json"):
        payload = _load_json(candidate)
        for split_name in ("dev", "test"):
            if split_name in payload:
                rows.append(
                    {
                        "family": f"{candidate.parent.name}_{split_name}",
                        "composite_score": payload[split_name].get("composite_score", 0.0),
                        "source": str(candidate),
                    }
                )
    return sorted(rows, key=lambda item: item["composite_score"], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate JAPCA milestone Markdown report.")
    parser.add_argument("--suite", required=True)
    args = parser.parse_args()

    paths = load_paths_config()
    reports_dir = ensure_directory(Path(paths["paths"]["reports_dir"]))
    leaderboard = _leaderboard_rows(paths, args.suite)
    baseline_metrics = Path(paths["paths"]["baselines_dir"]) / args.suite / "metrics.json"
    feature_lines = []
    if baseline_metrics.exists():
        payload = _load_json(baseline_metrics)
        importances = sorted(payload.get("feature_group_importance", []), key=lambda row: row["importance_delta"], reverse=True)
        for row in importances[:10]:
            feature_lines.append(f"| {row['holdout_year']} | {row['group']} | {row['importance_delta']:.4f} |")
    leaderboard_lines = [f"| {row['family']} | {row['composite_score']:.4f} | {row['source']} |" for row in leaderboard]
    markdown = "\n".join(
        [
            f"# JAPCA {args.suite} Summary",
            "",
            "## Leaderboard",
            "",
            "| Model | Composite score | Source |",
            "| --- | ---: | --- |",
            *leaderboard_lines,
            "",
            "## Top feature-group importance deltas",
            "",
            "| Holdout year | Group | Importance delta |",
            "| --- | --- | ---: |",
            *(feature_lines or ["| n/a | n/a | 0.0000 |"]),
        ]
    )
    output_path = reports_dir / f"{args.suite}_summary.md"
    output_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
