"""Incrementally summarize the auditable Exp-B early-training screen."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("./Code/Exp-B/reporting-v1")
OUT = ROOT / "analysis"
METRICS = ("state_tokens/coverage", "critic/rewards/mean", "env/finish_ratio", "timing_s/gen")


def parse_log(run: Path) -> dict[str, float]:
    text = (run / "logs" / "train.log").read_text(encoding="utf-8", errors="replace")
    line = next(line for line in text.splitlines() if "step:1 -" in line)
    values = dict(re.findall(r"([A-Za-z0-9_./]+):(-?[0-9.]+)", line))
    row = {metric: float(values[metric]) for metric in METRICS}
    matches = re.findall(r"step:2 - val/test_score/locomo:([0-9.]+)", text)
    row["val_test_score_locomo"] = float(matches[-1])
    return row


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows: list[dict[str, float | int]] = []
    for run in sorted((ROOT / "runs").glob("screen_q3x3x4_h*_seed*")):
        if not (run / "__SUCCESS__").exists():
            continue
        match = re.fullmatch(r"screen_q3x3x4_h([0-9]+p[0-9]+)_seed([0-9]+)", run.name)
        if not match:
            continue
        row: dict[str, float | int] = {"H": float(match.group(1).replace("p", ".")), "seed": int(match.group(2))}
        row.update(parse_log(run))
        rows.append(row)
    rows.sort(key=lambda row: (float(row["H"]), int(row["seed"])))
    fields = ["H", "seed", *METRICS, "val_test_score_locomo"]
    with (OUT / "trial_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)

    grouped: dict[float, list[dict[str, float | int]]] = defaultdict(list)
    for row in rows: grouped[float(row["H"])] .append(row)
    summary: list[dict[str, float | int]] = []
    for h in sorted(grouped):
        condition = grouped[h]
        result: dict[str, float | int] = {"H": h, "n": len(condition)}
        for metric in (*METRICS, "val_test_score_locomo"):
            values = np.array([float(row[metric]) for row in condition])
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(result)
    summary_fields = ["H", "n", *[item for metric in (*METRICS, "val_test_score_locomo") for item in (f"{metric}_mean", f"{metric}_std")]]
    with (OUT / "h_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader(); writer.writerows(summary)

    report = [
        "# Exp-B Early-Training Screen (Preliminary)",
        "",
        "Fixed Q=(m=3,n=3,l=4,k=1); 16 deterministic training examples; 16 held-out validation examples; one GRPO step; pre-training validation disabled.",
        "`val_test_score_locomo` is the framework's raw aggregate, not F1. Values are updated only after a run has `__SUCCESS__`.",
        "",
        "| H | n | State coverage (mean +/- sd) | Rollout reward (mean +/- sd) | Finish ratio (mean +/- sd) | Raw validation (mean +/- sd) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        report.append(
            f"| {row['H']:.2f} | {row['n']} | {row['state_tokens/coverage_mean']:.3f} +/- {row['state_tokens/coverage_std']:.3f} | "
            f"{row['critic/rewards/mean_mean']:.3f} +/- {row['critic/rewards/mean_std']:.3f} | "
            f"{row['env/finish_ratio_mean']:.3f} +/- {row['env/finish_ratio_std']:.3f} | "
            f"{row['val_test_score_locomo_mean']:.3f} +/- {row['val_test_score_locomo_std']:.3f} |"
        )
    report.extend(["", "This is a screening study, not convergence-level or paper-final evidence. Error bars are sample standard deviations over completed seeds."])
    (OUT / "screen_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    if summary:
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
        series = (("state_tokens/coverage", "State-token coverage", "tab:blue"), ("critic/rewards/mean", "Mean rollout reward", "tab:orange"), ("val_test_score_locomo", "Raw final validation aggregate", "tab:green"))
        x = [float(row["H"]) for row in summary]
        for axis, (metric, label, color) in zip(axes, series):
            mean = [float(row[f"{metric}_mean"]) for row in summary]
            std = [float(row[f"{metric}_std"]) for row in summary]
            axis.errorbar(x, mean, yerr=std, marker="o", color=color, capsize=4, linewidth=2)
            axis.set_xlabel("Coverage target H"); axis.set_ylabel(label); axis.set_xticks(x)
        fig.suptitle(f"Mem-T Exp-B early-training screen (completed trials: {len(rows)}/15)", fontsize=11)
        fig.savefig(OUT / "screen_progress.png", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
