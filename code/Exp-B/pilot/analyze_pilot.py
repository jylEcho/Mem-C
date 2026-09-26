"""Create auditable, explicitly preliminary summaries for the Exp-B pilot."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("./Code/Exp-B/pilot")
RUNS = ROOT / "runs"
OUTPUT = ROOT / "analysis"
CONDITIONS = ((0.0, "h0p0"), (0.5, "h0p5"), (1.0, "h1p0"))
METRICS = (
    "state_tokens/coverage",
    "state_tokens/total",
    "critic/rewards/mean",
    "response_length/mean",
    "env/number_of_actions/mean",
    "env/finish_ratio",
    "timing_s/gen",
)


def step_one_metrics(text: str) -> dict[str, float]:
    line = next(line for line in text.splitlines() if "step:1 -" in line)
    values = dict(re.findall(r"([A-Za-z0-9_./]+):(-?[0-9.]+)", line))
    return {key: float(values[key]) for key in METRICS}


def final_validation(text: str) -> float:
    matches = re.findall(r"step:2 - val/test_score/locomo:([0-9.]+)", text)
    if not matches:
        raise ValueError("Missing final validation metric")
    return float(matches[-1])


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    rows: list[dict[str, float]] = []
    for h, tag in CONDITIONS:
        run = RUNS / f"pilot_q3x3x4_{tag}_seed13"
        if not (run / "__SUCCESS__").exists():
            raise RuntimeError(f"Incomplete condition: {run.name}")
        text = (run / "logs" / "train.log").read_text(encoding="utf-8", errors="replace")
        row: dict[str, float] = {"H": h, **step_one_metrics(text)}
        row["val_test_score_locomo"] = final_validation(text)
        rows.append(row)

    columns = ["H", *METRICS, "val_test_score_locomo"]
    with (OUTPUT / "exp_b_pilot_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    table = [
        "# Exp-B Pilot Results (Preliminary, Non-final)",
        "",
        "Fixed Q=(m=3, n=3, l=4, k=1); H={0.0, 0.5, 1.0}; seed=13; 8 fixed training samples; one GRPO step; pre-training validation disabled.",
        "The table reports directly parsed trainer logs. `val_test_score_locomo` is the framework's raw final validation aggregate, not an F1 score.",
        "",
        "| H | State coverage | Mean reward | Mean response tokens | Mean actions | Finish ratio | Generation s | Raw final validation |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        table.append(
            f"| {row['H']:.2f} | {row['state_tokens/coverage']:.3f} | "
            f"{row['critic/rewards/mean']:.3f} | {row['response_length/mean']:.1f} | "
            f"{row['env/number_of_actions/mean']:.3f} | {row['env/finish_ratio']:.3f} | "
            f"{row['timing_s/gen']:.1f} | {row['val_test_score_locomo']:.3f} |"
        )
    table.extend([
        "",
        "Interpretation constraint: this one-seed, one-step pilot is suitable only for pipeline validation and an exploratory three-point plot. It cannot establish an optimum H or support a paper-level comparative claim.",
    ])
    (OUTPUT / "exp_b_pilot_report.md").write_text("\n".join(table) + "\n", encoding="utf-8")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    series = (
        ("state_tokens/coverage", "State-token coverage", "tab:blue"),
        ("critic/rewards/mean", "Mean rollout reward", "tab:orange"),
        ("val_test_score_locomo", "Raw final validation aggregate", "tab:green"),
    )
    x = [row["H"] for row in rows]
    for axis, (metric, label, color) in zip(axes, series):
        y = [row[metric] for row in rows]
        axis.plot(x, y, marker="o", linewidth=2, color=color)
        axis.set_xlabel("Coverage target H")
        axis.set_ylabel(label)
        axis.set_xticks(x)
    fig.suptitle("Mem-T Exp-B pilot: fixed Q, one seed, one GRPO step (preliminary)", fontsize=11)
    fig.savefig(OUTPUT / "exp_b_pilot_preliminary.png", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
