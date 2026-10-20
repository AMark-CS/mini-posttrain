"""Rebuild README_charts.png from retained run evidence; requires Matplotlib."""
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/verification"


def read_steps(stage):
    text = (EVIDENCE / f"{stage}_training.txt").read_text(encoding="utf-8")
    rows = re.findall(r"step (\d+)/(\d+) \| loss=([\d.]+)", text)
    assert len(rows) == 2 and [int(row[0]) for row in rows] == [1, 2]
    assert all(int(row[1]) == 2 for row in rows)
    return [float(row[2]) for row in rows]


def main():
    sft, dpo = read_steps("sft"), read_steps("dpo")
    tests = json.loads((EVIDENCE / "test_summary.json").read_text(encoding="utf-8"))
    assert sum(tests["default"].values()) == 17
    assert sum(tests["smoke"].values()) == 3
    text = (EVIDENCE / "sft_training.txt").read_text(encoding="utf-8")
    match = re.search(r"LoRA trainable params: (\d+) / (\d+)", text)
    assert match is not None
    trainable, total = map(int, match.groups())
    assert 0 < trainable < total
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
    fig.subplots_adjust(left=.09, right=.96, bottom=.24, top=.84, hspace=.70, wspace=.30)
    fig.suptitle("nano-posttrain | Local execution results", x=.09, ha="left",
                 fontsize=21, fontweight="bold", y=.97)
    fig.text(.09, .91, "2026-10-20 · Qwen2.5-0.5B · RTX 3060 · bf16 · LoRA r=16", fontsize=12)
    for ax, values, title, color, marker, upper in [
        (axes[0, 0], sft, "A  SFT: two logged batches", "#0072B2", "o", 3.3),
        (axes[0, 1], dpo, "B  DPO: two logged batches", "#9B4F00", "s", .85),
    ]:
        ax.scatter([1, 2], values, color=color, marker=marker, s=85, zorder=3)
        for step, value in enumerate(values, 1):
            ax.annotate(f"{value:.4f}", (step, value), xytext=(0, 13),
                        textcoords="offset points", ha="center", fontweight="bold")
        ax.set(title=title, xlabel="Optimizer step", ylabel="Logged training loss",
               xlim=(.5, 2.5), ylim=(0, upper), xticks=[1, 2])
        ax.grid(axis="y", alpha=.2)
    ax = axes[1, 0]
    left = [0, 0]
    for label, color, hatch in [("passed", "#0072B2", ""),
                                 ("failed", "#9B4F00", "//"),
                                 ("skipped", "#8A8A8A", "..")]:
        values = [tests[suite][label] for suite in ["default", "smoke"]]
        ax.barh([0, 1], values, left=left, color=color, hatch=hatch, label=label.title())
        for i, value in enumerate(values):
            if value:
                ax.text(left[i]+value/2, i, str(value), ha="center", va="center",
                        color="white", fontweight="bold")
        left = [a+b for a, b in zip(left, values)]
    ax.set(title="C  Test outcomes", xlabel="Test count",
           yticks=[0, 1], yticklabels=["Default", "Smoke"], xlim=(0, 18))
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.28), ncol=3, frameon=False, fontsize=9)
    ax = axes[1, 1]
    percentage = trainable / total * 100
    ax.barh([0], [percentage], color="#0072B2")
    ax.barh([0], [100-percentage], left=[percentage], color="#D8DDE3", hatch="//")
    ax.set(title="D  LoRA parameter allocation", xlabel="Share of parameters (%)",
           xlim=(0, 100), yticks=[])
    ax.text(.5, .68, f"{percentage:.2f}% trainable", transform=ax.transAxes,
            ha="center", fontweight="bold", color="#005483", bbox={"facecolor": "white", "edgecolor": "none", "pad": 3})
    ax.text(.5, .23, f"{trainable:,} / {total:,} parameters", transform=ax.transAxes,
            ha="center", fontsize=10, bbox={"facecolor": "white", "edgecolor": "none", "pad": 3})
    fig.text(.09, .035, "Two observations per objective; different batches, no uncertainty estimate.\n"
             "First-step LR = 0. Different loss scales; no convergence or model-quality claim.\n"
             "Test suites ran separately. Default failure: stale data path. Sources: docs/verification/.",
             fontsize=10, linespacing=1.5)
    fig.savefig(ROOT / "README_charts.png", dpi=160, facecolor="white",
                metadata={"Title": "nano-posttrain verified execution", "Description":
                          "Source logs and test counts in docs/verification; no smoothing or aggregation of loss."})
    plt.close(fig)


if __name__ == "__main__":
    main()
