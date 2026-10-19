#!/usr/bin/env python3
"""Figures for reports/E0.md from logs/e0/*.json (missing runs are skipped)."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "logs", "e0")
OUT = os.path.join(ROOT, "assets", "e0")
ARMS = ["gt-plan", "zero-plan", "random-plan", "wrong-goal", "no-field"]


def load(name):
    p = os.path.join(LOG, f"{name}.json")
    if not os.path.exists(p):
        return None
    j = json.load(open(p))
    j["rate"] = float(np.mean(j["success"])) * 100
    return j


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def bar_with_ci(ax, xs, rates, ns, color, label, width=0.36, offset=0.0):
    lo, hi = zip(*[wilson(r / 100 * n, n) for r, n in zip(rates, ns)])
    err = np.array([np.array(rates) / 100 - np.array(lo),
                    np.array(hi) - np.array(rates) / 100]) * 100
    ax.bar(xs + offset, rates, width, yerr=err, capsize=3, color=color,
           label=label, edgecolor="black", linewidth=0.5)


def fig_arms():
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    xs = np.arange(len(ARMS))
    for seed, color, off in [(0, "#4C72B0", -0.19), (1, "#DD8452", 0.19)]:
        rates, ns = [], []
        for arm in ARMS:
            j = load(f"vec_{arm}_s{seed}")
            rates.append(j["rate"] if j else 0.0)
            ns.append(j["n"] if j else 1)
        bar_with_ci(ax, xs, rates, ns, color, f"train seed {seed}", offset=off)
        for x, r in zip(xs, rates):
            if r > 0:
                ax.text(x + off, r + 1.5, f"{r:.1f}", ha="center",
                        va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(ARMS, rotation=15)
    ax.set_ylabel("full success (%)")
    ax.set_ylim(0, 100)
    ax.set_title("E0 closed loop: 5 arms x 2 seeds (200 scenes), "
                 "95% Wilson CI")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "arms.png"), dpi=160)
    plt.close(fig)


def fig_diag():
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    # A: deterministic-noise gt vs zero
    ax = axes[0]
    labels, rates, ns, cols = [], [], [], []
    for seed in (0, 1):
        for arm, c in [("gt-plan", "#4C72B0"), ("zero-plan", "#55A868")]:
            j = load(f"det_{arm}_s{seed}")
            labels.append(f"{arm}\ns{seed}")
            rates.append(j["rate"] if j else 0.0)
            ns.append(j["n"] if j else 1)
            cols.append(c)
    bar_with_ci(ax, np.arange(4), rates, ns, "#888", "", width=0.7)
    for i, (r, c) in enumerate(zip(rates, cols)):
        ax.patches[i].set_facecolor(c)
        ax.text(i, r + 1.5, f"{r:.1f}", ha="center", fontsize=8)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 100)
    ax.set_ylabel("full success (%)")
    ax.set_title("deterministic noise\n(paired flow noise)", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # B: teacher vs predicted
    ax = axes[1]
    xs, rates, ns, cols, labs = [], [], [], [], []
    for seed in (0, 1):
        j = load(f"vec_gt-plan_s{seed}")
        t = load(f"vec_teacher_gt-plan_s{seed}")
        xs += [len(xs), len(xs)]
        rates += [j["rate"] if j else 0.0, t["rate"] if t else 0.0]
        ns += [j["n"] if j else 1, t["n"] if t else 1]
        cols += ["#4C72B0", "#C44E52"]
        labs += [f"pred s{seed}", f"teacher s{seed}"]
    bar_with_ci(ax, np.arange(4), rates, ns, "#888", "", width=0.7)
    for i, (r, c) in enumerate(zip(rates, cols)):
        ax.patches[i].set_facecolor(c)
        ax.text(i, r + 1.5, f"{r:.1f}", ha="center", fontsize=8)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labs, fontsize=7)
    ax.set_ylim(0, 105)
    ax.set_title("high-level probe: predicted vs\noracle subgoals", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # C: field swap
    ax = axes[2]
    xs, rates, ns, cols, labs = [], [], [], [], []
    for seed in (0, 1):
        for tag, c in [("pred", "#4C72B0"), ("swap-random", "#DD8452"),
                       ("swap-wrong", "#8172B3")]:
            j = load(f"vec_{tag}_gt-plan_s{seed}")
            if j is None and tag == "pred":
                j = load(f"vec_gt-plan_s{seed}")
            if j is None:
                continue
            xs.append(len(xs))
            rates.append(j["rate"])
            ns.append(j["n"])
            cols.append(c)
            labs.append(f"{'own' if tag == 'pred' else tag.replace('swap-', '->')}"
                        f"\ns{seed}")
    bar_with_ci(ax, np.arange(len(rates)), rates, ns, "#888", "", width=0.7)
    for i, (r, c) in enumerate(zip(rates, cols)):
        ax.patches[i].set_facecolor(c)
        ax.text(i, r + 1.5, f"{r:.1f}", ha="center", fontsize=7)
    ax.set_xticks(np.arange(len(rates)))
    ax.set_xticklabels(labs, fontsize=6)
    ax.set_ylim(0, 100)
    ax.set_title("inference-time field swap\n(gt-plan checkpoints)", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "diagnostics.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_arms()
    fig_diag()
    print(f"wrote {OUT}/arms.png {OUT}/diagnostics.png")
