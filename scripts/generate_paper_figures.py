from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"
EXP1_DIR = ROOT / "results" / "paper" / "exp1_baseline"
EXP2_DIR = ROOT / "results" / "paper" / "exp2_ablation"


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "legend.fontsize": 7.6,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


METHODS = {
    "llm_episodic": "LLM+Episodic",
    "llm_only": "LLM-only",
    "bayesian": "Bayesian",
    "tpe_mv": "TPE+MV",
    "random": "Random",
    "legacy_full": "Legacy-Full",
    "ccrc_full": "CCRC-Full",
    "ccrc_no_episodic": "CCRC-NoEpisodic",
    "ccrc_no_coupled_risk": "CCRC-NoCoupledRisk",
}


COLORS = {
    "LLM+Episodic": "#007C7C",
    "LLM-only": "#1F4E79",
    "Bayesian": "#6B7280",
    "TPE+MV": "#D55E00",
    "Random": "#B23A48",
    "Legacy-Full": "#374151",
    "CCRC-Full": "#007C7C",
    "CCRC-NoEpisodic": "#B23A48",
    "CCRC-NoCoupledRisk": "#D55E00",
}


MARKERS = {
    "LLM+Episodic": "o",
    "LLM-only": "s",
    "Bayesian": "^",
    "TPE+MV": "D",
    "Random": "P",
    "Legacy-Full": "o",
    "CCRC-Full": "s",
    "CCRC-NoEpisodic": "^",
    "CCRC-NoCoupledRisk": "D",
}


def read_convergence(result_dir: Path, method_label: str) -> pd.DataFrame:
    by_seed = pd.read_csv(result_dir / "convergence_by_seed.csv")
    df = by_seed[by_seed["method"].eq(method_label)]
    grouped = (
        df.groupby("trial")["best_so_far"]
        .agg(mean_best_so_far="mean", std_best_so_far="std", n_seeds="count")
        .reset_index()
    )
    grouped["std_best_so_far"] = grouped["std_best_so_far"].fillna(0.0)
    required = {"trial", "mean_best_so_far", "std_best_so_far", "n_seeds"}
    missing = required.difference(grouped.columns)
    if missing:
        raise ValueError(f"{result_dir} is missing columns: {sorted(missing)}")
    return grouped


def read_summary(result_dir: Path, method_key: str) -> dict[str, float]:
    label = METHODS[method_key]
    conv = read_convergence(result_dir, label)
    aggregate = pd.read_csv(result_dir / "aggregate_metrics.csv")
    row = aggregate[aggregate["method"].eq(label)].iloc[0]
    return {
        "mean_accuracy": float(row["mean_acc"]),
        "std_accuracy": float(row["std"]),
        "final_bsf": float(conv["mean_best_so_far"].iloc[-1]),
        "auc_bsf": float(conv["mean_best_so_far"].mean()),
    }


def plot_exp1() -> None:
    methods = ["llm_episodic", "llm_only", "bayesian", "tpe_mv", "random"]

    fig, ax = plt.subplots(figsize=(6.9, 3.55))
    for method in methods:
        label = METHODS[method]
        df = read_convergence(EXP1_DIR, label)
        se = df["std_best_so_far"] / np.sqrt(df["n_seeds"].clip(lower=1))
        color = COLORS[label]
        ax.plot(
            df["trial"],
            df["mean_best_so_far"],
            label=label,
            color=color,
            linewidth=2.0 if method == "llm_episodic" else 1.65,
            marker=MARKERS[label],
            markevery=5,
            markersize=3.4,
        )
        ax.fill_between(
            df["trial"],
            df["mean_best_so_far"] - se,
            df["mean_best_so_far"] + se,
            color=color,
            alpha=0.12,
            linewidth=0,
        )

    ax.set_xlabel("Trial budget")
    ax.set_ylabel("Mean best-so-far validation accuracy")
    ax.set_xlim(1, 30)
    ax.set_ylim(0.40, 0.80)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.grid(True, axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.75)
    ax.legend(loc="lower right", frameon=False, ncol=1)
    fig.tight_layout(pad=0.6)
    save_figure(fig, "fig2_exp1_bsf_trajectory")


def plot_exp2() -> None:
    methods = [
        "legacy_full",
        "ccrc_full",
        "ccrc_no_episodic",
        "ccrc_no_coupled_risk",
    ]

    fig, ax_curve = plt.subplots(figsize=(6.8, 3.55))

    for method in methods:
        label = METHODS[method]
        df = read_convergence(EXP2_DIR, label)
        se = df["std_best_so_far"] / np.sqrt(df["n_seeds"].clip(lower=1))
        color = COLORS[label]
        ax_curve.plot(
            df["trial"],
            df["mean_best_so_far"],
            label=label,
            color=color,
            linewidth=1.8 if method != "ccrc_no_episodic" else 1.65,
            marker=MARKERS[label],
            markevery=5,
            markersize=3.2,
        )
        ax_curve.fill_between(
            df["trial"],
            df["mean_best_so_far"] - se,
            df["mean_best_so_far"] + se,
            color=color,
            alpha=0.12,
            linewidth=0,
        )

    ax_curve.set_xlabel("Trial budget")
    ax_curve.set_ylabel("Mean best-so-far validation accuracy")
    ax_curve.set_xlim(1, 25)
    ax_curve.set_ylim(0.67, 0.78)
    ax_curve.set_xticks([1, 5, 10, 15, 20, 25])
    ax_curve.grid(True, axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.75)
    ax_curve.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.86,
        handlelength=2.0,
    )
    fig.tight_layout(pad=0.6)
    save_figure(fig, "fig3_exp2_bsf_trajectory")

    fig, ax_metrics = plt.subplots(figsize=(6.8, 3.35))
    metric_rows = []
    for method in methods:
        metric_rows.append((METHODS[method], read_summary(EXP2_DIR, method)))

    labels = [row[0] for row in metric_rows]
    y = np.arange(len(labels))
    offsets = [-0.18, 0.0, 0.18]
    metric_specs = [
        ("Mean trial acc.", "mean_accuracy", "#6B7280", "o"),
        ("Final BSF", "final_bsf", "#007C7C", "s"),
        ("AUC-BSF", "auc_bsf", "#D55E00", "D"),
    ]
    for offset, (metric_label, key, color, marker) in zip(offsets, metric_specs):
        values = [row[1][key] for row in metric_rows]
        ax_metrics.scatter(values, y + offset, label=metric_label, color=color, marker=marker, s=28)

    ax_metrics.set_xlabel("Validation accuracy")
    ax_metrics.set_yticks(y)
    ax_metrics.set_yticklabels(labels)
    ax_metrics.invert_yaxis()
    ax_metrics.set_xlim(0.64, 0.775)
    ax_metrics.grid(True, axis="x", color="#D1D5DB", linewidth=0.6, alpha=0.75)
    ax_metrics.legend(loc="upper left", frameon=False)

    fig.subplots_adjust(left=0.33, right=0.985, top=0.96, bottom=0.16)
    save_figure(fig, "fig4_exp2_component_effect")


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = FIGURE_DIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(path.relative_to(ROOT))
    plt.close(fig)


def main() -> None:
    plot_exp1()
    plot_exp2()


if __name__ == "__main__":
    main()
