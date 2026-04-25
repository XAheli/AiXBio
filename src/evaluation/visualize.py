from __future__ import annotations

"""
Publication-quality visualization for function-aware screening results.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import roc_curve, precision_recall_curve

from src.config import FIGURES_DIR

logger = logging.getLogger(__name__)

# Paper-quality matplotlib settings
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.figsize": (6, 4.5),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

METHOD_COLORS = {
    "kmer_sim": "#999999",
    "cosine_nn_esm2": "#4DBEEE",
    "linear_esm2": "#77AC30",
    "knn_esm2": "#EDB120",
    "contrastive_esm2": "#D95319",
    "cosine_nn_protrek": "#7E2F8E",
    "linear_protrek": "#0072BD",
    "contrastive_protrek": "#A2142F",
}

METHOD_LABELS = {
    "kmer_sim": "K-mer Similarity (Homology Proxy)",
    "cosine_nn_esm2": "ESM-2 Cosine NN",
    "linear_esm2": "ESM-2 Linear",
    "knn_esm2": "ESM-2 KNN",
    "contrastive_esm2": "ESM-2 Contrastive (Ours)",
    "cosine_nn_protrek": "ProTrek Cosine NN",
    "linear_protrek": "ProTrek Linear",
    "contrastive_protrek": "ProTrek Contrastive (Ours)",
}


def plot_roc_curves(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    split_name: str = "test",
    output_dir: Path | None = None,
):
    """
    Plot ROC curves for all methods on a single split.

    Args:
        results: {method_name: (y_true, y_score)}
    """
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    for method_name, (y_true, y_score) in results.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, y_score)
        color = METHOD_COLORS.get(method_name, "#333333")
        label = METHOD_LABELS.get(method_name, method_name)
        linewidth = 2.5 if "contrastive" in method_name else 1.5
        linestyle = "-" if "contrastive" in method_name else "--"

        ax.plot(fpr, tpr, color=color, linewidth=linewidth, linestyle=linestyle,
                label=f"{label} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k:", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(f"ROC Curves — {split_name.replace('_', ' ').title()}")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    plt.tight_layout()
    plt.savefig(output_dir / f"roc_{split_name}.pdf")
    plt.savefig(output_dir / f"roc_{split_name}.png")
    plt.close()
    logger.info(f"Saved ROC plot: {output_dir / f'roc_{split_name}.pdf'}")


def plot_pr_curves(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    split_name: str = "test",
    output_dir: Path | None = None,
):
    """Plot Precision-Recall curves."""
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    for method_name, (y_true, y_score) in results.items():
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(y_true, y_score)
        color = METHOD_COLORS.get(method_name, "#333333")
        label = METHOD_LABELS.get(method_name, method_name)
        linewidth = 2.5 if "contrastive" in method_name else 1.5

        ax.plot(recall, precision, color=color, linewidth=linewidth,
                label=f"{label} (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall — {split_name.replace('_', ' ').title()}")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    plt.tight_layout()
    plt.savefig(output_dir / f"pr_{split_name}.pdf")
    plt.savefig(output_dir / f"pr_{split_name}.png")
    plt.close()


def plot_detection_vs_divergence(
    divergence_results: dict[str, dict],
    output_dir: Path | None = None,
):
    """
    THE KEY FIGURE: detection rate vs sequence divergence for each method.
    Shows that function-aware screening maintains detection where homology fails.
    """
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for method_name, bins_data in divergence_results.items():
        bin_labels = list(bins_data.keys())
        detection_rates = [bins_data[b]["detection_rate"] for b in bin_labels]
        n_samples = [bins_data[b]["n_samples"] for b in bin_labels]

        color = METHOD_COLORS.get(method_name, "#333333")
        label = METHOD_LABELS.get(method_name, method_name)
        linewidth = 2.5 if "contrastive" in method_name else 1.5
        marker = "o" if "contrastive" in method_name else "s"

        x = range(len(bin_labels))
        ax.plot(x, detection_rates, color=color, linewidth=linewidth,
                marker=marker, markersize=6, label=label)

        # Annotate sample counts
        for xi, n in zip(x, n_samples):
            ax.annotate(f"n={n}", (xi, detection_rates[xi]),
                       textcoords="offset points", xytext=(0, 10),
                       fontsize=7, alpha=0.6, ha="center")

    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, rotation=30, ha="right")
    ax.set_xlabel("K-mer Similarity to Nearest Training Threat")
    ax.set_ylabel("Detection Rate")
    ax.set_title("Detection Rate vs. Sequence Divergence")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_ylim([-0.05, 1.05])

    # Add annotation about the key insight
    ax.axhspan(0, 0.5, alpha=0.05, color="red")
    ax.text(len(bin_labels) - 1.5, 0.15, "Screening failure zone",
            fontsize=9, color="red", alpha=0.5, ha="center")

    plt.tight_layout()
    plt.savefig(output_dir / "detection_vs_divergence.pdf")
    plt.savefig(output_dir / "detection_vs_divergence.png")
    plt.close()
    logger.info("Saved detection vs divergence plot")


def plot_embedding_space(
    embeddings: np.ndarray,
    labels: np.ndarray,
    subcategories: np.ndarray | None = None,
    title: str = "Embedding Space",
    method_name: str = "",
    output_dir: Path | None = None,
):
    """
    2D UMAP visualization of the embedding/projection space.
    Shows threat vs benign clustering.
    """
    import umap

    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(7, 6))

    threat_mask = labels == 1
    benign_mask = labels == 0

    ax.scatter(coords[benign_mask, 0], coords[benign_mask, 1],
               c="#4DBEEE", s=20, alpha=0.5, label="Benign", edgecolors="none")
    ax.scatter(coords[threat_mask, 0], coords[threat_mask, 1],
               c="#D95319", s=30, alpha=0.7, label="Threat", edgecolors="none")

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    fname = f"umap_{method_name}" if method_name else "umap_embeddings"
    plt.savefig(output_dir / f"{fname}.pdf")
    plt.savefig(output_dir / f"{fname}.png")
    plt.close()


def plot_training_curves(history: dict, output_dir: Path | None = None):
    """Plot training loss and validation AUROC curves."""
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    ax1.plot(epochs, history["train_loss"], label="Total", color="#D95319", linewidth=2)
    ax1.plot(epochs, history["train_con"], label="Contrastive", color="#4DBEEE", linewidth=1.5, linestyle="--")
    ax1.plot(epochs, history["train_bce"], label="BCE", color="#77AC30", linewidth=1.5, linestyle="--")
    if history.get("val_loss"):
        ax1.plot(epochs[:len(history["val_loss"])], history["val_loss"],
                 label="Val Total", color="#D95319", linewidth=1.5, linestyle=":")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()

    # Validation AUROC
    if history.get("val_auroc"):
        ax2.plot(epochs[:len(history["val_auroc"])], history["val_auroc"],
                 color="#D95319", linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("AUROC")
        ax2.set_title("Validation AUROC")
        ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.pdf")
    plt.savefig(output_dir / "training_curves.png")
    plt.close()


def plot_results_heatmap(results_df: pd.DataFrame, metric: str = "AUROC",
                         output_dir: Path | None = None):
    """Heatmap of method x split performance."""
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    import seaborn as sns

    pivot = results_df.pivot(index="method", columns="split", values=metric)
    pivot = pivot.astype(float)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax,
                vmin=0.5, vmax=1.0, linewidths=0.5)
    ax.set_title(f"{metric} Across Methods and Evaluation Splits")
    ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(output_dir / f"heatmap_{metric.lower()}.pdf")
    plt.savefig(output_dir / f"heatmap_{metric.lower()}.png")
    plt.close()


def plot_mpnn_detection_vs_identity(
    method_results: dict[str, tuple[np.ndarray, np.ndarray]],
    seq_identities: np.ndarray,
    labels: np.ndarray,
    identity_bins: list[float] | None = None,
    output_dir: Path | None = None,
):
    """
    THE KEY MPNN FIGURE: detection rate as a function of sequence identity
    to the original threat protein.

    This shows whether function-aware screening can detect ProteinMPNN-designed
    variants even when sequence identity drops below 30%.

    Args:
        method_results: {method_name: (y_true, y_score)}
        seq_identities: per-sample sequence identity to source protein
        labels: ground truth labels (1=threat)
        identity_bins: upper bounds for identity bins
    """
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if identity_bins is None:
        identity_bins = [1.0, 0.9, 0.7, 0.5, 0.3, 0.2]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    threat_mask = labels == 1

    for method_name, (y_true, y_score) in method_results.items():
        threat_scores = y_score[threat_mask]
        threat_identities = seq_identities[threat_mask]

        bin_centers = []
        detection_rates = []
        n_samples_list = []

        prev_bound = 1.01
        for upper in identity_bins:
            mask = (threat_identities < prev_bound) & (threat_identities >= upper)
            if mask.sum() > 0:
                det_rate = (threat_scores[mask] >= 0.5).mean()
                bin_center = (prev_bound + upper) / 2
                bin_centers.append(f"{upper:.0%}-{prev_bound:.0%}" if prev_bound <= 1.0 else f">{upper:.0%}")
                detection_rates.append(det_rate)
                n_samples_list.append(int(mask.sum()))
            prev_bound = upper

        # Below lowest bin
        mask = threat_identities < identity_bins[-1]
        if mask.sum() > 0:
            det_rate = (threat_scores[mask] >= 0.5).mean()
            bin_centers.append(f"<{identity_bins[-1]:.0%}")
            detection_rates.append(det_rate)
            n_samples_list.append(int(mask.sum()))

        color = METHOD_COLORS.get(method_name, "#333333")
        label = METHOD_LABELS.get(method_name, method_name)
        linewidth = 2.5 if "contrastive" in method_name else 1.5
        marker = "o" if "contrastive" in method_name else "s"

        x = range(len(bin_centers))
        ax.plot(x, detection_rates, color=color, linewidth=linewidth,
                marker=marker, markersize=7, label=label)

        # Annotate sample counts
        for xi, (dr, n) in enumerate(zip(detection_rates, n_samples_list)):
            ax.annotate(f"n={n}", (xi, dr),
                       textcoords="offset points", xytext=(0, 12),
                       fontsize=7, alpha=0.5, ha="center")

    ax.set_xticks(range(len(bin_centers)))
    ax.set_xticklabels(bin_centers, rotation=25, ha="right")
    ax.set_xlabel("Sequence Identity to Original Threat (%)")
    ax.set_ylabel("Detection Rate")
    ax.set_title("Detection of ProteinMPNN-Designed Variants vs. Sequence Identity")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_ylim([-0.05, 1.05])

    # Danger zone annotation
    ax.axhspan(0, 0.5, alpha=0.06, color="red")
    ax.axvspan(len(bin_centers) - 2.5, len(bin_centers) - 0.5, alpha=0.06, color="orange")
    ax.text(len(bin_centers) - 1.5, 0.08, "Critical\nevasion\nzone",
            fontsize=8, color="red", alpha=0.6, ha="center", style="italic")

    plt.tight_layout()
    plt.savefig(output_dir / "mpnn_detection_vs_identity.pdf")
    plt.savefig(output_dir / "mpnn_detection_vs_identity.png")
    plt.close()
    logger.info("Saved MPNN detection vs identity plot")


def plot_identity_distribution(
    mpnn_df: "pd.DataFrame",
    output_dir: Path | None = None,
):
    """
    Plot the distribution of sequence identities for MPNN-designed variants,
    grouped by sampling temperature.
    """
    import seaborn as sns

    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if 'seq_identity_to_source' not in mpnn_df.columns or 'mpnn_temperature' not in mpnn_df.columns:
        logger.warning("MPNN DataFrame missing required columns for identity distribution plot")
        return

    threat_variants = mpnn_df[mpnn_df['subcategory'] == 'adversarial_mpnn'].copy()
    if len(threat_variants) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: violin plot by temperature
    temps = sorted(threat_variants['mpnn_temperature'].unique())
    violin_data = [
        threat_variants[threat_variants['mpnn_temperature'] == t]['seq_identity_to_source'].values
        for t in temps
    ]
    parts = ax1.violinplot(violin_data, positions=range(len(temps)), showmeans=True, showmedians=True)
    ax1.set_xticks(range(len(temps)))
    ax1.set_xticklabels([f"T={t}" for t in temps])
    ax1.set_ylabel("Sequence Identity to Original")
    ax1.set_title("MPNN Variant Diversity by Sampling Temperature")
    ax1.axhline(y=0.3, color="red", linestyle="--", alpha=0.5, label="30% identity threshold")
    ax1.legend()

    # Right: histogram of all identities
    ax2.hist(threat_variants['seq_identity_to_source'], bins=30, color="#D95319",
             alpha=0.7, edgecolor="black", linewidth=0.5)
    ax2.axvline(x=0.3, color="red", linestyle="--", alpha=0.7, label="30% identity")
    ax2.axvline(x=0.5, color="orange", linestyle="--", alpha=0.7, label="50% identity")
    ax2.set_xlabel("Sequence Identity to Original")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of MPNN Variant Identities")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "mpnn_identity_distribution.pdf")
    plt.savefig(output_dir / "mpnn_identity_distribution.png")
    plt.close()
    logger.info("Saved MPNN identity distribution plot")


def plot_detection_vs_divergence_with_ci(
    divergence_results: dict[str, dict[str, dict]],
    output_dir: Path | None = None,
):
    """
    Detection rate vs divergence with bootstrap CI error bars.

    Args:
        divergence_results: {method_name: {bin_label: {detection_rate, ci_lower, ci_upper, n_samples}}}
    """
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    bin_labels = None
    for method_name, bins_data in divergence_results.items():
        if bin_labels is None:
            bin_labels = list(bins_data.keys())
        det_rates = [bins_data[b]["detection_rate"] for b in bin_labels]
        ci_lo = [bins_data[b].get("ci_lower", det_rates[i]) for i, b in enumerate(bin_labels)]
        ci_hi = [bins_data[b].get("ci_upper", det_rates[i]) for i, b in enumerate(bin_labels)]
        n_samples = [bins_data[b]["n_samples"] for b in bin_labels]

        color = METHOD_COLORS.get(method_name, "#333333")
        label = METHOD_LABELS.get(method_name, method_name)
        lw = 2.5 if "contrastive" in method_name else 1.5
        marker = "o" if "contrastive" in method_name else "s"

        x = range(len(bin_labels))
        ax.plot(x, det_rates, color=color, linewidth=lw, marker=marker,
                markersize=6, label=label)

        # CI error bars
        yerr_lo = [d - l for d, l in zip(det_rates, ci_lo)]
        yerr_hi = [h - d for d, h in zip(det_rates, ci_hi)]
        ax.errorbar(x, det_rates, yerr=[yerr_lo, yerr_hi],
                    color=color, fmt='none', capsize=3, alpha=0.5)

        for xi, n in zip(x, n_samples):
            ax.annotate(f"n={n}", (xi, det_rates[xi]),
                       textcoords="offset points", xytext=(0, 12),
                       fontsize=7, alpha=0.5, ha="center")

    if bin_labels:
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels, rotation=30, ha="right")
    ax.set_xlabel("K-mer Similarity to Nearest Training Threat")
    ax.set_ylabel("Detection Rate")
    ax.set_title("Detection Rate vs. Sequence Divergence (with 95% CI)")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_ylim([-0.05, 1.05])

    plt.tight_layout()
    plt.savefig(output_dir / "detection_vs_divergence_ci.pdf")
    plt.savefig(output_dir / "detection_vs_divergence_ci.png")
    plt.close()
    logger.info("Saved detection vs divergence plot with CIs")


def plot_mpnn_detection_vs_identity_with_ci(
    method_results: dict[str, dict[str, dict]],
    output_dir: Path | None = None,
):
    """
    MPNN detection rate vs identity with bootstrap CI error bars.

    Args:
        method_results: {method_name: {bin_label: {detection_rate, ci_lower, ci_upper, n_samples}}}
    """
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bin_labels = None
    for method_name, bins_data in method_results.items():
        if bin_labels is None:
            bin_labels = list(bins_data.keys())

        det_rates = [bins_data.get(b, {}).get("detection_rate", 0) for b in bin_labels]
        ci_lo = [bins_data.get(b, {}).get("ci_lower", 0) for b in bin_labels]
        ci_hi = [bins_data.get(b, {}).get("ci_upper", 0) for b in bin_labels]
        n_samples = [bins_data.get(b, {}).get("n_samples", 0) for b in bin_labels]

        color = METHOD_COLORS.get(method_name, "#333333")
        label = METHOD_LABELS.get(method_name, method_name)
        lw = 2.5 if "contrastive" in method_name else 1.5
        marker = "o" if "contrastive" in method_name else "s"

        x = range(len(bin_labels))
        ax.plot(x, det_rates, color=color, linewidth=lw, marker=marker,
                markersize=7, label=label)

        yerr_lo = [max(0, d - l) for d, l in zip(det_rates, ci_lo)]
        yerr_hi = [max(0, h - d) for d, h in zip(det_rates, ci_hi)]
        ax.errorbar(x, det_rates, yerr=[yerr_lo, yerr_hi],
                    color=color, fmt='none', capsize=3, alpha=0.5)

        for xi, (dr, n) in enumerate(zip(det_rates, n_samples)):
            ax.annotate(f"n={n}", (xi, dr),
                       textcoords="offset points", xytext=(0, 14),
                       fontsize=7, alpha=0.5, ha="center")

    if bin_labels:
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels, rotation=25, ha="right")
    ax.set_xlabel("Sequence Identity to Original Threat (%)")
    ax.set_ylabel("Detection Rate")
    ax.set_title("Detection of MPNN Variants vs. Identity (with 95% CI)")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_ylim([-0.05, 1.05])

    ax.axhspan(0, 0.5, alpha=0.06, color="red")

    plt.tight_layout()
    plt.savefig(output_dir / "mpnn_detection_vs_identity_ci.pdf")
    plt.savefig(output_dir / "mpnn_detection_vs_identity_ci.png")
    plt.close()
    logger.info("Saved MPNN detection vs identity plot with CIs")


def plot_ablation_results(
    ablation_df: "pd.DataFrame",
    output_dir: Path | None = None,
):
    """
    2x3 grid of ablation plots: one subplot per ablation type,
    showing AUROC vs parameter value for each evaluation split.
    """
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    ablation_types = ablation_df["ablation_type"].unique()
    n_types = len(ablation_types)
    ncols = min(3, n_types)
    nrows = (n_types + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n_types == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    split_colors = {
        "test_standard": "#0072BD",
        "test_hard_negative": "#77AC30",
        "test_seq_divergent": "#D95319",
        "test_adversarial_mpnn": "#A2142F",
    }

    for i, atype in enumerate(ablation_types):
        ax = axes[i]
        subset = ablation_df[ablation_df["ablation_type"] == atype]

        for split_name, color in split_colors.items():
            split_data = subset[subset["split"] == split_name]
            if len(split_data) == 0:
                continue
            vals = split_data["ablation_value"].astype(str).values
            aurocs = split_data["AUROC"].astype(float).values
            ax.plot(range(len(vals)), aurocs, "o-", color=color,
                    linewidth=1.5, markersize=5,
                    label=split_name.replace("test_", ""))

        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(vals, rotation=30, ha="right")
        ax.set_title(atype.replace("_", " ").title())
        ax.set_ylabel("AUROC")
        ax.set_ylim([0.8, 1.02])
        ax.legend(fontsize=7, loc="lower left")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Ablation Study", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_results.pdf")
    plt.savefig(output_dir / "ablation_results.png")
    plt.close()
    logger.info("Saved ablation results plot")


def plot_loso_results(
    loso_df: "pd.DataFrame",
    output_dir: Path | None = None,
):
    """Bar chart of LOSO CV AUROC per held-out subcategory."""
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    subcats = loso_df["held_out_subcategory"].values
    aurocs = loso_df["AUROC"].astype(float).values
    n_held = loso_df["n_held_out_threats"].astype(int).values

    bars = ax.bar(range(len(subcats)), aurocs, color="#D95319", alpha=0.8,
                  edgecolor="black", linewidth=0.5)

    for i, (bar, n) in enumerate(zip(bars, n_held)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"n={n}", ha="center", fontsize=9)

    ax.set_xticks(range(len(subcats)))
    ax.set_xticklabels(subcats, rotation=30, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_title("Leave-One-Subcategory-Out Cross-Validation")
    ax.set_ylim([0, 1.1])
    ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5, label="0.9 threshold")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "loso_cv.pdf")
    plt.savefig(output_dir / "loso_cv.png")
    plt.close()
    logger.info("Saved LOSO CV plot")


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    sequence: str,
    accession: str = "",
    output_dir: Path | None = None,
):
    """
    Heatmap of attention weights over protein sequence positions.

    Shows which residues the model attends to for threat detection.
    For pore-forming toxins, we expect high attention on the
    pore-forming domain.
    """
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 2))

    # attention_weights shape: (n_heads, seq_len) or (seq_len,)
    if attention_weights.ndim == 2:
        weights = attention_weights.mean(axis=0)
    else:
        weights = attention_weights

    # Truncate for display
    max_display = 200
    if len(weights) > max_display:
        weights = weights[:max_display]
        sequence = sequence[:max_display]

    ax.imshow(weights.reshape(1, -1), aspect="auto", cmap="Reds",
              interpolation="nearest")
    ax.set_yticks([])
    ax.set_xlabel("Sequence Position")
    ax.set_title(f"Attention Weights — {accession}" if accession else "Attention Weights")

    # Mark positions with highest attention
    top_k = min(10, len(weights))
    top_positions = np.argsort(weights)[-top_k:]
    for pos in top_positions:
        ax.axvline(x=pos, color="blue", alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    fname = f"attention_{accession}" if accession else "attention_heatmap"
    plt.savefig(output_dir / f"{fname}.pdf")
    plt.savefig(output_dir / f"{fname}.png")
    plt.close()
