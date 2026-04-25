#!/usr/bin/env python3
"""
Step 4: Run full evaluation across all methods and splits.
Can run on CPU (uses pre-computed embeddings).

Usage:
    python scripts/04_evaluate.py --embedding esm2
    python scripts/04_evaluate.py --embedding esm2 protrek
"""
import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src.config import (
    SPLITS_DIR, EMBEDDINGS_DIR, CHECKPOINTS_DIR,
    FIGURES_DIR, TABLES_DIR, RESULTS_DIR,
)
from src.models.embeddings import load_embeddings
from src.models.contrastive import FunctionAwareScreener
from src.screening.baselines import (
    KmerScreener, CosineNNScreener, LinearScreener, KNNEmbeddingScreener,
)
from src.evaluation.metrics import (
    compute_screening_metrics, compute_screening_metrics_with_ci,
    detection_vs_divergence, detection_vs_divergence_with_ci,
    paired_bootstrap_test,
)
from src.evaluation.visualize import (
    plot_roc_curves, plot_pr_curves, plot_detection_vs_divergence,
    plot_detection_vs_divergence_with_ci,
    plot_embedding_space, plot_training_curves, plot_results_heatmap,
    plot_mpnn_detection_vs_identity, plot_mpnn_detection_vs_identity_with_ci,
    plot_identity_distribution,
)
from sklearn.metrics import roc_auc_score
from src.data.splits import compute_pairwise_identity_fast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_aligned(embedding_type: str, split_name: str):
    """Load embeddings and labels, aligned by accession."""
    emb, emb_accessions = load_embeddings(embedding_type, split_name)
    df = pd.read_parquet(SPLITS_DIR / f"{split_name}.parquet")
    label_map = dict(zip(df["accession"], df["label"].map({"threat": 1, "benign": 0})))
    seq_map = dict(zip(df["accession"], df["sequence"]))

    labels = np.array([label_map[acc] for acc in emb_accessions], dtype=np.int64)
    sequences = [seq_map[acc] for acc in emb_accessions]
    return emb, labels, sequences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", nargs="+", default=["esm2"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Load all splits
    split_names = ["train", "val", "test_standard", "test_hard_negative",
                    "test_seq_divergent", "test_adversarial", "test_adversarial_mpnn"]
    splits = {}
    for name in split_names:
        path = SPLITS_DIR / f"{name}.parquet"
        if path.exists():
            splits[name] = pd.read_parquet(path)

    train_df = splits["train"]
    train_seqs = train_df["sequence"].tolist()
    train_labels = train_df["label"].map({"threat": 1, "benign": 0}).values

    # K-mer screener (embedding-independent) — k=5 for proteins is a
    # better BLAST proxy than k=3 (avoids high collision rate with only 8000 3-mers)
    kmer = KmerScreener(k=5)
    kmer.fit(train_seqs, train_labels)

    for emb_type in args.embedding:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating with {emb_type} embeddings")
        logger.info(f"{'='*60}")

        # Load training embeddings (aligned with labels by accession)
        try:
            train_emb, train_labels_aligned, train_seqs_aligned = load_aligned(emb_type, "train")
        except FileNotFoundError:
            logger.error(f"Training embeddings for {emb_type} not found!")
            continue

        # Fit baselines on aligned data
        cosine_nn = CosineNNScreener(n_neighbors=5)
        cosine_nn.fit(train_emb, train_labels_aligned)

        linear = LinearScreener()
        linear.fit(train_emb, train_labels_aligned)

        knn = KNNEmbeddingScreener(n_neighbors=5)
        knn.fit(train_emb, train_labels_aligned)

        # Load contrastive model
        model_path = CHECKPOINTS_DIR / f"screener_{emb_type}.pt"
        screener = None
        if model_path.exists():
            input_dim = train_emb.shape[1]
            screener = FunctionAwareScreener(input_dim)
            screener.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
            screener.eval()
            logger.info(f"Loaded contrastive model from {model_path}")
        else:
            logger.warning(f"No contrastive model found at {model_path}")

        # Evaluate on each test split
        for split_name in ["test_standard", "test_hard_negative", "test_seq_divergent",
                          "test_adversarial", "test_adversarial_mpnn"]:
            if split_name not in splits:
                continue

            try:
                test_emb, test_labels, test_seqs = load_aligned(emb_type, split_name)
            except FileNotFoundError:
                logger.warning(f"No embeddings for {split_name}/{emb_type}")
                continue

            logger.info(f"\n--- {split_name} ({len(test_seqs)} seqs, {test_labels.sum()} threats) ---")

            # Collect predictions for ROC/PR plotting
            predictions = {}
            all_scores = {}

            # K-mer baseline
            kmer_scores = kmer.predict_proba(test_seqs)
            r = compute_screening_metrics_with_ci(test_labels, kmer_scores, "kmer_sim", split_name)
            all_results.append(r)
            predictions["kmer_sim"] = (test_labels, kmer_scores)
            all_scores["kmer_sim"] = kmer_scores
            logger.info(f"  K-mer:        AUROC={r['AUROC']} {r['AUROC_CI']}")

            # Cosine NN
            cosine_scores = cosine_nn.predict_proba(test_emb)
            r = compute_screening_metrics_with_ci(test_labels, cosine_scores, f"cosine_nn_{emb_type}", split_name)
            all_results.append(r)
            predictions[f"cosine_nn_{emb_type}"] = (test_labels, cosine_scores)
            all_scores[f"cosine_nn_{emb_type}"] = cosine_scores
            logger.info(f"  Cosine NN:    AUROC={r['AUROC']} {r['AUROC_CI']}")

            # Linear
            linear_scores = linear.predict_proba(test_emb)
            r = compute_screening_metrics_with_ci(test_labels, linear_scores, f"linear_{emb_type}", split_name)
            all_results.append(r)
            predictions[f"linear_{emb_type}"] = (test_labels, linear_scores)
            all_scores[f"linear_{emb_type}"] = linear_scores
            logger.info(f"  Linear:       AUROC={r['AUROC']} {r['AUROC_CI']}")

            # KNN
            knn_scores = knn.predict_proba(test_emb)
            r = compute_screening_metrics_with_ci(test_labels, knn_scores, f"knn_{emb_type}", split_name)
            all_results.append(r)
            predictions[f"knn_{emb_type}"] = (test_labels, knn_scores)
            all_scores[f"knn_{emb_type}"] = knn_scores
            logger.info(f"  KNN:          AUROC={r['AUROC']} {r['AUROC_CI']}")

            # Contrastive (ours)
            cont_scores = None
            if screener is not None:
                with torch.no_grad():
                    test_tensor = torch.from_numpy(test_emb).float()
                    cont_scores = screener.predict_proba(test_tensor).numpy()
                r = compute_screening_metrics_with_ci(test_labels, cont_scores, f"contrastive_{emb_type}", split_name)
                all_results.append(r)
                predictions[f"contrastive_{emb_type}"] = (test_labels, cont_scores)
                all_scores[f"contrastive_{emb_type}"] = cont_scores
                logger.info(f"  Contrastive:  AUROC={r['AUROC']} {r['AUROC_CI']}")

            # Paired bootstrap test: contrastive vs KNN
            if cont_scores is not None:
                p_val = paired_bootstrap_test(
                    test_labels, cont_scores, knn_scores, roc_auc_score
                )
                logger.info(f"  Paired bootstrap (contrastive vs KNN): p={p_val:.4f}")

            # Plot ROC and PR curves
            plot_roc_curves(predictions, split_name)
            plot_pr_curves(predictions, split_name)

        # ── Detection vs Divergence (THE KEY FIGURE) ───────────────
        if "test_seq_divergent" in splits:
            logger.info("\n--- Computing detection vs divergence ---")
            try:
                test_div_emb, test_div_labels, test_div_seqs = load_aligned(emb_type, "test_seq_divergent")
            except FileNotFoundError:
                logger.warning("Missing embeddings for divergence analysis")
                continue

            # Compute k-mer similarity of each test threat to nearest train threat
            k_div = 5
            train_threat_seqs = [s for s, l in zip(train_seqs_aligned, train_labels_aligned) if l == 1]

            if train_threat_seqs:
                kmer_sims = np.zeros(len(test_div_seqs))
                for i, seq in enumerate(test_div_seqs):
                    if test_div_labels[i] == 1:
                        query_kmers = set(seq[j:j+k_div] for j in range(len(seq) - k_div + 1))
                        max_sim = 0
                        for train_seq in train_threat_seqs:
                            train_kmers = set(train_seq[j:j+k_div] for j in range(len(train_seq) - k_div + 1))
                            intersection = len(query_kmers & train_kmers)
                            union = len(query_kmers | train_kmers)
                            sim = intersection / union if union > 0 else 0
                            max_sim = max(max_sim, sim)
                        kmer_sims[i] = max_sim

                div_results = {}

                # K-mer
                kmer_div_scores = kmer.predict_proba(test_div_seqs)
                div_results["kmer_sim"] = detection_vs_divergence(
                    test_div_labels, kmer_div_scores, kmer_sims)

                # Embedding-based methods
                cosine_div_scores = cosine_nn.predict_proba(test_div_emb)
                div_results[f"cosine_nn_{emb_type}"] = detection_vs_divergence(
                    test_div_labels, cosine_div_scores, kmer_sims)

                linear_div_scores = linear.predict_proba(test_div_emb)
                div_results[f"linear_{emb_type}"] = detection_vs_divergence(
                    test_div_labels, linear_div_scores, kmer_sims)

                if screener is not None:
                    with torch.no_grad():
                        t = torch.from_numpy(test_div_emb).float()
                        cont_div_scores = screener.predict_proba(t).numpy()
                    div_results[f"contrastive_{emb_type}"] = detection_vs_divergence(
                        test_div_labels, cont_div_scores, kmer_sims)

                plot_detection_vs_divergence(div_results)

        # ── UMAP visualizations ────────────────────────────────────
        if screener is not None:
            logger.info("\n--- Generating UMAP visualizations ---")
            try:
                # Raw embedding space
                plot_embedding_space(
                    train_emb, train_labels_aligned,
                    title=f"Raw {emb_type.upper()} Embedding Space",
                    method_name=f"raw_{emb_type}",
                )

                # Projected space (after contrastive learning)
                with torch.no_grad():
                    train_tensor = torch.from_numpy(train_emb).float()
                    projections = screener.get_projection(train_tensor).numpy()

                plot_embedding_space(
                    projections, train_labels_aligned,
                    title=f"Contrastive Projection Space ({emb_type.upper()})",
                    method_name=f"contrastive_{emb_type}",
                )
            except Exception as e:
                logger.warning(f"UMAP visualization failed: {e}")

        # ── MPNN Detection vs Identity (THE PAPER FIGURE) ──────────
        if "test_adversarial_mpnn" in splits:
            logger.info("\n--- MPNN Detection vs Sequence Identity ---")
            try:
                mpnn_emb, mpnn_labels, mpnn_seqs = load_aligned(emb_type, "test_adversarial_mpnn")
                mpnn_df = splits["test_adversarial_mpnn"]

                # Get sequence identities from the DataFrame
                if "seq_identity_to_source" in mpnn_df.columns:
                    # Align identities with embeddings
                    emb_data_raw, emb_acc = load_embeddings(emb_type, "test_adversarial_mpnn")
                    id_map = dict(zip(mpnn_df["accession"], mpnn_df["seq_identity_to_source"].fillna(1.0)))
                    seq_identities = np.array([id_map.get(acc, 1.0) for acc in emb_acc])

                    mpnn_predictions = {}

                    # K-mer
                    kmer_scores = kmer.predict_proba(mpnn_seqs)
                    mpnn_predictions["kmer_sim"] = (mpnn_labels, kmer_scores)

                    # Embedding methods
                    cosine_scores = cosine_nn.predict_proba(mpnn_emb)
                    mpnn_predictions[f"cosine_nn_{emb_type}"] = (mpnn_labels, cosine_scores)

                    linear_scores = linear.predict_proba(mpnn_emb)
                    mpnn_predictions[f"linear_{emb_type}"] = (mpnn_labels, linear_scores)

                    if screener is not None:
                        with torch.no_grad():
                            t = torch.from_numpy(mpnn_emb).float()
                            cont_scores = screener.predict_proba(t).numpy()
                        mpnn_predictions[f"contrastive_{emb_type}"] = (mpnn_labels, cont_scores)

                    plot_mpnn_detection_vs_identity(
                        mpnn_predictions, seq_identities, mpnn_labels
                    )

                    # Also plot identity distribution
                    plot_identity_distribution(mpnn_df)
                else:
                    logger.warning("MPNN split missing seq_identity_to_source column")

            except FileNotFoundError:
                logger.info("No MPNN embeddings found — run scripts/06_generate_mpnn_variants.py first")

    # ── Save all results ───────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(TABLES_DIR / "all_results.csv", index=False)
    logger.info(f"\nAll results saved to {TABLES_DIR / 'all_results.csv'}")

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)
    print(results_df.to_string(index=False))

    # Plot heatmap
    if len(results_df) > 0:
        plot_results_heatmap(results_df, "AUROC")

    # Training curves
    for emb_type in args.embedding:
        history_path = RESULTS_DIR / f"training_history_{emb_type}.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
            plot_training_curves(history)

    logger.info("\nAll figures saved to results/figures/")
    logger.info("All tables saved to results/tables/")


if __name__ == "__main__":
    main()
