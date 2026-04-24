from __future__ import annotations

"""
Experiment runner: evaluates all methods across all splits and produces
publication-quality figures and tables.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import (
    SPLITS_DIR, EMBEDDINGS_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR,
    ESM2_EMBEDDING_DIM, SEED,
)
from src.models.embeddings import load_embeddings
from src.models.contrastive import FunctionAwareScreener
from src.screening.baselines import (
    KmerScreener, CosineNNScreener, LinearScreener, KNNEmbeddingScreener,
)
from src.evaluation.metrics import compute_screening_metrics, detection_vs_divergence

logger = logging.getLogger(__name__)


def load_split(split_name: str) -> pd.DataFrame:
    """Load a data split."""
    return pd.read_parquet(SPLITS_DIR / f"{split_name}.parquet")


def run_all_experiments(
    model_path: Path | None = None,
    device: str = "cpu",
    embedding_types: list[str] | None = None,
) -> pd.DataFrame:
    """
    Run all screening methods on all evaluation splits.
    Returns a DataFrame of results.
    """
    if embedding_types is None:
        embedding_types = ["esm2"]

    # Load splits
    split_names = ["train", "val", "test_standard", "test_hard_negative", "test_seq_divergent"]
    splits = {}
    for name in split_names:
        try:
            splits[name] = load_split(name)
        except FileNotFoundError:
            logger.warning(f"Split {name} not found, skipping")

    if "train" not in splits:
        raise RuntimeError("Training split not found")

    all_results = []

    for emb_type in embedding_types:
        logger.info(f"\n=== Evaluating with {emb_type} embeddings ===")

        # Load embeddings for each split
        emb_data = {}
        for name in splits:
            try:
                emb, acc = load_embeddings(emb_type, name)
                emb_data[name] = emb
            except FileNotFoundError:
                logger.warning(f"Embeddings for {name}/{emb_type} not found")

        if "train" not in emb_data:
            logger.warning(f"No training embeddings for {emb_type}")
            continue

        train_emb = emb_data["train"]
        train_labels = splits["train"]["label"].map({"threat": 1, "benign": 0}).values
        train_seqs = splits["train"]["sequence"].tolist()

        # ── Baseline 1: K-mer screening ────────────────────────────
        kmer = KmerScreener(k=3)
        kmer.fit(train_seqs, train_labels)

        # ── Baseline 2: Cosine NN ──────────────────────────────────
        cosine_nn = CosineNNScreener(n_neighbors=5)
        cosine_nn.fit(train_emb, train_labels)

        # ── Baseline 3: Linear classifier ─────────────────────────
        linear = LinearScreener()
        linear.fit(train_emb, train_labels)

        # ── Baseline 4: KNN embedding ─────────────────────────────
        knn = KNNEmbeddingScreener(n_neighbors=5)
        knn.fit(train_emb, train_labels)

        # ── Our model: Function-aware screener ─────────────────────
        screener = None
        if model_path and model_path.exists():
            input_dim = train_emb.shape[1]
            screener = FunctionAwareScreener(input_dim)
            screener.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
            screener.eval()
            screener.to(device)

        # ── Evaluate on each test split ────────────────────────────
        eval_splits = [s for s in splits if s.startswith("test_")]

        for split_name in eval_splits:
            if split_name not in emb_data:
                continue

            test_emb = emb_data[split_name]
            test_labels = splits[split_name]["label"].map({"threat": 1, "benign": 0}).values
            test_seqs = splits[split_name]["sequence"].tolist()

            methods = {
                f"kmer_sim": ("kmer", kmer.predict_proba(test_seqs)),
                f"cosine_nn_{emb_type}": ("embedding", cosine_nn.predict_proba(test_emb)),
                f"linear_{emb_type}": ("embedding", linear.predict_proba(test_emb)),
                f"knn_{emb_type}": ("embedding", knn.predict_proba(test_emb)),
            }

            if screener is not None:
                with torch.no_grad():
                    test_tensor = torch.from_numpy(test_emb).float().to(device)
                    scores = screener.predict_proba(test_tensor).cpu().numpy()
                methods[f"contrastive_{emb_type}"] = ("learned", scores)

            for method_name, (method_type, scores) in methods.items():
                metrics = compute_screening_metrics(
                    test_labels, scores,
                    method_name=method_name,
                    split_name=split_name,
                )
                result = metrics.to_dict()
                result["embedding"] = emb_type
                result["method_type"] = method_type
                all_results.append(result)

                logger.info(
                    f"  {method_name} on {split_name}: "
                    f"AUROC={metrics.auroc:.4f} AUPRC={metrics.auprc:.4f} "
                    f"R@P95={metrics.recall_at_95_precision:.4f}"
                )

    results_df = pd.DataFrame(all_results)
    return results_df


def save_results(results_df: pd.DataFrame, output_dir: Path | None = None):
    """Save results as CSV and LaTeX table."""
    if output_dir is None:
        output_dir = TABLES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / "all_results.csv", index=False)

    # Pivot for paper table: methods as rows, splits as column groups
    for split in results_df["split"].unique():
        split_df = results_df[results_df["split"] == split]
        pivot = split_df[["method", "AUROC", "AUPRC", "R@P95", "FPR@R95", "F1*"]].copy()
        pivot.to_csv(output_dir / f"table_{split}.csv", index=False)

        # LaTeX
        latex = pivot.to_latex(index=False, escape=False)
        with open(output_dir / f"table_{split}.tex", "w") as f:
            f.write(latex)

    logger.info(f"Results saved to {output_dir}")
