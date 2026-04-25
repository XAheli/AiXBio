#!/usr/bin/env python3
"""
Step 8: Leave-One-Subcategory-Out (LOSO) cross-validation.

Tests whether the model memorizes specific subcategories or learns
general functional representations. Holds out all sequences from one
threat subcategory and tests if the model can detect them without
ever seeing them in training.

Usage:
    python scripts/08_loso_cv.py --embedding esm2 --device cuda
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

from src.config import SPLITS_DIR, EMBEDDINGS_DIR, CHECKPOINTS_DIR, TABLES_DIR, SEED
from src.models.embeddings import load_embeddings
from src.models.contrastive import train_screener
from src.evaluation.metrics import bootstrap_metric
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", default="esm2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Load full dataset with subcategory information
    full_df = pd.read_parquet(SPLITS_DIR / "train.parquet")
    val_df = pd.read_parquet(SPLITS_DIR / "val.parquet")
    combined_df = pd.concat([full_df, val_df], ignore_index=True)

    # Load all embeddings
    train_emb, train_acc = load_embeddings(args.embedding, "train")
    val_emb, val_acc = load_embeddings(args.embedding, "val")
    all_emb = np.concatenate([train_emb, val_emb], axis=0)
    all_acc = np.concatenate([train_acc, val_acc])

    # Build accession → index, label, subcategory maps
    acc_to_idx = {acc: i for i, acc in enumerate(all_acc)}
    label_map = dict(zip(combined_df["accession"], combined_df["label"]))
    subcat_map = dict(zip(combined_df["accession"], combined_df["subcategory"]))

    # Get threat subcategories
    threat_subcats = combined_df[combined_df["label"] == "threat"]["subcategory"].unique()
    logger.info(f"Threat subcategories: {list(threat_subcats)}")

    results = []

    for held_out in threat_subcats:
        logger.info(f"\n{'='*60}")
        logger.info(f"Holding out: {held_out}")
        logger.info(f"{'='*60}")

        # Split by subcategory
        held_out_accs = set(
            combined_df[
                (combined_df["subcategory"] == held_out) &
                (combined_df["label"] == "threat")
            ]["accession"]
        )

        # Training: everything except held-out threat subcategory
        train_mask = np.array([acc not in held_out_accs for acc in all_acc])
        # Test: held-out threats + a sample of benign
        test_threat_mask = np.array([acc in held_out_accs for acc in all_acc])

        benign_mask = np.array([label_map.get(acc) == "benign" for acc in all_acc])
        rng = np.random.RandomState(SEED)
        benign_indices = np.where(benign_mask)[0]
        test_benign_indices = rng.choice(
            benign_indices, size=min(100, len(benign_indices)), replace=False
        )
        test_benign_mask = np.zeros(len(all_acc), dtype=bool)
        test_benign_mask[test_benign_indices] = True

        test_mask = test_threat_mask | test_benign_mask

        # Build arrays
        fold_train_emb = all_emb[train_mask]
        fold_train_labels = np.array([
            1 if label_map.get(acc) == "threat" else 0
            for acc in all_acc[train_mask]
        ])

        fold_test_emb = all_emb[test_mask]
        fold_test_labels = np.array([
            1 if label_map.get(acc) == "threat" else 0
            for acc in all_acc[test_mask]
        ])

        n_held_threats = int(fold_test_labels.sum())
        n_held_benign = int(len(fold_test_labels) - n_held_threats)
        logger.info(f"  Train: {fold_train_emb.shape[0]} ({fold_train_labels.sum()} threats)")
        logger.info(f"  Test: {n_held_threats} held-out threats + {n_held_benign} benign")

        if n_held_threats < 3:
            logger.warning(f"  Too few held-out threats ({n_held_threats}), skipping")
            continue

        # Split train into train/val (80/20)
        n_train = int(len(fold_train_emb) * 0.8)
        perm = rng.permutation(len(fold_train_emb))
        tr_idx, va_idx = perm[:n_train], perm[n_train:]

        model, _ = train_screener(
            fold_train_emb[tr_idx], fold_train_labels[tr_idx],
            fold_train_emb[va_idx], fold_train_labels[va_idx],
            epochs=args.epochs, device=args.device,
            save_path=CHECKPOINTS_DIR / f"loso_{held_out}.pt",
        )

        # Evaluate
        model.eval()
        with torch.no_grad():
            scores = model.predict_proba(
                torch.from_numpy(fold_test_emb).float().to(args.device)
            ).cpu().numpy()

        if len(np.unique(fold_test_labels)) < 2:
            logger.warning("  Single class in test — skipping AUROC")
            continue

        auroc, auroc_lo, auroc_hi = bootstrap_metric(
            fold_test_labels, scores, roc_auc_score
        )

        logger.info(f"  AUROC: {auroc:.4f} [{auroc_lo:.4f}, {auroc_hi:.4f}]")

        results.append({
            "held_out_subcategory": held_out,
            "n_held_out_threats": n_held_threats,
            "n_test_benign": n_held_benign,
            "AUROC": f"{auroc:.4f}",
            "AUROC_CI": f"[{auroc_lo:.4f}, {auroc_hi:.4f}]",
            "detection_rate_at_0.5": f"{(scores[fold_test_labels == 1] >= 0.5).mean():.4f}",
        })

    # Save
    results_df = pd.DataFrame(results)
    results_df.to_csv(TABLES_DIR / "loso_results.csv", index=False)
    logger.info(f"\nLOSO results saved to {TABLES_DIR / 'loso_results.csv'}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
