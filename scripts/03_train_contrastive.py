#!/usr/bin/env python3
"""
Step 3: Train the contrastive function-aware screener.

Usage:
    # Original model
    python scripts/03_train_contrastive.py --embedding esm2 --device cuda --epochs 50

    # Full FuncScreen (all novelties)
    python scripts/03_train_contrastive.py --embedding esm2 --device cuda --epochs 50 \
        --multi-scale --mixup --adversarial-augment

    # Ablation: no hard negatives
    python scripts/03_train_contrastive.py --embedding esm2 --device cuda --epochs 50 \
        --no-hard-negatives
"""
import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.models.embeddings import load_embeddings
from src.models.contrastive import train_screener
from src.config import (
    SPLITS_DIR, CHECKPOINTS_DIR, RESULTS_DIR, EMBEDDINGS_DIR,
    CONTRASTIVE_LR, TEMPERATURE, HARD_NEGATIVE_RATIO, PROJECTION_DIM,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_aligned(embedding_type: str, split_name: str):
    """Load embeddings and labels, aligned by accession."""
    emb, emb_accessions = load_embeddings(embedding_type, split_name)
    df = pd.read_parquet(SPLITS_DIR / f"{split_name}.parquet")
    label_map = dict(zip(df["accession"], df["label"].map({"threat": 1, "benign": 0})))
    labels = np.array([label_map[acc] for acc in emb_accessions], dtype=np.int64)
    return emb, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", default="esm2", choices=["esm2", "protrek"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=CONTRASTIVE_LR)
    parser.add_argument("--projection-dim", type=int, default=PROJECTION_DIM)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--hard-negative-ratio", type=int, default=HARD_NEGATIVE_RATIO)
    parser.add_argument("--no-hard-negatives", action="store_true")
    parser.add_argument("--multi-scale", action="store_true",
                        help="Use multi-scale classifier (raw + projected features)")
    parser.add_argument("--mixup", action="store_true",
                        help="Enable Mixup augmentation in embedding space")
    parser.add_argument("--adversarial-augment", action="store_true",
                        help="Augment training with MPNN variants (T=0.8, 1.0)")
    parser.add_argument("--save-suffix", default="",
                        help="Suffix for checkpoint filename")
    args = parser.parse_args()

    # Load embeddings and labels, aligned by accession
    logger.info(f"Loading {args.embedding} embeddings...")
    train_emb, train_labels = _load_aligned(args.embedding, "train")
    val_emb, val_labels = _load_aligned(args.embedding, "val")

    # Adversarial augmentation: add high-temperature MPNN variants to training
    if args.adversarial_augment:
        mpnn_path = SPLITS_DIR / "test_adversarial_mpnn.parquet"
        if not mpnn_path.exists():
            logger.error("MPNN split not found. Run scripts/06_generate_mpnn_variants.py first.")
            return

        mpnn_df = pd.read_parquet(mpnn_path)
        # Use only high-temperature variants (T=0.8, 1.0) for training
        high_temp_mask = mpnn_df["mpnn_temperature"].isin([0.8, 1.0])
        threat_mask = mpnn_df["label"] == "threat"
        aug_df = mpnn_df[high_temp_mask & threat_mask]

        if len(aug_df) > 0:
            try:
                aug_emb, aug_acc = load_embeddings(args.embedding, "test_adversarial_mpnn")
                # Align augmentation embeddings with the filtered MPNN accessions
                aug_acc_set = set(aug_df["accession"])
                aug_mask = np.array([acc in aug_acc_set for acc in aug_acc])
                aug_emb_filtered = aug_emb[aug_mask]
                aug_labels_filtered = np.ones(aug_emb_filtered.shape[0], dtype=np.int64)

                train_emb = np.concatenate([train_emb, aug_emb_filtered], axis=0)
                train_labels = np.concatenate([train_labels, aug_labels_filtered], axis=0)
                logger.info(f"Adversarial augmentation: added {aug_emb_filtered.shape[0]} "
                           f"MPNN variants (T=0.8, 1.0) to training")
            except FileNotFoundError:
                logger.error("MPNN embeddings not found")
                return
        else:
            logger.warning("No high-temperature MPNN variants found for augmentation")

    logger.info(f"Train: {train_emb.shape} ({train_labels.sum()} threats)")
    logger.info(f"Val: {val_emb.shape} ({val_labels.sum()} threats)")

    # Build checkpoint name
    suffix = args.save_suffix
    if not suffix:
        parts = [args.embedding]
        if args.multi_scale:
            parts.append("ms")
        if args.mixup:
            parts.append("mixup")
        if args.adversarial_augment:
            parts.append("advtrain")
        suffix = "_".join(parts)

    save_path = CHECKPOINTS_DIR / f"screener_{suffix}.pt"

    model, history = train_screener(
        train_embeddings=train_emb,
        train_labels=train_labels,
        val_embeddings=val_emb,
        val_labels=val_labels,
        projection_dim=args.projection_dim,
        temperature=args.temperature,
        hard_negative_ratio=args.hard_negative_ratio,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_hard_negatives=not args.no_hard_negatives,
        multi_scale=args.multi_scale,
        use_mixup=args.mixup,
        device=args.device,
        save_path=save_path,
    )

    # Save training history
    history_path = RESULTS_DIR / f"training_history_{suffix}.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: [float(v) if not np.isnan(v) else None for v in vals]
                    for k, vals in history.items()}
    with open(history_path, "w") as f:
        json.dump(serializable, f, indent=2)

    logger.info(f"\nModel saved to {save_path}")
    logger.info(f"History saved to {history_path}")

    if history.get("val_auroc"):
        valid_aurocs = [v for v in history["val_auroc"] if not np.isnan(v)]
        if valid_aurocs:
            best_auroc = max(valid_aurocs)
            logger.info(f"Best validation AUROC: {best_auroc:.4f}")

    logger.info("\nDone! Next step: run scripts/04_evaluate.py")


if __name__ == "__main__":
    main()
