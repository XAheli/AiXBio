#!/usr/bin/env python3
"""
Step 3: Train the contrastive function-aware screener.
*** Run this on GPU (or CPU if embeddings are pre-computed) ***

Usage:
    python scripts/03_train_contrastive.py --embedding esm2 --device cuda --epochs 50
    python scripts/03_train_contrastive.py --embedding esm2 --device cpu --epochs 100
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
from src.config import SPLITS_DIR, CHECKPOINTS_DIR, RESULTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_aligned(embedding_type: str, split_name: str):
    """Load embeddings and labels, aligned by accession to prevent mismatch."""
    emb, emb_accessions = load_embeddings(embedding_type, split_name)
    df = pd.read_parquet(SPLITS_DIR / f"{split_name}.parquet")

    # Build accession -> label map from the DataFrame
    label_map = dict(zip(df["accession"], df["label"].map({"threat": 1, "benign": 0})))

    # Align: for each embedding row (in order of emb_accessions), look up the label
    labels = np.array([label_map[acc] for acc in emb_accessions], dtype=np.int64)

    n_missing = sum(1 for acc in emb_accessions if acc not in label_map)
    if n_missing > 0:
        raise RuntimeError(
            f"{n_missing} accessions in embeddings not found in {split_name} split. "
            f"Re-run steps 01 and 02 to regenerate data."
        )

    return emb, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", default="esm2", choices=["esm2", "protrek"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-hard-negatives", action="store_true")
    args = parser.parse_args()

    # Load embeddings and labels, aligned by accession
    logger.info(f"Loading {args.embedding} embeddings...")

    train_emb, train_labels = _load_aligned(args.embedding, "train")
    val_emb, val_labels = _load_aligned(args.embedding, "val")

    logger.info(f"Train: {train_emb.shape} ({train_labels.sum()} threats)")
    logger.info(f"Val: {val_emb.shape} ({val_labels.sum()} threats)")

    # Train
    save_path = CHECKPOINTS_DIR / f"screener_{args.embedding}.pt"

    model, history = train_screener(
        train_embeddings=train_emb,
        train_labels=train_labels,
        val_embeddings=val_emb,
        val_labels=val_labels,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_hard_negatives=not args.no_hard_negatives,
        device=args.device,
        save_path=save_path,
    )

    # Save training history
    history_path = RESULTS_DIR / f"training_history_{args.embedding}.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: [float(v) for v in vals] for k, vals in history.items()}
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
