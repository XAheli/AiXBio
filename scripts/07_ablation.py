#!/usr/bin/env python3
"""
Step 7: Ablation study across key hyperparameters.

Tests: projection dim, hard negative ratio, temperature, multi-scale, mixup.
Fine-tuning ablation is separate (requires end-to-end training).

Usage:
    python scripts/07_ablation.py --device cuda --embedding esm2
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
    SPLITS_DIR, EMBEDDINGS_DIR, CHECKPOINTS_DIR, TABLES_DIR,
    ABLATION_PROJECTION_DIMS, ABLATION_HARD_NEGATIVE_RATIOS,
    ABLATION_TEMPERATURES,
)
from src.models.embeddings import load_embeddings
from src.models.contrastive import FunctionAwareScreener, train_screener
from src.evaluation.metrics import compute_screening_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _load_aligned(embedding_type, split_name):
    emb, acc = load_embeddings(embedding_type, split_name)
    df = pd.read_parquet(SPLITS_DIR / f"{split_name}.parquet")
    label_map = dict(zip(df["accession"], df["label"].map({"threat": 1, "benign": 0})))
    labels = np.array([label_map[a] for a in acc], dtype=np.int64)
    return emb, labels


def evaluate_model(model, emb, labels, method_name, split_name, device="cpu"):
    model.eval()
    with torch.no_grad():
        scores = model.predict_proba(torch.from_numpy(emb).float().to(device)).cpu().numpy()
    return compute_screening_metrics(labels, scores, method_name, split_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", default="esm2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    train_emb, train_labels = _load_aligned(args.embedding, "train")
    val_emb, val_labels = _load_aligned(args.embedding, "val")

    eval_splits = {}
    for name in ["test_standard", "test_hard_negative", "test_seq_divergent", "test_adversarial_mpnn"]:
        try:
            eval_splits[name] = _load_aligned(args.embedding, name)
        except FileNotFoundError:
            pass

    all_results = []

    # ── Ablation 1: Projection dimension ──────────────────────────
    logger.info("\n=== Ablation: Projection Dimension ===")
    for dim in ABLATION_PROJECTION_DIMS:
        logger.info(f"  dim={dim}")
        model, _ = train_screener(
            train_emb, train_labels, val_emb, val_labels,
            projection_dim=dim, epochs=args.epochs, device=args.device,
            save_path=CHECKPOINTS_DIR / f"ablation_dim{dim}.pt",
        )
        for split_name, (emb, labels) in eval_splits.items():
            m = evaluate_model(model, emb, labels, f"dim_{dim}", split_name, args.device)
            row = m.to_dict()
            row["ablation_type"] = "projection_dim"
            row["ablation_value"] = dim
            all_results.append(row)
            logger.info(f"    {split_name}: AUROC={m.auroc:.4f}")

    # ── Ablation 2: Hard negative ratio ───────────────────────────
    logger.info("\n=== Ablation: Hard Negative Ratio ===")
    for k in ABLATION_HARD_NEGATIVE_RATIOS:
        logger.info(f"  k={k}")
        model, _ = train_screener(
            train_emb, train_labels, val_emb, val_labels,
            hard_negative_ratio=k, epochs=args.epochs, device=args.device,
            save_path=CHECKPOINTS_DIR / f"ablation_hnr{k}.pt",
        )
        for split_name, (emb, labels) in eval_splits.items():
            m = evaluate_model(model, emb, labels, f"hnr_{k}", split_name, args.device)
            row = m.to_dict()
            row["ablation_type"] = "hard_negative_ratio"
            row["ablation_value"] = k
            all_results.append(row)
            logger.info(f"    {split_name}: AUROC={m.auroc:.4f}")

    # ── Ablation 3: Temperature ───────────────────────────────────
    logger.info("\n=== Ablation: Temperature ===")
    for tau in ABLATION_TEMPERATURES:
        logger.info(f"  tau={tau}")
        model, _ = train_screener(
            train_emb, train_labels, val_emb, val_labels,
            temperature=tau, epochs=args.epochs, device=args.device,
            save_path=CHECKPOINTS_DIR / f"ablation_tau{tau}.pt",
        )
        for split_name, (emb, labels) in eval_splits.items():
            m = evaluate_model(model, emb, labels, f"tau_{tau}", split_name, args.device)
            row = m.to_dict()
            row["ablation_type"] = "temperature"
            row["ablation_value"] = tau
            all_results.append(row)
            logger.info(f"    {split_name}: AUROC={m.auroc:.4f}")

    # ── Ablation 4: Multi-scale vs single-scale ───────────────────
    logger.info("\n=== Ablation: Multi-scale ===")
    for ms in [False, True]:
        label = "multi_scale" if ms else "single_scale"
        logger.info(f"  {label}")
        model, _ = train_screener(
            train_emb, train_labels, val_emb, val_labels,
            multi_scale=ms, epochs=args.epochs, device=args.device,
            save_path=CHECKPOINTS_DIR / f"ablation_{label}.pt",
        )
        for split_name, (emb, labels) in eval_splits.items():
            m = evaluate_model(model, emb, labels, label, split_name, args.device)
            row = m.to_dict()
            row["ablation_type"] = "multi_scale"
            row["ablation_value"] = str(ms)
            all_results.append(row)
            logger.info(f"    {split_name}: AUROC={m.auroc:.4f}")

    # ── Ablation 5: Mixup vs no mixup ─────────────────────────────
    logger.info("\n=== Ablation: Mixup ===")
    for mx in [False, True]:
        label = "with_mixup" if mx else "no_mixup"
        logger.info(f"  {label}")
        model, _ = train_screener(
            train_emb, train_labels, val_emb, val_labels,
            use_mixup=mx, epochs=args.epochs, device=args.device,
            save_path=CHECKPOINTS_DIR / f"ablation_{label}.pt",
        )
        for split_name, (emb, labels) in eval_splits.items():
            m = evaluate_model(model, emb, labels, label, split_name, args.device)
            row = m.to_dict()
            row["ablation_type"] = "mixup"
            row["ablation_value"] = str(mx)
            all_results.append(row)
            logger.info(f"    {split_name}: AUROC={m.auroc:.4f}")

    # Save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(TABLES_DIR / "ablation_results.csv", index=False)
    logger.info(f"\nAblation results saved to {TABLES_DIR / 'ablation_results.csv'}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
