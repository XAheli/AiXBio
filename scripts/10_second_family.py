#!/usr/bin/env python3
"""
Step 10: Second threat family mini-experiment (Ribosome-Inactivating Proteins).

Demonstrates that FuncScreen generalizes beyond pore-forming toxins
by training and evaluating on a completely different threat family.

Usage:
    python scripts/10_second_family.py --device cuda
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
from sklearn.model_selection import train_test_split

from src.config import (
    SPLITS_DIR, EMBEDDINGS_DIR, CHECKPOINTS_DIR, TABLES_DIR,
    RIP_THREAT_QUERIES, RIP_BENIGN_QUERIES, SEED,
)
from src.data.curate import fetch_uniprot, parse_uniprot_entry, save_curated_dataset
from src.models.embeddings import ESM2Embedder
from src.models.contrastive import train_screener
from src.screening.baselines import (
    KmerScreener, CosineNNScreener, LinearScreener, KNNEmbeddingScreener,
)
from src.evaluation.metrics import compute_screening_metrics, bootstrap_metric
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Curate RIP dataset ────────────────────────────────────────
    logger.info("=== Curating Ribosome-Inactivating Protein dataset ===")

    all_entries = []
    seen = set()

    for qinfo in RIP_THREAT_QUERIES:
        logger.info(f"  Query: {qinfo['description']}")
        raw = fetch_uniprot(qinfo["query"], max_results=200)
        for entry in raw:
            parsed = parse_uniprot_entry(entry, qinfo["label"], qinfo["subcategory"])
            if parsed and parsed.accession not in seen and 50 <= parsed.length <= 2000:
                all_entries.append(parsed)
                seen.add(parsed.accession)
        logger.info(f"  Running total: {len(all_entries)}")

    n_threats = len(all_entries)
    logger.info(f"RIP threats: {n_threats}")

    for qinfo in RIP_BENIGN_QUERIES:
        logger.info(f"  Query: {qinfo['description']}")
        raw = fetch_uniprot(qinfo["query"], max_results=300)
        for entry in raw:
            parsed = parse_uniprot_entry(entry, qinfo["label"], qinfo["subcategory"])
            if parsed and parsed.accession not in seen and 50 <= parsed.length <= 2000:
                all_entries.append(parsed)
                seen.add(parsed.accession)
        logger.info(f"  Running total: {len(all_entries)}")

    if n_threats < 10:
        logger.error(f"Too few RIP threats ({n_threats}). Cannot run experiment.")
        return

    from dataclasses import asdict
    df = pd.DataFrame([asdict(e) for e in all_entries])
    n_benign = len(df) - n_threats
    logger.info(f"RIP dataset: {n_threats} threats, {n_benign} benign, {len(df)} total")

    # ── Split ─────────────────────────────────────────────────────
    labels = df["label"].map({"threat": 1, "benign": 0}).values
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.3, random_state=SEED, stratify=labels
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # ── Extract embeddings ────────────────────────────────────────
    logger.info("\nExtracting ESM-2 embeddings for RIP dataset...")
    embedder = ESM2Embedder(device=args.device)

    train_seqs = train_df["sequence"].tolist()
    test_seqs = test_df["sequence"].tolist()

    train_emb = embedder.embed_batch(train_seqs, batch_size=args.batch_size)
    test_emb = embedder.embed_batch(test_seqs, batch_size=args.batch_size)

    train_labels = train_df["label"].map({"threat": 1, "benign": 0}).values
    test_labels = test_df["label"].map({"threat": 1, "benign": 0}).values

    logger.info(f"Train: {train_emb.shape} ({train_labels.sum()} threats)")
    logger.info(f"Test: {test_emb.shape} ({test_labels.sum()} threats)")

    # ── Train FuncScreen on RIPs ──────────────────────────────────
    logger.info("\nTraining contrastive model on RIPs...")
    model, history = train_screener(
        train_emb, train_labels,
        epochs=args.epochs, device=args.device,
        save_path=CHECKPOINTS_DIR / "screener_rip.pt",
    )

    # ── Evaluate ──────────────────────────────────────────────────
    logger.info("\nEvaluating on RIP test set...")
    results = []

    # Baselines
    kmer = KmerScreener(k=5)
    kmer.fit(train_seqs, train_labels)
    kmer_scores = kmer.predict_proba(test_seqs)

    cosine_nn = CosineNNScreener(n_neighbors=5)
    cosine_nn.fit(train_emb, train_labels)
    cosine_scores = cosine_nn.predict_proba(test_emb)

    linear = LinearScreener()
    linear.fit(train_emb, train_labels)
    linear_scores = linear.predict_proba(test_emb)

    knn = KNNEmbeddingScreener(n_neighbors=5)
    knn.fit(train_emb, train_labels)
    knn_scores = knn.predict_proba(test_emb)

    model.eval()
    with torch.no_grad():
        cont_scores = model.predict_proba(
            torch.from_numpy(test_emb).float().to(args.device)
        ).cpu().numpy()

    methods = {
        "kmer_sim": kmer_scores,
        "cosine_nn": cosine_scores,
        "linear": linear_scores,
        "knn": knn_scores,
        "contrastive": cont_scores,
    }

    for method_name, scores in methods.items():
        m = compute_screening_metrics(test_labels, scores, method_name, "rip_test")
        auroc, auroc_lo, auroc_hi = bootstrap_metric(test_labels, scores, roc_auc_score)
        row = m.to_dict()
        row["AUROC_CI"] = f"[{auroc_lo:.4f}, {auroc_hi:.4f}]"
        row["family"] = "RIP"
        results.append(row)
        logger.info(f"  {method_name}: AUROC={m.auroc:.4f} [{auroc_lo:.4f}, {auroc_hi:.4f}]")

    # Save
    results_df = pd.DataFrame(results)
    results_df.to_csv(TABLES_DIR / "second_family_results.csv", index=False)
    logger.info(f"\nRIP results saved to {TABLES_DIR / 'second_family_results.csv'}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
