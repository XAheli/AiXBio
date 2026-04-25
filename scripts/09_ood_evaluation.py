#!/usr/bin/env python3
"""
Step 9: Out-of-Distribution false positive rate evaluation.

Tests whether the screener incorrectly flags completely unrelated
protein families (kinases, GPCRs, transcription factors) as threats.

Usage:
    python scripts/09_ood_evaluation.py --device cuda --embedding esm2
"""
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src.config import (
    SPLITS_DIR, EMBEDDINGS_DIR, CHECKPOINTS_DIR, TABLES_DIR,
    OOD_PROTEIN_FAMILIES, OOD_MAX_PER_FAMILY, SEED,
)
from src.data.curate import fetch_uniprot, parse_uniprot_entry
from src.models.embeddings import ESM2Embedder, load_embeddings
from src.models.contrastive import FunctionAwareScreener
from src.screening.baselines import (
    KmerScreener, CosineNNScreener, LinearScreener, KNNEmbeddingScreener,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding", default="esm2")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Load training data for baselines
    train_df = pd.read_parquet(SPLITS_DIR / "train.parquet")
    train_emb, train_acc = load_embeddings(args.embedding, "train")
    label_map = dict(zip(train_df["accession"], train_df["label"].map({"threat": 1, "benign": 0})))
    train_labels = np.array([label_map[a] for a in train_acc], dtype=np.int64)
    train_seqs = dict(zip(train_df["accession"], train_df["sequence"]))
    train_seq_list = [train_seqs[a] for a in train_acc]

    # Fit baselines
    kmer = KmerScreener(k=5)
    kmer.fit(train_seq_list, train_labels)

    cosine_nn = CosineNNScreener(n_neighbors=5)
    cosine_nn.fit(train_emb, train_labels)

    linear = LinearScreener()
    linear.fit(train_emb, train_labels)

    knn = KNNEmbeddingScreener(n_neighbors=5)
    knn.fit(train_emb, train_labels)

    # Load contrastive model
    model_path = CHECKPOINTS_DIR / f"screener_{args.embedding}.pt"
    screener = None
    if model_path.exists():
        screener = FunctionAwareScreener(train_emb.shape[1])
        screener.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
        screener.eval()

    # Load ESM-2 embedder for OOD proteins
    embedder = ESM2Embedder(device=args.device)

    results = []

    for family_name, query in OOD_PROTEIN_FAMILIES.items():
        logger.info(f"\n--- OOD Family: {family_name} ---")

        # Fetch proteins
        raw = fetch_uniprot(query, max_results=OOD_MAX_PER_FAMILY)
        sequences = []
        for entry in raw:
            parsed = parse_uniprot_entry(entry, "benign", family_name)
            if parsed and 50 <= parsed.length <= 2000:
                sequences.append(parsed.sequence)

        if len(sequences) == 0:
            logger.warning(f"  No sequences fetched for {family_name}")
            continue

        logger.info(f"  Fetched {len(sequences)} proteins")

        # Extract embeddings
        ood_emb = embedder.embed_batch(sequences, batch_size=args.batch_size)

        # Score with each method
        kmer_scores = kmer.predict_proba(sequences)
        cosine_scores = cosine_nn.predict_proba(ood_emb)
        linear_scores = linear.predict_proba(ood_emb)
        knn_scores = knn.predict_proba(ood_emb)

        methods = {
            "kmer_sim": kmer_scores,
            "cosine_nn": cosine_scores,
            "linear": linear_scores,
            "knn": knn_scores,
        }

        if screener is not None:
            with torch.no_grad():
                cont_scores = screener.predict_proba(
                    torch.from_numpy(ood_emb).float()
                ).numpy()
            methods["contrastive"] = cont_scores

        for method_name, scores in methods.items():
            fpr = float((scores >= 0.5).mean())
            mean_score = float(scores.mean())
            max_score = float(scores.max())

            results.append({
                "ood_family": family_name,
                "method": method_name,
                "n_proteins": len(sequences),
                "FPR_at_0.5": f"{fpr:.4f}",
                "mean_score": f"{mean_score:.4f}",
                "max_score": f"{max_score:.4f}",
            })
            logger.info(f"  {method_name}: FPR={fpr:.4f}, mean={mean_score:.4f}, max={max_score:.4f}")

    # Save
    results_df = pd.DataFrame(results)
    results_df.to_csv(TABLES_DIR / "ood_fpr.csv", index=False)
    logger.info(f"\nOOD results saved to {TABLES_DIR / 'ood_fpr.csv'}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
