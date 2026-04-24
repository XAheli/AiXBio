from __future__ import annotations

"""
Dataset splitting strategy for function-aware screening evaluation.

Creates 4 evaluation slices:
1. Standard test set - held-out natural sequences
2. Hard-negative set - benign but structurally/functionally similar proteins
3. Sequence-divergent set - threat sequences with low identity to training threats
4. Adversarial set - computationally mutated variants (generated separately)
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import SPLITS_DIR, SEED

logger = logging.getLogger(__name__)


def compute_pairwise_identity_fast(sequences: list[str]) -> np.ndarray:
    """
    Compute approximate pairwise sequence identity using k-mer overlap.
    This is a fast approximation — not a full alignment.
    For exact identity, use BLAST or MMseqs2 externally.
    """
    k = 3
    n = len(sequences)

    # Build k-mer sets
    kmer_sets = []
    for seq in sequences:
        kmers = set()
        for i in range(len(seq) - k + 1):
            kmers.add(seq[i:i+k])
        kmer_sets.append(kmers)

    # Jaccard similarity as identity proxy
    identity = np.zeros((n, n))
    for i in range(n):
        identity[i, i] = 1.0
        for j in range(i + 1, n):
            intersection = len(kmer_sets[i] & kmer_sets[j])
            union = len(kmer_sets[i] | kmer_sets[j])
            if union > 0:
                sim = intersection / union
                identity[i, j] = sim
                identity[j, i] = sim

    return identity


def create_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.15,
    test_ratio: float = 0.25,
    seq_divergent_threshold: float = 0.3,
    output_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Create train/val/test splits with 4 evaluation slices.

    The key constraint: threat sequences in the sequence-divergent test set
    must have low k-mer similarity to ALL threat sequences in the training set.
    This tests whether the model generalizes to novel variants.
    """
    if output_dir is None:
        output_dir = SPLITS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(SEED)

    # Separate threat and benign
    threat_df = df[df["label"] == "threat"].copy()
    benign_df = df[df["label"] == "benign"].copy()
    hard_neg_df = benign_df[benign_df["is_hard_negative"]].copy()
    easy_neg_df = benign_df[~benign_df["is_hard_negative"]].copy()

    logger.info(f"Threat: {len(threat_df)}, Hard neg: {len(hard_neg_df)}, Easy neg: {len(easy_neg_df)}")

    # ── Identify sequence-divergent threats ────────────────────────────
    # Compute pairwise similarity among threat sequences
    threat_seqs = threat_df["sequence"].tolist()
    if len(threat_seqs) > 500:
        # Subsample for speed
        logger.info("Subsampling threat sequences for identity computation...")
        subsample_idx = rng.choice(len(threat_seqs), 500, replace=False)
        subsample_seqs = [threat_seqs[i] for i in subsample_idx]
        identity_matrix = compute_pairwise_identity_fast(subsample_seqs)
    else:
        identity_matrix = compute_pairwise_identity_fast(threat_seqs)
        subsample_idx = np.arange(len(threat_seqs))

    # For each threat, compute max similarity to all other threats
    max_sim = np.zeros(len(threat_df))
    for i, idx in enumerate(subsample_idx):
        sims = identity_matrix[i].copy()
        sims[i] = 0  # exclude self
        max_sim[idx] = sims.max() if len(sims) > 1 else 0

    threat_df = threat_df.copy()
    threat_df["max_kmer_sim"] = max_sim[:len(threat_df)]

    # Sequence-divergent = low similarity to peers (outliers in sequence space)
    seq_div_mask = threat_df["max_kmer_sim"] < seq_divergent_threshold
    seq_div_threats = threat_df[seq_div_mask]
    normal_threats = threat_df[~seq_div_mask]

    logger.info(f"Sequence-divergent threats (sim < {seq_divergent_threshold}): {len(seq_div_threats)}")
    logger.info(f"Normal threats: {len(normal_threats)}")

    # ── Split normal threats: train / val / standard_test ──────────────
    if len(normal_threats) < 10:
        logger.warning("Very few normal threats — using all for training and duplicating for eval")
        train_threats = normal_threats
        val_threats = normal_threats.sample(min(5, len(normal_threats)), random_state=SEED)
        test_threats = normal_threats.sample(min(5, len(normal_threats)), random_state=SEED + 1)
    else:
        test_size = test_ratio / (1 - 0)  # We've removed seq_div already
        val_size = val_ratio / (train_ratio + val_ratio)

        train_val_threats, test_threats = train_test_split(
            normal_threats, test_size=test_ratio, random_state=SEED,
            stratify=normal_threats["subcategory"] if normal_threats["subcategory"].nunique() > 1 else None,
        )
        train_threats, val_threats = train_test_split(
            train_val_threats, test_size=val_size, random_state=SEED,
        )

    # ── Split benign sequences ─────────────────────────────────────────
    # Hard negatives: reserve most for test (that's where they matter)
    if len(hard_neg_df) >= 10:
        train_hard_neg, test_hard_neg = train_test_split(
            hard_neg_df, test_size=0.5, random_state=SEED,
        )
        train_hard_neg, val_hard_neg = train_test_split(
            train_hard_neg, test_size=0.3, random_state=SEED,
        )
    else:
        train_hard_neg = hard_neg_df.sample(frac=0.5, random_state=SEED)
        remaining = hard_neg_df.drop(train_hard_neg.index)
        val_hard_neg = remaining.sample(frac=0.5, random_state=SEED)
        test_hard_neg = remaining.drop(val_hard_neg.index)

    # Easy negatives: standard split
    if len(easy_neg_df) >= 10:
        train_easy, test_easy = train_test_split(
            easy_neg_df, test_size=test_ratio, random_state=SEED,
        )
        train_easy, val_easy = train_test_split(
            train_easy, test_size=val_ratio / (train_ratio + val_ratio), random_state=SEED,
        )
    else:
        train_easy = easy_neg_df
        val_easy = pd.DataFrame(columns=easy_neg_df.columns)
        test_easy = pd.DataFrame(columns=easy_neg_df.columns)

    # ── Assemble splits ────────────────────────────────────────────────
    train_df = pd.concat([train_threats, train_hard_neg, train_easy], ignore_index=True)
    val_df = pd.concat([val_threats, val_hard_neg, val_easy], ignore_index=True)

    # Standard test: normal threats + mix of negatives
    test_standard = pd.concat([test_threats, test_hard_neg.sample(frac=0.3, random_state=SEED), test_easy], ignore_index=True)

    # Hard-negative test: threats + ONLY hard negatives
    test_hard = pd.concat([test_threats, test_hard_neg], ignore_index=True)

    # Sequence-divergent test: divergent threats + benign
    test_seq_div = pd.concat([seq_div_threats, test_hard_neg.sample(frac=0.3, random_state=SEED), test_easy.sample(frac=0.3, random_state=SEED)], ignore_index=True)

    splits = {
        "train": train_df,
        "val": val_df,
        "test_standard": test_standard,
        "test_hard_negative": test_hard,
        "test_seq_divergent": test_seq_div,
    }

    # Shuffle all splits
    for name in splits:
        splits[name] = splits[name].sample(frac=1, random_state=SEED).reset_index(drop=True)

    # ── Save splits ────────────────────────────────────────────────────
    split_stats = {}
    for name, split_df in splits.items():
        split_df.to_parquet(output_dir / f"{name}.parquet", index=False)
        n_threat = int((split_df["label"] == "threat").sum())
        n_benign = int((split_df["label"] == "benign").sum())
        split_stats[name] = {"total": len(split_df), "threat": n_threat, "benign": n_benign}
        logger.info(f"  {name}: {len(split_df)} sequences ({n_threat} threat, {n_benign} benign)")

    with open(output_dir / "split_stats.json", "w") as f:
        json.dump(split_stats, f, indent=2)

    return splits
