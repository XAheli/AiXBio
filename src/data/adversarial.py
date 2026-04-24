from __future__ import annotations

"""
Adversarial variant generation for screening stress testing.

Generates function-preserving but sequence-divergent variants of threat
sequences using biologically plausible mutations. These simulate what an
adversary could produce using AI protein design tools or directed evolution.

For the hackathon prototype, we use computational mutation strategies.
A production system would integrate ProteinMPNN or similar inverse folding
tools for more realistic adversarial variants.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, SPLITS_DIR, SEED

logger = logging.getLogger(__name__)

# Blosum62-derived conservative substitution probabilities
# Higher probability = more conservative (more likely to preserve function)
BLOSUM62_CONSERVATIVE = {
    'A': 'GSTV', 'R': 'KHQ', 'N': 'DSHK', 'D': 'ENQ', 'C': 'S',
    'Q': 'EKNR', 'E': 'DQK', 'G': 'ASN', 'H': 'NQRY', 'I': 'LVMF',
    'L': 'IVMF', 'K': 'RQE', 'M': 'LIV', 'F': 'YLWI', 'P': 'AS',
    'S': 'TANPG', 'T': 'SANV', 'W': 'FY', 'Y': 'FWH', 'V': 'ILMA',
}


def generate_conservative_variant(
    sequence: str,
    mutation_rate: float,
    rng: np.random.RandomState,
) -> str:
    """
    Generate a conservative variant by mutating a fraction of positions
    using BLOSUM62-derived substitutions.

    This preserves function with high probability because each substitution
    is drawn from the set of amino acids that score positively in BLOSUM62.
    """
    seq = list(sequence)
    n_mutations = max(1, int(len(seq) * mutation_rate))
    positions = rng.choice(len(seq), size=min(n_mutations, len(seq)), replace=False)

    for pos in positions:
        original = seq[pos]
        alternatives = BLOSUM62_CONSERVATIVE.get(original, "")
        if alternatives:
            seq[pos] = rng.choice(list(alternatives))

    return "".join(seq)


def generate_block_shuffle_variant(
    sequence: str,
    block_size: int,
    n_swaps: int,
    rng: np.random.RandomState,
) -> str:
    """
    Swap non-overlapping blocks within the sequence.
    This disrupts local sequence patterns while preserving global composition,
    simulating what an adversary might do to evade k-mer-based screening.
    """
    seq = list(sequence)
    n_blocks = len(seq) // block_size

    for _ in range(n_swaps):
        if n_blocks < 2:
            break
        i, j = rng.choice(n_blocks, size=2, replace=False)
        start_i, start_j = i * block_size, j * block_size
        # Swap blocks
        block_i = seq[start_i:start_i + block_size]
        block_j = seq[start_j:start_j + block_size]
        seq[start_i:start_i + block_size] = block_j
        seq[start_j:start_j + block_size] = block_i

    return "".join(seq)


def generate_adversarial_split(
    train_df: pd.DataFrame,
    mutation_rates: list[float] | None = None,
    n_variants_per_rate: int = 2,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Generate adversarial variants from training threat sequences.

    Creates variants at multiple mutation rates to test how screening
    degrades as sequence divergence increases.
    """
    if mutation_rates is None:
        mutation_rates = [0.05, 0.10, 0.20, 0.30, 0.50]
    if output_dir is None:
        output_dir = SPLITS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(SEED)
    threat_df = train_df[train_df["label"] == "threat"].copy()

    if len(threat_df) == 0:
        logger.warning("No threat sequences in training set for adversarial generation")
        return pd.DataFrame()

    adversarial_rows = []

    for _, row in threat_df.iterrows():
        original_seq = row["sequence"]

        for rate in mutation_rates:
            for variant_idx in range(n_variants_per_rate):
                # Conservative mutation variant
                variant_seq = generate_conservative_variant(original_seq, rate, rng)
                adversarial_rows.append({
                    "accession": f"{row['accession']}_adv_mut{rate:.0%}_v{variant_idx}",
                    "name": f"{row['name']}_adversarial",
                    "sequence": variant_seq,
                    "length": len(variant_seq),
                    "organism": row.get("organism", "synthetic"),
                    "label": "threat",
                    "subcategory": "adversarial_conservative",
                    "is_hard_negative": False,
                    "source_accession": row["accession"],
                    "mutation_rate": rate,
                    "mutation_type": "conservative",
                    "pfam_ids": row.get("pfam_ids", []),
                    "go_terms": row.get("go_terms", []),
                    "function_description": row.get("function_description", ""),
                })

        # Also generate block-shuffle variants at high divergence
        for variant_idx in range(2):
            variant_seq = generate_block_shuffle_variant(
                original_seq, block_size=10, n_swaps=5, rng=rng
            )
            adversarial_rows.append({
                "accession": f"{row['accession']}_adv_shuffle_v{variant_idx}",
                "name": f"{row['name']}_adversarial",
                "sequence": variant_seq,
                "length": len(variant_seq),
                "organism": row.get("organism", "synthetic"),
                "label": "threat",
                "subcategory": "adversarial_shuffle",
                "is_hard_negative": False,
                "source_accession": row["accession"],
                "mutation_rate": 0.0,
                "mutation_type": "block_shuffle",
                "pfam_ids": row.get("pfam_ids", []),
                "go_terms": row.get("go_terms", []),
                "function_description": row.get("function_description", ""),
            })

    adv_df = pd.DataFrame(adversarial_rows)

    # Add some benign sequences from training for balanced evaluation
    benign_df = train_df[train_df["label"] == "benign"].sample(
        n=min(len(adv_df), len(train_df[train_df["label"] == "benign"])),
        random_state=SEED,
    ).copy()
    # Add missing columns for consistency
    for col in ["source_accession", "mutation_rate", "mutation_type"]:
        if col not in benign_df.columns:
            benign_df[col] = None

    test_adversarial = pd.concat([adv_df, benign_df], ignore_index=True)
    test_adversarial = test_adversarial.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Save
    test_adversarial.to_parquet(output_dir / "test_adversarial.parquet", index=False)

    n_adv_threat = int((test_adversarial["label"] == "threat").sum())
    n_adv_benign = int((test_adversarial["label"] == "benign").sum())
    logger.info(f"Adversarial split: {len(test_adversarial)} sequences "
                f"({n_adv_threat} threat variants, {n_adv_benign} benign)")
    logger.info(f"  Mutation rates: {mutation_rates}")
    logger.info(f"  Saved to {output_dir / 'test_adversarial.parquet'}")

    return test_adversarial
