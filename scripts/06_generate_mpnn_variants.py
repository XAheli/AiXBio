#!/usr/bin/env python3
"""
Step 6: Generate ProteinMPNN adversarial variants.
*** Run this on GPU ***

This is the strongest adversarial evaluation: ProteinMPNN generates
sequences that fold into the same 3D structure as known threat proteins
but have divergent amino acid sequences.

Pipeline:
  1. Fetch structures for threat proteins (AlphaFold DB or ESMFold)
  2. Run ProteinMPNN inverse folding at multiple temperatures
  3. Save variants as test_adversarial_mpnn split
  4. Extract ESM-2 embeddings for the new split

Prerequisites:
  - ProteinMPNN installed:
      git clone https://github.com/dauparas/ProteinMPNN.git
      export PYTHONPATH=$PYTHONPATH:$(pwd)/ProteinMPNN
  - OR: pip install git+https://github.com/dauparas/ProteinMPNN.git

Usage:
  python scripts/06_generate_mpnn_variants.py --device cuda --structure-source alphafold
  python scripts/06_generate_mpnn_variants.py --device cuda --structure-source esmfold
  python scripts/06_generate_mpnn_variants.py --device cuda --temps 0.1 0.3 0.5 0.8 1.0
"""
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import SPLITS_DIR, EMBEDDINGS_DIR
from src.data.structures import fetch_structures_batch
from src.data.proteinmpnn import generate_mpnn_adversarial_split
from src.models.embeddings import extract_and_save_embeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate ProteinMPNN adversarial variants for screening stress test"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--structure-source", choices=["alphafold", "esmfold"], default="alphafold",
        help="Where to get protein structures. 'alphafold' downloads from AlphaFold DB "
             "(no GPU needed, most Swiss-Prot proteins available). 'esmfold' predicts "
             "structures on-the-fly (GPU required, works for any sequence)."
    )
    parser.add_argument(
        "--temps", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.8, 1.0],
        help="ProteinMPNN sampling temperatures. Higher = more diverse sequences."
    )
    parser.add_argument(
        "--num-seqs", type=int, default=5,
        help="Number of designed sequences per temperature per protein"
    )
    parser.add_argument(
        "--max-proteins", type=int, default=None,
        help="Limit number of threat proteins to process (for speed)"
    )
    parser.add_argument(
        "--embedding", default="esm2", choices=["esm2", "protrek"],
        help="Embedding model for the new variants"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for embedding extraction"
    )
    args = parser.parse_args()

    # ── Load threat sequences from training split ─────────────────
    train_path = SPLITS_DIR / "train.parquet"
    if not train_path.exists():
        logger.error("Training split not found. Run scripts/01_curate_data.py first.")
        return

    train_df = pd.read_parquet(train_path)
    threat_df = train_df[train_df["label"] == "threat"].copy()

    if args.max_proteins:
        threat_df = threat_df.head(args.max_proteins)

    logger.info(f"Processing {len(threat_df)} threat proteins")

    # ── Step 1: Fetch structures ──────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"Fetching structures from {args.structure_source}")
    logger.info(f"{'='*60}")

    accessions = threat_df["accession"].tolist()
    sequences = threat_df["sequence"].tolist()

    structure_paths = fetch_structures_batch(
        accessions=accessions,
        sequences=sequences if args.structure_source == "esmfold" else None,
        source=args.structure_source,
        device=args.device,
    )

    if not structure_paths:
        logger.error("No structures obtained. Cannot generate MPNN variants.")
        return

    # ── Step 2: Generate ProteinMPNN variants ─────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"Generating ProteinMPNN variants")
    logger.info(f"  Temperatures: {args.temps}")
    logger.info(f"  Sequences per temp: {args.num_seqs}")
    logger.info(f"  Expected variants: ~{len(structure_paths) * len(args.temps) * args.num_seqs}")
    logger.info(f"{'='*60}")

    mpnn_df = generate_mpnn_adversarial_split(
        threat_df=threat_df,
        structure_paths=structure_paths,
        num_sequences_per_temp=args.num_seqs,
        sampling_temps=args.temps,
        device=args.device,
    )

    if len(mpnn_df) == 0:
        logger.error("No variants generated. Exiting.")
        return

    # ── Step 3: Extract embeddings for MPNN variants ──────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"Extracting {args.embedding} embeddings for MPNN variants")
    logger.info(f"{'='*60}")

    extract_and_save_embeddings(
        df=mpnn_df,
        model_type=args.embedding,
        device=args.device,
        batch_size=args.batch_size,
        split_name="test_adversarial_mpnn",
    )

    logger.info(f"\nDone! MPNN adversarial split saved.")
    logger.info(f"Re-run evaluation: python scripts/04_evaluate.py --embedding {args.embedding}")


if __name__ == "__main__":
    main()
