#!/usr/bin/env python3
"""
Step 1: Curate dataset from UniProt.
Run this on CPU — no GPU needed.

Usage:
    python scripts/01_curate_data.py
"""
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.curate import curate_dataset, save_curated_dataset
from src.data.splits import create_splits
from src.data.adversarial import generate_adversarial_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Step 1: Curating dataset from UniProt")
    logger.info("=" * 60)

    # Curate
    df = curate_dataset(
        max_per_query=300,
        max_random_benign=200,
        min_length=50,
        max_length=2000,
    )

    # Save
    stats = save_curated_dataset(df)

    logger.info("\n" + "=" * 60)
    logger.info("Dataset summary:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")

    # Create splits
    logger.info("\n" + "=" * 60)
    logger.info("Creating train/val/test splits")
    logger.info("=" * 60)

    splits = create_splits(df)

    # Generate adversarial variants from training threats
    logger.info("\n" + "=" * 60)
    logger.info("Generating adversarial evaluation split")
    logger.info("=" * 60)

    generate_adversarial_split(splits["train"])

    logger.info("\nDone! Next step: run scripts/02_extract_embeddings.py on GPU")


if __name__ == "__main__":
    main()
