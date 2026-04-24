#!/usr/bin/env python3
"""
Step 2: Extract ESM-2 (and optionally ProTrek) embeddings.
*** Run this on GPU ***

Usage:
    python scripts/02_extract_embeddings.py --model esm2 --device cuda --batch-size 8
    python scripts/02_extract_embeddings.py --model protrek --device cuda --batch-size 4
"""
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.models.embeddings import extract_and_save_embeddings
from src.config import SPLITS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["esm2", "protrek"], default="esm2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--splits", nargs="+",
                        default=["train", "val", "test_standard", "test_hard_negative",
                                 "test_seq_divergent", "test_adversarial"])
    args = parser.parse_args()

    logger.info(f"Extracting {args.model} embeddings on {args.device}")

    for split_name in args.splits:
        split_path = SPLITS_DIR / f"{split_name}.parquet"
        if not split_path.exists():
            logger.warning(f"Split {split_name} not found at {split_path}, skipping")
            continue

        df = pd.read_parquet(split_path)
        logger.info(f"\n--- {split_name}: {len(df)} sequences ---")

        extract_and_save_embeddings(
            df=df,
            model_type=args.model,
            device=args.device,
            batch_size=args.batch_size,
            split_name=split_name,
        )

    logger.info("\nDone! Next step: run scripts/03_train_contrastive.py")


if __name__ == "__main__":
    main()
