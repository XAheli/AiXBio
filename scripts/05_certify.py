#!/usr/bin/env python3
"""
Step 5 (stretch): Run certified robustness experiments.
Requires GPU for re-embedding noisy sequences.

Usage:
    python scripts/05_certify.py --embedding esm2 --device cuda --n-samples 200 --max-sequences 50
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
import matplotlib
import matplotlib.pyplot as plt

from src.config import SPLITS_DIR, CHECKPOINTS_DIR, RESULTS_DIR, FIGURES_DIR
from src.models.embeddings import load_embeddings, ESM2Embedder, ProTrekEmbedder
from src.models.contrastive import FunctionAwareScreener
from src.screening.certified import run_certification_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

matplotlib.rcParams.update({
    "font.family": "serif", "font.size": 11, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", default="esm2", choices=["esm2", "protrek"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Monte Carlo samples per sequence for certification")
    parser.add_argument("--max-sequences", type=int, default=50,
                        help="Max sequences to certify (for speed)")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load test data
    test_df = pd.read_parquet(SPLITS_DIR / "test_standard.parquet")
    test_seqs = test_df["sequence"].tolist()
    test_labels = test_df["label"].map({"threat": 1, "benign": 0}).values

    # Subsample for speed
    if len(test_seqs) > args.max_sequences:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(test_seqs), args.max_sequences, replace=False)
        test_seqs = [test_seqs[i] for i in idx]
        test_labels = test_labels[idx]

    logger.info(f"Certifying {len(test_seqs)} sequences with {args.n_samples} samples each")

    # Load model
    model_path = CHECKPOINTS_DIR / f"screener_{args.embedding}.pt"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return

    # Load embedder matching the model's training embedding type
    logger.info(f"Loading {args.embedding} embedder on {args.device}...")
    if args.embedding == "esm2":
        embedder = ESM2Embedder(device=args.device)
    elif args.embedding == "protrek":
        embedder = ProTrekEmbedder(device=args.device)

    # Load contrastive model
    sample_emb, _ = load_embeddings(args.embedding, "train")
    input_dim = sample_emb.shape[1]
    screener = FunctionAwareScreener(input_dim)
    screener.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
    screener.eval()

    # Wrap for certification
    def embedder_fn(sequences: list[str]) -> np.ndarray:
        return embedder.embed_batch(sequences, batch_size=4)

    def classifier_fn(embeddings: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(embeddings).float()
            return screener.predict_proba(t).numpy()

    # Run certification
    results = run_certification_experiment(
        sequences=test_seqs,
        labels=test_labels,
        base_classifier=classifier_fn,
        embedder=embedder_fn,
        mutation_counts=[1, 2, 3, 5, 8, 10],
        mutation_types=["conservative", "random"],
        n_samples=args.n_samples,
    )

    # Save results
    results_path = RESULTS_DIR / "certification_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # ── Plot: Certified accuracy vs mutation budget ────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for mut_type, ax, title in [
        ("conservative", ax1, "Conservative Substitutions"),
        ("random", ax2, "Random Substitutions"),
    ]:
        ks = []
        empirical_accs = []
        certified_accs = []

        for key, data in results.items():
            if data["mutation_type"] == mut_type:
                ks.append(data["n_mutations"])
                empirical_accs.append(data["accuracy"])
                certified_accs.append(data["certified_accuracy"])

        # Sort by k
        order = np.argsort(ks)
        ks = [ks[i] for i in order]
        empirical_accs = [empirical_accs[i] for i in order]
        certified_accs = [certified_accs[i] for i in order]

        ax.plot(ks, empirical_accs, "o-", color="#D95319", linewidth=2,
                markersize=6, label="Empirical Accuracy")
        ax.plot(ks, certified_accs, "s--", color="#0072BD", linewidth=2,
                markersize=6, label="Certified Accuracy")
        ax.fill_between(ks, certified_accs, empirical_accs,
                        alpha=0.15, color="#EDB120", label="Certifiability Gap")

        ax.set_xlabel("Number of Mutations (k)")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.legend()
        ax.set_ylim([-0.05, 1.05])
        ax.set_xticks(ks)

    plt.suptitle("Empirical vs. Certified Robustness Under Biological Mutations", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "certified_robustness.pdf")
    plt.savefig(FIGURES_DIR / "certified_robustness.png")
    plt.close()

    logger.info(f"Certification figure saved to {FIGURES_DIR / 'certified_robustness.pdf'}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("CERTIFICATION SUMMARY")
    logger.info("=" * 60)
    for key, data in sorted(results.items()):
        logger.info(
            f"  {key:25s} | Emp: {data['accuracy']:.3f} | "
            f"Cert: {data['certified_accuracy']:.3f} | "
            f"Gap: {data['accuracy'] - data['certified_accuracy']:.3f} | "
            f"p_lower: {data['mean_p_lower']:.3f}"
        )


if __name__ == "__main__":
    main()
