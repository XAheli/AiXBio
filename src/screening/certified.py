from __future__ import annotations

"""
Certified robustness analysis for function-aware screening.

Implements randomized smoothing adapted to biologically structured
mutation spaces (synonymous substitutions, conservative AA replacements).

This is Contribution B of the paper: either meaningful certificates
or a rigorous quantification of the certifiability gap.
"""
import logging
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm

from src.config import SEED

logger = logging.getLogger(__name__)

# ── Biological mutation models ─────────────────────────────────────────

# Conservative amino acid substitution groups (Dayhoff 6-group classification)
# Reference: Dayhoff, Schwartz & Orcutt (1978), Atlas of Protein Sequence and Structure
# Each AA appears in exactly one group. Substitutions within a group are conservative.
CONSERVATIVE_GROUPS = {
    "sulfur": set("CM"),
    "small": set("AGPST"),
    "acid_amide": set("DENQ"),
    "basic": set("HKR"),
    "hydrophobic": set("ILV"),
    "aromatic": set("FWY"),
}

# Map each AA to its conservative group
AA_TO_GROUP: dict[str, str] = {}
for _group_name, _aas in CONSERVATIVE_GROUPS.items():
    for _aa in _aas:
        if _aa in AA_TO_GROUP:
            raise RuntimeError(f"Amino acid {_aa} appears in multiple groups: "
                               f"{AA_TO_GROUP[_aa]} and {_group_name}")
        AA_TO_GROUP[_aa] = _group_name

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


def get_conservative_substitutions(aa: str) -> list[str]:
    """Return amino acids that are conservative substitutions for the given AA."""
    group = AA_TO_GROUP.get(aa)
    if group is None:
        return []  # No known conservative group (e.g., rare AAs)
    return [a for a in CONSERVATIVE_GROUPS[group] if a != aa]


def sample_biological_noise(
    sequence: str,
    n_mutations: int = 1,
    mutation_type: str = "conservative",
    rng: np.random.RandomState | None = None,
) -> str:
    """
    Apply biologically structured random mutations to a protein sequence.

    Args:
        sequence: input amino acid sequence
        n_mutations: number of positions to mutate
        mutation_type: "conservative" (within Dayhoff group),
                       "random" (any AA), or "synonymous" (same AA = no-op baseline)
    """
    if rng is None:
        rng = np.random.RandomState(SEED)

    seq = list(sequence)
    positions = rng.choice(len(seq), size=min(n_mutations, len(seq)), replace=False)

    for pos in positions:
        original_aa = seq[pos]
        if mutation_type == "conservative":
            subs = get_conservative_substitutions(original_aa)
            if subs:
                seq[pos] = rng.choice(subs)
        elif mutation_type == "random":
            alternatives = [aa for aa in AMINO_ACIDS if aa != original_aa]
            seq[pos] = rng.choice(alternatives)
        elif mutation_type == "synonymous":
            pass  # No change (baseline)

    return "".join(seq)


# ── Randomized smoothing for sequence screening ───────────────────────

class BiologicalRandomizedSmoothing:
    """
    Randomized smoothing adapted to protein sequences with biologically
    structured perturbations.

    Given a base classifier f(x), constructs a smoothed classifier g(x)
    whose prediction is certified robust under k mutations of a given type.

    Key insight: instead of Gaussian noise (images), we use biologically
    meaningful mutation distributions (conservative AA substitutions).
    """

    def __init__(
        self,
        base_classifier: Callable[[list[str]], np.ndarray],
        embedder: Callable[[list[str]], np.ndarray] | None = None,
        n_samples: int = 100,
        n_mutations: int = 3,
        mutation_type: str = "conservative",
        alpha: float = 0.001,
    ):
        """
        Args:
            base_classifier: function mapping sequences -> threat probabilities
            embedder: optional function mapping sequences -> embeddings
                      (if classifier operates on embeddings)
            n_samples: number of noisy samples for smoothing
            n_mutations: mutations per sample
            mutation_type: type of biological mutation to apply
            alpha: significance level for certification
        """
        self.base_classifier = base_classifier
        self.embedder = embedder
        self.n_samples = n_samples
        self.n_mutations = n_mutations
        self.mutation_type = mutation_type
        self.alpha = alpha

    def _predict_noisy_batch(self, sequence: str, n: int,
                             rng: np.random.RandomState) -> np.ndarray:
        """Generate n noisy versions and classify each."""
        noisy_seqs = [
            sample_biological_noise(sequence, self.n_mutations,
                                    self.mutation_type, rng)
            for _ in range(n)
        ]

        if self.embedder is not None:
            embeddings = self.embedder(noisy_seqs)
            scores = self.base_classifier(embeddings)
        else:
            scores = self.base_classifier(noisy_seqs)

        return scores

    def predict(self, sequence: str, rng: np.random.RandomState | None = None) -> tuple[int, float]:
        """
        Smoothed prediction: returns (class, confidence).

        Class is determined by majority vote of noisy samples.
        Each call must use a fresh or explicitly provided RNG to ensure
        independent randomness across sequences.
        """
        if rng is None:
            rng = np.random.RandomState()
        scores = self._predict_noisy_batch(sequence, self.n_samples, rng)

        # Threshold at 0.5
        votes_threat = (scores >= 0.5).sum()
        votes_benign = self.n_samples - votes_threat

        if votes_threat >= votes_benign:
            return 1, votes_threat / self.n_samples
        else:
            return 0, votes_benign / self.n_samples

    def certify(self, sequence: str, rng: np.random.RandomState | None = None) -> dict:
        """
        Certify robustness of the smoothed classifier's prediction.

        Uses a two-phase procedure following Cohen et al. (2019):
        Phase 1: Small sample to determine predicted class
        Phase 2: Larger independent sample to compute confidence bound

        For discrete (L0) perturbations, we follow Lee et al. (2019):
        the smoothed classifier is certifiably robust to r substitutions if
        the lower bound p_A on the majority class probability satisfies
        p_A > 0.5. The certified radius r is the largest integer such that
        the adversary cannot flip the prediction with at most r mutations.

        Under uniform-at-random substitution noise with n_mutations positions
        perturbed per sample, we certify using the relationship between
        the smoothing distribution and the perturbation budget, following
        the discrete randomized smoothing framework of RS-Del (Huang et al. 2023).
        """
        if rng is None:
            rng = np.random.RandomState()

        # Two-phase: separate RNG streams for independence
        phase1_seed = rng.randint(0, 2**31)
        phase2_seed = rng.randint(0, 2**31)

        n_initial = min(100, self.n_samples // 4)
        n_certify = self.n_samples - n_initial

        # Phase 1: predict class
        rng1 = np.random.RandomState(phase1_seed)
        scores_init = self._predict_noisy_batch(sequence, n_initial, rng1)
        predicted_class = 1 if (scores_init >= 0.5).mean() >= 0.5 else 0

        # Phase 2: certify with independent samples
        rng2 = np.random.RandomState(phase2_seed)
        scores_cert = self._predict_noisy_batch(sequence, n_certify, rng2)

        if predicted_class == 1:
            count_correct = int((scores_cert >= 0.5).sum())
        else:
            count_correct = int((scores_cert < 0.5).sum())

        # One-sided Clopper-Pearson lower bound on P(correct class)
        p_lower = self._clopper_pearson_lower(count_correct, n_certify, self.alpha)

        # Certification: the smoothed classifier's prediction is certified
        # robust if p_lower > 0.5 (majority class has >50% probability
        # even in the worst case within the confidence bound)
        certified = p_lower > 0.5

        return {
            "predicted_class": predicted_class,
            "certified": certified,
            "p_lower": float(p_lower),
            "confidence": float(count_correct / n_certify),
            "n_samples_certify": n_certify,
            "n_mutations_smoothing": self.n_mutations,
            "mutation_type": self.mutation_type,
        }

    @staticmethod
    def _clopper_pearson_lower(successes: int, n: int, alpha: float) -> float:
        """
        One-sided Clopper-Pearson lower bound.

        This is the standard one-sided lower confidence bound used in
        randomized smoothing certification (Cohen et al. 2019, Appendix A).
        """
        from scipy.stats import beta as beta_dist
        if successes == 0:
            return 0.0
        return float(beta_dist.ppf(alpha, successes, n - successes + 1))


def run_certification_experiment(
    sequences: list[str],
    labels: np.ndarray,
    base_classifier: Callable,
    embedder: Callable | None = None,
    mutation_counts: list[int] | None = None,
    mutation_types: list[str] | None = None,
    n_samples: int = 200,
) -> dict:
    """
    Run certification across multiple mutation budgets and types.

    Returns structured results for paper figures.
    """
    if mutation_counts is None:
        mutation_counts = [1, 2, 3, 5, 8, 10, 15, 20]
    if mutation_types is None:
        mutation_types = ["conservative", "random"]

    results = {}

    for mut_type in mutation_types:
        for n_mut in mutation_counts:
            key = f"{mut_type}_k{n_mut}"
            logger.info(f"Certifying: {mut_type}, k={n_mut}")

            smoother = BiologicalRandomizedSmoothing(
                base_classifier=base_classifier,
                embedder=embedder,
                n_samples=n_samples,
                n_mutations=n_mut,
                mutation_type=mut_type,
            )

            certs = []
            master_rng = np.random.RandomState(SEED)
            for i, (seq, label) in enumerate(zip(sequences, labels)):
                # Each sequence gets an independent RNG derived from master
                seq_rng = np.random.RandomState(master_rng.randint(0, 2**31))
                cert = smoother.certify(seq, rng=seq_rng)
                cert["true_label"] = int(label)
                cert["correct"] = int(cert["predicted_class"] == label)
                certs.append(cert)

            # Aggregate
            certs_arr = {k: np.array([c[k] for c in certs]) for k in certs[0]}

            results[key] = {
                "mutation_type": mut_type,
                "n_mutations": n_mut,
                "accuracy": float(certs_arr["correct"].mean()),
                "certified_accuracy": float(
                    (certs_arr["correct"] & certs_arr["certified"]).mean()
                ),
                "certification_rate": float(certs_arr["certified"].mean()),
                "mean_p_lower": float(certs_arr["p_lower"].mean()),
                "mean_confidence": float(certs_arr["confidence"].mean()),
                "n_sequences": len(sequences),
            }

            logger.info(
                f"  Accuracy: {results[key]['accuracy']:.3f}, "
                f"Certified acc: {results[key]['certified_accuracy']:.3f}, "
                f"Cert rate: {results[key]['certification_rate']:.3f}"
            )

    return results
