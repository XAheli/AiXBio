from __future__ import annotations

"""
Evaluation metrics for biosecurity screening.

Includes both standard ML metrics and security-relevant operational metrics.
"""
import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, f1_score, accuracy_score,
    confusion_matrix,
)

from src.config import BOOTSTRAP_ITERATIONS, BOOTSTRAP_CI_ALPHA, SEED

logger = logging.getLogger(__name__)


@dataclass
class ScreeningMetrics:
    """Complete evaluation metrics for a screening method."""
    method_name: str
    split_name: str
    auroc: float
    auprc: float
    recall_at_95_precision: float
    recall_at_99_precision: float
    fpr_at_95_recall: float
    fpr_at_99_recall: float
    f1_optimal: float
    accuracy_optimal: float
    n_samples: int
    n_threat: int
    n_benign: int

    def to_dict(self) -> dict:
        return {
            "method": self.method_name,
            "split": self.split_name,
            "AUROC": f"{self.auroc:.4f}",
            "AUPRC": f"{self.auprc:.4f}",
            "R@P95": f"{self.recall_at_95_precision:.4f}",
            "R@P99": f"{self.recall_at_99_precision:.4f}",
            "FPR@R95": f"{self.fpr_at_95_recall:.4f}",
            "FPR@R99": f"{self.fpr_at_99_recall:.4f}",
            "F1*": f"{self.f1_optimal:.4f}",
            "n": self.n_samples,
        }


def compute_screening_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    method_name: str = "",
    split_name: str = "",
) -> ScreeningMetrics:
    """Compute the full suite of screening metrics."""
    n_threat = int(y_true.sum())
    n_benign = int(len(y_true) - n_threat)

    # Handle degenerate cases
    if n_threat == 0 or n_benign == 0:
        logger.warning(f"Degenerate split: {n_threat} threats, {n_benign} benign")
        return ScreeningMetrics(
            method_name=method_name, split_name=split_name,
            auroc=0.0, auprc=0.0,
            recall_at_95_precision=0.0, recall_at_99_precision=0.0,
            fpr_at_95_recall=1.0, fpr_at_99_recall=1.0,
            f1_optimal=0.0, accuracy_optimal=0.0,
            n_samples=len(y_true), n_threat=n_threat, n_benign=n_benign,
        )

    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    # Recall at precision thresholds
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    recall_at_95_precision = _recall_at_precision(precision, recall, 0.95)
    recall_at_99_precision = _recall_at_precision(precision, recall, 0.99)

    # FPR at recall thresholds
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fpr_at_95_recall = _fpr_at_recall(fpr, tpr, 0.95)
    fpr_at_99_recall = _fpr_at_recall(fpr, tpr, 0.99)

    # Optimal F1
    thresholds = np.linspace(0, 1, 200)
    f1s = [f1_score(y_true, y_score >= t, zero_division=0) for t in thresholds]
    best_t = thresholds[np.argmax(f1s)]
    f1_optimal = max(f1s)
    accuracy_optimal = accuracy_score(y_true, y_score >= best_t)

    return ScreeningMetrics(
        method_name=method_name,
        split_name=split_name,
        auroc=auroc,
        auprc=auprc,
        recall_at_95_precision=recall_at_95_precision,
        recall_at_99_precision=recall_at_99_precision,
        fpr_at_95_recall=fpr_at_95_recall,
        fpr_at_99_recall=fpr_at_99_recall,
        f1_optimal=f1_optimal,
        accuracy_optimal=accuracy_optimal,
        n_samples=len(y_true),
        n_threat=n_threat,
        n_benign=n_benign,
    )


def _recall_at_precision(precision: np.ndarray, recall: np.ndarray,
                         target_precision: float) -> float:
    """Find maximum recall where precision >= target."""
    valid = precision >= target_precision
    if not valid.any():
        return 0.0
    return float(recall[valid].max())


def _fpr_at_recall(fpr: np.ndarray, tpr: np.ndarray,
                   target_recall: float) -> float:
    """Find minimum FPR where recall (TPR) >= target."""
    valid = tpr >= target_recall
    if not valid.any():
        return 1.0
    return float(fpr[valid].min())


def detection_vs_divergence(
    y_true: np.ndarray,
    y_score: np.ndarray,
    kmer_sims: np.ndarray,
    bins: list[float] | None = None,
    threshold: float = 0.5,
) -> dict[str, dict]:
    """
    Compute detection rate as a function of sequence divergence.

    This is the key figure: how does detection degrade as sequences
    become less similar to known threats?

    Args:
        y_true: ground truth labels
        y_score: predicted scores
        kmer_sims: max k-mer similarity to nearest training threat
        bins: similarity bins (upper bounds)
        threshold: classification threshold

    Returns:
        Dict mapping bin label to detection rate
    """
    if bins is None:
        bins = [1.0, 0.7, 0.5, 0.3, 0.2, 0.1]

    threat_mask = y_true == 1
    threat_scores = y_score[threat_mask]
    threat_sims = kmer_sims[threat_mask]

    results = {}
    prev_bound = 1.01

    for upper in bins:
        mask = (threat_sims < prev_bound) & (threat_sims >= upper)
        if mask.sum() > 0:
            detection_rate = (threat_scores[mask] >= threshold).mean()
            bin_label = f"[{upper:.1f}, {prev_bound:.1f})"
            results[bin_label] = {
                "detection_rate": float(detection_rate),
                "n_samples": int(mask.sum()),
                "mean_score": float(threat_scores[mask].mean()),
            }
        prev_bound = upper

    # Below lowest bin
    mask = threat_sims < bins[-1]
    if mask.sum() > 0:
        detection_rate = (threat_scores[mask] >= threshold).mean()
        results[f"[0, {bins[-1]:.1f})"] = {
            "detection_rate": float(detection_rate),
            "n_samples": int(mask.sum()),
            "mean_score": float(threat_scores[mask].mean()),
        }

    return results


# ── Bootstrap Confidence Intervals ────────────────────────────────────

def bootstrap_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_iterations: int = BOOTSTRAP_ITERATIONS,
    ci_alpha: float = BOOTSTRAP_CI_ALPHA,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """
    Compute a metric with bootstrap 95% confidence interval.

    Returns (point_estimate, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    point = metric_fn(y_true, y_score)

    bootstrapped = []
    for _ in range(n_iterations):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        bootstrapped.append(metric_fn(y_true[idx], y_score[idx]))

    if len(bootstrapped) < 10:
        logger.warning(f"Only {len(bootstrapped)} valid bootstrap samples")
        return point, point, point

    bootstrapped = np.array(bootstrapped)
    lower = float(np.percentile(bootstrapped, 100 * ci_alpha / 2))
    upper = float(np.percentile(bootstrapped, 100 * (1 - ci_alpha / 2)))

    return point, lower, upper


def paired_bootstrap_test(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    metric_fn=roc_auc_score,
    n_iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = SEED,
) -> float:
    """
    Paired bootstrap test for comparing two methods.

    Returns p-value for the null hypothesis that method A and B
    have equal performance. Uses the same bootstrap indices for both
    methods to account for correlation.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    observed_diff = metric_fn(y_true, scores_a) - metric_fn(y_true, scores_b)

    count_greater = 0
    valid = 0
    for _ in range(n_iterations):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        diff = metric_fn(y_true[idx], scores_a[idx]) - metric_fn(y_true[idx], scores_b[idx])
        if observed_diff >= 0 and diff <= 0:
            count_greater += 1
        elif observed_diff < 0 and diff >= 0:
            count_greater += 1
        valid += 1

    if valid == 0:
        return 1.0
    return float(count_greater / valid)


def compute_screening_metrics_with_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    method_name: str = "",
    split_name: str = "",
    n_bootstrap: int = BOOTSTRAP_ITERATIONS,
) -> dict:
    """
    Compute screening metrics with bootstrap 95% CIs.

    Returns dict with point estimates and CI bounds for AUROC and AUPRC.
    """
    base = compute_screening_metrics(y_true, y_score, method_name, split_name)
    result = base.to_dict()

    n_threat = int(y_true.sum())
    n_benign = int(len(y_true) - n_threat)
    if n_threat == 0 or n_benign == 0:
        result["AUROC_CI"] = "[N/A]"
        result["AUPRC_CI"] = "[N/A]"
        return result

    auroc_point, auroc_lo, auroc_hi = bootstrap_metric(
        y_true, y_score, roc_auc_score, n_bootstrap
    )
    auprc_point, auprc_lo, auprc_hi = bootstrap_metric(
        y_true, y_score, average_precision_score, n_bootstrap
    )

    result["AUROC_CI"] = f"[{auroc_lo:.4f}, {auroc_hi:.4f}]"
    result["AUPRC_CI"] = f"[{auprc_lo:.4f}, {auprc_hi:.4f}]"
    result["AUROC_lo"] = f"{auroc_lo:.4f}"
    result["AUROC_hi"] = f"{auroc_hi:.4f}"

    return result


def detection_vs_divergence_with_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    kmer_sims: np.ndarray,
    bins: list[float] | None = None,
    threshold: float = 0.5,
    n_bootstrap: int = BOOTSTRAP_ITERATIONS,
    seed: int = SEED,
) -> dict[str, dict]:
    """
    Detection rate vs divergence with bootstrap CIs per bin.
    """
    if bins is None:
        bins = [1.0, 0.7, 0.5, 0.3, 0.2, 0.1]

    threat_mask = y_true == 1
    threat_scores = y_score[threat_mask]
    threat_sims = kmer_sims[threat_mask]

    rng = np.random.RandomState(seed)
    results = {}
    prev_bound = 1.01

    all_bounds = list(bins) + [0.0]

    for i, upper in enumerate(bins):
        mask = (threat_sims < prev_bound) & (threat_sims >= upper)
        if mask.sum() > 0:
            scores_in_bin = threat_scores[mask]
            n = len(scores_in_bin)
            det_rate = float((scores_in_bin >= threshold).mean())

            # Bootstrap CI for detection rate
            boot_rates = []
            for _ in range(n_bootstrap):
                idx = rng.choice(n, n, replace=True)
                boot_rates.append((scores_in_bin[idx] >= threshold).mean())
            boot_rates = np.array(boot_rates)
            ci_lo = float(np.percentile(boot_rates, 2.5))
            ci_hi = float(np.percentile(boot_rates, 97.5))

            bin_label = f"[{upper:.1f}, {prev_bound:.1f})"
            results[bin_label] = {
                "detection_rate": det_rate,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "n_samples": int(n),
                "mean_score": float(scores_in_bin.mean()),
            }
        prev_bound = upper

    # Below lowest bin
    mask = threat_sims < bins[-1]
    if mask.sum() > 0:
        scores_in_bin = threat_scores[mask]
        n = len(scores_in_bin)
        det_rate = float((scores_in_bin >= threshold).mean())

        boot_rates = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n, n, replace=True)
            boot_rates.append((scores_in_bin[idx] >= threshold).mean())
        boot_rates = np.array(boot_rates)
        ci_lo = float(np.percentile(boot_rates, 2.5))
        ci_hi = float(np.percentile(boot_rates, 97.5))

        results[f"[0, {bins[-1]:.1f})"] = {
            "detection_rate": det_rate,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "n_samples": int(n),
            "mean_score": float(scores_in_bin.mean()),
        }

    return results
