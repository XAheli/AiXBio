from __future__ import annotations

"""
Contrastive learning model for function-aware biosecurity screening.

This is the core contribution: a learned decision boundary in embedding space
that separates threat functions from benign homologs, trained with hard-negative
mining and supervised contrastive loss.
"""
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.config import (
    PROJECTION_DIM, TEMPERATURE, CONTRASTIVE_LR,
    CONTRASTIVE_EPOCHS, CONTRASTIVE_BATCH_SIZE,
    HARD_NEGATIVE_RATIO, CHECKPOINTS_DIR, SEED,
)

logger = logging.getLogger(__name__)


# ── Dataset ────────────────────────────────────────────────────────────

class EmbeddingDataset(Dataset):
    """Dataset of pre-computed embeddings with threat/benign labels."""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray,
                 subcategories: np.ndarray | None = None):
        self.embeddings = torch.from_numpy(embeddings).float()
        self.labels = torch.from_numpy(labels).long()  # 1=threat, 0=benign
        self.subcategories = subcategories

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class HardNegativeSampler:
    """
    Mines hard negatives: for each threat anchor, finds the closest benign
    sequences in embedding space. This forces the model to learn a tight
    decision boundary rather than relying on trivial separation.
    """

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray,
                 n_hard_negatives: int = HARD_NEGATIVE_RATIO):
        self.n_hard = n_hard_negatives
        self.threat_idx = np.where(labels == 1)[0]
        self.benign_idx = np.where(labels == 0)[0]

        # Pre-compute cosine similarity between threats and benigns
        threat_emb = embeddings[self.threat_idx]
        benign_emb = embeddings[self.benign_idx]

        # Normalize
        threat_norm = threat_emb / (np.linalg.norm(threat_emb, axis=1, keepdims=True) + 1e-8)
        benign_norm = benign_emb / (np.linalg.norm(benign_emb, axis=1, keepdims=True) + 1e-8)

        # Similarity matrix: (n_threats, n_benign)
        self.sim_matrix = threat_norm @ benign_norm.T

    def get_hard_negatives(self, threat_local_idx: int) -> np.ndarray:
        """Get indices of hardest benign negatives for a given threat."""
        sims = self.sim_matrix[threat_local_idx]
        top_k = min(self.n_hard, len(sims))
        hard_idx = np.argpartition(sims, -top_k)[-top_k:]
        return self.benign_idx[hard_idx]


class HardNegativeDataset(Dataset):
    """
    Upsampled dataset that ensures each batch contains threats paired with
    their hardest benign negatives in embedding space, forcing the model
    to learn a tight decision boundary.
    """

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray,
                 subcategories: np.ndarray | None = None,
                 n_hard_negatives: int = HARD_NEGATIVE_RATIO):
        self.embeddings = torch.from_numpy(embeddings).float()
        self.labels = torch.from_numpy(labels).long()
        self.n_hard_negatives = n_hard_negatives

        self.threat_idx = np.where(labels == 1)[0]
        self.benign_idx = np.where(labels == 0)[0]
        self.sampler = HardNegativeSampler(embeddings, labels, n_hard_negatives)
        self._rng = np.random.RandomState(SEED)

    def __len__(self):
        return len(self.threat_idx) * (1 + self.n_hard_negatives)

    def __getitem__(self, idx):
        # Cycle through threats, yielding the threat then its hard negatives
        group_size = 1 + self.n_hard_negatives
        threat_cycle = idx // group_size
        position = idx % group_size

        threat_local_idx = threat_cycle % len(self.threat_idx)
        threat_global_idx = self.threat_idx[threat_local_idx]

        if position == 0:
            # Return the threat itself
            return self.embeddings[threat_global_idx], self.labels[threat_global_idx]
        else:
            # Return a hard negative (with slight randomization to avoid overfitting)
            hard_neg_indices = self.sampler.get_hard_negatives(threat_local_idx)
            neg_idx = hard_neg_indices[(position - 1) % len(hard_neg_indices)]
            return self.embeddings[neg_idx], self.labels[neg_idx]


# ── Model ──────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """
    Learned projection from frozen PLM embeddings to a discriminative
    screening space. This is where the decision boundary lives.
    """

    def __init__(self, input_dim: int, projection_dim: int = PROJECTION_DIM,
                 hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, projection_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class FunctionAwareScreener(nn.Module):
    """
    Full screening model: projection head + classification head.

    The projection head maps embeddings to a space where contrastive loss
    separates threat functions from benign ones. The classification head
    provides the final threat/benign decision.
    """

    def __init__(self, input_dim: int, projection_dim: int = PROJECTION_DIM):
        super().__init__()
        self.projection = ProjectionHead(input_dim, projection_dim)
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        z = self.projection(x)
        logit = self.classifier(z).squeeze(-1)
        return z, logit

    def predict_proba(self, x):
        """Return threat probability."""
        _, logit = self.forward(x)
        return torch.sigmoid(logit)

    def get_projection(self, x):
        """Return normalized projection (for similarity computation)."""
        return self.projection(x)


# ── Loss functions ─────────────────────────────────────────────────────

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020).

    Pulls together embeddings of the same class (threat or benign)
    and pushes apart embeddings of different classes.
    """

    def __init__(self, temperature: float = TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]

        # Normalize features (ProjectionHead already normalizes, but this
        # ensures correctness if SupConLoss is used with other inputs)
        features = F.normalize(features, dim=1)

        # Compute similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature  # (B, B)

        # Mask: same class = 1, different class = 0, diagonal = 0
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask.fill_diagonal_(0)

        # For numerical stability
        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        # Mask out self-contrast
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        # Compute log prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Mean of log-likelihood over positive pairs
        n_positives = mask.sum(1)
        mean_log_prob = (mask * log_prob).sum(1) / n_positives.clamp(min=1)

        loss = -mean_log_prob.mean()
        return loss


class CombinedLoss(nn.Module):
    """Combined supervised contrastive + binary cross-entropy loss."""

    def __init__(self, temperature: float = TEMPERATURE, alpha: float = 0.5):
        super().__init__()
        self.supcon = SupConLoss(temperature)
        self.bce = nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, projections: torch.Tensor, logits: torch.Tensor,
                labels: torch.Tensor) -> tuple[torch.Tensor, dict]:
        loss_con = self.supcon(projections, labels)
        loss_bce = self.bce(logits, labels.float())
        loss_total = self.alpha * loss_con + (1 - self.alpha) * loss_bce

        return loss_total, {
            "loss_total": loss_total.item(),
            "loss_contrastive": loss_con.item(),
            "loss_bce": loss_bce.item(),
        }


# ── Training ───────────────────────────────────────────────────────────

def train_screener(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    val_embeddings: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
    input_dim: int | None = None,
    projection_dim: int = PROJECTION_DIM,
    lr: float = CONTRASTIVE_LR,
    epochs: int = CONTRASTIVE_EPOCHS,
    batch_size: int = CONTRASTIVE_BATCH_SIZE,
    use_hard_negatives: bool = True,
    device: str = "cuda",
    save_path: Path | None = None,
) -> tuple[FunctionAwareScreener, dict]:
    """
    Train the function-aware screener on pre-computed embeddings.

    Returns the trained model and training history.
    """
    torch.manual_seed(SEED)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    if input_dim is None:
        input_dim = train_embeddings.shape[1]

    # Build datasets
    if use_hard_negatives:
        train_dataset = HardNegativeDataset(train_embeddings, train_labels)
    else:
        train_dataset = EmbeddingDataset(train_embeddings, train_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=0,
    )

    val_loader = None
    if val_embeddings is not None and val_labels is not None:
        val_dataset = EmbeddingDataset(val_embeddings, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Build model
    model = FunctionAwareScreener(input_dim, projection_dim).to(device)
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "train_con": [], "train_bce": [],
               "val_loss": [], "val_auroc": []}

    best_val_auroc = -1.0
    if save_path is None:
        save_path = CHECKPOINTS_DIR / "best_screener.pt"

    for epoch in range(epochs):
        # ── Train ──────────────────────────────────────────────────
        model.train()
        epoch_losses = {"loss_total": [], "loss_contrastive": [], "loss_bce": []}

        for embeddings_batch, labels_batch in train_loader:
            embeddings_batch = embeddings_batch.to(device)
            labels_batch = labels_batch.to(device)

            projections, logits = model(embeddings_batch)
            loss, loss_dict = criterion(projections, logits, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k, v in loss_dict.items():
                epoch_losses[k].append(v)

        scheduler.step()

        # Record training metrics
        avg_train = {k: np.mean(v) for k, v in epoch_losses.items()}
        history["train_loss"].append(avg_train["loss_total"])
        history["train_con"].append(avg_train["loss_contrastive"])
        history["train_bce"].append(avg_train["loss_bce"])

        # ── Validate ───────────────────────────────────────────────
        if val_loader is not None:
            model.eval()
            val_preds, val_true = [], []
            val_loss_total = []

            with torch.no_grad():
                for emb_b, lab_b in val_loader:
                    emb_b, lab_b = emb_b.to(device), lab_b.to(device)
                    proj, logit = model(emb_b)
                    loss, _ = criterion(proj, logit, lab_b)
                    val_loss_total.append(loss.item())

                    probs = torch.sigmoid(logit).cpu().numpy()
                    val_preds.extend(probs)
                    val_true.extend(lab_b.cpu().numpy())

            from sklearn.metrics import roc_auc_score
            val_preds = np.array(val_preds)
            val_true = np.array(val_true)

            if len(np.unique(val_true)) > 1:
                val_auroc = roc_auc_score(val_true, val_preds)
            else:
                logger.warning("Validation set has only one class — AUROC undefined, skipping")
                val_auroc = float("nan")

            history["val_loss"].append(np.mean(val_loss_total))
            history["val_auroc"].append(val_auroc)

            if not np.isnan(val_auroc) and val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train loss: {avg_train['loss_total']:.4f} "
                    f"(con: {avg_train['loss_contrastive']:.4f}, bce: {avg_train['loss_bce']:.4f}) | "
                    f"Val AUROC: {val_auroc:.4f}"
                )
        elif (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train loss: {avg_train['loss_total']:.4f}"
            )

    # Load best model if we saved one
    if save_path and save_path.exists():
        model.load_state_dict(torch.load(save_path, weights_only=True))
        logger.info(f"Loaded best model (val AUROC: {best_val_auroc:.4f})")

    return model, history
