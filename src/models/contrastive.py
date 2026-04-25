from __future__ import annotations

"""
Contrastive learning model for function-aware biosecurity screening.

Architecture components:
  - AttentionPooling: learned attention over ESM-2 hidden states (replaces mean pooling)
  - ProjectionHead: maps pooled embeddings to a contrastive screening space
  - FunctionAwareScreener: multi-scale classifier with optional attention pooling
  - MixupAugmenter: embedding-space data augmentation for small datasets
  - EndToEndScreener: wraps ESM-2 backbone for fine-tuning ablation

Training uses supervised contrastive loss with hard-negative mining,
adversarial augmentation from ProteinMPNN variants, and Mixup regularization.
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
    ESM2_EMBEDDING_DIM, ATTENTION_HEADS, ATTENTION_HIDDEN_DIM,
    MULTI_SCALE_HIDDEN, MIXUP_ALPHA, MIXUP_PROB,
)

logger = logging.getLogger(__name__)


# ── Datasets ──────────────────────────────────────────────────────────

class EmbeddingDataset(Dataset):
    """Dataset of pre-computed embeddings with threat/benign labels."""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray,
                 subcategories: np.ndarray | None = None):
        self.embeddings = torch.from_numpy(embeddings).float()
        self.labels = torch.from_numpy(labels).long()
        self.subcategories = subcategories

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class HardNegativeSampler:
    """
    Mines hard negatives: for each threat anchor, finds the closest benign
    sequences in embedding space.
    """

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray,
                 n_hard_negatives: int = HARD_NEGATIVE_RATIO):
        self.n_hard = n_hard_negatives
        self.threat_idx = np.where(labels == 1)[0]
        self.benign_idx = np.where(labels == 0)[0]

        threat_emb = embeddings[self.threat_idx]
        benign_emb = embeddings[self.benign_idx]

        threat_norm = threat_emb / (np.linalg.norm(threat_emb, axis=1, keepdims=True) + 1e-8)
        benign_norm = benign_emb / (np.linalg.norm(benign_emb, axis=1, keepdims=True) + 1e-8)

        self.sim_matrix = threat_norm @ benign_norm.T

    def get_hard_negatives(self, threat_local_idx: int) -> np.ndarray:
        sims = self.sim_matrix[threat_local_idx]
        top_k = min(self.n_hard, len(sims))
        hard_idx = np.argpartition(sims, -top_k)[-top_k:]
        return self.benign_idx[hard_idx]


class HardNegativeDataset(Dataset):
    """
    Upsampled dataset pairing each threat with its hardest benign negatives.
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

    def __len__(self):
        return len(self.threat_idx) * (1 + self.n_hard_negatives)

    def __getitem__(self, idx):
        group_size = 1 + self.n_hard_negatives
        threat_cycle = idx // group_size
        position = idx % group_size

        threat_local_idx = threat_cycle % len(self.threat_idx)
        threat_global_idx = self.threat_idx[threat_local_idx]

        if position == 0:
            return self.embeddings[threat_global_idx], self.labels[threat_global_idx]
        else:
            hard_neg_indices = self.sampler.get_hard_negatives(threat_local_idx)
            neg_idx = hard_neg_indices[(position - 1) % len(hard_neg_indices)]
            return self.embeddings[neg_idx], self.labels[neg_idx]


# ── Attention Pooling ─────────────────────────────────────────────────

class AttentionPooling(nn.Module):
    """
    Learned attention-weighted pooling over ESM-2 per-position hidden states.

    Instead of treating every amino acid position equally (mean pooling),
    this module learns which positions are most informative for threat
    detection. For pore-forming toxins, we expect the model to attend
    to the pore-forming domain rather than signal peptides or linker regions.

    The attention weights are interpretable: they show which protein
    regions drive the screening decision.
    """

    def __init__(self, input_dim: int = ESM2_EMBEDDING_DIM,
                 hidden_dim: int = ATTENTION_HIDDEN_DIM,
                 n_heads: int = ATTENTION_HEADS):
        super().__init__()
        self.n_heads = n_heads
        self.query = nn.Parameter(torch.randn(n_heads, hidden_dim))
        self.key_proj = nn.Linear(input_dim, hidden_dim * n_heads)
        self.value_proj = nn.Linear(input_dim, input_dim)
        self.scale = hidden_dim ** -0.5

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor | None = None):
        """
        Args:
            hidden_states: (batch, seq_len, input_dim)
            attention_mask: (batch, seq_len) — 1 for real, 0 for padding

        Returns:
            pooled: (batch, input_dim)
            attention_weights: (batch, n_heads, seq_len)
        """
        B, L, D = hidden_states.shape

        # Keys: (batch, seq_len, n_heads * hidden_dim) → (batch, n_heads, seq_len, hidden_dim)
        keys = self.key_proj(hidden_states).view(B, L, self.n_heads, -1).transpose(1, 2)

        # Query: (n_heads, hidden_dim) → (1, n_heads, 1, hidden_dim)
        query = self.query.unsqueeze(0).unsqueeze(2)

        # Attention scores: (batch, n_heads, 1, seq_len)
        scores = (query @ keys.transpose(-2, -1)) * self.scale
        scores = scores.squeeze(2)  # (batch, n_heads, seq_len)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).expand_as(scores)  # (batch, n_heads, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)  # (batch, n_heads, seq_len)

        # Values: (batch, seq_len, input_dim)
        values = self.value_proj(hidden_states)

        # Weighted sum: average over heads
        # (batch, n_heads, seq_len) × (batch, seq_len, input_dim)
        pooled = torch.bmm(
            attention_weights.mean(dim=1).unsqueeze(1),  # (batch, 1, seq_len)
            values  # (batch, seq_len, input_dim)
        ).squeeze(1)  # (batch, input_dim)

        return pooled, attention_weights


# ── Mixup Augmentation ────────────────────────────────────────────────

class MixupAugmenter:
    """
    Embedding-space Mixup for small-dataset augmentation.

    Generates synthetic training examples by interpolating between
    pairs of same-class embeddings:
      x_mix = lambda * x_a + (1 - lambda) * x_b
      y_mix = y_a (same class)

    Lambda ~ Beta(alpha, alpha), with alpha=0.2 producing most
    interpolations near the endpoints (preserving class semantics).

    Reference: Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018
    """

    def __init__(self, alpha: float = MIXUP_ALPHA, prob: float = MIXUP_PROB):
        self.alpha = alpha
        self.prob = prob

    def __call__(self, embeddings: torch.Tensor, labels: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply Mixup to a batch. Returns augmented (embeddings, labels)."""
        if torch.rand(1).item() > self.prob:
            return embeddings, labels

        batch_size = embeddings.shape[0]

        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)

        # Shuffle indices for pairing (same-class pairs only)
        threat_mask = labels == 1
        benign_mask = labels == 0

        mixed_embeddings = embeddings.clone()

        for mask in [threat_mask, benign_mask]:
            indices = torch.where(mask)[0]
            if len(indices) < 2:
                continue
            perm = indices[torch.randperm(len(indices))]
            mixed_embeddings[indices] = lam * embeddings[indices] + (1 - lam) * embeddings[perm]

        return mixed_embeddings, labels


# ── Model ─────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """
    Learned projection from PLM embeddings to a discriminative screening space.
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
    Full screening model with optional multi-scale classification.

    When multi_scale=False (original): classifier sees only projection (256-dim).
    When multi_scale=True (improved): classifier sees concat(raw, projection) (1536-dim).

    The SupCon loss always operates on the projection only.
    The multi-scale classifier retains the full raw representation that
    makes KNN competitive, while adding the learned contrastive boundary.
    """

    def __init__(self, input_dim: int, projection_dim: int = PROJECTION_DIM,
                 multi_scale: bool = False):
        super().__init__()
        self.multi_scale = multi_scale
        self.input_dim = input_dim
        self.projection = ProjectionHead(input_dim, projection_dim)

        if multi_scale:
            classifier_input = input_dim + projection_dim
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input, MULTI_SCALE_HIDDEN),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(MULTI_SCALE_HIDDEN, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1),
            )
        else:
            classifier_input = projection_dim
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1),
            )

    def forward(self, x):
        z = self.projection(x)
        if self.multi_scale:
            classifier_input = torch.cat([x, z], dim=-1)
        else:
            classifier_input = z
        logit = self.classifier(classifier_input).squeeze(-1)
        return z, logit

    def predict_proba(self, x):
        _, logit = self.forward(x)
        return torch.sigmoid(logit)

    def get_projection(self, x):
        return self.projection(x)


# ── End-to-End Screener (for fine-tuning ablation) ────────────────────

class EndToEndScreener(nn.Module):
    """
    Wraps ESM-2 backbone + attention pooling + projection + classifier
    for end-to-end fine-tuning experiments.

    Used in ablation studies to compare frozen vs fine-tuned ESM-2.
    """

    def __init__(self, esm_model, tokenizer,
                 projection_dim: int = PROJECTION_DIM,
                 finetune_layers: int = 0,
                 use_attention_pool: bool = True,
                 multi_scale: bool = False):
        super().__init__()
        self.esm = esm_model
        self.tokenizer = tokenizer
        self.use_attention_pool = use_attention_pool

        # Freeze all ESM-2, then selectively unfreeze
        for param in self.esm.parameters():
            param.requires_grad = False

        if finetune_layers > 0:
            layers = list(self.esm.encoder.layer)
            for layer in layers[-finetune_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        if use_attention_pool:
            self.attention_pool = AttentionPooling(ESM2_EMBEDDING_DIM)
        self.screener = FunctionAwareScreener(
            ESM2_EMBEDDING_DIM, projection_dim, multi_scale
        )

    def forward(self, sequences: list[str], device: str = "cuda"):
        inputs = self.tokenizer(
            sequences, return_tensors="pt", padding=True,
            truncation=True, max_length=1024,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.esm(**inputs)
        hidden_states = outputs.last_hidden_state

        if self.use_attention_pool:
            pooled, _ = self.attention_pool(hidden_states, inputs["attention_mask"])
        else:
            # Mean pool (excluding CLS/EOS)
            mask = inputs["attention_mask"].clone()
            mask[:, 0] = 0
            for j in range(mask.shape[0]):
                n_real = int(inputs["attention_mask"][j].sum().item())
                if n_real > 1:
                    mask[j, n_real - 1] = 0
            mask_f = mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)

        return self.screener(pooled)


# ── Loss functions ────────────────────────────────────────────────────

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020)."""

    def __init__(self, temperature: float = TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]

        features = F.normalize(features, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature

        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask.fill_diagonal_(0)

        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        n_positives = mask.sum(1)
        mean_log_prob = (mask * log_prob).sum(1) / n_positives.clamp(min=1)

        return -mean_log_prob.mean()


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


# ── Training ──────────────────────────────────────────────────────────

def train_screener(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    val_embeddings: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
    input_dim: int | None = None,
    projection_dim: int = PROJECTION_DIM,
    temperature: float = TEMPERATURE,
    hard_negative_ratio: int = HARD_NEGATIVE_RATIO,
    lr: float = CONTRASTIVE_LR,
    epochs: int = CONTRASTIVE_EPOCHS,
    batch_size: int = CONTRASTIVE_BATCH_SIZE,
    use_hard_negatives: bool = True,
    multi_scale: bool = False,
    use_mixup: bool = False,
    device: str = "cuda",
    save_path: Path | None = None,
) -> tuple[FunctionAwareScreener, dict]:
    """
    Train the function-aware screener on pre-computed embeddings.
    """
    torch.manual_seed(SEED)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    if input_dim is None:
        input_dim = train_embeddings.shape[1]

    # Build datasets
    if use_hard_negatives:
        train_dataset = HardNegativeDataset(
            train_embeddings, train_labels,
            n_hard_negatives=hard_negative_ratio,
        )
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
    model = FunctionAwareScreener(input_dim, projection_dim, multi_scale).to(device)
    criterion = CombinedLoss(temperature=temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    mixup = MixupAugmenter() if use_mixup else None

    history = {"train_loss": [], "train_con": [], "train_bce": [],
               "val_loss": [], "val_auroc": []}

    best_val_auroc = -1.0
    if save_path is None:
        save_path = CHECKPOINTS_DIR / "best_screener.pt"

    for epoch in range(epochs):
        model.train()
        epoch_losses = {"loss_total": [], "loss_contrastive": [], "loss_bce": []}

        for embeddings_batch, labels_batch in train_loader:
            embeddings_batch = embeddings_batch.to(device)
            labels_batch = labels_batch.to(device)

            # Apply Mixup augmentation if enabled
            if mixup is not None:
                embeddings_batch, labels_batch = mixup(embeddings_batch, labels_batch)

            projections, logits = model(embeddings_batch)
            loss, loss_dict = criterion(projections, logits, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k, v in loss_dict.items():
                epoch_losses[k].append(v)

        scheduler.step()

        avg_train = {k: np.mean(v) for k, v in epoch_losses.items()}
        history["train_loss"].append(avg_train["loss_total"])
        history["train_con"].append(avg_train["loss_contrastive"])
        history["train_bce"].append(avg_train["loss_bce"])

        # Validate
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
                    val_preds.extend(torch.sigmoid(logit).cpu().numpy())
                    val_true.extend(lab_b.cpu().numpy())

            from sklearn.metrics import roc_auc_score
            val_preds = np.array(val_preds)
            val_true = np.array(val_true)

            if len(np.unique(val_true)) > 1:
                val_auroc = roc_auc_score(val_true, val_preds)
            else:
                logger.warning("Validation set has only one class — AUROC undefined")
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

    if save_path and save_path.exists():
        model.load_state_dict(torch.load(save_path, weights_only=True))
        logger.info(f"Loaded best model (val AUROC: {best_val_auroc:.4f})")

    return model, history
