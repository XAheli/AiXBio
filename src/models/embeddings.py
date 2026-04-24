from __future__ import annotations

"""
Embedding extraction using ESM-2 and ProTrek.

Produces per-protein embeddings from frozen backbones, saved to disk
for downstream contrastive learning and screening.
"""
import logging
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

from src.config import (
    ESM2_MODEL_NAME, ESM2_EMBEDDING_DIM, ESM2_MAX_LENGTH,
    PROTREK_MODEL_NAME, PROTREK_EMBEDDING_DIM,
    EMBEDDINGS_DIR,
)

logger = logging.getLogger(__name__)


class ESM2Embedder:
    """Extract per-protein embeddings from ESM-2 (mean-pooled last hidden state)."""

    def __init__(self, model_name: str = ESM2_MODEL_NAME, device: str = "cuda"):
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading ESM-2 model {model_name} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        logger.info("ESM-2 loaded successfully")

    @torch.no_grad()
    def embed_single(self, sequence: str) -> np.ndarray:
        """Embed a single protein sequence. Returns shape (embedding_dim,)."""
        # Truncate if needed
        seq = sequence[:ESM2_MAX_LENGTH]
        inputs = self.tokenizer(seq, return_tensors="pt", padding=False, truncation=True,
                                max_length=ESM2_MAX_LENGTH + 2)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        # Mean pool over sequence length (excluding CLS/EOS tokens)
        hidden = outputs.last_hidden_state[0, 1:-1, :]  # (seq_len, dim)
        embedding = hidden.mean(dim=0)  # (dim,)

        return embedding.cpu().numpy()

    @torch.no_grad()
    def embed_batch(self, sequences: list[str], batch_size: int = 8) -> np.ndarray:
        """Embed a batch of sequences. Returns shape (n_sequences, embedding_dim)."""
        all_embeddings = []

        for i in tqdm(range(0, len(sequences), batch_size), desc="ESM-2 embedding"):
            batch_seqs = [s[:ESM2_MAX_LENGTH] for s in sequences[i:i + batch_size]]
            inputs = self.tokenizer(
                batch_seqs, return_tensors="pt", padding=True,
                truncation=True, max_length=ESM2_MAX_LENGTH + 2,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state  # (batch, seq_len, dim)
            attention_mask = inputs["attention_mask"]  # (batch, seq_len)

            # Mask out padding + special tokens for mean pooling
            # ESM-2 wraps as: <cls> [AA tokens] <eos> [pad...]
            # We exclude CLS (position 0) and EOS (last real token)
            mask = attention_mask.clone()
            mask[:, 0] = 0  # CLS token
            for j in range(mask.shape[0]):
                # Find EOS position from ORIGINAL attention_mask (before modification)
                n_real_tokens = int(attention_mask[j].sum().item())
                eos_pos = n_real_tokens - 1
                if eos_pos > 0:
                    mask[j, eos_pos] = 0  # EOS token

            mask_expanded = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            summed = (hidden * mask_expanded).sum(dim=1)  # (batch, dim)
            counts = mask_expanded.sum(dim=1).clamp(min=1)  # (batch, 1)
            embeddings = summed / counts  # (batch, dim)

            all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)


class ProTrekEmbedder:
    """
    Extract per-protein embeddings from ProTrek's sequence encoder.

    ProTrek is a tri-modal model (sequence + structure + function text).
    We use the sequence encoder branch, which is an ESM-2-based backbone
    with a projection head trained via contrastive learning across modalities.

    ProTrek must be installed separately:
        pip install git+https://github.com/westlake-repl/ProTrek.git

    The model uses its own loading API (not HuggingFace AutoModel).
    """

    def __init__(self, model_name: str = PROTREK_MODEL_NAME, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading ProTrek model {model_name} on {self.device}...")

        try:
            from protrek.model import ProTrekTrimodalModel
            self.model = ProTrekTrimodalModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            logger.info("ProTrek loaded successfully")
        except ImportError:
            raise ImportError(
                "ProTrek is not installed. Install it with:\n"
                "  pip install git+https://github.com/westlake-repl/ProTrek.git\n"
                "If you cannot install ProTrek, use --model esm2 instead."
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load ProTrek model '{model_name}': {e}\n"
                "Check that the model name is correct. Known models:\n"
                "  westlake-repl/ProTrek_650M\n"
                "  westlake-repl/ProTrek_35M_UniRef50"
            )

    @torch.no_grad()
    def embed_batch(self, sequences: list[str], batch_size: int = 8) -> np.ndarray:
        all_embeddings = []
        for i in tqdm(range(0, len(sequences), batch_size), desc="ProTrek embedding"):
            batch_seqs = sequences[i:i + batch_size]
            # ProTrek's API: get_protein_repr returns normalized embeddings
            embeddings = self.model.get_protein_repr(batch_seqs)
            all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)


def extract_and_save_embeddings(
    df: pd.DataFrame,
    model_type: str = "esm2",
    device: str = "cuda",
    batch_size: int = 8,
    output_dir: Path | None = None,
    split_name: str = "all",
):
    """
    Extract embeddings for all sequences in a DataFrame and save to disk.

    Args:
        df: DataFrame with 'accession' and 'sequence' columns
        model_type: "esm2" or "protrek"
        device: "cuda" or "cpu"
        batch_size: batch size for inference
        output_dir: where to save embeddings
        split_name: name prefix for saved files
    """
    if output_dir is None:
        output_dir = EMBEDDINGS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = df["sequence"].tolist()
    accessions = df["accession"].tolist()

    if model_type == "esm2":
        embedder = ESM2Embedder(device=device)
    elif model_type == "protrek":
        embedder = ProTrekEmbedder(device=device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info(f"Extracting {model_type} embeddings for {len(sequences)} sequences...")
    embeddings = embedder.embed_batch(sequences, batch_size=batch_size)

    # Save embeddings
    out_path = output_dir / f"{split_name}_{model_type}_embeddings.npz"
    np.savez_compressed(
        out_path,
        embeddings=embeddings,
        accessions=np.array(accessions),
    )
    logger.info(f"Saved embeddings to {out_path}: shape {embeddings.shape}")

    return embeddings


def load_embeddings(
    model_type: str = "esm2",
    split_name: str = "all",
    embeddings_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load pre-computed embeddings. Returns (embeddings, accessions)."""
    if embeddings_dir is None:
        embeddings_dir = EMBEDDINGS_DIR

    path = embeddings_dir / f"{split_name}_{model_type}_embeddings.npz"
    data = np.load(path)
    return data["embeddings"], data["accessions"]
