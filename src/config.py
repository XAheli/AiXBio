from __future__ import annotations

"""
Central configuration for the Function-Aware Biosecurity Screening project.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"

# ── Threat family: Pore-Forming Toxins (PFTs) ─────────────────────────
# Why PFTs:
#   - Well-annotated in UniProt (PF01338/Aerolysin, PF07968/CDC family, etc.)
#   - Clear regulatory relevance (many are Select Agent associated)
#   - Rich set of benign homologs (aerolysin-like lectins, complement proteins)
#   - Enough structural/functional diversity for hard negatives
#   - Sequence identity within family can drop to <20% while function is preserved
THREAT_FAMILY = "pore_forming_toxins"

# UniProt query components for data curation
TOXIN_PFAM_IDS = [
    "PF01338",  # Aerolysin/ETX pore-forming domain
    "PF07968",  # Cholesterol-dependent cytolysin (CDC)
    "PF01375",  # Epsilon toxin (ETX) / Mosquitocidal toxin (MTX)
]

# Keywords for threat sequences
TOXIN_KEYWORDS = [
    "pore-forming toxin",
    "cytolysin",
    "hemolysin",
    "aerolysin",
    "perfringolysin",
    "listeriolysin",
    "streptolysin",
    "alpha-toxin",
    "epsilon toxin",
]

# Keywords for hard negatives (benign homologs)
BENIGN_HOMOLOG_KEYWORDS = [
    "aerolysin-like lectin",
    "membrane attack complex",
    "complement component",
    "perforin",
    "MACPF domain",
]

# ── Model configs ──────────────────────────────────────────────────────
ESM2_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"  # 650M params
ESM2_EMBEDDING_DIM = 1280
ESM2_MAX_LENGTH = 1024  # tokens

PROTREK_MODEL_NAME = "westlake-repl/ProTrek_650M"
PROTREK_EMBEDDING_DIM = 1024

# ── Training configs ───────────────────────────────────────────────────
CONTRASTIVE_LR = 1e-3
CONTRASTIVE_EPOCHS = 50
CONTRASTIVE_BATCH_SIZE = 64
CONTRASTIVE_MARGIN = 0.5
PROJECTION_DIM = 256  # dimension of learned projection head
TEMPERATURE = 0.07  # for supervised contrastive loss
HARD_NEGATIVE_RATIO = 3  # hard negatives per positive

# ── Evaluation configs ─────────────────────────────────────────────────
EVAL_SPLITS = ["standard", "hard_negative", "seq_divergent", "adversarial"]
SEQUENCE_IDENTITY_BINS = [0.9, 0.7, 0.5, 0.3, 0.2]  # for detection-vs-divergence curves

# ── Random seeds ───────────────────────────────────────────────────────
SEED = 42
