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

# ── Attention pooling ──────────────────────────────────────────────────
ATTENTION_HEADS = 1
ATTENTION_HIDDEN_DIM = 256

# ── Multi-scale classifier ────────────────────────────────────────────
MULTI_SCALE_HIDDEN = 128

# ── Mixup augmentation ────────────────────────────────────────────────
MIXUP_ALPHA = 0.2  # Beta distribution parameter
MIXUP_PROB = 0.5  # probability of applying mixup per batch

# ── Bootstrap and statistical testing ─────────────────────────────────
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_CI_ALPHA = 0.05  # 95% CI

# ── Ablation configs ──────────────────────────────────────────────────
ABLATION_PROJECTION_DIMS = [128, 256, 512, 1024]
ABLATION_HARD_NEGATIVE_RATIOS = [1, 3, 5]
ABLATION_TEMPERATURES = [0.05, 0.07, 0.1, 0.2]
ABLATION_FINETUNE_LAYERS = [0, 2, 4]

# ── Certification upgrade ─────────────────────────────────────────────
CERTIFICATION_N_SAMPLES = 1000
CERTIFICATION_MAX_SEQUENCES = 100

# ── OOD evaluation ────────────────────────────────────────────────────
OOD_PROTEIN_FAMILIES = {
    "kinases": '(keyword:KW-0418) AND (reviewed:true) AND (organism_id:9606)',
    "gpcrs": '(keyword:KW-0297) AND (reviewed:true) AND (organism_id:9606)',
    "transcription_factors": '(keyword:KW-0805) AND (reviewed:true) AND (organism_id:9606)',
}
OOD_MAX_PER_FAMILY = 100

# ── Second threat family: Ribosome-Inactivating Proteins (RIPs) ──────
RIP_THREAT_QUERIES = [
    {
        "query": '("ribosome-inactivating") AND (keyword:KW-0800) AND (reviewed:true)',
        "label": "threat",
        "subcategory": "rip_toxin",
        "description": "Ribosome-inactivating proteins",
    },
    {
        "query": '(gene:ricin OR gene:abrin) AND (keyword:KW-0800) AND (reviewed:true)',
        "label": "threat",
        "subcategory": "ricin_family",
        "description": "Ricin/abrin family toxins",
    },
]
RIP_BENIGN_QUERIES = [
    {
        "query": '(keyword:KW-0326) AND (NOT keyword:KW-0800) AND (reviewed:true)',
        "label": "benign",
        "subcategory": "glycosidase_nontoxin",
        "description": "Glycosidases that are not toxins",
    },
]

# ── Random seeds ───────────────────────────────────────────────────────
SEED = 42
