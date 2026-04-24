from __future__ import annotations

"""
ProteinMPNN-based adversarial variant generation.

Uses ProteinMPNN inverse folding to generate protein sequences that
are predicted to fold into the same 3D structure as known threat proteins
but have divergent amino acid sequences — exactly the kind of evasion
an adversary could produce.

This is the strongest possible stress test for DNA screening:
functional equivalence with arbitrary sequence divergence.

ProteinMPNN reference:
  Dauparas et al. "Robust deep learning–based protein sequence design
  using ProteinMPNN" Science 378, 49–56 (2022)

Installation:
  pip install git+https://github.com/dauparas/ProteinMPNN.git
  OR clone and add to PYTHONPATH
"""
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RAW_DIR, SPLITS_DIR, SEED
from src.data.structures import parse_pdb_backbone

logger = logging.getLogger(__name__)

PROTEINMPNN_WEIGHTS_DIR = RAW_DIR / "proteinmpnn_weights"


def check_proteinmpnn_available() -> bool:
    """Check if ProteinMPNN is importable."""
    try:
        # ProteinMPNN can be imported different ways depending on installation
        import protein_mpnn_utils
        return True
    except ImportError:
        pass

    try:
        # Some installations use a package name
        from proteinmpnn import protein_mpnn_utils
        return True
    except ImportError:
        pass

    return False


def design_sequences_proteinmpnn(
    pdb_path: Path,
    num_sequences: int = 10,
    sampling_temps: list[float] | None = None,
    device: str = "cuda",
    model_type: str = "v_48_020",
) -> list[dict]:
    """
    Run ProteinMPNN inverse folding on a single structure.

    Generates diverse sequences predicted to fold into the given backbone.
    Higher sampling temperatures produce more diverse (lower identity) sequences.

    Args:
        pdb_path: path to input PDB file
        num_sequences: sequences to generate per temperature
        sampling_temps: list of sampling temperatures
            - 0.1: very conservative, ~90%+ identity
            - 0.3: moderate diversity, ~60-80% identity
            - 0.5: high diversity, ~40-60% identity
            - 1.0: maximum diversity, ~20-40% identity
        device: "cuda" or "cpu"
        model_type: ProteinMPNN model variant

    Returns:
        list of dicts with: sequence, temperature, score, seq_identity
    """
    if sampling_temps is None:
        sampling_temps = [0.1, 0.3, 0.5, 0.8, 1.0]

    results = []

    try:
        results = _design_via_python_api(
            pdb_path, num_sequences, sampling_temps, device, model_type
        )
    except (ImportError, ModuleNotFoundError):
        logger.info("ProteinMPNN Python API not available, using CLI")
        backbone = parse_pdb_backbone(pdb_path)
        original_seq = backbone['sequence']
        if len(original_seq) == 0:
            raise RuntimeError(f"No residues parsed from {pdb_path}")
        results = _design_via_cli(
            pdb_path, original_seq, num_sequences, sampling_temps, device
        )

    return results


def _design_via_python_api(
    pdb_path: Path,
    num_sequences: int,
    sampling_temps: list[float],
    device: str,
    model_type: str,
) -> list[dict]:
    """
    Run ProteinMPNN via Python API using its native parse_PDB
    and tied_featurize functions.
    """
    import torch
    from protein_mpnn_utils import ProteinMPNN, tied_featurize, parse_PDB

    # Use ProteinMPNN's own PDB parser
    pdb_dict_list = parse_PDB(str(pdb_path))
    if not pdb_dict_list:
        raise RuntimeError(f"ProteinMPNN parse_PDB returned empty for {pdb_path}")

    original_seq = pdb_dict_list[0].get('seq', '')
    if not original_seq:
        raise RuntimeError(f"No sequence extracted from {pdb_path}")

    # Determine chains
    all_chain_list = [
        k.split('_')[-1] for k in pdb_dict_list[0]
        if k.startswith('seq_chain_')
    ]
    if not all_chain_list:
        raise RuntimeError(f"No chains found in {pdb_path}")

    # Build chain_id_dict: all chains are designable
    chain_id_dict = {pdb_dict_list[0]['name']: (all_chain_list, [])}

    # Load model
    model = _load_proteinmpnn_model(model_type, device)

    # Featurize using ProteinMPNN's own function
    batch = [pdb_dict_list[0]]
    (X, S, mask, lengths, chain_M, chain_encoding_all,
     chain_list_list, visible_list_list, masked_list_list,
     masked_chain_length_list_list, chain_M_pos, omit_AA_mask,
     residue_idx, dihedral_mask, tied_pos_list_of_lists_list,
     pssm_coef, pssm_bias, pssm_log_odds_all,
     bias_by_res_all, tied_beta) = tied_featurize(
        batch, device, chain_id_dict,
        fixed_position_dict=None,
        omit_AA_dict=None,
        tied_positions_dict=None,
        pssm_dict=None,
        bias_by_res_dict=None,
    )

    # Amino acid alphabet used by ProteinMPNN
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

    # Default arrays from ProteinMPNN's protein_mpnn_run.py:
    # omit_AAs_np: all zeros = allow all amino acids (no omissions)
    # bias_AAs_np: all zeros = no compositional bias
    # These cannot be None — model.sample() calls torch.from_numpy() on them.
    omit_AAs_np = np.zeros(len(alphabet), dtype=np.float32)
    bias_AAs_np = np.zeros(len(alphabet), dtype=np.float32)

    results = []
    for temp in sampling_temps:
        for seq_idx in range(num_sequences):
            with torch.no_grad():
                randn = torch.randn(chain_M.shape, device=device)
                sample_dict = model.sample(
                    X, randn, S, chain_M, chain_encoding_all,
                    residue_idx, mask=mask, temperature=temp,
                    omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np,
                    chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask,
                    pssm_coef=pssm_coef, pssm_bias=pssm_bias,
                    pssm_multi=0.0, pssm_log_odds_flag=False,
                    pssm_log_odds_mask=pssm_log_odds_all,
                    pssm_bias_flag=False, bias_by_res=bias_by_res_all,
                )

            S_sample = sample_dict["S"]
            all_probs = sample_dict["probs"]

            # Decode designed sequence
            indices = S_sample[0].cpu().numpy()
            designed_seq = ''.join(
                alphabet[i] if i < len(alphabet) else 'X' for i in indices
            )

            # Score: mean log-probability of designed residues
            # all_probs shape: (1, L, 21) — probability at each position
            designed_probs = all_probs[0].cpu().numpy()
            log_prob_sum = 0.0
            n_designed = 0
            for pos_idx, aa_idx in enumerate(indices):
                p = designed_probs[pos_idx, aa_idx] if aa_idx < 21 else 0.0
                if p > 0:
                    log_prob_sum += np.log(p)
                    n_designed += 1
            score = log_prob_sum / max(n_designed, 1)

            # Compute metrics
            seq_identity = _compute_sequence_identity(original_seq, designed_seq)

            results.append({
                'sequence': designed_seq,
                'temperature': temp,
                'score': score,
                'seq_identity': seq_identity,
                'original_sequence': original_seq,
                'length': len(designed_seq),
            })

    return results


def _design_via_cli(
    pdb_path: Path,
    original_seq: str,
    num_sequences: int,
    sampling_temps: list[float],
    device: str,
) -> list[dict]:
    """Run ProteinMPNN via command-line script."""
    import tempfile

    # Find the ProteinMPNN script
    mpnn_script = _find_proteinmpnn_script()

    results = []
    for temp in sampling_temps:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "output"
            out_dir.mkdir()

            cmd = [
                sys.executable, str(mpnn_script),
                "--pdb_path", str(pdb_path),
                "--out_folder", str(out_dir),
                "--num_seq_per_target", str(num_sequences),
                "--sampling_temp", str(temp),
                "--seed", str(SEED),
                "--batch_size", "1",
            ]

            if device == "cpu":
                cmd.append("--use_cpu")

            logger.debug(f"Running: {' '.join(cmd)}")
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )

            if proc.returncode != 0:
                raise RuntimeError(
                    f"ProteinMPNN CLI failed (exit {proc.returncode}):\n"
                    f"stderr: {proc.stderr[:1000]}"
                )

            # Parse output FASTA
            seqs_dir = out_dir / "seqs"
            if seqs_dir.exists():
                for fasta_file in seqs_dir.glob("*.fa"):
                    designed_seqs = _parse_mpnn_fasta(fasta_file)
                    for seq_info in designed_seqs:
                        seq_identity = _compute_sequence_identity(
                            original_seq, seq_info['sequence']
                        )
                        results.append({
                            'sequence': seq_info['sequence'],
                            'temperature': temp,
                            'score': seq_info.get('score', 0.0),
                            'seq_identity': seq_identity,
                            'original_sequence': original_seq,
                            'length': len(seq_info['sequence']),
                        })

    return results


def _find_proteinmpnn_script() -> Path:
    """Locate the ProteinMPNN inference script."""
    # Check common locations
    candidates = [
        Path("ProteinMPNN/protein_mpnn_run.py"),
        Path.home() / "ProteinMPNN" / "protein_mpnn_run.py",
        RAW_DIR / "ProteinMPNN" / "protein_mpnn_run.py",
    ]

    for path in candidates:
        if path.exists():
            return path

    # Try to find via Python path
    try:
        import protein_mpnn_run
        return Path(protein_mpnn_run.__file__)
    except ImportError:
        pass

    raise RuntimeError(
        "ProteinMPNN not found. Install it:\n"
        "  git clone https://github.com/dauparas/ProteinMPNN.git\n"
        "  OR: pip install git+https://github.com/dauparas/ProteinMPNN.git\n"
        "Then either add to PYTHONPATH or clone into the project directory."
    )


def _load_proteinmpnn_model(model_type: str, device: str):
    """Load ProteinMPNN model weights."""
    import torch

    try:
        from protein_mpnn_utils import ProteinMPNN
    except ImportError:
        from proteinmpnn.protein_mpnn_utils import ProteinMPNN

    # Standard model config for v_48_020
    model = ProteinMPNN(
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        vocab=21,
        k_neighbors=48,
        augment_eps=0.0,
    )

    # Try to load weights from standard locations
    weight_paths = [
        PROTEINMPNN_WEIGHTS_DIR / f"{model_type}.pt",
        Path("ProteinMPNN/vanilla_model_weights") / f"{model_type}.pt",
        Path.home() / "ProteinMPNN/vanilla_model_weights" / f"{model_type}.pt",
    ]

    for wpath in weight_paths:
        if wpath.exists():
            checkpoint = torch.load(wpath, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            logger.info(f"Loaded ProteinMPNN weights from {wpath}")
            return model

    raise RuntimeError(
        f"ProteinMPNN weights not found. Checked:\n"
        + "\n".join(f"  - {p}" for p in weight_paths)
        + "\nDownload weights from: https://github.com/dauparas/ProteinMPNN"
    )


def _decode_sequence(indices: np.ndarray) -> str:
    """Convert ProteinMPNN integer indices to amino acid string."""
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    return ''.join(alphabet[i] for i in indices if i < len(alphabet))


def _parse_mpnn_fasta(fasta_path: Path) -> list[dict]:
    """Parse ProteinMPNN output FASTA file."""
    sequences = []
    current_header = None
    current_seq = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header and current_seq:
                    score = _extract_score_from_header(current_header)
                    sequences.append({
                        'header': current_header,
                        'sequence': ''.join(current_seq),
                        'score': score,
                    })
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)

        if current_header and current_seq:
            score = _extract_score_from_header(current_header)
            sequences.append({
                'header': current_header,
                'sequence': ''.join(current_seq),
                'score': score,
            })

    # Skip the first entry (usually the original sequence)
    if len(sequences) > 1:
        sequences = sequences[1:]

    return sequences


def _extract_score_from_header(header: str) -> float:
    """Extract score from ProteinMPNN FASTA header."""
    # Headers look like: "T=0.1, sample=1, score=1.234, ..."
    for part in header.split(','):
        part = part.strip()
        if part.startswith('score='):
            try:
                return float(part.split('=')[1])
            except (ValueError, IndexError):
                pass
    return 0.0


def _compute_sequence_identity(seq1: str, seq2: str) -> float:
    """Compute fraction of identical residues between two sequences."""
    min_len = min(len(seq1), len(seq2))
    if min_len == 0:
        return 0.0
    matches = sum(1 for a, b in zip(seq1[:min_len], seq2[:min_len]) if a == b)
    return matches / max(len(seq1), len(seq2))


def generate_mpnn_adversarial_split(
    threat_df: pd.DataFrame,
    structure_paths: dict[str, Path],
    num_sequences_per_temp: int = 5,
    sampling_temps: list[float] | None = None,
    device: str = "cuda",
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Generate adversarial variants using ProteinMPNN for all threat proteins
    with available structures.

    This is the strongest adversarial evaluation: sequences that are
    structurally (and thus functionally) equivalent but sequence-divergent.

    Args:
        threat_df: DataFrame of threat proteins (must have 'accession', 'sequence' cols)
        structure_paths: dict mapping accession -> PDB file path
        num_sequences_per_temp: designs per temperature per protein
        sampling_temps: ProteinMPNN sampling temperatures
        device: GPU device
        output_dir: where to save the split

    Returns:
        DataFrame of adversarial variants ready for evaluation
    """
    if sampling_temps is None:
        sampling_temps = [0.1, 0.3, 0.5, 0.8, 1.0]
    if output_dir is None:
        output_dir = SPLITS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    all_variants = []
    n_proteins = 0
    n_failed = 0

    for _, row in threat_df.iterrows():
        acc = row['accession']
        if acc not in structure_paths:
            continue

        n_proteins += 1
        pdb_path = structure_paths[acc]

        try:
            designs = design_sequences_proteinmpnn(
                pdb_path=pdb_path,
                num_sequences=num_sequences_per_temp,
                sampling_temps=sampling_temps,
                device=device,
            )

            for design in designs:
                all_variants.append({
                    'accession': f"{acc}_mpnn_t{design['temperature']:.1f}_{len(all_variants)}",
                    'name': f"{row.get('name', acc)}_mpnn",
                    'sequence': design['sequence'],
                    'length': design['length'],
                    'organism': 'designed',
                    'label': 'threat',
                    'subcategory': 'adversarial_mpnn',
                    'is_hard_negative': False,
                    'source_accession': acc,
                    'mutation_rate': 1.0 - design['seq_identity'],
                    'mutation_type': f"mpnn_t{design['temperature']:.1f}",
                    'seq_identity_to_source': design['seq_identity'],
                    'mpnn_score': design['score'],
                    'mpnn_temperature': design['temperature'],
                    'pfam_ids': row.get('pfam_ids', []),
                    'go_terms': row.get('go_terms', []),
                    'function_description': row.get('function_description', ''),
                })

            logger.info(
                f"  {acc}: {len(designs)} variants, "
                f"identity range: [{min(d['seq_identity'] for d in designs):.2f}, "
                f"{max(d['seq_identity'] for d in designs):.2f}]"
            )

        except Exception as e:
            logger.warning(f"  {acc}: ProteinMPNN failed — {e}")
            n_failed += 1

    if not all_variants:
        logger.error("No MPNN variants generated. Check ProteinMPNN installation.")
        return pd.DataFrame()

    mpnn_df = pd.DataFrame(all_variants)

    # Add benign sequences for balanced evaluation
    # Use the training benign sequences
    benign_df = threat_df[threat_df['label'] == 'benign'] if 'label' in threat_df.columns else pd.DataFrame()
    if len(benign_df) == 0:
        # Load from training split if threat_df is threats-only
        try:
            train_df = pd.read_parquet(SPLITS_DIR / "train.parquet")
            benign_df = train_df[train_df['label'] == 'benign'].sample(
                n=min(len(mpnn_df), len(train_df[train_df['label'] == 'benign'])),
                random_state=SEED,
            )
        except FileNotFoundError:
            logger.warning("No training split found for benign sequences")

    if len(benign_df) > 0:
        # Add missing columns
        for col in mpnn_df.columns:
            if col not in benign_df.columns:
                benign_df[col] = None
        # Keep only columns that exist in mpnn_df
        benign_df = benign_df[[c for c in mpnn_df.columns if c in benign_df.columns]]
        mpnn_df = pd.concat([mpnn_df, benign_df], ignore_index=True)

    mpnn_df = mpnn_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Save
    mpnn_df.to_parquet(output_dir / "test_adversarial_mpnn.parquet", index=False)

    n_threat = int((mpnn_df['label'] == 'threat').sum())
    n_benign = int((mpnn_df['label'] == 'benign').sum())
    logger.info(f"\nMPNN adversarial split: {len(mpnn_df)} sequences "
                f"({n_threat} threat variants from {n_proteins} proteins, "
                f"{n_benign} benign, {n_failed} proteins failed)")

    # Log identity distribution
    threat_variants = mpnn_df[mpnn_df['subcategory'] == 'adversarial_mpnn']
    if 'seq_identity_to_source' in threat_variants.columns:
        identities = threat_variants['seq_identity_to_source'].dropna()
        if len(identities) > 0:
            logger.info(f"  Sequence identity distribution:")
            logger.info(f"    Mean: {identities.mean():.3f}")
            logger.info(f"    Median: {identities.median():.3f}")
            logger.info(f"    Min: {identities.min():.3f}")
            logger.info(f"    Max: {identities.max():.3f}")
            for threshold in [0.3, 0.5, 0.7]:
                n_below = (identities < threshold).sum()
                logger.info(f"    Below {threshold:.0%} identity: {n_below} "
                            f"({n_below/len(identities):.0%})")

    return mpnn_df
