from __future__ import annotations

"""
Protein structure acquisition for ProteinMPNN-based adversarial generation.

Two sources:
  1. AlphaFold DB — pre-computed structures for most Swiss-Prot proteins
  2. ESMFold — on-the-fly structure prediction from sequence (GPU required)

Both return PDB files with backbone coordinates (N, CA, C, O) needed
for ProteinMPNN inverse folding.
"""
import logging
import time
from pathlib import Path

import numpy as np
import requests

from src.config import RAW_DIR

logger = logging.getLogger(__name__)

ALPHAFOLD_DB_API = "https://alphafold.ebi.ac.uk/api/prediction"
STRUCTURES_DIR = RAW_DIR / "structures"


def fetch_alphafold_structure(
    uniprot_id: str,
    output_dir: Path | None = None,
) -> Path:
    """
    Download a predicted structure from AlphaFold DB.

    Args:
        uniprot_id: UniProt accession (e.g., "P0A9D4")
        output_dir: where to save PDB files

    Returns:
        Path to the downloaded PDB file

    Raises:
        RuntimeError: if the structure is not available in AlphaFold DB
    """
    if output_dir is None:
        output_dir = STRUCTURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for any cached version first
    cached = list(output_dir.glob(f"AF-{uniprot_id}-F1-model_v*.pdb"))
    if cached:
        return cached[0]

    # Query the AlphaFold API to get the correct PDB URL
    # (the model version changes over time — v4, v5, v6, etc.)
    api_url = f"{ALPHAFOLD_DB_API}/{uniprot_id}"

    for attempt in range(3):
        try:
            resp = requests.get(api_url, timeout=30)
            if resp.status_code == 200:
                entries = resp.json()
                if not entries:
                    raise RuntimeError(
                        f"AlphaFold API returned empty response for {uniprot_id}"
                    )
                pdb_url = entries[0].get("pdbUrl")
                if not pdb_url:
                    raise RuntimeError(
                        f"No PDB URL in AlphaFold API response for {uniprot_id}"
                    )

                # Download the PDB file
                pdb_resp = requests.get(pdb_url, timeout=30)
                pdb_resp.raise_for_status()

                # Extract filename from URL
                pdb_filename = pdb_url.split("/")[-1]
                pdb_path = output_dir / pdb_filename
                pdb_path.write_text(pdb_resp.text)
                logger.debug(f"Downloaded AlphaFold structure for {uniprot_id}")
                return pdb_path

            elif resp.status_code == 404:
                raise RuntimeError(
                    f"No AlphaFold structure available for {uniprot_id}"
                )
            else:
                logger.warning(f"AlphaFold API returned {resp.status_code} for {uniprot_id}")
        except requests.RequestException as e:
            if attempt == 2:
                raise RuntimeError(f"Failed to fetch AlphaFold structure for {uniprot_id}: {e}")
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Failed to fetch AlphaFold structure for {uniprot_id}")


def predict_structure_esmfold(
    sequence: str,
    uniprot_id: str,
    output_dir: Path | None = None,
    device: str = "cuda",
) -> Path:
    """
    Predict protein structure using ESMFold (GPU required).

    ESMFold is a single-sequence structure predictor that runs in a single
    forward pass. It's less accurate than AlphaFold2 but sufficient for
    generating backbone coordinates for ProteinMPNN.

    Args:
        sequence: amino acid sequence
        uniprot_id: identifier for saving
        output_dir: where to save PDB files
        device: "cuda" or "cpu"

    Returns:
        Path to the predicted PDB file
    """
    if output_dir is None:
        output_dir = STRUCTURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_path = output_dir / f"ESMFold-{uniprot_id}.pdb"
    if pdb_path.exists():
        return pdb_path

    import torch
    from transformers import EsmForProteinFolding, AutoTokenizer

    logger.info(f"Predicting structure for {uniprot_id} ({len(sequence)} AA) with ESMFold...")

    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    model = model.to(device)
    model.eval()

    # ESMFold has a practical length limit ~1024 residues
    if len(sequence) > 1024:
        logger.warning(f"{uniprot_id}: sequence length {len(sequence)} > 1024, truncating")
        sequence = sequence[:1024]

    with torch.no_grad():
        inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

    # Convert to PDB
    pdb_string = _esmfold_output_to_pdb(outputs, sequence)
    pdb_path.write_text(pdb_string)
    logger.info(f"Saved ESMFold structure to {pdb_path}")

    return pdb_path


def _esmfold_output_to_pdb(outputs, sequence: str) -> str:
    """Convert ESMFold output to PDB format string."""
    # ESMFold outputs positions as (batch, n_residues, n_atoms, 3)
    # Atom order: N, CA, C, O (first 4)
    positions = outputs.positions[-1, 0].cpu().numpy()  # Last layer, first batch
    atom_names = ["N", "CA", "C", "O"]

    lines = []
    lines.append("REMARK   Generated by ESMFold")
    atom_idx = 1

    for res_idx in range(len(sequence)):
        res_name = _one_to_three(sequence[res_idx])
        for atom_i, atom_name in enumerate(atom_names):
            if res_idx < positions.shape[0] and atom_i < positions.shape[1]:
                x, y, z = positions[res_idx, atom_i]
                lines.append(
                    f"ATOM  {atom_idx:5d}  {atom_name:<3s} {res_name:>3s} A"
                    f"{res_idx + 1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    f"  1.00  0.00           {atom_name[0]:>2s}"
                )
                atom_idx += 1

    lines.append("END")
    return "\n".join(lines)


def parse_pdb_backbone(pdb_path: Path) -> dict:
    """
    Parse a PDB file and extract backbone atom coordinates.

    Returns:
        dict with:
            - 'N': (n_residues, 3) array of N coordinates
            - 'CA': (n_residues, 3) array of CA coordinates
            - 'C': (n_residues, 3) array of C coordinates
            - 'O': (n_residues, 3) array of O coordinates
            - 'sequence': extracted amino acid sequence
            - 'chain_id': chain identifier
    """
    coords = {'N': [], 'CA': [], 'C': [], 'O': []}
    sequence = []
    current_res_idx = None
    chain_id = None

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            atom_name = line[12:16].strip()
            if atom_name not in coords:
                continue

            res_idx = int(line[22:26].strip())
            res_name = line[17:20].strip()

            if chain_id is None:
                chain_id = line[21]

            # Only process first chain
            if line[21] != chain_id:
                continue

            if res_idx != current_res_idx:
                current_res_idx = res_idx
                sequence.append(_three_to_one(res_name))

            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords[atom_name].append([x, y, z])

    # Validate: all atom types should have the same count
    lengths = {k: len(v) for k, v in coords.items()}
    if len(set(lengths.values())) > 1:
        # Some residues may be missing atoms — align by taking the minimum
        min_len = min(lengths.values())
        logger.warning(
            f"Unequal backbone atom counts in {pdb_path.name}: {lengths}. "
            f"Truncating to {min_len} residues."
        )
        for k in coords:
            coords[k] = coords[k][:min_len]
        sequence = sequence[:min_len]

    return {
        'N': np.array(coords['N']),
        'CA': np.array(coords['CA']),
        'C': np.array(coords['C']),
        'O': np.array(coords['O']),
        'sequence': ''.join(sequence),
        'chain_id': chain_id or 'A',
    }


def fetch_structures_batch(
    accessions: list[str],
    sequences: list[str] | None = None,
    source: str = "alphafold",
    device: str = "cuda",
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """
    Fetch structures for a batch of proteins.

    Args:
        accessions: list of UniProt accession IDs
        sequences: list of sequences (required if source="esmfold")
        source: "alphafold" or "esmfold"
        device: GPU device for ESMFold
        output_dir: where to save PDB files

    Returns:
        dict mapping accession -> PDB file path (only successful ones)
    """
    results = {}
    failed = []

    for i, acc in enumerate(accessions):
        try:
            if source == "alphafold":
                pdb_path = fetch_alphafold_structure(acc, output_dir)
            elif source == "esmfold":
                if sequences is None:
                    raise ValueError("sequences must be provided for ESMFold")
                pdb_path = predict_structure_esmfold(sequences[i], acc, output_dir, device)
            else:
                raise ValueError(f"Unknown structure source: {source}")

            results[acc] = pdb_path

        except RuntimeError as e:
            logger.warning(f"Skipping {acc}: {e}")
            failed.append(acc)

        if (i + 1) % 10 == 0:
            logger.info(f"  Structures: {i+1}/{len(accessions)} "
                        f"({len(results)} success, {len(failed)} failed)")

    logger.info(f"Fetched {len(results)}/{len(accessions)} structures "
                f"({len(failed)} failed)")
    if failed:
        logger.info(f"Failed accessions: {failed[:20]}{'...' if len(failed) > 20 else ''}")

    return results


# ── Amino acid conversion tables ──────────────────────────────────────

_THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}
_ONE_TO_THREE = {v: k for k, v in _THREE_TO_ONE.items()}


def _three_to_one(three: str) -> str:
    return _THREE_TO_ONE.get(three.upper(), 'X')


def _one_to_three(one: str) -> str:
    return _ONE_TO_THREE.get(one.upper(), 'UNK')
