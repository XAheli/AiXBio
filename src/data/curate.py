from __future__ import annotations

"""
Data curation pipeline for function-aware biosecurity screening.

Fetches threat sequences (pore-forming toxins) and hard negatives (benign
homologs) from UniProt, applies quality filters, and saves curated datasets.
"""
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict

import requests
import pandas as pd
from Bio import SeqIO
from io import StringIO

from src.config import (
    RAW_DIR, PROCESSED_DIR,
    TOXIN_PFAM_IDS, TOXIN_KEYWORDS, BENIGN_HOMOLOG_KEYWORDS,
)

logger = logging.getLogger(__name__)

UNIPROT_API = "https://rest.uniprot.org/uniprotkb"

# ── Curated threat categories ──────────────────────────────────────────
# Each entry: (query, label, subcategory, description)
# We use reviewed (Swiss-Prot) entries for high-confidence labels.

# UniProt keyword IDs (from https://www.uniprot.org/keywords/)
# KW-0800 = Toxin, KW-0472 = Membrane, KW-0812 = Transmembrane
# Note: "Pore-forming toxin" is NOT a UniProt keyword. Use free-text + Toxin keyword.

THREAT_QUERIES = [
    {
        # Pore-forming toxins: use "pore-forming" as free text + KW-0800 (Toxin)
        "query": '("pore-forming toxin") AND (keyword:KW-0800) AND (reviewed:true)',
        "label": "threat",
        "subcategory": "pore_forming_toxin",
        "description": "Reviewed pore-forming toxins (free text + Toxin keyword)",
    },
    {
        # Aerolysin family via Pfam domain PF01338
        "query": '(xref:pfam-PF01338) AND (keyword:KW-0800) AND (reviewed:true)',
        "label": "threat",
        "subcategory": "aerolysin_family",
        "description": "Aerolysin/ETX domain toxins (Pfam PF01338)",
    },
    {
        # Cholesterol-dependent cytolysins via Pfam PF01289
        "query": '(xref:pfam-PF01289) AND (keyword:KW-0800) AND (reviewed:true)',
        "label": "threat",
        "subcategory": "cytolysin_family",
        "description": "Cholesterol-dependent cytolysins (Pfam PF01289)",
    },
    {
        # Hemolytic toxins via keyword KW-0354 (Hemolysis)
        "query": '(keyword:KW-0354) AND (keyword:KW-0800) AND (reviewed:true)',
        "label": "threat",
        "subcategory": "hemolysin",
        "description": "Hemolytic toxins (Hemolysis + Toxin keywords)",
    },
]

BENIGN_QUERIES = [
    {
        # MACPF domain proteins that are NOT toxins (complement, perforin)
        "query": '(xref:pfam-PF01823) AND (NOT keyword:KW-0800) AND (reviewed:true)',
        "label": "benign",
        "subcategory": "macpf_nontoxin",
        "description": "MACPF domain non-toxin proteins (complement, perforin)",
    },
    {
        # Pore-forming non-toxic proteins (e.g., gasdermin, membrane channels)
        "query": '("pore-forming") AND (NOT keyword:KW-0800) AND (reviewed:true)',
        "label": "benign",
        "subcategory": "pore_nontoxin",
        "description": "Pore-forming but non-toxic proteins",
    },
    {
        # Complement C9 — structurally similar to PFTs
        "query": '(gene:C9) AND (keyword:KW-0180) AND (reviewed:true)',
        "label": "benign",
        "subcategory": "complement",
        "description": "Complement component C9 (structurally similar to PFTs)",
    },
    {
        # Antimicrobial peptides — pore-forming but beneficial
        "query": '(keyword:KW-0929) AND ("pore") AND (NOT keyword:KW-0800) AND (reviewed:true)',
        "label": "benign",
        "subcategory": "antimicrobial_pore",
        "description": "Antimicrobial pore-forming peptides",
    },
    {
        # Perforin family — immune defense
        "query": '(gene:PRF1 OR gene:MPEG1) AND (reviewed:true)',
        "label": "benign",
        "subcategory": "perforin",
        "description": "Perforin family - immune defense, structurally similar to PFTs",
    },
]

# Random benign proteins for easy negatives
RANDOM_BENIGN_QUERY = {
    # KW-0378 = Hydrolase (a common enzyme keyword)
    "query": '(keyword:KW-0378) AND (NOT keyword:KW-0800) AND (organism_id:9606) AND (reviewed:true)',
    "label": "benign",
    "subcategory": "random_enzyme",
    "description": "Random human hydrolases as easy negatives",
}


@dataclass
class ProteinEntry:
    accession: str
    name: str
    sequence: str
    length: int
    organism: str
    label: str  # "threat" or "benign"
    subcategory: str
    pfam_ids: list[str] = field(default_factory=list)
    go_terms: list[str] = field(default_factory=list)
    function_description: str = ""
    sequence_identity_to_nearest_threat: float | None = None


def fetch_uniprot(query: str, max_results: int = 500, fields: str | None = None) -> list[dict]:
    """Fetch entries from UniProt REST API with pagination."""
    if fields is None:
        fields = "accession,id,protein_name,sequence,length,organism_name,xref_pfam,go,cc_function"

    all_results = []
    url = f"{UNIPROT_API}/search"
    params = {
        "query": query,
        "format": "json",
        "size": min(max_results, 500),
        "fields": fields,
    }

    while url and len(all_results) < max_results:
        for attempt in range(3):
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == 2:
                    logger.error(f"Failed to fetch from UniProt after 3 attempts: {e}")
                    return all_results
                time.sleep(2 ** attempt)

        data = resp.json()
        results = data.get("results", [])
        all_results.extend(results)

        # Handle pagination
        link_header = resp.headers.get("Link", "")
        if 'rel="next"' in link_header:
            url = link_header.split(";")[0].strip("<>")
            params = None  # URL already contains params
        else:
            break

        time.sleep(0.5)  # Rate limiting

    return all_results[:max_results]


def parse_uniprot_entry(entry: dict, label: str, subcategory: str) -> ProteinEntry | None:
    """Parse a UniProt JSON entry into a ProteinEntry."""
    try:
        accession = entry["primaryAccession"]
        name = entry.get("uniProtkbId", accession)

        # Extract sequence
        seq_data = entry.get("sequence", {})
        sequence = seq_data.get("value", "")
        length = seq_data.get("length", len(sequence))

        if not sequence or length < 50:
            return None

        # Organism
        organism = entry.get("organism", {}).get("scientificName", "Unknown")

        # Pfam IDs
        pfam_ids = []
        for xref in entry.get("uniProtKBCrossReferences", []):
            if xref.get("database") == "Pfam":
                pfam_ids.append(xref.get("id", ""))

        # GO terms
        go_terms = []
        for xref in entry.get("uniProtKBCrossReferences", []):
            if xref.get("database") == "GO":
                go_id = xref.get("id", "")
                props = xref.get("properties", [])
                term = next((p["value"] for p in props if p.get("key") == "GoTerm"), "")
                go_terms.append(f"{go_id}:{term}")

        # Function description
        func_desc = ""
        for comment in entry.get("comments", []):
            if comment.get("commentType") == "FUNCTION":
                texts = comment.get("texts", [])
                if texts:
                    func_desc = texts[0].get("value", "")

        return ProteinEntry(
            accession=accession,
            name=name,
            sequence=sequence,
            length=length,
            organism=organism,
            label=label,
            subcategory=subcategory,
            pfam_ids=pfam_ids,
            go_terms=go_terms,
            function_description=func_desc,
        )
    except (KeyError, IndexError) as e:
        logger.warning(f"Failed to parse entry: {e}")
        return None


def curate_dataset(
    max_per_query: int = 300,
    max_random_benign: int = 200,
    min_length: int = 50,
    max_length: int = 2000,
) -> pd.DataFrame:
    """
    Curate the full dataset: threat sequences + hard negatives + easy negatives.

    Returns a DataFrame with columns matching ProteinEntry fields.
    """
    all_entries: list[ProteinEntry] = []
    seen_accessions: set[str] = set()

    # Fetch threat sequences
    logger.info("=== Fetching threat sequences ===")
    for qinfo in THREAT_QUERIES:
        logger.info(f"  Query: {qinfo['description']}")
        raw = fetch_uniprot(qinfo["query"], max_results=max_per_query)
        logger.info(f"  Got {len(raw)} raw results")
        for entry in raw:
            parsed = parse_uniprot_entry(entry, qinfo["label"], qinfo["subcategory"])
            if parsed and parsed.accession not in seen_accessions:
                if min_length <= parsed.length <= max_length:
                    all_entries.append(parsed)
                    seen_accessions.add(parsed.accession)
        logger.info(f"  Running total: {len(all_entries)} entries")

    n_threat = sum(1 for e in all_entries if e.label == "threat")
    logger.info(f"Total threat sequences: {n_threat}")

    # Fetch hard negatives (benign homologs)
    logger.info("\n=== Fetching hard negatives (benign homologs) ===")
    for qinfo in BENIGN_QUERIES:
        logger.info(f"  Query: {qinfo['description']}")
        raw = fetch_uniprot(qinfo["query"], max_results=max_per_query)
        logger.info(f"  Got {len(raw)} raw results")
        for entry in raw:
            parsed = parse_uniprot_entry(entry, qinfo["label"], qinfo["subcategory"])
            if parsed and parsed.accession not in seen_accessions:
                if min_length <= parsed.length <= max_length:
                    all_entries.append(parsed)
                    seen_accessions.add(parsed.accession)
        logger.info(f"  Running total: {len(all_entries)} entries")

    # Fetch random easy negatives
    logger.info("\n=== Fetching easy negatives (random enzymes) ===")
    raw = fetch_uniprot(RANDOM_BENIGN_QUERY["query"], max_results=max_random_benign)
    logger.info(f"  Got {len(raw)} raw results")
    for entry in raw:
        parsed = parse_uniprot_entry(
            entry, RANDOM_BENIGN_QUERY["label"], RANDOM_BENIGN_QUERY["subcategory"]
        )
        if parsed and parsed.accession not in seen_accessions:
            if min_length <= parsed.length <= max_length:
                all_entries.append(parsed)
                seen_accessions.add(parsed.accession)

    n_benign = sum(1 for e in all_entries if e.label == "benign")
    logger.info(f"Total benign sequences: {n_benign}")
    logger.info(f"Total dataset size: {len(all_entries)}")

    # Convert to DataFrame
    df = pd.DataFrame([asdict(e) for e in all_entries])

    # Add metadata
    df["is_hard_negative"] = df["subcategory"].isin([
        "macpf_nontoxin", "pore_nontoxin", "complement", "perforin",
        "antimicrobial_pore",
    ])

    return df


def save_curated_dataset(df: pd.DataFrame, output_dir: Path | None = None):
    """Save curated dataset in multiple formats."""
    if output_dir is None:
        output_dir = PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as parquet (efficient)
    df.to_parquet(output_dir / "curated_dataset.parquet", index=False)

    # Save as CSV (human-readable)
    df_no_seq = df.drop(columns=["sequence"])
    df_no_seq.to_csv(output_dir / "curated_metadata.csv", index=False)

    # Save sequences as FASTA
    fasta_path = output_dir / "curated_sequences.fasta"
    with open(fasta_path, "w") as f:
        for _, row in df.iterrows():
            header = f">{row['accession']}|{row['label']}|{row['subcategory']}"
            f.write(f"{header}\n{row['sequence']}\n")

    # Save summary stats
    stats = {
        "total_sequences": len(df),
        "threat_sequences": int((df["label"] == "threat").sum()),
        "benign_sequences": int((df["label"] == "benign").sum()),
        "hard_negatives": int(df["is_hard_negative"].sum()),
        "subcategory_counts": df["subcategory"].value_counts().to_dict(),
        "length_stats": {
            "mean": float(df["length"].mean()),
            "median": float(df["length"].median()),
            "min": int(df["length"].min()),
            "max": int(df["length"].max()),
        },
    }
    with open(output_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Saved curated dataset to {output_dir}")
    logger.info(f"  {stats['total_sequences']} sequences")
    logger.info(f"  {stats['threat_sequences']} threat, {stats['benign_sequences']} benign")
    logger.info(f"  {stats['hard_negatives']} hard negatives")

    return stats
