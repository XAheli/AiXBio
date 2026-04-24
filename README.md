# FuncScreen: Contrastive PLM Embeddings for Evasion-Resistant Biosecurity Screening

Biosecurity screening for DNA synthesis orders currently relies on sequence homology -- checking whether a query sequence *looks like* a known threat. AI protein design tools like ProteinMPNN can generate sequences that fold into the same dangerous structure but share less than 30% sequence identity with any known threat, completely evading homology-based screening.

This project introduces **function-aware screening**: a contrastive learning framework over protein language model (ESM-2) embeddings that detects threat proteins by *predicted biological function* rather than sequence similarity. We demonstrate that function-aware screening maintains detection capability in regimes where all homology-based methods fail.

## Key Results

| Evaluation Split | K-mer (Homology) | ESM-2 Cosine NN | ESM-2 Linear | **Contrastive (Ours)** |
|---|---|---|---|---|
| Standard | 0.990 | 0.993 | 0.995 | **1.000** |
| Hard Negative | 0.989 | 0.994 | 0.995 | **1.000** |
| Sequence Divergent | 0.821 | 0.886 | 0.943 | **0.974** |
| MPNN Adversarial | 0.952 | 0.965 | 0.994 | **0.991** |

On ProteinMPNN-designed variants with <20% sequence identity, homology screening achieves **0% detection** while our contrastive model maintains **~65% detection**.

## Method

1. **Data curation**: 985 proteins from UniProt Swiss-Prot (335 pore-forming toxins + 650 benign homologs including MACPF domain proteins, perforin, complement components)
2. **Embedding extraction**: Frozen ESM-2 (650M) mean-pooled representations (1280-dim)
3. **Contrastive learning**: Supervised contrastive loss + BCE with hard-negative mining over threat/benign protein pairs
4. **Adversarial stress testing**: 4,100 ProteinMPNN-designed variants at 5 sampling temperatures (T=0.1 to T=1.0), producing variants with 7-60% sequence identity to source threats
5. **Certified robustness**: Randomized smoothing adapted to biological mutation spaces (conservative and random amino acid substitutions)

## Repository Structure

```
src/
  config.py                     Central configuration
  data/
    curate.py                   UniProt data fetching and quality filtering
    splits.py                   Train/val/test splitting with 4 evaluation slices
    adversarial.py              BLOSUM62-conservative adversarial variant generation
    structures.py               AlphaFold DB structure download and PDB parsing
    proteinmpnn.py              ProteinMPNN inverse folding wrapper
  models/
    embeddings.py               ESM-2 and ProTrek embedding extraction
    contrastive.py              Supervised contrastive learning with hard-negative mining
  screening/
    baselines.py                K-mer, Cosine NN, Linear, KNN baseline screeners
    certified.py                Randomized smoothing for biological mutation spaces
  evaluation/
    metrics.py                  AUROC, AUPRC, recall@precision, detection-vs-divergence
    experiments.py              Full experiment runner
    visualize.py                Publication-quality figures

scripts/
  01_curate_data.py             Fetch and split data from UniProt
  02_extract_embeddings.py      Extract ESM-2 embeddings (GPU)
  03_train_contrastive.py       Train contrastive screener (GPU)
  04_evaluate.py                Evaluate all methods, generate figures
  05_certify.py                 Certified robustness analysis (GPU)
  06_generate_mpnn_variants.py  ProteinMPNN adversarial generation (GPU)

data/                           Curated datasets, embeddings, splits
results/                        Checkpoints, figures, tables
```

## Reproducing Results

### Requirements

```bash
git clone https://github.com/XAheli/AiXBio.git
cd aixbio
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Pipeline

```bash
# Step 1: Curate data from UniProt
python scripts/01_curate_data.py

# Step 2: Extract ESM-2 embeddings 
python scripts/02_extract_embeddings.py --model esm2 --device cuda --batch-size 32

# Step 3: Train contrastive screener
python scripts/03_train_contrastive.py --embedding esm2 --device cuda --epochs 50

# Step 4: Evaluate + generate figures 
python scripts/04_evaluate.py --embedding esm2

# Step 5: Certified robustness 
python scripts/05_certify.py --embedding esm2 --device cuda --n-samples 200

# Step 6: ProteinMPNN adversarial variants 
git clone https://github.com/dauparas/ProteinMPNN.git
export PYTHONPATH=$PYTHONPATH:$(pwd)/ProteinMPNN
python scripts/06_generate_mpnn_variants.py --device cuda --structure-source alphafold

# Step 7: Re-evaluate with MPNN split
python scripts/04_evaluate.py --embedding esm2
```

## Threat Family

This prototype focuses on **pore-forming toxins (PFTs)** -- a well-annotated threat family with rich benign homologs (MACPF domain proteins, perforin, complement components). The framework generalizes to other threat families by modifying the UniProt queries in `src/config.py`.

## Certified Robustness

We adapt randomized smoothing to biologically structured perturbation models:
- **Conservative substitutions**: mutations within Dayhoff amino acid groups (function-preserving)
- **Random substitutions**: unrestricted amino acid replacements

The empirical-certified gap is **<=2%** across all tested mutation budgets (k=1 to k=10), indicating that individual screening decisions are provably stable under biological mutations.

## Citation

If you use this work, please cite:

```
@inproceedings{poddar2026funcscreen,
  title={FuncScreen: Contrastive PLM Embeddings for Evasion-Resistant Biosecurity Screening},
  author={Poddar, Aheli},
  booktitle={AIxBio Hackathon},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE).

## Limitations and Dual-Use Considerations

- Scoped to one threat family (pore-forming toxins) -- generalization to other families requires validation
- Detection rate degrades at extreme sequence divergence (<20% identity) -- not a replacement for multi-layered screening
- ProteinMPNN variant generation demonstrates a known attack vector -- we disclose this to motivate defensive improvements
- This work is intended to strengthen biosecurity screening infrastructure, not to enable evasion
