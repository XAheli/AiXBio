# FuncScreen: Contrastive PLM Embeddings for Evasion-Resistant Biosecurity Screening

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![ESM-2](https://img.shields.io/badge/backbone-ESM--2%20650M-green.svg)](https://github.com/facebookresearch/esm)
[![ProteinMPNN](https://img.shields.io/badge/adversarial-ProteinMPNN-orange.svg)](https://github.com/dauparas/ProteinMPNN)
[![Hackathon](https://img.shields.io/badge/AIxBio-Hackathon%202026-purple.svg)](https://apartresearch.com)

> [!IMPORTANT]
> **We came top 25% 👀**
> 
> **Project Submission**: View the official project page and hackathon submission on **[Apart Research](https://apartresearch.com/project/funcscreen-contrastive-plm-embeddings-for-evasionresistant-biosecurity-screening-j842)**

> **TL;DR**: Current DNA screening checks if a sequence *looks like* a known threat. AI tools like ProteinMPNN design sequences that *function* identically but look nothing alike — evading all homology-based screening. FuncScreen detects threats by *function*, not *sequence*, closing this gap.

---

## The Problem

<p align="center">
  <img src="results/figures/mpnn_detection_vs_identity.png" width="85%" alt="Detection rate vs sequence identity"/>
</p>

<p align="center"><em>At &lt;20% sequence identity, homology screening achieves <b>0% detection</b>. FuncScreen maintains detection signal in this critical regime.</em></p>

AI protein design tools (ProteinMPNN) generate functional threat variants with as low as **7% sequence identity** to known threats. Current screening infrastructure — SecureDNA, IBBIS Common Mechanism — relies on sequence homology and **completely misses these variants**.

## Key Results

All metrics reported with **95% bootstrap confidence intervals** (1,000 iterations).

| Evaluation Split | K-mer (Homology) | ESM-2 Cosine NN | ESM-2 Linear | KNN | **FuncScreen** |
|---|---|---|---|---|---|
| Standard | .990 [.973, 1.0] | .993 [.977, 1.0] | .995 [.988, .999] | 1.00 [.999, 1.0] | **1.00** [1.0, 1.0] |
| Hard Negative | .989 [.969, 1.0] | .994 [.980, 1.0] | .995 [.990, .999] | .995 [.988, 1.0] | **1.00** [1.0, 1.0] |
| Seq. Divergent | .821 [.744, .897] | .886 [.826, .940] | .943 [.905, .973] | .966 [.934, .992] | **.974** [.947, .994] |
| MPNN Adversarial | .952 [.944, .959] | .965 [.958, .972] | .994 [.991, .996] | **.997** [.996, .998] | .991 [.988, .993] |

<details>
<summary><b>AUROC Heatmap (all methods x all splits)</b></summary>
<p align="center">
  <img src="results/figures/heatmap_auroc.png" width="80%" alt="AUROC Heatmap"/>
</p>
</details>

<details>
<summary><b>ROC Curves — ProteinMPNN Adversarial Split</b></summary>
<p align="center">
  <img src="results/figures/roc_test_adversarial_mpnn.png" width="70%" alt="ROC MPNN"/>
</p>
</details>

## Ablation Study

| Ablation | Best Config | MPNN AUROC | Seq. Div. AUROC |
|---|---|---|---|
| Projection dim | 512 | **.997** | .983 |
| Temperature | 0.2 | **.997** | **.987** |
| Hard neg. ratio | k=3 (default) | .991 | .974 |
| Multi-scale | No improvement | .989 | .974 |
| Mixup | No improvement | .986 | .976 |

Temperature 0.2 and projection dim 512 close the gap with KNN on MPNN adversarial while maintaining strong sequence-divergent performance.

## Generalization

<details>
<summary><b>Leave-One-Subcategory-Out Cross-Validation</b></summary>

| Held-out Subcategory | n | AUROC | 95% CI |
|---|---|---|---|
| Hemolysin | 169 | 0.963 | [0.935, 0.986] |
| Cytolysin family | 15 | 1.000 | [1.000, 1.000] |
| Pore-forming toxin | 17 | 0.615 | [0.452, 0.775] |
| Aerolysin family | 4 | 0.998 | [0.985, 1.000] |

Model generalizes to cytolysins and aerolysins without seeing them in training. Pore-forming toxin subcategory (0.615) is an honest limitation.
</details>

<details>
<summary><b>Second Threat Family: Ribosome-Inactivating Proteins</b></summary>

| Method | AUROC | 95% CI |
|---|---|---|
| K-mer | 0.992 | [0.973, 1.000] |
| FuncScreen | 0.962 | [0.879, 1.000] |

FuncScreen generalizes to a completely different threat family (ricin, abrin) with >0.96 AUROC.
</details>

<details>
<summary><b>Out-of-Distribution False Positive Rates</b></summary>

| Method | Kinases | GPCRs | Transcription Factors |
|---|---|---|---|
| K-mer | 0.000 | 0.000 | 0.000 |
| Cosine NN | **0.990** | **0.969** | **1.000** |
| KNN | 0.000 | 0.000 | 0.011 |
| FuncScreen | 0.041 | 0.031 | 0.242 |

Cosine NN is catastrophically unreliable on OOD proteins. FuncScreen has low FPR on kinases/GPCRs but elevated FPR on transcription factors — a calibration issue for future work.
</details>

## Method

<p align="center">
  <img src="results/figures/umap_raw_esm2.png" width="45%" alt="Raw ESM-2 Space"/>
  <img src="results/figures/umap_contrastive_esm2.png" width="45%" alt="Contrastive Projection"/>
</p>
<p align="center"><em>Left: Raw ESM-2 embeddings (overlapping clusters). Right: After contrastive learning (clean separation).</em></p>

1. **Data curation**: 985 proteins from UniProt Swiss-Prot (335 pore-forming toxins + 650 benign homologs)
2. **Embedding extraction**: Frozen ESM-2 (650M) mean-pooled representations (1280-dim)
3. **Contrastive learning**: Supervised contrastive loss + BCE with hard-negative mining
4. **Mixup augmentation**: Embedding-space interpolation for small-dataset regularization
5. **Adversarial training**: High-temperature ProteinMPNN variants (T=0.8, 1.0) added to training
6. **Adversarial stress testing**: 4,100 ProteinMPNN-designed variants at 5 sampling temperatures
7. **Certified robustness**: Randomized smoothing with 1,000 MC samples, empirical-certified gap ≤1%

<details>
<summary><b>ProteinMPNN Variant Diversity Distribution</b></summary>
<p align="center">
  <img src="results/figures/mpnn_identity_distribution.png" width="80%" alt="MPNN Identity Distribution"/>
</p>
</details>

<details>
<summary><b>Certified Robustness</b></summary>
<p align="center">
  <img src="results/figures/certified_robustness.png" width="85%" alt="Certified Robustness"/>
</p>
<p align="center"><em>Empirical vs certified accuracy under biological mutations. Gap ≤1%.</em></p>
</details>

<details>
<summary><b>Training Curves</b></summary>
<p align="center">
  <img src="results/figures/training_curves.png" width="80%" alt="Training Curves"/>
</p>
</details>

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
    embeddings.py               ESM-2 embedding and hidden state extraction
    contrastive.py              AttentionPooling, MixupAugmenter, contrastive learning
  screening/
    baselines.py                K-mer, Cosine NN, Linear, KNN baseline screeners
    certified.py                Randomized smoothing for biological mutation spaces
  evaluation/
    metrics.py                  Bootstrap CIs, paired significance tests, detection-vs-divergence
    experiments.py              Full experiment runner
    visualize.py                Publication-quality figures with CI error bars

scripts/
  01_curate_data.py             Fetch and split data from UniProt
  02_extract_embeddings.py      Extract ESM-2 embeddings (GPU)
  03_train_contrastive.py       Train screener (--multi-scale, --mixup, --adversarial-augment)
  04_evaluate.py                Evaluate with bootstrap CIs and paired tests
  05_certify.py                 Certified robustness (1,000 MC samples)
  06_generate_mpnn_variants.py  ProteinMPNN adversarial generation (GPU)
  07_ablation.py                Projection dim, temperature, hard neg ratio ablations
  08_loso_cv.py                 Leave-one-subcategory-out cross-validation
  09_ood_evaluation.py          Out-of-distribution false positive rates
  10_second_family.py           Ribosome-inactivating protein generalization

data/                           Curated datasets, embeddings, splits
results/                        Checkpoints, figures, tables, ablation results
```

## Reproducing Results

```bash
git clone https://github.com/XAheli/AiXBio.git aixbio
cd aixbio
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Core pipeline
python scripts/01_curate_data.py
python scripts/02_extract_embeddings.py --model esm2 --device cuda --batch-size 32
python scripts/03_train_contrastive.py --embedding esm2 --device cuda --epochs 50
python scripts/04_evaluate.py --embedding esm2

# ProteinMPNN adversarial variants
git clone https://github.com/dauparas/ProteinMPNN.git
export PYTHONPATH=$PYTHONPATH:$(pwd)/ProteinMPNN
python scripts/06_generate_mpnn_variants.py --device cuda --structure-source alphafold
python scripts/04_evaluate.py --embedding esm2

# Additional experiments
python scripts/05_certify.py --embedding esm2 --device cuda
python scripts/07_ablation.py --device cuda --embedding esm2
python scripts/08_loso_cv.py --embedding esm2 --device cuda
python scripts/09_ood_evaluation.py --device cuda --embedding esm2
python scripts/10_second_family.py --device cuda
```

## Citation

```bibtex
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

- KNN baseline outperforms FuncScreen on MPNN adversarial split in aggregate AUROC (0.997 vs 0.991); ablation shows this gap closes with dim=512 or τ=0.2
- Elevated OOD false positive rate on transcription factors (24%) — calibration needed
- LOSO CV: pore-forming toxin subcategory poorly detected when held out (0.615 AUROC)
- Primarily evaluated on one threat family; RIP mini-experiment provides preliminary generalization evidence
- Certified robustness uses 1,000 MC samples (production requires 100,000+)
- ProteinMPNN variant generation demonstrates a known attack vector — disclosed to motivate defensive improvements
- This work is intended to strengthen biosecurity screening infrastructure, not to enable evasion
