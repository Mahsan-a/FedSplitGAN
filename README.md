# FedSplitGAN
Privacy-Preserving Synthetic CGM Generation via Federated Split GANs

Federated Split Generative Adversarial Networks for synthetic medical time-series generation across distributed clinical sites.


This project implements a **Federated Split GAN (FedSplitGAN)** framework for generating synthetic continuous glucose monitoring (CGM) scalograms from distributed Type 1 diabetes (T1D) cohorts. The architecture separates the GAN model across participating medical entities: a **single central generator** produces synthetic spectral-temporal representations of glucose dynamics, while **multiple distributed discriminators** (each hosted at an independent clinical site) evaluate sample quality against their local patient population without ever transmitting raw data.

A **peer-to-peer ring communication** protocol between discriminators propagates compressed gradient statistics across sites before they are aggregated at the central node, reducing communication rounds by up to ~40% compared to a star-topology federated baseline while improving generator convergence.

The synthetic outputs are dual-channel CWT scalograms of shape `(96 × 288 × 2)` > the same spectral-temporal representation used in the [MAHP hypoglycemia prediction framework](https://github.com/Mahsan-a/MAHP) making them directly compatible with downstream classifiers trained on real CGM data.

---

## Clinical Motivation

Federated learning is particularly well-suited to clinical CGM data for two reasons. First, patient glucose records are governed by HIPAA and equivalent international frameworks, meaning centralized data pooling is often legally or institutionally infeasible. Second, the rare-event patient subgroup identified in real-world T1D cohorts contributes very few hypoglycemic examples per site — synthetic augmentation of this class, generated from the joint distribution learned across sites, directly addresses the class imbalance problem without data sharing.

The datasets supported by this framework include:

| Dataset | N patients | Ages | Monitoring duration |
|---|---|---|---|
| T1DEXI | 491 | 18–70 | 28 days |
| T1DEXIP | 227 | 12–17 | 28 days |
| CL3 | 168 | ≤14 | 6–8 months |
| CL5 | 100 | 6–13 | 16–20 weeks |
| CITY | 149 | 14–25 | 26 weeks |
| PEDAP | 98 | 2–6 | 26–32 weeks |
| AIDET1D | 82 | ≥65 | 54 weeks |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CENTRAL NODE                             │
│                                                             │
│   z ~ N(0,I)  →  [ Generator G_θ ]  →  X_syn (96×288×2)   │
│                         ↑                                   │
│                  Aggregated ∇G                              │
└──────────────────────┬──────────────────────────────────────┘
                       │  broadcast X_syn
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   ┌────────────┐ ┌────────────┐ ┌────────────┐
   │  Client 1  │ │  Client 2  │ │  Client K  │
   │ D_1(X_syn, │ │ D_2(X_syn, │ │ D_K(X_syn, │
   │  X_real_1) │ │  X_real_2) │ │  X_real_K) │
   └──────┬─────┘ └──────┬─────┘ └──────┬─────┘
          │  ∇D_1         │  ∇D_2         │  ∇D_K
          └───────────────┴───────────────┘
               P2P ring gradient communication
               (compressed via top-k sparsification)
```

**Key properties:**
- Raw patient data never leaves its originating site
- Generator learns from the joint distribution across all sites simultaneously
- P2P ring topology reduces required communication rounds vs. star topology
- Gradient compression (top-k sparsification + quantization) reduces per-round bandwidth
- Differential privacy noise (Gaussian mechanism, ε-δ DP) applied to discriminator gradients before transmission

---

## Installation

```bash
git clone https://github.com/yourusername/fed-split-gan-cgm.git
cd fed-split-gan-cgm
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, PyTorch ≥ 2.0, pywavelets, numpy, pandas, scikit-learn, matplotlib

---

## Quickstart

```bash
# Prepare CGM data (computes CWT scalograms from raw glucose CSVs)
python src/data_utils.py --data_dir /path/to/cgm_csvs --out_dir data/scalograms

# Run federated training
python train.py --config configs/default.yaml

# Evaluate synthetic sample quality (FID, MMD, clinical glucose statistics)
python evaluate.py --checkpoint checkpoints/gen_final.pt --real_dir data/scalograms
```

---

## Project Structure

```
fed_split_gan/
├── src/
│   ├── generator.py          # Central generator network
│   ├── discriminator.py      # Site-local discriminator network
│   ├── fed_split_gan.py      # Federated training orchestrator
│   ├── communication.py      # P2P ring gradient communication
│   └── data_utils.py         # CGM loading, CWT computation, dataset classes
├── configs/
│   └── default.yaml          # Hyperparameters
├── notebooks/
│   └── visualize_samples.ipynb
├── train.py                  # Training entry point
├── evaluate.py               # FID / MMD / clinical metrics
└── requirements.txt
```

---

## Evaluation Metrics

Beyond standard image generation metrics, synthetic CGM scalograms are evaluated against clinical glucose quality criteria:

| Metric | Description |
|---|---|
| FID | Fréchet Inception Distance on scalogram feature embeddings |
| MMD | Maximum Mean Discrepancy between real and synthetic distributions |
| TIR | Time-in-range (70–180 mg/dL) preservation |
| Hypo rate | Synthetic hypoglycemia prevalence vs. real cohort |
| Wavelet energy | Spectral energy distribution across scales |

