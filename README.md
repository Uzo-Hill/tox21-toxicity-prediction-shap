# tox21-toxicity-prediction-shap
Multi-task toxicity prediction on the Tox21 dataset using Random Forest and XGBoost, with SHAP-based interpretability mapping predictions to real chemical substructures via RDKit.

---

# Machine Learning-Based Prediction and Structural Interpretation of Tox21 Toxicity

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![RDKit](https://img.shields.io/badge/RDKit-2026.03.4-brightgreen.svg)](https://www.rdkit.org/)
[![DeepChem](https://img.shields.io/badge/DeepChem-2.8.0-orange.svg)](https://deepchem.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Predicting chemical toxicity directly from molecular structure using machine learning, benchmarked across all 12 Tox21 endpoints, with SHAP-based interpretability mapping model predictions back to real chemical substructures.

## Overview

This project combines a background in Analytical Chemistry with machine learning to address a core problem in computational toxicology: predicting whether a chemical compound is likely to be toxic, directly from its molecular structure, before physical lab testing.

Two tree-based ensemble models (Random Forest and XGBoost) were trained and benchmarked across all 12 toxicity endpoints in the **Tox21** dataset. For four mechanistically distinct endpoints, **SHAP (SHapley Additive exPlanations)** was used to identify which molecular substructures most influenced each prediction, and **RDKit** was used to map those substructures back to real, interpretable chemical structures.

## Motivation

Traditional toxicity testing is slow, expensive, and often relies on animal testing. Computational toxicology enables rapid, low-cost pre-screening of large chemical libraries, but a model that predicts toxicity without explaining *why* offers limited value to a chemist. This project pairs predictive benchmarking with genuine structural interpretability.

## Dataset

- **Source:** [Tox21](https://tripod.nih.gov/tox21/challenge/) via [DeepChem's MoleculeNet](https://deepchem.io/)
- **Compounds:** ~7,800
- **Endpoints:** 12 (7 nuclear receptor assays, 5 stress response assays)
- **Featurization:** 1024-bit Extended Connectivity Fingerprints (ECFP, radius 2)
- **Split:** Scaffold-based (stricter, more realistic than random split)

## Methodology

1. **Data loading & featurization** — Tox21 loaded via DeepChem, scaffold split applied
2. **Multi-task benchmarking** — Random Forest and XGBoost trained independently per task (12 tasks × 2 models)
3. **Interpretability analysis** — SHAP TreeExplainer applied to 4 selected tasks (NR-AhR, NR-AR, SR-p53, NR-ER-LBD)
4. **Substructure mapping** — RDKit used to trace top SHAP-important fingerprint bits back to real molecular substructures

## 📊 Results

### 12-Task Benchmark (Random Forest vs. XGBoost)

| Metric | Random Forest | XGBoost |
|---|---|---|
| Mean ROC-AUC (12 tasks) | 0.703 | 0.714 |
| Best task | NR-AR-LBD (0.848) | NR-AR-LBD (0.807) |
| Weakest task | NR-ER (0.603) | NR-PPAR-gamma (0.605) |

Full per-task results: [`results/tox21_12task_benchmark.csv`](https://github.com/Uzo-Hill/tox21-toxicity-prediction-shap/blob/main/tox21_12task_benchmark.csv)

### Deep-Dive Interpretability (4 Tasks)

| Task | Biological Category | RF ROC-AUC | Dominant Substructure Theme |
|---|---|---|---|
| NR-AhR | Nuclear Receptor | 0.796 | Nitrogen-rich groups (amines, sulfonamide, lactone) |
| SR-p53 | Stress Response | 0.752 | Aromatic/electrophilic motifs (nitrile-aromatic, O-heterocycles) |
| NR-AR | Nuclear Receptor | 0.734 | N-heterocycles and carbonyls (imidazole, hydantoin) |
| NR-ER-LBD | Nuclear Receptor | 0.626 | Mixed/noisy (includes spurious organometallic outlier) |

**Key finding:** The top 5 SHAP-identified fingerprint bits were entirely non-overlapping across all four tasks — model predictions were driven by mechanistically distinct chemistry per receptor. Notably, explanation coherence tracked model performance: the weakest-performing task also produced the least chemically plausible SHAP output.

## Figures


| NR-AhR | NR-AR |
|---|---|
| ![NR-AhR](https://github.com/Uzo-Hill/tox21-toxicity-prediction-shap/blob/main/all_substructures_grid_NRAhR.png) | ![NR-AR](https://github.com/Uzo-Hill/tox21-toxicity-prediction-shap/blob/main/all_substructures_grid_NRAR.png) |

| SR-p53 | NR-ER-LBD |
|---|---|
| ![SR-p53](https://github.com/Uzo-Hill/tox21-toxicity-prediction-shap/blob/main/all_substructures_grid_SRp53.png) | ![NR-ER-LBD](https://github.com/Uzo-Hill/tox21-toxicity-prediction-shap/blob/main/all_substructures_grid_NRERLBD.png) |

## Repository Structure

```
tox21-toxicity-prediction-shap/
├── README.md
├── LICENSE
├── requirements.txt
├── notebooks/
│   └── tox21_shap_analysis.ipynb
├── figures/
├── results/
└── paper/
    └── tox21_paper_draft.docx
```

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/Uzo-Hill/tox21-toxicity-prediction-shap.git
cd tox21-toxicity-prediction-shap

# Create and activate a conda environment
conda create -n tox21_env python=3.10 -y
conda activate tox21_env

# Install RDKit
pip install rdkit

# Install remaining dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter lab
```

## Requirements

```
deepchem==2.8.0
torch
scikit-learn
xgboost
shap
pandas
numpy
matplotlib
seaborn
jupyter
ipykernel
```

## Limitations

- SHAP sample size limited to n=50 per task for computational tractability
- Single scaffold-based train/test split (no repeated cross-validation)
- XGBoost SHAP values not computed due to a library version incompatibility (XGBoost 3.x vs. installed SHAP release)

See the full paper for a complete discussion of limitations and future work.

## Paper

The full research paper draft is available at [`paper/tox21_paper_draft.docx`](paper/tox21_paper_draft.docx).

## Citation

If you use this work, please cite:

```
Uzoh, C. H. (2026). Machine Learning-Based Prediction and Structural 
Interpretation of Nuclear Receptor and Stress Response Toxicity Using 
the Tox21 Dataset. [Preprint/Repository].
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Chukwudum Hillary Uzoh**

Data Scientist & AI Engineer | Analytical & Industrial Chemistry background

[LinkedIn](https://www.linkedin.com/in/hillaryuzoh/) · [Twitter/X](https://x.com/UzohHillary) · [GitHub](https://github.com/Uzo-Hill)

## 🙏 Acknowledgments

Built using [RDKit](https://www.rdkit.org/), [DeepChem](https://deepchem.io/), [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.ai/), and [SHAP](https://shap.readthedocs.io/).
