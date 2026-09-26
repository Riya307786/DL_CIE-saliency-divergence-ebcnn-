# Implementation Audit: Saliency-Divergence Branch Selection

**Project Title:** Saliency-Divergence Branch Selection: Replacing Accuracy-Heuristic Ensembling with an Explainability-Driven Criterion for Spatially-Scattered Defect Classification  
**Date of Audit:** September 26, 2026  
**Auditor:** Antigravity Autonomous Senior ML Research Engineer

---

## 1. System & Component Status Matrix

| Component | Current Status | Issues / Gaps Identified | Required Engineering Action | Priority | Validation Method |
| :--- | :---: | :--- | :--- | :---: | :--- |
| **Dataset Files** | Complete | Files were inside `archive (5)/` subdirectory; Int64Index unpickling incompatibility with pandas 2.2+. | Fixed by shimming `pandas.core.indexes.numeric` and copying files directly to `dataset/extracted/`. | HIGH | `scripts/audit_dataset.py` passes with zero errors. |
| **Dataset Report** | Complete | Missing before audit. | Generated comprehensive audit in `docs/dataset_report.md` with complete distribution and leakage checks. | HIGH | Verified 62,248 samples, 0 leakage, exact $3:1:1$ ratio. |
| **Preprocessing** | Complete | Was unspecified; risk of blur via continuous bilinear interpolation. | Implemented `src/data/preprocessing.py` (Padded Nearest Neighbor, 224x224, $\div 255.0$, 3-channel). | HIGH | Verified 100% ternary purity and 40x lower distortion in `docs/preprocessing_verification.md`. |
| **Data Loader** | Missing | No PyTorch Dataset/DataLoader exists to stream batches into deep models. | Implement `src/data/dataset.py` with memory-efficient batching and stratification. | HIGH | Unit tests checking batch shapes, dtypes, and labels. |
| **EB-CNN Model** | Missing | Need exact implementation of Abdullah et al. (2025) VGG-16 multi-branch architecture. | Implement `src/models/ebcnn.py` with 5 branches ($B_1..B_5$), stacking classifiers ($C_1..C_5$), and Grad-CAM layer hooks. | CRITICAL | Forward/backward pass unit test, feature map dimensions ($7 \times 7$ at B5). |
| **Training Pipeline** | Missing | Need reproducible training loop with deterministic seeds and multi-branch loss. | Implement `src/training/trainer.py` with multi-branch cross-entropy loss, checkpointing, and metrics tracking. | HIGH | Training loss decrease, checkpoint saved/loaded cleanly. |
| **Grad-CAM Engine** | Missing | Need gradient-weighted activation mapping on the final conv layer of each branch. | Implement `src/explainability/gradcam.py` computing $\alpha_k^c$ via backward gradients and ReLU combination. | HIGH | Saliency maps verified on real Donut and Random wafers. |
| **Saliency Divergence** | Missing | Need pairwise divergence formulations (IoU and Cosine). | Implement `src/branch_selection/divergence.py` calculating spatial divergence matrix $D(i, j) \in [0, 1]^{5 \times 5}$. | HIGH | Unit test checking symmetry, diagonal 0, range $[0, 1]$. |
| **Branch Selection** | Missing | Need Baseline (original EB-CNN heuristic) and Proposed (Predictive Quality + Saliency Divergence). | Implement `src/branch_selection/selector.py` with validation-only thresholding and equal/weighted ensembling. | CRITICAL | Subset selection logic tested on validation splits. |
| **HPO Pipeline** | Missing | Need controlled hyperparameter optimization adhering to AlMuhaideb & Khan (2026). | Implement `src/training/hpo.py` using Optuna, tuning LR, dropout, weight decay with fixed budget. | HIGH | Optuna study completes; saves `best_params.json` and retrains fresh instance. |
| **Multi-Seed Runner** | Missing | Need paired evaluation across multiple seeds (10 fixed seeds). | Implement `scripts/run_experiments.py` evaluating Baseline vs Proposed vs Ablations with shared seed pool. | HIGH | 10-seed experiment completes and outputs structured results JSON/CSV. |
| **Statistical Analysis**| Missing | Need Friedman test, Wilcoxon signed-rank with Holm correction, and rank-biserial effect sizes. | Implement `src/statistics/statistical_tests.py` computing seed-level paired statistical comparisons. | HIGH | Statistical report generated with exact p-values and effect sizes. |
| **Publication Figures** | Partial | Audit & Preprocessing figures complete; architecture, branch performance, and divergence figures needed. | Implement visualization scripts for Grad-CAM maps, divergence matrices, and baseline vs proposed comparisons. | MEDIUM | Generated PNG files in `artifacts/figures/`. |
| **FastAPI Backend** | Missing | Need real-time inference server serving real PyTorch models and Grad-CAM maps. | Implement `backend/main.py` with upload, sample test, and toggle endpoints. | HIGH | API endpoints return 200 OK with valid JSON payloads. |
| **React Frontend** | Missing | Need interactive web UI for demonstration. | Build React + Vite + Tailwind single-page app displaying real predictions, heatmaps, and toggle. | HIGH | Browser verification of UI upload and toggle flows. |

---

## 2. Immediate Execution Roadmap

1. Implement PyTorch data loader in `src/data/dataset.py`.
2. Implement exact EB-CNN architecture in `src/models/ebcnn.py` and write `docs/ebcnn_architecture.md`.
3. Implement unit tests in `tests/test_data_and_model.py` and run them.
4. Implement multi-branch training engine in `src/training/trainer.py`.
5. Implement Grad-CAM engine in `src/explainability/gradcam.py`.
6. Implement Saliency Divergence metrics in `src/branch_selection/divergence.py`.
7. Implement Baseline & Proposed Branch Selectors in `src/branch_selection/selector.py`.
8. Implement HPO pipeline in `src/training/hpo.py`.
9. Execute single-seed and multi-seed training runs.
10. Execute statistical analysis and generate all figures.
11. Build FastAPI backend and React frontend.
12. Run end-to-end validation.
