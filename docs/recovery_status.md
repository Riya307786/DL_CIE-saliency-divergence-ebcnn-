# Project Recovery & Execution Status

**Project Title:** Saliency-Divergence Branch Selection: Replacing Accuracy-Heuristic Ensembling with an Explainability-Driven Criterion for Spatially-Scattered Defect Classification  
**Date of Recovery:** September 26, 2026  
**Recovery Agent:** Antigravity Autonomous Research & Development Assistant  

---

## 1. Recovery Audit Matrix

| Pipeline Stage | Status | Verification Evidence / Artifacts | Action Taken / Next Step |
| :--- | :---: | :--- | :--- |
| **1. Dataset Integrity & Audit** | VERIFIED | `dataset/extracted/` (train/val/test splits), `docs/dataset_report.md` | Complete. 62,248 samples, zero leakage, exact 3:1:1 split. |
| **2. Preprocessing & Purity** | VERIFIED | `src/data/preprocessing.py`, `docs/preprocessing_verification.md` | Complete. Padded Nearest Neighbor 224x224, 100% ternary purity. |
| **3. Data Loader Pipeline** | VERIFIED | `src/data/dataset.py`, `tests/test_data_and_model.py` | Complete. `WM811KDataset` & dataloaders pass all tests. |
| **4. EB-CNN Model Architecture** | VERIFIED | `src/models/ebcnn.py`, `docs/ebcnn_architecture.md` | Complete. Abdullah et al. (2025) 5-branch architecture & Grad-CAM layer hooks. |
| **5. Multi-Branch Trainer** | VERIFIED | `src/training/trainer.py` | Complete. Joint multi-branch loss, checkpointing, and evaluation metrics. |
| **6. Grad-CAM Saliency Engine** | VERIFIED | `src/explainability/gradcam.py` | Complete. Backward-gradient feature map weighted activations across B1..B5. |
| **7. Saliency Divergence Metrics** | VERIFIED | `src/branch_selection/divergence.py` | Complete. Pairwise Cosine & IoU divergence matrix calculations ($D \in [0, 1]^{5 \times 5}$). |
| **8. Baseline & Proposed Selectors** | VERIFIED | `src/branch_selection/selector.py` | Complete. `BaselineEBCNNSelector` & `SaliencyDivergenceBranchSelector` implemented. |
| **9. Controlled HPO Pipeline** | INCOMPLETE | `src/training/hpo.py` | Execute Optuna HPO to freeze optimal hyperparameters in `artifacts/hpo/best_params.json`. |
| **10. 10-Seed Experiment Pipeline** | INCOMPLETE | `scripts/run_experiments.py` | Execute full 10-seed paired evaluation across baseline, proposed, and ablations. |
| **11. Statistical & Figures Report** | INCOMPLETE | `src/statistics/statistical_tests.py` | Generate `results/statistical_report.json` and publication figures in `artifacts/figures/`. |
| **12. FastAPI Backend Service** | MISSING | `backend/main.py` | Implement REST API for real model inference, Grad-CAM maps, and selection. |
| **13. React Frontend Interactive Web UI** | MISSING | `frontend/` | Implement React + Vite frontend for end-to-end interactive MVP. |

---

## 2. Last Verified Stage
- **Stage 8: Branch Selection Logic & Unit Tests**. Unit test suite (`pytest tests/`) executed cleanly with 6/6 tests passing in 8.10s.

---

## 3. Autonomous Continuation Plan
1. **Execute HPO & 10-Seed Research Pipeline:** Launch `scripts/run_experiments.py` to optimize hyperparameters, complete all 10-seed paired experiments, compute statistical tests (Friedman, Wilcoxon, McNemar), save results to `results/`, and render publication-quality figures to `artifacts/figures/`.
2. **Document Findings:** Compile full experimental findings into `docs/experimental_results.md`.
3. **Build FastAPI Backend:** Implement `backend/main.py` with endpoints (`/api/predict`, `/api/sample`, `/api/health`, `/api/stats`) loading real model weights and returning real Grad-CAM heatmaps & selection outputs.
4. **Build React Frontend:** Initialize and build a React SPA in `frontend/` displaying interactive wafer inference, Grad-CAM visualization, divergence metrics matrix, and baseline vs proposed toggle.
5. **End-to-End Verification:** Verify tests, backend API, and frontend build end-to-end.
