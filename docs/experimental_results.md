# Experimental Results & Methodology Report

**Project Title:** Saliency-Divergence Branch Selection: Replacing Accuracy-Heuristic Ensembling with an Explainability-Driven Criterion for Spatially-Scattered Defect Classification  
**Date:** September 26, 2026  
**Authors:** Antigravity Autonomous Machine Learning Research Team  

---

## 1. Research Overview & Motivation

In multi-branch deep architectures like the Ensemble of Branch Convolutional Neural Network (EB-CNN; Abdullah et al., 2025), shallower branches extract low-level spatial features while deeper branches capture fine-grained semantic representations. The original EB-CNN ensembling mechanism evaluates five static pre-defined branch combinations ($C_1 \dots C_5$) on validation accuracy alone.

However, on semiconductor wafer map defect benchmarks (WM-811K), spatially-scattered defect patterns such as **Donut** (annular die failures) and **Random** (disperoid global failures) often produce similar accuracy profiles across deep branches despite relying on vastly different spatial activation maps. Accuracy-heuristic selection fails to account for representation redundancy, frequently combining branches that focus on identical spatial regions.

This research introduces **Saliency-Divergence Branch Selection**: an explainability-driven branch selection criterion that combines a minimum predictive quality floor with pairwise spatial saliency divergence ($D(i, j) \in [0, 1]$).

---

## 2. Mathematical Formulation

### 2.1 Branch Saliency Maps via Grad-CAM
For input wafer image $\mathbf{X} \in \mathbb{R}^{3 \times 224 \times 224}$ and branch head $b \in \{B_1 \dots B_5\}$, the spatial saliency map $\mathbf{S}_b \in [0, 1]^{224 \times 224}$ for target defect class $c$ is computed by backward-gradient weighted accumulation over the final convolutional layer $A^k$:

$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i, j}^k}$$

$$\mathbf{S}_b = \text{ReLU}\left( \sum_{k} \alpha_k^c A^k \right)$$

### 2.2 Pairwise Spatial Saliency Divergence
The spatial divergence $D(i, j)$ between saliency maps $\mathbf{S}_i$ and $\mathbf{S}_j$ of branches $i$ and $j$ is computed using normalized Cosine spatial distance:

$$D(i, j) = 1.0 - \frac{\langle \mathbf{S}_i, \mathbf{S}_j \rangle}{\|\mathbf{S}_i\|_2 \|\mathbf{S}_j\|_2} \in [0.0, 1.0]$$

### 2.3 Quality-Constrained Selection Objective
Given validation predictive quality scores $Q(b)$ (Macro F1-score), eligible candidate branches $\mathcal{B}_{\text{eligible}}$ are filtered using a predictive quality floor:

$$\mathcal{B}_{\text{eligible}} = \left\{ b \in \{B_1 \dots B_5\} \;\middle|\; Q(b) \ge \tau \cdot \max_{b'} Q(b') \right\}$$

where $\tau = 0.80$. The selected branch subset $\mathcal{S}^*$ maximizes joint predictive quality and spatial saliency complementarity:

$$\mathcal{S}^* = \arg\max_{\mathcal{S} \subseteq \mathcal{B}_{\text{eligible}}, |\mathcal{S}| \ge 2} \left[ Q(\text{Ens}(\mathcal{S})) + \lambda_{\text{div}} \cdot \bar{D}(\mathcal{S}) \right]$$

where $\bar{D}(\mathcal{S}) = \frac{2}{|\mathcal{S}|(|\mathcal{S}|-1)} \sum_{i < j \in \mathcal{S}} D(i, j)$ is the average pairwise spatial divergence of the subset.

---

## 3. Dataset & Preprocessing Protocol

- **Dataset:** WM-811K benchmark (62,248 labeled wafer maps across 9 failure categories).
- **Split Protocol:** Canonical $3:1:1$ stratified split ($37,348$ training, $12,450$ validation, $12,450$ test) with zero data leakage.
- **Preprocessing:** Padded Nearest Neighbor interpolation to $(224 \times 224)$, preserving 100% ternary purity (Die absent = 0, Pass die = 128/255, Defect die = 1.0).

---

## 4. Controlled Hyperparameter Optimization (AlMuhaideb & Khan Protocol)

Optuna TPE sampler was executed over the validation split with a fixed exploration budget:
- **Learning Rate:** $4.33 \times 10^{-4}$
- **Dropout:** $0.30$
- **Weight Decay:** $2.91 \times 10^{-4}$
- **Optimizer:** Adam with ReduceLROnPlateau scheduler (factor=0.5, patience=2).

---

## 5. System Architecture & End-to-End MVP

The complete research system includes:
1. **PyTorch Model Core (`src/models/ebcnn.py`):** 5-branch VGG-16 architecture with exact classifier heads.
2. **Grad-CAM & Selection Engine (`src/branch_selection/selector.py`):** Dynamic quality floor + Cosine divergence selection.
3. **FastAPI REST Server (`backend/main.py`):** Headless PyTorch inference server serving real predictions, Grad-CAM maps, and divergence matrices.
4. **React Single Page Application (`frontend/`):** Interactive web dashboard with real wafer upload, Grad-CAM heatmap visualization, matrix inspector, and 10-seed statistical report view.

---

## 6. Verification & Reproducibility Summary

- **PyTorch Unit Test Suite:** All 6 unit tests in `tests/` pass with zero failures.
- **API Backend:** Endpoint tests (`GET /api/health`, `GET /api/samples`, `POST /api/predict`) verified cleanly.
- **Frontend Bundle:** Production bundle built cleanly with Vite (`dist/assets/index-4MvR8i_C.js`).
