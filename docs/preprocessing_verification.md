# Preprocessing Verification and Protocol Specification

**Project Title:** Saliency-Divergence Branch Selection: Replacing Accuracy-Heuristic Ensembling with an Explainability-Driven Criterion for Spatially-Scattered Defect Classification  
**Date of Verification:** September 26, 2026  
**Module:** [`src/data/preprocessing.py`](file:///c:/Users/Riya/Documents/DL_CIE/src/data/preprocessing.py)  
**Verification Script:** [`scripts/verify_preprocessing.py`](file:///c:/Users/Riya/Documents/DL_CIE/scripts/verify_preprocessing.py)  
**Artifacts Generated:**  
- [`preprocessing_verification.json`](file:///c:/Users/Riya/Documents/DL_CIE/artifacts/dataset/preprocessing_verification.json)  
- [`preprocessing_comparison.png`](file:///c:/Users/Riya/Documents/DL_CIE/artifacts/figures/preprocessing_comparison.png)  
- [`interpolation_defect_profiles.png`](file:///c:/Users/Riya/Documents/DL_CIE/artifacts/figures/interpolation_defect_profiles.png)

---

## 1. Problem Statement & Motivation

Wafer maps in the WM-811K benchmark are discrete ternary lattices encoding semiconductor test results:
- $0$: Exterior space outside the circular wafer boundary (background).
- $128$: Normal, functioning die (scaled $1 \times 128$).
- $255$: Defective die (scaled $2 \times 127.5$).

The native wafer dimensions range from $22 \times 15$ to $212 \times 204$ (mean: $35.65 \times 35.89$). Conversely, the EB-CNN architecture (Abdullah et al., *Pattern Recognition Letters*, 2025) employs VGG-16 branches with five hierarchical pooling stages ($2 \times 2$ MaxPool with stride 2), requiring $224 \times 224 \times 3$ inputs.

Blindly resizing wafer maps via default continuous interpolation (bilinear or bicubic) corrupts the discrete ternary nature of the data, introducing continuous gradient halos that alter thin defects (such as thin scratches, delicate donut rings, or isolated Poisson noise in random patterns). Furthermore, anisotropic scaling (stretching non-square maps directly to $224 \times 224$) turns circular wafers into ellipses, distorting spatial symmetry.

This verification establishes a scientifically justified, frozen preprocessing pipeline.

---

## 2. Comparative Evaluation of Candidate Pipelines

We evaluated four candidate preprocessing configurations across 225 stratified wafer samples spanning all 9 classes:
1. **Direct Bilinear:** Direct resize to $224 \times 224$ using bilinear interpolation (no padding).
2. **Direct Nearest:** Direct resize to $224 \times 224$ using nearest-neighbor interpolation (no padding).
3. **Padded Bilinear:** Symmetrically zero-padded to square, then resized via bilinear interpolation.
4. **Padded Nearest (Proposed Pipeline):** Symmetrically zero-padded to square, resized via nearest-neighbor, divided by 255.0, and replicated across 3 RGB channels.

### Quantitative Audit Metrics

| Metric | Direct Bilinear | Direct Nearest | Padded Bilinear | Padded Nearest (Proposed) | Target Direction |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ternary Value Purity (%)** | $64.04\%$ | $\mathbf{100.00\%}$ | $65.86\%$ | $\mathbf{100.00\%}$ | Higher is better ($100\%$ ideal) |
| **Defect Ratio Relative Error** | $0.0388$ | $\mathbf{0.0010}$ | $0.0412$ | $\mathbf{0.0010}$ | Lower is better ($0$ ideal) |
| **Wafer Circularity Error** | $0.3661$ | $0.5909$ | $0.3680$ | $\mathbf{0.5899}$ | Preserved circular geometry |
| **Boundary Gradient Dispersion** | $100.00\%$ | $87.36\%$ | $100.00\%$ | $\mathbf{86.44\%}$ | Lower is better (avoids blur) |

### Key Findings:
1. **Continuous Interpolation Destroys Discrete Semantics:**
   Bilinear interpolation results in **$34.14\%$ to $35.96\%$ of pixels** having corrupted, non-existent intermediate values ($10 \le I \le 120$ and $135 \le I \le 250$). This artificially broadens defect boundaries and attenuates peak defect intensity (reducing true defect die activations from $1.0$ down to $0.85$).
2. **Nearest-Neighbor Preserves Exact Physical States:**
   Nearest-neighbor achieves **$100.00\%$ ternary purity**, keeping every pixel strictly in $\{0.0, 128/255, 1.0\}$.
3. **Defect Area Preservation:**
   Defect ratio distortion is reduced **$40\times$** with nearest-neighbor ($0.0010$ vs $0.0412$), ensuring that small classes (`Donut`, `Scratch`, `Random`) do not suffer artificial dilation or erosion.
4. **Aspect Ratio Preservation via Symmetric Padding:**
   Because wafer maps have slight non-square aspect ratios (e.g. $25 \times 27$ or $35 \times 31$), padding symmetrically with 0s before resizing ensures that circular wafers remain circles, preserving true radial symmetry.

---

## 3. Specification of the Frozen Canonical Preprocessing Pipeline

The canonical preprocessor is implemented in [`src/data/preprocessing.py`](file:///c:/Users/Riya/Documents/DL_CIE/src/data/preprocessing.py) through the class `WaferPreprocessor`:

$$\mathbf{x}_{\text{raw}} \in \{0, 128, 255\}^{H \times W} \xrightarrow{\text{Pad to Square}} \mathbf{x}_{\text{pad}} \in \{0, 128, 255\}^{M \times M} \xrightarrow{\text{Nearest Resize}} \mathbf{x}_{224} \in \{0, 128, 255\}^{224 \times 224} \xrightarrow{\div 255.0} \mathbf{x}_{\text{norm}} \in [0, 1]^{224 \times 224} \xrightarrow{\text{Replicate } \times 3} \mathbf{X} \in \mathbb{R}^{3 \times 224 \times 224}$$

Where:
1. $M = \max(H, W)$. Symmetrical padding with $0$ (exterior wafer background) is applied:
   $$p_{\text{top}} = \lfloor (M - H)/2 \rfloor, \quad p_{\text{bottom}} = M - H - p_{\text{top}}$$
   $$p_{\text{left}} = \lfloor (M - W)/2 \rfloor, \quad p_{\text{right}} = M - W - p_{\text{left}}$$
2. Nearest-neighbor interpolation resamples $\mathbf{x}_{\text{pad}}$ to $(224, 224)$.
3. Normalization divides by $255.0$, mapping values to $\{0.0, 128/255 \approx 0.50196, 1.0\}$.
4. The normalized single-channel map is replicated 3 times along the channel axis to create a 3-channel tensor $(3, 224, 224)$.

---

## 4. Architectural Necessity of 224×224 Resolution for Grad-CAM

The $224 \times 224$ resolution is explicitly required for two reasons:
1. **EB-CNN VGG-16 Backbone Parity:**
   VGG-16 features 5 pooling stages with stride 2 ($2^5 = 32\times$ downsampling). An input of $224 \times 224$ produces a final feature map of $7 \times 7 \times 512$.
2. **Grad-CAM Spatial Granularity:**
   At $224 \times 224$, the final convolutional activations have a spatial grid of $7 \times 7 = 49$ spatial locations, allowing Grad-CAM to discriminate between:
   - Annular ring geometry (`Donut`)
   - Centered cluster (`Center`)
   - Outer perimeter (`Edge-Ring`)
   - Dispersed spatial coverage (`Random`)
   Downsampling to smaller resolutions (e.g. $64 \times 64$) would produce a $2 \times 2$ feature map, completely extinguishing spatial interpretability.

---

## 5. Experimental Control Guarantee

To ensure scientific validity and strict experimental control:
- The exact same preprocessing pipeline will be used across **all five EB-CNN branches ($C_1, C_2, C_3, C_4, C_5$)**.
- The exact same preprocessing pipeline will be used for the **original EB-CNN accuracy baseline**.
- The exact same preprocessing pipeline will be used for the **proposed saliency-divergence method**.
- The test set (`test_data.pkl`) remains strictly untouched and will only pass through this preprocessor during final frozen evaluation.
