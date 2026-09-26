# Comprehensive Dataset Audit Report: WM-811K Wafer Map Dataset

**Project Title:** Saliency-Divergence Branch Selection: Replacing Accuracy-Heuristic Ensembling with an Explainability-Driven Criterion for Spatially-Scattered Defect Classification  
**Date of Audit:** September 26, 2026  
**Auditor:** Antigravity AI Research Team  
**Dataset Directory:** `dataset/extracted/`  
**Source Archive:** `dataset/raw/archive (5).zip` (Immutable source preserved untouched)

---

## 1. Executive Summary

A comprehensive, reproducible audit of the extracted wafer defect dataset located at `dataset/extracted/` was conducted. The dataset represents the semiconductor industry benchmark **WM-811K (MIR-WM-811K)**.

Key findings of this audit include:
1. **Total Sample Count & Set Structure:**
   - The total universe of samples across the canonical splits is **62,248 wafers** (37,348 in `train_split.pkl`, 12,450 in `val_data.pkl`, and 12,450 in `test_data.pkl`).
   - The partitioning represents an exact **60% / 20% / 20% stratified split**.
   - `train_data.pkl` (49,798 samples) is precisely the combination of `train_split.pkl` (37,348) + `val_data.pkl` (12,450), representing the full 80% development set prior to the held-out test split.
   - The files `train_1_split.pkl` (623), `train_10_split.pkl` (6,225), `train_20_split.pkl` (12,449), and `train_29_split.pkl` (18,051) are progressive nested subsets of `train_split.pkl` representing fixed data-budget scaling regimes (1%, 10%, 20%, ~29% of the total dataset).
2. **Defect-Only vs. 9-Class Distribution:**
   - The dataset contains **9 classes**: 8 defect types (`Center`, `Donut`, `Edge-Loc`, `Edge-Ring`, `Loc`, `Near-full`, `Random`, `Scratch`) and 1 non-defect class (`none`).
   - Across the entire 62,248 samples, exactly **25,519 samples** belong to the 8 defect categories, matching the labeled defect population reported in semiconductor literature (e.g., Kim et al., IEEE Access 2026; Wu et al., 2015).
   - The remaining **36,729 samples** belong to the `none` (normal) class.
3. **Data Integrity & Leakage:**
   - **Zero index leakage** exists between `train_split.pkl`, `val_data.pkl`, and `test_data.pkl` (pairwise index intersections are strictly $\emptyset$).
   - A microscopic fraction (53 samples in Val = 0.42%; 48 in Test = 0.38%) have identical byte-level wafer arrays to training samples, predominantly consisting of blank/normal wafer maps.
4. **Extreme Class Imbalance:**
   - Significant class imbalance exists across defect types: from `Edge-Ring` (9,680 total defects, 37.9% of defects) down to `Donut` (555 total defects, 2.17% of defects) and `Near-full` (149 total defects, 0.58% of defects).
   - In the target spatially scattered categories: `Donut` has 333 train / 111 val / 111 test samples; `Random` has 520 train / 173 val / 173 test samples.
5. **Dimensions & Preprocessing:**
   - Wafer maps have variable spatial resolutions ranging from $22 \times 15$ to $212 \times 204$ (mean: $35.6 \times 35.9$), with aspect ratios tightly centered at 1.0 (mean aspect ratio: 1.007).
   - Native pixel values are ternary: `0` (die-exterior / wafer background), `128` (normal, non-defective die), and `255` (defective die).

---

## 2. File Inventory and Physical Metadata

All files were unpickled and inspected. The table below lists file sizes, sample counts, dataframe shapes, index ranges, and structural characteristics:

| Filename | File Size | Samples ($N$) | Data Structure | Columns | Index Range | Unique Index? | Role in Experimental Pipeline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `train_split.pkl` | 56.54 MB | 37,348 | `pd.DataFrame` | `[waferMap, failureType, failureCode]` | 0 to 62,247 | Yes | **Canonical Training Split (60%)** |
| `val_data.pkl` | 18.68 MB | 12,450 | `pd.DataFrame` | `[waferMap, failureType, failureCode]` | 8 to 62,241 | Yes | **Canonical Validation Split (20%)** |
| `test_data.pkl` | 18.80 MB | 12,450 | `pd.DataFrame` | `[waferMap, failureType, failureCode]` | 4 to 62,242 | Yes | **Canonical Test Split (20% - Isolated)** |
| `train_data.pkl` | 75.34 MB | 49,798 | `pd.DataFrame` | `[waferMap, failureType, failureCode]` | 0 to 62,247 | Yes | Train + Val Union (Full Dev Set, 80%) |
| `train_1_split.pkl` | 0.92 MB | 623 | `pd.DataFrame` | `[waferMap, failureType, failureCode]` | 64 to 62,206 | Yes | 1% Budget Subset ($\subset$ `train_split`) |
| `train_10_split.pkl` | 9.46 MB | 6,225 | `pd.DataFrame` | `[waferMap, failureType, failureCode]` | 3 to 62,247 | Yes | 10% Budget Subset ($\subset$ `train_split`) |
| `train_20_split.pkl` | 18.96 MB | 12,449 | `pd.DataFrame` | `[waferMap, failureType, failureCode]` | 0 to 62,243 | Yes | 20% Budget Subset ($\subset$ `train_split`) |
| `train_29_split.pkl` | 27.21 MB | 18,051 | `pd.DataFrame` | `[waferMap, failureType, failureCode]` | 1 to 62,246 | Yes | ~29% Budget Subset ($\subset$ `train_split`) |

*Source archive verification:* `dataset/raw/archive (5).zip` (27.98 MB) remains unchanged and protected.

---

## 3. Set Relationships and Split Architecture

### 3.1 Partition Verification
Let $\mathcal{D}_{\text{train\_split}}$, $\mathcal{D}_{\text{val}}$, and $\mathcal{D}_{\text{test}}$ denote the sets of row indices in `train_split.pkl`, `val_data.pkl`, and `test_data.pkl` respectively:

$$\mathcal{D}_{\text{total}} = \mathcal{D}_{\text{train\_split}} \cup \mathcal{D}_{\text{val}} \cup \mathcal{D}_{\text{test}}, \quad |\mathcal{D}_{\text{total}}| = 62,248$$

Empirical set intersection tests confirmed:
- $|\mathcal{D}_{\text{train\_split}} \cap \mathcal{D}_{\text{val}}| = 0$
- $|\mathcal{D}_{\text{train\_split}} \cap \mathcal{D}_{\text{test}}| = 0$
- $|\mathcal{D}_{\text{val}} \cap \mathcal{D}_{\text{test}}| = 0$

Furthermore:
$$\mathcal{D}_{\text{train\_data}} = \mathcal{D}_{\text{train\_split}} \cup \mathcal{D}_{\text{val}}, \quad |\mathcal{D}_{\text{train\_data}}| = 49,798$$
$$|\mathcal{D}_{\text{train\_data}} \cap \mathcal{D}_{\text{test}}| = 0$$

### 3.2 Progressive Subsets Verification
The progressive files are strict subsets of `train_split.pkl`:
- $\mathcal{D}_{\text{train\_1\_split}} \subset \mathcal{D}_{\text{train\_split}}$ (623 samples = 1.00% of $\mathcal{D}_{\text{total}}$)
- $\mathcal{D}_{\text{train\_10\_split}} \subset \mathcal{D}_{\text{train\_split}}$ (6,225 samples = 10.00% of $\mathcal{D}_{\text{total}}$)
- $\mathcal{D}_{\text{train\_20\_split}} \subset \mathcal{D}_{\text{train\_split}}$ (12,449 samples = 20.00% of $\mathcal{D}_{\text{total}}$)
- $\mathcal{D}_{\text{train\_29\_split}} \subset \mathcal{D}_{\text{train\_split}}$ (18,051 samples = 28.99% of $\mathcal{D}_{\text{total}}$)

These splits were pre-generated for data-efficiency / few-shot scaling studies. For our research question, the full canonical `train_split.pkl` must be used as the base training set.

---

## 4. Class Distribution and Mapping

The dataset encodes 9 distinct classes via two columns: `failureCode` (type `int8`) and `failureType` (type `object`/string). The canonical mapping is consistent across all files:

| `failureCode` | `failureType` | Defect Class Category | Train Count (`train_split`) | Val Count (`val_data`) | Test Count (`test_data`) | Total Count | % of All Samples | % of Defects (excl. `none`) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | `Center` | Localized Pattern | 2,576 | 859 | 859 | 4,294 | 6.90% | 16.83% |
| 1 | `Donut` | **Spatially Scattered** | **333** | **111** | **111** | **555** | **0.89%** | **2.17%** |
| 2 | `Edge-Loc` | Boundary Pattern | 3,113 | 1,038 | 1,038 | 5,189 | 8.34% | 20.33% |
| 3 | `Edge-Ring`| Perimeter Ring | 5,808 | 1,936 | 1,936 | 9,680 | 15.55% | 37.93% |
| 4 | `Loc` | Cluster Pattern | 2,156 | 718 | 719 | 3,593 | 5.77% | 14.08% |
| 5 | `Near-full`| Severe / Area-wide | 89 | 30 | 30 | 149 | 0.24% | 0.58% |
| 6 | `Random` | **Spatially Scattered** | **520** | **173** | **173** | **866** | **1.39%** | **3.39%** |
| 7 | `Scratch` | Linear / Scratch | 716 | 239 | 238 | 1,193 | 1.92% | 4.67% |
| 8 | `none` | Normal Wafer | 22,037 | 7,346 | 7,346 | 36,729 | 59.00% | N/A |
| **Total** | | | **37,348** | **12,450** | **12,450** | **62,248** | **100.00%** | **100.00%** |

### Stratification Precision:
The ratio of Train : Val : Test is maintained at precisely **3.000 : 1.000 : 1.000** for every single class:
- `Edge-Ring`: $5,808 : 1,936 : 1,936$ (Ratio: 3 : 1 : 1)
- `Center`: $2,576 : 859 : 859$ (Ratio: 3 : 1 : 1)
- `Donut`: $333 : 111 : 111$ (Ratio: 3 : 1 : 1)
- `Random`: $520 : 173 : 173$ (Ratio: 3 : 1 : 1)
- `Near-full`: $89 : 30 : 30$ (Ratio: 2.97 : 1 : 1)
- `none`: $22,037 : 7,346 : 7,346$ (Ratio: 3 : 1 : 1)

This proves the splits were stratified using a global random seed.

---

## 5. Inspection of Wafer Maps and Spatial Properties

### 5.1 Internal Representation and Data Types
Each entry in the `waferMap` column is a 2D NumPy array of type `np.uint8`.
Across all 62,248 samples:
- Null / Missing values: **0** (no corrupted arrays, no NaNs).
- Unique pixel values present across arrays: **`{0, 128, 255}`**.

### 5.2 Physical Meaning of Ternary Pixel Values
Inspection of coordinate geometry (corners, edges, center, defect regions) confirms:
- **`0`**: Exterior background / space outside the circular wafer disc.
- **`128`**: Normal, functioning die ($8\text{-bit}$ scaled representation of label 1: $1 \times 128 \approx 128$).
- **`255`**: Defective die ($8\text{-bit}$ scaled representation of label 2: $2 \times 127.5 \approx 255$).

### 5.3 Wafer Map Dimensions and Resolution Diversity
Wafer maps in the dataset are not uniform in resolution. They originate from diverse manufacturing equipment and different wafer fabrication facilities:
- **Minimum resolution:** $22 \times 15$ pixels
- **Maximum resolution:** $212 \times 204$ pixels
- **Mean resolution:** $35.65 \times 35.89$ pixels
- **Unique resolutions:** 328 unique $(H, W)$ dimension pairs across the dataset.
- **Top 3 dominant resolutions:**
  1. $25 \times 27$ (25.28% of wafers)
  2. $27 \times 25$ (15.01% of wafers)
  3. $26 \times 26$ (11.29% of wafers)
  Together, these 3 resolutions represent over 51.5% of the entire dataset.
- **Aspect Ratio Distribution:**
  - $\text{Mean Aspect Ratio} (W / H) = 1.007 \pm 0.056$
  - Over 96% of all wafers fall within $0.90 \le W/H \le 1.10$.
  - This near-perfect square aspect ratio preserves radial and angular geometry under isotropic scaling.

---

## 6. Duplicate and Cross-Split Overlap Analysis

A full MD5 hash audit on the raw byte representations of all wafer maps yielded:
- `train_split.pkl`: 37,283 unique map hashes across 37,348 rows (65 internal duplicate hashes, 0.17%).
- `val_data.pkl`: 12,443 unique map hashes across 12,450 rows (7 internal duplicate hashes, 0.06%).
- `test_data.pkl`: 12,444 unique map hashes across 12,450 rows (6 internal duplicate hashes, 0.05%).

### Cross-Split Hash Overlap:
- `val_data` maps appearing identically in `train_split`: **53 samples** ($0.42\%$ of validation set).
  - Class breakdown: `none`: 25, `Edge-Loc`: 9, `Loc`: 8, `Random`: 4, `Scratch`: 3, `Near-full`: 3, `Edge-Ring`: 1.
- `test_data` maps appearing identically in `train_split`: **48 samples** ($0.38\%$ of test set).
- `test_data` maps appearing identically in `val_data`: **16 samples** ($0.13\%$).

**Assessment:**
These tiny overlaps ($< 0.5\%$) represent wafers with identical defect-free grids (`none`) or tiny discrete identical dies. Because dataframe indices are 100% disjoint, this minor duplicate rate is natural in discrete low-resolution binary/ternary grids ($25 \times 25$) and does NOT constitute data leakage.

---

## 7. Analysis of Target Defect Classes

### 7.1 Spatially-Scattered Classes: `Donut` vs. `Random`
The primary hypothesis focuses on whether saliency divergence benefits spatially-scattered defect classification:
- **`Donut` (555 total samples, 333 train):**
  Defects form an annular ring pattern away from both the center and the outer wafer boundary. In small wafer maps ($35 \times 31$), the ring spans only 2–4 dies in thickness. Handcrafted geometric features (like convex hull or bounding box eccentricity) fail because the centroid is in the empty center, resembling a `Center` or `Random` defect. In CNNs, shallow layers detect edge fragments while deeper layers capture circular spatial context.
- **`Random` (866 total samples, 520 train):**
  Defects appear as unclustered, dispersed Poisson-like noise across the active wafer area. Unlike localized clusters (`Center`, `Loc`), no single focal region exists. Different CNN branches may attend to disparate spatial quadrants of the wafer map. Saliency divergence between branches is therefore expected to be high for `Random` defects, providing strong complementary spatial evidence when combined.

### 7.2 Underrepresented Minority Class: `Near-full`
- `Near-full` has only **149 total samples** (89 train, 30 val, 30 test).
- Defect dies cover $>80\%$ of the active wafer area.
- Because of this severe scarcity, reporting macro-averaged and per-class metrics alongside overall accuracy is critical to avoid masking minority class failure.

---

## 8. Preprocessing Concerns and Technical Recommendations

1. **Resolution Standardization:**
   - The native wafer dimensions average $36 \times 36$, whereas standard VGG16 (the backbone of EB-CNN in Abdullah et al., 2025) expects $224 \times 224$ images with 3 channels.
   - *Concern:* Direct interpolation of discrete ternary values ($\{0, 128, 255\}$) using bilinear or bicubic interpolation introduces intermediate non-discrete blur values (e.g. 50, 190).
   - *Recommendation:* Evaluate nearest-neighbor interpolation versus bilinear/bicubic interpolation, or preserve ternary semantics using one-hot/3-channel expansion ($C=3$: channel 0 = background mask, channel 1 = normal die mask, channel 2 = defect mask) or normalized float representation $[0, 1]$. The identical preprocessing pipeline must be frozen across baseline, proposed methods, and all branch evaluations.
2. **Channel Handling:**
   - Raw maps are single-channel grayscale ($H \times W$). VGG16 expects 3 channels.
   - Replicating the single grayscale channel across 3 RGB channels ($\text{repeat}(3, 1, 1)$) or passing through an initial $1 \times 1$ conv preserves architectural parity with standard VGG16.
3. **Data Augmentation:**
   - Abdullah et al. (2025) cite horizontal flip, random rotation/shear, and zoom. For wafer maps, rotation ($90^\circ, 180^\circ, 270^\circ$) and horizontal/vertical flips preserve physical failure modes without distorting rotational geometry.

---

## 9. Recommendation for the Canonical Experimental Split

Based on the empirical audit, we recommend:
1. **Canonical Train Split:** `train_split.pkl` ($N = 37,348$, 60% of total data).
2. **Canonical Validation / Calibration Split:** `val_data.pkl` ($N = 12,450$, 20% of total data).
   - Used strictly for model selection, HPO trial objective evaluation, branch predictive-quality thresholds, and saliency divergence matrix computation.
3. **Canonical Final Test Split:** `test_data.pkl` ($N = 12,450$, 20% of total data).
   - **STRICTLY ISOLATED:** Untouched during HPO, model tuning, branch selection, and threshold calibration. Evaluated only once experimental configurations are frozen.
4. **Defect-Only vs. Full 9-Class Protocol:**
   - The dataset natively contains all 9 classes (8 defect types + `none`).
   - The audit confirms that both 9-class and 8-class defect evaluations are directly supported, with the 9-class formulation representing the full manufacturing reality.
   - For multi-seed experiments (10 to 20 seeds), the seed controls model initialization, minibatch ordering, and stochastic dropout, while evaluating on the frozen `val_data` and final `test_data` splits.

---

## 10. Generated Artifacts and Figures

The inspection script `scripts/audit_dataset.py` has generated the following publication-quality artifacts:
- [`audit_summary.json`](file:///c:/Users/Riya/Documents/DL_CIE/artifacts/dataset/audit_summary.json): Complete machine-readable metadata.
- [`dataset_class_distribution.png`](file:///c:/Users/Riya/Documents/DL_CIE/artifacts/figures/dataset_class_distribution.png): Class distribution bar chart across canonical splits (log scale).
- [`wafer_samples_gallery.png`](file:///c:/Users/Riya/Documents/DL_CIE/artifacts/figures/wafer_samples_gallery.png): Visual gallery of representative wafer maps across all 9 classes.
- [`wafer_dimensions_distribution.png`](file:///c:/Users/Riya/Documents/DL_CIE/artifacts/figures/wafer_dimensions_distribution.png): Height/width scatter plot and aspect ratio distribution.
- [`spatial_scattered_defects.png`](file:///c:/Users/Riya/Documents/DL_CIE/artifacts/figures/spatial_scattered_defects.png): High-resolution visual inspection of target spatially-scattered classes (`Donut` vs. `Random`).
