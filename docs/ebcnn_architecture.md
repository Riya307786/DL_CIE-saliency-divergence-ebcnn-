# EB-CNN Architecture Documentation

**Reference Paper:** Azizi Abdullah, Wei Soong Wong, Dheeb Albashish. "EB-CNN: Ensemble of branch convolutional neural network for image classification." *Pattern Recognition Letters* 189 (2025) 1–7.  
**Implementation Source:** [`src/models/ebcnn.py`](file:///c:/Users/Riya/Documents/DL_CIE/src/models/ebcnn.py)

---

## 1. Architectural Philosophy & Overview

Conventional CNNs rely on a single output layer at the deepest level of the network. This assumes that all target defect categories are distinguished at the same level of visual abstraction. However:
- Some defect patterns (e.g., `Edge-Ring`, `Scratch`, or localized clusters) exhibit sharp, fine-grained geometric edges discernible in early/intermediate stages.
- Other defect patterns (e.g., `Donut`, `Random`, `Near-full`) require global spatial reasoning across the entire wafer disc.

EB-CNN introduces a **multi-branch, multi-scale hierarchical ensemble** based on the 16-layer VGG network (VGG-16). The network splits into five distinct branch classifiers ($B_1$ through $B_5$), each tapping into the hierarchical representations of the network after every pooling stage.

---

## 2. Detailed Layer Specifications

The input tensor has shape $(B, 3, 224, 224)$.

### 2.1 Feature Extraction Backbone (5 Stages)

| Stage | Convolutional Layers | Output Channels | Intermediate Feature Map | Pooling Layer | Pooled Spatial Dim |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **Stage 1** | $\text{Conv}(3 \to 64, 3\times3) \to \text{BN} \to \text{ReLU} \to \text{Conv}(64 \to 64, 3\times3) \to \text{BN} \to \text{ReLU}$ | $64$ | $224 \times 224$ | $\text{MaxPool}(2, 2)$ | $112 \times 112$ |
| **Stage 2** | $\text{Conv}(64 \to 128, 3\times3) \to \text{BN} \to \text{ReLU} \to \text{Conv}(128 \to 128, 3\times3) \to \text{BN} \to \text{ReLU}$ | $128$ | $112 \times 112$ | $\text{MaxPool}(2, 2)$ | $56 \times 56$ |
| **Stage 3** | $\text{Conv}(128 \to 256, 3\times3) \times 3 \to \text{BN} \to \text{ReLU}$ | $256$ | $56 \times 56$ | $\text{MaxPool}(2, 2)$ | $28 \times 28$ |
| **Stage 4** | $\text{Conv}(256 \to 512, 3\times3) \times 3 \to \text{BN} \to \text{ReLU}$ | $512$ | $28 \times 28$ | $\text{MaxPool}(2, 2)$ | $14 \times 14$ |
| **Stage 5** | $\text{Conv}(512 \to 512, 3\times3) \times 3 \to \text{BN} \to \text{ReLU}$ | $512$ | $14 \times 14$ | $\text{MaxPool}(2, 2)$ | $7 \times 7$ |

### 2.2 Branch Classification Heads ($H_1$ through $H_5$)

Each branch classification head attaches directly after the corresponding stage's max pooling layer:
$$\text{Input } (C_k, H_k, W_k) \xrightarrow{\text{AdaptiveAvgPool}(7 \times 7)} (C_k, 7, 7) \xrightarrow{\text{Flatten}} \mathbb{R}^{49 \cdot C_k} \xrightarrow{\text{Linear}(49 \cdot C_k \to 512)} \xrightarrow{\text{BN}} \xrightarrow{\text{ReLU}} \xrightarrow{\text{Dropout}(p=0.5)} \xrightarrow{\text{Linear}(512 \to 9)} \text{Logits}$$

- **Branch 1 Head ($H_1$):** Input $64 \times 112 \times 112 \to \text{Pool}(7, 7) \to 3,136 \to 512 \to 9$
- **Branch 2 Head ($H_2$):** Input $128 \times 56 \times 56 \to \text{Pool}(7, 7) \to 6,272 \to 512 \to 9$
- **Branch 3 Head ($H_3$):** Input $256 \times 28 \times 28 \to \text{Pool}(7, 7) \to 12,544 \to 512 \to 9$
- **Branch 4 Head ($H_4$):** Input $512 \times 14 \times 14 \to \text{Pool}(7, 7) \to 25,088 \to 512 \to 9$
- **Branch 5 Head ($H_5$):** Input $512 \times 7 \times 7 \to \text{Pool}(7, 7) \to 25,088 \to 512 \to 9$

---

## 3. Original EB-CNN Branch Stacking & Heuristic Selection

Abdullah et al. (Section 3.2, Figure 2) define five candidate stacked ensemble models constructed from the trained branches:

| Stacked Model | Constituent Branches | Abstraction Focus |
| :---: | :---: | :--- |
| **$C_1$** | $\{B_5\}$ | Deepest semantic features alone |
| **$C_2$** | $\{B_5, B_4\}$ | High-level + intermediate semantic features |
| **$C_3$** | $\{B_5, B_4, B_3\}$ | Intermediate-to-high level features |
| **$C_4$** | $\{B_5, B_4, B_3, B_2\}$ | Coarse-to-fine multi-scale hierarchy |
| **$C_5$** | $\{B_5, B_4, B_3, B_2, B_1\}$ | Complete hierarchical spectrum |

### Original Branch-Selection Heuristic:
The reference paper defines model selection via Equation (4):
$$\text{Best Model } c^* = \arg\max_{k \in \{1..5\}} \text{Accuracy}(C_k \mid \mathcal{D}_{\text{val}})$$

The candidate combinations are evaluated on validation data using an algebraic combining rule:
1. **Mean Rule:** $P(x, j) = \frac{1}{N} \sum_{k=1}^N P_k(j \mid x)$
2. **Product Rule:** $P(x, j) \propto \prod_{k=1}^N P_k(j \mid x)$
3. **Max Rule:** $P(x, j) = \max_k P_k(j \mid x)$

This original heuristic is purely accuracy-driven and evaluates only nested fixed combinations $\{C_1..C_5\}$, ignoring spatial explainability and feature complementarity.

---

## 4. Grad-CAM Layer Targets for Explainability

To extract spatial saliency from each branch:
- **Branch 1:** Hook on last Conv2d of `stage1` (`conv1_2`, 64 feature maps, $224 \times 224$).
- **Branch 2:** Hook on last Conv2d of `stage2` (`conv2_2`, 128 feature maps, $112 \times 112$).
- **Branch 3:** Hook on last Conv2d of `stage3` (`conv3_3`, 256 feature maps, $56 \times 56$).
- **Branch 4:** Hook on last Conv2d of `stage4` (`conv4_3`, 512 feature maps, $28 \times 28$).
- **Branch 5:** Hook on last Conv2d of `stage5` (`conv5_3`, 512 feature maps, $14 \times 14$).

These layers capture the richest class-discriminative spatial activations before classification pooling.
