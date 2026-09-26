import torch
import numpy as np
from typing import Dict, List, Tuple, Union

class SaliencyDivergence:
    """
    Computes spatial saliency divergence and complementarity metrics across EB-CNN branches.
    """
    def __init__(self, method: str = "cosine", binarize_threshold: float = 0.4):
        self.method = method.lower()
        self.binarize_threshold = binarize_threshold

    def compute_pairwise_divergence(
        self,
        map1: torch.Tensor,
        map2: torch.Tensor
    ) -> float:
        """
        Computes divergence D(i, j) between two normalized 2D saliency maps.
        Returns a float in [0.0, 1.0], where higher indicates greater spatial divergence/complementarity.
        """
        if self.method == "iou":
            # Method A: IoU-based divergence
            m1_bin = (map1 >= self.binarize_threshold).float()
            m2_bin = (map2 >= self.binarize_threshold).float()
            
            intersection = torch.sum(m1_bin * m2_bin).item()
            union = torch.sum((m1_bin + m2_bin) > 0).item()
            
            if union == 0:
                iou = 1.0 # Both empty -> 0 divergence
            else:
                iou = intersection / union
            return float(np.clip(1.0 - iou, 0.0, 1.0))

        elif self.method == "cosine":
            # Method B: Cosine-based divergence
            f1 = map1.flatten().float()
            f2 = map2.flatten().float()
            
            norm1 = torch.norm(f1, p=2).item()
            norm2 = torch.norm(f2, p=2).item()
            
            if norm1 < 1e-7 or norm2 < 1e-7:
                cos_sim = 1.0
            else:
                cos_sim = torch.dot(f1, f2).item() / (norm1 * norm2)
            cos_sim = np.clip(cos_sim, 0.0, 1.0)
            return float(1.0 - cos_sim)

        elif self.method == "correlation":
            # Method C: Pearson correlation-based divergence (ablation)
            f1 = map1.flatten().float()
            f2 = map2.flatten().float()
            
            f1_mean = f1 - torch.mean(f1)
            f2_mean = f2 - torch.mean(f2)
            
            denom = torch.norm(f1_mean, p=2) * torch.norm(f2_mean, p=2)
            if denom.item() < 1e-7:
                r = 1.0
            else:
                r = (torch.dot(f1_mean, f2_mean) / denom).item()
            # Map correlation [-1, 1] to divergence [0, 1]
            return float(np.clip(1.0 - max(0.0, r), 0.0, 1.0))

        else:
            raise ValueError(f"Unknown divergence method: {self.method}")

    def compute_divergence_matrix(
        self,
        saliency_dict: Dict[str, torch.Tensor],
        branch_order: List[str] = ["B1", "B2", "B3", "B4", "B5"]
    ) -> np.ndarray:
        """
        Computes 5x5 pairwise divergence matrix for a single sample or mean maps.
        """
        n = len(branch_order)
        mat = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    mat[i, j] = 0.0
                else:
                    b1 = branch_order[i]
                    b2 = branch_order[j]
                    m1 = saliency_dict[b1]
                    m2 = saliency_dict[b2]
                    d = self.compute_pairwise_divergence(m1, m2)
                    mat[i, j] = d
                    mat[j, i] = d
        return mat

    def subset_divergence(self, subset: List[str], divergence_matrix: np.ndarray, branch_order: List[str] = ["B1", "B2", "B3", "B4", "B5"]) -> float:
        """Computes average pairwise divergence across all pairs in subset."""
        if len(subset) <= 1:
            return 0.0
        indices = [branch_order.index(b) for b in subset]
        total_d = 0.0
        pairs = 0
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                total_d += divergence_matrix[indices[i], indices[j]]
                pairs += 1
        return float(total_d / pairs) if pairs > 0 else 0.0
