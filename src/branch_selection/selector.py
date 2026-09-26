import itertools
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import accuracy_score, f1_score

from src.branch_selection.divergence import SaliencyDivergence

class BaselineEBCNNSelector:
    """
    Original EB-CNN branch selection heuristic (Abdullah et al., 2025).
    Evaluates the five predefined stacked ensembles (C1 to C5) on validation data
    and selects the best model using:
        best_model = argmax(accuracy_k)
    """
    def __init__(self, combining_rule: str = "mean"):
        self.combining_rule = combining_rule.lower()
        self.best_combo_name: Optional[str] = None
        self.best_branches: List[str] = []
        self.val_scores: Dict[str, float] = {}

    def fit(self, val_logits_dict: Dict[str, torch.Tensor], val_targets: torch.Tensor) -> Dict[str, Any]:
        """
        Evaluates candidate stacked models C1..C5 on validation logits.
        """
        targets_np = val_targets.cpu().numpy()
        probs_dict = {k: F.softmax(v, dim=-1) for k, v in val_logits_dict.items()}

        candidate_combos = {
            "C1": ["B5"],
            "C2": ["B5", "B4"],
            "C3": ["B5", "B4", "B3"],
            "C4": ["B5", "B4", "B3", "B2"],
            "C5": ["B5", "B4", "B3", "B2", "B1"]
        }

        best_acc = -1.0
        best_name = "C1"
        combo_metrics = {}

        for c_name, branches in candidate_combos.items():
            b_probs = [probs_dict[b] for b in branches]
            if self.combining_rule == "mean":
                ens_p = torch.mean(torch.stack(b_probs, dim=0), dim=0)
            elif self.combining_rule == "max":
                ens_p, _ = torch.max(torch.stack(b_probs, dim=0), dim=0)
            elif self.combining_rule == "product":
                ens_p = b_probs[0]
                for p in b_probs[1:]:
                    ens_p = ens_p * p
                ens_p = ens_p / (torch.sum(ens_p, dim=-1, keepdim=True) + 1e-8)
            else:
                raise ValueError(f"Unknown rule: {self.combining_rule}")

            preds = ens_p.argmax(dim=-1).cpu().numpy()
            acc = float(accuracy_score(targets_np, preds))
            macro_f1 = float(f1_score(targets_np, preds, average="macro", zero_division=0))

            combo_metrics[c_name] = {
                "branches": branches,
                "accuracy": acc,
                "macro_f1": macro_f1
            }

            if acc > best_acc:
                best_acc = acc
                best_name = c_name

        self.best_combo_name = best_name
        self.best_branches = candidate_combos[best_name]
        self.val_scores = {k: v["accuracy"] for k, v in combo_metrics.items()}

        return {
            "best_combo": self.best_combo_name,
            "selected_branches": self.best_branches,
            "best_val_accuracy": best_acc,
            "all_combos": combo_metrics
        }

    def predict(self, logits_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference for selected baseline ensemble."""
        if not self.best_branches:
            raise RuntimeError("Selector must be fitted before predict.")
        probs = [F.softmax(logits_dict[b], dim=-1) for b in self.best_branches]
        if self.combining_rule == "mean":
            ens_prob = torch.mean(torch.stack(probs, dim=0), dim=0)
        elif self.combining_rule == "max":
            ens_prob, _ = torch.max(torch.stack(probs, dim=0), dim=0)
        elif self.combining_rule == "product":
            ens_prob = probs[0]
            for p in probs[1:]:
                ens_prob = ens_prob * p
            ens_prob = ens_prob / (torch.sum(ens_prob, dim=-1, keepdim=True) + 1e-8)
        preds = ens_prob.argmax(dim=-1)
        return preds, ens_prob


class SaliencyDivergenceBranchSelector:
    """
    Proposed Explainability-Driven Branch Selection:
    Combines a minimum predictive-quality floor with spatial saliency divergence.
    """
    def __init__(
        self,
        divergence_metric: str = "cosine",
        quality_metric: str = "macro_f1",
        quality_floor_ratio: float = 0.85, # Keep branches with performance >= 85% of top branch
        lambda_div: float = 0.15,          # Weight for saliency divergence
        combining_rule: str = "mean",
        min_branches: int = 2,
        max_branches: int = 4
    ):
        self.divergence_metric = divergence_metric
        self.quality_metric = quality_metric
        self.quality_floor_ratio = quality_floor_ratio
        self.lambda_div = lambda_div
        self.combining_rule = combining_rule
        self.min_branches = min_branches
        self.max_branches = max_branches
        
        self.divergence_calculator = SaliencyDivergence(method=divergence_metric)
        self.selected_branches: List[str] = []
        self.branch_quality_scores: Dict[str, float] = {}
        self.mean_divergence_matrix: Optional[np.ndarray] = None
        self.selection_log: Dict[str, Any] = {}

    def fit(
        self,
        val_logits_dict: Dict[str, torch.Tensor],
        val_targets: torch.Tensor,
        val_saliency_dict: Dict[str, torch.Tensor], # (N, 224, 224) for each branch
        branch_order: List[str] = ["B1", "B2", "B3", "B4", "B5"]
    ) -> Dict[str, Any]:
        """
        Executes proposed selection on validation data:
        1. Evaluate predictive quality of individual branches.
        2. Apply predictive quality floor.
        3. Compute validation saliency divergence matrix.
        4. Evaluate candidate subsets based on predictive performance and saliency divergence.
        5. Select the best branch subset.
        """
        targets_np = val_targets.cpu().numpy()
        branch_qualities = {}
        branch_accuracies = {}
        branch_macro_f1s = {}

        for b in branch_order:
            preds_b = val_logits_dict[b].argmax(dim=-1).cpu().numpy()
            acc = float(accuracy_score(targets_np, preds_b))
            mf1 = float(f1_score(targets_np, preds_b, average="macro", zero_division=0))
            branch_accuracies[b] = acc
            branch_macro_f1s[b] = mf1
            branch_qualities[b] = mf1 if self.quality_metric == "macro_f1" else acc

        self.branch_quality_scores = branch_qualities
        top_quality = max(branch_qualities.values())
        threshold = top_quality * self.quality_floor_ratio

        # Step 2: Quality Floor Filtering
        eligible_branches = [b for b, q in branch_qualities.items() if q >= threshold]
        if len(eligible_branches) < self.min_branches:
            # Fallback: take top min_branches
            sorted_b = sorted(branch_qualities.keys(), key=lambda k: branch_qualities[k], reverse=True)
            eligible_branches = sorted_b[:self.min_branches]

        # Step 3: Compute Validation Saliency Divergence Matrix
        # Compute pairwise divergence over a sample of validation maps
        n_samples = val_saliency_dict[branch_order[0]].shape[0]
        sample_indices = np.random.choice(n_samples, min(200, n_samples), replace=False)
        
        all_matrices = []
        for idx in sample_indices:
            sample_sal = {b: val_saliency_dict[b][idx] for b in branch_order}
            mat = self.divergence_calculator.compute_divergence_matrix(sample_sal, branch_order=branch_order)
            all_matrices.append(mat)
            
        self.mean_divergence_matrix = np.mean(all_matrices, axis=0)

        # Step 4: Evaluate Candidate Subsets
        probs_dict = {b: F.softmax(val_logits_dict[b], dim=-1) for b in branch_order}
        candidate_subsets = []
        
        for k in range(self.min_branches, min(self.max_branches + 1, len(eligible_branches) + 1)):
            for sub in itertools.combinations(eligible_branches, k):
                candidate_subsets.append(list(sub))

        subset_evaluations = []
        best_score = -1e9
        best_sub = eligible_branches

        for sub in candidate_subsets:
            # Ensemble predictions on val
            sub_probs = [probs_dict[b] for b in sub]
            ens_p = torch.mean(torch.stack(sub_probs, dim=0), dim=0)
            preds = ens_p.argmax(dim=-1).cpu().numpy()
            
            ens_acc = float(accuracy_score(targets_np, preds))
            ens_f1 = float(f1_score(targets_np, preds, average="macro", zero_division=0))
            
            # Saliency divergence of subset
            div_val = self.divergence_calculator.subset_divergence(sub, self.mean_divergence_matrix, branch_order)
            
            # Selection Objective: Joint Quality + Diversity
            # Score = Val_F1 + lambda_div * Divergence
            target_metric = ens_f1 if self.quality_metric == "macro_f1" else ens_acc
            joint_score = target_metric + self.lambda_div * div_val
            
            rec = {
                "subset": sub,
                "val_accuracy": ens_acc,
                "val_macro_f1": ens_f1,
                "divergence": div_val,
                "joint_score": joint_score
            }
            subset_evaluations.append(rec)
            
            if joint_score > best_score:
                best_score = joint_score
                best_sub = sub

        self.selected_branches = best_sub
        self.selection_log = {
            "eligible_branches": eligible_branches,
            "threshold": threshold,
            "selected_branches": self.selected_branches,
            "best_joint_score": best_score,
            "all_subsets": subset_evaluations,
            "branch_qualities": branch_qualities,
            "divergence_matrix": self.mean_divergence_matrix.tolist()
        }
        return self.selection_log

    def predict(self, logits_dict: Dict[str, torch.Tensor], weights: Optional[Dict[str, float]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ensemble inference using selected branches.
        Primary: equal weighting (mean rule).
        Ablation: optional performance weighting.
        """
        if not self.selected_branches:
            raise RuntimeError("Selector must be fitted before predict.")
            
        probs = [F.softmax(logits_dict[b], dim=-1) for b in self.selected_branches]
        
        if weights is not None:
            # Performance-weighted ablation
            w_tensor = torch.tensor([weights[b] for b in self.selected_branches], dtype=probs[0].dtype, device=probs[0].device)
            w_tensor = w_tensor / w_tensor.sum()
            stacked = torch.stack(probs, dim=0) # (num_b, B, num_classes)
            ens_prob = torch.sum(stacked * w_tensor.view(-1, 1, 1), dim=0)
        else:
            # Primary: Equal-weight mean rule
            ens_prob = torch.mean(torch.stack(probs, dim=0), dim=0)
            
        preds = ens_prob.argmax(dim=-1)
        return preds, ens_prob
