import os
import sys
import types
import pytest
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

# Compatibility shim
if 'pandas.core.indexes.numeric' not in sys.modules:
    mod = types.ModuleType('pandas.core.indexes.numeric')
    mod.Int64Index = pd.Index
    sys.modules['pandas.core.indexes.numeric'] = mod

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.ebcnn import EBCNN
from src.explainability.gradcam import BranchGradCAM
from src.branch_selection.divergence import SaliencyDivergence
from src.branch_selection.selector import BaselineEBCNNSelector, SaliencyDivergenceBranchSelector

def test_gradcam_computation():
    model = EBCNN(num_classes=9, hidden_dim=128, dropout=0.1)
    cam_engine = BranchGradCAM(model, target_size=(224, 224))

    # Dummy batch of 2 images
    dummy_input = torch.randn(2, 3, 224, 224, requires_grad=True)
    saliency_dict = cam_engine.generate_saliency_maps(dummy_input, target_class=1) # Target Donut (1)

    assert set(saliency_dict.keys()) == {"B1", "B2", "B3", "B4", "B5"}
    for b, smap in saliency_dict.items():
        assert smap.shape == (2, 224, 224), f"Expected (2, 224, 224), got {smap.shape}"
        assert torch.all(smap >= 0.0) and torch.all(smap <= 1.0)

    cam_engine.remove_hooks()

def test_saliency_divergence_metrics():
    div_cos = SaliencyDivergence(method="cosine")
    div_iou = SaliencyDivergence(method="iou", binarize_threshold=0.5)

    # Identical maps -> Divergence should be 0
    map_a = torch.ones(224, 224)
    d_cos_ident = div_cos.compute_pairwise_divergence(map_a, map_a)
    d_iou_ident = div_iou.compute_pairwise_divergence(map_a, map_a)
    assert abs(d_cos_ident) < 1e-5
    assert abs(d_iou_ident) < 1e-5

    # Completely disjoint maps -> Divergence should be 1
    map1 = torch.zeros(224, 224)
    map1[:112, :] = 1.0
    map2 = torch.zeros(224, 224)
    map2[112:, :] = 1.0

    d_cos_disjoint = div_cos.compute_pairwise_divergence(map1, map2)
    d_iou_disjoint = div_iou.compute_pairwise_divergence(map1, map2)
    assert d_cos_disjoint > 0.99
    assert d_iou_disjoint > 0.99

    # Matrix symmetry and diagonal
    sal_dict = {
        "B1": map1,
        "B2": map2,
        "B3": (map1 + map2) / 2,
        "B4": map1 * 0.8,
        "B5": map2 * 0.5
    }
    mat = div_cos.compute_divergence_matrix(sal_dict)
    assert mat.shape == (5, 5)
    assert np.allclose(mat, mat.T)
    assert np.allclose(np.diag(mat), 0.0)

def test_branch_selectors():
    # Simulate validation logits for 100 samples
    N = 100
    targets = torch.randint(0, 9, (N,))
    
    # Simulate B5 being strong, B1 being weak
    val_logits = {
        "B1": torch.randn(N, 9),
        "B2": torch.randn(N, 9),
        "B3": torch.randn(N, 9),
        "B4": torch.randn(N, 9) + F.one_hot(targets, 9) * 2.0,
        "B5": torch.randn(N, 9) + F.one_hot(targets, 9) * 3.0,
    }

    # Baseline selector
    baseline = BaselineEBCNNSelector(combining_rule="mean")
    res_base = baseline.fit(val_logits, targets)
    assert "best_combo" in res_base
    assert len(res_base["selected_branches"]) >= 1

    test_preds, test_probs = baseline.predict(val_logits)
    assert test_preds.shape == (N,)
    assert test_probs.shape == (N, 9)

    # Proposed selector
    dummy_saliency = {b: torch.rand(N, 224, 224) for b in ["B1", "B2", "B3", "B4", "B5"]}
    proposed = SaliencyDivergenceBranchSelector(divergence_metric="cosine", lambda_div=0.2)
    res_prop = proposed.fit(val_logits, targets, dummy_saliency)

    assert "selected_branches" in res_prop
    assert len(res_prop["selected_branches"]) >= 2
    assert "divergence_matrix" in res_prop

    p_preds, p_probs = proposed.predict(val_logits)
    assert p_preds.shape == (N,)
    assert p_probs.shape == (N, 9)
