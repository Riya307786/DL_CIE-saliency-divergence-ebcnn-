import os
import sys
import types
import pytest
import torch
import numpy as np
import pandas as pd

# Compatibility shim
if 'pandas.core.indexes.numeric' not in sys.modules:
    mod = types.ModuleType('pandas.core.indexes.numeric')
    mod.Int64Index = pd.Index
    sys.modules['pandas.core.indexes.numeric'] = mod

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.preprocessing import WaferPreprocessor
from src.data.dataset import WM811KDataset, CLASS_NAMES
from src.models.ebcnn import EBCNN

def test_preprocessor():
    prep = WaferPreprocessor(target_size=224, preserve_aspect_ratio=True, interpolation="nearest")
    # Simulate a 26x28 wafer map with ternary values
    dummy_map = np.zeros((26, 28), dtype=np.uint8)
    dummy_map[5:20, 5:22] = 128
    dummy_map[10:15, 10:15] = 255
    
    tensor, meta = prep.preprocess_numpy(dummy_map)
    assert tensor.shape == (3, 224, 224), f"Expected (3, 224, 224), got {tensor.shape}"
    assert tensor.dtype == torch.float32
    assert torch.all(tensor >= 0.0) and torch.all(tensor <= 1.0)
    # Check ternary preservation: only values near 0, 128/255, 1.0
    uvals = torch.unique(tensor)
    for u in uvals:
        diffs = [abs(u.item() - target) for target in [0.0, 128.0/255.0, 1.0]]
        assert min(diffs) < 1e-4, f"Found unexpected interpolated value: {u.item()}"

def test_dataset():
    data_path = os.path.join("dataset", "extracted", "train_split.pkl")
    if not os.path.exists(data_path):
        pytest.skip("train_split.pkl not found")
        
    ds = WM811KDataset(data_path, preprocessor=WaferPreprocessor(target_size=224), augment=True)
    assert len(ds) == 37348
    
    sample = ds[0]
    assert "image" in sample and sample["image"].shape == (3, 224, 224)
    assert "label" in sample and 0 <= sample["label"].item() < 9
    assert "type_name" in sample and sample["type_name"] in CLASS_NAMES
    assert "id" in sample

def test_ebcnn_forward_backward():
    model = EBCNN(num_classes=9, hidden_dim=256, dropout=0.2)
    model.eval()
    
    # Batch of 2 dummy images
    dummy_input = torch.randn(2, 3, 224, 224)
    outputs = model(dummy_input)
    
    assert set(outputs.keys()) == {"B1", "B2", "B3", "B4", "B5"}
    for b_name, logits in outputs.items():
        assert logits.shape == (2, 9), f"{b_name} logits shape {logits.shape} != (2, 9)"
        
    # Check combining rules
    combos = model.predict_stacked_combinations(outputs, rule="mean")
    assert set(combos.keys()) == {"C1", "C2", "C3", "C4", "C5"}
    for c_name, p in combos.items():
        assert p.shape == (2, 9)
        assert torch.allclose(p.sum(dim=-1), torch.ones(2), atol=1e-5)
        
    # Test backward pass
    model.train()
    total_loss = sum(logits.sum() for logits in model(dummy_input).values())
    total_loss.backward()
    
    # Check gradients exist in all 5 stages and heads
    assert model.stage1.convs[0].weight.grad is not None
    assert model.stage5.convs[0].weight.grad is not None
    assert model.head1.fc2.weight.grad is not None
    assert model.head5.fc2.weight.grad is not None
    
    param_counts = model.count_parameters()
    assert param_counts["total_parameters"] > 10_000_000
    print(f"\nEB-CNN Total Parameters: {param_counts['total_parameters']:,}")
