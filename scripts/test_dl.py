import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    print("Testing get_dataloaders...")
    from src.data.dataset import get_dataloaders
    tr, va, te = get_dataloaders(subsample_frac=0.05)
    print(f"Success! Train len: {len(tr.dataset)}, Val len: {len(va.dataset)}, Test len: {len(te.dataset)}")
except Exception as e:
    print("Error:")
    traceback.print_exc()
