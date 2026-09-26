import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
torch.backends.mkldnn.enabled = False
torch.set_num_threads(1)
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    print("Testing ConvBlock with mkldnn.enabled = False...")
    from src.models.ebcnn import ConvBlock

    block = ConvBlock(3, 64, 2)
    print("1. Random tensor test...")
    dummy = torch.randn(8, 3, 224, 224)
    c, p = block(dummy)
    print("SUCCESS! Output shape:", p.shape)
except Exception as e:
    print("ConvBlock error:")
    traceback.print_exc()
