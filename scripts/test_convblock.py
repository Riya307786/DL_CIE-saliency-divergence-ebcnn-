import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    print("Testing ConvBlock directly...")
    from src.models.ebcnn import ConvBlock
    from src.data.dataset import get_dataloaders

    block = ConvBlock(3, 64, 2)
    print("1. Random tensor test...")
    dummy = torch.randn(32, 3, 224, 224)
    c, p = block(dummy)
    print("Random tensor output shape:", p.shape)

    print("2. Dataloader tensor test...")
    train_loader, _, _ = get_dataloaders(subsample_frac=0.01)
    batch = next(iter(train_loader))
    img = batch["image"]
    print("DataLoader img shape:", img.shape, "dtype:", img.dtype, "is_contiguous:", img.is_contiguous())

    c2, p2 = block(img)
    print("DataLoader image forward success! Shape:", p2.shape)
except Exception as e:
    print("ConvBlock error:")
    traceback.print_exc()
