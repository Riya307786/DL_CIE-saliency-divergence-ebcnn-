import sys
import os
import traceback
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    print("Testing train batch iteration...")
    from src.data.dataset import get_dataloaders
    from src.models.ebcnn import EBCNN
    from src.training.trainer import EBCNNTrainer

    train_loader, val_loader, _ = get_dataloaders(subsample_frac=0.01)
    print(f"Train loader len: {len(train_loader)}")

    print("Fetching first batch...")
    batch = next(iter(train_loader))
    print("Batch keys:", batch.keys())
    print("Image shape:", batch["image"].shape)
    print("Label shape:", batch["label"].shape)

    model = EBCNN(num_classes=9)
    trainer = EBCNNTrainer(model)

    print("Running single epoch...")
    res = trainer.train_epoch(train_loader)
    print("Train epoch res:", res)
except Exception as e:
    print("Batch test error:")
    traceback.print_exc()
