import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "2"

import torch
torch.set_num_threads(2)

import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    print("Testing train batch iteration with OpenMP fixes...")
    from src.data.dataset import get_dataloaders
    from src.models.ebcnn import EBCNN
    from src.training.trainer import EBCNNTrainer

    train_loader, val_loader, _ = get_dataloaders(subsample_frac=0.01)
    model = EBCNN(num_classes=9)
    trainer = EBCNNTrainer(model)

    print("Running single epoch...")
    res = trainer.train_epoch(train_loader)
    print("Train epoch completed successfully! Res:", res)
except Exception as e:
    print("Batch test error:")
    traceback.print_exc()
