import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import traceback
import torch
torch.backends.mkldnn.enabled = False
torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

err_file = open("fit_error.txt", "w")

def log(msg):
    print(msg, flush=True)
    err_file.write(str(msg) + "\n")
    err_file.flush()

try:
    log("Testing trainer.fit with explicit file logger...")
    from src.data.dataset import get_dataloaders
    from src.models.ebcnn import EBCNN
    from src.training.trainer import EBCNNTrainer

    train_loader, val_loader, _ = get_dataloaders(subsample_frac=0.01)
    model = EBCNN(num_classes=9)
    trainer = EBCNNTrainer(model, checkpoint_dir="checkpoints_test")

    log("Calling trainer.fit...")
    res = trainer.fit(train_loader, val_loader, epochs=1, checkpoint_name="test.pt")
    log(f"SUCCESS! Fit completed cleanly! Res: {res}")
except Exception as e:
    log("Fit test error:")
    err_str = traceback.format_exc()
    log(err_str)
finally:
    err_file.close()
