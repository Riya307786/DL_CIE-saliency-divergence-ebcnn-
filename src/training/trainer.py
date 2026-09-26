import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "2"
import time
import json
import torch
torch.backends.mkldnn.enabled = False
torch.set_num_threads(2)
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from src.models.ebcnn import EBCNN
from src.data.dataset import CLASS_NAMES

class EBCNNTrainer:
    """
    Multi-Branch Trainer for EB-CNN architecture.
    """
    def __init__(
        self,
        model: EBCNN,
        device: Optional[torch.device] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        checkpoint_dir: str = "checkpoints"
    ):
        self.model = model
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5, patience=2)

    def train_epoch(self, train_loader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        branch_losses = {f"loss_B{i}": 0.0 for i in range(1, 6)}
        num_batches = 0

        for batch in train_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            logits_dict = self.model(images)

            # Multi-branch joint loss
            batch_loss = 0.0
            for b_name, logits in logits_dict.items():
                l_b = self.criterion(logits, labels)
                batch_loss += l_b
                branch_losses[f"loss_{b_name}"] += l_b.item()

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += batch_loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        res = {"train_total_loss": avg_loss}
        for k, v in branch_losses.items():
            res[k] = v / max(1, num_batches)
        return res

    def evaluate(self, loader) -> Dict[str, Any]:
        """Evaluates all 5 branches on a given DataLoader."""
        self.model.eval()
        all_logits = {f"B{i}": [] for i in range(1, 6)}
        all_targets = []
        all_ids = []

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                ids = batch["id"]

                logits_dict = self.model(images)
                for b in all_logits:
                    all_logits[b].append(logits_dict[b].cpu())
                all_targets.append(labels.cpu())
                if isinstance(ids, torch.Tensor):
                    all_ids.extend(ids.tolist())
                elif isinstance(ids, (list, tuple)):
                    all_ids.extend([int(x) for x in ids])
                else:
                    all_ids.extend(list(ids))

        # Concatenate
        logits_tensor = {b: torch.cat(all_logits[b], dim=0) for b in all_logits}
        targets_tensor = torch.cat(all_targets, dim=0)
        targets_np = targets_tensor.numpy()

        branch_metrics = {}
        for b, logit_t in logits_tensor.items():
            preds = logit_t.argmax(dim=-1).numpy()
            acc = float(accuracy_score(targets_np, preds))
            prec, rec, f1, _ = precision_recall_fscore_support(targets_np, preds, average="macro", zero_division=0)
            w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(targets_np, preds, average="weighted", zero_division=0)
            
            # Per-class metrics
            c_prec, c_rec, c_f1, _ = precision_recall_fscore_support(targets_np, preds, average=None, labels=list(range(9)), zero_division=0)
            
            branch_metrics[b] = {
                "accuracy": acc,
                "macro_precision": float(prec),
                "macro_recall": float(rec),
                "macro_f1": float(f1),
                "weighted_f1": float(w_f1),
                "donut_f1": float(c_f1[1]),      # Class 1: Donut
                "random_f1": float(c_f1[6]),     # Class 6: Random
                "nearfull_f1": float(c_f1[5]),   # Class 5: Near-full
                "per_class_f1": [float(x) for x in c_f1]
            }

        return {
            "branch_metrics": branch_metrics,
            "logits": logits_tensor,
            "targets": targets_tensor,
            "ids": all_ids
        }

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int = 10,
        checkpoint_name: str = "ebcnn_model.pt",
        early_stopping_patience: int = 4
    ) -> Dict[str, Any]:
        """Full training loop across epochs with checkpointing."""
        history = []
        best_val_score = -1.0
        patience_counter = 0
        best_checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

        print(f"Starting training on device: {self.device} for {epochs} epochs...")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            ep_start = time.time()
            train_res = self.train_epoch(train_loader)
            val_eval = self.evaluate(val_loader)
            val_metrics = val_eval["branch_metrics"]

            # Use deepest branch B5 macro F1 for model selection
            val_score = val_metrics["B5"]["macro_f1"]
            self.scheduler.step(val_score)

            ep_time = time.time() - ep_start
            print(f"Epoch {epoch}/{epochs} [{ep_time:.1f}s]: TrainLoss={train_res['train_total_loss']:.4f} | "
                  f"B5 ValAcc={val_metrics['B5']['accuracy']:.4f}, ValMacroF1={val_score:.4f}, "
                  f"DonutF1={val_metrics['B5']['donut_f1']:.4f}, RandomF1={val_metrics['B5']['random_f1']:.4f}")

            history_entry = {
                "epoch": epoch,
                "train_metrics": train_res,
                "val_metrics": val_metrics,
                "epoch_time": ep_time
            }
            history.append(history_entry)

            if val_score > best_val_score:
                best_val_score = val_score
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "val_metrics": val_metrics,
                    "best_val_score": best_val_score
                }, best_checkpoint_path)
                print(f"  --> Saved new best checkpoint to {best_checkpoint_path} (B5 Macro F1: {best_val_score:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.1f}s. Best B5 Val Macro F1: {best_val_score:.4f}")

        # Load best checkpoint before returning
        if os.path.exists(best_checkpoint_path):
            ckpt = torch.load(best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])

        return {
            "history": history,
            "best_val_score": best_val_score,
            "total_time": total_time,
            "checkpoint_path": best_checkpoint_path
        }
