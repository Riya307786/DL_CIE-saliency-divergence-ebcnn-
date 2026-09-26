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
        branch_correct = {f"B{i}": 0 for i in range(1, 6)}
        total_samples = 0
        num_batches = 0

        for batch in train_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            batch_sz = labels.size(0)
            total_samples += batch_sz

            self.optimizer.zero_grad()
            logits_dict = self.model(images)

            # Multi-branch joint loss
            batch_loss = 0.0
            for b_name, logits in logits_dict.items():
                l_b = self.criterion(logits, labels)
                batch_loss += l_b
                branch_losses[f"loss_{b_name}"] += l_b.item()
                preds = logits.argmax(dim=-1)
                branch_correct[b_name] += (preds == labels).sum().item()

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += batch_loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        res = {"train_total_loss": avg_loss}
        for i in range(1, 6):
            b_name = f"B{i}"
            res[f"loss_{b_name}"] = branch_losses[f"loss_{b_name}"] / max(1, num_batches)
            res[f"acc_{b_name}"] = branch_correct[b_name] / max(1, total_samples)
        return res

    def evaluate(self, loader) -> Dict[str, Any]:
        """Evaluates all 5 branches on a given DataLoader."""
        self.model.eval()
        all_logits = {f"B{i}": [] for i in range(1, 6)}
        all_targets = []
        all_ids = []
        val_losses = {f"B{i}": 0.0 for i in range(1, 6)}
        num_batches = 0

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                ids = batch["id"]

                logits_dict = self.model(images)
                for b in all_logits:
                    all_logits[b].append(logits_dict[b].cpu())
                    l_b = self.criterion(logits_dict[b], labels).item()
                    val_losses[b] += l_b

                all_targets.append(labels.cpu())
                if isinstance(ids, torch.Tensor):
                    all_ids.extend(ids.tolist())
                elif isinstance(ids, (list, tuple)):
                    all_ids.extend([int(x) for x in ids])
                else:
                    all_ids.extend(list(ids))
                num_batches += 1

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
            c_prec, c_rec, c_f1, c_supp = precision_recall_fscore_support(targets_np, preds, average=None, labels=list(range(9)), zero_division=0)
            cm = confusion_matrix(targets_np, preds, labels=list(range(9))).tolist()
            
            branch_metrics[b] = {
                "loss": float(val_losses[b] / max(1, num_batches)),
                "accuracy": acc,
                "precision": float(prec),
                "recall": float(rec),
                "macro_f1": float(f1),
                "weighted_f1": float(w_f1),
                "donut_precision": float(c_prec[1]),
                "donut_recall": float(c_rec[1]),
                "donut_f1": float(c_f1[1]),          # Class 1: Donut
                "random_precision": float(c_prec[6]),
                "random_recall": float(c_rec[6]),
                "random_f1": float(c_f1[6]),         # Class 6: Random
                "nearfull_precision": float(c_prec[5]),
                "nearfull_recall": float(c_rec[5]),
                "nearfull_f1": float(c_f1[5]),       # Class 5: Near-full
                "per_class_precision": [float(x) for x in c_prec],
                "per_class_recall": [float(x) for x in c_rec],
                "per_class_f1": [float(x) for x in c_f1],
                "class_support": [int(x) for x in c_supp],
                "confusion_matrix": cm
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
        early_stopping_patience: int = 4,
        seed: int = 101
    ) -> Dict[str, Any]:
        """Full training loop across epochs with comprehensive checkpointing and metric logging."""
        history = []
        best_val_score = -1.0
        patience_counter = 0
        best_checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        current_lr = self.optimizer.param_groups[0]["lr"]

        print(f"Starting training on device: {self.device} for {epochs} epochs (Seed={seed}, LR={current_lr})...")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            ep_start = time.time()
            train_res = self.train_epoch(train_loader)
            val_eval = self.evaluate(val_loader)
            val_metrics = val_eval["branch_metrics"]

            # Use deepest branch B5 macro F1 for model selection
            val_score = val_metrics["B5"]["macro_f1"]
            self.scheduler.step(val_score)
            current_lr = self.optimizer.param_groups[0]["lr"]

            ep_time = time.time() - ep_start
            print(f"Epoch {epoch}/{epochs} [{ep_time:.1f}s, LR={current_lr:.6f}]: "
                  f"TrainLoss={train_res['train_total_loss']:.4f} | "
                  f"B5 ValAcc={val_metrics['B5']['accuracy']:.4f}, ValMacroF1={val_score:.4f}, "
                  f"DonutF1={val_metrics['B5']['donut_f1']:.4f}, RandomF1={val_metrics['B5']['random_f1']:.4f}")

            # Structure per-branch metrics for this epoch
            branch_epoch_summary = {}
            for i in range(1, 6):
                b_name = f"B{i}"
                branch_epoch_summary[b_name] = {
                    "train_loss": train_res.get(f"loss_{b_name}", 0.0),
                    "train_accuracy": train_res.get(f"acc_{b_name}", 0.0),
                    "val_loss": val_metrics[b_name]["loss"],
                    "val_accuracy": val_metrics[b_name]["accuracy"],
                    "val_precision": val_metrics[b_name]["precision"],
                    "val_recall": val_metrics[b_name]["recall"],
                    "val_macro_f1": val_metrics[b_name]["macro_f1"],
                    "val_weighted_f1": val_metrics[b_name]["weighted_f1"],
                    "donut_f1": val_metrics[b_name]["donut_f1"],
                    "random_f1": val_metrics[b_name]["random_f1"],
                    "nearfull_f1": val_metrics[b_name]["nearfull_f1"],
                }

            history_entry = {
                "epoch": epoch,
                "learning_rate": current_lr,
                "epoch_time": ep_time,
                "train_metrics": train_res,
                "val_metrics": val_metrics,
                "branch_summary": branch_epoch_summary
            }
            history.append(history_entry)

            if val_score > best_val_score:
                best_val_score = val_score
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "seed": seed,
                    "learning_rate": current_lr,
                    "model_state_dict": self.model.state_dict(),
                    "val_metrics": val_metrics,
                    "best_val_score": best_val_score,
                    "branch_summary": branch_epoch_summary,
                    "checkpoint_path": best_checkpoint_path
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

        # Persist branch training metrics json for UI and API consumption
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")
        os.makedirs(results_dir, exist_ok=True)
        branch_metrics_file = os.path.join(results_dir, "branch_training_metrics.json")
        try:
            with open(branch_metrics_file, "w") as f:
                json.dump({
                    "seed": seed,
                    "epochs_trained": len(history),
                    "best_val_macro_f1": best_val_score,
                    "total_training_duration": total_time,
                    "checkpoint_path": best_checkpoint_path,
                    "history": history,
                    "final_branch_metrics": val_metrics
                }, f, indent=2)
            print(f"Persisted branch training metrics to {branch_metrics_file}")
        except Exception as e:
            print(f"Warning: could not write branch training metrics: {e}")

        return {
            "history": history,
            "best_val_score": best_val_score,
            "total_time": total_time,
            "checkpoint_path": best_checkpoint_path
        }
