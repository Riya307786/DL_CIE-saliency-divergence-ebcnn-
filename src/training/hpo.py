import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "2"
import json
import optuna
import pandas as pd
import numpy as np
import torch
torch.set_num_threads(2)
from typing import Dict, Any

from src.models.ebcnn import EBCNN
from src.data.dataset import get_dataloaders
from src.training.trainer import EBCNNTrainer

optuna.logging.set_verbosity(optuna.logging.WARNING)

class EBCCNHyperparameterOptimizer:
    """
    Controlled Hyperparameter Optimization following AlMuhaideb & Khan (2026).
    Tunes learning rate, dropout, and weight decay on the validation split with a fixed budget.
    """
    def __init__(
        self,
        data_dir: str = os.path.join("dataset", "extracted"),
        artifacts_dir: str = os.path.join("artifacts", "hpo"),
        n_trials: int = 8,
        epochs_per_trial: int = 3,
        subsample_frac: float = 0.08, # Subsample for fast, representative exploration
        seed: int = 42
    ):
        self.data_dir = data_dir
        self.artifacts_dir = artifacts_dir
        self.n_trials = n_trials
        self.epochs_per_trial = epochs_per_trial
        self.subsample_frac = subsample_frac
        self.seed = seed
        os.makedirs(self.artifacts_dir, exist_ok=True)

    def objective(self, trial: optuna.Trial) -> float:
        try:
            lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
            dropout = trial.suggest_float("dropout", 0.2, 0.6, step=0.1)
            weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

            print(f"  [HPO Trial {trial.number}] Parameters: lr={lr:.6f}, dropout={dropout:.2f}, weight_decay={weight_decay:.6f}")
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

            train_loader, val_loader, _ = get_dataloaders(
                data_dir=self.data_dir,
                batch_size=32,
                subsample_frac=self.subsample_frac,
                seed=self.seed
            )

            model = EBCNN(num_classes=9, hidden_dim=256, dropout=dropout)
            trainer = EBCNNTrainer(model, lr=lr, weight_decay=weight_decay, checkpoint_dir=self.artifacts_dir)

            fit_res = trainer.fit(
                train_loader,
                val_loader,
                epochs=self.epochs_per_trial,
                checkpoint_name=f"hpo_trial_{trial.number}.pt",
                early_stopping_patience=2
            )

            val_eval = trainer.evaluate(val_loader)
            b5_macro_f1 = val_eval["branch_metrics"]["B5"]["macro_f1"]
            print(f"  [HPO Trial {trial.number}] Complete. B5 Val Macro F1: {b5_macro_f1:.4f}")
            return float(b5_macro_f1)
        except Exception as e:
            import traceback
            print(f"ERROR in HPO Trial {trial.number}:")
            traceback.print_exc()
            raise e

    def run(self) -> Dict[str, Any]:
        print(f"\n{'='*60}\nRUNNING CONTROLLED HPO (Optuna: {self.n_trials} Trials)\n{'='*60}")
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(self.objective, n_trials=self.n_trials)

        best_params = study.best_params
        best_val = study.best_value
        print(f"\nHPO Completed. Best Trial {study.best_trial.number}:")
        print(f"  Params: {best_params}")
        print(f"  Best Val Macro F1: {best_val:.4f}")

        # Save trials DataFrame
        trials_df = study.trials_dataframe()
        csv_path = os.path.join(self.artifacts_dir, "trials.csv")
        trials_df.to_csv(csv_path, index=False)

        best_json_path = os.path.join(self.artifacts_dir, "best_params.json")
        with open(best_json_path, "w") as f:
            json.dump({
                "best_trial": study.best_trial.number,
                "best_value": best_val,
                "best_params": best_params,
                "n_trials": self.n_trials
            }, f, indent=2)

        summary_path = os.path.join(self.artifacts_dir, "hpo_summary.json")
        with open(summary_path, "w") as f:
            json.dump({
                "study_name": study.study_name,
                "best_params": best_params,
                "best_value": best_val,
                "trials_count": len(study.trials)
            }, f, indent=2)

        print(f"Saved HPO results to {self.artifacts_dir}")
        return best_params
