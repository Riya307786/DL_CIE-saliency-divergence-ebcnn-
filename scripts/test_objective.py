import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    print("Testing HPO objective directly...")
    import optuna
    from src.training.hpo import EBCCNHyperparameterOptimizer
    hpo = EBCCNHyperparameterOptimizer(n_trials=1, epochs_per_trial=1, subsample_frac=0.01)
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    print("Calling objective on trial...")
    val = hpo.objective(trial)
    print("Objective returned:", val)
except Exception as e:
    print("Direct objective error:")
    traceback.print_exc()
