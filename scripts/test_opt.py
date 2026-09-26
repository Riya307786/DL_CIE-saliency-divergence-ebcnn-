import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    print("Testing optuna create_study...")
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    print("Study created:", study)
    
    def dummy_obj(trial):
        print("Trial start:", trial.number)
        return 0.5
        
    study.optimize(dummy_obj, n_trials=1)
    print("Study optimized!")
except Exception as e:
    print("Optuna test error:")
    traceback.print_exc()
