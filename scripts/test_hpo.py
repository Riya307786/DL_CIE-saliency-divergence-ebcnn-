import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log_file = open("hpo_debug.txt", "w")

def log(msg):
    print(msg)
    log_file.write(str(msg) + "\n")
    log_file.flush()
    sys.stdout.flush()

try:
    log("Testing HPO step 1...")
    from src.training.hpo import EBCCNHyperparameterOptimizer
    log("Testing HPO step 2: Instantiating optimizer...")
    hpo = EBCCNHyperparameterOptimizer(n_trials=2, epochs_per_trial=1, subsample_frac=0.01)
    log("Testing HPO step 3: Running Optuna study...")
    res = hpo.run()
    log(f"HPO Result: {res}")
except Exception as e:
    log("Error in HPO:")
    err = traceback.format_exc()
    log(err)
    sys.exit(1)
finally:
    log_file.close()
