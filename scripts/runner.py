import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "2"
import torch
torch.backends.mkldnn.enabled = False
torch.set_num_threads(2)

class Tee:
    def __init__(self, filename):
        self.file = open(filename, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.file.flush()
        self.stdout.write(data)
        self.stdout.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()

tee = Tee("runner.log")
sys.stdout = tee
sys.stderr = tee

print("Starting scripts/run_experiments via runner.py...")

try:
    from scripts.run_experiments import main
    main()
    print("\nFINISHED RUNNER SUCCESSFULLY!")
except Exception as e:
    print("\nFATAL ERROR IN RUNNER:")
    traceback.print_exc()
    sys.exit(1)
