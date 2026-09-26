import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.backends.mkldnn.enabled = False
torch.set_num_threads(1)

import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    print("Testing FastAPI app startup & endpoints locally...")
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)
    
    print("1. Testing GET /api/health...")
    r1 = client.get("/api/health")
    print("  Status:", r1.status_code, "Body:", r1.json())
    assert r1.status_code == 200

    print("2. Testing GET /api/samples...")
    r2 = client.get("/api/samples?limit=3")
    print("  Status:", r2.status_code, "Num samples:", len(r2.json().get("samples", [])))
    assert r2.status_code == 200

    print("3. Testing POST /api/predict (sample_id=0)...")
    r3 = client.post("/api/predict", data={"sample_id": 0})
    print("  Status:", r3.status_code)
    pred_data = r3.json()
    print("  Baseline selected:", pred_data["baseline_selection"]["selected_branches"])
    print("  Proposed selected:", pred_data["proposed_selection"]["selected_branches"])
    print("  Grad-CAM keys:", list(pred_data["gradcam_heatmaps"].keys()))
    assert r3.status_code == 200

    print("\nSUCCESS! All API endpoints verified cleanly!")
except Exception as e:
    print("API test error:")
    traceback.print_exc()
