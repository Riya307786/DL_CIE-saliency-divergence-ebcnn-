import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "2"
import sys
import io
import json
import base64
import types
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(2)
torch.backends.mkldnn.enabled = False
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Compatibility shim for pandas Int64Index unpickling
if 'pandas.core.indexes.numeric' not in sys.modules:
    mod = types.ModuleType('pandas.core.indexes.numeric')
    mod.Int64Index = pd.Index
    sys.modules['pandas.core.indexes.numeric'] = mod

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.preprocessing import WaferPreprocessor
from src.data.dataset import CLASS_NAMES
from src.models.ebcnn import EBCNN
from src.explainability.gradcam import BranchGradCAM
from src.branch_selection.divergence import SaliencyDivergence
from src.branch_selection.selector import BaselineEBCNNSelector, SaliencyDivergenceBranchSelector

from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="Saliency-Divergence Branch Selection API",
    description="Backend API serving real EB-CNN inference, Grad-CAM maps, saliency divergence, and branch selection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dist = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL: Optional[EBCNN] = None
PREPROCESSOR = WaferPreprocessor(target_size=224, preserve_aspect_ratio=True, interpolation="nearest")
DIVERGENCE_CALCULATOR = SaliencyDivergence(method="cosine")
TEST_SAMPLES_DF: Optional[pd.DataFrame] = None

def get_or_load_model() -> EBCNN:
    global MODEL
    if MODEL is None:
        MODEL = EBCNN(num_classes=9, hidden_dim=256, dropout=0.3)
        ckpt_dir = "checkpoints"
        ckpt_path = os.path.join(ckpt_dir, "ebcnn_seed_101.pt")
        if not os.path.exists(ckpt_path):
            # Check for any available checkpoint in checkpoints directory
            ckpts = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pt")] if os.path.exists(ckpt_dir) else []
            if ckpts:
                ckpt_path = ckpts[0]
        
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            MODEL.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded trained model checkpoint from {ckpt_path}")
        else:
            print("No checkpoint found. Operating with initialized model weights.")
            
        MODEL.to(DEVICE)
        MODEL.eval()
    return MODEL

def load_test_samples():
    global TEST_SAMPLES_DF
    if TEST_SAMPLES_DF is None:
        test_path = os.path.join("dataset", "extracted", "test_data.pkl")
        if os.path.exists(test_path):
            with open(test_path, "rb") as f:
                import pickle
                TEST_SAMPLES_DF = pickle.load(f)

def heatmap_to_base64(heatmap: np.ndarray, original_map: Optional[np.ndarray] = None) -> str:
    """Converts a 2D float heatmap array (0..1) into a base64 encoded PNG image string."""
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    ax.imshow(heatmap, cmap="jet", vmin=0.0, vmax=1.0)
    ax.axis("off")
    plt.tight_layout(pad=0)
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def wafer_map_to_base64(wafer_map: np.ndarray) -> str:
    """Converts a 2D wafer map numpy array into a base64 encoded PNG image."""
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    ax.imshow(wafer_map, cmap="inferno", vmin=0, vmax=255)
    ax.axis("off")
    plt.tight_layout(pad=0)
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

@app.on_event("startup")
def startup_event():
    get_or_load_model()
    load_test_samples()

from fastapi.responses import FileResponse

@app.get("/")
def read_root():
    index_path = os.path.join(frontend_dist, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Saliency-Divergence API Backend Running. UI bundle build pending."}

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "device": str(DEVICE),
        "model_loaded": MODEL is not None,
        "classes": CLASS_NAMES
    }

@app.get("/api/stats")
def get_experimental_stats():
    exp_summary_path = os.path.join("results", "experiments_summary.json")
    stat_report_path = os.path.join("results", "statistical_report.json")
    
    summary_data = {}
    stat_data = {}
    
    if os.path.exists(exp_summary_path):
        with open(exp_summary_path, "r") as f:
            summary_data = json.load(f)
            
    if os.path.exists(stat_report_path):
        with open(stat_report_path, "r") as f:
            stat_data = json.load(f)
            
    return {
        "experiments_summary": summary_data,
        "statistical_report": stat_data
    }

@app.get("/api/samples")
def get_sample_wafers(limit: int = 18):
    load_test_samples()
    if TEST_SAMPLES_DF is None:
        raise HTTPException(status_code=404, detail="Test dataset not found")
        
    # Group by failureCode to ensure balanced presentation
    sampled = TEST_SAMPLES_DF.groupby("failureCode", group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), max(1, limit // 9)), random_state=42)
    ).head(limit)
    
    samples_list = []
    for idx, row in sampled.iterrows():
        w_map = row["waferMap"]
        f_code = int(row["failureCode"])
        c_name = CLASS_NAMES[f_code]
        img_b64 = wafer_map_to_base64(w_map)
        
        samples_list.append({
            "sample_id": int(idx),
            "failure_code": f_code,
            "class_name": c_name,
            "dimensions": list(w_map.shape),
            "wafer_image_b64": img_b64
        })
        
    return {"samples": samples_list}

class InferenceRequest(BaseModel):
    sample_id: Optional[int] = None
    wafer_matrix: Optional[List[List[int]]] = None

@app.post("/api/predict")
def predict_wafer(
    sample_id: Optional[int] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    model = get_or_load_model()
    wafer_map = None
    true_label = None
    true_class_name = "Unknown"
    
    if file is not None:
        # Read file as image or npy
        contents = file.file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert("L")
            wafer_map = np.array(image, dtype=np.uint8)
        except Exception:
            try:
                wafer_map = np.load(io.BytesIO(contents))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to parse uploaded image: {str(e)}")
    elif sample_id is not None:
        load_test_samples()
        if TEST_SAMPLES_DF is not None and len(TEST_SAMPLES_DF) > 0:
            if sample_id in TEST_SAMPLES_DF.index:
                row = TEST_SAMPLES_DF.loc[sample_id]
            else:
                row = TEST_SAMPLES_DF.iloc[abs(sample_id) % len(TEST_SAMPLES_DF)]
            wafer_map = row["waferMap"]
            true_label = int(row["failureCode"])
            true_class_name = CLASS_NAMES[true_label]
        else:
            raise HTTPException(status_code=404, detail="Test dataset not loaded")
    else:
        # Fallback dummy sample
        wafer_map = np.zeros((26, 28), dtype=np.uint8)
        wafer_map[5:20, 5:22] = 128
        wafer_map[10:15, 10:15] = 255
        true_label = 1 # Donut
        true_class_name = "Donut"
        
    # Preprocess wafer map into (3, 224, 224) tensor
    tensor, meta = PREPROCESSOR.preprocess_numpy(wafer_map)
    input_batch = tensor.unsqueeze(0).to(DEVICE) # (1, 3, 224, 224)
    
    # 1. Forward Pass through EB-CNN
    with torch.no_grad():
        logits_dict = model(input_batch)
        probs_dict = {b: F.softmax(logits_dict[b], dim=-1)[0].cpu().numpy().tolist() for b in logits_dict}
        preds_dict = {b: int(np.argmax(probs_dict[b])) for b in probs_dict}
        top_conf_dict = {b: float(np.max(probs_dict[b])) for b in probs_dict}

    # 2. Grad-CAM Saliency Maps across B1..B5
    cam_engine = BranchGradCAM(model, target_size=(224, 224))
    pred_class = preds_dict["B5"] # Target class is predicted class of B5
    
    with torch.enable_grad():
        sal_dict = cam_engine.generate_saliency_maps(input_batch, target_class=pred_class)
        
    saliency_heatmaps_b64 = {}
    saliency_tensors = {}
    for b in ["B1", "B2", "B3", "B4", "B5"]:
        s_map = sal_dict[b][0].cpu()
        saliency_tensors[b] = s_map
        saliency_heatmaps_b64[b] = heatmap_to_base64(s_map.numpy())
        
    cam_engine.remove_hooks()

    # 3. Pairwise Spatial Saliency Divergence Matrix (5x5)
    div_mat = DIVERGENCE_CALCULATOR.compute_divergence_matrix(saliency_tensors)
    
    # 4. Fit Selectors on single sample / default criteria
    # Baseline Selector
    baseline_sel = BaselineEBCNNSelector(combining_rule="mean")
    baseline_fit = baseline_sel.fit(logits_dict, torch.tensor([pred_class]).to(DEVICE))
    base_preds, base_probs = baseline_sel.predict(logits_dict)
    
    # Proposed Selector
    # Provide single sample saliency tensor formatted (1, 224, 224)
    sal_batch_dict = {b: saliency_tensors[b].unsqueeze(0) for b in saliency_tensors}
    prop_sel = SaliencyDivergenceBranchSelector(divergence_metric="cosine", lambda_div=0.15)
    prop_fit = prop_sel.fit(logits_dict, torch.tensor([pred_class]).to(DEVICE), sal_batch_dict)
    prop_preds, prop_probs = prop_sel.predict(logits_dict)
    
    # Render wafer image b64
    wafer_img_b64 = wafer_map_to_base64(wafer_map)

    return {
        "true_label": true_label,
        "true_class_name": true_class_name,
        "original_dimensions": list(wafer_map.shape),
        "wafer_image_b64": wafer_img_b64,
        "per_branch_predictions": {
            b: {
                "predicted_class": preds_dict[b],
                "class_name": CLASS_NAMES[preds_dict[b]],
                "confidence": top_conf_dict[b],
                "probabilities": probs_dict[b]
            } for b in ["B1", "B2", "B3", "B4", "B5"]
        },
        "gradcam_heatmaps": saliency_heatmaps_b64,
        "divergence_matrix": div_mat.tolist(),
        "baseline_selection": {
            "best_combo": baseline_fit["best_combo"],
            "selected_branches": baseline_fit["selected_branches"],
            "predicted_class": int(base_preds[0]),
            "class_name": CLASS_NAMES[int(base_preds[0])],
            "confidence": float(torch.max(base_probs[0]).item()),
            "probabilities": base_probs[0].cpu().numpy().tolist()
        },
        "proposed_selection": {
            "selected_branches": prop_fit["selected_branches"],
            "predicted_class": int(prop_preds[0]),
            "class_name": CLASS_NAMES[int(prop_preds[0])],
            "confidence": float(torch.max(prop_probs[0]).item()),
            "probabilities": prop_probs[0].cpu().numpy().tolist(),
            "best_joint_score": float(prop_fit["best_joint_score"])
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
