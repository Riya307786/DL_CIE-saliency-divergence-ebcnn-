import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "2"
import sys
import io
import json
import base64
import pickle
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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
frontend_dist = os.path.join(BASE_DIR, "frontend", "dist")
if os.path.exists(frontend_dist) and os.path.exists(os.path.join(frontend_dist, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")

figures_dir = os.path.join(BASE_DIR, "artifacts", "figures")
os.makedirs(figures_dir, exist_ok=True)
app.mount("/figures", StaticFiles(directory=figures_dir), name="figures")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL: Optional[EBCNN] = None
LOADED_CKPT_PATH: Optional[str] = None
PREPROCESSOR = WaferPreprocessor(target_size=224, preserve_aspect_ratio=True, interpolation="nearest")
DIVERGENCE_CALCULATOR = SaliencyDivergence(method="cosine")
TEST_SAMPLES_DF: Optional[pd.DataFrame] = None
CACHED_SAMPLES_DICT: Dict[int, List[Dict[str, Any]]] = {}

def find_test_data_path() -> Optional[str]:
    candidates = [
        os.path.join(BASE_DIR, "dataset", "extracted", "test_data.pkl"),
        os.path.join(os.getcwd(), "dataset", "extracted", "test_data.pkl"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "extracted", "test_data.pkl"),
        os.path.join(BASE_DIR, "dataset", "test_data.pkl"),
        os.path.join(os.getcwd(), "test_data.pkl"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def find_best_checkpoint() -> Optional[str]:
    candidate_dirs = [
        os.path.join(BASE_DIR, "checkpoints"),
        os.path.join(os.getcwd(), "checkpoints"),
    ]
    for cdir in candidate_dirs:
        if os.path.exists(cdir):
            ckpts = [os.path.join(cdir, f) for f in os.listdir(cdir) if f.endswith(".pt")]
            if ckpts:
                seed_ckpts = [c for c in ckpts if "ebcnn_seed_101" in c]
                return seed_ckpts[0] if seed_ckpts else ckpts[0]
    return None

def get_or_load_model(force_reload: bool = False) -> EBCNN:
    global MODEL, LOADED_CKPT_PATH
    ckpt_path = find_best_checkpoint()

    # Determine if reload is necessary
    if MODEL is None or force_reload or (ckpt_path and ckpt_path != LOADED_CKPT_PATH):
        hidden_dim = 256
        state_dict_to_load = None
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location=DEVICE)
                state_dict_to_load = ckpt.get("model_state_dict", ckpt)
                if "head1.fc1.weight" in state_dict_to_load:
                    hidden_dim = state_dict_to_load["head1.fc1.weight"].shape[0]
            except Exception as e:
                print(f"Warning: could not read checkpoint {ckpt_path}: {e}")
                state_dict_to_load = None

        MODEL = EBCNN(num_classes=9, hidden_dim=hidden_dim, dropout=0.3)
        if state_dict_to_load:
            try:
                MODEL.load_state_dict(state_dict_to_load)
                LOADED_CKPT_PATH = ckpt_path
                print(f"Loaded trained model checkpoint from {ckpt_path} (hidden_dim={hidden_dim})")
            except Exception as e:
                print(f"Warning: load_state_dict failed, using default weights: {e}")
        else:
            print("No checkpoint found. Operating with initialized model weights.")
            
        MODEL.to(DEVICE)
        MODEL.eval()
    return MODEL

def load_test_samples():
    global TEST_SAMPLES_DF
    if TEST_SAMPLES_DF is None:
        p = find_test_data_path()
        if p and os.path.exists(p):
            with open(p, "rb") as f:
                TEST_SAMPLES_DF = pickle.load(f)
                print(f"Loaded test dataset from {p} with {len(TEST_SAMPLES_DF)} samples.")

def heatmap_to_base64(heatmap: np.ndarray) -> str:
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

@app.get("/")
def read_root():
    index_path = os.path.join(frontend_dist, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Saliency-Divergence API Backend Running. UI bundle build pending."}

@app.get("/api/health")
def health_check():
    ckpt_path = find_best_checkpoint()
    return {
        "status": "online",
        "device": str(DEVICE),
        "model_loaded": MODEL is not None,
        "checkpoint_loaded": LOADED_CKPT_PATH is not None,
        "checkpoint_path": LOADED_CKPT_PATH if LOADED_CKPT_PATH else (ckpt_path if ckpt_path else "None (Initialized weights)"),
        "classes": CLASS_NAMES
    }

@app.get("/api/training_status")
def get_training_status():
    metrics_path = os.path.join(BASE_DIR, "results", "branch_training_metrics.json")
    exp_path = os.path.join(BASE_DIR, "results", "experiments_summary.json")
    ckpt_path = find_best_checkpoint()

    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            data = json.load(f)
        final_m = data.get("final_branch_metrics", {})
        branches_status = {}
        for b in ["B1", "B2", "B3", "B4", "B5"]:
            bm = final_m.get(b, {})
            branches_status[b] = {
                "status": "TRAINED",
                "best_validation_accuracy": bm.get("accuracy", "Not yet evaluated"),
                "best_macro_f1": bm.get("macro_f1", "Not yet evaluated"),
                "best_validation_loss": bm.get("loss", "Not yet evaluated"),
                "epochs": data.get("epochs_trained", 3),
                "checkpoint": "available" if ckpt_path else "not found",
                "donut_f1": bm.get("donut_f1", "Not yet evaluated"),
                "random_f1": bm.get("random_f1", "Not yet evaluated"),
                "nearfull_f1": bm.get("nearfull_f1", "Not yet evaluated")
            }
        return {
            "training_completed": True,
            "seed": data.get("seed", 101),
            "epochs_trained": data.get("epochs_trained", 3),
            "total_duration_seconds": data.get("total_training_duration", 0),
            "checkpoint_path": data.get("checkpoint_path", ckpt_path),
            "branches": branches_status,
            "history": data.get("history", [])
        }
    else:
        # Untrained state
        branches_status = {}
        for b in ["B1", "B2", "B3", "B4", "B5"]:
            branches_status[b] = {
                "status": "Training not completed",
                "best_validation_accuracy": "Not yet evaluated",
                "best_macro_f1": "Not yet evaluated",
                "best_validation_loss": "Not yet evaluated",
                "epochs": 0,
                "checkpoint": "available" if ckpt_path else "not found",
                "donut_f1": "Not yet evaluated",
                "random_f1": "Not yet evaluated",
                "nearfull_f1": "Not yet evaluated"
            }
        return {
            "training_completed": False,
            "seed": 101,
            "epochs_trained": 0,
            "total_duration_seconds": 0,
            "checkpoint_path": ckpt_path if ckpt_path else "Not created",
            "branches": branches_status,
            "history": []
        }

@app.get("/api/branch_comparison")
def get_branch_comparison():
    exp_path = os.path.join(BASE_DIR, "results", "experiments_summary.json")
    metrics_path = os.path.join(BASE_DIR, "results", "branch_training_metrics.json")
    
    metrics = None
    if os.path.exists(exp_path):
        with open(exp_path, "r") as f:
            exp_data = json.load(f)
            metrics = exp_data.get("branch_test_metrics", None)
            
    if metrics is None and os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            t_data = json.load(f)
            metrics = t_data.get("final_branch_metrics", None)

    if metrics is None:
        return {
            "status": "Not yet evaluated",
            "rows": []
        }

    rows = []
    for b in ["B1", "B2", "B3", "B4", "B5"]:
        bm = metrics.get(b, {})
        rows.append({
            "branch": b,
            "accuracy": bm.get("accuracy", "Not yet evaluated"),
            "precision": bm.get("precision", "Not yet evaluated"),
            "recall": bm.get("recall", "Not yet evaluated"),
            "macro_f1": bm.get("macro_f1", "Not yet evaluated"),
            "weighted_f1": bm.get("weighted_f1", "Not yet evaluated"),
            "donut_f1": bm.get("donut_f1", "Not yet evaluated"),
            "random_f1": bm.get("random_f1", "Not yet evaluated"),
            "nearfull_f1": bm.get("nearfull_f1", "Not yet evaluated")
        })
    return {"status": "Evaluated", "rows": rows}

@app.get("/api/per_class_metrics")
def get_per_class_metrics():
    exp_path = os.path.join(BASE_DIR, "results", "experiments_summary.json")
    metrics_path = os.path.join(BASE_DIR, "results", "branch_training_metrics.json")
    
    metrics = None
    if os.path.exists(exp_path):
        with open(exp_path, "r") as f:
            metrics = json.load(f).get("branch_test_metrics", None)
    if metrics is None and os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f).get("final_branch_metrics", None)

    if metrics is None:
        return {"status": "Not yet evaluated", "per_branch": {}}

    res = {}
    for b in ["B1", "B2", "B3", "B4", "B5"]:
        bm = metrics.get(b, {})
        p_list = bm.get("per_class_precision", [None]*9)
        r_list = bm.get("per_class_recall", [None]*9)
        f_list = bm.get("per_class_f1", [None]*9)
        s_list = bm.get("class_support", [None]*9)
        classes_data = []
        for i, c_name in enumerate(CLASS_NAMES):
            classes_data.append({
                "class_name": c_name,
                "precision": p_list[i],
                "recall": r_list[i],
                "f1": f_list[i],
                "support": s_list[i]
            })
        res[b] = classes_data
    return {"status": "Evaluated", "per_branch": res}

@app.get("/api/graphs")
def get_research_graphs():
    fig_names = [
        "branch_accuracy_comparison.png",
        "branch_macro_f1_comparison.png",
        "branch_precision_recall_comparison.png",
        "spatially_scattered_f1_comparison.png",
        "baseline_vs_proposed_performance.png",
        "saliency_divergence_matrix.png",
        "confusion_matrix_proposed.png"
    ]
    graph_list = []
    for fn in fig_names:
        full_p = os.path.join(figures_dir, fn)
        graph_list.append({
            "name": fn.replace(".png", "").replace("_", " ").title(),
            "filename": fn,
            "url": f"/figures/{fn}",
            "available": os.path.exists(full_p)
        })
    return {"graphs": graph_list}

@app.get("/api/stats")
def get_experimental_stats():
    exp_summary_paths = [
        os.path.join(BASE_DIR, "results", "experiments_summary.json"),
        os.path.join(os.getcwd(), "results", "experiments_summary.json"),
    ]
    stat_report_paths = [
        os.path.join(BASE_DIR, "results", "statistical_report.json"),
        os.path.join(os.getcwd(), "results", "statistical_report.json"),
    ]
    summary_data = {}
    stat_data = {}
    for p in exp_summary_paths:
        if os.path.exists(p):
            with open(p, "r") as f:
                summary_data = json.load(f)
            break
    for p in stat_report_paths:
        if os.path.exists(p):
            with open(p, "r") as f:
                stat_data = json.load(f)
            break
            
    return {
        "experiments_summary": summary_data,
        "statistical_report": stat_data
    }

@app.get("/api/samples")
def get_sample_wafers(limit: int = 9):
    load_test_samples()
    if TEST_SAMPLES_DF is None or len(TEST_SAMPLES_DF) == 0:
        raise HTTPException(status_code=404, detail="Test dataset not found at dataset/extracted/test_data.pkl")
        
    if limit in CACHED_SAMPLES_DICT:
        return {"samples": CACHED_SAMPLES_DICT[limit]}
        
    # Balanced sampling across all 9 classes (Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch, none)
    sample_dfs = []
    per_class = max(1, limit // 9)
    for code in range(9):
        class_df = TEST_SAMPLES_DF[TEST_SAMPLES_DF["failureCode"] == code]
        if len(class_df) > 0:
            sample_dfs.append(class_df.sample(n=min(len(class_df), per_class), random_state=42))
            
    if sample_dfs:
        sampled = pd.concat(sample_dfs).head(limit)
    else:
        sampled = TEST_SAMPLES_DF.head(limit)
        
    samples_list = []
    for idx, row in sampled.iterrows():
        w_map = row["waferMap"]
        f_code = int(row["failureCode"])
        c_name = CLASS_NAMES[f_code] if f_code < len(CLASS_NAMES) else str(row.get("failureType", "Unknown"))
        img_b64 = wafer_map_to_base64(w_map)
        
        samples_list.append({
            "sample_id": int(idx),
            "failure_code": f_code,
            "class_name": c_name,
            "dimensions": list(w_map.shape),
            "wafer_image_b64": img_b64
        })
        
    CACHED_SAMPLES_DICT[limit] = samples_list
    return {"samples": samples_list}

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
        # Fallback sample (Donut)
        wafer_map = np.zeros((26, 28), dtype=np.uint8)
        wafer_map[5:20, 5:22] = 128
        wafer_map[10:15, 10:15] = 255
        true_label = 1
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

    # Retrieve branch historical metrics if available
    metrics_path = os.path.join(BASE_DIR, "results", "branch_training_metrics.json")
    saved_branch_metrics = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                saved_branch_metrics = json.load(f).get("final_branch_metrics", {})
        except Exception:
            pass

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
    sal_batch_dict = {b: saliency_tensors[b].unsqueeze(0) for b in saliency_tensors}
    prop_sel = SaliencyDivergenceBranchSelector(divergence_metric="cosine", lambda_div=0.15)
    prop_fit = prop_sel.fit(logits_dict, torch.tensor([pred_class]).to(DEVICE), sal_batch_dict)
    prop_preds, prop_probs = prop_sel.predict(logits_dict)
    
    # Render wafer image b64
    wafer_img_b64 = wafer_map_to_base64(wafer_map)

    # Calculate average pairwise divergence for selected branch subsets
    prop_sub = prop_fit["selected_branches"]
    prop_avg_div = DIVERGENCE_CALCULATOR.subset_divergence(prop_sub, div_mat)
    
    base_sub = baseline_fit["selected_branches"]
    base_avg_div = DIVERGENCE_CALCULATOR.subset_divergence(base_sub, div_mat)

    # Ensemble weights
    base_weights = {b: round(100.0 / len(base_sub), 2) for b in base_sub}
    prop_weights = {b: round(100.0 / len(prop_sub), 2) for b in prop_sub}

    # Step-by-step panel data
    return {
        "true_label": true_label,
        "true_class_name": true_class_name,
        "original_dimensions": list(wafer_map.shape),
        "wafer_image_b64": wafer_img_b64,
        "checkpoint_active": LOADED_CKPT_PATH is not None,
        "step_by_step": {
            "step_1_input": {
                "title": "STEP 1 — INPUT WAFER MAP",
                "description": f"Loaded wafer map of dimensions {list(wafer_map.shape)}. True defect category: {true_class_name}."
            },
            "step_2_branches": {
                "title": "STEP 2 — INDEPENDENT BRANCH PREDICTIONS",
                "description": "Each branch independently inspects the wafer map from coarse to fine scale."
            },
            "step_3_gradcam": {
                "title": "STEP 3 — GRAD-CAM SALIENCY HEATMAPS",
                "description": "Bright regions indicate areas that contributed more strongly to this branch's prediction."
            },
            "step_4_divergence": {
                "title": "STEP 4 — SPATIAL SALIENCY DIVERGENCE MATRIX (5×5)",
                "description": "Higher divergence means the selected branches focus on more different spatial regions. Divergence is not automatically good; branch predictive quality is also considered."
            },
            "step_5_original": {
                "title": "STEP 5 — ORIGINAL EB-CNN ACCURACY-BASED SELECTION",
                "description": "Original heuristic ranks predefined stacked combinations (C1..C5) on validation accuracy alone."
            },
            "step_6_proposed": {
                "title": "STEP 6 — PROPOSED QUALITY + SALIENCY-DIVERGENCE SELECTION",
                "description": "Proposed method ensures branches meet a predictive quality floor, then selects the combination that maximizes spatial complementarity."
            },
            "step_7_ensemble": {
                "title": "STEP 7 — ENSEMBLE PREDICTIONS",
                "description": "Equal-weight ensemble: each selected branch contributes equally."
            },
            "step_8_comparison": {
                "title": "STEP 8 — COMPARISON: ORIGINAL VS PROPOSED",
                "description": "Direct comparison of predictions, confidence, and spatial diversity."
            }
        },
        "per_branch_predictions": {
            b: {
                "branch": b,
                "predicted_class": preds_dict[b],
                "class_name": CLASS_NAMES[preds_dict[b]],
                "confidence": top_conf_dict[b],
                "probabilities": {CLASS_NAMES[i]: round(probs_dict[b][i], 4) for i in range(9)},
                "accuracy": saved_branch_metrics.get(b, {}).get("accuracy", "Not yet evaluated"),
                "macro_f1": saved_branch_metrics.get(b, {}).get("macro_f1", "Not yet evaluated")
            } for b in ["B1", "B2", "B3", "B4", "B5"]
        },
        "gradcam_heatmaps": saliency_heatmaps_b64,
        "divergence_matrix": div_mat.tolist(),
        "baseline_selection": {
            "method_name": "Original EB-CNN Accuracy-Based Selection",
            "best_combo": baseline_fit["best_combo"],
            "selected_branches": base_sub,
            "ensemble_weights": base_weights,
            "weights_explanation": "Equal-weight ensemble: each selected branch contributes equally.",
            "average_divergence": round(base_avg_div, 4),
            "predicted_class": int(base_preds[0]),
            "class_name": CLASS_NAMES[int(base_preds[0])],
            "confidence": float(torch.max(base_probs[0]).item()),
            "probabilities": {CLASS_NAMES[i]: round(float(base_probs[0][i].item()), 4) for i in range(9)}
        },
        "proposed_selection": {
            "method_name": "Proposed Quality + Saliency-Divergence Selection",
            "quality_floor": round(float(prop_fit["threshold"]), 4),
            "eligible_branches": prop_fit["eligible_branches"],
            "selected_branches": prop_sub,
            "ensemble_weights": prop_weights,
            "weights_explanation": "Equal-weight ensemble: each selected branch contributes equally.",
            "average_divergence": round(prop_avg_div, 4),
            "best_joint_score": round(float(prop_fit["best_joint_score"]), 4),
            "predicted_class": int(prop_preds[0]),
            "class_name": CLASS_NAMES[int(prop_preds[0])],
            "confidence": float(torch.max(prop_probs[0]).item()),
            "probabilities": {CLASS_NAMES[i]: round(float(prop_probs[0][i].item()), 4) for i in range(9)}
        },
        "comparison": {
            "original_prediction": CLASS_NAMES[int(base_preds[0])],
            "original_confidence": float(torch.max(base_probs[0]).item()),
            "original_branches": base_sub,
            "proposed_prediction": CLASS_NAMES[int(prop_preds[0])],
            "proposed_confidence": float(torch.max(prop_probs[0]).item()),
            "proposed_branches": prop_sub,
            "divergence_gain": round(prop_avg_div - base_avg_div, 4)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
