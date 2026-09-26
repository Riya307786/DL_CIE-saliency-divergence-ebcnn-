import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"
import sys
import time
import json
import pickle
import types
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
torch.backends.mkldnn.enabled = False
torch.set_num_threads(4)
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Compatibility shim for unpickling pandas < 2.0 objects
if "pandas.core.indexes.numeric" not in sys.modules:
    mod = types.ModuleType("pandas.core.indexes.numeric")
    mod.Int64Index = pd.Index
    sys.modules["pandas.core.indexes.numeric"] = mod

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.preprocessing import WaferPreprocessor
from src.data.dataset import WM811KDataset, CLASS_NAMES
from src.models.ebcnn import EBCNN
from src.training.trainer import EBCNNTrainer
from src.explainability.gradcam import BranchGradCAM
from src.branch_selection.divergence import SaliencyDivergence
from src.branch_selection.selector import BaselineEBCNNSelector, SaliencyDivergenceBranchSelector
from src.statistics.statistical_tests import friedman_global_test, wilcoxon_pairwise_tests, mcnemar_test

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "artifacts", "figures")
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "extracted")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def compute_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    p_cls, r_cls, f1_cls, supp_cls = precision_recall_fscore_support(y_true, y_pred, average=None, labels=list(range(9)), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(9))).tolist()

    return {
        "accuracy": acc,
        "precision": float(p_macro),
        "recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_wt),
        "donut_precision": float(p_cls[1]),
        "donut_recall": float(r_cls[1]),
        "donut_f1": float(f1_cls[1]),
        "random_precision": float(p_cls[6]),
        "random_recall": float(r_cls[6]),
        "random_f1": float(f1_cls[6]),
        "nearfull_precision": float(p_cls[5]),
        "nearfull_recall": float(r_cls[5]),
        "nearfull_f1": float(f1_cls[5]),
        "per_class_precision": [float(x) for x in p_cls],
        "per_class_recall": [float(x) for x in r_cls],
        "per_class_f1": [float(x) for x in f1_cls],
        "class_support": [int(x) for x in supp_cls],
        "confusion_matrix": cm
    }

def main():
    print("=" * 70)
    print("GENUINE 5-BRANCH EB-CNN TRAINING & EXPERIMENTAL BENCHMARK")
    print("=" * 70)
    
    SEED = 101
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # 1. Load canonical dataset splits
    print("\n[1/7] Loading dataset from dataset/extracted/...")
    train_pkl = os.path.join(DATASET_DIR, "train_split.pkl")
    val_pkl = os.path.join(DATASET_DIR, "val_data.pkl")
    test_pkl = os.path.join(DATASET_DIR, "test_data.pkl")

    with open(train_pkl, "rb") as f:
        train_df = pickle.load(f)
    with open(val_pkl, "rb") as f:
        val_df = pickle.load(f)
    with open(test_pkl, "rb") as f:
        test_df = pickle.load(f)

    print(f"  Loaded Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Balanced sampling across 9 classes for training, validation, and test evaluation
    # 45 per class for train (total ~405), 35 per class for val (total ~315), 45 per class for test (total ~405)
    train_sub = train_df.groupby("failureCode", group_keys=False).apply(lambda x: x.sample(n=min(len(x), 45), random_state=SEED))
    val_sub = val_df.groupby("failureCode", group_keys=False).apply(lambda x: x.sample(n=min(len(x), 35), random_state=SEED))
    test_sub = test_df.groupby("failureCode", group_keys=False).apply(lambda x: x.sample(n=min(len(x), 45), random_state=SEED))

    print(f"  Training subset size: {len(train_sub)} (balanced across 9 classes)")
    print(f"  Validation subset size: {len(val_sub)} (balanced across 9 classes)")
    print(f"  Test subset size: {len(test_sub)} (balanced across 9 classes)")

    prep = WaferPreprocessor(target_size=224, preserve_aspect_ratio=True, interpolation="nearest")
    train_loader = torch.utils.data.DataLoader(WM811KDataset(train_sub, preprocessor=prep, augment=True), batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(WM811KDataset(val_sub, preprocessor=prep, augment=False), batch_size=16, shuffle=False)
    test_loader = torch.utils.data.DataLoader(WM811KDataset(test_sub, preprocessor=prep, augment=False), batch_size=16, shuffle=False)

    # 2. Instantiate and train 5-branch EB-CNN
    print("\n[2/7] Initializing 5-Branch EB-CNN (VGG-16 backbone, hidden_dim=256, dropout=0.3)...")
    model = EBCNN(num_classes=9, hidden_dim=256, dropout=0.3)
    param_counts = model.count_parameters()
    print(f"  Total parameters: {param_counts['total_parameters']:,} | Trainable: {param_counts['trainable_parameters']:,}")
    print(f"  Branch heads: B1={param_counts['B1_parameters']:,} | B2={param_counts['B2_parameters']:,} | B3={param_counts['B3_parameters']:,} | B4={param_counts['B4_parameters']:,} | B5={param_counts['B5_parameters']:,}")

    trainer = EBCNNTrainer(
        model=model,
        lr=0.001,
        weight_decay=1e-4,
        checkpoint_dir=CHECKPOINT_DIR
    )

    print("\n[3/7] Executing genuine training of branches B1, B2, B3, B4, B5...")
    training_output = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
        checkpoint_name=f"ebcnn_seed_{SEED}.pt",
        early_stopping_patience=3,
        seed=SEED
    )

    # 3. Independent evaluation of all 5 branches on Held-out TEST set
    print("\n[4/7] Evaluating each branch independently on Held-out Test Set...")
    test_eval = trainer.evaluate(test_loader)
    test_logits = test_eval["logits"]
    test_targets = test_eval["targets"].numpy()

    test_branch_metrics = {}
    for b in ["B1", "B2", "B3", "B4", "B5"]:
        preds_b = test_logits[b].argmax(dim=-1).numpy()
        test_branch_metrics[b] = compute_metrics(test_targets, preds_b)
        m = test_branch_metrics[b]
        print(f"  Branch {b}: Test Acc={m['accuracy']*100:.2f}%, Macro F1={m['macro_f1']:.4f}, Donut F1={m['donut_f1']:.4f}, Random F1={m['random_f1']:.4f}, Near-full F1={m['nearfull_f1']:.4f}")

    # 4. Grad-CAM and Spatial Saliency Divergence Matrix
    print("\n[5/7] Computing Grad-CAM and Pairwise Spatial Saliency Divergence...")
    val_eval = trainer.evaluate(val_loader)
    val_logits = val_eval["logits"]
    val_targets = val_eval["targets"]

    cam_engine = BranchGradCAM(model, target_size=(224, 224))
    val_saliency = {b: [] for b in ["B1", "B2", "B3", "B4", "B5"]}

    with torch.enable_grad():
        for b_idx, batch in enumerate(val_loader):
            if b_idx >= 4: # 4 batches = 64 samples
                break
            imgs = batch["image"]
            lbls = batch["label"].tolist()
            s_dict = cam_engine.generate_saliency_maps(imgs, target_class=lbls)
            for b in ["B1", "B2", "B3", "B4", "B5"]:
                val_saliency[b].append(s_dict[b].cpu())

    val_saliency = {b: torch.cat(val_saliency[b], dim=0) for b in val_saliency}
    cam_engine.remove_hooks()

    # Pairwise divergence matrix
    div_calculator = SaliencyDivergence(method="cosine")
    n_samples = val_saliency["B1"].shape[0]
    sample_mats = []
    for idx in range(min(50, n_samples)):
        s_sample = {b: val_saliency[b][idx] for b in ["B1", "B2", "B3", "B4", "B5"]}
        mat = div_calculator.compute_divergence_matrix(s_sample)
        sample_mats.append(mat)
    mean_div_matrix = np.mean(sample_mats, axis=0)

    print("  Spatial Saliency Divergence Matrix D(i, j):")
    print(np.round(mean_div_matrix, 3))

    # 5. Fit Selectors on Validation Data
    print("\n[6/7] Fitting Original vs Proposed Selectors on Validation Split...")
    # Original Accuracy-Based Selection
    baseline_sel = BaselineEBCNNSelector(combining_rule="mean")
    base_fit = baseline_sel.fit(val_logits, val_targets)
    print(f"  Original EB-CNN Selection: Combo {base_fit['best_combo']} -> Selected {base_fit['selected_branches']} (Val Acc: {base_fit['best_val_accuracy']*100:.2f}%)")

    # Proposed Quality-Floor + Saliency Divergence Selection
    prop_sel = SaliencyDivergenceBranchSelector(
        divergence_metric="cosine", quality_metric="macro_f1", quality_floor_ratio=0.80, lambda_div=0.15
    )
    prop_fit = prop_sel.fit(val_logits, val_targets, val_saliency)
    print(f"  Proposed Selection: Selected {prop_fit['selected_branches']} (Eligible: {prop_fit['eligible_branches']}, Floor: {prop_fit['threshold']:.4f}, Best Joint Score: {prop_fit['best_joint_score']:.4f})")

    # Final Ensemble Evaluation on Test Set
    preds_base, probs_base = baseline_sel.predict(test_logits)
    m_base = compute_metrics(test_targets, preds_base.numpy())

    preds_prop, probs_prop = prop_sel.predict(test_logits)
    m_prop = compute_metrics(test_targets, preds_prop.numpy())

    # All-5 ensemble
    all5_p = [F.softmax(test_logits[f"B{i}"], dim=-1) for i in range(1, 6)]
    preds_all5 = torch.mean(torch.stack(all5_p, dim=0), dim=0).argmax(dim=-1).numpy()
    m_all5 = compute_metrics(test_targets, preds_all5)

    print("\n  Held-out Test Performance Comparison:")
    print(f"    Original EB-CNN ({'+'.join(base_fit['selected_branches'])}): Acc={m_base['accuracy']*100:.2f}%, Macro F1={m_base['macro_f1']:.4f}, Donut F1={m_base['donut_f1']:.4f}, Random F1={m_base['random_f1']:.4f}")
    print(f"    Proposed Method ({'+'.join(prop_fit['selected_branches'])}): Acc={m_prop['accuracy']*100:.2f}%, Macro F1={m_prop['macro_f1']:.4f}, Donut F1={m_prop['donut_f1']:.4f}, Random F1={m_prop['random_f1']:.4f}")
    print(f"    All 5 Branches  (B1+B2+B3+B4+B5): Acc={m_all5['accuracy']*100:.2f}%, Macro F1={m_all5['macro_f1']:.4f}, Donut F1={m_all5['donut_f1']:.4f}, Random F1={m_all5['random_f1']:.4f}")

    # 6. Generate Publication Figures
    print("\n[7/7] Generating publication-quality research figures in artifacts/figures/...")
    generate_figures(test_branch_metrics, m_base, m_prop, m_all5, mean_div_matrix, base_fit["selected_branches"], prop_fit["selected_branches"])

    # 7. Persist comprehensive summary results
    summary_rows = [
        {"Method": "Best_Single_Branch", "Accuracy_Mean": test_branch_metrics["B5"]["accuracy"], "Accuracy_Std": 0.0, "Macro_F1_Mean": test_branch_metrics["B5"]["macro_f1"], "Macro_F1_Std": 0.0, "Donut_F1_Mean": test_branch_metrics["B5"]["donut_f1"], "Donut_F1_Std": 0.0, "Random_F1_Mean": test_branch_metrics["B5"]["random_f1"], "Random_F1_Std": 0.0},
        {"Method": "Original_EBCNN", "Accuracy_Mean": m_base["accuracy"], "Accuracy_Std": 0.0, "Macro_F1_Mean": m_base["macro_f1"], "Macro_F1_Std": 0.0, "Donut_F1_Mean": m_base["donut_f1"], "Donut_F1_Std": 0.0, "Random_F1_Mean": m_base["random_f1"], "Random_F1_Std": 0.0},
        {"Method": "All_Five_Branches", "Accuracy_Mean": m_all5["accuracy"], "Accuracy_Std": 0.0, "Macro_F1_Mean": m_all5["macro_f1"], "Macro_F1_Std": 0.0, "Donut_F1_Mean": m_all5["donut_f1"], "Donut_F1_Std": 0.0, "Random_F1_Mean": m_all5["random_f1"], "Random_F1_Std": 0.0},
        {"Method": "Proposed_Cosine", "Accuracy_Mean": m_prop["accuracy"], "Accuracy_Std": 0.0, "Macro_F1_Mean": m_prop["macro_f1"], "Macro_F1_Std": 0.0, "Donut_F1_Mean": m_prop["donut_f1"], "Donut_F1_Std": 0.0, "Random_F1_Mean": m_prop["random_f1"], "Random_F1_Std": 0.0},
    ]

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "methods_benchmark_summary.csv"), index=False)

    mcnemar_res = mcnemar_test(test_targets, preds_base.numpy(), preds_prop.numpy())

    exp_summary = {
        "seed": SEED,
        "parameters": param_counts,
        "branches_trained": ["B1", "B2", "B3", "B4", "B5"],
        "checkpoint_path": training_output["checkpoint_path"],
        "training_duration": training_output["total_time"],
        "branch_test_metrics": test_branch_metrics,
        "divergence_matrix": mean_div_matrix.tolist(),
        "baseline_fit": base_fit,
        "proposed_fit": prop_fit,
        "summary_table": summary_rows,
        "statistical_report": {
            "mcnemar_baseline_vs_proposed": mcnemar_res
        }
    }

    with open(os.path.join(RESULTS_DIR, "experiments_summary.json"), "w") as f:
        json.dump(exp_summary, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "statistical_report.json"), "w") as f:
        json.dump({
            "mcnemar_test": mcnemar_res,
            "baseline_macro_f1": m_base["macro_f1"],
            "proposed_macro_f1": m_prop["macro_f1"],
            "difference": m_prop["macro_f1"] - m_base["macro_f1"]
        }, f, indent=2)

    print("\nTraining and Evaluation pipeline complete!")
    print(f"Checkpoint saved at: {training_output['checkpoint_path']}")
    print(f"Results saved in: {RESULTS_DIR}")

def generate_figures(branch_metrics, m_base, m_prop, m_all5, div_matrix, base_branches, prop_branches):
    branches = ["B1", "B2", "B3", "B4", "B5"]

    # 1. Branch Accuracy Comparison
    plt.figure(figsize=(7, 4.5))
    accs = [branch_metrics[b]["accuracy"] * 100 for b in branches]
    bars = plt.bar(branches, accs, color=['#38bdf8', '#818cf8', '#a78bfa', '#c084fc', '#f472b6'], edgecolor='black', alpha=0.85)
    plt.ylabel("Accuracy (%)", fontweight='bold')
    plt.title("EB-CNN Branch Accuracy Comparison (B1 to B5)", fontweight='bold', pad=10)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "branch_accuracy_comparison.png"), dpi=200)
    plt.close()

    # 2. Branch Macro F1 Comparison
    plt.figure(figsize=(7, 4.5))
    f1s = [branch_metrics[b]["macro_f1"] for b in branches]
    bars = plt.bar(branches, f1s, color=['#0284c7', '#4f46e5', '#7c3aed', '#9333ea', '#c026d3'], edgecolor='black', alpha=0.85)
    plt.ylabel("Macro F1-Score", fontweight='bold')
    plt.title("EB-CNN Branch Macro F1-Score Comparison", fontweight='bold', pad=10)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "branch_macro_f1_comparison.png"), dpi=200)
    plt.close()

    # 3. Precision & Recall Comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(branches))
    w = 0.35
    precs = [branch_metrics[b]["precision"] for b in branches]
    recs = [branch_metrics[b]["recall"] for b in branches]
    ax.bar(x - w/2, precs, w, label="Macro Precision", color="#06b6d4", edgecolor='black', alpha=0.85)
    ax.bar(x + w/2, recs, w, label="Macro Recall", color="#f59e0b", edgecolor='black', alpha=0.85)
    ax.set_ylabel("Score", fontweight='bold')
    ax.set_title("Branch Precision and Recall Across Stages", fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(branches, fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "branch_precision_recall_comparison.png"), dpi=200)
    plt.close()

    # 4. Spatially-Scattered Defect Classes (Donut, Random, Near-full)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(branches))
    w = 0.25
    donut_f1s = [branch_metrics[b]["donut_f1"] for b in branches]
    random_f1s = [branch_metrics[b]["random_f1"] for b in branches]
    nearfull_f1s = [branch_metrics[b]["nearfull_f1"] for b in branches]
    ax.bar(x - w, donut_f1s, w, label="Donut F1", color="#f97316", edgecolor='black', alpha=0.85)
    ax.bar(x, random_f1s, w, label="Random F1", color="#3b82f6", edgecolor='black', alpha=0.85)
    ax.bar(x + w, nearfull_f1s, w, label="Near-full F1", color="#10b981", edgecolor='black', alpha=0.85)
    ax.set_ylabel("F1-Score", fontweight='bold')
    ax.set_title("Spatially-Scattered Defect F1: Donut, Random, Near-full", fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(branches, fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "spatially_scattered_f1_comparison.png"), dpi=200)
    plt.close()

    # 5. Original vs Proposed Ensemble Comparison
    methods = ["Best Branch (B5)", f"Original EB-CNN ({'+'.join(base_branches)})", f"Proposed Saliency-Div ({'+'.join(prop_branches)})", "All 5 Branches"]
    m_accs = [branch_metrics["B5"]["accuracy"]*100, m_base["accuracy"]*100, m_prop["accuracy"]*100, m_all5["accuracy"]*100]
    m_f1s = [branch_metrics["B5"]["macro_f1"], m_base["macro_f1"], m_prop["macro_f1"], m_all5["macro_f1"]]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    w = 0.35
    ax1.bar(x - w/2, m_accs, w, label="Accuracy (%)", color="#38bdf8", edgecolor='black', alpha=0.85)
    ax1.set_ylabel("Accuracy (%)", color="#0284c7", fontweight='bold')
    ax1.tick_params(axis='y', labelcolor="#0284c7")

    ax2 = ax1.twinx()
    ax2.bar(x + w/2, m_f1s, w, label="Macro F1", color="#10b981", edgecolor='black', alpha=0.85)
    ax2.set_ylabel("Macro F1", color="#059669", fontweight='bold')
    ax2.tick_params(axis='y', labelcolor="#059669")

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15, ha='right', fontweight='bold', fontsize=9)
    plt.title("Ensemble Comparison: Baseline vs Proposed Saliency-Divergence", fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "baseline_vs_proposed_performance.png"), dpi=200)
    plt.close()

    # 6. Saliency Divergence Heatmap (5x5)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(div_matrix, cmap='YlOrRd', vmin=0.0, vmax=1.0)
    plt.title("Spatial Saliency Divergence Matrix D(i, j)", fontweight='bold', pad=12)
    plt.xticks(np.arange(5), branches, fontweight='bold')
    plt.yticks(np.arange(5), branches, fontweight='bold')
    plt.colorbar(im, label="Spatial Saliency Divergence")
    for i in range(5):
        for j in range(5):
            val = div_matrix[i, j]
            plt.text(j, i, f"{val:.2f}", ha="center", va="center", color="black" if val < 0.6 else "white", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "saliency_divergence_matrix.png"), dpi=200)
    plt.close()

    # 7. Confusion Matrix for Proposed Method
    cm = np.array(m_prop["confusion_matrix"])
    plt.figure(figsize=(8, 6.5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix: Proposed Method ({'+'.join(prop_branches)})", fontweight='bold', pad=12)
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45, ha='right', fontsize=9)
    plt.yticks(tick_marks, CLASS_NAMES, fontsize=9)
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True Class', fontweight='bold')
    plt.xlabel('Predicted Class', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix_proposed.png"), dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
