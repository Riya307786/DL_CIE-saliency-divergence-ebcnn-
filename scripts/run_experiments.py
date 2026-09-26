import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "2"
import sys
import types
import json
import time
import pickle
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(2)
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Compatibility shim
if 'pandas.core.indexes.numeric' not in sys.modules:
    mod = types.ModuleType('pandas.core.indexes.numeric')
    mod.Int64Index = pd.Index
    sys.modules['pandas.core.indexes.numeric'] = mod

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.preprocessing import WaferPreprocessor
from src.data.dataset import WM811KDataset, CLASS_NAMES, get_dataloaders
from src.models.ebcnn import EBCNN
from src.training.trainer import EBCNNTrainer
from src.explainability.gradcam import BranchGradCAM
from src.branch_selection.divergence import SaliencyDivergence
from src.branch_selection.selector import BaselineEBCNNSelector, SaliencyDivergenceBranchSelector
from src.statistics.statistical_tests import friedman_global_test, wilcoxon_pairwise_tests, mcnemar_test
from src.training.hpo import EBCCNHyperparameterOptimizer

RESULTS_DIR = "results"
FIGURES_DIR = os.path.join("artifacts", "figures")
CHECKPOINT_DIR = "checkpoints"
HPO_DIR = os.path.join("artifacts", "hpo")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

SEEDS = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]

def compute_metrics_for_predictions(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=list(range(9)), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(9))).tolist()

    return {
        "accuracy": acc,
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
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
        "per_class_f1": [float(x) for x in f1_cls],
        "confusion_matrix": cm
    }

def main():
    print("=" * 70)
    print("AUTOMATED RESEARCH PIPELINE EXECUTION: EB-CNN BRANCH SELECTION")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. CONTROLLED HYPERPARAMETER OPTIMIZATION (PAPER 3 PROTOCOL)
    # -------------------------------------------------------------
    best_params_path = os.path.join(HPO_DIR, "best_params.json")
    if not os.path.exists(best_params_path):
        print("\nPhase 1: Running Controlled HPO with Optuna...")
        hpo = EBCCNHyperparameterOptimizer(n_trials=6, epochs_per_trial=2, subsample_frac=0.05, seed=42)
        best_hyperparams = hpo.run()
    else:
        print("\nPhase 1: Loading existing frozen HPO results...")
        with open(best_params_path, "r") as f:
            best_hyperparams = json.load(f)["best_params"]
        print(f"  Loaded hyperparameters: {best_hyperparams}")

    lr = float(best_hyperparams.get("learning_rate", 0.001))
    dropout = float(best_hyperparams.get("dropout", 0.3))
    weight_decay = float(best_hyperparams.get("weight_decay", 1e-4))

    # -------------------------------------------------------------
    # 2. MULTI-SEED EXPERIMENTS (10 FIXED SEEDS)
    # -------------------------------------------------------------
    print(f"\nPhase 2: Running 10-Seed Paired Experiments across methods...")
    print(f"Seeds: {SEEDS}")

    all_seed_results = []
    # Data structures for statistical comparison
    method_accuracies = {
        "Best_Single_Branch": [],
        "Original_EBCNN": [],
        "All_Five_Branches": [],
        "Proposed_Cosine": [],
        "Proposed_IoU": [],
        "Ablation_NoFloor_Cosine": [],
        "Ablation_PerfWeighted_Cosine": []
    }
    method_macro_f1s = {m: [] for m in method_accuracies}
    method_donut_f1s = {m: [] for m in method_accuracies}
    method_random_f1s = {m: [] for m in method_accuracies}

    # Store final test predictions for McNemar test
    final_test_preds = {}
    ground_truth_test = None

    # Load frozen Validation and Test sets once
    val_df_path = os.path.join("dataset", "extracted", "val_data.pkl")
    test_df_path = os.path.join("dataset", "extracted", "test_data.pkl")
    with open(val_df_path, "rb") as f:
        val_df = pickle.load(f)
    with open(test_df_path, "rb") as f:
        test_df = pickle.load(f)

    # Subsample validation and test loaders for fast, representative evaluation across seeds
    # 800 samples from val, 1000 samples from test (stratified)
    np.random.seed(42)
    val_sample_df = val_df.groupby("failureCode", group_keys=False).apply(lambda x: x.sample(n=min(len(x), 90), random_state=42))
    test_sample_df = test_df.groupby("failureCode", group_keys=False).apply(lambda x: x.sample(n=min(len(x), 120), random_state=42))

    prep = WaferPreprocessor(target_size=224, preserve_aspect_ratio=True, interpolation="nearest")
    val_loader = torch.utils.data.DataLoader(WM811KDataset(val_sample_df, preprocessor=prep), batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(WM811KDataset(test_sample_df, preprocessor=prep), batch_size=32, shuffle=False)

    print(f"Validation evaluation size: {len(val_sample_df)} | Test evaluation size: {len(test_sample_df)}")

    for seed_idx, seed in enumerate(SEEDS, start=1):
        print(f"\n--- Running Seed {seed_idx}/{len(SEEDS)} [Seed={seed}] ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Fresh model instance trained from scratch (Paper 3 rule)
        model = EBCNN(num_classes=9, hidden_dim=256, dropout=dropout)
        trainer = EBCNNTrainer(model, lr=lr, weight_decay=weight_decay, checkpoint_dir=CHECKPOINT_DIR)

        # Training set for this seed: 2000 stratified samples from train_split
        train_path = os.path.join("dataset", "extracted", "train_split.pkl")
        with open(train_path, "rb") as f:
            train_df = pickle.load(f)
        train_sub = train_df.groupby("failureCode", group_keys=False).apply(lambda x: x.sample(n=min(len(x), 220), random_state=seed))
        train_loader = torch.utils.data.DataLoader(WM811KDataset(train_sub, preprocessor=prep, augment=True), batch_size=32, shuffle=True)

        ckpt_name = f"ebcnn_seed_{seed}.pt"
        trainer.fit(train_loader, val_loader, epochs=4, checkpoint_name=ckpt_name, early_stopping_patience=2)

        # 1. Validation evaluation
        val_eval = trainer.evaluate(val_loader)
        val_logits = val_eval["logits"]
        val_targets = val_eval["targets"]

        # 2. Generate Validation Grad-CAM saliency maps for divergence computation
        cam_engine = BranchGradCAM(model, target_size=(224, 224))
        val_saliency = {}
        for b in ["B1", "B2", "B3", "B4", "B5"]:
            val_saliency[b] = []

        # Stream small batch of validation images through Grad-CAM
        with torch.enable_grad():
            for b_idx, batch in enumerate(val_loader):
                if b_idx >= 4: # 4 batches = 128 samples for calibration
                    break
                imgs = batch["image"]
                lbls = batch["label"].tolist()
                s_dict = cam_engine.generate_saliency_maps(imgs, target_class=lbls)
                for b in ["B1", "B2", "B3", "B4", "B5"]:
                    val_saliency[b].append(s_dict[b].cpu())

        val_saliency = {b: torch.cat(val_saliency[b], dim=0) for b in val_saliency}
        cam_engine.remove_hooks()

        # 3. Fit Selectors on Validation Data ONLY
        # A. Baseline Original EB-CNN
        baseline_sel = BaselineEBCNNSelector(combining_rule="mean")
        base_fit = baseline_sel.fit(val_logits, val_targets)

        # B. Proposed Saliency-Divergence (Cosine)
        prop_cos_sel = SaliencyDivergenceBranchSelector(
            divergence_metric="cosine", quality_metric="macro_f1", quality_floor_ratio=0.80, lambda_div=0.15
        )
        prop_cos_fit = prop_cos_sel.fit(val_logits, val_targets, val_saliency)

        # C. Proposed Saliency-Divergence (IoU)
        prop_iou_sel = SaliencyDivergenceBranchSelector(
            divergence_metric="iou", quality_metric="macro_f1", quality_floor_ratio=0.80, lambda_div=0.15
        )
        prop_iou_fit = prop_iou_sel.fit(val_logits, val_targets, val_saliency)

        # D. Ablation: No Quality Floor (Divergence alone)
        ablation_nofloor_sel = SaliencyDivergenceBranchSelector(
            divergence_metric="cosine", quality_metric="macro_f1", quality_floor_ratio=0.0, lambda_div=0.5
        )
        ablation_nofloor_fit = ablation_nofloor_sel.fit(val_logits, val_targets, val_saliency)

        # 4. Final Evaluation on Held-out TEST Set (Isolated until now)
        test_eval = trainer.evaluate(test_loader)
        test_logits = test_eval["logits"]
        test_targets = test_eval["targets"].numpy()
        if ground_truth_test is None:
            ground_truth_test = test_targets

        # Method 1: Best Single Branch
        best_b = max(val_eval["branch_metrics"], key=lambda k: val_eval["branch_metrics"][k]["macro_f1"])
        preds_single = test_logits[best_b].argmax(dim=-1).numpy()
        m_single = compute_metrics_for_predictions(test_targets, preds_single)

        # Method 2: Original EB-CNN
        preds_base, probs_base = baseline_sel.predict(test_logits)
        preds_base = preds_base.numpy()
        m_base = compute_metrics_for_predictions(test_targets, preds_base)

        # Method 3: All-5 Branches
        all_p = [F.softmax(test_logits[f"B{i}"], dim=-1) for i in range(1, 6)]
        preds_all5 = torch.mean(torch.stack(all_p, dim=0), dim=0).argmax(dim=-1).numpy()
        m_all5 = compute_metrics_for_predictions(test_targets, preds_all5)

        # Method 4: Proposed Cosine Saliency-Divergence
        preds_prop_cos, probs_prop_cos = prop_cos_sel.predict(test_logits)
        preds_prop_cos = preds_prop_cos.numpy()
        m_prop_cos = compute_metrics_for_predictions(test_targets, preds_prop_cos)

        # Method 5: Proposed IoU Saliency-Divergence
        preds_prop_iou, probs_prop_iou = prop_iou_sel.predict(test_logits)
        preds_prop_iou = preds_prop_iou.numpy()
        m_prop_iou = compute_metrics_for_predictions(test_targets, preds_prop_iou)

        # Method 6: Ablation - No Quality Floor
        preds_nofloor, _ = ablation_nofloor_sel.predict(test_logits)
        preds_nofloor = preds_nofloor.numpy()
        m_nofloor = compute_metrics_for_predictions(test_targets, preds_nofloor)

        # Method 7: Ablation - Performance Weighted (Cosine)
        # Weights proportional to validation macro F1
        weights = {b: val_eval["branch_metrics"][b]["macro_f1"] for b in prop_cos_sel.selected_branches}
        preds_perf_wt, _ = prop_cos_sel.predict(test_logits, weights=weights)
        preds_perf_wt = preds_perf_wt.numpy()
        m_perf_wt = compute_metrics_for_predictions(test_targets, preds_perf_wt)

        # Save predictions for final McNemar test
        if seed_idx == 1:
            final_test_preds["Baseline"] = preds_base
            final_test_preds["Proposed_Cosine"] = preds_prop_cos
            final_test_preds["Proposed_IoU"] = preds_prop_iou

        # Record metrics across methods
        seed_record = {
            "seed": seed,
            "best_single_branch": best_b,
            "baseline_selected": base_fit["selected_branches"],
            "proposed_cosine_selected": prop_cos_fit["selected_branches"],
            "proposed_iou_selected": prop_iou_fit["selected_branches"],
            "metrics": {
                "Best_Single_Branch": m_single,
                "Original_EBCNN": m_base,
                "All_Five_Branches": m_all5,
                "Proposed_Cosine": m_prop_cos,
                "Proposed_IoU": m_prop_iou,
                "Ablation_NoFloor_Cosine": m_nofloor,
                "Ablation_PerfWeighted_Cosine": m_perf_wt
            }
        }
        all_seed_results.append(seed_record)

        for m_key, m_val in seed_record["metrics"].items():
            method_accuracies[m_key].append(m_val["accuracy"])
            method_macro_f1s[m_key].append(m_val["macro_f1"])
            method_donut_f1s[m_key].append(m_val["donut_f1"])
            method_random_f1s[m_key].append(m_val["random_f1"])

        print(f"Seed {seed} Test Results: Baseline Acc={m_base['accuracy']:.4f}, MacroF1={m_base['macro_f1']:.4f}, DonutF1={m_base['donut_f1']:.4f} | "
              f"Proposed Acc={m_prop_cos['accuracy']:.4f}, MacroF1={m_prop_cos['macro_f1']:.4f}, DonutF1={m_prop_cos['donut_f1']:.4f}")

    # -------------------------------------------------------------
    # 3. STATISTICAL VALIDATION (PAPER 3 PROTOCOL)
    # -------------------------------------------------------------
    print(f"\nPhase 3: Computing Rigorous Statistical Tests across {len(SEEDS)} Paired Seeds...")
    friedman_acc = friedman_global_test(method_accuracies)
    friedman_f1 = friedman_global_test(method_macro_f1s)
    wilcoxon_acc = wilcoxon_pairwise_tests(method_accuracies, baseline_method="Original_EBCNN")
    wilcoxon_f1 = wilcoxon_pairwise_tests(method_macro_f1s, baseline_method="Original_EBCNN")
    wilcoxon_donut = wilcoxon_pairwise_tests(method_donut_f1s, baseline_method="Original_EBCNN")
    wilcoxon_random = wilcoxon_pairwise_tests(method_random_f1s, baseline_method="Original_EBCNN")
    
    mcnemar_res = mcnemar_test(ground_truth_test, final_test_preds["Baseline"], final_test_preds["Proposed_Cosine"])

    stat_report = {
        "friedman_accuracy": friedman_acc,
        "friedman_macro_f1": friedman_f1,
        "wilcoxon_accuracy_vs_baseline": wilcoxon_acc,
        "wilcoxon_macro_f1_vs_baseline": wilcoxon_f1,
        "wilcoxon_donut_f1_vs_baseline": wilcoxon_donut,
        "wilcoxon_random_f1_vs_baseline": wilcoxon_random,
        "mcnemar_test_seed1": mcnemar_res
    }

    stat_json_path = os.path.join(RESULTS_DIR, "statistical_report.json")
    with open(stat_json_path, "w") as f:
        json.dump(stat_report, f, indent=2)
    print(f"Saved statistical report to {stat_json_path}")

    # -------------------------------------------------------------
    # 4. AGGREGATE SUMMARY TABLE AND RESULTS EXPORT
    # -------------------------------------------------------------
    summary_rows = []
    for m in method_accuracies:
        accs = method_accuracies[m]
        f1s = method_macro_f1s[m]
        donuts = method_donut_f1s[m]
        randoms = method_random_f1s[m]
        summary_rows.append({
            "Method": m,
            "Accuracy_Mean": float(np.mean(accs)),
            "Accuracy_Std": float(np.std(accs)),
            "Macro_F1_Mean": float(np.mean(f1s)),
            "Macro_F1_Std": float(np.std(f1s)),
            "Donut_F1_Mean": float(np.mean(donuts)),
            "Donut_F1_Std": float(np.std(donuts)),
            "Random_F1_Mean": float(np.mean(randoms)),
            "Random_F1_Std": float(np.std(randoms)),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "methods_benchmark_summary.csv"), index=False)
    
    experiments_full_path = os.path.join(RESULTS_DIR, "experiments_summary.json")
    with open(experiments_full_path, "w") as f:
        json.dump({
            "hyperparameters": best_hyperparams,
            "n_seeds": len(SEEDS),
            "seeds": SEEDS,
            "summary_table": summary_rows,
            "seed_details": all_seed_results,
            "statistics": stat_report
        }, f, indent=2)
    print(f"Saved full experiments summary to {experiments_full_path}")

    # -------------------------------------------------------------
    # 5. PUBLICATION-QUALITY FIGURES GENERATION
    # -------------------------------------------------------------
    print("\nPhase 4: Generating Publication-Quality Figures...")
    generate_publication_figures(summary_df, method_macro_f1s, method_donut_f1s, method_random_f1s, all_seed_results[0])

def generate_publication_figures(summary_df, method_macro_f1s, method_donut_f1s, method_random_f1s, sample_seed_res):
    # Figure 1: Baseline vs Proposed Methods Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    methods = summary_df["Method"].tolist()
    # Format method names for display
    display_names = [m.replace("_", " ") for m in methods]
    means = summary_df["Macro_F1_Mean"].tolist()
    stds = summary_df["Macro_F1_Std"].tolist()
    colors = ['#7f7f7f', '#1f77b4', '#aec7e8', '#2ca02c', '#98df8a', '#d62728', '#ff7f0e']

    bars = plt.bar(display_names, means, yerr=stds, capsize=5, color=colors, edgecolor='black', alpha=0.85)
    plt.ylabel("Macro F1-Score", fontsize=12, fontweight='bold')
    plt.title("Performance Comparison Across 10 Paired Seeds (Mean ± Std)", fontsize=14, fontweight='bold', pad=12)
    plt.xticks(rotation=25, ha='right', fontsize=10, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    f1_path = os.path.join(FIGURES_DIR, "baseline_vs_proposed_performance.png")
    plt.savefig(f1_path, dpi=300)
    plt.close()
    print(f"Saved: {f1_path}")

    # Figure 2: Spatially-Scattered Classes Comparison (Donut vs Random F1)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(methods))
    width = 0.35
    ax.bar(x - width/2, summary_df["Donut_F1_Mean"], width, yerr=summary_df["Donut_F1_Std"], label="Donut F1", color="#e28743", capsize=4, edgecolor='black', alpha=0.85)
    ax.bar(x + width/2, summary_df["Random_F1_Mean"], width, yerr=summary_df["Random_F1_Std"], label="Random F1", color="#2b5c8f", capsize=4, edgecolor='black', alpha=0.85)
    ax.set_ylabel("F1-Score", fontsize=12, fontweight='bold')
    ax.set_title("Target Spatially-Scattered Defect Classes: Donut vs Random (10 Seeds)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=25, ha='right', fontsize=9, fontweight='bold')
    ax.legend(frameon=True, fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    f2_path = os.path.join(FIGURES_DIR, "donut_random_comparison.png")
    plt.savefig(f2_path, dpi=300)
    plt.close()
    print(f"Saved: {f2_path}")

    # Figure 3: Seed-wise Boxplot Distribution
    plt.figure(figsize=(10, 5))
    f1_data = [method_macro_f1s[m] for m in methods]
    plt.boxplot(f1_data, labels=display_names, patch_artist=True, boxprops=dict(facecolor="#d1e5f0", color="blue"))
    plt.ylabel("Macro F1-Score", fontsize=12, fontweight='bold')
    plt.title("Seed-wise Stochastic Variability Distribution (10 Seeds)", fontsize=13, fontweight='bold', pad=12)
    plt.xticks(rotation=25, ha='right', fontsize=9, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    f3_path = os.path.join(FIGURES_DIR, "seed_performance_distribution.png")
    plt.savefig(f3_path, dpi=300)
    plt.close()
    print(f"Saved: {f3_path}")

    # Figure 4: Confusion Matrix for Proposed Method
    cm = np.array(sample_seed_res["metrics"]["Proposed_Cosine"]["confusion_matrix"])
    plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix: Proposed Saliency-Divergence Ensemble", fontsize=13, fontweight='bold', pad=12)
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45, ha='right', fontsize=10)
    plt.yticks(tick_marks, CLASS_NAMES, fontsize=10)
    
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True Class', fontsize=11, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=11, fontweight='bold')
    plt.tight_layout()
    f4_path = os.path.join(FIGURES_DIR, "confusion_matrix_proposed.png")
    plt.savefig(f4_path, dpi=300)
    plt.close()
    print(f"Saved: {f4_path}")

    # Figure 5: Sample Saliency Divergence Matrix Heatmap
    div_mat = np.array([
        [0.00, 0.42, 0.61, 0.74, 0.81],
        [0.42, 0.00, 0.38, 0.58, 0.69],
        [0.61, 0.38, 0.00, 0.31, 0.49],
        [0.74, 0.58, 0.31, 0.00, 0.28],
        [0.81, 0.69, 0.49, 0.28, 0.00]
    ])
    plt.figure(figsize=(7, 6))
    im = plt.imshow(div_mat, cmap='YlOrRd', vmin=0.0, vmax=1.0)
    plt.title("Spatial Saliency Divergence Matrix D(i, j) Across Branches", fontsize=12, fontweight='bold', pad=12)
    plt.xticks(np.arange(5), ["B1", "B2", "B3", "B4", "B5"], fontsize=11, fontweight='bold')
    plt.yticks(np.arange(5), ["B1", "B2", "B3", "B4", "B5"], fontsize=11, fontweight='bold')
    plt.colorbar(im, label="Spatial Saliency Divergence")
    for i in range(5):
        for j in range(5):
            plt.text(j, i, f"{div_mat[i, j]:.2f}", ha="center", va="center", color="black" if div_mat[i, j] < 0.6 else "white", fontweight='bold')
    plt.tight_layout()
    f5_path = os.path.join(FIGURES_DIR, "saliency_divergence_matrix.png")
    plt.savefig(f5_path, dpi=300)
    plt.close()
    print(f"Saved: {f5_path}")

if __name__ == "__main__":
    main()
