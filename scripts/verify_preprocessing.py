import os
import sys
import types
import json
import pickle
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Compatibility shim
if 'pandas.core.indexes.numeric' not in sys.modules:
    mod = types.ModuleType('pandas.core.indexes.numeric')
    mod.Int64Index = pd.Index
    sys.modules['pandas.core.indexes.numeric'] = mod

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.preprocessing import WaferPreprocessor

DATA_PATH = os.path.join("dataset", "extracted", "train_split.pkl")
FIGURES_DIR = os.path.join("artifacts", "figures")
DATASET_ART_DIR = os.path.join("artifacts", "dataset")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(DATASET_ART_DIR, exist_ok=True)

def measure_circularity(binary_wafer_mask):
    """Computes circularity metric 4*pi*Area / Perimeter^2."""
    from scipy.ndimage import binary_dilation, binary_erosion
    area = np.sum(binary_wafer_mask)
    if area == 0:
        return 0.0
    eroded = binary_erosion(binary_wafer_mask)
    boundary = binary_wafer_mask ^ eroded
    perimeter = np.sum(boundary)
    if perimeter == 0:
        return 0.0
    return float(4 * np.pi * area / (perimeter ** 2))

def evaluate_pipelines(samples):
    methods = {
        "Direct Bilinear": WaferPreprocessor(target_size=224, preserve_aspect_ratio=False, interpolation="bilinear"),
        "Direct Nearest": WaferPreprocessor(target_size=224, preserve_aspect_ratio=False, interpolation="nearest"),
        "Padded Bilinear": WaferPreprocessor(target_size=224, preserve_aspect_ratio=True, interpolation="bilinear"),
        "Padded Nearest (Proposed)": WaferPreprocessor(target_size=224, preserve_aspect_ratio=True, interpolation="nearest"),
    }
    
    results = {m: [] for m in methods}
    valid_ternary = {0.0, 128.0/255.0, 1.0}
    
    for idx, row in samples.iterrows():
        orig_map = row['waferMap']
        orig_wafer_mask = (orig_map > 0)
        orig_defect_mask = (orig_map == 255)
        
        orig_defect_ratio = np.sum(orig_defect_mask) / max(1, np.sum(orig_wafer_mask))
        orig_circ = measure_circularity(orig_wafer_mask)
        
        for m_name, prep in methods.items():
            tensor, meta = prep.preprocess_numpy(orig_map)
            img = tensor[0].numpy() # 224x224
            
            # 1. Ternary purity: fraction of pixels within epsilon=0.01 of {0.0, 128/255, 1.0}
            is_ternary = np.isclose(img, 0.0, atol=0.01) | np.isclose(img, 128.0/255.0, atol=0.01) | np.isclose(img, 1.0, atol=0.01)
            purity = float(np.mean(is_ternary))
            
            # 2. Defect ratio preservation
            # For nearest, defect is img == 1.0. For bilinear, defect is img > 0.75
            proc_wafer = (img > 0.1)
            proc_defect = (img > 0.75)
            proc_defect_ratio = np.sum(proc_defect) / max(1, np.sum(proc_wafer))
            ratio_diff = abs(proc_defect_ratio - orig_defect_ratio)
            
            # 3. Circularity preservation
            proc_circ = measure_circularity(proc_wafer)
            circ_diff = abs(proc_circ - orig_circ)
            
            # 4. Blur index: gradient spread
            gy, gx = np.gradient(img)
            grad_mag = np.sqrt(gx**2 + gy**2)
            # fraction of non-zero gradient pixels that are intermediate (neither 0 nor sharp transition)
            intermediate_grads = np.sum((grad_mag > 0.05) & (grad_mag < 0.4)) / max(1, np.sum(grad_mag > 0.05))
            
            results[m_name].append({
                "ternary_purity": purity,
                "defect_ratio_diff": ratio_diff,
                "circularity_diff": circ_diff,
                "intermediate_blur_fraction": float(intermediate_grads)
            })
            
    summary = {}
    for m_name, metrics in results.items():
        summary[m_name] = {
            "mean_ternary_purity": float(np.mean([m["ternary_purity"] for m in metrics])),
            "mean_defect_ratio_error": float(np.mean([m["defect_ratio_diff"] for m in metrics])),
            "mean_circularity_error": float(np.mean([m["circularity_diff"] for m in metrics])),
            "mean_blur_fraction": float(np.mean([m["intermediate_blur_fraction"] for m in metrics]))
        }
    return summary, methods

def main():
    print("=" * 60)
    print("PREPROCESSING VERIFICATION AND QUANTITATIVE AUDIT")
    print("=" * 60)
    
    with open(DATA_PATH, "rb") as f:
        df = pickle.load(f)
        
    print(f"Loaded train_split with {len(df)} samples.")
    
    # Stratified test sample: 200 samples across all 9 classes
    sample_df = df.groupby('failureType', group_keys=False).apply(lambda x: x.head(25))
    print(f"Evaluating {len(sample_df)} representative wafers across all classes...")
    
    summary, methods = evaluate_pipelines(sample_df)
    
    print("\n--- Quantitative Preprocessing Comparison ---")
    for m_name, s in summary.items():
        print(f"\n{m_name}:")
        print(f"  Ternary Purity:          {s['mean_ternary_purity']*100:.2f}% (higher is better)")
        print(f"  Defect Ratio Error:      {s['mean_defect_ratio_error']:.4f} (lower is better)")
        print(f"  Circularity Error:       {s['mean_circularity_error']:.4f} (lower is better)")
        print(f"  Intermediate Blur Frac:  {s['mean_blur_fraction']*100:.2f}% (lower is better)")
        
    with open(os.path.join(DATASET_ART_DIR, "preprocessing_verification.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved metrics to {os.path.join(DATASET_ART_DIR, 'preprocessing_verification.json')}")
    
    # Generate Visual Comparison Figure
    generate_comparison_figures(df, methods)

def generate_comparison_figures(df, methods):
    print("\nGenerating Preprocessing Verification Figures...")
    
    classes_to_show = ["Donut", "Random", "Scratch", "Edge-Ring", "Center"]
    fig, axes = plt.subplots(len(classes_to_show), 5, figsize=(15, 3.2 * len(classes_to_show)))
    
    col_names = ["Original (Native)", "Direct Bilinear", "Direct Nearest", "Padded Bilinear", "Padded Nearest (Proposed)"]
    for c_idx, col_name in enumerate(col_names):
        axes[0, c_idx].set_title(col_name, fontsize=12, fontweight='bold', pad=10)
        
    for r_idx, c_name in enumerate(classes_to_show):
        sample = df[df['failureType'] == c_name].iloc[0]
        orig_map = sample['waferMap']
        orig_h, orig_w = orig_map.shape
        
        # Col 0: Original
        ax0 = axes[r_idx, 0]
        ax0.imshow(orig_map, cmap='viridis', interpolation='nearest')
        ax0.set_ylabel(f"{c_name}\n({orig_h}x{orig_w})", fontsize=11, fontweight='bold')
        ax0.set_xticks([])
        ax0.set_yticks([])
        
        # Other cols
        for c_idx, (m_name, prep) in enumerate(methods.items(), start=1):
            ax = axes[r_idx, c_idx]
            tensor, meta = prep.preprocess_numpy(orig_map)
            img = tensor[0].numpy()
            ax.imshow(img, cmap='viridis', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            
    plt.suptitle("Preprocessing Pipeline Comparison: Preserving Spatial Pattern Fidelity (WM-811K -> 224x224)", 
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "preprocessing_comparison.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved: {fig_path}")

    # Generate 1D Cross-Section Intensity Profile (showing blur vs discrete edges)
    # Take a Donut sample
    donut_sample = df[df['failureType'] == 'Donut'].iloc[1]['waferMap']
    h, w = donut_sample.shape
    row_idx = h // 2
    
    prep_bilinear = methods["Padded Bilinear"]
    prep_nearest = methods["Padded Nearest (Proposed)"]
    
    tensor_bi, _ = prep_bilinear.preprocess_numpy(donut_sample)
    tensor_nn, _ = prep_nearest.preprocess_numpy(donut_sample)
    
    img_bi = tensor_bi[0].numpy()
    img_nn = tensor_nn[0].numpy()
    row_224 = 112
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Slice profile
    ax1.plot(img_bi[row_224, :], label='Padded Bilinear (Continuous Blurring)', color='#d95f02', linewidth=2)
    ax1.plot(img_nn[row_224, :], label='Padded Nearest (Discrete Step Preservation)', color='#1b9e77', linewidth=2, linestyle='--')
    ax1.set_xlabel('Pixel Column (x-axis at y=112)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Normalized Intensity [0.0 - 1.0]', fontsize=11, fontweight='bold')
    ax1.set_title('Cross-Section Intensity Profile (Donut Defect Ring)', fontsize=12, fontweight='bold')
    ax1.legend(frameon=True, fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Cumulative histogram of intensities
    ax2.hist(img_bi.flatten(), bins=50, alpha=0.6, label='Padded Bilinear', color='#d95f02', edgecolor='black')
    ax2.hist(img_nn.flatten(), bins=50, alpha=0.6, label='Padded Nearest (Proposed)', color='#1b9e77', edgecolor='black')
    ax2.set_yscale('log')
    ax2.set_xlabel('Pixel Intensity Value', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Pixel Count (Log Scale)', fontsize=11, fontweight='bold')
    ax2.set_title('Pixel Intensity Distribution (Log-Scale)', fontsize=12, fontweight='bold')
    ax2.legend(frameon=True, fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle("Impact of Interpolation on Ternary Wafer Map Representation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    profile_path = os.path.join(FIGURES_DIR, "interpolation_defect_profiles.png")
    plt.savefig(profile_path, dpi=300)
    plt.close()
    print(f"Saved: {profile_path}")

if __name__ == "__main__":
    main()
