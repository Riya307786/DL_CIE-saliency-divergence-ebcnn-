
import os
import sys
import types
import json
import pickle
import hashlib
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Fix compatibility for pickles saved with older pandas versions
if 'pandas.core.indexes.numeric' not in sys.modules:
    mod = types.ModuleType('pandas.core.indexes.numeric')
    mod.Int64Index = pd.Index
    sys.modules['pandas.core.indexes.numeric'] = mod

DATA_DIR = os.path.join("dataset", "extracted")
FIGURES_DIR = os.path.join("artifacts", "figures")
DATASET_ART_DIR = os.path.join("artifacts", "dataset")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(DATASET_ART_DIR, exist_ok=True)

PKL_FILES = [
    "train_data.pkl",
    "train_split.pkl",
    "val_data.pkl",
    "test_data.pkl",
    "train_1_split.pkl",
    "train_10_split.pkl",
    "train_20_split.pkl",
    "train_29_split.pkl"
]

def load_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

def analyze_df(name, df):
    file_path = os.path.join(DATA_DIR, name)
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    num_samples = len(df)
    cols = list(df.columns)
    col_dtypes = {col: str(df[col].dtype) for col in cols}
    null_counts = {col: int(df[col].isnull().sum()) for col in cols}
    
    # Class distribution
    has_type = 'failureType' in df.columns
    has_code = 'failureCode' in df.columns
    
    class_dist = {}
    code_dist = {}
    code_to_type = {}
    
    if has_type:
        type_counts = df['failureType'].value_counts().to_dict()
        class_dist = {str(k): int(v) for k, v in type_counts.items()}
    if has_code:
        c_counts = df['failureCode'].value_counts().to_dict()
        code_dist = {int(k): int(v) for k, v in c_counts.items()}
    if has_type and has_code:
        mapping = df.drop_duplicates(subset=['failureCode', 'failureType'])[['failureCode', 'failureType']]
        for _, row in mapping.iterrows():
            code_to_type[int(row['failureCode'])] = str(row['failureType'])
            
    # Wafer map inspection
    shapes = []
    unique_pixel_vals = set()
    sample_wafer_dtypes = set()
    for wm in df['waferMap'].head(200):
        if isinstance(wm, np.ndarray):
            shapes.append(wm.shape)
            unique_pixel_vals.update(np.unique(wm).tolist())
            sample_wafer_dtypes.add(str(wm.dtype))
        elif isinstance(wm, list):
            arr = np.array(wm)
            shapes.append(arr.shape)
            unique_pixel_vals.update(np.unique(arr).tolist())
            sample_wafer_dtypes.add(str(arr.dtype))

    # All shapes
    all_shapes = [wm.shape if hasattr(wm, 'shape') else (len(wm), len(wm[0])) for wm in df['waferMap']]
    heights = [s[0] for s in all_shapes]
    widths = [s[1] for s in all_shapes]
    
    index_list = df.index.tolist()
    index_set = set(index_list)
    is_index_unique = len(index_list) == len(index_set)

    return {
        "filename": name,
        "size_mb": round(file_size_mb, 2),
        "num_samples": num_samples,
        "columns": cols,
        "dtypes": col_dtypes,
        "null_counts": null_counts,
        "index_min": int(min(index_list)) if index_list else None,
        "index_max": int(max(index_list)) if index_list else None,
        "is_index_unique": is_index_unique,
        "class_distribution": class_dist,
        "code_distribution": code_dist,
        "code_to_type": code_to_type,
        "wafer_dtypes": list(sample_wafer_dtypes),
        "unique_pixel_values": sorted(list(unique_pixel_vals)),
        "height_min": int(min(heights)),
        "height_max": int(max(heights)),
        "width_min": int(min(widths)),
        "width_max": int(max(widths)),
        "height_mean": float(np.mean(heights)),
        "width_mean": float(np.mean(widths)),
        "unique_resolutions_count": len(set(all_shapes)),
        "indices": index_set
    }

def main():
    print("=" * 60)
    print("AUDITING DATASET FILES IN:", DATA_DIR)
    print("=" * 60)

    reports = {}
    loaded_dfs = {}

    for name in PKL_FILES:
        path = os.path.join(DATA_DIR, name)
        if not os.path.exists(path):
            print(f"WARNING: File {path} not found!")
            continue
        print(f"Loading and analyzing {name}...")
        df = load_pickle(path)
        loaded_dfs[name] = df
        rep = analyze_df(name, df)
        reports[name] = rep
        print(f"  -> Samples: {rep['num_samples']}, Size: {rep['size_mb']} MB, Index Unique: {rep['is_index_unique']}")

    # Set relationship and Overlap analysis
    print("\n--- Overlap and Set Relationship Analysis ---")
    splits = ["train_split.pkl", "val_data.pkl", "test_data.pkl"]
    for i in range(len(splits)):
        for j in range(i+1, len(splits)):
            s1, s2 = splits[i], splits[j]
            overlap = reports[s1]["indices"].intersection(reports[s2]["indices"])
            print(f"Index overlap between {s1} and {s2}: {len(overlap)} samples")

    if "train_data.pkl" in reports:
        train_data_indices = reports["train_data.pkl"]["indices"]
        union_splits = reports["train_split.pkl"]["indices"] | reports["val_data.pkl"]["indices"] | reports["test_data.pkl"]["indices"]
        print(f"Total samples in train_data.pkl: {len(train_data_indices)}")
        print(f"Total union of (train_split + val_data + test_data): {len(union_splits)}")
        print(f"Difference (train_data - union): {len(train_data_indices - union_splits)}")
        print(f"Difference (union - train_data): {len(union_splits - train_data_indices)}")
        
        train_split_in_train_data = reports["train_split.pkl"]["indices"].issubset(train_data_indices)
        val_in_train_data = reports["val_data.pkl"]["indices"].issubset(train_data_indices)
        test_in_train_data = reports["test_data.pkl"]["indices"].issubset(train_data_indices)
        print(f"train_split subset of train_data: {train_split_in_train_data}")
        print(f"val_data subset of train_data: {val_in_train_data}")
        print(f"test_data subset of train_data: {test_in_train_data}")

    # Inspect the progressive splits: train_1_split, train_10_split, etc.
    train_split_indices = reports["train_split.pkl"]["indices"]
    for prog in ["train_1_split.pkl", "train_10_split.pkl", "train_20_split.pkl", "train_29_split.pkl"]:
        if prog in reports:
            prog_indices = reports[prog]["indices"]
            is_sub = prog_indices.issubset(train_split_indices)
            ratio = len(prog_indices) / len(train_split_indices)
            print(f"{prog}: {len(prog_indices)} samples ({ratio*100:.2f}% of train_split), subset of train_split: {is_sub}")

    # Master class mapping
    master_mapping = reports["train_split.pkl"]["code_to_type"]
    print("\nMaster Class Mapping (failureCode -> failureType):")
    for code in sorted(master_mapping.keys()):
        print(f"  {code}: {master_mapping[code]}")

    # Clean reports for JSON dump (remove set of indices)
    json_summary = {}
    for k, v in reports.items():
        v_clean = dict(v)
        del v_clean["indices"]
        json_summary[k] = v_clean

    with open(os.path.join(DATASET_ART_DIR, "audit_summary.json"), "w") as f:
        json.dump(json_summary, f, indent=2)
    print(f"\nSaved audit summary JSON to {os.path.join(DATASET_ART_DIR, 'audit_summary.json')}")

    # Generate Visualizations
    generate_figures(loaded_dfs, master_mapping)

def generate_figures(loaded_dfs, code_to_type):
    print("\nGenerating audit visualization figures...")

    # 1. Class distributions comparison across train_split, val_data, test_data
    df_train = loaded_dfs["train_split.pkl"]
    df_val = loaded_dfs["val_data.pkl"]
    df_test = loaded_dfs["test_data.pkl"]

    train_counts = df_train['failureType'].value_counts()
    val_counts = df_val['failureType'].value_counts()
    test_counts = df_test['failureType'].value_counts()

    classes = sorted(list(set(train_counts.index) | set(val_counts.index) | set(test_counts.index)))

    x = np.arange(len(classes))
    width = 0.28

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width, [train_counts.get(c, 0) for c in classes], width, label=f'Train Split (N={len(df_train)})', color='#2b5c8f')
    ax.bar(x, [val_counts.get(c, 0) for c in classes], width, label=f'Val Data (N={len(df_val)})', color='#e28743')
    ax.bar(x + width, [test_counts.get(c, 0) for c in classes], width, label=f'Test Data (N={len(df_test)})', color='#2e8b57')

    ax.set_xlabel('Defect Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sample Count (Log Scale)', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.set_title('WM-811K Canonical Dataset Split - Class Distributions (Log-Scale)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha='right', fontsize=10, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
    plt.tight_layout()

    fig_path1 = os.path.join(FIGURES_DIR, "dataset_class_distribution.png")
    plt.savefig(fig_path1, dpi=300)
    plt.close()
    print(f"Saved: {fig_path1}")

    # 2. Representative Wafer Map Gallery (3 samples per class)
    # Target classes: Donut, Random, Edge-Ring, Edge-Loc, Center, Loc, Scratch, Near-full, none
    fig, axes = plt.subplots(len(classes), 4, figsize=(12, 2.5 * len(classes)))
    
    cmap = plt.colormaps['viridis']

    for i, c_name in enumerate(classes):
        samples = df_train[df_train['failureType'] == c_name].head(4)
        for j in range(4):
            ax = axes[i, j]
            if j < len(samples):
                wm = samples.iloc[j]['waferMap']
                im = ax.imshow(wm, cmap='viridis', interpolation='nearest')
                shape_str = f"{wm.shape[0]}x{wm.shape[1]}"
                if j == 0:
                    ax.set_ylabel(f"{c_name}\n({len(df_train[df_train['failureType']==c_name])})", fontsize=11, fontweight='bold')
                ax.set_title(f"Sample {j+1} [{shape_str}]", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle("Representative Wafer Defect Patterns Across Classes (WM-811K)", fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    fig_path2 = os.path.join(FIGURES_DIR, "wafer_samples_gallery.png")
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    print(f"Saved: {fig_path2}")

    # 3. Wafer Map Dimensions Scatter & Histogram
    all_shapes = [wm.shape for wm in df_train['waferMap']]
    h = [s[0] for s in all_shapes]
    w = [s[1] for s in all_shapes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.scatter(w, h, alpha=0.2, c='#1f77b4', edgecolors='none', s=20)
    ax1.set_xlabel('Width (pixels)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Height (pixels)', fontsize=11, fontweight='bold')
    ax1.set_title('Wafer Map Dimensions Distribution (Width vs Height)', fontsize=12, fontweight='bold')

    aspect_ratios = [width / height for width, height in zip(w, h)]
    ax2.hist(aspect_ratios, bins=30, color='#2ca02c', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Aspect Ratio (Width / Height)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('Aspect Ratio Distribution across Wafer Maps', fontsize=12, fontweight='bold')
    plt.tight_layout()

    fig_path3 = os.path.join(FIGURES_DIR, "wafer_dimensions_distribution.png")
    plt.savefig(fig_path3, dpi=300)
    plt.close()
    print(f"Saved: {fig_path3}")

    # 4. Spatially scattered vs localized defect focus: Donut & Random vs others
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for idx, focus_cls in enumerate(["Donut", "Random"]):
        f_samples = df_train[df_train['failureType'] == focus_cls].head(5)
        for j in range(5):
            ax = axes[idx, j]
            if j < len(f_samples):
                wm = f_samples.iloc[j]['waferMap']
                ax.imshow(wm, cmap='inferno', interpolation='nearest')
                ax.set_title(f"{focus_cls} #{j+1} ({wm.shape[0]}x{wm.shape[1]})", fontsize=10, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.suptitle("Target Spatially-Scattered Classes: Donut vs Random Patterns", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path4 = os.path.join(FIGURES_DIR, "spatial_scattered_defects.png")
    plt.savefig(fig_path4, dpi=300)
    plt.close()
    print(f"Saved: {fig_path4}")

if __name__ == "__main__":
    main()
