import os
import sys
import types
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

# Compatibility shim for older pandas pickles
if 'pandas.core.indexes.numeric' not in sys.modules:
    mod = types.ModuleType('pandas.core.indexes.numeric')
    mod.Int64Index = pd.Index
    sys.modules['pandas.core.indexes.numeric'] = mod

from src.data.preprocessing import WaferPreprocessor

CLASS_NAMES = [
    "Center",      # 0
    "Donut",       # 1
    "Edge-Loc",    # 2
    "Edge-Ring",   # 3
    "Loc",         # 4
    "Near-full",   # 5
    "Random",      # 6
    "Scratch",     # 7
    "none"         # 8
]

CODE_TO_NAME = {i: name for i, name in enumerate(CLASS_NAMES)}
NAME_TO_CODE = {name: i for i, name in enumerate(CLASS_NAMES)}

class WM811KDataset(Dataset):
    """
    PyTorch Dataset for WM-811K Wafer Map benchmark.
    """
    def __init__(self, data_or_path, preprocessor=None, augment=False):
        super().__init__()
        if isinstance(data_or_path, str):
            if not os.path.exists(data_or_path):
                raise FileNotFoundError(f"Dataset path not found: {data_or_path}")
            with open(data_or_path, "rb") as f:
                self.df = pickle.load(f)
        elif isinstance(data_or_path, pd.DataFrame):
            self.df = data_or_path.copy()
        else:
            raise ValueError(f"Expected file path or pandas DataFrame, got {type(data_or_path)}")

        self.preprocessor = preprocessor if preprocessor is not None else WaferPreprocessor()
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wafer_map = row['waferMap']
        label = int(row['failureCode'])
        type_str = str(row['failureType'])
        sample_id = int(self.df.index[idx])

        # Preprocess map into (3, 224, 224)
        tensor, meta = self.preprocessor.preprocess_numpy(wafer_map)

        # Data augmentation for training: rotation by 90-degree multiples and flips
        if self.augment:
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                tensor = TF.hflip(tensor)
            # Random vertical flip
            if torch.rand(1).item() > 0.5:
                tensor = TF.vflip(tensor)
            # Random orthogonal rotation (0, 90, 180, 270) to preserve discrete die grid
            rot_choice = torch.randint(0, 4, (1,)).item()
            if rot_choice == 1:
                tensor = torch.rot90(tensor, 1, [1, 2])
            elif rot_choice == 2:
                tensor = torch.rot90(tensor, 2, [1, 2])
            elif rot_choice == 3:
                tensor = torch.rot90(tensor, 3, [1, 2])

        return {
            "image": tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "type_name": type_str,
            "id": sample_id
        }

def get_dataloaders(
    data_dir=os.path.join("dataset", "extracted"),
    batch_size=32,
    num_workers=0,
    target_size=224,
    augment=True,
    subsample_frac=None,
    seed=42
):
    """
    Constructs canonical Train, Validation, and Test DataLoaders.
    Optionally allows stratified subsampling for faster debugging / testing.
    """
    preprocessor = WaferPreprocessor(target_size=target_size, preserve_aspect_ratio=True, interpolation="nearest")
    
    train_path = os.path.join(data_dir, "train_split.pkl")
    val_path = os.path.join(data_dir, "val_data.pkl")
    test_path = os.path.join(data_dir, "test_data.pkl")

    with open(train_path, "rb") as f:
        df_train = pickle.load(f)
    with open(val_path, "rb") as f:
        df_val = pickle.load(f)
    with open(test_path, "rb") as f:
        df_test = pickle.load(f)

    if subsample_frac is not None and subsample_frac < 1.0:
        np.random.seed(seed)
        df_train = df_train.groupby("failureCode", group_keys=False).apply(
            lambda x: x.sample(frac=subsample_frac, random_state=seed)
        )
        df_val = df_val.groupby("failureCode", group_keys=False).apply(
            lambda x: x.sample(frac=subsample_frac, random_state=seed)
        )
        df_test = df_test.groupby("failureCode", group_keys=False).apply(
            lambda x: x.sample(frac=subsample_frac, random_state=seed)
        )

    train_ds = WM811KDataset(df_train, preprocessor=preprocessor, augment=augment)
    val_ds = WM811KDataset(df_val, preprocessor=preprocessor, augment=False)
    test_ds = WM811KDataset(df_test, preprocessor=preprocessor, augment=False)

    use_pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin)

    return train_loader, val_loader, test_loader
