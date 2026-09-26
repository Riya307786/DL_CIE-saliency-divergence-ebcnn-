import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

class WaferPreprocessor:
    """
    Standardized, reproducible wafer map preprocessor for EB-CNN.
    
    Adheres strictly to the EB-CNN reference (Abdullah et al., 2025):
    - Adjusts resolution to 224x224 for VGG16 branch compatibility and Grad-CAM spatial granularity (7x7 final feature map).
    - Preserves aspect ratio via symmetric zero-padding to keep circular wafer geometry undistorted.
    - Employs Nearest-Neighbor interpolation to preserve discrete ternary semantics {0, 128, 255}.
    - Normalizes by dividing by 255.0 to [0.0, 1.0].
    - Replicates 1-channel wafer map to 3 RGB channels (3, 224, 224) for VGG16 branch input.
    """
    def __init__(self, target_size=224, preserve_aspect_ratio=True, interpolation="nearest", normalize_div255=True):
        self.target_size = target_size
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.interpolation = interpolation.lower()
        self.normalize_div255 = normalize_div255

    def pad_to_square(self, img_arr):
        """Symmetrically pads non-square wafer maps with 0 (exterior background)."""
        h, w = img_arr.shape[:2]
        if h == w:
            return img_arr, (0, 0, 0, 0)
        max_dim = max(h, w)
        pad_top = (max_dim - h) // 2
        pad_bottom = max_dim - h - pad_top
        pad_left = (max_dim - w) // 2
        pad_right = max_dim - w - pad_left
        
        padded = np.pad(img_arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
        return padded, (pad_top, pad_bottom, pad_left, pad_right)

    def preprocess_numpy(self, wafer_map):
        """
        Processes a single 2D wafer map numpy array.
        Returns:
            processed_tensor: torch.FloatTensor of shape (3, target_size, target_size)
            metadata: dict with original and intermediate dimensions
        """
        orig_h, orig_w = wafer_map.shape[:2]
        
        if self.preserve_aspect_ratio:
            square_map, padding = self.pad_to_square(wafer_map)
        else:
            square_map = wafer_map
            padding = (0, 0, 0, 0)
            
        # Resizing using PIL
        pil_img = Image.fromarray(square_map)
        if self.interpolation == "nearest":
            resample_mode = Image.Resampling.NEAREST
        elif self.interpolation == "bilinear":
            resample_mode = Image.Resampling.BILINEAR
        elif self.interpolation == "bicubic":
            resample_mode = Image.Resampling.BICUBIC
        else:
            resample_mode = Image.Resampling.NEAREST
            
        resized_img = pil_img.resize((self.target_size, self.target_size), resample=resample_mode)
        resized_arr = np.array(resized_img, dtype=np.float32)
        
        # Normalization
        if self.normalize_div255:
            normalized_arr = resized_arr / 255.0
        else:
            normalized_arr = resized_arr
            
        # Channel handling: replicate to 3 channels (C, H, W)
        tensor = torch.from_numpy(normalized_arr).unsqueeze(0).repeat(3, 1, 1).float()
        
        metadata = {
            "orig_shape": (orig_h, orig_w),
            "square_shape": square_map.shape,
            "padding": padding,
            "target_size": (self.target_size, self.target_size),
            "interpolation": self.interpolation
        }
        return tensor, metadata

    def __call__(self, wafer_map):
        tensor, _ = self.preprocess_numpy(wafer_map)
        return tensor
