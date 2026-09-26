import torch
torch.backends.mkldnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

class ConvBlock(nn.Module):
    """VGG-style Convolutional Block with 2 or 3 Conv-ReLU layers followed by MaxPool."""
    def __init__(self, in_channels: int, out_channels: int, num_convs: int):
        super().__init__()
        layers = []
        for i in range(num_convs):
            cin = in_channels if i == 0 else out_channels
            layers.append(nn.Conv2d(cin, out_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=False))
        self.convs = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Returns (features_before_pool, features_after_pool)
        conv_feats = self.convs(x)
        pooled = self.pool(conv_feats)
        return conv_feats, pooled

class BranchClassifier(nn.Module):
    """
    Branch classifier head attached after a convolutional stage pool.
    Contains: AdaptivePool -> Flatten -> FC -> BatchNorm -> ReLU -> Dropout -> Linear.
    """
    def __init__(self, in_channels: int, num_classes: int = 9, hidden_dim: int = 512, dropout: float = 0.5, pool_size: int = 7):
        super().__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.flatten = nn.Flatten()
        in_features = in_channels * pool_size * pool_size
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=False)
        self.drop = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.adaptive_pool(x)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.drop(out)
        logits = self.fc2(out)
        return logits

class EBCNN(nn.Module):
    """
    EB-CNN: Ensemble of Branch Convolutional Neural Network (Abdullah et al., 2025).
    
    Backbone: VGG-16 structured into 5 hierarchical stages:
      - Stage 1 (B1): 2x Conv(64), Pool -> Branch 1 Head
      - Stage 2 (B2): 2x Conv(128), Pool -> Branch 2 Head
      - Stage 3 (B3): 3x Conv(256), Pool -> Branch 3 Head
      - Stage 4 (B4): 3x Conv(512), Pool -> Branch 4 Head
      - Stage 5 (B5): 3x Conv(512), Pool -> Branch 5 Head
    
    Each branch classifier extracts representations from coarse to fine scale.
    """
    def __init__(self, num_classes: int = 9, hidden_dim: int = 512, dropout: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        
        # 5 VGG Stages
        self.stage1 = ConvBlock(in_channels=3, out_channels=64, num_convs=2)
        self.stage2 = ConvBlock(in_channels=64, out_channels=128, num_convs=2)
        self.stage3 = ConvBlock(in_channels=128, out_channels=256, num_convs=3)
        self.stage4 = ConvBlock(in_channels=256, out_channels=512, num_convs=3)
        self.stage5 = ConvBlock(in_channels=512, out_channels=512, num_convs=3)

        # 5 Branch Classifiers
        self.head1 = BranchClassifier(64, num_classes=num_classes, hidden_dim=hidden_dim, dropout=dropout)
        self.head2 = BranchClassifier(128, num_classes=num_classes, hidden_dim=hidden_dim, dropout=dropout)
        self.head3 = BranchClassifier(256, num_classes=num_classes, hidden_dim=hidden_dim, dropout=dropout)
        self.head4 = BranchClassifier(512, num_classes=num_classes, hidden_dim=hidden_dim, dropout=dropout)
        self.head5 = BranchClassifier(512, num_classes=num_classes, hidden_dim=hidden_dim, dropout=dropout)

        # Storage for Grad-CAM hooks
        self.branch_conv_features: Dict[str, torch.Tensor] = {}
        self.branch_conv_gradients: Dict[str, torch.Tensor] = {}
        self._hooks = []

    def get_last_conv_layer(self, branch_idx: int) -> nn.Module:
        """Returns the last convolutional layer for a given branch (1 to 5)."""
        stages = [self.stage1, self.stage2, self.stage3, self.stage4, self.stage5]
        stage = stages[branch_idx - 1]
        # The convs Sequential contains [Conv, BN, ReLU, Conv, BN, ReLU, ...]
        # Find the last Conv2d
        last_conv = None
        for layer in stage.convs:
            if isinstance(layer, nn.Conv2d):
                last_conv = layer
        return last_conv

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        conv1, pool1 = self.stage1(x)
        out1 = self.head1(pool1)

        conv2, pool2 = self.stage2(pool1)
        out2 = self.head2(pool2)

        conv3, pool3 = self.stage3(pool2)
        out3 = self.head3(pool3)

        conv4, pool4 = self.stage4(pool3)
        out4 = self.head4(pool4)

        conv5, pool5 = self.stage5(pool4)
        out5 = self.head5(pool5)

        return {
            "B1": out1,
            "B2": out2,
            "B3": out3,
            "B4": out4,
            "B5": out5
        }

    def predict_stacked_combinations(self, logits_dict: Dict[str, torch.Tensor], rule: str = "mean") -> Dict[str, torch.Tensor]:
        """
        Computes the five stacked branch classifier combinations defined in Abdullah et al. (Fig 2):
          C1: B5
          C2: Ensemble {B5, B4}
          C3: Ensemble {B5, B4, B3}
          C4: Ensemble {B5, B4, B3, B2}
          C5: Ensemble {B5, B4, B3, B2, B1}
        """
        probs = {k: F.softmax(v, dim=-1) for k, v in logits_dict.items()}
        
        combinations = {
            "C1": ["B5"],
            "C2": ["B5", "B4"],
            "C3": ["B5", "B4", "B3"],
            "C4": ["B5", "B4", "B3", "B2"],
            "C5": ["B5", "B4", "B3", "B2", "B1"]
        }

        combo_probs = {}
        for c_name, branch_list in combinations.items():
            branch_p = [probs[b] for b in branch_list]
            if rule == "mean":
                stacked_p = torch.mean(torch.stack(branch_p, dim=0), dim=0)
            elif rule == "max":
                stacked_p, _ = torch.max(torch.stack(branch_p, dim=0), dim=0)
            elif rule == "product":
                stacked_p = branch_p[0]
                for p in branch_p[1:]:
                    stacked_p = stacked_p * p
                # Re-normalize to sum to 1
                stacked_p = stacked_p / (torch.sum(stacked_p, dim=-1, keepdim=True) + 1e-8)
            else:
                raise ValueError(f"Unknown combining rule: {rule}")
            combo_probs[c_name] = stacked_p

        return combo_probs

    def count_parameters(self) -> Dict[str, int]:
        """Calculates total and branch-specific parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        b1_params = sum(p.numel() for p in self.stage1.parameters()) + sum(p.numel() for p in self.head1.parameters())
        b2_params = sum(p.numel() for p in self.stage2.parameters()) + sum(p.numel() for p in self.head2.parameters())
        b3_params = sum(p.numel() for p in self.stage3.parameters()) + sum(p.numel() for p in self.head3.parameters())
        b4_params = sum(p.numel() for p in self.stage4.parameters()) + sum(p.numel() for p in self.head4.parameters())
        b5_params = sum(p.numel() for p in self.stage5.parameters()) + sum(p.numel() for p in self.head5.parameters())
        
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "B1_parameters": b1_params,
            "B2_parameters": b2_params,
            "B3_parameters": b3_params,
            "B4_parameters": b4_params,
            "B5_parameters": b5_params
        }
