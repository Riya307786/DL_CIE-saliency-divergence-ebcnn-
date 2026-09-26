import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

class BranchGradCAM:
    """
    Grad-CAM engine for multi-branch EB-CNN architecture.
    Calculates gradient-weighted class activation maps for each branch (B1..B5)
    targeting their respective final convolutional layers.
    """
    def __init__(self, model: nn.Module, target_size: Tuple[int, int] = (224, 224)):
        self.model = model
        self.target_size = target_size
        self.hooks = []
        self.activations: Dict[str, torch.Tensor] = {}
        self.gradients: Dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self):
        # Register hooks on the final conv layer of each stage
        stages = [
            ("B1", self.model.stage1),
            ("B2", self.model.stage2),
            ("B3", self.model.stage3),
            ("B4", self.model.stage4),
            ("B5", self.model.stage5)
        ]

        for b_name, stage in stages:
            # Find the last Conv2d layer
            last_conv = None
            for layer in stage.convs:
                if isinstance(layer, nn.Conv2d):
                    last_conv = layer

            if last_conv is not None:
                self._attach_hook(b_name, last_conv)

    def _attach_hook(self, name: str, layer: nn.Module):
        def forward_hook(module, input, output):
            self.activations[name] = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients[name] = grad_output[0].detach()

        h1 = layer.register_forward_hook(forward_hook)
        h2 = layer.register_full_backward_hook(backward_hook)
        self.hooks.extend([h1, h2])

    def generate_saliency_maps(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[Union[int, List[int]]] = None,
        normalize: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Generates Grad-CAM saliency maps for all 5 branches on input_tensor.
        input_tensor: shape (B, 3, H, W)
        target_class: int or None (if None, uses predicted class of each branch)
        
        Returns:
            Dict[str, torch.Tensor] mapping branch name ("B1".."B5") to saliency maps of shape (B, target_size[0], target_size[1])
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        logits_dict = self.model(input_tensor)
        saliency_maps = {}

        for b_name in ["B1", "B2", "B3", "B4", "B5"]:
            logits = logits_dict[b_name] # (B, num_classes)
            batch_size = logits.shape[0]

            if target_class is None:
                chosen_class = logits.argmax(dim=-1) # (B,)
            elif isinstance(target_class, int):
                chosen_class = torch.full((batch_size,), target_class, dtype=torch.long, device=logits.device)
            else:
                chosen_class = torch.tensor(target_class, dtype=torch.long, device=logits.device)

            # Score to backpropagate
            score = logits.gather(1, chosen_class.unsqueeze(1)).sum()
            
            self.model.zero_grad()
            score.backward(retain_graph=(b_name != "B5"))

            acts = self.activations[b_name] # (B, C, H_f, W_f)
            grads = self.gradients[b_name]   # (B, C, H_f, W_f)

            # Equation (1): Global average pooling of gradients -> weights alpha_k
            alpha = torch.mean(grads, dim=(2, 3), keepdim=True) # (B, C, 1, 1)

            # Equation (2): Linear combination of activation maps followed by ReLU
            cam = torch.sum(alpha * acts, dim=1, keepdim=True) # (B, 1, H_f, W_f)
            cam = F.relu(cam)

            # Upsample to common target size (224, 224)
            cam_upsampled = F.interpolate(cam, size=self.target_size, mode='bilinear', align_corners=False)
            cam_upsampled = cam_upsampled.squeeze(1) # (B, 224, 224)

            # Normalization per image to [0, 1]
            if normalize:
                cam_min = cam_upsampled.view(batch_size, -1).min(dim=-1)[0].view(batch_size, 1, 1)
                cam_max = cam_upsampled.view(batch_size, -1).max(dim=-1)[0].view(batch_size, 1, 1)
                cam_norm = (cam_upsampled - cam_min) / (cam_max - cam_min + 1e-8)
                saliency_maps[b_name] = cam_norm
            else:
                saliency_maps[b_name] = cam_upsampled

        return saliency_maps

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
