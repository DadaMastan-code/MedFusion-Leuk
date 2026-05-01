"""
Grad-CAM visualization for EfficientNet-B4 backbone.

# [NOVELTY] Dual XAI: Grad-CAM highlights WHICH cells are abnormal
# (visual explanation) while SHAP explains WHICH features drive the
# overall risk score (numerical explanation).
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from pathlib import Path
from typing import Optional

from config import IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD, GRADCAM_DIR


class GradCAM:
    """
    Hooks into the last convolutional block of EfficientNet-B4 to produce
    a class-discriminative heatmap overlaid on the original blood smear.
    """

    def __init__(self, model: torch.nn.Module, target_layer_name: str = "efficientnet.blocks"):
        self.model = model
        self.model.eval()
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None

        # Resolve target layer (last conv block of EfficientNet)
        target = self._get_layer(model, target_layer_name)
        target.register_forward_hook(self._save_activations)
        target.register_full_backward_hook(self._save_gradients)

    def _get_layer(self, model, layer_name: str):
        parts = layer_name.split(".")
        layer = model
        for p in parts:
            if p.isdigit():
                layer = layer[int(p)]
            else:
                layer = getattr(layer, p)
        return layer

    def _save_activations(self, module, input, output):
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Returns a (H, W) heatmap in [0, 1] for the predicted or specified class.
        """
        input_tensor = input_tensor.unsqueeze(0).requires_grad_(True)
        logits, _ = self.model(input_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Pool gradients across channels
        weights = self._gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1).squeeze()   # (H', W')
        cam = F.relu(cam)

        # Normalise to [0, 1]
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

    def overlay_on_image(
        self,
        original_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.45,
    ) -> Image.Image:
        """Blend the Grad-CAM heatmap with the original blood smear image."""
        img_np = np.array(original_image.resize((IMAGE_SIZE, IMAGE_SIZE)))
        heatmap_resized = cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))
        heatmap_color = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        overlay = (alpha * heatmap_color + (1 - alpha) * img_np).astype(np.uint8)
        return Image.fromarray(overlay)

    def save(
        self,
        original_image: Image.Image,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        filename: str = "gradcam.png",
    ) -> Path:
        heatmap = self.generate(input_tensor, class_idx)
        overlay = self.overlay_on_image(original_image, heatmap)
        out_path = GRADCAM_DIR / filename
        GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
        overlay.save(out_path)
        return out_path
