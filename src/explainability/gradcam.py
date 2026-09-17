"""Grad-CAM for Phase 09: visualize what the model is looking at.

Standard Grad-CAM (Selvaraju et al. 2017): hook a target conv layer's
forward activations and backward gradients, weight activation channels by
their gradient's global average, ReLU, normalize to [0,1].

Motivation from measured results (docs/ERROR_ANALYSIS.md): the model
completely fails on "spider" and has poor "swiss cheese" recall — Grad-CAM
lets us check whether it's looking at plausible terrain features for
those classes or latching onto something spurious. Answering that
question requires running this against the real trained checkpoint
(not done in this module — see notebooks/colab_train.ipynb's
explainability cells for the actual run).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activations)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def generate(self, input_tensor: torch.Tensor, target_class: int | None = None) -> tuple[torch.Tensor, int]:
        """input_tensor: (1, C, H, W). Returns (heatmap of shape (H, W) in
        [0, 1], the target_class used — the argmax prediction if not given)."""
        if input_tensor.dim() != 4 or input_tensor.size(0) != 1:
            raise ValueError("input_tensor must have shape (1, C, H, W)")

        self.model.eval()
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        score = output[0, target_class]
        score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Hooks did not fire — is target_layer part of the forward path?")

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = F.relu((weights * self._activations).sum(dim=1, keepdim=True))  # (1, 1, h, w)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze()

        cam_min, cam_max = cam.min(), cam.max()
        if (cam_max - cam_min).item() > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam, target_class
