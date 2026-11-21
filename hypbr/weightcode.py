"""Zero-training analytic weight surgery ("WeightCode") for neural PBR priors.

Modifies frozen pretrained PyTorch convolutional networks directly in weight space
without backpropagation, datasets, or LoRA adapters:

  1. Exact 9-mode orthonormal 2D spatial basis decomposition of 3x3 Conv2d kernels
     into DC (global offset), linear ramp (perspective tilt), and high-frequency
     surface relief / curvature modes.
  2. Multi-scale DPT pyramid spectral rebalancing: suppresses coarse perspective
     ramps while amplifying fine surface micro-relief directly inside frozen weights.
  3. Seamless circular Conv2d padding conversion + 4-quadrant separable raised-cosine
     cyclic-shift inference (exact partition of unity, zero ViT border seams).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


def _build_3x3_spatial_basis() -> torch.Tensor:
    """Return an orthonormal (9, 9) 2D separable DCT-II basis for 3x3 conv kernels.

    Row ordering by spatial frequency:
      - index 0:     DC (u=0, v=0) — constant offset
      - indices 1,2: Linear tilt/ramp (u=1,v=0) and (u=0,v=1) — perspective slant
      - indices 3..8: High-frequency relief, curvature, corners, micro-detail
    """
    n = 3
    k = np.arange(n, dtype=np.float64)[:, None]
    i = np.arange(n, dtype=np.float64)[None, :]
    c = np.cos(np.pi * (2.0 * i + 1.0) * k / (2.0 * n))
    c[0] *= np.sqrt(1.0 / n)
    c[1:] *= np.sqrt(2.0 / n)

    # Order pairs so DC is 0, first-order linear ramps are 1..2, rest are 3..8
    pairs = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1), (2, 2)]
    basis = np.stack([np.outer(c[u], c[v]).reshape(9) for u, v in pairs], axis=0)
    return torch.from_numpy(basis.astype(np.float32))


BASIS_3X3: torch.Tensor = _build_3x3_spatial_basis()


@dataclass(frozen=True)
class WeightCodeReport:
    tensors_edited: int
    params_edited: int
    total_params: int
    circular_conv_layers: int

    @property
    def pct_edited(self) -> float:
        return 100.0 * self.params_edited / max(self.total_params, 1)


@torch.no_grad()
def spectral_filter_3x3_kernel(
    weight: torch.Tensor,
    dc_gain: float = 1.0,
    ramp_gain: float = 1.0,
    detail_gain: float = 1.0,
) -> torch.Tensor:
    """Re-weight DC, linear-ramp, and high-frequency relief modes inside a (Out, In, 3, 3) kernel.

    Because BASIS_3X3 is orthonormal, setting all gains to 1.0 is an exact identity transform.
    """
    if weight.ndim != 4 or weight.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (Out, In, 3, 3) conv weight, got {tuple(weight.shape)}")
    orig_dtype = weight.dtype
    w = weight.detach().to(torch.float32)
    B = BASIS_3X3.to(w.device)
    gains = torch.tensor(
        [dc_gain, ramp_gain, ramp_gain, detail_gain, detail_gain, detail_gain, detail_gain, detail_gain, detail_gain],
        dtype=torch.float32,
        device=w.device,
    )
    flat = w.reshape(-1, 9)
    coeffs = flat @ B.T
    filtered = (coeffs * gains) @ B
    return filtered.reshape_as(w).to(orig_dtype)


@torch.no_grad()
def svd_energy_shrink(weight: torch.Tensor, energy_keep: float = 0.96, tail_scale: float = 0.35) -> torch.Tensor:
    """Softly damp low-energy singular modes of a 2D or 1x1 conv projection matrix."""
    orig_dtype = weight.dtype
    orig_shape = weight.shape
    w = weight.detach().to(torch.float32).reshape(orig_shape[0], -1)
    if min(w.shape) < 2:
        return weight.clone()
    U, S, Vh = torch.linalg.svd(w, full_matrices=False)
    energy = torch.cumsum(S * S, dim=0) / (torch.sum(S * S) + 1e-12)
    mask = torch.where(energy <= energy_keep, torch.ones_like(S), torch.full_like(S, tail_scale))
    mask[0] = 1.0
    out = (U * (S * mask)) @ Vh
    return out.reshape(orig_shape).to(orig_dtype)


def set_circular_conv_padding(module: nn.Module, enabled: bool = True) -> int:
    """Switch padded Conv2d layers in `module` to circular (seamless) or zeros padding."""
    mode = "circular" if enabled else "zeros"
    count = 0
    for m in module.modules():
        if isinstance(m, nn.Conv2d) and m.padding != (0, 0):
            m.padding_mode = mode
            count += 1
    return count


@torch.no_grad()
def apply_depthpro_weightcode(
    model: nn.Module,
    scale_weights: tuple[float, float, float, float] = (1.85, 1.35, 0.65, 0.20),
    boost_fine_res: float = 1.35,
    coarse_ramp_gain: float = 0.35,
    fine_detail_gain: float = 1.25,
    circular_pad: bool = False,
) -> WeightCodeReport:
    """Apply zero-training analytic WeightCode surgery to a DPT / DepthPro2 model in-place.

    Combines:
      - Multi-scale DPT pyramid gain rebalancing (`decoder.convs.0..3`)
      - Spectral 3x3 kernel filtering (suppresses perspective ramp in coarse scales,
        boosts micro-relief in fine scales)
      - Fine residual fusion gain boost (`decoder.fusions.0..1`)
      - Optional circular Conv2d padding across decoder and head
    """
    total_params = sum(p.numel() for p in model.parameters())
    sd = model.state_dict()
    tensors_edited = 0
    params_edited = 0

    for lvl, s in enumerate(scale_weights):
        k = f"decoder.convs.{lvl}.weight"
        if k not in sd:
            continue
        w = sd[k] * float(s)
        if w.ndim == 4 and w.shape[-2:] == (3, 3):
            if lvl <= 1:
                w = spectral_filter_3x3_kernel(w, dc_gain=0.90, ramp_gain=0.75, detail_gain=fine_detail_gain)
            else:
                w = spectral_filter_3x3_kernel(w, dc_gain=0.70, ramp_gain=coarse_ramp_gain, detail_gain=0.85)
        sd[k] = w
        tensors_edited += 1
        params_edited += w.numel()

    for lvl, s in [(0, boost_fine_res), (1, 1.0 + 0.5 * (boost_fine_res - 1.0))]:
        for suf in ("weight", "bias"):
            k = f"decoder.fusions.{lvl}.resnet1.residual.3.{suf}"
            if k in sd:
                w = sd[k] * float(s)
                if suf == "weight" and w.ndim == 4 and w.shape[-2:] == (3, 3):
                    w = spectral_filter_3x3_kernel(w, dc_gain=0.95, ramp_gain=0.80, detail_gain=1.15)
                sd[k] = w
                tensors_edited += 1
                params_edited += w.numel()

    model.load_state_dict(sd, strict=True)

    circular_layers = 0
    for attr in ("decoder", "head"):
        sub = getattr(model, attr, None)
        if isinstance(sub, nn.Module):
            circular_layers += set_circular_conv_padding(sub, enabled=circular_pad)

    return WeightCodeReport(
        tensors_edited=tensors_edited,
        params_edited=params_edited,
        total_params=total_params,
        circular_conv_layers=circular_layers,
    )


def tileable_4quad_blend(
    rgb_uint8: np.ndarray,
    infer_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Exact 4-quadrant separable raised-cosine cyclic shift blending.

    Evaluates `infer_fn` on 4 half-tile cyclic shifts: (0,0), (H/2,0), (0,W/2), (H/2,W/2),
    un-shifts each prediction, and blends with separable weights:
      w_y(y) = sin^2(pi * y / H),  w_x(x) = sin^2(pi * x / W)

    Because sin^2(theta) + cos^2(theta) == 1 identically, the 4 weights form an exact
    partition of unity where every pass has weight 0 along BOTH of its own border seams,
    eliminating ViT patch boundary artifacts mathematically.
    """
    H, W = rgb_uint8.shape[:2]
    hy, hx = H // 2, W // 2

    y = np.linspace(0.0, np.pi, H, endpoint=False, dtype=np.float32)
    x = np.linspace(0.0, np.pi, W, endpoint=False, dtype=np.float32)
    wy0 = np.sin(y) ** 2          # 0 at y=0, 1 at y=H/2
    wy1 = np.cos(y) ** 2          # 1 at y=0, 0 at y=H/2
    wx0 = np.sin(x) ** 2          # 0 at x=0, 1 at x=W/2
    wx1 = np.cos(x) ** 2          # 1 at x=0, 0 at x=W/2

    shifts_and_weights = (
        (0, 0, np.outer(wy0, wx0)),
        (hy, 0, np.outer(wy1, wx0)),
        (0, hx, np.outer(wy0, wx1)),
        (hy, hx, np.outer(wy1, wx1)),
    )

    acc = np.zeros((H, W), dtype=np.float32)
    for sy, sx, w in shifts_and_weights:
        inp = np.roll(np.roll(rgb_uint8, sy, axis=0), sx, axis=1) if (sy or sx) else rgb_uint8
        pred = infer_fn(np.ascontiguousarray(inp)).astype(np.float32)
        if sy or sx:
            pred = np.roll(np.roll(pred, -sy, axis=0), -sx, axis=1)
        acc += w * pred
    return acc
