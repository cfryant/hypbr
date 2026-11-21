"""Physically-based height-lerp PBR material blending (`hypbr blend`).

Blends two seamless hypbr material directories using heightmap-driven masking
(analogous to Substance / Unreal HeightLerp), preserving physical correctness:
  - Linear-light albedo blending (sRGB -> linear -> blend -> sRGB)
  - Tangent-space normal blending with unit re-normalization
  - Height-masked roughness, metallic, AO, specular, thickness, and ORM repacking
  - Exact circular-padded curvature, cavity, and edge-wear recomputation
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from . import depthpro_pbr as dp
from . import normal_graph as ng
from . import procedural as gd
from .synthesize import render_studio_preview, save_maps_parallel


def _load_srgb_linear(path: Path, size: tuple[int, int]) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if im.size != size:
        im = im.resize(size, Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return np.power(np.clip(arr, 0.0, 1.0), 2.2)


def _load_gray_f32(path: Path, size: tuple[int, int], default: float = 0.5) -> np.ndarray:
    if not path.exists():
        return np.full((size[1], size[0]), default, dtype=np.float32)
    im = Image.open(path).convert("L")
    if im.size != size:
        im = im.resize(size, Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def _load_height_f32(mat_dir: Path, size: tuple[int, int]) -> np.ndarray:
    """Prefer full-precision 16-bit height16.png when available, falling back to 8-bit height.png."""
    h16_path = mat_dir / "height16.png"
    if h16_path.exists():
        im = Image.open(h16_path)
        arr = np.asarray(im, dtype=np.float32)
        if im.mode in ("I;16", "I") or arr.max() > 255.0:
            arr = arr / 65535.0
        else:
            arr = arr / 255.0
        if im.size != size:
            im_f = Image.fromarray(arr, mode="F").resize(size, Image.BILINEAR)
            arr = np.asarray(im_f, dtype=np.float32)
        return np.clip(arr, 0.0, 1.0).astype(np.float32)
    return _load_gray_f32(mat_dir / "height.png", size, 0.5)


def _load_normal_f32(path: Path, size: tuple[int, int]) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if im.size != size:
        im = im.resize(size, Image.BILINEAR)
    n = np.asarray(im, dtype=np.float32) / 127.5 - 1.0
    return ng.normalize_n(n)


def compute_height_blend_mask(
    h_a: np.ndarray,
    h_b: np.ndarray,
    factor: float = 0.5,
    contrast: float = 0.12,
    mode: str = "height",
) -> np.ndarray:
    """Compute organic, C1-smooth HeightLerp blend weight `m in [0, 1]` selecting material B over A.

    - `mode="height"`: Unreal/Substance HeightLerp — material B rises where
      `h_b + factor` exceeds `h_a + (1 - factor)`, with organic micro-fracture breakup.
    - `mode="cavity"`: Material B settles preferentially into crevices/valleys of A,
      modulated by material B's height and periodic micro-variation to avoid uniform blobs.
    - `mode="linear"`: Uniform scalar blend `m = factor`.
    """
    f = float(np.clip(factor, 0.0, 1.0))
    c = max(float(contrast), 1e-3)
    if mode == "linear":
        return np.full_like(h_a, f, dtype=np.float32)

    H, W = h_a.shape
    # Seamless organic edge breakup prevents artificial uniform blobs on regular patterns
    breakup = gd._periodic_fbm(H, W, 9137, (6, 12, 24, 48)) * 0.14
    if mode == "cavity":
        # Low valleys of A + high relief of B + organic breakup favors B
        score = (1.0 - h_a) * 0.55 + h_b * 0.45 + breakup + (f * 1.4 - 0.7)
        m = np.clip((score - 0.5) / c + 0.5, 0.0, 1.0)
    elif mode == "height":
        ma = h_a + (1.0 - f)
        mb = h_b + f + breakup
        diff = mb - ma
        m = np.clip(diff / c + 0.5, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown blend mode {mode!r}. Choose 'height', 'cavity', or 'linear'.")
    # Smoothstep for crisp yet anti-aliased transitions
    return (m * m * (3.0 - 2.0 * m)).astype(np.float32)


def blend_materials(
    dir_a: Path | str,
    dir_b: Path | str,
    out_dir: Path | str,
    factor: float = 0.5,
    contrast: float = 0.09,
    mode: str = "height",
) -> Path:
    """Physically blend two hypbr material directories into a new seamless 15-map directory."""
    pa, pb, out = Path(dir_a), Path(dir_b), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with Image.open(pa / "albedo_delit.png") as im0:
        size = im0.size

    h_a = _load_height_f32(pa, size)
    h_b = _load_height_f32(pb, size)
    mask = compute_height_blend_mask(h_a, h_b, factor=factor, contrast=contrast, mode=mode)

    # Unified crisp physical boundary mask so albedo, roughness, and metallic share clean edges
    # without semi-metallic gray halos or muddy/cyan transition fringes.
    if mode == "linear":
        surf_mask = mask
        met_mask = mask
    else:
        sm = np.clip((mask - 0.5) / 0.24 + 0.5, 0.0, 1.0)
        surf_mask = (sm * sm * (3.0 - 2.0 * sm)).astype(np.float32)
        mm = np.clip((mask - 0.5) / 0.16 + 0.5, 0.0, 1.0)
        met_mask = (mm * mm * (3.0 - 2.0 * mm)).astype(np.float32)
    m3 = surf_mask[..., None]

    # 1. Linear-light albedo blend + subtle contact darkening at overlay rim
    alb_a = _load_srgb_linear(pa / "albedo_delit.png", size)
    alb_b = _load_srgb_linear(pb / "albedo_delit.png", size)
    rim_band = 4.0 * surf_mask * (1.0 - surf_mask)  # peaks at 1.0 along material boundary
    alb_lin = ((1.0 - m3) * alb_a + m3 * alb_b) * (1.0 - 0.10 * rim_band[..., None])
    alb_u8 = np.clip(np.power(np.clip(alb_lin, 0.0, 1.0), 1.0 / 2.2) * 255.0 + 0.5, 0, 255).astype(np.uint8)

    # 2. Height & slope-consistent normal blend with physical transition bevel
    h_blend = (1.0 - surf_mask) * h_a + surf_mask * (h_b + (0.04 if mode == "height" else 0.0))
    h_out = dp.normalize_height(h_blend, clip_pct=0.5)
    n_a = _load_normal_f32(pa / "normal.png", size)
    n_b = _load_normal_f32(pb / "normal.png", size)
    n_lip = gd._height_to_normal_periodic(surf_mask, slope=0.35 if mode != "linear" else 0.0)
    n_mix = ng.normalize_n((1.0 - m3) * n_a + m3 * n_b)
    n_out = ng.normalize_n(np.stack([n_mix[..., 0] + n_lip[..., 0] * 0.45, n_mix[..., 1] + n_lip[..., 1] * 0.45, np.maximum(n_mix[..., 2], 0.15)], axis=-1))

    # 3. Roughness (with Specular AA) & crisp physical Metallic
    r_a = _load_gray_f32(pa / "roughness.png", size, 0.5)
    r_b = _load_gray_f32(pb / "roughness.png", size, 0.5)
    rough = np.clip((1.0 - surf_mask) * r_a + surf_mask * r_b + 0.06 * rim_band, 0.06, 0.98).astype(np.float32)
    rough = ng.node_roughness_specular_aa({"circular": True}, {"roughness": rough, "normal": n_out}, strength=0.2, max_boost=0.06)["roughness"]

    met_a = _load_gray_f32(pa / "metallic.png", size, 0.0)
    met_b = _load_gray_f32(pb / "metallic.png", size, 0.0)
    met = np.clip((1.0 - met_mask) * met_a + met_mask * met_b, 0.0, 1.0).astype(np.float32)
    met_u8 = np.clip(met * 255.0 + 0.5, 0, 255).astype(np.uint8)

    # 4. Recompute physical 8-azimuth HBAO + bent normal + curvature + anisotropy + thickness
    hbao = ng.node_horizon_bent_normal_ao({"circular": True}, {"height": h_out, "normal": n_out}, strength=1.45, height_scale=18.0)
    ao = hbao["ao"]
    curv = ng.node_curvature_multiscale({"circular": True}, {"normal": n_out}, gain=2.8)
    aniso = ng.node_anisotropy_flow({"circular": True}, {"height": h_out, "ao": ao, "roughness": rough})
    thick = ng.node_thickness_subsurface({"circular": True}, {"height": h_out})
    rough_u8 = np.clip(rough * 255.0 + 0.5, 0, 255).astype(np.uint8)
    orm_u8 = np.stack([hbao["ao_u8"], rough_u8, met_u8], axis=-1)

    # 5. Physically-based studio preview (with metallic Fresnel IBL)
    prev_u8 = render_studio_preview(alb_lin, n_out, rough, met, ao)

    save_maps_parallel([
        (out / "albedo_delit.png", alb_u8),
        (out / "preview.png", prev_u8),
        (out / "height.png", np.clip(h_out * 255.0 + 0.5, 0, 255).astype(np.uint8)),
        (out / "height16.png", np.clip(h_out * 65535.0 + 0.5, 0, 65535).astype(np.uint16)),
        (out / "normal.png", dp.normal_to_u8(n_out)),
        (out / "bent_normal.png", hbao["bent_normal_u8"]),
        (out / "ao.png", hbao["ao_u8"]),
        (out / "roughness.png", rough_u8),
        (out / "metallic.png", met_u8),
        (out / "orm.png", orm_u8),
        (out / "curvature.png", curv["curvature_u8"]),
        (out / "cavity.png", curv["cavity_u8"]),
        (out / "edge_wear.png", np.clip(curv["convex"] * 255.0 + 0.5, 0, 255).astype(np.uint8)),
        (out / "anisotropy.png", aniso["anisotropy_u8"]),
        (out / "specular.png", aniso["specular_u8"]),
        (out / "thickness.png", thick["thickness_u8"]),
        (out / "blend_mask.png", np.clip(surf_mask * 255.0 + 0.5, 0, 255).astype(np.uint8)),
    ])
    return out
