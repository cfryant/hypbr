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
    contrast: float = 0.15,
    mode: str = "height",
) -> np.ndarray:
    """Compute smooth blend weight `m in [0, 1]` selecting material B over material A.

    - `mode="height"`: Unreal/Substance HeightLerp — material B fills or rises where
      `h_b + factor` exceeds `h_a + (1 - factor)`, with transition sharpness controlled by `contrast`.
    - `mode="cavity"`: Material B settles preferentially into crevices/low regions of A.
    - `mode="linear"`: Uniform scalar blend `m = factor`.
    """
    f = float(np.clip(factor, 0.0, 1.0))
    c = max(float(contrast), 1e-3)
    if mode == "linear":
        return np.full_like(h_a, f, dtype=np.float32)
    if mode == "cavity":
        # Low height of A + high height of B favors B
        score = (1.0 - h_a) * 0.65 + h_b * 0.35 + (f * 2.0 - 1.0)
        m = np.clip((score - 0.5) / c + 0.5, 0.0, 1.0)
    elif mode == "height":
        # Classic HeightLerp: compare biased heightfields
        ma = h_a + (1.0 - f)
        mb = h_b + f
        diff = mb - ma
        m = np.clip(diff / c + 0.5, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown blend mode {mode!r}. Choose 'height', 'cavity', or 'linear'.")
    # Smoothstep for C1-continuous transitions without hard stepping
    return (m * m * (3.0 - 2.0 * m)).astype(np.float32)


def blend_materials(
    dir_a: Path | str,
    dir_b: Path | str,
    out_dir: Path | str,
    factor: float = 0.5,
    contrast: float = 0.15,
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
    m3 = mask[..., None]

    # 1. Linear-light albedo blend
    alb_a = _load_srgb_linear(pa / "albedo_delit.png", size)
    alb_b = _load_srgb_linear(pb / "albedo_delit.png", size)
    alb_lin = (1.0 - m3) * alb_a + m3 * alb_b
    alb_u8 = np.clip(np.power(np.clip(alb_lin, 0.0, 1.0), 1.0 / 2.2) * 255.0 + 0.5, 0, 255).astype(np.uint8)

    # 2. Height & slope-consistent normal blend
    h_out = dp.normalize_height((1.0 - mask) * h_a + mask * h_b, clip_pct=0.5)
    n_a = _load_normal_f32(pa / "normal.png", size)
    n_b = _load_normal_f32(pb / "normal.png", size)
    n_out = ng.normalize_n((1.0 - m3) * n_a + m3 * n_b)

    # 3. Roughness (with Specular AA) & Metallic
    r_a = _load_gray_f32(pa / "roughness.png", size, 0.5)
    r_b = _load_gray_f32(pb / "roughness.png", size, 0.5)
    rough = np.clip((1.0 - mask) * r_a + mask * r_b, 0.06, 0.98).astype(np.float32)
    rough = ng.node_roughness_specular_aa({"circular": True}, {"roughness": rough, "normal": n_out}, strength=0.2, max_boost=0.06)["roughness"]

    met_a = _load_gray_f32(pa / "metallic.png", size, 0.0)
    met_b = _load_gray_f32(pb / "metallic.png", size, 0.0)
    met = np.clip((1.0 - mask) * met_a + mask * met_b, 0.0, 1.0).astype(np.float32)
    met_u8 = np.clip(met * 255.0 + 0.5, 0, 255).astype(np.uint8)

    # 4. Recompute physical 8-azimuth HBAO + bent normal + curvature + anisotropy + thickness
    hbao = ng.node_horizon_bent_normal_ao({"circular": True}, {"height": h_out, "normal": n_out}, strength=1.45, height_scale=18.0)
    ao = hbao["ao"]
    curv = ng.node_curvature_multiscale({"circular": True}, {"normal": n_out}, gain=2.8)
    aniso = ng.node_anisotropy_flow({"circular": True}, {"height": h_out, "ao": ao, "roughness": rough})
    thick = ng.node_thickness_subsurface({"circular": True}, {"height": h_out})
    rough_u8 = np.clip(rough * 255.0 + 0.5, 0, 255).astype(np.uint8)
    orm_u8 = np.stack([hbao["ao_u8"], rough_u8, met_u8], axis=-1)

    # 5. Shaded studio preview
    L1 = np.array([0.48, 0.52, 0.71], np.float32); L1 /= np.linalg.norm(L1)
    L2 = np.array([-0.60, -0.30, 0.74], np.float32); L2 /= np.linalg.norm(L2)
    H1 = L1 + np.array([0.0, 0.0, 1.0], np.float32); H1 /= np.linalg.norm(H1)
    ndotl1 = np.clip((n_out * L1[None, None, :]).sum(-1), 0.0, 1.0)
    ndotl2 = np.clip((n_out * L2[None, None, :]).sum(-1), 0.0, 1.0)
    ndoth1 = np.clip((n_out * H1[None, None, :]).sum(-1), 0.0, 1.0)
    shin = np.clip(2.0 / (np.maximum(rough, 0.08) ** 2.4) - 2.0, 2.0, 180.0)
    spec_col = (1.0 - met[..., None]) * 0.8 + met[..., None] * alb_lin
    spec = (ndoth1 ** shin)[..., None] * (1.0 - rough[..., None] * 0.80) * 0.75 * spec_col
    col = alb_lin * (0.18 + 0.80 * ndotl1[..., None] + 0.22 * ndotl2[..., None]) * (0.28 + 0.72 * ao[..., None]) + spec * (0.35 + 0.65 * ao[..., None])
    prev_u8 = np.clip(np.power(np.clip(col, 0.0, 1.0), 1.0 / 2.2) * 255.0 + 0.5, 0, 255).astype(np.uint8)

    Image.fromarray(alb_u8).save(out / "albedo_delit.png")
    Image.fromarray(prev_u8).save(out / "preview.png")
    Image.fromarray(np.clip(h_out * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(out / "height.png")
    Image.fromarray(np.clip(h_out * 65535.0 + 0.5, 0, 65535).astype(np.uint16)).save(out / "height16.png")
    Image.fromarray(dp.normal_to_u8(n_out)).save(out / "normal.png")
    Image.fromarray(hbao["bent_normal_u8"]).save(out / "bent_normal.png")
    Image.fromarray(hbao["ao_u8"]).save(out / "ao.png")
    Image.fromarray(rough_u8).save(out / "roughness.png")
    Image.fromarray(met_u8).save(out / "metallic.png")
    Image.fromarray(orm_u8).save(out / "orm.png")
    Image.fromarray(curv["curvature_u8"]).save(out / "curvature.png")
    Image.fromarray(curv["cavity_u8"]).save(out / "cavity.png")
    Image.fromarray(np.clip(curv["convex"] * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(out / "edge_wear.png")
    Image.fromarray(aniso["anisotropy_u8"]).save(out / "anisotropy.png")
    Image.fromarray(aniso["specular_u8"]).save(out / "specular.png")
    Image.fromarray(thick["thickness_u8"]).save(out / "thickness.png")
    Image.fromarray(np.clip(mask * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(out / "blend_mask.png")
    return out
