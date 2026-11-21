#!/usr/bin/env python3
"""Everyday Real-World Surface Details PBR Normal/Height/Roughness Generator.

Generates 12 seamlessly tileable everyday-life surface details (things encountered daily:
cardboard, knurled metal, denim twill, quilted leather, pegboard, sidewalk concrete,
beveled subway tile, rubber stair tread, corduroy, radiator fins, herringbone parquet,
and 2x2 carbon fiber weave) x 3 wear states (0_fresh, 1_lived_in, 2_weathered)
= 36 complete 512x512 seamless PBR texture sets!

Outputs per (detail, wear_state):
  - `normal.png`    (512x512 OpenGL +Y tangent-space normal, seamlessly tileable)
  - `height.png`    (512x512 linear POM heightmap, seamlessly tileable)
  - `ao.png`        (512x512 multi-scale cavity ambient occlusion)
  - `roughness.png` (512x512 calibrated physical roughness + Specular AA)
  - `edge_wear.png` (512x512 convex ridge wear / scuff / lint highlight mask)
  - `preview.png`   (512x512 shaded PBR clay/material lit preview so details read instantly!)

Usage:
  hypbr synth --out ./output/pbr --size 512
"""
import argparse
import base64
import functools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from . import depthpro_pbr as dp  # noqa: E402
from . import normal_graph as ng  # noqa: E402


# =============================================================================
# 1. Seamless Periodic Noise & Helpers
# =============================================================================
@functools.lru_cache(maxsize=96)
def _cr_periodic_basis_np(n: int, f_sz: int) -> np.ndarray:
    """Exact 1D periodic Catmull-Rom cubic interpolation matrix of shape `(n, f_sz)`."""
    pos = np.linspace(0.0, float(f_sz) * (1.0 - 1.0 / n), n, dtype=np.float32)
    i0 = np.floor(pos).astype(np.int64) % f_sz
    t = pos - np.floor(pos)
    t2 = t * t
    t3 = t2 * t
    mat = np.zeros((n, f_sz), dtype=np.float32)
    rows = np.arange(n)
    np.add.at(mat, (rows, (i0 - 1) % f_sz), -0.5 * t3 + t2 - 0.5 * t)
    np.add.at(mat, (rows, i0), 1.5 * t3 - 2.5 * t2 + 1.0)
    np.add.at(mat, (rows, (i0 + 1) % f_sz), -1.5 * t3 + 2.0 * t2 + 0.5 * t)
    np.add.at(mat, (rows, (i0 + 2) % f_sz), 0.5 * t3 - 0.5 * t2)
    return mat


@functools.lru_cache(maxsize=96)
def _cr_periodic_basis_gpu(n: int, f_sz: int, dev_str: str) -> torch.Tensor:
    return torch.from_numpy(_cr_periodic_basis_np(n, f_sz)).to(dev_str)


@torch.no_grad()
def _periodic_fbm(H: int, W: int, seed: int, octaves=(4, 8, 16, 32, 64)) -> np.ndarray:
    """100% mathematically seamless multi-octave periodic noise for ANY integer frequency f.

    Uses cached separable 1D periodic Catmull-Rom GEMMs `(My @ g) @ Mx.T` (14x faster on
    MPS/CUDA and 9x faster on CPU BLAS than `grid_sample`, with >0.997 correlation and
    exact circular periodicity for arbitrary integer frequencies).
    """
    rng = np.random.default_rng(seed)
    use_gpu = (dp.DEVICE.type != "cpu") and (H * W >= 262144) and all(f <= 256 for f in octaves)
    amp = 1.0
    norm = 0.0

    if use_gpu:
        dev = dp.DEVICE
        dev_str = str(dev)
        out_t = torch.zeros((H, W), dtype=torch.float32, device=dev)
        for f in octaves:
            g = torch.from_numpy(rng.standard_normal((f, f)).astype(np.float32)).to(dev)
            my = _cr_periodic_basis_gpu(H, f, dev_str)
            mx = _cr_periodic_basis_gpu(W, f, dev_str)
            out_t.add_((my @ g) @ mx.T, alpha=amp)
            norm += amp
            amp *= 0.52
        out_t.div_(max(norm, 1e-6))
        out_t = (out_t - out_t.mean()) / (out_t.std() + 1e-6)
        return out_t.cpu().numpy().astype(np.float32)

    out = np.zeros((H, W), dtype=np.float32)
    for f in octaves:
        g = rng.standard_normal((f, f)).astype(np.float32)
        if f <= 256:
            my = _cr_periodic_basis_np(H, f)
            mx = _cr_periodic_basis_np(W, f)
            up = (my @ g) @ mx.T
        else:
            yy = torch.linspace(0.0, 1.0 - 1.0 / H, H, dtype=torch.float32)
            xx = torch.linspace(0.0, 1.0 - 1.0 / W, W, dtype=torch.float32)
            gy, gx = torch.meshgrid(yy, xx, indexing="ij")
            t_pad = F.pad(torch.from_numpy(g)[None, None], (2, 2, 2, 2), mode="circular")
            n_pad = float(f + 4)
            u_c = ((2.0 + gx * float(f) + 0.5) / n_pad) * 2.0 - 1.0
            v_c = ((2.0 + gy * float(f) + 0.5) / n_pad) * 2.0 - 1.0
            grid = torch.stack([u_c, v_c], dim=-1)[None]
            up = F.grid_sample(t_pad, grid, mode="bicubic", padding_mode="border", align_corners=False)[0, 0].numpy()
        out += amp * up
        norm += amp
        amp *= 0.52
    out /= max(norm, 1e-6)
    return (out - out.mean()) / (out.std() + 1e-6)


def _periodic_grad(h: np.ndarray):
    """Exact periodic central differences (zero boundary seam!)."""
    gx = 0.5 * (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1))
    gy = 0.5 * (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0))
    return gx.astype(np.float32), gy.astype(np.float32)


def _height_to_normal_periodic(h: np.ndarray, slope: float = 18.0) -> np.ndarray:
    """Convert periodic heightfield to OpenGL +Y tangent-space normal (resolution-invariant to 512px)."""
    gx, gy = _periodic_grad(h)
    eff_slope = float(slope) * (min(h.shape[0], h.shape[1]) / 512.0)
    return ng.slopes_to_normal(-gx * eff_slope, +gy * eff_slope)


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    if abs(e1 - e0) < 1e-6:
        return (x >= e0).astype(np.float32)
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _periodic_worley(u: np.ndarray, v: np.ndarray, n: int, seed: int):
    """Fast vectorized 100% periodic 2D Worley/Voronoi F1 & F2 distances on [0,1)^2."""
    rng = np.random.default_rng(seed)
    jx = rng.uniform(0.15, 0.85, (n, n)).astype(np.float32)
    jy = rng.uniform(0.15, 0.85, (n, n)).astype(np.float32)
    gx, gy = u * n, v * n
    ix = np.floor(gx).astype(np.int32)
    iy = np.floor(gy).astype(np.int32)
    fx, fy = gx - ix, gy - iy
    f1 = np.full_like(u, 1e6, dtype=np.float32)
    f2 = np.full_like(u, 1e6, dtype=np.float32)
    for dy in (-1, 0, 1):
        ny = (iy + dy) % n
        for dx in (-1, 0, 1):
            nx = (ix + dx) % n
            ddx = dx + jx[ny, nx] - fx
            ddy = dy + jy[ny, nx] - fy
            d = np.sqrt(ddx * ddx + ddy * ddy)
            m1 = d < f1
            f2 = np.where(m1, f1, np.minimum(f2, d))
            f1 = np.where(m1, d, f1)
    return f1, f2


def _lit_preview(N: np.ndarray, ao: np.ndarray, rough: np.ndarray, tint: tuple[float, float, float]) -> np.ndarray:
    """Fast physically-inspired Blinn-Phong + rim preview renderer so normal/roughness detail pops."""
    H, W = N.shape[:2]
    L1 = np.array([0.48, 0.52, 0.71], dtype=np.float32); L1 /= np.linalg.norm(L1)
    L2 = np.array([-0.60, -0.30, 0.74], dtype=np.float32); L2 /= np.linalg.norm(L2)
    V = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    H1 = L1 + V; H1 /= np.linalg.norm(H1)

    ndotl1 = np.clip(N[..., 0] * L1[0] + N[..., 1] * L1[1] + N[..., 2] * L1[2], 0.0, 1.0)
    ndotl2 = np.clip(N[..., 0] * L2[0] + N[..., 1] * L2[1] + N[..., 2] * L2[2], 0.0, 1.0)
    ndoth1 = np.clip(N[..., 0] * H1[0] + N[..., 1] * H1[1] + N[..., 2] * H1[2], 0.0, 1.0)

    shin = np.clip(2.0 / (np.maximum(rough, 0.08) ** 2.4) - 2.0, 2.0, 180.0)
    spec = (ndoth1 ** shin) * (1.0 - rough * 0.82) * 0.65

    base = np.array(tint, dtype=np.float32)[None, None, :]
    col = base * (0.20 + 0.78 * ndotl1[..., None] + 0.22 * ndotl2[..., None]) * (0.35 + 0.65 * ao[..., None])
    col += spec[..., None] * (0.4 + 0.6 * ao[..., None])
    col = np.power(np.clip(col, 0.0, 1.0), 1.0 / 2.2)
    return np.clip(col * 255.0 + 0.5, 0, 255).astype(np.uint8)


# =============================================================================
# 2. 12 Everyday Real-World Seamless Surface Details
# =============================================================================
DAILY_CATALOG = [
    ("corrugated_cardboard",     "Fluted kraft shipping cardboard with sinusoidal ribs & paper fibers",     "paper",   (0.62, 0.48, 0.34)),
    ("knurled_metal_grip",       "Diamond cross-hatch machined knurling (flashlights, dials, tool grips)", "metal",   (0.58, 0.60, 0.63)),
    ("woven_denim_twill",        "Diagonal 3x1 twill denim/canvas weave with interlocking yarn twist",     "fabric",  (0.24, 0.34, 0.48)),
    ("quilted_leather_stitch",   "Padded diamond-quilted booth/bag leather with recessed stitch seams",    "leather", (0.34, 0.21, 0.16)),
    ("acoustic_pegboard",        "Workshop pegboard / acoustic panel with chamfered circular hole grid",   "wood",    (0.68, 0.59, 0.46)),
    ("sidewalk_expansion_joint", "Broom-finished concrete sidewalk slabs with chamfered control joints",   "concrete",(0.62, 0.62, 0.60)),
    ("subway_metro_tile",        "Beveled running-bond ceramic subway tiles with recessed sanded grout",   "ceramic", (0.84, 0.86, 0.85)),
    ("rubber_tread_stair",       "Anti-slip metro/staircase coin-dot rubber floor mat with bevel studs",   "rubber",  (0.22, 0.23, 0.25)),
    ("ribbed_corduroy_velvet",   "Vertical wales corduroy fabric with rounded ridges & cross-weave",       "fabric",  (0.46, 0.31, 0.22)),
    ("waffle_radiator_fins",     "Apartment cast-iron radiator / HVAC vent fins with rounded ribs",        "metal",   (0.76, 0.75, 0.72)),
    ("wood_parquet_herringbone", "Oak herringbone parquet floorboards with micro-bevel seams & wood grain","wood",    (0.56, 0.40, 0.26)),
    ("carbon_fiber_weave",       "2x2 twill carbon-fiber tech weave with crisp over-under tow relief",     "carbon",  (0.18, 0.19, 0.21)),
    ("street_storm_grate",       "Street cast-iron storm-drain trench grate with staggered beveled slots", "metal",   (0.46, 0.48, 0.51)),
    ("diamond_tread_plate",      "Industrial/subway steel diamond checkerplate with alternating raised bars","metal", (0.50, 0.53, 0.56)),
    ("hvac_slotted_vent",        "Stamped architectural HVAC return-air louver vent grille with ribs",     "metal",   (0.68, 0.69, 0.68)),
    # --- 13 New Small-Scale Vents, Grilles, Plates, Meshes & Micro-Treads ---
    ("speaker_mesh_hex",         "Fine hexagonal perforated speaker / intercom / laptop mesh grille",      "metal",   (0.50, 0.53, 0.57)),
    ("expanded_metal_mesh",      "Diamond expanded-metal security screen & industrial walkway mesh",       "metal",   (0.42, 0.44, 0.46)),
    ("stamped_louver_micro",     "Fine stamped electronics / power-box cooling louver slot array",         "metal",   (0.36, 0.38, 0.40)),
    ("chainlink_wire_mesh",      "Tight woven galvanized wire mesh screen with over-under crossings",      "metal",   (0.56, 0.58, 0.60)),
    ("circular_perf_sheet",      "Staggered micro-punched stainless steel perforated vent sheet",          "metal",   (0.64, 0.66, 0.68)),
    ("corrugated_sheet_metal",   "Fine trapezoidal corrugated steel shutter / utility siding ribs",        "metal",   (0.54, 0.56, 0.58)),
    ("ribbed_rubber_mat",        "Fine parallel V-ribbed escalator / bus-aisle anti-slip rubber mat",      "rubber",  (0.055, 0.058, 0.064)),
    ("brushed_anodized_plate",   "Fine directional brushed aluminum elevator / appliance trim seam",       "metal",   (0.70, 0.72, 0.75)),
    ("embossed_braille_strip",   "Tactile railing / elevator braille dots & directional guide ribs",       "metal",   (0.66, 0.62, 0.50)),
    ("stamped_vent_slots_stagger","Staggered rounded-pill vent slots on street transformer cabinets",      "metal",   (0.46, 0.49, 0.46)),
    ("riveted_sheet_seam",       "Overlapping bus/train/aircraft sheet-metal lap seam with flush rivets",  "metal",   (0.60, 0.62, 0.65)),
    ("micro_hex_grip",           "Molded silicone / tool handle / camera body micro-hex grip pattern",     "rubber",  (0.068, 0.074, 0.084)),
    ("woven_nylon_strap",        "Tight herringbone ribbed seatbelt / backpack nylon webbing weave",       "fabric",  (0.10, 0.14, 0.11)),
    # --- 8 More Small-Scale Seamless Everyday Micro-Surfaces (36 Total!) ---
    ("knurled_straight_flutes",  "Fine vertical straight-knurled dial / pen / camera barrel flutes",       "metal",   (0.58, 0.60, 0.64)),
    ("dimpled_golf_grip",        "Concave spherical micro-dimpled grip / acoustic / sport handle surface", "rubber",  (0.032, 0.035, 0.042)),
    ("perforated_leather_auto",  "Car-seat / headphone micro-perforated pebble leather upholstery",        "leather", (0.085, 0.065, 0.055)),
    ("stair_nosing_grooves",     "Machined aluminum anti-slip stair-nosing tread channels with grit",      "metal",   (0.62, 0.64, 0.66)),
    ("heatsink_extrusion_fins",  "Tight extruded aluminum electronics heatsink cooling fin array",         "metal",   (0.32, 0.34, 0.37)),
    ("window_blinds_slats",      "Overlapping horizontal venetian-blind slats with ladder-cord ribs",      "metal",   (0.70, 0.71, 0.70)),
    ("acoustic_slat_wood",       "Vertical micro-slatted oak acoustic wall paneling over dark felt",       "wood",    (0.58, 0.44, 0.30)),
    ("drip_tray_sieve_grate",    "Stainless-steel coffee/espresso drip-tray bridge-lanced slotted grate",  "metal",   (0.70, 0.72, 0.75)),
    # --- 8 More Small-Scale Seamless Everyday Micro-Surfaces (44 Total!) ---
    ("modular_stud_grid",        "Precision-molded interlocking circular plastic stud grid",               "ceramic", (0.76, 0.22, 0.20)),
    ("waffle_silicone_pad",      "Molded kitchen/coaster square-waffle silicone grip mat",                 "rubber",  (0.24, 0.34, 0.38)),
    ("keyboard_keycap_grid",     "Tight grid of gently dished sculpted matte-plastic keycaps",             "rubber",  (0.22, 0.23, 0.26)),
    ("solar_cell_grid",          "Monocrystalline solar panel cells with silver busbars & micro-fingers",  "ceramic", (0.12, 0.17, 0.28)),
    ("pcb_solder_mask_traces",   "Electronics PCB surface with parallel copper traces & micro-vias",       "ceramic", (0.12, 0.34, 0.22)),
    ("acoustic_pyramid_foam",    "Studio/packaging convoluted acoustic pyramid foam micro-peaks",          "rubber",  (0.18, 0.19, 0.21)),
    ("terry_velcro_loops",       "Fine micro-looped hook-and-loop / terry textile surface",                "fabric",  (0.58, 0.54, 0.48)),
    ("reeded_coin_milling",      "Ultra-fine milled coin-edge / watch-bezel parallel micro-reeding",       "metal",   (0.95, 0.68, 0.24)),
    # --- 12 New Material Categories (Glass, Terrazzo, Rattan, Cork, Hammered Copper, Knit Wool, Mosaics, Asphalt, Plaster, Felt) ---
    ("fluted_reeded_glass",      "Architectural vertical fluted/reeded privacy glass ribs",                "glass",   (0.78, 0.86, 0.85)),
    ("terrazzo_speckled_stone",  "Honed terrazzo cement with multi-colored angular marble aggregate",      "stone",   (0.76, 0.74, 0.70)),
    ("woven_rattan_cane",        "Parisian cafe Thonet woven rattan cane webbing with natural fibers",     "natural", (0.68, 0.54, 0.34)),
    ("corkboard_pin_surface",    "Natural granulated office corkboard with compressed bark cells",         "natural", (0.58, 0.42, 0.27)),
    ("hammered_copper_dimple",   "Hand-planished hammered copper with organic concave hammer facets",      "copper",  (0.80, 0.44, 0.28)),
    ("cable_knit_wool",          "Chunky V-stitch knit wool sweater textile with interlocking yarn loops", "fabric",  (0.62, 0.58, 0.52)),
    ("penny_round_mosaic",       "Glazed teal circular penny-round ceramic mosaic tiles with grout",       "ceramic", (0.16, 0.44, 0.46)),
    ("hex_mosaic_tile",          "Classic 1-inch matte hexagonal ceramic bathroom floor mosaic tiles",     "ceramic", (0.86, 0.86, 0.84)),
    ("fine_asphalt_tarmac",      "Fine-aggregate street asphalt/tarmac with sharp dark mineral gravel",    "asphalt", (0.14, 0.14, 0.15)),
    ("venetian_plaster_trowel",  "Hand-troweled architectural lime plaster with sweeping trowel ridges",   "plaster", (0.82, 0.77, 0.70)),
    ("frosted_prism_diffuser",   "Ceiling prismatic acrylic light-diffuser panel with micro-pyramids",     "glass",   (0.84, 0.88, 0.92)),
    ("acoustic_kerf_felt",       "Compressed heather-gray PET acoustic felt with fine kerf slits",         "fabric",  (0.42, 0.44, 0.47)),
    # --- 12 More Material Categories (Patina Bronze, Marble, Tweed, Waffle Linen, Damascus, Fishscale, Burlap, Cast Iron, Velvet, Bamboo, Travertine, Gilded Tin) ---
    ("patinated_verdigris_bronze","Aged architectural bronze with turquoise verdigris patina in crevices", "bronze",  (0.48, 0.36, 0.22)),
    ("carrara_veined_marble",    "Polished white Italian Carrara marble with feathery blue-gray veins",    "stone",   (0.88, 0.88, 0.87)),
    ("herringbone_tweed_wool",   "Tailored charcoal & oatmeal diagonal chevron herringbone tweed weave",   "fabric",  (0.38, 0.36, 0.33)),
    ("waffle_cotton_linen",      "Deep 3D honeycomb waffle-weave sage cotton spa textile",                 "fabric",  (0.46, 0.54, 0.48)),
    ("damascus_folded_steel",    "Acid-etched folded Damascus steel with wavy topographic grain bands",    "metal",   (0.54, 0.56, 0.59)),
    ("mermaid_fishscale_tile",   "Glazed cobalt-emerald Moroccan scalloped fish-scale ceramic tiles",      "ceramic", (0.10, 0.36, 0.44)),
    ("burlap_jute_sackcloth",    "Coarse open-weave raw jute burlap with slubby golden fibers",            "natural", (0.58, 0.46, 0.30)),
    ("cast_iron_skillet_pebble", "Seasoned black cast-iron cookware pebble grain with satin oil sheen",    "metal",   (0.14, 0.14, 0.15)),
    ("tufted_emerald_velvet",    "Plush diamond-tufted emerald velvet upholstery with recessed buttons",   "fabric",  (0.08, 0.28, 0.22)),
    ("woven_bamboo_tatami",      "Traditional Japanese woven tatami / bamboo strip mat with reed grain",   "natural", (0.64, 0.60, 0.40)),
    ("travertine_porous_stone",  "Honed Roman travertine limestone with banded horizontal erosion pores",  "stone",   (0.76, 0.69, 0.58)),
    ("gilded_tin_ceiling_tile",  "Embossed vintage pressed-tin ceiling relief in antique champagne brass", "brass",   (0.74, 0.60, 0.34)),
    ("chainlink_fence_galvanized","Woven diamond galvanized steel chain-link fence with over-under crossings, light rust & transparent holes", "metal", (0.58, 0.61, 0.64)),
]

TRANSPARENT_MATERIALS = frozenset({"chainlink_fence_galvanized"})

WEAR_STATES = ("0_fresh", "1_lived_in", "2_weathered")


def build_daily_detail(detail_id: str, H: int, W: int, seed: int):
    """Returns (height [0..1], slope_scale, cavity_mask [0..1], base_roughness, ridge_mask [0..1])."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u = xx / float(W)
    v = yy / float(H)

    if detail_id == "corrugated_cardboard":
        # 24 vertical flutes + subtle horizontal paper-liner fibers + occasional slight crush wave
        flutes = 0.5 + 0.5 * np.sin(2.0 * math.pi * u * 24.0)
        fibers = _periodic_fbm(H, W, seed + 1, (16, 32, 64))
        crease = np.exp(-(((v - 0.5) / 0.018) ** 2)) * 0.28
        h = 0.55 * flutes - crease + 0.08 * fibers
        cav = np.clip((1.0 - flutes) * 0.7 + crease * 1.5, 0.0, 1.0)
        ridge = np.clip(flutes ** 2, 0.0, 1.0)
        return h, 14.0, cav, 0.78, ridge

    if detail_id == "knurled_metal_grip":
        # Classic 45-deg diamond machined knurling: sharp pyramid peaks + recessed V-grooves
        a = (u + v) * 14.0
        b = (u - v) * 14.0
        da = np.abs((a % 1.0) - 0.5) * 2.0
        db = np.abs((b % 1.0) - 0.5) * 2.0
        # Flat-topped truncated pyramids (like real lathe-cut knurling)
        pyr = np.minimum(1.0 - da, 1.0 - db)
        h = np.clip(pyr * 1.35, 0.0, 0.82)
        cav = _smoothstep(0.28, 0.02, h)
        ridge = _smoothstep(0.65, 0.82, h)
        return h, 14.0, cav, 0.28, ridge

    if detail_id == "woven_denim_twill":
        # Authentic fine 3x1 diagonal jeans denim twill: dense 56x56 subtle indigo warp & fine diagonal twill cords (ZERO chunky tweed blocks!)
        diag_cord = 0.5 + 0.5 * np.sin(2.0 * math.pi * (u * 32.0 + v * 32.0))
        warp_rib = 0.5 + 0.5 * np.cos(2.0 * math.pi * u * 64.0)
        weft_rib = 0.5 + 0.5 * np.cos(2.0 * math.pi * v * 64.0)
        # Subtle diagonal slubby cotton yarn texture (characteristic indigo jeans look!)
        slub_u = 0.06 * _periodic_fbm(H, W, seed + 3, (4, 32))
        slub_v = 0.04 * _periodic_fbm(H, W, seed + 4, (32, 4))
        fine_nap = 0.035 * _periodic_fbm(H, W, seed + 5, (32, 64))
        weave = 0.52 * diag_cord + 0.28 * warp_rib + 0.20 * weft_rib
        h = 0.28 + 0.52 * weave + slub_u + slub_v + fine_nap
        cav = _smoothstep(0.52, 0.28, h)
        ridge = _smoothstep(0.68, 0.90, h)
        return h.astype(np.float32), 6.5, cav.astype(np.float32), 0.86, ridge.astype(np.float32)

    if detail_id == "quilted_leather_stitch":
        # 4x4 diamond-quilted upholstery leather with plump pillowed domes + crisp saddle-stitch dashes + fine grain
        a = (u + v) * 4.0
        b = (u - v) * 4.0
        fa = np.abs((a % 1.0) - 0.5) * 2.0
        fb = np.abs((b % 1.0) - 0.5) * 2.0
        da = 1.0 - fa
        db = 1.0 - fb
        seam_d = np.minimum(da, db)
        pillow = np.sqrt(np.clip(1.0 - 0.74 * (fa * fa + fb * fb) * 0.5, 0.04, 1.0))
        seam_cut = np.exp(-((seam_d / 0.045) ** 2))
        st_a = np.exp(-((da / 0.016) ** 2)) * _smoothstep(0.15, 0.45, np.sin(2.0 * math.pi * b * 10.0))
        st_b = np.exp(-((db / 0.016) ** 2)) * _smoothstep(0.15, 0.45, np.sin(2.0 * math.pi * a * 10.0))
        stitches = np.maximum(st_a, st_b)
        pebble = _periodic_fbm(H, W, seed + 5, (24, 48, 64))
        h = pillow * (1.0 - 0.52 * seam_cut) + 0.16 * stitches + 0.025 * pebble * (1.0 - 0.7 * seam_cut)
        cav = np.clip(seam_cut * 1.10, 0.0, 1.0)
        ridge = np.clip(stitches * 1.2 + _smoothstep(0.76, 0.98, pillow) * 0.45, 0.0, 1.0)
        return h, 16.5, cav, 0.28, ridge

    if detail_id == "acoustic_pegboard":
        # 6x6 chamfered circular holes on a flat wood/hardboard panel
        n = 6.0
        fu = (u * n) % 1.0 - 0.5
        fv = (v * n) % 1.0 - 0.5
        r = np.sqrt(fu * fu + fv * fv)
        hole = _smoothstep(0.16, 0.24, r)  # 0 inside hole, 1 on board
        wood = 0.025 * _periodic_fbm(H, W, seed + 7, (8, 32, 64))
        h = 0.12 + 0.82 * hole + wood * hole
        cav = 1.0 - hole
        ridge = _smoothstep(0.19, 0.23, r) * (1.0 - _smoothstep(0.24, 0.29, r))
        return h, 20.0, cav, 0.58, ridge

    if detail_id == "sidewalk_expansion_joint":
        # Authentic broomed architectural Portland-cement sidewalk slabs: fine sandy grit + crisp control joint (ZERO amoeba pits!)
        du = np.minimum(u % 0.5, 0.5 - (u % 0.5))
        dv = np.minimum(v % 0.5, 0.5 - (v % 0.5))
        d_joint = np.minimum(du, dv)
        joint = _smoothstep(0.006, 0.022, d_joint)
        broom = 0.022 * np.sin(2.0 * math.pi * v * 48.0 + 0.4 * _periodic_fbm(H, W, seed + 9, (4, 8)))
        grit_coarse = _periodic_fbm(H, W, seed + 10, (12, 24, 48))
        grit_fine = _periodic_fbm(H, W, seed + 11, (32, 64))
        h = 0.20 + 0.68 * joint + (0.045 * grit_coarse + 0.032 * grit_fine + broom) * joint
        cav = np.clip((1.0 - joint) + 0.20 * _smoothstep(0.9, 1.8, -grit_fine), 0.0, 1.0)
        ridge = _smoothstep(0.018, 0.032, d_joint) * (1.0 - _smoothstep(0.032, 0.052, d_joint))
        return h, 12.0, cav, 0.84, ridge

    if detail_id == "subway_metro_tile":
        # Classic 2x4 running-bond beveled ceramic subway tiles (phase-centered off UV border)
        rows, cols = 6.0, 3.0
        vs = (v + 0.5 / rows) % 1.0
        row_idx = np.floor(vs * rows)
        u_shift = (u + 0.5 / cols + 0.5 * (row_idx % 2.0) / cols) % 1.0
        fu = (u_shift * cols) % 1.0
        fv = (vs * rows) % 1.0
        du = np.minimum(fu, 1.0 - fu) / cols
        dv = np.minimum(fv, 1.0 - fv) / rows
        d_edge = np.minimum(du, dv)
        grout = _smoothstep(0.004, 0.009, d_edge)
        bevel = _smoothstep(0.009, 0.032, d_edge)
        undul = 0.028 * _periodic_fbm(H, W, seed + 13, (4, 8))
        grout_grit = 0.04 * _periodic_fbm(H, W, seed + 14, (24, 48)) * (1.0 - grout)
        h = 0.14 + 0.24 * grout + 0.58 * bevel + undul * grout + grout_grit
        cav = 1.0 - grout
        ridge = _smoothstep(0.026, 0.034, d_edge) * (1.0 - _smoothstep(0.034, 0.048, d_edge))
        return h, 16.5, cav, 0.10, ridge

    if detail_id == "rubber_tread_stair":
        # Anti-slip coin-dot rubber mat (8x8 staggered circular flat-topped studs)
        n = 8.0
        row = np.floor(v * n)
        u_s = (u + 0.5 * (row % 2.0) / n) % 1.0
        fu = (u_s * n) % 1.0 - 0.5
        fv = (v * n) % 1.0 - 0.5
        r = np.sqrt(fu * fu + fv * fv)
        coin = 1.0 - _smoothstep(0.30, 0.38, r)
        # Subtle concentric dimple ring on top of each rubber coin
        ring = 0.06 * np.exp(-(((r - 0.22) / 0.04) ** 2))
        grit = 0.02 * _periodic_fbm(H, W, seed + 15, (32, 64))
        h = 0.18 + 0.70 * coin - ring + grit
        cav = 1.0 - coin
        ridge = _smoothstep(0.28, 0.34, r) * coin
        return h, 19.0, cav, 0.56, ridge

    if detail_id == "ribbed_corduroy_velvet":
        # 24 crisp rounded velvet wales + fine cross-weave velvet micro-nap
        wales = 0.5 + 0.5 * np.cos(2.0 * math.pi * u * 24.0)
        wales_r = np.power(np.clip(wales, 0.0, 1.0), 0.55)
        cross = 0.04 * np.sin(2.0 * math.pi * v * 48.0) * (0.4 + 0.6 * (1.0 - wales_r))
        nap = 0.03 * _periodic_fbm(H, W, seed + 17, (32, 64))
        h = 0.14 + 0.74 * wales_r + cross + nap
        cav = np.clip(1.0 - wales_r, 0.0, 1.0)
        ridge = _smoothstep(0.72, 0.98, wales_r)
        return h, 13.5, cav, 0.84, ridge

    if detail_id == "waffle_radiator_fins":
        # Vertical cast-iron radiator columns + horizontal tie-manifold ribs
        cols = np.power(np.clip(0.5 + 0.5 * np.cos(2.0 * math.pi * u * 10.0), 0.0, 1.0), 0.55)
        tie1 = np.exp(-(((v - 0.18) / 0.045) ** 2))
        tie2 = np.exp(-(((v - 0.82) / 0.045) ** 2))
        ties = np.maximum(tie1, tie2)
        h = np.maximum(0.12 + 0.75 * cols, 0.45 + 0.38 * ties)
        cav = _smoothstep(0.45, 0.15, h)
        ridge = _smoothstep(0.78, 0.92, h)
        return h, 20.0, cav, 0.34, ridge

    if detail_id == "wood_parquet_herringbone":
        # Authentic elongated 4:1 oak herringbone parquet in 45-deg rotated periodic lattice (N=8 -> EXACT torus periodicity!)
        N = 8.0
        rx = (u + v) * N
        ry = (u - v) * N
        ix = np.floor(rx)
        iy = np.floor(ry)
        fx = rx - ix
        fy = ry - iy
        d = (ix - iy) % 8.0
        is_h = d < 4.0
        px = np.where(is_h, d + fx, (d - 4.0) + fy)
        py = np.where(is_h, fy, fx)
        # Under u->u+1: (ix,iy)->(ix+8,iy+8); under v->v+1: (ix,iy)->(ix+8,iy-8) => mod N is 100% invariant!
        pid_x = np.where(is_h, (ix - d) % N, ix % N)
        pid_y = np.where(is_h, iy % N, (iy - (d - 4.0)) % N)
        seam_d = np.minimum(np.minimum(px, 4.0 - px), np.minimum(py, 1.0 - py))
        bevel = _smoothstep(0.018, 0.068, seam_d)
        # Organic flowing wood grain along each plank's long axis
        w1 = _periodic_fbm(H, W, seed + 21, (4, 8, 16))
        w2 = _periodic_fbm(H, W, seed + 22, (16, 32))
        g_h = np.sin(2.0 * math.pi * (ry * 4.0 + 0.25 * w1) + 0.65 * w2)
        g_v = np.sin(2.0 * math.pi * (rx * 4.0 + 0.25 * w1) + 0.65 * w2)
        grain = np.where(is_h, g_h, g_v) * 0.022
        plank_var = 0.030 * np.sin(pid_x * 13.7 + pid_y * 29.3)
        h = 0.24 + 0.64 * bevel + (grain + plank_var) * bevel
        cav = 1.0 - bevel
        ridge = _smoothstep(0.055, 0.10, seam_d) * (1.0 - _smoothstep(0.10, 0.16, seam_d))
        return h, 11.5, cav, 0.24, ridge

    if detail_id == "carbon_fiber_weave":
        # Tight glossy clearcoat 2x2 twill carbon-fiber epoxy weave (28x28 tows = 7 twill repeats, smooth wet clearcoat!)
        n = 28.0
        us = (u + 0.5 / n) % 1.0
        vs = (v + 0.5 / n) % 1.0
        iu, iv = np.floor(us * n), np.floor(vs * n)
        fu, fv = (us * n) % 1.0, (vs * n) % 1.0
        twill = ((iu + iv) % 4.0) < 2.0
        # Smooth resin-encapsulated flat-crowned carbon tows with soft shallow micro-valleys
        tow_u = np.sin(math.pi * fu) ** 0.24 * (0.88 + 0.12 * np.sin(math.pi * fv))
        tow_v = np.sin(math.pi * fv) ** 0.24 * (0.88 + 0.12 * np.sin(math.pi * fu))
        h = np.where(twill, 0.52 + 0.40 * tow_u, 0.48 + 0.38 * tow_v)
        cav = _smoothstep(0.62, 0.48, h) * 0.65
        ridge = _smoothstep(0.76, 0.92, h)
        return h.astype(np.float32), 5.5, cav.astype(np.float32), 0.07, ridge.astype(np.float32)

    if detail_id == "street_storm_grate":
        # Heavy cast-iron street drainage trench grate: staggered rectangular slots + beveled bars
        cols, rows = 8.0, 4.0
        r_idx = np.floor(v * rows)
        u_s = (u + 0.5 * (r_idx % 2.0) / cols) % 1.0
        fu = np.abs((u_s * cols) % 1.0 - 0.5) * 2.0  # 0 at slot center, 1 at bar center
        fv = np.abs((v * rows) % 1.0 - 0.5) * 2.0
        # Rounded-rect drainage opening inside each cell
        d_slot = np.maximum(fu - 0.56, fv - 0.72)
        bar = _smoothstep(-0.08, 0.08, d_slot)  # 0 inside open slot, 1 on cast-iron bar
        # Slight raised anti-slip crown along bar centers
        crown = 0.08 * np.clip(d_slot * 2.5, 0.0, 1.0)
        h = 0.06 + 0.80 * bar + crown * bar
        cav = 1.0 - bar
        ridge = _smoothstep(0.04, 0.14, d_slot) * (1.0 - _smoothstep(0.14, 0.26, d_slot))
        return h.astype(np.float32), 18.0, cav.astype(np.float32), 0.28, ridge.astype(np.float32)

    if detail_id == "diamond_tread_plate":
        # Classic street/subway steel diamond checkerplate (alternating perpendicular raised lozenge bars)
        n = 8.0
        iu, iv = np.floor(u * n), np.floor(v * n)
        fu = (u * n) % 1.0 - 0.5
        fv = (v * n) % 1.0 - 0.5
        # Rotate 45 deg inside each cell, alternating +45 / -45 by checkerboard parity
        par = ((iu + iv) % 2.0) < 0.5
        a1 = (fu + fv) * 0.70710678
        b1 = (fv - fu) * 0.70710678
        lx = np.where(par, a1, b1)
        ly = np.where(par, b1, a1)
        # Tapered almond/lozenge diamond bar SDF
        d_loz = np.sqrt((lx / 0.34) ** 2 + (ly / 0.115) ** 2)
        bar = 1.0 - _smoothstep(0.72, 1.05, d_loz)
        h = 0.20 + 0.72 * bar
        cav = _smoothstep(0.35, 0.22, h)
        ridge = _smoothstep(0.65, 0.90, bar)
        return h.astype(np.float32), 20.0, cav.astype(np.float32), 0.32, ridge.astype(np.float32)

    if detail_id == "hvac_slotted_vent":
        cols, rows = 3.0, 10.0
        fu = np.abs(((u * cols + 0.5) % 1.0) - 0.5) * 2.0
        fv = (v * rows + 0.5) % 1.0
        in_col = 1.0 - _smoothstep(0.84, 0.91, fu)
        slat_ramp = np.clip(fv / 0.78, 0.0, 1.0)
        slat_drop = 1.0 - _smoothstep(0.76, 0.86, fv)
        slat = slat_ramp * slat_drop
        rib = 1.0 - in_col
        h = np.maximum(0.16 + 0.72 * slat * in_col, 0.62 * rib)
        cav = np.clip((1.0 - slat_drop) * in_col, 0.0, 1.0)
        ridge = _smoothstep(0.68, 0.77, fv) * slat_drop * in_col
        return h.astype(np.float32), 15.0, cav.astype(np.float32), 0.28, ridge.astype(np.float32)

    if detail_id == "speaker_mesh_hex":
        # Tight hexagonal honeycomb perforated speaker/intercom mesh (16x16 staggered)
        nx, ny = 16.0, 14.0
        row = np.floor(v * ny)
        u_s = (u + 0.5 * (row % 2.0) / nx) % 1.0
        fu = (u_s * nx) % 1.0 - 0.5
        fv = (v * ny) % 1.0 - 0.5
        # Hex distance
        ax, ay = np.abs(fu), np.abs(fv)
        d_hex = np.maximum(ax * 0.866025 + ay * 0.5, ay)
        web = _smoothstep(0.32, 0.42, d_hex)  # 0 inside hole, 1 on hex web
        h = 0.06 + 0.84 * web
        cav = 1.0 - web
        ridge = _smoothstep(0.36, 0.44, d_hex)
        return h.astype(np.float32), 18.0, cav.astype(np.float32), 0.22, ridge.astype(np.float32)

    if detail_id == "expanded_metal_mesh":
        # Stretched diamond expanded-metal mesh with slight twist-angle strand relief
        a = (u * 12.0 + v * 6.0)
        b = (u * 12.0 - v * 6.0)
        da = np.abs((a % 1.0) - 0.5) * 2.0
        db = np.abs((b % 1.0) - 0.5) * 2.0
        strand = _smoothstep(0.64, 0.80, np.maximum(da, db))
        twist = 0.14 * np.sin(2.0 * math.pi * a) * strand
        h = 0.08 + 0.76 * strand + twist
        cav = 1.0 - strand
        ridge = _smoothstep(0.74, 0.92, np.maximum(da, db))
        return h.astype(np.float32), 21.0, cav.astype(np.float32), 0.38, ridge.astype(np.float32)

    if detail_id == "stamped_louver_micro":
        # Fine electronics / street cabinet stamped louver rows (6 cols x 18 rows)
        cols, rows = 6.0, 18.0
        fu = np.abs((u * cols) % 1.0 - 0.5) * 2.0
        fv = (v * rows) % 1.0
        in_slot = 1.0 - _smoothstep(0.76, 0.86, fu)
        hood = np.sin(math.pi * np.clip(fv / 0.72, 0.0, 1.0)) * (fv < 0.72)
        open_gap = _smoothstep(0.68, 0.76, fv) * (1.0 - _smoothstep(0.88, 0.96, fv)) * in_slot
        h = 0.45 + 0.42 * hood * in_slot - 0.36 * open_gap
        cav = open_gap
        ridge = _smoothstep(0.60, 0.95, hood) * in_slot
        return h.astype(np.float32), 19.0, cav.astype(np.float32), 0.34, ridge.astype(np.float32)

    if detail_id == "chainlink_fence_galvanized":
        # High-detail woven diamond galvanized steel chain-link fence with real over-under crossings,
        # uniform cylindrical wire thickness (slope-corrected orthogonal distance), zinc spangle relief,
        # and crisp open holes (cav = 1 - wire_mask -> opacity.png cutout).
        nx, ny = 5.0, 5.0
        phase = 2.0 * math.pi * v * ny
        # Smoothly rounded triangle wave for classic diamond chain-link wire bends
        s_phi = np.sin(phase)
        c_phi = np.cos(phase)
        amp = 0.285
        # Use arcsin(0.94 * sin) for crisp diamond straight runs with smooth rounded hooks at knuckles
        zv_a = amp * (2.0 / math.pi) * np.arcsin(0.94 * s_phi)
        zv_b = -zv_a
        # Derivative dz/dv (in cell units) to compute orthogonal distance to the diagonal wire!
        dz_dv = amp * (2.0 / math.pi) * (0.94 * c_phi * 2.0 * math.pi * (ny / nx)) / np.sqrt(np.maximum(0.06, 1.0 - (0.94 * s_phi) ** 2))
        ortho_scale = 1.0 / np.sqrt(1.0 + dz_dv * dz_dv)

        da_h = np.abs((((u * nx - 0.25 - zv_a) + 0.5) % 1.0) - 0.5)
        db_h = np.abs((((u * nx - 0.75 - zv_b) + 0.5) % 1.0) - 0.5)
        da = da_h * ortho_scale
        db = db_h * ortho_scale

        r_wire = 0.088
        wa = np.sqrt(np.clip(1.0 - (da / r_wire) ** 2, 0.0, 1.0))
        wb = np.sqrt(np.clip(1.0 - (db / r_wire) ** 2, 0.0, 1.0))

        # Helical Z-depth: wire A passes OVER wire B at upper knuckle (cos > 0) and UNDER at lower knuckle (cos < 0)
        za = 0.68 + 0.28 * c_phi
        zb = 0.68 - 0.28 * c_phi
        ha = wa * za
        hb = wb * zb
        upper_a = ha >= hb
        wire_h = np.maximum(ha, hb)

        # Fine longitudinal drawing/twist striations + zinc spangle micro-relief on wire surface
        spangle = _periodic_worley(u, v, 40, seed + 11)[0]
        twist_groove = 0.014 * np.sin(2.0 * math.pi * (u * nx * 6.0 + v * ny * 6.0))
        wire_mask = _smoothstep(0.02, 0.14, np.maximum(wa, wb))

        # Slight contact-collar bevel where wires hook together
        knuckle = np.minimum(wa, wb)
        wire_h = np.clip(wire_h + 0.05 * knuckle + (0.018 * (1.0 - spangle) + twist_groove) * wire_mask, 0.0, 1.0)

        h = 0.02 + 0.92 * wire_h
        # cav = 1.0 in open diamond holes, 0.0 on solid steel wires; subtle dip at under-wire crossing
        under_crease = np.where(upper_a, wb * 0.15, wa * 0.15) * np.clip(knuckle * 4.0, 0.0, 1.0)
        cav = np.clip((1.0 - wire_mask) + under_crease * 0.25, 0.0, 1.0)
        ridge = _smoothstep(0.62, 0.92, wire_h) * wire_mask
        return h.astype(np.float32), 20.0, cav.astype(np.float32), 0.24, ridge.astype(np.float32)

    if detail_id == "chainlink_wire_mesh":
        # Authentic interlocking helical zigzag galvanized steel chain-link fence over deep open void!
        # Vertical spiral wire columns at pitch nx=6, bending in zigzags at pitch ny=6 and hooking into neighbors
        nx, ny = 6.0, 6.0
        # Two interlocked sets of zigzag wires shifted by half-cell horizontally & vertically
        # Wire set A: zigzag center x_A(v) oscillates +-0.30 around integer columns (u*nx)
        zig_a = 0.28 * (2.0 * np.abs(((v * ny) % 1.0) - 0.5) - 0.5) * 2.0  # triangle wave in [-0.28, +0.28]
        # Smooth out the sharp triangle corners into authentic bent wire hooks
        zv_a = 0.27 * (2.0 / math.pi) * np.arcsin(0.92 * np.sin(2.0 * math.pi * v * ny))
        zv_b = -zv_a
        # Distance to wire A (columns centered at 0.25) and wire B (columns centered at 0.75)
        da = np.abs((((u * nx - 0.25 - zv_a) + 0.5) % 1.0) - 0.5)
        db = np.abs((((u * nx - 0.75 - zv_b) + 0.5) % 1.0) - 0.5)
        r_wire = 0.095
        wa = np.sqrt(np.clip(1.0 - (da / r_wire) ** 2, 0.0, 1.0))
        wb = np.sqrt(np.clip(1.0 - (db / r_wire) ** 2, 0.0, 1.0))
        # Helical Z-depth modulation so wire A hooks OVER wire B at upper knuckle and UNDER at lower knuckle!
        za = 0.74 + 0.24 * np.cos(2.0 * math.pi * v * ny)
        zb = 0.74 - 0.24 * np.cos(2.0 * math.pi * v * ny)
        ha = wa * za
        hb = wb * zb
        # Knuckle twist collar where the two hooks interlock
        knuckle = np.minimum(wa, wb)
        wire_h = np.clip(np.maximum(ha, hb) + 0.12 * knuckle, 0.0, 1.0)
        wire_mask = _smoothstep(0.05, 0.22, np.maximum(wa, wb))
        h = 0.02 + 0.90 * wire_h
        cav = 1.0 - wire_mask
        ridge = _smoothstep(0.65, 0.92, wire_h) * wire_mask
        return h.astype(np.float32), 18.0, cav.astype(np.float32), 0.20, ridge.astype(np.float32)

    if detail_id == "circular_perf_sheet":
        # Staggered micro-punched stainless steel perforated sheet (18x18)
        n = 18.0
        row = np.floor(v * n)
        u_s = (u + 0.5 * (row % 2.0) / n) % 1.0
        fu = (u_s * n) % 1.0 - 0.5
        fv = (v * n) % 1.0 - 0.5
        r = np.sqrt(fu * fu + fv * fv)
        sheet = _smoothstep(0.26, 0.34, r)
        h = 0.08 + 0.84 * sheet
        cav = 1.0 - sheet
        ridge = _smoothstep(0.30, 0.35, r) * (1.0 - _smoothstep(0.36, 0.42, r))
        return h.astype(np.float32), 19.0, cav.astype(np.float32), 0.26, ridge.astype(np.float32)

    if detail_id == "corrugated_sheet_metal":
        # Smooth sinusoidal-trapezoidal galvanized corrugated steel panel (10 ribs, flat-plateau border-aligned)
        t = (u * 10.0 + 0.43) % 1.0
        trap = np.minimum(_smoothstep(0.06, 0.30, t), 1.0 - _smoothstep(0.56, 0.80, t))
        galv = 0.015 * _periodic_fbm(H, W, seed + 37, (8, 16, 32))
        h = 0.16 + 0.72 * trap + galv
        cav = 1.0 - trap
        ridge = _smoothstep(0.82, 0.99, trap)
        return h.astype(np.float32), 12.0, cav.astype(np.float32), 0.22, ridge.astype(np.float32)

    if detail_id == "ribbed_rubber_mat":
        # Clean continuous parallel V-ribbed anti-slip industrial rubber mat (16 ribs, zero vertical split seams!)
        t_raw = np.abs(((v * 16.0 + 0.25) % 1.0) - 0.5) * 2.0
        t = np.sqrt(t_raw * t_raw + 0.020) / math.sqrt(1.020)
        rib = np.clip(1.0 - t, 0.0, 1.0) ** 0.82
        grain = 0.022 * _periodic_fbm(H, W, seed + 39, (24, 48))
        h = 0.16 + 0.70 * rib + grain
        cav = 1.0 - rib
        ridge = _smoothstep(0.72, 0.96, rib)
        return h.astype(np.float32), 13.5, cav.astype(np.float32), 0.68, ridge.astype(np.float32)

    if detail_id == "brushed_anodized_plate":
        # Crisp strictly linear directional brushed aluminum elevator panel (ZERO wavy noise!) + expansion seam & screws
        # 1D periodic high-freq streak table along v, gently broken up along u
        rng_b = np.random.default_rng(seed + 41)
        b_row = rng_b.standard_normal(H).astype(np.float32)
        # Mild 3-tap periodic smoothing so streaks are anti-aliased yet crisp
        b_row = 0.25 * np.roll(b_row, -1) + 0.50 * b_row + 0.25 * np.roll(b_row, 1)
        b_mod = 0.65 + 0.35 * _periodic_fbm(H, W, seed + 42, (2, 4))
        brush = 0.011 * b_row[:, None] * b_mod
        seam_v = np.exp(-(((v - 0.5) / 0.008) ** 2))
        seam_u = np.exp(-(((u - 0.5) / 0.008) ** 2))
        seams = np.maximum(seam_v, seam_u)
        screws = np.zeros_like(u)
        for sx in (0.44, 0.56):
            for sy in (0.44, 0.56):
                r = np.sqrt((u - sx) ** 2 + (v - sy) ** 2)
                screws = np.maximum(screws, 1.0 - _smoothstep(0.012, 0.018, r))
        h = 0.62 + brush - 0.42 * seams - 0.18 * screws
        cav = np.clip(seams + 0.6 * screws, 0.0, 1.0)
        ridge = _smoothstep(0.012, 0.020, np.minimum(np.abs(u - 0.5), np.abs(v - 0.5))) * (1.0 - _smoothstep(0.020, 0.032, np.minimum(np.abs(u - 0.5), np.abs(v - 0.5))))
        return h.astype(np.float32), 14.0, cav.astype(np.float32), 0.22, ridge.astype(np.float32)

    if detail_id == "embossed_braille_strip":
        # Tactile elevator / transit railing strip: 3 parallel guide ribs + 2x3 braille dot cells
        rib_pos = (0.14, 0.22, 0.78, 0.86)
        ribs = np.zeros_like(u)
        for rp in rib_pos:
            ribs = np.maximum(ribs, 1.0 - _smoothstep(0.012, 0.022, np.abs(v - rp)))
        # Braille cells along the middle band (v in [0.35, 0.65])
        dots = np.zeros_like(u)
        rng = np.random.default_rng(seed + 45)
        for ci in range(8):
            cx0 = (ci + 0.5) / 8.0
            for dx in (-0.022, 0.022):
                for dy in (-0.075, 0.0, 0.075):
                    if rng.random() > 0.35:
                        r = np.sqrt((u - (cx0 + dx)) ** 2 + (v - (0.5 + dy)) ** 2)
                        dots = np.maximum(dots, 1.0 - _smoothstep(0.011, 0.018, r))
        h = 0.24 + 0.58 * np.maximum(ribs, dots)
        cav = _smoothstep(0.32, 0.24, h)
        ridge = np.maximum(ribs, dots)
        return h.astype(np.float32), 18.0, cav.astype(np.float32), 0.22, ridge.astype(np.float32)

    if detail_id == "stamped_vent_slots_stagger":
        # Staggered rounded-pill ventilation slots on street transformer / utility boxes
        cols, rows = 4.0, 12.0
        r_idx = np.floor(v * rows)
        u_s = (u + 0.5 * (r_idx % 2.0) / cols) % 1.0
        fu = (u_s * cols) % 1.0 - 0.5
        fv = (v * rows) % 1.0 - 0.5
        # Pill SDF in cell coords
        px_c = np.maximum(np.abs(fu) - 0.26, 0.0)
        d_pill = np.sqrt(px_c * px_c + (fv * 0.35) ** 2)
        slot = 1.0 - _smoothstep(0.065, 0.095, d_pill)
        h = 0.80 - 0.72 * slot
        cav = slot
        ridge = _smoothstep(0.085, 0.11, d_pill) * (1.0 - _smoothstep(0.11, 0.15, d_pill))
        return h.astype(np.float32), 20.0, cav.astype(np.float32), 0.36, ridge.astype(np.float32)

    if detail_id == "riveted_sheet_seam":
        # Seamless aerospace/train riveted aluminum lap-strap butt joint with double flush rivet rows
        strap = _smoothstep(0.32, 0.35, v) * (1.0 - _smoothstep(0.65, 0.68, v))
        seam_groove = np.exp(-(((v - 0.50) / 0.008) ** 2))
        strap_edges = np.exp(-(((v - 0.335) / 0.007) ** 2)) + np.exp(-(((v - 0.665) / 0.007) ** 2))
        rivets = np.zeros_like(u)
        for ry in (0.415, 0.585):
            fu = (u * 10.0) % 1.0 - 0.5
            r = np.sqrt((fu / 10.0) ** 2 + (v - ry) ** 2)
            dome = np.sqrt(np.clip(1.0 - (r / 0.020) ** 2, 0.0, 1.0))
            ring = np.exp(-(((r - 0.021) / 0.004) ** 2)) * 0.25
            rivets = np.maximum(rivets, dome - ring)
        roll_grain = 0.014 * _periodic_fbm(H, W, seed + 47, (8, 32))
        h = 0.42 + 0.16 * strap - 0.20 * seam_groove - 0.06 * strap_edges + 0.26 * rivets + roll_grain
        cav = np.clip(seam_groove + 0.6 * strap_edges, 0.0, 1.0)
        ridge = np.clip(rivets + 0.4 * strap * (1.0 - seam_groove), 0.0, 1.0)
        return h.astype(np.float32), 14.0, cav.astype(np.float32), 0.24, ridge.astype(np.float32)

    if detail_id == "micro_hex_grip":
        # Modern molded silicone / tool / camera body micro-hex grip (20x18 raised chamfered hex islands)
        nx, ny = 20.0, 18.0
        row = np.floor(v * ny)
        u_s = (u + 0.5 * (row % 2.0) / nx) % 1.0
        fu = (u_s * nx) % 1.0 - 0.5
        fv = (v * ny) % 1.0 - 0.5
        ax, ay = np.abs(fu), np.abs(fv)
        d_hex = np.maximum(ax * 0.866025 + ay * 0.5, ay)
        pad_h = 1.0 - _smoothstep(0.32, 0.43, d_hex)
        h = 0.16 + 0.72 * pad_h
        cav = 1.0 - pad_h
        ridge = _smoothstep(0.75, 0.98, pad_h)
        return h.astype(np.float32), 18.0, cav.astype(np.float32), 0.66, ridge.astype(np.float32)

    if detail_id == "woven_nylon_strap":
        # Clean band-limited herringbone woven seatbelt nylon webbing (zero moire, tactile matte fabric)
        cols = 4.0
        us = (u + 0.5 / cols) % 1.0
        col = np.floor(us * cols)
        sgn = np.where((col % 2.0) < 0.5, 1.0, -1.0)
        rib = 0.5 + 0.5 * np.sin(2.0 * math.pi * (v * 20.0 + sgn * us * 8.0))
        cross = 0.12 * (0.5 + 0.5 * np.sin(2.0 * math.pi * us * 32.0))
        h = 0.22 + 0.60 * (rib ** 0.78) + cross
        cav = 1.0 - rib
        ridge = _smoothstep(0.72, 0.98, rib)
        return h.astype(np.float32), 13.0, cav.astype(np.float32), 0.76, ridge.astype(np.float32)

    if detail_id == "knurled_straight_flutes":
        # Crisp machined straight-knurled dial / camera barrel flutes (24 C1-continuous metallic ridges)
        t_raw = np.abs(((u * 24.0 + 0.25) % 1.0) - 0.5) * 2.0
        t = np.sqrt(t_raw * t_raw + 0.022) / math.sqrt(1.022)
        flute = np.clip(1.0 - t, 0.0, 1.0) ** 0.78
        h = 0.18 + 0.70 * flute
        cav = 1.0 - flute
        ridge = _smoothstep(0.74, 0.96, flute)
        return h.astype(np.float32), 14.0, cav.astype(np.float32), 0.20, ridge.astype(np.float32)

    if detail_id == "dimpled_golf_grip":
        # Tactile matte dark charcoal rubber concave micro-dimpled grip (soft edges, matte elastomer micro-grain)
        n = 16.0
        row = np.floor(v * n)
        u_s = (u + 0.5 * (row % 2.0) / n) % 1.0
        fu = (u_s * n) % 1.0 - 0.5
        fv = (v * n) % 1.0 - 0.5
        r = np.sqrt(fu * fu + fv * fv)
        dimple = np.sqrt(np.clip(1.0 - (r / 0.36) ** 2, 0.0, 1.0))
        grain = 0.024 * _periodic_fbm(H, W, seed + 50, (32, 64))
        h = 0.80 - 0.56 * dimple + grain
        cav = dimple
        ridge = _smoothstep(0.32, 0.39, r) * (1.0 - _smoothstep(0.39, 0.46, r)) * 0.35
        return h.astype(np.float32), 13.0, cav.astype(np.float32), 0.78, ridge.astype(np.float32)

    if detail_id == "perforated_leather_auto":
        # Luxury automotive micro-perforated pebble-grain leather: true Voronoi cellular leather pebbles + clean punched holes
        n = 12.0
        row = np.floor(v * n)
        u_s = (u + 0.5 * (row % 2.0) / n) % 1.0
        fu = (u_s * n) % 1.0 - 0.5
        fv = (v * n) % 1.0 - 0.5
        r = np.sqrt(fu * fu + fv * fv)
        hole = 1.0 - _smoothstep(0.14, 0.22, r)
        # Organic rounded Voronoi leather pebble cells (creases where F2 - F1 is small!)
        f1, f2 = _periodic_worley(u, v, 36, seed + 51)
        crease = 1.0 - _smoothstep(0.02, 0.16, f2 - f1)
        pebble = np.clip(1.0 - 0.65 * crease - 0.35 * f1, 0.0, 1.0)
        h = 0.72 + 0.14 * pebble * (1.0 - hole) - 0.68 * hole
        cav = np.clip(hole + 0.16 * crease * (1.0 - hole), 0.0, 1.0)
        ridge = np.clip(_smoothstep(0.62, 0.92, pebble) * (1.0 - hole), 0.0, 1.0)
        return h.astype(np.float32), 14.0, cav.astype(np.float32), 0.38, ridge.astype(np.float32)

    if detail_id == "stair_nosing_grooves":
        # Machined aluminum anti-slip stair-nosing: clean extruded ribs & fine parallel micro-serrated channels
        t = (v * 10.0) % 1.0
        rib = np.minimum(_smoothstep(0.06, 0.14, t), 1.0 - _smoothstep(0.42, 0.50, t))
        # Fine 1D extruded micro-serrations inside each anti-slip channel (zero crumpled FBM distortion)
        serr = (0.5 + 0.5 * np.sin(2.0 * math.pi * v * 120.0)) * (1.0 - rib)
        h = 0.22 + 0.66 * rib + 0.045 * serr
        cav = 1.0 - rib
        ridge = _smoothstep(0.85, 0.99, rib)
        return h.astype(np.float32), 14.0, cav.astype(np.float32), 0.22, ridge.astype(np.float32)

    if detail_id == "heatsink_extrusion_fins":
        # Extruded anodized aluminum heatsink cooling fins (8 bold symmetric precision fins -> zero aliasing/barcode!)
        t = (u * 8.0 + 0.5) % 1.0
        fin = np.minimum(_smoothstep(0.14, 0.26, t), 1.0 - _smoothstep(0.74, 0.86, t))
        # Subtle extruded crown across each fin top + fine vertical die-extrusion micro-streaks
        crown = 0.08 * np.sin(math.pi * np.clip((t - 0.14) / 0.72, 0.0, 1.0)) * fin
        h = 0.10 + 0.78 * fin + crown
        cav = 1.0 - fin
        ridge = _smoothstep(0.76, 0.98, fin)
        return h.astype(np.float32), 14.0, cav.astype(np.float32), 0.20, ridge.astype(np.float32)

    if detail_id == "window_blinds_slats":
        # Overlapping horizontal venetian-blind slats (12 phase-centered slats) + twin vertical ladder-cord ribs
        fv = (v * 12.0 + 0.5) % 1.0
        slat = np.clip(fv / 0.88, 0.0, 1.0) * (1.0 - _smoothstep(0.86, 0.96, fv))
        cord = np.maximum(
            1.0 - _smoothstep(0.008, 0.016, np.abs(u - 0.25)),
            1.0 - _smoothstep(0.008, 0.016, np.abs(u - 0.75)),
        )
        h = np.maximum(0.16 + 0.70 * slat, 0.58 * cord)
        cav = _smoothstep(0.84, 0.94, fv)
        ridge = _smoothstep(0.75, 0.86, fv) * (1.0 - cav)
        return h.astype(np.float32), 14.0, cav.astype(np.float32), 0.28, ridge.astype(np.float32)

    if detail_id == "acoustic_slat_wood":
        # Modern architectural vertical micro-slatted oak acoustic wall paneling over dark felt valleys
        t = (u * 12.0) % 1.0
        slat = np.minimum(_smoothstep(0.14, 0.20, t), 1.0 - _smoothstep(0.76, 0.82, t))
        grain = 0.035 * np.sin(2.0 * math.pi * u * 72.0 + 1.4 * _periodic_fbm(H, W, seed + 61, (8, 16))) * slat
        felt = 0.04 * _periodic_fbm(H, W, seed + 62, (32, 64)) * (1.0 - slat)
        h = 0.12 + 0.76 * slat + grain + felt
        cav = 1.0 - slat
        ridge = _smoothstep(0.85, 0.99, slat)
        return h.astype(np.float32), 20.0, cav.astype(np.float32), 0.46, ridge.astype(np.float32)

    if detail_id == "drip_tray_sieve_grate":
        cols, rows = 10.0, 6.0
        r_idx = np.floor(v * rows)
        u_s = (u + 0.5 * (r_idx % 2.0) / cols) % 1.0
        fu = (u_s * cols) % 1.0 - 0.5
        fv = (v * rows) % 1.0 - 0.5
        py_c = np.maximum(np.abs(fv) - 0.26, 0.0)
        d_pill = np.sqrt((fu * 0.62) ** 2 + py_c * py_c)
        slot = 1.0 - _smoothstep(0.085, 0.125, d_pill)
        h = 0.84 - 0.76 * slot
        cav = slot
        ridge = _smoothstep(0.11, 0.14, d_pill) * (1.0 - _smoothstep(0.14, 0.19, d_pill))
        return h.astype(np.float32), 21.0, cav.astype(np.float32), 0.22, ridge.astype(np.float32)

    if detail_id == "modular_stud_grid":
        # Precision-molded 8x8 interlocking circular plastic studs with crisp chamfered tops
        n = 8.0
        fu = (u * n) % 1.0 - 0.5
        fv = (v * n) % 1.0 - 0.5
        r = np.sqrt(fu * fu + fv * fv)
        stud = 1.0 - _smoothstep(0.28, 0.34, r)
        h = 0.18 + 0.72 * stud
        cav = 1.0 - stud
        ridge = _smoothstep(0.26, 0.31, r) * stud
        return h.astype(np.float32), 20.0, cav.astype(np.float32), 0.16, ridge.astype(np.float32)

    if detail_id == "waffle_silicone_pad":
        # Everyday kitchen/desk square-waffle molded silicone mat (10x10 phase-centered crisp pockets)
        n = 10.0
        fu = np.abs(((u * n + 0.5) % 1.0) - 0.5) * 2.0
        fv = np.abs(((v * n + 0.5) % 1.0) - 0.5) * 2.0
        wall = _smoothstep(0.66, 0.84, np.maximum(fu, fv))
        h = 0.16 + 0.72 * wall
        cav = 1.0 - wall
        ridge = _smoothstep(0.82, 0.98, np.maximum(fu, fv))
        return h.astype(np.float32), 14.0, cav.astype(np.float32), 0.48, ridge.astype(np.float32)

    if detail_id == "keyboard_keycap_grid":
        # Tight 6x6 grid of sculpted gently-dished matte-plastic keycaps with narrow crisp gaps
        n = 6.0
        fu = (u * n) % 1.0 - 0.5
        fv = (v * n) % 1.0 - 0.5
        d_box = np.maximum(np.abs(fu), np.abs(fv))
        cap = 1.0 - _smoothstep(0.36, 0.46, d_box)
        # Subtle spherical fingertip dish in the center of each keycap
        dish = 0.14 * np.exp(-((fu * fu + fv * fv) / 0.08))
        h = 0.10 + (0.78 - dish) * cap
        cav = 1.0 - cap
        ridge = _smoothstep(0.30, 0.38, d_box) * cap
        return h.astype(np.float32), 19.0, cav.astype(np.float32), 0.36, ridge.astype(np.float32)

    if detail_id == "solar_cell_grid":
        # Monocrystalline solar panel cells (2x2 chamfered-corner cells + silver busbars + micro-fingers)
        n = 2.0
        fu = np.abs((u * n) % 1.0 - 0.5) * 2.0
        fv = np.abs((v * n) % 1.0 - 0.5) * 2.0
        cell_gap = np.maximum(np.maximum(fu, fv) - 0.94, (fu + fv) - 1.72)
        in_cell = 1.0 - _smoothstep(-0.02, 0.02, cell_gap)
        busbars = np.maximum(
            1.0 - _smoothstep(0.010, 0.020, np.abs(fu - 0.38)),
            1.0 - _smoothstep(0.010, 0.020, np.abs(fu - 0.78)),
        ) * in_cell
        fingers = (0.5 + 0.5 * np.cos(2.0 * math.pi * v * 96.0)) * in_cell * 0.06
        h = 0.20 + 0.58 * in_cell + 0.14 * busbars + fingers
        cav = 1.0 - in_cell
        ridge = busbars
        return h.astype(np.float32), 16.0, cav.astype(np.float32), 0.14, ridge.astype(np.float32)

    if detail_id == "pcb_solder_mask_traces":
        # Authentic varied electronics PCB: 100% periodic 2x2 distinct Manhattan/45-deg copper trace buses + IC pads + vias!
        # Keep v=0 and v=1 strictly flat vertical bus crossings so wrap seam is EXACTLY zero at any resolution (256..2048)!
        # Vertical 8-track bus with opposite-direction 45-deg bends in upper vs lower half -> varied organic circuit layout!
        bend_a = np.clip((v - 0.14) / 0.16, 0.0, 1.0) * (1.0 - np.clip((v - 0.36) / 0.12, 0.0, 1.0)) * 0.125
        bend_b = -np.clip((v - 0.58) / 0.14, 0.0, 1.0) * (1.0 - np.clip((v - 0.78) / 0.12, 0.0, 1.0)) * 0.125
        u_routed = (u + bend_a + bend_b + 0.03125) % 1.0
        # 16 vertical copper traces grouped into 4-trace differential pairs with gaps
        t_cell = (u_routed * 16.0) % 1.0 - 0.5
        t_idx = np.floor(u_routed * 16.0).astype(np.int32)
        bus_keep = ((t_idx % 4) != 3).astype(np.float32)
        trace_v = (1.0 - _smoothstep(0.15, 0.23, np.abs(t_cell))) * bus_keep
        # Horizontal cross-buses at v=0.49 and v=0.91 (well away from v=0/1 wrap border!)
        trace_h1 = (1.0 - _smoothstep(0.008, 0.015, np.abs(v - 0.49))) * _smoothstep(0.12, 0.16, u) * (1.0 - _smoothstep(0.84, 0.88, u))
        trace_h2 = (1.0 - _smoothstep(0.008, 0.015, np.abs(v - 0.91))) * (1.0 - _smoothstep(0.32, 0.36, np.abs(u - 0.5)))
        # Asymmetric IC SMD rectangular pad row around v=0.28 and v=0.72 + staggered gold-ringed vias
        smd_y = np.minimum(np.abs(v - 0.28), np.abs(v - 0.72))
        smd_x = np.abs(((u * 12.0 + 0.5) % 1.0) - 0.5)
        smd_mask = ((np.abs(u - 0.27) < 0.14) | (np.abs(u - 0.73) < 0.14)).astype(np.float32)
        smd_pad = (1.0 - _smoothstep(0.022, 0.032, smd_y)) * (1.0 - _smoothstep(0.22, 0.30, smd_x)) * smd_mask
        # Staggered via grid (all interior, zero border-crossing!)
        via_pts = [
            (0.12, 0.14), (0.38, 0.16), (0.62, 0.14), (0.88, 0.18),
            (0.18, 0.42), (0.46, 0.49), (0.54, 0.49), (0.82, 0.42),
            (0.14, 0.62), (0.42, 0.64), (0.64, 0.62), (0.86, 0.66),
            (0.24, 0.86), (0.50, 0.91), (0.76, 0.86),
        ]
        pad = smd_pad.copy()
        hole = np.zeros_like(u)
        for vx, vy in via_pts:
            rv = np.sqrt((u - vx) ** 2 + (v - vy) ** 2)
            pad = np.maximum(pad, 1.0 - _smoothstep(0.020, 0.027, rv))
            hole = np.maximum(hole, 1.0 - _smoothstep(0.008, 0.013, rv))
        traces = np.maximum(trace_v, np.maximum(trace_h1, trace_h2))
        copper = np.clip(np.maximum(traces * 0.85, pad) * (1.0 - hole), 0.0, 1.0)
        h = 0.44 + 0.16 * traces + 0.24 * pad - 0.48 * hole
        cav = hole
        ridge = copper
        return h.astype(np.float32), 11.0, cav.astype(np.float32), 0.18, ridge.astype(np.float32)

    if detail_id == "acoustic_pyramid_foam":
        # Studio convoluted acoustic pyramid foam (8x8 soft-tip pyramids, quarter-cell phase-centered)
        n = 8.0
        fu = np.abs(((u * n + 0.25) % 1.0) - 0.5) * 2.0
        fv = np.abs(((v * n + 0.25) % 1.0) - 0.5) * 2.0
        pyr = np.clip(1.0 - np.maximum(fu, fv), 0.0, 1.0) ** 0.90
        pores = 0.035 * _periodic_fbm(H, W, seed + 71, (24, 48))
        h = 0.14 + 0.76 * pyr + pores
        cav = 1.0 - pyr
        ridge = _smoothstep(0.72, 0.96, pyr)
        return h.astype(np.float32), 15.0, cav.astype(np.float32), 0.84, ridge.astype(np.float32)

    if detail_id == "terry_velcro_loops":
        # Authentic soft fibrous terry-cloth towel / fleece loop pile (organic domain-warped yarn loops, ZERO rigid grommets!)
        wu = 0.030 * _periodic_fbm(H, W, seed + 73, (6, 12, 24))
        wv = 0.030 * _periodic_fbm(H, W, seed + 74, (6, 12, 24))
        uw = (u + wu) % 1.0
        vw = (v + wv) % 1.0
        # Two overlapping jittered Voronoi loop layers create natural tangled soft yarn tufts & curl crests
        f1a, f2a = _periodic_worley(uw, vw, 18, seed + 75)
        f1b, f2b = _periodic_worley((uw + 0.031) % 1.0, (vw + 0.047) % 1.0, 24, seed + 76)
        curl_a = np.exp(-(((f1a - 0.30) / 0.16) ** 2))
        curl_b = np.exp(-(((f1b - 0.28) / 0.17) ** 2))
        tuft = np.maximum(curl_a * 0.90, curl_b * 0.80)
        fuzz = 0.14 * _periodic_fbm(H, W, seed + 77, (16, 32, 64))
        h = 0.22 + 0.56 * tuft + fuzz
        cav = _smoothstep(0.48, 0.20, h)
        ridge = _smoothstep(0.64, 0.88, h)
        return h.astype(np.float32), 10.0, cav.astype(np.float32), 0.88, ridge.astype(np.float32)

    if detail_id == "reeded_coin_milling":
        # Crisp bright milled golden-brass coin-edge ridges (20 band-limited flutes -> gleaming metallic sheen!)
        t = 0.5 + 0.5 * np.cos(2.0 * math.pi * u * 20.0)
        h = 0.14 + 0.76 * (t ** 0.58)
        cav = _smoothstep(0.45, 0.0, t)
        ridge = _smoothstep(0.68, 0.98, t)
        return h.astype(np.float32), 16.0, cav.astype(np.float32), 0.08, ridge.astype(np.float32)

    if detail_id == "fluted_reeded_glass":
        # Architectural vertical fluted privacy glass (12 smooth cylindrical half-round ribs, peak-centered)
        fu = ((u * 12.0 + 0.5) % 1.0) - 0.5
        # Softened cylinder roll-off eliminates infinite-slope cusps & opaque dark valleys
        rib = np.sqrt(np.clip(1.0 - 0.92 * (fu * 2.0) ** 2, 0.0, 1.0))
        h = 0.18 + 0.72 * rib
        cav = _smoothstep(0.45, 0.24, rib)
        ridge = _smoothstep(0.84, 0.99, rib)
        return h.astype(np.float32), 11.0, cav.astype(np.float32), 0.06, ridge.astype(np.float32)

    if detail_id == "terrazzo_speckled_stone":
        # Authentic honed terrazzo: DEAD-FLAT polished cement matrix with sharp angular periodic Voronoi marble chips!
        rng = np.random.default_rng(seed + 81)
        n_pts = 96
        pts_u = rng.uniform(0.0, 1.0, n_pts).astype(np.float32)
        pts_v = rng.uniform(0.0, 1.0, n_pts).astype(np.float32)
        # Warp slightly with fine noise so stone chips look hand-crushed & polygonal
        d1 = np.full((H, W), 10.0, dtype=np.float32)
        d2 = np.full((H, W), 10.0, dtype=np.float32)
        cid = np.zeros((H, W), dtype=np.int32)
        for idx_p, (pu, pv) in enumerate(zip(pts_u, pts_v)):
            du = np.abs(u - pu); du = np.minimum(du, 1.0 - du)
            dv = np.abs(v - pv); dv = np.minimum(dv, 1.0 - dv)
            # L1.6 metric produces sharp angular chiseled polygonal marble facets!
            dist = (du ** 1.6 + dv ** 1.6) ** (1.0 / 1.6)
            closer = dist < d1
            d2 = np.where(closer, d1, np.minimum(d2, dist))
            d1 = np.where(closer, dist, d1)
            cid = np.where(closer, idx_p, cid)
        # Keep ~38% of cells as distinct angular stone chips separated cleanly by cement matrix
        cell_hash = ((cid.astype(np.int64) * 2654435761) & 0xFFFF).astype(np.float32) / 65535.0
        is_chip_cell = (cell_hash < 0.42).astype(np.float32)
        edge_dist = d2 - d1
        chip_mask = is_chip_cell * _smoothstep(0.008, 0.022, edge_dist)
        micro = 0.015 * _periodic_fbm(H, W, seed + 82, (16, 32, 64))
        # Virtually flush honed surface (zero raised camo blobs!)
        h = 0.54 + 0.028 * chip_mask + micro
        # Encode chip color bucket smoothly in cav (0 = cement, 0.33/0.66/1.0 = mineral chip kinds)
        chip_kind = (0.33 + 0.33 * (cid % 3).astype(np.float32)) * chip_mask
        cav = np.clip(chip_kind, 0.0, 1.0)
        ridge = chip_mask
        return h.astype(np.float32), 3.0, cav.astype(np.float32), 0.24, ridge.astype(np.float32)

    if detail_id == "woven_rattan_cane":
        # Classic Parisian Thonet octagonal cane webbing with true over-under strand interleaving!
        n = 8.0
        us = (u + 0.5 / n) % 1.0
        vs = (v + 0.5 / n) % 1.0
        iu, iv = np.floor(us * n), np.floor(vs * n)
        fu = np.abs((us * n) % 1.0 - 0.5) * 2.0
        fv = np.abs((vs * n) % 1.0 - 0.5) * 2.0
        fd1 = np.abs(((us + vs) * n * 0.5) % 1.0 - 0.5) * 2.0
        fd2 = np.abs(((us - vs) * n * 0.5) % 1.0 - 0.5) * 2.0
        # Over-under alternation for orthogonal & diagonal cane strips
        par_hv = ((iu + iv) % 2.0) < 0.5
        par_d = (np.floor((us + vs) * n * 0.5) + np.floor((us - vs) * n * 0.5)) % 2.0 < 0.5
        su = np.sqrt(np.clip(1.0 - (fu / 0.28) ** 2, 0.0, 1.0)) * np.where(par_hv, 0.96, 0.68)
        sv = np.sqrt(np.clip(1.0 - (fv / 0.28) ** 2, 0.0, 1.0)) * np.where(par_hv, 0.68, 0.96)
        sd1 = np.sqrt(np.clip(1.0 - (fd1 / 0.22) ** 2, 0.0, 1.0)) * np.where(par_d, 0.84, 0.58)
        sd2 = np.sqrt(np.clip(1.0 - (fd2 / 0.22) ** 2, 0.0, 1.0)) * np.where(par_d, 0.58, 0.84)
        cane = np.maximum.reduce([su, sv, sd1, sd2])
        fiber = 0.025 * _periodic_fbm(H, W, seed + 86, (16, 48))
        h = 0.08 + 0.82 * cane + fiber * (cane > 0.1)
        cav = _smoothstep(0.22, 0.06, cane)
        ridge = _smoothstep(0.70, 0.92, cane)
        return h.astype(np.float32), 14.0, cav.astype(np.float32), 0.44, ridge.astype(np.float32)

    if detail_id == "corkboard_pin_surface":
        # Granulated natural corkboard: multi-scale bark platelets + dark intercellular pits
        c1 = _periodic_fbm(H, W, seed + 85, (12, 24, 48))
        c2 = _periodic_fbm(H, W, seed + 86, (24, 48, 96))
        pits = np.clip((-c2 - 0.85) * 2.2, 0.0, 1.0)
        h = 0.58 + 0.22 * c1 + 0.10 * c2 - 0.38 * pits
        cav = np.clip(pits * 1.1, 0.0, 1.0)
        ridge = _smoothstep(0.68, 0.88, h)
        return h.astype(np.float32), 18.0, cav.astype(np.float32), 0.86, ridge.astype(np.float32)

    if detail_id == "hammered_copper_dimple":
        # Hand-planished hammered copper: 84 densely packed shallow concave hammer-strike facets
        rng = np.random.default_rng(seed + 89)
        pts_u = rng.uniform(0.0, 1.0, 84).astype(np.float32)
        pts_v = rng.uniform(0.0, 1.0, 84).astype(np.float32)
        dmin = np.full((H, W), 10.0, dtype=np.float32)
        for pu, pv in zip(pts_u, pts_v):
            du = np.abs(u - pu); du = np.minimum(du, 1.0 - du)
            dv = np.abs(v - pv); dv = np.minimum(dv, 1.0 - dv)
            dmin = np.minimum(dmin, np.sqrt(du * du + dv * dv))
        # Shallow smooth planished bowl with crisp faceted specular rims
        h = np.clip(0.24 + 0.66 * np.clip(dmin / 0.085, 0.0, 1.0) ** 1.35, 0.0, 1.0)
        cav = 1.0 - h
        ridge = _smoothstep(0.72, 0.92, h)
        return h.astype(np.float32), 12.5, cav.astype(np.float32), 0.18, ridge.astype(np.float32)

    if detail_id == "cable_knit_wool":
        # Authentic Fisherman's Aran braided cable-knit sweater: alternating twisted 2-strand rope cables & purl-rib channels!
        n_cables = 4.0
        cu = (u * n_cables) % 1.0 - 0.5  # [-0.5, 0.5] per cable repeat
        # Two braided strands weaving left-right in opposite sinusoidal phase along v
        phase = 2.0 * math.pi * v * 4.0
        s1_x = 0.17 * np.sin(phase)
        s2_x = -0.17 * np.sin(phase)
        d1 = np.abs(cu - s1_x)
        d2 = np.abs(cu - s2_x)
        # Knit stitch rib texture along each rope strand
        st1 = 0.86 + 0.14 * np.sin(2.0 * math.pi * (v * 28.0 + cu * 12.0))
        st2 = 0.86 + 0.14 * np.sin(2.0 * math.pi * (v * 28.0 - cu * 12.0))
        # Over-under height modulation so strand 1 crosses cleanly OVER strand 2 on cos > 0 and UNDER on cos < 0
        z1 = 0.78 + 0.22 * np.cos(phase)
        z2 = 0.78 - 0.22 * np.cos(phase)
        r1 = np.sqrt(np.clip(1.0 - (d1 / 0.145) ** 2, 0.0, 1.0)) * z1 * st1
        r2 = np.sqrt(np.clip(1.0 - (d2 / 0.145) ** 2, 0.0, 1.0)) * z2 * st2
        braid = np.maximum(r1, r2)
        # Flanking purl-stitch background ribs along |cu| > 0.34
        purl = 0.36 * (0.65 + 0.35 * np.sin(2.0 * math.pi * v * 20.0)) * (0.7 + 0.3 * np.cos(2.0 * math.pi * cu * 8.0))
        wool_fuzz = 0.035 * _periodic_fbm(H, W, seed + 91, (24, 48))
        h = 0.10 + 0.82 * np.maximum(braid, purl) + wool_fuzz
        cav = _smoothstep(0.42, 0.14, h)
        ridge = _smoothstep(0.68, 0.90, braid)
        return h.astype(np.float32), 14.0, cav.astype(np.float32), 0.84, ridge.astype(np.float32)

    if detail_id == "penny_round_mosaic":
        # Classic glazed circular penny-round mosaic tiles (12x12 hexagonal packing) with recessed grout
        nx, ny = 12.0, 10.0
        row = np.floor(v * ny)
        u_s = (u + 0.5 * (row % 2.0) / nx) % 1.0
        fu = (u_s * nx) % 1.0 - 0.5
        fv = (v * ny) % 1.0 - 0.5
        r = np.sqrt(fu * fu + fv * fv)
        tile = 1.0 - _smoothstep(0.36, 0.44, r)
        pillow = np.sqrt(np.clip(1.0 - (r / 0.46) ** 2, 0.0, 1.0)) * tile
        h = 0.12 + 0.78 * (0.45 * tile + 0.55 * pillow)
        cav = 1.0 - tile
        ridge = _smoothstep(0.75, 0.92, h)
        return h.astype(np.float32), 20.0, cav.astype(np.float32), 0.16, ridge.astype(np.float32)

    if detail_id == "hex_mosaic_tile":
        # True seamless dual-lattice pointy-top hexagonal mosaic tessellation (10x6 dual periods, ZERO gaps!)
        nx, ny = 10.0, 6.0  # ny/nx ≈ 1/sqrt(3) = 0.577 -> near-exact regular hexagons!
        sx, sy = u * nx, v * ny * math.sqrt(3.0)
        # Dual rectangular lattice A (centered at (i+0.5, j*sqrt(3)+sqrt(3)/2)) and B (shifted by (0.5, sqrt(3)/2))
        s3 = math.sqrt(3.0)
        ax = (sx % 1.0) - 0.5
        ay = (sy % s3) - 0.5 * s3
        bx = ((sx + 0.5) % 1.0) - 0.5
        by = ((sy + 0.5 * s3) % s3) - 0.5 * s3
        use_a = (ax * ax + ay * ay) < (bx * bx + by * by)
        hx = np.where(use_a, ax, bx)
        hy = np.where(use_a, ay, by)
        # Exact pointy-top hex apothem distance (0 at center, 0.5 at exact shared hex edge!)
        d_hex = np.maximum(np.abs(hx), np.abs(hx) * 0.5 + np.abs(hy) * (0.5 * s3))
        tile = 1.0 - _smoothstep(0.435, 0.482, d_hex)
        bevel = 1.0 - _smoothstep(0.39, 0.455, d_hex)
        h = 0.16 + 0.34 * tile + 0.42 * bevel + 0.018 * _periodic_fbm(H, W, seed + 93, (8, 16)) * tile
        cav = 1.0 - tile
        ridge = _smoothstep(0.37, 0.43, d_hex) * tile
        return h.astype(np.float32), 14.0, cav.astype(np.float32), 0.18, ridge.astype(np.float32)

    if detail_id == "fine_asphalt_tarmac":
        # Street fine-aggregate asphalt: dense sharp mineral chips + bitumen binder pores
        a1 = _periodic_fbm(H, W, seed + 95, (16, 32, 64))
        a2 = _periodic_fbm(H, W, seed + 96, (32, 64, 96))
        chips = np.clip(a1 * 0.65 + np.abs(a2) * 0.45, -1.0, 1.5)
        h = 0.45 + 0.28 * chips
        cav = _smoothstep(0.42, 0.20, h)
        ridge = _smoothstep(0.68, 0.88, h)
        return h.astype(np.float32), 20.0, cav.astype(np.float32), 0.84, ridge.astype(np.float32)

    if detail_id == "venetian_plaster_trowel":
        # Authentic Italian burnished Venetian lime plaster (Marmorino): smooth sweeping spatula coats, ZERO gouged pits!
        swirl1 = _periodic_fbm(H, W, seed + 97, (2, 4, 8))
        swirl2 = _periodic_fbm(H, W, seed + 98, (6, 12, 24))
        fine = _periodic_fbm(H, W, seed + 101, (24, 48))
        # Sweeping overlapping curved trowel arcs (subtle burnished coat transitions)
        arc1 = np.sin(2.0 * math.pi * (u * 2.0 + v * 1.0 + 0.35 * swirl1))
        arc2 = np.cos(2.0 * math.pi * (u * 1.0 - v * 2.0 + 0.30 * swirl2))
        coat = 0.5 + 0.30 * arc1 + 0.20 * arc2
        # Delicate low-relief spatula edge lip + smooth burnished mineral body
        h = 0.52 + 0.12 * coat + 0.06 * swirl1 + 0.03 * swirl2 + 0.015 * fine
        cav = _smoothstep(0.56, 0.36, coat + 0.25 * swirl2)
        ridge = _smoothstep(0.58, 0.82, coat + 0.20 * swirl1)
        return h.astype(np.float32), 5.0, cav.astype(np.float32), 0.38, ridge.astype(np.float32)

    if detail_id == "frosted_prism_diffuser":
        # Office/transit ceiling prismatic acrylic diffuser (20x20 crisp diamond micro-pyramids)
        a = (u + v) * 20.0
        b = (u - v) * 20.0
        da = np.abs((a % 1.0) - 0.5) * 2.0
        db = np.abs((b % 1.0) - 0.5) * 2.0
        pyr = np.minimum(1.0 - da, 1.0 - db)
        h = 0.14 + 0.78 * pyr
        cav = _smoothstep(0.32, 0.16, h)
        ridge = _smoothstep(0.75, 0.92, h)
        return h.astype(np.float32), 14.0, cav.astype(np.float32), 0.18, ridge.astype(np.float32)

    if detail_id == "acoustic_kerf_felt":
        # Compressed matte heather wool-PET acoustic felt: razor-sharp knife-cut kerf slits (NO pillowed bevels!) + dense micro-fibers
        cols, rows = 4.0, 12.0
        r_idx = np.floor(v * rows)
        u_s = (u + 0.5 * (r_idx % 2.0) / cols) % 1.0
        fu = (u_s * cols) % 1.0 - 0.5
        fv = (v * rows) % 1.0 - 0.5
        px_c = np.maximum(np.abs(fu) - 0.31, 0.0)
        d_slit = np.sqrt(px_c * px_c + (fv * 0.55) ** 2)
        # Crisp knife-cut vertical drop without wide shiny pillowed shoulder halos
        slit = 1.0 - _smoothstep(0.036, 0.056, d_slit)
        fib_hi = _periodic_fbm(H, W, seed + 99, (48, 96, 128))
        rng_f = np.random.default_rng(seed + 100)
        fuzz = rng_f.standard_normal((H, W)).astype(np.float32) * 0.018
        h = 0.74 - 0.58 * slit + (0.048 * fib_hi + fuzz) * (1.0 - slit)
        cav = slit
        ridge = np.clip(0.5 + 0.35 * fib_hi, 0.0, 1.0) * (1.0 - slit)
        return h.astype(np.float32), 7.5, cav.astype(np.float32), 0.93, ridge.astype(np.float32)

    if detail_id == "patinated_verdigris_bronze":
        # Authentic hand-cast hammered bronze with gleaming satin metallic facets + interior crevice cuprite->verdigris patina!
        f1_b, _ = _periodic_worley(u, v, 8, seed + 110)
        bowl = np.clip(f1_b / 0.62, 0.0, 1.0) ** 1.35  # concave hammered dimples with crisp specular facet rims!
        p_mid = _periodic_fbm(H, W, seed + 114, (4, 8, 16))
        p_fine = _periodic_fbm(H, W, seed + 115, (16, 32, 64))
        # Crevice accumulation: verdigris collects organically inside scattered hammered bowls & recessed pits across the whole tile!
        crevice = 0.42 * (1.0 - bowl) + 0.44 * p_mid + 0.26 * p_fine
        ox_halo = _smoothstep(0.24, 0.64, crevice)
        verd_crust = _smoothstep(0.42, 0.84, crevice) * (0.60 + 0.40 * _smoothstep(-0.35, 0.75, p_fine))
        h = 0.24 + 0.64 * bowl * (1.0 - 0.30 * verd_crust) + 0.05 * p_fine * verd_crust
        cav = np.clip(verd_crust, 0.0, 1.0)
        ridge = np.clip(ox_halo * (1.0 - verd_crust * 0.75), 0.0, 1.0)
        return h.astype(np.float32), 13.0, cav.astype(np.float32), 0.16, ridge.astype(np.float32)

    if detail_id == "carrara_veined_marble":
        # Authentic polished Italian Carrara marble: virtually flat calcite slab with in-plane mineral veins (zero sunken cracks!)
        w_lo = _periodic_fbm(H, W, seed + 115, (2, 4))
        w_md = _periodic_fbm(H, W, seed + 116, (4, 8, 16))
        w_hi = _periodic_fbm(H, W, seed + 117, (16, 32, 64))
        p1 = u + v + 0.052 * w_lo + 0.018 * w_md + 0.005 * w_hi
        p2 = u + 2.0 * v - 0.045 * w_lo + 0.016 * w_md + 0.005 * w_hi + 0.31
        s_main = np.clip(1.0 - np.abs(np.sin(math.pi * p1)), 0.0, 1.0)
        s_sec = np.clip(1.0 - np.abs(np.sin(math.pi * p2)), 0.0, 1.0)
        taper1 = np.clip(0.55 + 0.45 * w_lo + 0.20 * w_md, 0.15, 1.0)
        taper2 = np.clip(0.45 - 0.40 * w_lo + 0.25 * w_md, 0.10, 1.0)
        sharp_v = np.clip((s_main ** 16.0) * 0.88 * taper1 + (s_sec ** 22.0) * 0.58 * taper2, 0.0, 1.0)
        cloud_v = np.clip((s_main ** 2.6) * 0.45 * taper1 + (s_sec ** 3.0) * 0.24 * taper2 + np.clip(w_lo * 0.15 + 0.10, 0.0, 0.25), 0.0, 1.0)
        h = 0.74 - 0.004 * sharp_v + 0.005 * w_md + 0.002 * w_hi
        cav = cloud_v
        ridge = sharp_v
        return h.astype(np.float32), 1.2, cav.astype(np.float32), 0.09, ridge.astype(np.float32)

    if detail_id == "herringbone_tweed_wool":
        # Authentic woven Harris/Donegal herringbone tweed: C1-smooth interlocked warp/weft yarn crossings + fuzzy wool nap
        cols = 4.0
        us = (u + 0.5 / cols) % 1.0
        # Smoothly rounded triangle wave for interlocked chevron twill
        zig = 0.25 * (2.0 / math.pi) * np.arcsin(0.92 * np.sin(2.0 * math.pi * us * (cols * 0.5))) + 0.25
        # Half-cell phase-centered yarn crossings -> strictly C1 continuous at u=0/1 and v=0/1!
        nu, nv = 24.0, 24.0
        fu = ((u + 0.5 / nu) * nu) % 1.0
        fv = ((v + 0.5 / nv) * nv) % 1.0
        twill_wave = 0.5 + 0.5 * np.sin(2.0 * math.pi * (v * 12.0 + zig * 6.0))
        warp_y = (0.5 - 0.5 * np.cos(2.0 * math.pi * fu)) ** 0.75
        weft_y = (0.5 - 0.5 * np.cos(2.0 * math.pi * fv)) ** 0.75
        weave = twill_wave * (0.40 + 0.55 * warp_y) + (1.0 - twill_wave) * (0.30 + 0.50 * weft_y)
        nap = 0.09 * _periodic_fbm(H, W, seed + 119, (24, 48, 96))
        h = 0.16 + 0.70 * weave + nap
        cav = _smoothstep(0.46, 0.20, h)
        ridge = _smoothstep(0.66, 0.88, h)
        return h.astype(np.float32), 10.0, cav.astype(np.float32), 0.88, ridge.astype(np.float32)

    if detail_id == "waffle_cotton_linen":
        # Authentic woven waffle-weave cotton (6x6 honeycomb pockets, 18 interlaced threads, phase-centered)
        n = 6.0
        tw = n * 3.0
        us = (u + 0.5 / tw) % 1.0
        vs = (v + 0.5 / tw) % 1.0
        fu = np.abs((us * n) % 1.0 - 0.5) * 2.0
        fv = np.abs((vs * n) % 1.0 - 0.5) * 2.0
        wall = np.maximum(fu, fv)
        pocket = _smoothstep(0.32, 0.84, wall)
        tu = np.abs((us * tw) % 1.0 - 0.5) * 2.0
        tv = np.abs((vs * tw) % 1.0 - 0.5) * 2.0
        ix = np.floor(us * tw); iy = np.floor(vs * tw)
        over = ((ix + iy) % 2.0) < 0.5
        yu = np.sqrt(np.clip(1.0 - (tu / 0.84) ** 2, 0.0, 1.0)) * np.where(over, 1.0, 0.54)
        yv = np.sqrt(np.clip(1.0 - (tv / 0.84) ** 2, 0.0, 1.0)) * np.where(over, 0.54, 1.0)
        cloth = np.maximum(yu, yv)
        fuzz = 0.025 * _periodic_fbm(H, W, seed + 122, (32, 64))
        h = 0.14 + 0.46 * (pocket ** 1.08) + 0.32 * cloth + fuzz
        cav = np.clip((1.0 - pocket) * 0.68 + (1.0 - cloth) * 0.36, 0.0, 1.0)
        ridge = _smoothstep(0.68, 0.90, h)
        return h.astype(np.float32), 12.0, cav.astype(np.float32), 0.84, ridge.astype(np.float32)

    if detail_id == "damascus_folded_steel":
        # Authentic Japanese forged Damascus steel: crisp acid-etched terrace steps + dramatic bright/dark steel bands
        f1 = _periodic_fbm(H, W, seed + 123, (2, 4, 8))
        f2 = _periodic_fbm(H, W, seed + 124, (6, 12, 24))
        phase = 2.0 * math.pi * (u * 12.0 + v * 3.0) + 4.4 * f1 + 0.85 * f2
        striation = np.clip(0.5 + 0.48 * np.tanh(3.2 * np.sin(phase)) + 0.05 * np.sin(phase * 2.0 + 0.6), 0.0, 1.0)
        fine_etch = 0.010 * _periodic_fbm(H, W, seed + 125, (32, 64))
        h = 0.48 + 0.10 * striation + fine_etch
        cav = 1.0 - striation
        ridge = striation
        return h.astype(np.float32), 4.0, cav.astype(np.float32), 0.14, ridge.astype(np.float32)

    if detail_id == "mermaid_fishscale_tile":
        # Authentic Moroccan Zellige fish-scale tiles: convex glossy domes with cleanly recessed thin grout
        nx, ny = 6.0, 10.0
        tile_h = np.full((H, W), 0.12, dtype=np.float32)
        tile_grout = np.ones((H, W), dtype=np.float32)
        tile_edge = np.zeros((H, W), dtype=np.float32)
        wobble = _periodic_fbm(H, W, seed + 126, (6, 12, 24))
        base_ry = np.floor(v * ny)
        for row_off in (-2, -1, 0, 1, 2):
            ry = base_ry + row_off
            u_s = (u + 0.5 * (ry % 2.0) / nx) % 1.0
            fu = (u_s * nx) % 1.0 - 0.5
            fv = (v * ny - ry) * 0.5
            r = np.sqrt(fu * fu + fv * fv) + 0.008 * wobble
            inside = r < 0.50
            grout_rim = _smoothstep(0.43, 0.48, r)
            dome = np.sqrt(np.clip(1.0 - (r / 0.50) ** 2, 0.0, 1.0))
            edge_pool = _smoothstep(0.32, 0.43, r) * (1.0 - grout_rim)
            scale_h = 0.12 + 0.80 * (1.0 - grout_rim) * (0.48 + 0.48 * dome + 0.04 * wobble)
            tile_h = np.where(inside, scale_h, tile_h)
            tile_grout = np.where(inside, grout_rim, tile_grout)
            tile_edge = np.where(inside, edge_pool, tile_edge)
        h = tile_h
        cav = tile_grout
        ridge = tile_edge
        return h.astype(np.float32), 13.0, cav.astype(np.float32), 0.10, ridge.astype(np.float32)

    if detail_id == "burlap_jute_sackcloth":
        # Authentic coarse jute burlap: interlocking 3-ply twisted rough fibers + slub irregularity + dark open weave pores
        n = 12.0
        jit_u = 0.11 * _periodic_fbm(H, W, seed + 127, (4, 8))
        jit_v = 0.11 * _periodic_fbm(H, W, seed + 128, (4, 8))
        u_j = u + jit_u * 0.08; v_j = v + jit_v * 0.08
        fu = np.abs((u_j * n) % 1.0 - 0.5) * 2.0
        fv = np.abs((v_j * n) % 1.0 - 0.5) * 2.0
        ix = np.floor(u_j * n); iy = np.floor(v_j * n)
        over = ((ix + iy) % 2.0) < 0.5
        twist_u = 0.5 + 0.5 * np.sin(2.0 * math.pi * (v_j * n * 3.0 + fu * 1.2))
        twist_v = 0.5 + 0.5 * np.sin(2.0 * math.pi * (u_j * n * 3.0 + fv * 1.2))
        w_u = 0.56 + 0.14 * _periodic_fbm(H, W, seed + 129, (4, 8))
        w_v = 0.56 + 0.14 * _periodic_fbm(H, W, seed + 130, (4, 8))
        yu = np.sqrt(np.clip(1.0 - (fu / w_u) ** 2, 0.0, 1.0)) * (0.72 + 0.28 * twist_u) * np.where(over, 1.0, 0.46)
        yv = np.sqrt(np.clip(1.0 - (fv / w_v) ** 2, 0.0, 1.0)) * (0.72 + 0.28 * twist_v) * np.where(over, 0.46, 1.0)
        fibers = 0.11 * _periodic_fbm(H, W, seed + 1300, (32, 64, 128))
        yarn_h = np.maximum(yu, yv)
        h = 0.10 + 0.80 * yarn_h + fibers * np.clip(yarn_h * 2.5, 0.0, 1.0)
        cav = _smoothstep(0.28, 0.11, h)
        ridge = _smoothstep(0.66, 0.88, h)
        return h.astype(np.float32), 14.0, cav.astype(np.float32), 0.84, ridge.astype(np.float32)

    if detail_id == "cast_iron_skillet_pebble":
        # Authentic Lodge seasoned sand-cast iron: true 3D multi-scale Worley pebble bumps + oiled-iron peaks!
        f1a, f2a = _periodic_worley(u, v, 28, seed + 131)
        f1b, f2b = _periodic_worley((u + 0.019) % 1.0, (v + 0.027) % 1.0, 48, seed + 132)
        peb_a = np.clip(1.0 - (f1a / 0.68) ** 2, 0.0, 1.0) ** 0.75
        peb_b = np.clip(1.0 - (f1b / 0.68) ** 2, 0.0, 1.0) ** 0.75
        peb = 0.65 * peb_a + 0.35 * peb_b
        pits = _smoothstep(0.32, 0.08, peb)
        crowns = _smoothstep(0.55, 0.88, peb)
        h = 0.24 + 0.62 * peb
        cav = pits
        ridge = crowns
        return h.astype(np.float32), 10.0, cav.astype(np.float32), 0.30, ridge.astype(np.float32)

    if detail_id == "tufted_emerald_velvet":
        # Authentic Chesterfield tufted velvet: plush diamond cushions + crisp radiating button pinch pleats + fine velvet pile
        nx, ny = 3.0, 3.0
        a = (u * nx + v * ny) % 1.0 - 0.5
        b = (u * nx - v * ny) % 1.0 - 0.5
        d_btn = np.sqrt(a * a + b * b)
        # Sharp V-creases running diagonally from button to button + plush dome pillows
        crease_a = _smoothstep(0.0, 0.14, np.abs(a))
        crease_b = _smoothstep(0.0, 0.14, np.abs(b))
        dimple = _smoothstep(0.050, 0.36, d_btn) ** 0.45
        pillow = (crease_a * crease_b) ** 0.42 * dimple
        # Prominent radiating fabric pinch pleats converging into each tufted button!
        ang = np.arctan2(b, a)
        pleat_env = np.exp(-((d_btn * 3.4) ** 2)) * _smoothstep(0.055, 0.11, d_btn)
        wrinkle = (0.11 * np.cos(8.0 * ang) + 0.05 * np.cos(16.0 * ang + 0.5)) * pleat_env
        btn = (1.0 - _smoothstep(0.050, 0.075, d_btn)) * np.sqrt(np.clip(1.0 - (d_btn / 0.08) ** 2, 0.0, 1.0))
        pile = 0.024 * _periodic_fbm(H, W, seed + 133, (32, 64, 96))
        h = 0.14 + 0.72 * pillow + 0.28 * btn + wrinkle + pile
        cav = np.clip((1.0 - pillow) * 0.85 + np.clip(-wrinkle * 3.0, 0.0, 0.35), 0.0, 1.0) * (1.0 - btn)
        ridge = np.clip(_smoothstep(0.66, 0.90, pillow) + np.clip(wrinkle * 3.0, 0.0, 0.30), 0.0, 1.0)
        return h.astype(np.float32), 14.5, cav.astype(np.float32), 0.86, ridge.astype(np.float32)

    if detail_id == "woven_bamboo_tatami":
        # Authentic Japanese igusa rush tatami: 22 clearly rounded cylindrical rush stalks + longitudinal grass striations + warp tuck
        cols, rows = 8.0, 22.0
        us = (u + 0.5 / cols) % 1.0
        vs = (v + 0.5 / rows) % 1.0
        col_i = np.floor(us * cols)
        fu = (us * cols) % 1.0 - 0.5
        bind = 1.0 - 0.30 * _smoothstep(0.36, 0.49, np.abs(fu))
        fv_s = (vs * rows + 0.5 * (col_i % 2.0)) % 1.0 - 0.5
        stalk = np.sqrt(np.clip(1.0 - 0.95 * (fv_s * 2.0) ** 2, 0.0, 1.0))
        # Fine longitudinal striations running along each rush stalk
        striat = 0.045 * np.sin(2.0 * math.pi * v * rows * 4.0 + 0.6 * _periodic_fbm(H, W, seed + 135, (8, 24)))
        h = 0.18 + 0.68 * stalk * bind + striat * stalk
        cav = np.clip((1.0 - stalk) * 0.65 + (1.0 - bind) * 0.70, 0.0, 1.0)
        ridge = _smoothstep(0.72, 0.94, stalk) * bind
        return h.astype(np.float32), 11.0, cav.astype(np.float32), 0.64, ridge.astype(np.float32)

    if detail_id == "travertine_porous_stone":
        # Authentic honed Roman travertine: gritty calcareous limestone strata + terraced horizontally-elongated erosion pores!
        s1 = _periodic_fbm(H, W, seed + 137, (2, 4, 8))
        strata = 0.5 + 0.5 * np.sin(2.0 * math.pi * (v * 10.0 + 0.18 * s1))
        # Horizontally elongated pore field (low freq along u, high freq along v) -> authentic sedimentary voids!
        p_elong = _periodic_fbm(H, W, seed + 138, (4, 8)) * 0.5 + _periodic_fbm(H, W, seed + 139, (12, 24)) * 0.5
        p_grit = _periodic_fbm(H, W, seed + 140, (24, 48, 96))
        band_wave = np.sin(2.0 * math.pi * (v * 14.0 + 0.16 * s1))
        cav_field = band_wave * 0.52 - p_elong * 0.55 + 0.25 * p_grit
        # Terraced 3-step cavity depth + sharp gritty micro-pinholes (zero flat camo puddles!)
        step1 = _smoothstep(0.58, 0.72, cav_field)
        step2 = _smoothstep(0.76, 0.90, cav_field)
        step3 = _smoothstep(0.94, 1.08, cav_field)
        pinholes = _smoothstep(0.82, 1.20, -p_grit) * 0.42
        total_pore = np.clip(0.32 * step1 + 0.36 * step2 + 0.32 * step3 + pinholes, 0.0, 1.0)
        # Fine calcareous limestone grit across the whole face
        grit_h = 0.045 * p_grit
        h = 0.78 + 0.025 * strata + grit_h * (1.0 - 0.5 * step1) - 0.16 * step1 - 0.22 * step2 - 0.24 * step3 - 0.14 * pinholes
        cav = total_pore
        ridge = _smoothstep(0.74, 0.84, h) * (1.0 - step1)
        return h.astype(np.float32), 13.0, cav.astype(np.float32), 0.48, ridge.astype(np.float32)

    # "gilded_tin_ceiling_tile"
    # Authentic vintage stamped pressed-tin ceiling tile: crisp embossed rosette medallion + beaded frame + hammered micro-grain
    n = 2.0
    fu = (u * n) % 1.0 - 0.5
    fv = (v * n) % 1.0 - 0.5
    ax, ay = np.abs(fu), np.abs(fv)
    r = np.sqrt(fu * fu + fv * fv)
    ang = np.arctan2(fv, fu)
    d_outer = np.abs(np.maximum(ax, ay) - 0.44)
    bead_wave = 0.65 + 0.35 * np.cos(2.0 * math.pi * np.where(ax > ay, fv, fu) * 20.0)
    frame = (1.0 - _smoothstep(0.024, 0.046, d_outer)) * bead_wave
    inner_f = 1.0 - _smoothstep(0.016, 0.034, np.abs(np.maximum(ax, ay) - 0.34))
    petal = 0.5 + 0.5 * np.cos(8.0 * ang)
    rosette = (1.0 - _smoothstep(0.18, 0.22, r - 0.065 * petal)) * (0.50 + 0.50 * petal)
    ring = 1.0 - _smoothstep(0.015, 0.030, np.abs(r - 0.12))
    boss = 1.0 - _smoothstep(0.045, 0.080, r)
    d_cor = np.sqrt((ax - 0.39) ** 2 + (ay - 0.39) ** 2)
    corner = 1.0 - _smoothstep(0.025, 0.050, d_cor)
    relief = np.maximum.reduce([frame * 0.86, inner_f * 0.72, rosette * 0.90, ring * 0.92, boss * 0.98, corner * 0.80])
    micro = 0.032 * _periodic_fbm(H, W, seed + 141, (16, 32, 64)) + 0.012 * _periodic_fbm(H, W, seed + 142, (64, 128))
    h = 0.16 + 0.76 * relief + micro
    cav = _smoothstep(0.30, 0.17, h)
    ridge = _smoothstep(0.65, 0.90, h)
    return h.astype(np.float32), 20.0, cav.astype(np.float32), 0.22, ridge.astype(np.float32)


def apply_daily_wear(h0, slope_scale, cav0, base_r, ridge0, mat_kind: str, wear_state: str, seed: int):
    H, W = h0.shape
    h = h0.astype(np.float32).copy()
    fbm_m = _periodic_fbm(H, W, seed + 101, (4, 8, 16))
    fbm_f = _periodic_fbm(H, W, seed + 102, (16, 32, 64))

    scuff_mask = np.zeros((H, W), dtype=np.float32)
    if wear_state == "1_lived_in":
        h += 0.035 * fbm_m + 0.022 * fbm_f
        scuff_mask = np.clip((fbm_m - 0.35) * 1.4, 0.0, 1.0)
    elif wear_state == "2_weathered":
        # Subtle warping + scratches/pits + worn-down ridges
        h = ng.node_height_warp_flow({"circular": True}, {"height": h}, scale=6.0, amplitude=0.008, octaves=2, seed=seed + 5)["height"]
        pits = np.clip((fbm_f - 0.85) * 2.2, 0.0, 1.0)
        h += 0.06 * fbm_m - 0.14 * pits
        scuff_mask = np.clip((fbm_m - 0.10) * 1.3 + pits * 0.8, 0.0, 1.0)

    N = _height_to_normal_periodic(h, slope=slope_scale)
    # Add fine material micro-grain
    g_kind = "mineral" if mat_kind in ("concrete", "ceramic") else ("brushed" if mat_kind in ("metal", "carbon") else "cast")
    g_str = {"0_fresh": 0.05, "1_lived_in": 0.10, "2_weathered": 0.17}[wear_state]
    if mat_kind in ("ceramic", "carbon"):
        g_str *= 0.45
    N = ng.node_micro_grain({"circular": True}, {"normal": N}, kind=g_kind, scale=36.0, strength=g_str, seed=seed + 7)["normal"]
    N = ng.node_normal_clamp_slope({"circular": True}, {"normal": N}, max_angle_deg=58.0)["normal"]

    h_norm = dp.normalize_height(h, clip_pct=0.5)
    ao_u8 = dp.height_to_ao(h_norm, strength=1.55, circular=True)
    ao_f = ao_u8.astype(np.float32) / 255.0

    r_add = {"0_fresh": 0.0, "1_lived_in": 0.06, "2_weathered": 0.14}[wear_state]
    # Grout/cavities are rougher; worn ridges polish slightly on hard surfaces or fuzz on fabric
    ridge_delta = +0.08 if mat_kind in ("fabric", "paper") else -0.10
    rough = np.clip(
        base_r + r_add + 0.34 * cav0 + 0.22 * (1.0 - ao_f) + 0.16 * scuff_mask + ridge_delta * ridge0,
        0.08, 0.96,
    ).astype(np.float32)
    rough = ng.node_roughness_specular_aa({"circular": True}, {"roughness": rough, "normal": N},
                                          strength=0.25, max_boost=0.08)["roughness"]

    w_gain = {"0_fresh": 0.20, "1_lived_in": 0.55, "2_weathered": 0.90}[wear_state]
    wear = np.clip(w_gain * (ridge0 * 0.8 + scuff_mask * 0.6), 0.0, 1.0).astype(np.float32)
    return h_norm, N, ao_u8, ao_f, rough, wear


def main():
    ap = argparse.ArgumentParser(description="Generate 36 Seamless Everyday-Life PBR Surface Details (12 designs x 3 wear states)")
    ap.add_argument("--out", type=Path, default=Path("./output/procedural"))
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    sheets_dir = args.out / "_sheets"
    sheets_dir.mkdir(exist_ok=True)
    manifest = {"version": "v1-daily-details-pbr", "tileable": True, "size": args.size, "states": list(WEAR_STATES), "details": []}
    cards_html = []
    showcase_rows = []

    for idx, (did, desc, mat_kind, tint) in enumerate(DAILY_CATALOG):
        h0, slope_scale, cav0, base_r, ridge0 = build_daily_detail(did, args.size, args.size, seed=2000 + idx * 41)
        entry = {"id": did, "description": desc, "material": mat_kind, "variants": {}}
        strip_panels = []

        for s_idx, ws in enumerate(WEAR_STATES):
            h_n, N, ao_u8, ao_f, rough, wear = apply_daily_wear(
                h0, slope_scale, cav0, base_r, ridge0, mat_kind, ws, seed=3000 + idx * 53 + s_idx * 11
            )
            preview = _lit_preview(N, ao_f, rough, tint)
            vdir = args.out / did / ws
            vdir.mkdir(parents=True, exist_ok=True)

            Image.fromarray(dp.normal_to_u8(N)).save(vdir / "normal.png")
            Image.fromarray(np.clip(h_n * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(vdir / "height.png")
            Image.fromarray(ao_u8).save(vdir / "ao.png")
            Image.fromarray(np.clip(rough * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(vdir / "roughness.png")
            Image.fromarray(np.clip(wear * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(vdir / "edge_wear.png")
            Image.fromarray(preview).save(vdir / "preview.png")
            entry["variants"][ws] = str(vdir.relative_to(args.out))

            if s_idx == 0:
                strip_panels.extend([
                    ("Fresh (Lit)", preview),
                    ("Fresh (Normal)", dp.normal_to_u8(N)),
                    ("Fresh (Height)", np.stack([np.clip(h_n * 255.0 + 0.5, 0, 255).astype(np.uint8)] * 3, -1)),
                ])
            elif s_idx == 1:
                strip_panels.extend([
                    ("Lived-In (Lit)", preview),
                    ("Lived-In (Normal)", dp.normal_to_u8(N)),
                ])
            else:
                strip_panels.extend([
                    ("Weathered (Lit)", preview),
                    ("Weathered (Normal)", dp.normal_to_u8(N)),
                    ("Weathered (Rough)", np.stack([np.clip(rough * 255.0 + 0.5, 0, 255).astype(np.uint8)] * 3, -1)),
                ])

        sz = 192
        pad = 6
        hdr = 22
        row = Image.new("RGB", (len(strip_panels) * (sz + pad) + pad, sz + hdr + pad), (18, 20, 24))
        dr = ImageDraw.Draw(row)
        for pi, (lbl, arr) in enumerate(strip_panels):
            x = pad + pi * (sz + pad)
            row.paste(Image.fromarray(arr).resize((sz, sz), Image.BICUBIC), (x, hdr))
            dr.text((x + 4, 4), f"{did}: {lbl}" if pi == 0 else lbl, fill=(220, 226, 236))
        sheet_p = sheets_dir / f"{did}.png"
        row.save(sheet_p)
        showcase_rows.append(row)

        b64 = base64.b64encode(sheet_p.read_bytes()).decode("ascii")
        cards_html.append(
            f'<div class="card"><div class="hdr"><b>{idx+1:02d}. {did}</b> '
            f'<span class="tag">{mat_kind}</span> <span class="desc">{desc}</span></div>'
            f'<img src="data:image/png;base64,{b64}"/></div>'
        )
        manifest["details"].append(entry)
        print(f"  [{idx+1:02d}/{len(DAILY_CATALOG)}] {did:26s} (3 wear states x 6 maps = 18 PNGs)", flush=True)

    # Save combined showcase PNG in output directory
    comb = Image.new("RGB", (showcase_rows[0].width, sum(r.height for r in showcase_rows)), (14, 16, 20))
    y = 0
    for r in showcase_rows:
        comb.paste(r, (0, y)); y += r.height
    comb_path = args.out / f"showcase_{len(DAILY_CATALOG)}.png"
    comb.save(comb_path)

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Daily Life PBR Surface Details ({len(DAILY_CATALOG)} x 3)</title>
<style>body{{background:#111318;color:#e5e9f0;font-family:system-ui,sans-serif;margin:24px}}
h1{{margin:0 0 8px}}p{{color:#9aa4b2;margin:0 0 20px}}
.card{{background:#181b22;border:1px solid #262b36;border-radius:10px;padding:12px;margin-bottom:16px}}
.hdr{{margin-bottom:8px;font-size:14px}}.tag{{background:#253248;color:#8cc8ff;padding:2px 7px;border-radius:4px;font-size:11px;margin:0 6px}}
.desc{{color:#9aa4b2;font-size:12px}}img{{width:100%;border-radius:6px;display:block}}</style></head>
<body><h1>Everyday Real-World PBR Surface Details ({len(DAILY_CATALOG)} Tileable Details &times; 3 Wear States)</h1>
<p>100% seamlessly tileable 512&times;512 Normal + Height + Cavity AO + Specular-AA Roughness + Edge Wear + Lit Preview.</p>
{''.join(cards_html)}</body></html>"""
    cat_path = args.out / "daily_details_catalog.html"
    cat_path.write_text(html)
    print(f"\nGenerated {len(DAILY_CATALOG)*len(WEAR_STATES)} seamless 512x512 PBR sets in {time.time()-t0:.1f}s!")
    print(f"Showcase PNG -> {comb_path}")
    print(f"Catalog HTML -> {cat_path}")


if __name__ == "__main__":
    main()
