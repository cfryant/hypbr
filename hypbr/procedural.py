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
def _periodic_fbm(H: int, W: int, seed: int, octaves=(4, 8, 16, 32, 64)) -> np.ndarray:
    """100% mathematically seamless multi-octave periodic noise for ANY integer frequency f."""
    rng = np.random.default_rng(seed)
    out = np.zeros((H, W), dtype=np.float32)
    amp = 1.0
    norm = 0.0
    yy = torch.linspace(0.0, 1.0 - 1.0 / H, H, dtype=torch.float32)
    xx = torch.linspace(0.0, 1.0 - 1.0 / W, W, dtype=torch.float32)
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    for f in octaves:
        g = rng.standard_normal((f, f)).astype(np.float32)
        t = torch.from_numpy(g)[None, None]
        # Circular-pad by 2 cells so bicubic kernel is 100% periodic for any f (even 3, 6, 12)
        t_pad = F.pad(t, (2, 2, 2, 2), mode="circular")
        n_pad = float(f + 4)
        # Exact continuous coords in align_corners=False normalized space [-1, 1]
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
    gx, gy = _periodic_grad(h)
    return ng.slopes_to_normal(-gx * slope, +gy * slope)


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    if abs(e1 - e0) < 1e-6:
        return (x >= e0).astype(np.float32)
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


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
    ("street_storm_grate",       "Street cast-iron storm-drain trench grate with staggered beveled slots", "metal",   (0.30, 0.31, 0.33)),
    ("diamond_tread_plate",      "Industrial/subway steel diamond checkerplate with alternating raised bars","metal", (0.50, 0.53, 0.56)),
    ("hvac_slotted_vent",        "Stamped architectural HVAC return-air louver vent grille with ribs",     "metal",   (0.78, 0.79, 0.77)),
    # --- 13 New Small-Scale Vents, Grilles, Plates, Meshes & Micro-Treads ---
    ("speaker_mesh_hex",         "Fine hexagonal perforated speaker / intercom / laptop mesh grille",      "metal",   (0.24, 0.25, 0.28)),
    ("expanded_metal_mesh",      "Diamond expanded-metal security screen & industrial walkway mesh",       "metal",   (0.42, 0.44, 0.46)),
    ("stamped_louver_micro",     "Fine stamped electronics / power-box cooling louver slot array",         "metal",   (0.36, 0.38, 0.40)),
    ("chainlink_wire_mesh",      "Tight woven galvanized wire mesh screen with over-under crossings",      "metal",   (0.56, 0.58, 0.60)),
    ("circular_perf_sheet",      "Staggered micro-punched stainless steel perforated vent sheet",          "metal",   (0.64, 0.66, 0.68)),
    ("corrugated_sheet_metal",   "Fine trapezoidal corrugated steel shutter / utility siding ribs",        "metal",   (0.54, 0.56, 0.58)),
    ("ribbed_rubber_mat",        "Fine parallel V-ribbed escalator / bus-aisle anti-slip rubber mat",      "rubber",  (0.18, 0.19, 0.21)),
    ("brushed_anodized_plate",   "Fine directional brushed aluminum elevator / appliance trim seam",       "metal",   (0.70, 0.72, 0.75)),
    ("embossed_braille_strip",   "Tactile railing / elevator braille dots & directional guide ribs",       "metal",   (0.66, 0.62, 0.50)),
    ("stamped_vent_slots_stagger","Staggered rounded-pill vent slots on street transformer cabinets",      "metal",   (0.46, 0.49, 0.46)),
    ("riveted_sheet_seam",       "Overlapping bus/train/aircraft sheet-metal lap seam with flush rivets",  "metal",   (0.60, 0.62, 0.65)),
    ("micro_hex_grip",           "Molded silicone / tool handle / camera body micro-hex grip pattern",     "rubber",  (0.20, 0.21, 0.23)),
    ("woven_nylon_strap",        "Tight herringbone ribbed seatbelt / backpack nylon webbing weave",       "fabric",  (0.22, 0.24, 0.28)),
    # --- 8 More Small-Scale Seamless Everyday Micro-Surfaces (36 Total!) ---
    ("knurled_straight_flutes",  "Fine vertical straight-knurled dial / pen / camera barrel flutes",       "metal",   (0.56, 0.58, 0.62)),
    ("dimpled_golf_grip",        "Concave spherical micro-dimpled grip / acoustic / sport handle surface", "rubber",  (0.26, 0.28, 0.30)),
    ("perforated_leather_auto",  "Car-seat / headphone micro-perforated pebble leather upholstery",        "leather", (0.20, 0.20, 0.22)),
    ("stair_nosing_grooves",     "Machined aluminum anti-slip stair-nosing tread channels with grit",      "metal",   (0.62, 0.64, 0.66)),
    ("heatsink_extrusion_fins",  "Tight extruded aluminum electronics heatsink cooling fin array",         "metal",   (0.32, 0.34, 0.37)),
    ("window_blinds_slats",      "Overlapping horizontal venetian-blind slats with ladder-cord ribs",      "metal",   (0.82, 0.83, 0.82)),
    ("acoustic_slat_wood",       "Vertical micro-slatted oak acoustic wall paneling over dark felt",       "wood",    (0.58, 0.44, 0.30)),
    ("drip_tray_sieve_grate",    "Stainless-steel coffee/espresso drip-tray bridge-lanced slotted grate",  "metal",   (0.70, 0.72, 0.75)),
    # --- 8 More Small-Scale Seamless Everyday Micro-Surfaces (44 Total!) ---
    ("modular_stud_grid",        "Precision-molded interlocking circular plastic stud grid",               "ceramic", (0.76, 0.22, 0.20)),
    ("waffle_silicone_pad",      "Molded kitchen/coaster square-waffle silicone grip mat",                 "rubber",  (0.24, 0.34, 0.38)),
    ("keyboard_keycap_grid",     "Tight grid of gently dished sculpted matte-plastic keycaps",             "rubber",  (0.22, 0.23, 0.26)),
    ("solar_cell_grid",          "Monocrystalline solar panel cells with silver busbars & micro-fingers",  "ceramic", (0.12, 0.17, 0.28)),
    ("pcb_solder_mask_traces",   "Electronics PCB surface with parallel copper traces & micro-vias",       "ceramic", (0.12, 0.34, 0.22)),
    ("acoustic_pyramid_foam",    "Studio/packaging convoluted acoustic pyramid foam micro-peaks",          "rubber",  (0.18, 0.19, 0.21)),
    ("terry_velcro_loops",       "Fine micro-looped hook-and-loop / terry textile surface",                "fabric",  (0.26, 0.27, 0.30)),
    ("reeded_coin_milling",      "Ultra-fine milled coin-edge / watch-bezel parallel micro-reeding",       "metal",   (0.74, 0.64, 0.42)),
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
]

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
        return h, 22.0, cav, 0.28, ridge

    if detail_id == "woven_denim_twill":
        # 3x1 diagonal twill weave (32 warp x 32 weft yarns) with yarn cross-section arching
        nu, nv = 32.0, 32.0
        iu = np.floor(u * nu)
        iv = np.floor(v * nv)
        fu = (u * nu) % 1.0
        fv = (v * nv) % 1.0
        phase = (iu + iv) % 4.0
        warp_over = (phase < 3.0).astype(np.float32)
        # Rounded yarn profiles
        warp_h = np.sin(math.pi * fu) ** 0.75 * (0.65 + 0.35 * np.sin(math.pi * fv))
        weft_h = np.sin(math.pi * fv) ** 0.75 * (0.65 + 0.35 * np.sin(math.pi * fu))
        twist = 0.08 * np.sin(2.0 * math.pi * (u * 96.0 + v * 96.0))
        h = warp_over * (0.35 + 0.65 * warp_h) + (1.0 - warp_over) * (0.20 + 0.55 * weft_h) + twist
        cav = _smoothstep(0.45, 0.15, h)
        ridge = _smoothstep(0.72, 0.98, h)
        return h, 16.0, cav, 0.84, ridge

    if detail_id == "quilted_leather_stitch":
        # 4x4 diamond-quilted upholstery leather with soft pillowed domes + dashed stitch grooves + pebble grain
        a = (u + v) * 4.0
        b = (u - v) * 4.0
        fa = np.abs((a % 1.0) - 0.5) * 2.0  # 0 at center, 1 at seam
        fb = np.abs((b % 1.0) - 0.5) * 2.0
        seam_d = np.minimum(1.0 - fa, 1.0 - fb)
        pillow = np.sqrt(np.clip(1.0 - 0.68 * (fa * fa + fb * fb) * 0.5, 0.05, 1.0))
        seam_cut = np.exp(-((seam_d / 0.045) ** 2))
        # Subtle dashed thread stitches running along the seams
        stitch_wave = (np.sin(2.0 * math.pi * a * 12.0) * np.sin(2.0 * math.pi * b * 12.0) > 0.1).astype(np.float32)
        pebble = _periodic_fbm(H, W, seed + 5, (24, 48, 96))
        h = pillow * (1.0 - 0.55 * seam_cut) + 0.08 * seam_cut * stitch_wave + 0.035 * pebble
        cav = np.clip(seam_cut * 1.15, 0.0, 1.0)
        ridge = _smoothstep(0.75, 0.98, pillow)
        return h, 18.0, cav, 0.36, ridge

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
        # 2x2 poured concrete sidewalk slabs with broom-finish striations + chamfered control joints + pores
        du = np.minimum(u % 0.5, 0.5 - (u % 0.5))
        dv = np.minimum(v % 0.5, 0.5 - (v % 0.5))
        d_joint = np.minimum(du, dv)
        joint = _smoothstep(0.008, 0.026, d_joint)
        # Fine directional broom-finish brush lines across concrete
        broom = 0.035 * np.sin(2.0 * math.pi * v * 110.0 + 0.8 * _periodic_fbm(H, W, seed + 9, (8, 16)))
        agg = _periodic_fbm(H, W, seed + 10, (8, 16, 32, 64))
        pores = np.clip((_periodic_fbm(H, W, seed + 11, (32, 64)) - 1.35) * 2.5, 0.0, 1.0)
        h = 0.15 + 0.72 * joint + (0.06 * agg + broom - 0.16 * pores) * joint
        cav = np.clip((1.0 - joint) + 0.7 * pores, 0.0, 1.0)
        ridge = _smoothstep(0.020, 0.035, d_joint) * (1.0 - _smoothstep(0.035, 0.055, d_joint))
        return h, 18.0, cav, 0.80, ridge

    if detail_id == "subway_metro_tile":
        # Classic 2x4 running-bond beveled ceramic subway tiles with crisp chamfered faces & recessed grout
        rows, cols = 6.0, 3.0
        row_idx = np.floor(v * rows)
        u_shift = (u + 0.5 * (row_idx % 2.0) / cols) % 1.0
        fu = (u_shift * cols) % 1.0
        fv = (v * rows) % 1.0
        du = np.minimum(fu, 1.0 - fu) / cols
        dv = np.minimum(fv, 1.0 - fv) / rows
        d_edge = np.minimum(du, dv)
        grout = _smoothstep(0.004, 0.008, d_edge)
        bevel = _smoothstep(0.008, 0.032, d_edge)
        # Subtle handmade ceramic glaze undulation
        undul = 0.025 * _periodic_fbm(H, W, seed + 13, (4, 8))
        grout_grit = 0.05 * _periodic_fbm(H, W, seed + 14, (32, 64)) * (1.0 - grout)
        h = 0.10 + 0.25 * grout + 0.62 * bevel + undul * grout + grout_grit
        cav = 1.0 - grout
        ridge = _smoothstep(0.026, 0.034, d_edge) * (1.0 - _smoothstep(0.034, 0.048, d_edge))
        return h, 20.0, cav, 0.18, ridge

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
        # 20 vertical rounded wales + fine horizontal cross-weave fuzz inside valleys
        wales = 0.5 + 0.5 * np.cos(2.0 * math.pi * u * 20.0)
        wales_r = np.power(np.clip(wales, 0.0, 1.0), 0.65)
        cross = 0.08 * np.sin(2.0 * math.pi * v * 96.0) * (1.0 - 0.5 * wales_r)
        nap = 0.04 * _periodic_fbm(H, W, seed + 17, (32, 64))
        h = 0.15 + 0.72 * wales_r + cross + nap
        cav = np.clip(1.0 - wales_r, 0.0, 1.0)
        ridge = _smoothstep(0.75, 1.0, wales_r)
        return h, 15.0, cav, 0.86, ridge

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
        # Classic oak herringbone parquet: alternating 45-deg rectangular planks with micro-bevels & grain
        # Transform into 45-deg rotated grid
        rx = (u + v) * 4.0
        ry = (u - v) * 4.0
        ix, iy = np.floor(rx), np.floor(ry)
        fx, fy = rx - ix, ry - iy
        # Herringbone parity: pair cells into 2:1 planks alternating horizontal/vertical in rotated space
        s = (ix + iy) % 2.0
        # Distance to plank borders
        # Horizontal plank spans 2 cells in x; vertical spans 2 cells in y
        dx_p = np.where(s < 0.5, np.minimum((rx % 2.0), 2.0 - (rx % 2.0)) * 0.5, np.minimum(fx, 1.0 - fx))
        dy_p = np.where(s < 0.5, np.minimum(fy, 1.0 - fy), np.minimum((ry % 2.0), 2.0 - (ry % 2.0)) * 0.5)
        seam_d = np.minimum(dx_p, dy_p)
        bevel = _smoothstep(0.015, 0.065, seam_d)
        # Plank-aligned wood grain
        g_h = np.sin(2.0 * math.pi * ry * 14.0 + 1.5 * _periodic_fbm(H, W, seed + 21, (8, 16)))
        g_v = np.sin(2.0 * math.pi * rx * 14.0 + 1.5 * _periodic_fbm(H, W, seed + 22, (8, 16)))
        grain = np.where(s < 0.5, g_h, g_v) * 0.035
        plank_var = 0.04 * np.sin(ix * 3.1 + iy * 5.7)
        h = 0.18 + 0.68 * bevel + (grain + plank_var) * bevel
        cav = 1.0 - bevel
        ridge = _smoothstep(0.05, 0.08, seam_d) * (1.0 - _smoothstep(0.08, 0.13, seam_d))
        return h, 17.0, cav, 0.44, ridge

    if detail_id == "carbon_fiber_weave":
        n = 16.0
        iu, iv = np.floor(u * n), np.floor(v * n)
        fu, fv = (u * n) % 1.0, (v * n) % 1.0
        twill = ((iu + iv) % 4.0) < 2.0
        tow_u = np.sin(math.pi * fu) ** 0.55 * (0.72 + 0.28 * np.sin(math.pi * fv))
        tow_v = np.sin(math.pi * fv) ** 0.55 * (0.72 + 0.28 * np.sin(math.pi * fu))
        fil_u = 0.04 * np.sin(2.0 * math.pi * u * n * 6.0)
        fil_v = 0.04 * np.sin(2.0 * math.pi * v * n * 6.0)
        h = np.where(twill, 0.22 + 0.72 * tow_u + fil_u, 0.18 + 0.68 * tow_v + fil_v)
        cav = _smoothstep(0.42, 0.18, h)
        ridge = _smoothstep(0.78, 0.96, h)
        return h.astype(np.float32), 18.0, cav.astype(np.float32), 0.22, ridge.astype(np.float32)

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
        return h.astype(np.float32), 21.0, cav.astype(np.float32), 0.44, ridge.astype(np.float32)

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
        fu = np.abs((u * cols) % 1.0 - 0.5) * 2.0
        fv = (v * rows) % 1.0
        in_col = 1.0 - _smoothstep(0.84, 0.91, fu)
        slat_ramp = np.clip(fv / 0.78, 0.0, 1.0)
        slat_drop = 1.0 - _smoothstep(0.76, 0.86, fv)
        slat = slat_ramp * slat_drop
        rib = 1.0 - in_col
        h = np.maximum(0.12 + 0.76 * slat * in_col, 0.62 * rib)
        cav = np.clip((1.0 - slat_drop) * in_col, 0.0, 1.0)
        ridge = _smoothstep(0.68, 0.77, fv) * slat_drop * in_col
        return h.astype(np.float32), 20.0, cav.astype(np.float32), 0.34, ridge.astype(np.float32)

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
        return h.astype(np.float32), 20.0, cav.astype(np.float32), 0.32, ridge.astype(np.float32)

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

    if detail_id == "chainlink_wire_mesh":
        # Tight woven wire screen (20x20 over-under crossing metal wires)
        n = 20.0
        iu, iv = np.floor(u * n), np.floor(v * n)
        fu = np.abs((u * n) % 1.0 - 0.5) * 2.0
        fv = np.abs((v * n) % 1.0 - 0.5) * 2.0
        wu = np.sqrt(np.clip(1.0 - (fu / 0.36) ** 2, 0.0, 1.0))
        wv = np.sqrt(np.clip(1.0 - (fv / 0.36) ** 2, 0.0, 1.0))
        par = ((iu + iv) % 2.0) < 0.5
        hu = wu * np.where(par, 0.92, 0.62)
        hv = wv * np.where(par, 0.62, 0.92)
        h = 0.06 + 0.86 * np.maximum(hu, hv)
        cav = _smoothstep(0.25, 0.08, h)
        ridge = _smoothstep(0.75, 0.92, h)
        return h.astype(np.float32), 20.0, cav.astype(np.float32), 0.30, ridge.astype(np.float32)

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
        # Fine trapezoidal corrugated steel shutter / utility panel (12 ribs)
        t = (u * 12.0) % 1.0
        # Flat-topped trapezoid profile
        trap = np.minimum(_smoothstep(0.08, 0.22, t), 1.0 - _smoothstep(0.58, 0.72, t))
        h = 0.14 + 0.74 * trap
        cav = 1.0 - trap
        ridge = _smoothstep(0.85, 0.99, trap)
        return h.astype(np.float32), 20.0, cav.astype(np.float32), 0.34, ridge.astype(np.float32)

    if detail_id == "ribbed_rubber_mat":
        # Fine parallel V-ribbed escalator / bus-aisle anti-slip rubber (28 sharp ribs)
        t = np.abs(((v * 28.0) % 1.0) - 0.5) * 2.0
        rib = np.clip(1.0 - t, 0.0, 1.0) ** 0.75
        # Subtle cross-drain channels every 1/4 width
        ch = 1.0 - 0.45 * np.exp(-((((u * 4.0) % 1.0 - 0.5) / 0.04) ** 2))
        h = 0.14 + 0.74 * rib * ch
        cav = 1.0 - rib * ch
        ridge = _smoothstep(0.75, 0.98, rib) * ch
        return h.astype(np.float32), 19.0, cav.astype(np.float32), 0.54, ridge.astype(np.float32)

    if detail_id == "brushed_anodized_plate":
        # Fine brushed aluminum elevator/appliance panel with horizontal expansion seam & corner micro-screws
        brush = 0.035 * np.sin(2.0 * math.pi * v * 140.0 + 1.2 * _periodic_fbm(H, W, seed + 41, (4, 8)))
        brush += 0.025 * _periodic_fbm(H, W, seed + 42, (8, 64))
        seam_v = np.exp(-(((v - 0.5) / 0.008) ** 2))
        seam_u = np.exp(-(((u - 0.5) / 0.008) ** 2))
        seams = np.maximum(seam_v, seam_u)
        # Small countersunk panel screws near seam intersections
        screws = np.zeros_like(u)
        for sx in (0.44, 0.56):
            for sy in (0.44, 0.56):
                r = np.sqrt((u - sx) ** 2 + (v - sy) ** 2)
                screws = np.maximum(screws, 1.0 - _smoothstep(0.012, 0.018, r))
        h = 0.62 + brush - 0.42 * seams - 0.18 * screws
        cav = np.clip(seams + 0.6 * screws, 0.0, 1.0)
        ridge = _smoothstep(0.012, 0.020, np.minimum(np.abs(u - 0.5), np.abs(v - 0.5))) * (1.0 - _smoothstep(0.020, 0.032, np.minimum(np.abs(u - 0.5), np.abs(v - 0.5))))
        return h.astype(np.float32), 18.0, cav.astype(np.float32), 0.24, ridge.astype(np.float32)

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
        return h.astype(np.float32), 20.0, cav.astype(np.float32), 0.30, ridge.astype(np.float32)

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
        # Overlapping bus/train/aircraft aluminum sheet-metal lap seam with flush countersunk rivet rows
        step = _smoothstep(0.485, 0.515, v)
        seam_groove = np.exp(-(((v - 0.50) / 0.008) ** 2))
        # Row of 10 small rivets on each side of the lap seam
        rivets = np.zeros_like(u)
        for ry in (0.42, 0.58):
            fu = (u * 10.0) % 1.0 - 0.5
            r = np.sqrt((fu / 10.0) ** 2 + (v - ry) ** 2)
            dome = np.sqrt(np.clip(1.0 - (r / 0.020) ** 2, 0.0, 1.0))
            ring = np.exp(-(((r - 0.021) / 0.004) ** 2)) * 0.25
            rivets = np.maximum(rivets, dome - ring)
        h = 0.42 + 0.18 * step - 0.22 * seam_groove + 0.28 * rivets
        cav = np.clip(seam_groove, 0.0, 1.0)
        ridge = np.clip(rivets + 0.5 * np.exp(-(((v - 0.52) / 0.012) ** 2)), 0.0, 1.0)
        return h.astype(np.float32), 19.0, cav.astype(np.float32), 0.28, ridge.astype(np.float32)

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
        return h.astype(np.float32), 20.0, cav.astype(np.float32), 0.52, ridge.astype(np.float32)

    if detail_id == "woven_nylon_strap":
        col = np.floor(u * 4.0)
        sgn = np.where((col % 2.0) < 0.5, 1.0, -1.0)
        rib = 0.5 + 0.5 * np.sin(2.0 * math.pi * (v * 36.0 + sgn * u * 18.0))
        fine = 0.15 * np.sin(2.0 * math.pi * u * 72.0)
        h = 0.22 + 0.62 * (rib ** 0.8) + fine
        cav = 1.0 - rib
        ridge = _smoothstep(0.72, 0.98, rib)
        return h.astype(np.float32), 16.0, cav.astype(np.float32), 0.66, ridge.astype(np.float32)

    if detail_id == "knurled_straight_flutes":
        # Fine vertical straight-knurled dial / pen / camera barrel flutes (40 crisp ridges)
        t = np.abs(((u * 40.0) % 1.0) - 0.5) * 2.0
        flute = np.clip(1.0 - t, 0.0, 1.0) ** 0.70
        h = 0.16 + 0.72 * flute
        cav = 1.0 - flute
        ridge = _smoothstep(0.78, 0.99, flute)
        return h.astype(np.float32), 20.0, cav.astype(np.float32), 0.26, ridge.astype(np.float32)

    if detail_id == "dimpled_golf_grip":
        # Concave spherical micro-dimpled grip (16x16 staggered circular concave dimples)
        n = 16.0
        row = np.floor(v * n)
        u_s = (u + 0.5 * (row % 2.0) / n) % 1.0
        fu = (u_s * n) % 1.0 - 0.5
        fv = (v * n) % 1.0 - 0.5
        r = np.sqrt(fu * fu + fv * fv)
        dimple = np.sqrt(np.clip(1.0 - (r / 0.38) ** 2, 0.0, 1.0))
        h = 0.84 - 0.66 * dimple
        cav = dimple
        ridge = _smoothstep(0.32, 0.42, r)
        return h.astype(np.float32), 18.0, cav.astype(np.float32), 0.48, ridge.astype(np.float32)

    if detail_id == "perforated_leather_auto":
        # Car-seat / headphone micro-perforated pebble leather (20x20 staggered pinholes + fine pebble)
        n = 20.0
        row = np.floor(v * n)
        u_s = (u + 0.5 * (row % 2.0) / n) % 1.0
        fu = (u_s * n) % 1.0 - 0.5
        fv = (v * n) % 1.0 - 0.5
        r = np.sqrt(fu * fu + fv * fv)
        hole = 1.0 - _smoothstep(0.14, 0.22, r)
        pebble = 0.055 * _periodic_fbm(H, W, seed + 51, (32, 64))
        h = 0.78 - 0.66 * hole + pebble * (1.0 - hole)
        cav = hole
        ridge = _smoothstep(0.18, 0.26, r) * (1.0 - _smoothstep(0.26, 0.34, r))
        return h.astype(np.float32), 18.0, cav.astype(np.float32), 0.38, ridge.astype(np.float32)

    if detail_id == "stair_nosing_grooves":
        # Machined aluminum anti-slip stair-nosing: alternating smooth metal ribs & coarse grit channels
        t = (v * 10.0) % 1.0
        rib = np.minimum(_smoothstep(0.06, 0.14, t), 1.0 - _smoothstep(0.42, 0.50, t))
        grit = 0.08 * _periodic_fbm(H, W, seed + 55, (32, 64)) * (1.0 - rib)
        h = 0.24 + 0.64 * rib + grit
        cav = 1.0 - rib
        ridge = _smoothstep(0.85, 0.99, rib)
        return h.astype(np.float32), 20.0, cav.astype(np.float32), 0.30, ridge.astype(np.float32)

    if detail_id == "heatsink_extrusion_fins":
        # Tight extruded aluminum electronics heatsink cooling fins (24 deep narrow fins)
        t = (u * 24.0) % 1.0
        fin = np.minimum(_smoothstep(0.18, 0.26, t), 1.0 - _smoothstep(0.56, 0.64, t))
        h = 0.08 + 0.84 * fin
        cav = 1.0 - fin
        ridge = _smoothstep(0.85, 0.99, fin)
        return h.astype(np.float32), 22.0, cav.astype(np.float32), 0.28, ridge.astype(np.float32)

    if detail_id == "window_blinds_slats":
        # Overlapping horizontal venetian-blind slats (14 slats) + twin vertical ladder-cord ribs
        fv = (v * 14.0) % 1.0
        slat = np.clip(fv / 0.88, 0.0, 1.0) * (1.0 - _smoothstep(0.86, 0.96, fv))
        cord = np.maximum(
            1.0 - _smoothstep(0.008, 0.016, np.abs(u - 0.25)),
            1.0 - _smoothstep(0.008, 0.016, np.abs(u - 0.75)),
        )
        h = np.maximum(0.14 + 0.72 * slat, 0.58 * cord)
        cav = _smoothstep(0.84, 0.94, fv)
        ridge = _smoothstep(0.75, 0.86, fv) * (1.0 - cav)
        return h.astype(np.float32), 18.0, cav.astype(np.float32), 0.32, ridge.astype(np.float32)

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
        # Everyday kitchen/desk square-waffle molded silicone mat (12x12 crisp pockets)
        n = 12.0
        fu = np.abs((u * n) % 1.0 - 0.5) * 2.0
        fv = np.abs((v * n) % 1.0 - 0.5) * 2.0
        wall = _smoothstep(0.68, 0.84, np.maximum(fu, fv))
        h = 0.14 + 0.74 * wall
        cav = 1.0 - wall
        ridge = _smoothstep(0.82, 0.98, np.maximum(fu, fv))
        return h.astype(np.float32), 19.0, cav.astype(np.float32), 0.52, ridge.astype(np.float32)

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
        # Everyday electronics PCB: parallel copper traces, solder-mask micro-step & periodic micro-vias
        traces = _smoothstep(0.35, 0.65, np.sin(2.0 * math.pi * u * 24.0 + 0.5 * np.sin(2.0 * math.pi * v * 4.0)))
        # Periodic circular vias / test pads (8x8)
        nv = 8.0
        gu = (u * nv) % 1.0 - 0.5
        gv = (v * nv) % 1.0 - 0.5
        rv = np.sqrt(gu * gu + gv * gv)
        pad = 1.0 - _smoothstep(0.14, 0.19, rv)
        hole = 1.0 - _smoothstep(0.05, 0.09, rv)
        h = 0.42 + 0.18 * traces + 0.26 * pad - 0.58 * hole
        cav = hole
        ridge = np.clip(pad * (1.0 - hole) + 0.4 * traces, 0.0, 1.0)
        return h.astype(np.float32), 17.0, cav.astype(np.float32), 0.24, ridge.astype(np.float32)

    if detail_id == "acoustic_pyramid_foam":
        # Studio / packaging convoluted acoustic pyramid foam (12x12 sharp soft-tip peaks)
        n = 12.0
        fu = np.abs((u * n) % 1.0 - 0.5) * 2.0
        fv = np.abs((v * n) % 1.0 - 0.5) * 2.0
        pyr = np.clip(1.0 - np.maximum(fu, fv), 0.0, 1.0) ** 0.85
        pores = 0.04 * _periodic_fbm(H, W, seed + 71, (32, 64))
        h = 0.12 + 0.78 * pyr + pores
        cav = 1.0 - pyr
        ridge = _smoothstep(0.75, 0.98, pyr)
        return h.astype(np.float32), 20.0, cav.astype(np.float32), 0.90, ridge.astype(np.float32)

    if detail_id == "terry_velcro_loops":
        # Fine micro-looped hook-and-loop / terry textile surface
        loops = 0.5 + 0.5 * np.sin(2.0 * math.pi * u * 48.0) * np.cos(2.0 * math.pi * v * 48.0)
        fuzz = _periodic_fbm(H, W, seed + 75, (16, 32, 64))
        h = 0.25 + 0.45 * loops + 0.18 * fuzz
        cav = _smoothstep(0.45, 0.15, h)
        ridge = _smoothstep(0.70, 0.95, h)
        return h.astype(np.float32), 16.0, cav.astype(np.float32), 0.88, ridge.astype(np.float32)

    if detail_id == "reeded_coin_milling":
        t = 0.5 + 0.5 * np.cos(2.0 * math.pi * u * 64.0)
        h = 0.18 + 0.72 * (t ** 0.75)
        cav = 1.0 - t
        ridge = _smoothstep(0.78, 0.99, t)
        return h.astype(np.float32), 21.0, cav.astype(np.float32), 0.22, ridge.astype(np.float32)

    if detail_id == "fluted_reeded_glass":
        # Architectural vertical fluted/reeded privacy glass (18 smooth cylindrical half-round ribs)
        fu = (u * 18.0) % 1.0 - 0.5
        rib = np.sqrt(np.clip(1.0 - (fu * 2.0) ** 2, 0.0, 1.0))
        h = 0.12 + 0.80 * rib
        cav = _smoothstep(0.32, 0.05, rib)
        ridge = _smoothstep(0.82, 0.99, rib)
        return h.astype(np.float32), 20.0, cav.astype(np.float32), 0.09, ridge.astype(np.float32)

    if detail_id == "terrazzo_speckled_stone":
        # Honed terrazzo cement with crisp angular embedded marble/quartz aggregate chips
        n1 = _periodic_fbm(H, W, seed + 81, (8, 16, 32))
        n2 = _periodic_fbm(H, W, seed + 82, (16, 32, 64))
        chips = _smoothstep(0.55, 0.78, n1)
        micro = 0.04 * n2
        h = 0.52 + 0.14 * chips + micro
        cav = _smoothstep(0.55, 0.45, h) * 0.4
        ridge = chips
        return h.astype(np.float32), 12.0, cav.astype(np.float32), 0.34, ridge.astype(np.float32)

    if detail_id == "woven_rattan_cane":
        # Classic Thonet / Parisian cafe woven rattan cane webbing (over-under horizontal, vertical & diagonal strands)
        n = 10.0
        fu = np.abs((u * n) % 1.0 - 0.5) * 2.0
        fv = np.abs((v * n) % 1.0 - 0.5) * 2.0
        fd1 = np.abs(((u + v) * n * 0.5) % 1.0 - 0.5) * 2.0
        fd2 = np.abs(((u - v) * n * 0.5) % 1.0 - 0.5) * 2.0
        su = np.sqrt(np.clip(1.0 - (fu / 0.30) ** 2, 0.0, 1.0))
        sv = np.sqrt(np.clip(1.0 - (fv / 0.30) ** 2, 0.0, 1.0))
        sd1 = np.sqrt(np.clip(1.0 - (fd1 / 0.24) ** 2, 0.0, 1.0)) * 0.78
        sd2 = np.sqrt(np.clip(1.0 - (fd2 / 0.24) ** 2, 0.0, 1.0)) * 0.78
        h = 0.06 + 0.84 * np.maximum.reduce([su, sv, sd1, sd2])
        cav = _smoothstep(0.22, 0.08, h)
        ridge = _smoothstep(0.75, 0.90, h)
        return h.astype(np.float32), 20.0, cav.astype(np.float32), 0.62, ridge.astype(np.float32)

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
        # Hand-planished hammered copper: Voronoi-like organic concave hammer-strike facets
        rng = np.random.default_rng(seed + 89)
        pts_u = rng.uniform(0.0, 1.0, 42).astype(np.float32)
        pts_v = rng.uniform(0.0, 1.0, 42).astype(np.float32)
        dmin = np.full((H, W), 10.0, dtype=np.float32)
        for pu, pv in zip(pts_u, pts_v):
            du = np.abs(u - pu); du = np.minimum(du, 1.0 - du)
            dv = np.abs(v - pv); dv = np.minimum(dv, 1.0 - dv)
            dmin = np.minimum(dmin, np.sqrt(du * du + dv * dv))
        # Concave dimple at cell center, sharp hammered ridge along Voronoi edges
        h = np.clip(0.18 + 0.74 * np.clip(dmin / 0.11, 0.0, 1.0) ** 1.4, 0.0, 1.0)
        cav = 1.0 - h
        ridge = _smoothstep(0.72, 0.92, h)
        return h.astype(np.float32), 19.0, cav.astype(np.float32), 0.24, ridge.astype(np.float32)

    if detail_id == "cable_knit_wool":
        # Cozy V-stitch rib-knit wool sweater textile (12 vertical ribs of interlocking V-loops)
        cols, rows = 12.0, 16.0
        fu = (u * cols) % 1.0 - 0.5
        fv = (v * rows) % 1.0
        # Classic knit V-chevron loop profile
        v_shift = (fv + np.abs(fu) * 0.95) % 1.0
        loop = np.sin(math.pi * v_shift) ** 0.75 * (1.0 - 0.55 * (np.abs(fu) * 2.0) ** 1.4)
        fuzz = 0.05 * _periodic_fbm(H, W, seed + 91, (32, 64))
        h = 0.14 + 0.76 * loop + fuzz
        cav = _smoothstep(0.40, 0.15, h)
        ridge = _smoothstep(0.72, 0.92, h)
        return h.astype(np.float32), 18.0, cav.astype(np.float32), 0.90, ridge.astype(np.float32)

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
        # Classic 1-inch hexagonal bathroom mosaic tiles (10x9 staggered hex grid) with crisp fine grout
        nx, ny = 10.0, 9.0
        row = np.floor(v * ny)
        u_s = (u + 0.5 * (row % 2.0) / nx) % 1.0
        fu = (u_s * nx) % 1.0 - 0.5
        fv = (v * ny) % 1.0 - 0.5
        ax, ay = np.abs(fu), np.abs(fv)
        d_hex = np.maximum(ax * 0.866025 + ay * 0.5, ay)
        tile = 1.0 - _smoothstep(0.39, 0.45, d_hex)
        h = 0.14 + 0.76 * tile + 0.02 * _periodic_fbm(H, W, seed + 93, (8, 16)) * tile
        cav = 1.0 - tile
        ridge = _smoothstep(0.35, 0.41, d_hex) * tile
        return h.astype(np.float32), 19.0, cav.astype(np.float32), 0.28, ridge.astype(np.float32)

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
        # Hand-troweled architectural lime plaster with sweeping curved trowel marks & subtle mineral pores
        swirl = _periodic_fbm(H, W, seed + 97, (3, 6, 12))
        trowel = np.sin(2.0 * math.pi * (u * 2.0 + v * 1.0 + 0.45 * swirl)) * 0.35
        fine = _periodic_fbm(H, W, seed + 98, (16, 32, 64))
        pits = np.clip((-fine - 1.35) * 2.5, 0.0, 1.0)
        h = 0.54 + 0.22 * trowel + 0.10 * swirl + 0.04 * fine - 0.24 * pits
        cav = np.clip(pits + _smoothstep(0.46, 0.32, h) * 0.45, 0.0, 1.0)
        ridge = _smoothstep(0.68, 0.86, h)
        return h.astype(np.float32), 14.0, cav.astype(np.float32), 0.66, ridge.astype(np.float32)

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
        return h.astype(np.float32), 21.0, cav.astype(np.float32), 0.18, ridge.astype(np.float32)

    if detail_id == "acoustic_kerf_felt":
        # Compressed heather-gray PET acoustic felt with fine laser-cut staggered kerf slits
        cols, rows = 4.0, 16.0
        r_idx = np.floor(v * rows)
        u_s = (u + 0.5 * (r_idx % 2.0) / cols) % 1.0
        fu = (u_s * cols) % 1.0 - 0.5
        fv = (v * rows) % 1.0 - 0.5
        px_c = np.maximum(np.abs(fu) - 0.34, 0.0)
        d_slit = np.sqrt(px_c * px_c + (fv * 0.55) ** 2)
        slit = 1.0 - _smoothstep(0.055, 0.095, d_slit)
        fibers = 0.045 * _periodic_fbm(H, W, seed + 99, (24, 48, 96))
        h = 0.78 - 0.66 * slit + fibers * (1.0 - slit)
        cav = slit
        ridge = _smoothstep(0.08, 0.14, d_slit) * (1.0 - _smoothstep(0.14, 0.22, d_slit))
        return h.astype(np.float32), 18.0, cav.astype(np.float32), 0.92, ridge.astype(np.float32)

    if detail_id == "patinated_verdigris_bronze":
        # Authentic verdigris bronze: hammered satin cast-bronze substrate + crevice-driven granular corrosion & cuprite transition
        p_macro = _periodic_fbm(H, W, seed + 111, (4, 8))
        p_pits = _periodic_fbm(H, W, seed + 112, (16, 32, 64))
        p_grit = _periodic_fbm(H, W, seed + 113, (64, 128, 256))
        # Crevice-driven patina: settles in hammered dimples & fine corrosion pits with soft organic feathered transition
        crevice = np.clip(-0.52 * p_macro - 0.38 * p_pits + 0.16 * p_grit, -1.0, 1.5)
        ox_halo = _smoothstep(0.18, 0.45, crevice)
        verd_crust = _smoothstep(0.42, 0.72, crevice)
        # Subtle hammered cast-bronze relief + granular crystalline patina grit in oxidized crevices
        h = 0.60 + 0.07 * p_macro + 0.03 * p_pits - 0.14 * ox_halo + 0.045 * p_grit * verd_crust
        cav = verd_crust
        ridge = ox_halo * (1.0 - verd_crust)  # cuprite brown oxide transition rim!
        return h.astype(np.float32), 7.5, cav.astype(np.float32), 0.34, ridge.astype(np.float32)

    if detail_id == "carrara_veined_marble":
        # Authentic Italian Carrara marble: strictly monotonic directional geological strata (mathematically zero closed loops!) + misty calcite clouding
        w_lo = _periodic_fbm(H, W, seed + 115, (2, 4))
        w_md = _periodic_fbm(H, W, seed + 116, (4, 8, 16))
        w_hi = _periodic_fbm(H, W, seed + 117, (16, 32, 64))
        # Monotonic low-amplitude warp guarantees directional flow never folds back into closed contour rings
        p1 = u + v + 0.052 * w_lo + 0.018 * w_md + 0.005 * w_hi
        p2 = u + 2.0 * v - 0.045 * w_lo + 0.016 * w_md + 0.005 * w_hi + 0.31
        s_main = np.clip(1.0 - np.abs(np.sin(math.pi * p1)), 0.0, 1.0)
        s_sec = np.clip(1.0 - np.abs(np.sin(math.pi * p2)), 0.0, 1.0)
        # Modulate vein intensity along strata so veins organically feather & taper rather than uniform stripes
        taper1 = np.clip(0.55 + 0.45 * w_lo + 0.20 * w_md, 0.15, 1.0)
        taper2 = np.clip(0.45 - 0.40 * w_lo + 0.25 * w_md, 0.10, 1.0)
        sharp_v = np.clip((s_main ** 18.0) * 0.92 * taper1 + (s_sec ** 24.0) * 0.62 * taper2, 0.0, 1.0)
        cloud_v = np.clip((s_main ** 2.6) * 0.45 * taper1 + (s_sec ** 3.0) * 0.24 * taper2 + np.clip(w_lo * 0.15 + 0.10, 0.0, 0.25), 0.0, 1.0)
        h = 0.74 - 0.018 * sharp_v + 0.004 * w_hi
        cav = cloud_v
        ridge = sharp_v
        return h.astype(np.float32), 1.5, cav.astype(np.float32), 0.10, ridge.astype(np.float32)

    if detail_id == "herringbone_tweed_wool":
        # Authentic Harris/Donegal herringbone tweed: thick lofted 2-ply wool yarns (6 chevron cols, 18 yarns — zero moire!)
        cols = 6.0
        col_i = np.floor(u * cols)
        sign = np.where((col_i % 2.0) < 0.5, 1.0, -1.0)
        diag = (u * 18.0 + sign * v * 18.0)
        yarn_u = np.abs((diag % 1.0) - 0.5) * 2.0
        cross_p = (u * 18.0 - sign * v * 18.0)
        ply_twist = 0.5 + 0.5 * np.sin(2.0 * math.pi * cross_p * 2.0)
        yarn_prof = np.sqrt(np.clip(1.0 - (yarn_u / 0.94) ** 2, 0.0, 1.0)) * (0.74 + 0.26 * ply_twist)
        # Soft fuzzy wool fiber loft
        nap = 0.08 * _periodic_fbm(H, W, seed + 119, (32, 64, 128))
        h = 0.14 + 0.76 * yarn_prof + nap
        cav = _smoothstep(0.40, 0.16, h)
        ridge = _smoothstep(0.66, 0.88, h)
        return h.astype(np.float32), 14.0, cav.astype(np.float32), 0.94, ridge.astype(np.float32)

    if detail_id == "waffle_cotton_linen":
        # Authentic woven waffle-weave cotton: 8x8 crisp honeycomb pockets built from cleanly interlaced twisted warp/weft yarns
        n = 8.0
        fu = np.abs((u * n) % 1.0 - 0.5) * 2.0
        fv = np.abs((v * n) % 1.0 - 0.5) * 2.0
        wall = np.maximum(fu, fv)
        pocket = _smoothstep(0.30, 0.84, wall)
        # Cleanly interlaced warp/weft yarns (3 crisp threads per waffle cell, zero mushy noise)
        tw = n * 3.0
        tu = np.abs((u * tw) % 1.0 - 0.5) * 2.0
        tv = np.abs((v * tw) % 1.0 - 0.5) * 2.0
        ix = np.floor(u * tw); iy = np.floor(v * tw)
        over = ((ix + iy) % 2.0) < 0.5
        yu = np.sqrt(np.clip(1.0 - (tu / 0.82) ** 2, 0.0, 1.0)) * np.where(over, 1.0, 0.52)
        yv = np.sqrt(np.clip(1.0 - (tv / 0.82) ** 2, 0.0, 1.0)) * np.where(over, 0.52, 1.0)
        cloth = np.maximum(yu, yv)
        fuzz = 0.03 * _periodic_fbm(H, W, seed + 122, (64, 128))
        h = 0.10 + 0.50 * (pocket ** 1.10) + 0.34 * cloth + fuzz
        cav = np.clip((1.0 - pocket) * 0.72 + (1.0 - cloth) * 0.38, 0.0, 1.0)
        ridge = _smoothstep(0.72, 0.94, h)
        return h.astype(np.float32), 16.0, cav.astype(np.float32), 0.92, ridge.astype(np.float32)

    if detail_id == "damascus_folded_steel":
        # Authentic Japanese forged Damascus steel: flat ground blade with crisp sub-mm acid-etched layered striations
        f1 = _periodic_fbm(H, W, seed + 123, (2, 4, 8))
        f2 = _periodic_fbm(H, W, seed + 124, (8, 16, 32))
        phase = 2.0 * math.pi * (u * 12.0 + v * 4.0) + 4.8 * f1 + 1.1 * f2
        # Crisp acid-etched terrace bands (flat ground steel, not crumpled foil)
        striation = np.clip(0.5 + 0.42 * np.tanh(1.6 * np.sin(phase)) + 0.08 * np.sin(phase * 2.0 + 0.6), 0.0, 1.0)
        fine_etch = 0.015 * _periodic_fbm(H, W, seed + 125, (64, 128))
        h = 0.48 + 0.08 * striation + fine_etch
        cav = 1.0 - striation
        ridge = striation
        return h.astype(np.float32), 1.6, cav.astype(np.float32), 0.19, ridge.astype(np.float32)

    if detail_id == "mermaid_fishscale_tile":
        # Authentic Moroccan Zellige fish-scale tiles: overlapping scallops + handmade kiln glaze wobble + deeply recessed grout!
        nx, ny = 6.0, 10.0
        tile_h = np.full((H, W), 0.06, dtype=np.float32)
        tile_grout = np.ones((H, W), dtype=np.float32)
        tile_edge = np.zeros((H, W), dtype=np.float32)
        wobble = _periodic_fbm(H, W, seed + 126, (8, 16, 32))
        base_ry = np.floor(v * ny)
        for row_off in (-2, -1, 0, 1, 2):
            ry = base_ry + row_off
            u_s = (u + 0.5 * (ry % 2.0) / nx) % 1.0
            fu = (u_s * nx) % 1.0 - 0.5
            fv = (v * ny - ry) * 0.5
            r = np.sqrt(fu * fu + fv * fv) + 0.010 * wobble
            inside = r < 0.50
            # Recessed grout valley around outer scallop rim (tile drops cleanly INTO grout, no raised piping!)
            grout_rim = _smoothstep(0.41, 0.47, r)
            dome = np.sqrt(np.clip(1.0 - (r / 0.52) ** 2, 0.0, 1.0))
            edge_pool = _smoothstep(0.28, 0.41, r) * (1.0 - grout_rim)
            scale_h = 0.06 + 0.84 * (1.0 - grout_rim) * (0.58 + 0.36 * dome + 0.06 * wobble)
            tile_h = np.where(inside, scale_h, tile_h)
            tile_grout = np.where(inside, grout_rim, tile_grout)
            tile_edge = np.where(inside, edge_pool, tile_edge)
        h = tile_h
        cav = tile_grout
        ridge = tile_edge
        return h.astype(np.float32), 16.0, cav.astype(np.float32), 0.13, ridge.astype(np.float32)

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
        fibers = 0.09 * _periodic_fbm(H, W, seed + 1300, (32, 64, 128))
        yarn_h = np.maximum(yu, yv)
        h = 0.04 + 0.86 * yarn_h + fibers * np.clip(yarn_h * 2.5, 0.0, 1.0)
        cav = _smoothstep(0.24, 0.07, h)
        ridge = _smoothstep(0.66, 0.88, h)
        return h.astype(np.float32), 20.0, cav.astype(np.float32), 0.93, ridge.astype(np.float32)

    if detail_id == "cast_iron_skillet_pebble":
        # Authentic Lodge seasoned black cast-iron: tight uniform sand-cast micro-grain with satin polymerized sheen
        s1 = _periodic_fbm(H, W, seed + 131, (64, 128))
        s2 = _periodic_fbm(H, W, seed + 132, (128, 256))
        raw = 0.55 * s1 + 0.45 * s2
        pits = _smoothstep(0.30, 1.10, -raw)
        crowns = _smoothstep(-0.25, 0.85, raw)
        h = 0.54 + 0.09 * crowns - 0.16 * pits
        cav = pits
        ridge = crowns
        return h.astype(np.float32), 3.4, cav.astype(np.float32), 0.42, ridge.astype(np.float32)

    if detail_id == "tufted_emerald_velvet":
        # Authentic Chesterfield tufted velvet: continuous diamond crease folds connecting every recessed button + plush pillows!
        nx, ny = 3.0, 3.0
        a = (u * nx + v * ny) % 1.0 - 0.5
        b = (u * nx - v * ny) % 1.0 - 0.5
        d_btn = np.sqrt(a * a + b * b)
        # Continuous V-shaped diagonal diamond fold channels running button-to-button along a=0 and b=0
        crease_a = _smoothstep(0.0, 0.14, np.abs(a))
        crease_b = _smoothstep(0.0, 0.14, np.abs(b))
        # Deep dimple around each button + puffed diamond cushion
        dimple = _smoothstep(0.055, 0.34, d_btn) ** 0.55
        pillow = (crease_a * crease_b) ** 0.52 * dimple
        # Soft pinched fabric folds radiating slightly around button dimple
        ang = np.arctan2(b, a)
        wrinkle = 0.035 * np.cos(4.0 * ang) * np.exp(-((d_btn * 5.5) ** 2)) * (1.0 - _smoothstep(0.05, 0.10, d_btn))
        # Recessed fabric-covered button sitting cleanly inside the dimple
        btn = (1.0 - _smoothstep(0.050, 0.075, d_btn)) * np.sqrt(np.clip(1.0 - (d_btn / 0.08) ** 2, 0.0, 1.0))
        pile = 0.018 * _periodic_fbm(H, W, seed + 133, (64, 128, 256))
        h = 0.06 + 0.82 * pillow + 0.28 * btn + wrinkle + pile
        cav = np.clip((1.0 - pillow) * 0.92, 0.0, 1.0) * (1.0 - btn)
        ridge = _smoothstep(0.70, 0.90, pillow)
        return h.astype(np.float32), 20.0, cav.astype(np.float32), 0.86, ridge.astype(np.float32)

    if detail_id == "woven_bamboo_tatami":
        # Authentic Japanese igusa rush tatami: tightly packed fine cylindrical horizontal rush stalks + subtle vertical stitch lines
        cols, rows = 14.0, 56.0
        col_i = np.floor(u * cols)
        fu = (u * cols) % 1.0 - 0.5
        # Subtle woven warp thread groove (NOT a deep brick gap!)
        bind = 1.0 - 0.24 * _smoothstep(0.38, 0.49, np.abs(fu))
        fv_s = (v * rows + 0.5 * (col_i % 2.0)) % 1.0 - 0.5
        stalk = np.sqrt(np.clip(1.0 - (fv_s * 2.0) ** 2, 0.0, 1.0))
        # Fine longitudinal grass fiber striations along each rush stalk
        fiber = 0.05 * _periodic_fbm(H, W, seed + 135, (16, 128))
        h = 0.22 + 0.64 * stalk * bind + fiber
        cav = np.clip((1.0 - stalk) * 0.65 + (1.0 - bind) * 0.80, 0.0, 1.0)
        ridge = _smoothstep(0.72, 0.94, stalk) * bind
        return h.astype(np.float32), 7.5, cav.astype(np.float32), 0.66, ridge.astype(np.float32)

    if detail_id == "travertine_porous_stone":
        # Authentic Roman travertine: warm honed sedimentary strata + multi-scale chiseled horizontal erosion cavities!
        s1 = _periodic_fbm(H, W, seed + 137, (2, 4, 8))
        strata = 0.5 + 0.5 * np.sin(2.0 * math.pi * (v * 8.0 + 0.20 * s1))
        p_h = _periodic_fbm(H, W, seed + 138, (4, 8, 16))
        p_v = _periodic_fbm(H, W, seed + 139, (16, 32, 64))
        p_micro = _periodic_fbm(H, W, seed + 140, (32, 64, 128))
        band_wave = np.sin(2.0 * math.pi * (v * 14.0 + 0.18 * s1))
        # Multi-scale eroded cavities: chiseled outer rim + deep rough inner pocket + fine micro-pinholes
        cav_field = band_wave * 0.50 - p_h * 0.55 - p_v * 0.35 + 0.12 * p_micro
        pores_rim = _smoothstep(0.42, 0.72, cav_field)
        pores_deep = _smoothstep(0.64, 0.92, cav_field)
        pinholes = _smoothstep(0.85, 1.25, -p_micro) * 0.35
        total_pore = np.clip(0.45 * pores_rim + 0.55 * pores_deep + pinholes, 0.0, 1.0)
        h = 0.78 + 0.03 * strata + 0.025 * p_micro - 0.24 * pores_rim - 0.44 * pores_deep - 0.15 * pinholes
        cav = total_pore
        ridge = _smoothstep(0.76, 0.84, h) * (1.0 - pores_rim)
        return h.astype(np.float32), 18.0, cav.astype(np.float32), 0.56, ridge.astype(np.float32)

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
