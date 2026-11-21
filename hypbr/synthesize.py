#!/usr/bin/env python3
"""V13 Material Quality Upgrade Pass for ALL 44 Seamless Daily Materials + HardOps Plate.

Upgrades every material from flat single-tint albedo & basic AO to AAA studio quality:
  1. True V13 8-Azimuth Horizon-Based AO (`HorizonBentNormalAO`) + Tangent-Space Bent Normal.
  2. Rich multi-tone physically-based Albedo per material class:
       - Glazed subway tile: warm cream ceramic faces + cool sanded mortar grout
       - Denim twill: classic indigo warp + ecru cotton weft yarns
       - PCB: emerald solder mask + bright copper/gold traces & pads + dark drilled vias
       - Solar cell: deep sapphire silicon + silver busbars & fingers + white backsheet gaps
       - Herringbone / acoustic oak: multi-tone honey/walnut wood grain + charcoal acoustic felt valleys
       - Quilted leather: rich espresso/saddle leather + warm tan contrast thread stitches
       - Machined metals / vents / grates: anisotropic steel/brass/iron tonal breakup + dark oil crevices + bright chamfer crests
       - Carbon fiber: alternating glossy graphite/charcoal tows
       - Cardboard: warm kraft pulp fiber mottling
  3. Calibrated per-pixel Roughness & Metallic separation (grout/felt/crevices matte dielectric,
     exposed crests/metal/glaze crisp low roughness) + Specular AA.
  4. Generates `hardops-plate` as a pure gunmetal machined HardOps panel
     (replacing the green palm-tree photo backdrop).
"""
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import depthpro_pbr as dp          # noqa: E402
from . import normal_graph as ng          # noqa: E402
from . import procedural as gd            # noqa: E402


def _rich_albedo_and_metal(did: str, mat_kind: str, tint: tuple, h: np.ndarray, cav: np.ndarray,
                           ridge: np.ndarray, ao: np.ndarray, seed: int):
    """Synthesize physically-grounded multi-tone linear RGB albedo, metallic, and roughness offset."""
    H, W = h.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u, v = xx / W, yy / H

    m1 = gd._periodic_fbm(H, W, seed + 201, (4, 8, 16))
    m2 = gd._periodic_fbm(H, W, seed + 202, (16, 32, 64))
    var = 0.06 * m1 + 0.03 * m2

    base = np.array(tint, dtype=np.float32)[None, None, :]
    alb = base * (0.92 + var[..., None])
    met = np.zeros((H, W), dtype=np.float32)
    r_mod = np.zeros((H, W), dtype=np.float32)

    if did == "subway_metro_tile":
        # Warm glossy off-white ceramic face + cool gray matte sanded mortar grout
        tile_col = np.array([0.86, 0.87, 0.84], np.float32)[None, None, :] * (0.96 + 0.04 * m1[..., None])
        grout_col = np.array([0.36, 0.37, 0.36], np.float32)[None, None, :] * (0.90 + 0.10 * m2[..., None])
        alb = tile_col * (1.0 - cav[..., None]) + grout_col * cav[..., None]
        r_mod = -0.08 * (1.0 - cav) + 0.58 * cav

    elif did == "pcb_solder_mask_traces":
        # Deep emerald solder mask + bright copper/gold traces & pads + dark drilled vias
        mask_col = np.array([0.04, 0.24, 0.11], np.float32)[None, None, :] * (0.94 + 0.06 * m1[..., None])
        copper_col = np.array([0.78, 0.56, 0.26], np.float32)[None, None, :]
        trace_m = np.clip(ridge * 1.4, 0.0, 1.0)
        via_dark = np.clip(cav * 1.3, 0.0, 1.0)
        alb = mask_col * (1.0 - trace_m[..., None]) + copper_col * trace_m[..., None]
        alb *= (1.0 - 0.85 * via_dark[..., None])
        met = trace_m * 0.92 * (1.0 - via_dark)
        r_mod = -0.06 * trace_m + 0.35 * via_dark

    elif did == "solar_cell_grid":
        # Deep sapphire monocrystalline silicon + bright silver busbars + white backsheet gaps
        si_col = np.array([0.05, 0.10, 0.26], np.float32)[None, None, :] * (0.92 + 0.08 * m1[..., None])
        bus_col = np.array([0.82, 0.84, 0.88], np.float32)[None, None, :]
        gap_col = np.array([0.84, 0.85, 0.86], np.float32)[None, None, :]
        alb = si_col * (1.0 - cav[..., None]) + gap_col * cav[..., None]
        alb = alb * (1.0 - ridge[..., None]) + bus_col * ridge[..., None]
        met = ridge * 0.95
        r_mod = -0.04 * (1.0 - cav) + 0.45 * cav

    elif did == "woven_denim_twill":
        # Authentic fine indigo jeans denim twill: dense indigo-dyed warp + subtle narrow diagonal ecru weft flecks!
        diag_f = 0.5 + 0.5 * np.sin(2.0 * math.pi * (u * 32.0 + v * 32.0))
        weft_fleck = gd._smoothstep(0.68, 0.94, diag_f) * (0.5 + 0.5 * np.cos(2.0 * math.pi * u * 64.0))
        slub = 0.5 + 0.5 * gd._periodic_fbm(H, W, seed + 205, (4, 32))
        indigo_lo = np.array([0.055, 0.105, 0.235], np.float32)[None, None, :]
        indigo_hi = np.array([0.115, 0.200, 0.380], np.float32)[None, None, :]
        ecru = np.array([0.48, 0.49, 0.50], np.float32)[None, None, :]
        indigo = indigo_lo * (1.0 - 0.55 * slub[..., None]) + indigo_hi * (0.55 * slub[..., None])
        alb = indigo * (1.0 - 0.46 * weft_fleck[..., None]) + ecru * (0.46 * weft_fleck[..., None])
        alb *= (0.86 + 0.14 * ao[..., None])

    elif did == "quilted_leather_stitch":
        # Rich espresso/saddle leather + warm tan contrast stitching in the seams!
        leath = np.array([0.26, 0.13, 0.08], np.float32)[None, None, :] * (0.95 + 0.04 * m1[..., None] + 0.03 * m2[..., None])
        stitch = np.array([0.74, 0.56, 0.36], np.float32)[None, None, :]
        stitch_m = np.clip(ridge * cav * 1.8, 0.0, 1.0)
        alb = leath * (1.0 - stitch_m[..., None]) + stitch * stitch_m[..., None]
        alb *= (0.76 + 0.24 * ao[..., None])
        r_mod = -0.08 * ridge + 0.30 * cav

    elif did == "wood_parquet_herringbone":
        # Rich multi-tone 4:1 oak herringbone planks aligned identically with N=8 torus-periodic procedural geometry!
        N = 8.0
        rx, ry = (u + v) * N, (u - v) * N
        ix, iy = np.floor(rx), np.floor(ry)
        d = (ix - iy) % 8.0
        is_h = d < 4.0
        pid_x = np.where(is_h, (ix - d) % N, ix % N)
        pid_y = np.where(is_h, iy % N, (iy - (d - 4.0)) % N)
        plank_id = np.sin(pid_x * 13.7 + pid_y * 29.3) * 0.5 + 0.5
        grain = 0.5 + 0.5 * np.where(
            is_h,
            np.sin(2.0 * math.pi * (ry * 4.0 + 0.25 * m1) + 0.65 * m2),
            np.sin(2.0 * math.pi * (rx * 4.0 + 0.25 * m1) + 0.65 * m2),
        )
        oak_light = np.array([0.60, 0.41, 0.23], np.float32)[None, None, :]
        oak_dark = np.array([0.30, 0.18, 0.09], np.float32)[None, None, :]
        plank_mix = np.clip(0.24 + 0.52 * plank_id + 0.24 * grain, 0.0, 1.0)[..., None]
        alb = oak_dark * (1.0 - plank_mix) + oak_light * plank_mix
        alb *= (1.0 - 0.55 * cav[..., None])
        r_mod = 0.28 * cav - 0.06 * grain

    elif did == "acoustic_slat_wood":
        # Warm natural oak slats over deep charcoal acoustic felt valleys
        grain = 0.5 + 0.5 * np.sin(2.0 * math.pi * u * 72.0 + 1.5 * m1)
        oak = np.array([0.56, 0.39, 0.23], np.float32)[None, None, :] * (0.86 + 0.20 * grain[..., None] + 0.04 * m1[..., None])
        felt = np.array([0.05, 0.05, 0.06], np.float32)[None, None, :] * (0.90 + 0.10 * m2[..., None])
        alb = oak * (1.0 - cav[..., None]) + felt * cav[..., None]
        r_mod = 0.48 * cav

    elif did == "carbon_fiber_weave":
        # Ultra-glossy clearcoat 2x2 twill carbon-fiber weave (28x28 tows = 7 repeats, strong orthogonal anisotropic sheen!)
        n = 28.0
        us = (u + 0.5 / n) % 1.0
        vs = (v + 0.5 / n) % 1.0
        iu, iv = np.floor(us * n), np.floor(vs * n)
        fu, fv = (us * n) % 1.0, (vs * n) % 1.0
        twill = ((iu + iv) % 4.0) < 2.0
        # Anisotropic fiber sheen gradient across each curved tow crown
        sheen_u = np.sin(math.pi * fu) ** 0.55
        sheen_v = np.sin(math.pi * fv) ** 0.55
        c_dark = np.array([0.024, 0.027, 0.034], np.float32)[None, None, :] * (0.82 + 0.30 * sheen_u[..., None])
        c_bright = np.array([0.23, 0.25, 0.29], np.float32)[None, None, :] * (0.72 + 0.42 * sheen_v[..., None])
        alb = np.where(twill[..., None], c_dark, c_bright)
        met = np.where(twill, 0.22, 0.62).astype(np.float32) * (1.0 - 0.35 * cav)
        r_mod = -0.02 * ridge + 0.05 * cav

    elif did == "corrugated_cardboard":
        # Warm kraft paper pulp with fine fiber speckles
        kraft = np.array([0.58, 0.43, 0.28], np.float32)[None, None, :]
        speck = np.clip((m2 - 1.1) * 2.0, 0.0, 1.0)[..., None]
        alb = kraft * (0.90 + 0.10 * m1[..., None]) - 0.12 * speck
        r_mod = 0.10 * cav

    elif did == "terrazzo_speckled_stone":
        # Honed cream terrazzo cement matrix + crisp angular Voronoi marble aggregate chips (terracotta, slate, warm ochre)!
        cem = np.array([0.78, 0.75, 0.70], np.float32)[None, None, :] * (0.96 + 0.04 * m2[..., None])
        c_terra = np.array([0.58, 0.27, 0.18], np.float32)[None, None, :] * (0.92 + 0.08 * m1[..., None])
        c_slate = np.array([0.20, 0.25, 0.30], np.float32)[None, None, :] * (0.92 + 0.08 * m1[..., None])
        c_ochre = np.array([0.70, 0.54, 0.30], np.float32)[None, None, :] * (0.92 + 0.08 * m2[..., None])
        chip = np.clip(ridge, 0.0, 1.0)
        # cav encodes mineral bucket in {0.33, 0.66, 0.99} inside chips
        b_slate = gd._smoothstep(0.45, 0.55, cav) * (1.0 - gd._smoothstep(0.78, 0.88, cav))
        b_ochre = gd._smoothstep(0.78, 0.88, cav)
        b_terra = np.clip(1.0 - b_slate - b_ochre, 0.0, 1.0)
        chip_col = c_terra * b_terra[..., None] + c_slate * b_slate[..., None] + c_ochre * b_ochre[..., None]
        alb = cem * (1.0 - chip[..., None]) + chip_col * chip[..., None]
        r_mod = 0.08 * (1.0 - chip) - 0.09 * chip

    elif did == "venetian_plaster_trowel":
        # Warm Italian alabaster/sand burnished Marmorino Venetian lime plaster with soft trowel sheen modulation
        ivory = np.array([0.78, 0.74, 0.67], np.float32)[None, None, :]
        sand = np.array([0.66, 0.61, 0.53], np.float32)[None, None, :]
        t_mix = np.clip(0.48 + 0.28 * ridge - 0.24 * cav + 0.12 * m1, 0.0, 1.0)[..., None]
        alb = sand * (1.0 - t_mix) + ivory * t_mix
        alb *= (0.95 + 0.05 * m2[..., None])
        # Burnished spatula ridges are satin-smooth (~0.24), matte recesses slightly chalkier (~0.48)
        r_mod = -0.14 * ridge + 0.12 * cav

    elif did == "cable_knit_wool":
        # Cozy oatmeal/cream Fisherman's Aran cable-knit wool with deep warm shadow pockets between braided strands
        cream = np.array([0.72, 0.67, 0.58], np.float32)[None, None, :] * (0.92 + 0.08 * m1[..., None])
        oat = np.array([0.48, 0.43, 0.36], np.float32)[None, None, :] * (0.92 + 0.08 * m2[..., None])
        alb = oat * cav[..., None] + cream * (1.0 - cav[..., None])
        alb = alb * (0.80 + 0.20 * ao[..., None]) + 0.03 * ridge[..., None]
        r_mod = -0.04 * ridge + 0.08 * cav

    elif did == "penny_round_mosaic":
        # Deep glazed teal/sea-glass penny tiles + crisp off-white sanded grout
        glaze = np.array([0.09, 0.34, 0.37], np.float32)[None, None, :] * (0.85 + 0.22 * m1[..., None])
        grout = np.array([0.78, 0.77, 0.74], np.float32)[None, None, :] * (0.92 + 0.08 * m2[..., None])
        alb = glaze * (1.0 - cav[..., None]) + grout * cav[..., None]
        r_mod = -0.06 * (1.0 - cav) + 0.62 * cav

    elif did == "hex_mosaic_tile":
        # Classic satin-white 1-inch honeycomb hex mosaic tiles + tight cool-gray sanded mortar grout
        white_t = np.array([0.82, 0.82, 0.80], np.float32)[None, None, :] * (0.96 + 0.04 * m1[..., None])
        grout = np.array([0.36, 0.38, 0.40], np.float32)[None, None, :] * (0.92 + 0.08 * m2[..., None])
        alb = white_t * (1.0 - cav[..., None]) + grout * cav[..., None]
        r_mod = -0.04 * (1.0 - cav) + 0.58 * cav

    elif did == "hammered_copper_dimple":
        # Hand-planished warm rose copper with luminous metallic specular crests & rich burnished dimples
        cop = np.array([0.88, 0.49, 0.29], np.float32)[None, None, :] * (0.94 + 0.08 * m1[..., None])
        pat = np.array([0.52, 0.26, 0.16], np.float32)[None, None, :]
        alb = pat * (0.28 * cav[..., None]) + cop * (1.0 - 0.28 * cav[..., None])
        met = np.clip(0.97 * (1.0 - 0.14 * cav), 0.75, 0.99).astype(np.float32)
        r_mod = -0.06 * ridge + 0.16 * cav

    elif did in ("fluted_reeded_glass", "frosted_prism_diffuser"):
        # Architectural luminous sea-green / crystal frosted reeded glass
        glass = np.array([0.72, 0.84, 0.82], np.float32)[None, None, :] * (0.95 + 0.05 * m1[..., None])
        alb = glass * (0.90 + 0.10 * ao[..., None])
        r_mod = -0.03 * ridge + 0.06 * cav

    elif did == "fine_asphalt_tarmac":
        # Dark charcoal bitumen with lighter gray/brown mineral aggregate sparkles
        bit = np.array([0.09, 0.09, 0.10], np.float32)[None, None, :]
        agg = np.array([0.34, 0.33, 0.31], np.float32)[None, None, :]
        alb = bit * (1.0 - 0.65 * ridge[..., None]) + agg * (0.65 * ridge[..., None])
        r_mod = -0.14 * ridge + 0.12 * cav

    elif did == "corkboard_pin_surface":
        # Warm multi-tone natural cork cells with darker bark flecks
        c_hi = np.array([0.62, 0.45, 0.29], np.float32)[None, None, :]
        c_lo = np.array([0.34, 0.22, 0.12], np.float32)[None, None, :]
        t_c = np.clip(0.5 + 0.35 * m1 - 0.45 * cav, 0.0, 1.0)[..., None]
        alb = c_lo * (1.0 - t_c) + c_hi * t_c

    elif did == "patinated_verdigris_bronze":
        # Three-stage hammered antique bronze: gleaming dark-to-bright satin bronze -> rich cuprite -> powdery interior verdigris!
        bronze_lo = np.array([0.44, 0.27, 0.11], np.float32)[None, None, :]
        bronze_hi = np.array([0.86, 0.58, 0.24], np.float32)[None, None, :]
        b_mix = np.clip(0.25 + 0.75 * h, 0.0, 1.0)[..., None]
        bronze = bronze_lo * (1.0 - b_mix) + bronze_hi * b_mix
        cuprite = np.array([0.22, 0.11, 0.05], np.float32)[None, None, :] * (0.90 + 0.10 * m2[..., None])
        verd_lo = np.array([0.13, 0.44, 0.37], np.float32)[None, None, :]
        verd_hi = np.array([0.30, 0.62, 0.50], np.float32)[None, None, :]
        v_mix = np.clip(0.5 + 0.45 * m2, 0.0, 1.0)[..., None]
        verd = verd_lo * (1.0 - v_mix) + verd_hi * v_mix
        alb = bronze * (1.0 - 0.72 * ridge[..., None]) + cuprite * (0.72 * ridge[..., None])
        alb = alb * (1.0 - cav[..., None]) + verd * cav[..., None]
        met = np.clip(0.97 * (1.0 - 0.30 * ridge) * (1.0 - 0.96 * cav), 0.0, 0.99).astype(np.float32)
        r_mod = -0.04 * (h - 0.5) + 0.18 * ridge + 0.70 * cav

    elif did == "acoustic_kerf_felt":
        # Genuine matte heather-gray compressed wool-PET acoustic felt: crisp fine micro-fiber flecks (ZERO low-freq clouds!)
        f_hi = np.array([0.48, 0.49, 0.51], np.float32)[None, None, :]
        f_lo = np.array([0.24, 0.25, 0.27], np.float32)[None, None, :]
        slit_c = np.array([0.025, 0.026, 0.030], np.float32)[None, None, :]
        f_grain = gd._periodic_fbm(H, W, seed + 209, (48, 96, 128))
        rng_h = np.random.default_rng(seed + 210)
        speck = rng_h.standard_normal((H, W)).astype(np.float32) * 0.22
        heather = np.clip(0.52 + 0.32 * f_grain + speck, 0.0, 1.0)[..., None]
        felt = f_lo * (1.0 - heather) + f_hi * heather
        alb = felt * (1.0 - 0.92 * cav[..., None]) + slit_c * (0.92 * cav[..., None])
        r_mod = 0.02 * cav

    elif did == "carrara_veined_marble":
        # Luminous cool-white calcite + broad smoky-grey subsurface clouding + crisp directional charcoal mineral veins!
        calcite = np.array([0.85, 0.84, 0.82], np.float32)[None, None, :] * (0.97 + 0.03 * m1[..., None])
        cloud_c = np.array([0.66, 0.68, 0.72], np.float32)[None, None, :]
        core_c = np.array([0.30, 0.32, 0.36], np.float32)[None, None, :]
        alb = calcite * (1.0 - cav[..., None]) + cloud_c * cav[..., None]
        alb = alb * (1.0 - ridge[..., None]) + core_c * ridge[..., None]
        r_mod = 0.04 * cav + 0.14 * ridge

    elif did == "herringbone_tweed_wool":
        # Authentic woven Harris/Donegal herringbone tweed: interlocked smooth zigzag twill yarns + Donegal rust flecks
        cols = 4.0
        us = (u + 0.5 / cols) % 1.0
        zig = 0.25 * (2.0 / math.pi) * np.arcsin(0.92 * np.sin(2.0 * math.pi * us * (cols * 0.5))) + 0.25
        yarn_t = 0.5 + 0.5 * np.sin(2.0 * math.pi * (v * 12.0 + zig * 6.0))
        heather = np.array([0.22, 0.21, 0.20], np.float32)[None, None, :] * (0.92 + 0.08 * m1[..., None])
        oat = np.array([0.56, 0.50, 0.42], np.float32)[None, None, :] * (0.92 + 0.08 * m2[..., None])
        fleck = gd._smoothstep(1.30, 1.70, m2)[..., None]
        rust = np.array([0.48, 0.26, 0.16], np.float32)[None, None, :]
        alb = heather * (1.0 - yarn_t[..., None]) + oat * yarn_t[..., None]
        alb = alb * (1.0 - 0.55 * fleck) + rust * (0.55 * fleck)
        alb *= (0.78 + 0.22 * ao[..., None])

    elif did == "waffle_cotton_linen":
        # Natural woven sage-eucalyptus cotton with bright sunlit interlaced thread crests & warm shadowed waffle pockets
        sage = np.array([0.40, 0.49, 0.42], np.float32)[None, None, :] * (0.94 + 0.06 * m1[..., None])
        linen_hi = np.array([0.56, 0.62, 0.55], np.float32)[None, None, :]
        alb = sage * (1.0 - 0.35 * ridge[..., None]) + linen_hi * (0.35 * ridge[..., None])
        alb *= (0.72 + 0.28 * ao[..., None])

    elif did == "damascus_folded_steel":
        # Authentic Japanese folded Damascus steel: dramatic bright nickel-steel vs acid-etched dark carbon-steel bands!
        dark_s = np.array([0.16, 0.18, 0.22], np.float32)[None, None, :]
        bright_s = np.array([0.88, 0.90, 0.94], np.float32)[None, None, :]
        alb = dark_s * (1.0 - ridge[..., None]) + bright_s * ridge[..., None]
        met = np.clip(0.92 + 0.07 * ridge, 0.88, 0.99).astype(np.float32)
        r_mod = -0.05 * ridge + 0.18 * cav

    elif did == "mermaid_fishscale_tile":
        # Handmade Moroccan Zellige fish-scale tiles: convex turquoise/jade glaze + recessed cool light-stone grout!
        nx, ny = 6.0, 10.0
        wobble = gd._periodic_fbm(H, W, seed + 126, (6, 12, 24))
        tile_var = np.full((H, W), 0.5, dtype=np.float32)
        base_ry = np.floor(v * ny)
        for row_off in (-2, -1, 0, 1, 2):
            ry = base_ry + row_off
            u_s = (u + 0.5 * (ry % 2.0) / nx) % 1.0
            rx = np.floor(u_s * nx)
            fu = (u_s * nx) % 1.0 - 0.5
            fv = (v * ny - ry) * 0.5
            r = np.sqrt(fu * fu + fv * fv) + 0.008 * wobble
            inside = r < 0.50
            tid = np.sin((rx % nx) * 17.3 + (ry % ny) * 31.7) * 0.5 + 0.5
            tile_var = np.where(inside, tid, tile_var)
        teal = np.array([0.05, 0.32, 0.38], np.float32)[None, None, :]
        jade = np.array([0.12, 0.46, 0.44], np.float32)[None, None, :]
        deep = np.array([0.03, 0.20, 0.28], np.float32)[None, None, :]
        glaze = teal * (1.0 - tile_var[..., None]) + jade * tile_var[..., None]
        glaze = glaze * (0.90 + 0.14 * m1[..., None])
        glaze = glaze * (1.0 - 0.35 * ridge[..., None]) + deep * (0.35 * ridge[..., None])
        grout = np.array([0.74, 0.74, 0.72], np.float32)[None, None, :] * (0.94 + 0.06 * m2[..., None])
        alb = glaze * (1.0 - cav[..., None]) + grout * cav[..., None]
        r_mod = -0.03 * (1.0 - cav) + 0.74 * cav

    elif did == "burlap_jute_sackcloth":
        # Authentic raw jute burlap: warm golden-tan twisted fibers + natural deep shadowed weave gaps (no pure black crush!)
        jute_hi = np.array([0.74, 0.60, 0.42], np.float32)[None, None, :]
        jute_mid = np.array([0.52, 0.38, 0.23], np.float32)[None, None, :]
        jute_lo = np.array([0.11, 0.075, 0.042], np.float32)[None, None, :]
        j_mix = np.clip(0.32 + 0.48 * ridge + 0.16 * m1, 0.0, 1.0)[..., None]
        alb = (jute_mid * (1.0 - j_mix) + jute_hi * j_mix) * (1.0 - cav[..., None]) + jute_lo * cav[..., None]

    elif did == "cast_iron_skillet_pebble":
        # Authentic Lodge seasoned black cast-iron: 3D Worley pebble crowns with oiled-iron sheen & matte dark valleys!
        pit_c = np.array([0.032, 0.034, 0.040], np.float32)[None, None, :]
        crown_c = np.array([0.17, 0.18, 0.20], np.float32)[None, None, :]
        alb = pit_c * (1.0 - ridge[..., None]) + crown_c * ridge[..., None]
        met = np.clip(0.82 * (1.0 - 0.55 * cav), 0.18, 0.88).astype(np.float32)
        r_mod = -0.10 * ridge + 0.36 * cav

    elif did == "tufted_emerald_velvet":
        # Rich deep jewel-toned Chesterfield emerald velvet (clean plush pile, zero dark macro blotches!)
        em_hi = np.array([0.035, 0.27, 0.17], np.float32)[None, None, :]
        em_lo = np.array([0.008, 0.065, 0.042], np.float32)[None, None, :]
        alb = em_lo * cav[..., None] + em_hi * (1.0 - cav[..., None]) * (0.95 + 0.04 * m1[..., None] + 0.03 * m2[..., None])
        r_mod = -0.04 * ridge + 0.08 * cav

    elif did == "woven_bamboo_tatami":
        # Authentic Japanese igusa rush tatami: warm golden-olive-straw stalks aligned with 22x8 phase-centered geometry!
        cols, rows = 8.0, 22.0
        us = (u + 0.5 / cols) % 1.0
        vs = (v + 0.5 / rows) % 1.0
        stalk_id = np.sin(np.floor(vs * rows) * 23.7 + np.floor(us * cols) * 11.3) * 0.5 + 0.5
        straw = np.array([0.62, 0.54, 0.30], np.float32)[None, None, :]
        igusa = np.array([0.42, 0.46, 0.24], np.float32)[None, None, :]
        gold = np.array([0.70, 0.58, 0.32], np.float32)[None, None, :]
        t1 = np.clip(0.25 + 0.55 * stalk_id + 0.12 * m1, 0.0, 1.0)[..., None]
        alb = igusa * (1.0 - t1) + straw * t1
        alb = alb * (1.0 - 0.22 * ridge[..., None]) + gold * (0.22 * ridge[..., None])
        alb *= (0.78 + 0.22 * ao[..., None])

    elif did == "travertine_porous_stone":
        # Authentic honed Roman travertine: warm cream-ochre calcareous strata + natural gritty mineral cavities!
        ivory = np.array([0.76, 0.68, 0.55], np.float32)[None, None, :]
        honey = np.array([0.58, 0.48, 0.35], np.float32)[None, None, :]
        vein = np.array([0.48, 0.39, 0.28], np.float32)[None, None, :]
        pore = np.array([0.24, 0.19, 0.13], np.float32)[None, None, :] * (0.88 + 0.16 * m2[..., None])
        band1 = (0.5 + 0.5 * np.sin(2.0 * math.pi * (v * 10.0 + 0.18 * m1)))[..., None]
        band2 = (0.5 + 0.5 * np.sin(2.0 * math.pi * (v * 20.0 - 0.14 * m1)))[..., None]
        stone = (honey * (1.0 - band1) + ivory * band1) * (0.92 + 0.08 * band2) * (0.94 + 0.06 * m2[..., None])
        stone = stone * 0.90 + vein * 0.10 * (1.0 - band2)
        alb = stone * (1.0 - 0.72 * cav[..., None]) + pore * (0.72 * cav[..., None])
        r_mod = 0.36 * cav

    elif did == "gilded_tin_ceiling_tile":
        # Authentic warm burnished gold-leaf / antique brass pressed tin with dark umber-soot recess patina
        gold = np.array([0.92, 0.72, 0.34], np.float32)[None, None, :] * (0.95 + 0.05 * m1[..., None])
        patina = np.array([0.18, 0.12, 0.06], np.float32)[None, None, :]
        alb = gold * (1.0 - 0.82 * cav[..., None]) + patina * (0.82 * cav[..., None])
        met = np.clip(0.97 * (1.0 - 0.78 * cav), 0.18, 0.99).astype(np.float32)
        r_mod = -0.06 * ridge + 0.52 * cav

    elif did in (
        "chainlink_wire_mesh", "expanded_metal_mesh", "street_storm_grate",
        "hvac_slotted_vent", "stamped_vent_slots_stagger", "drip_tray_sieve_grate",
        "stamped_louver_micro", "speaker_mesh_hex", "circular_perf_sheet",
    ):
        # Open-void perforated/mesh metals: gleaming clean metal frame + DEEP dark open void interiors!
        void_c = np.array([0.016, 0.017, 0.020], np.float32)[None, None, :]
        crest = np.clip(base * 1.24 + 0.05, 0.0, 0.94)
        alb = base * (0.96 + 0.035 * m1[..., None] + 0.025 * m2[..., None])
        alb = alb * (1.0 - 0.30 * ridge[..., None]) + crest * (0.30 * ridge[..., None])
        alb = alb * (1.0 - 0.96 * cav[..., None]) + void_c * (0.96 * cav[..., None])
        met = np.clip(0.95 * (1.0 - 0.96 * cav), 0.0, 0.98).astype(np.float32)
        r_mod = -0.07 * ridge + 0.65 * cav

    elif did in ("acoustic_pegboard", "perforated_leather_auto"):
        # Perforated wood / pebble leather: clean surface + deep dark punched through-holes!
        void_c = np.array([0.016, 0.016, 0.018], np.float32)[None, None, :]
        alb = base * (0.96 + 0.035 * m1[..., None] + 0.03 * m2[..., None])
        alb = alb * (0.88 + 0.12 * ao[..., None]) + 0.02 * ridge[..., None]
        hole_m = gd._smoothstep(0.35, 0.85, cav)
        alb = alb * (1.0 - 0.94 * hole_m[..., None]) + void_c * (0.94 * hole_m[..., None])
        r_mod = 0.50 * hole_m - 0.05 * ridge

    elif mat_kind == "metal":
        # Clean machined/cast metal: oil-darkened recesses + bright specular-polished chamfered crests (zero cloudy blotch!)
        oil = np.array([0.10, 0.10, 0.12], np.float32)[None, None, :]
        crest = np.clip(base * 1.22 + 0.05, 0.0, 0.94)
        alb = base * (0.96 + 0.035 * m1[..., None] + 0.025 * m2[..., None])
        alb = alb * (1.0 - 0.52 * cav[..., None]) + oil * (0.52 * cav[..., None])
        alb = alb * (1.0 - 0.30 * ridge[..., None]) + crest * (0.30 * ridge[..., None])
        met = np.clip(0.95 * (1.0 - 0.42 * cav), 0.28, 0.99).astype(np.float32)
        r_mod = -0.08 * ridge + 0.24 * cav

    else:
        # Clean tactile dielectric fallback (rubber / leather / fabric / concrete / ceramic) without dirty Perlin clouds
        alb = base * (0.95 + 0.035 * m1[..., None] + 0.03 * m2[..., None])
        alb = alb * (0.80 + 0.20 * ao[..., None]) + 0.028 * ridge[..., None]
        r_mod = 0.18 * cav - 0.05 * ridge

    # Enforce physically plausible dielectric albedo ceiling (~238 sRGB) while allowing bright metals up to 0.95
    alb_max = np.where(met[..., None] > 0.25, 0.95, 0.855).astype(np.float32)
    alb_clamped = np.clip(alb, 0.015, alb_max).astype(np.float32)
    alb_srgb = np.power(alb_clamped, 1.0 / 2.2)
    return (
        np.clip(alb_srgb * 255.0 + 0.5, 0, 255).astype(np.uint8),
        alb_clamped,
        np.clip(met * 255.0 + 0.5, 0, 255).astype(np.uint8),
        r_mod.astype(np.float32),
    )


def synthesize_hardops_plate(out_dir: Path, size: int = 512) -> Path:
    """Synthesize a pure gunmetal machined HardOps panel using 8-Azimuth HBAO + Bent Normal."""
    H = W = size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u, v = xx / W, yy / H

    # Start with clean machined gunmetal steel plate + subtle rolled-steel variation
    h0 = 0.50 + 0.015 * gd._periodic_fbm(H, W, 901, (4, 8, 16))
    N0 = gd._height_to_normal_periodic(h0, slope=10.0)

    # Stack HardOps panel seams + bolt plate + circular lug boss + brushed micro-grain + weld fillets!
    ctx = {"circular": True}
    s1 = ng.node_hardops_plug_stamp(ctx, {"height": h0, "normal": N0}, plug="panel_seams", array_x=2, array_y=2, bevel_radius=0.04, bevel_profile="chamfer", depth=0.55, normal_strength=2.0)
    s2 = ng.node_hardops_plug_stamp(ctx, {"height": s1["height"], "normal": s1["normal"]}, plug="bolt_plate", array_x=2, array_y=2, scale=0.74, bevel_radius=0.042, bevel_profile="round", depth=0.48, normal_strength=1.9)
    s3 = ng.node_hardops_plug_stamp(ctx, {"height": s2["height"], "normal": s2["normal"]}, plug="lug_boss", array_x=2, array_y=2, scale=0.42, bevel_radius=0.048, bevel_profile="round", depth=0.55, normal_strength=2.1)
    n_br = ng.node_normal_anisotropic_brush(ctx, {"normal": s3["normal"]}, angle_deg=0.0, freq=72.0, anisotropy=14.0, strength=0.18, seed=902)["normal"]
    curv = ng.node_curvature_multiscale(ctx, {"normal": n_br}, gain=3.4)
    n_weld = ng.node_normal_weld_fillet(ctx, {"normal": n_br, "concave": curv["concave"]}, radius=2.6, strength=0.50)["normal"]
    n_fin = ng.node_normal_clamp_slope(ctx, {"normal": n_weld}, max_angle_deg=58.0)["normal"]

    h_n = dp.normalize_height(s3["height"], clip_pct=0.5)
    hbao = ng.node_horizon_bent_normal_ao(ctx, {"height": h_n, "normal": n_fin}, strength=1.22, height_scale=13.0)
    ao = hbao["ao"]

    # Pure gunmetal anodized steel albedo with worn silver chamfer highlights & dark oil crevices
    m1 = gd._periodic_fbm(H, W, 903, (8, 16, 32))
    base = np.array([0.24, 0.27, 0.31], np.float32)[None, None, :] * (0.90 + 0.10 * m1[..., None])
    silver = np.array([0.74, 0.77, 0.81], np.float32)[None, None, :]
    oil = np.array([0.07, 0.08, 0.09], np.float32)[None, None, :]
    w_edge = np.clip(curv["convex"] * 0.85, 0.0, 1.0)[..., None]
    w_oil = np.clip((1.0 - ao) * 1.15, 0.0, 1.0)[..., None]
    alb_lin = np.clip(base * (1.0 - w_oil) + oil * w_oil, 0.02, 0.95)
    alb_lin = np.clip(alb_lin * (1.0 - 0.55 * w_edge) + silver * (0.55 * w_edge), 0.02, 0.95)
    alb_u8 = np.clip(np.power(alb_lin, 1.0 / 2.2) * 255.0 + 0.5, 0, 255).astype(np.uint8)

    rough = np.clip(0.28 + 0.34 * (1.0 - ao) - 0.12 * curv["convex"] + 0.04 * m1, 0.10, 0.90).astype(np.float32)
    rough = ng.node_roughness_specular_aa(ctx, {"roughness": rough, "normal": n_fin}, strength=0.25, max_boost=0.08)["roughness"]
    metal = np.clip(0.92 * (0.55 + 0.45 * ao) + 0.08 * curv["convex"], 0.25, 1.0).astype(np.float32)
    aniso = ng.node_anisotropy_flow(ctx, {"height": h_n, "ao": ao, "roughness": rough})
    thick = ng.node_thickness_subsurface(ctx, {"height": h_n})
    prev_u8 = render_studio_preview(alb_lin, n_fin, rough, metal, ao)
    rough_u8 = np.clip(rough * 255.0 + 0.5, 0, 255).astype(np.uint8)
    met_u8 = np.clip(metal * 255.0 + 0.5, 0, 255).astype(np.uint8)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_maps_parallel([
        (out_dir / "albedo_delit.png", alb_u8),
        (out_dir / "source_512.png", alb_u8),
        (out_dir / "preview.png", prev_u8),
        (out_dir / "height.png", np.clip(h_n * 255.0 + 0.5, 0, 255).astype(np.uint8)),
        (out_dir / "height16.png", np.clip(h_n * 65535.0 + 0.5, 0, 65535).astype(np.uint16)),
        (out_dir / "normal.png", dp.normal_to_u8(n_fin)),
        (out_dir / "bent_normal.png", hbao["bent_normal_u8"]),
        (out_dir / "ao.png", hbao["ao_u8"]),
        (out_dir / "roughness.png", rough_u8),
        (out_dir / "metallic.png", met_u8),
        (out_dir / "orm.png", np.stack([hbao["ao_u8"], rough_u8, met_u8], -1)),
        (out_dir / "curvature.png", curv["curvature_u8"]),
        (out_dir / "cavity.png", curv["cavity_u8"]),
        (out_dir / "edge_wear.png", np.clip(curv["convex"] * 255.0 + 0.5, 0, 255).astype(np.uint8)),
        (out_dir / "anisotropy.png", aniso["anisotropy_u8"]),
        (out_dir / "specular.png", aniso["specular_u8"]),
        (out_dir / "thickness.png", thick["thickness_u8"]),
    ])
    return out_dir


def list_materials() -> list[dict[str, str]]:
    """Return metadata for all built-in procedural PBR materials."""
    return [
        {"id": did, "description": desc, "category": mat_kind}
        for did, desc, mat_kind, _ in gd.DAILY_CATALOG
    ]


def save_maps_parallel(items: list[tuple[Path, np.ndarray]], compress_level: int = 4) -> None:
    """Save multiple lossless PNG maps concurrently (Pillow releases the GIL during zlib)."""
    def _write(pair: tuple[Path, np.ndarray]) -> None:
        p, arr = pair
        Image.fromarray(arr).save(p, compress_level=compress_level)

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(items)))) as pool:
        list(pool.map(_write, items))


@torch.no_grad()
def render_studio_preview(
    alb_lin: np.ndarray,
    N: np.ndarray,
    rough: np.ndarray,
    met: np.ndarray,
    ao: np.ndarray,
) -> np.ndarray:
    """Physically-based Cook-Torrance GGX + Lagarde Fresnel IBL studio preview (zero matte plastic glare)."""
    if dp.DEVICE.type != "cpu" and rough.size >= 262144:
        dev = dp.DEVICE
        alb_t = torch.from_numpy(np.ascontiguousarray(alb_lin, dtype=np.float32)).to(dev)
        n_t = torch.from_numpy(np.ascontiguousarray(N, dtype=np.float32)).to(dev)
        r_t = torch.clamp(torch.from_numpy(np.ascontiguousarray(rough, dtype=np.float32)).to(dev), 0.06, 0.98)
        m3 = torch.from_numpy(np.ascontiguousarray(met, dtype=np.float32)).to(dev).unsqueeze(-1)
        ao3 = torch.from_numpy(np.ascontiguousarray(ao, dtype=np.float32)).to(dev).unsqueeze(-1)

        l1 = torch.tensor([0.48, 0.55, 0.68], dtype=torch.float32, device=dev); l1 = l1 / torch.linalg.norm(l1)
        l2 = torch.tensor([-0.62, -0.25, 0.74], dtype=torch.float32, device=dev); l2 = l2 / torch.linalg.norm(l2)
        v_vec = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=dev)
        h1 = l1 + v_vec; h1 = h1 / torch.linalg.norm(h1)
        h2 = l2 + v_vec; h2 = h2 / torch.linalg.norm(h2)

        ndotl1 = torch.clamp((n_t * l1).sum(dim=-1), 0.0, 1.0)
        ndotl2 = torch.clamp((n_t * l2).sum(dim=-1), 0.0, 1.0)
        ndoth1 = torch.clamp((n_t * h1).sum(dim=-1), 0.0, 1.0)
        ndoth2 = torch.clamp((n_t * h2).sum(dim=-1), 0.0, 1.0)
        ndotv = torch.clamp(n_t[..., 2], 0.02, 1.0)
        vdoth1 = float(torch.clamp((v_vec * h1).sum(), 0.0, 1.0).item())
        vdoth2 = float(torch.clamp((v_vec * h2).sum(), 0.0, 1.0).item())

        r3 = r_t.unsqueeze(-1)
        diff_col = alb_t * (1.0 - m3)
        f0 = 0.04 * (1.0 - m3) + alb_t * m3
        # Lagarde roughness-attenuated Fresnel for IBL (eliminates milky plastic halo on rough textiles)
        fresnel_ibl = f0 + (torch.maximum(1.0 - r3, f0) - f0) * torch.pow(1.0 - ndotv, 5.0).unsqueeze(-1)

        rx = 2.0 * ndotv * n_t[..., 0]
        ry = 2.0 * ndotv * n_t[..., 1]
        sky_grad = torch.clamp(0.5 + 0.5 * ry, 0.0, 1.0).unsqueeze(-1)
        softbox = torch.clamp(1.0 - torch.hypot(rx - 0.35, ry - 0.45) * 1.15, 0.0, 1.0).unsqueeze(-1).square()
        c_gnd = torch.tensor([0.12, 0.13, 0.16], dtype=torch.float32, device=dev)
        c_sky = torch.tensor([0.66, 0.71, 0.80], dtype=torch.float32, device=dev)
        c_box = torch.tensor([2.25, 2.15, 1.98], dtype=torch.float32, device=dev)
        c_blr = torch.tensor([0.36, 0.37, 0.40], dtype=torch.float32, device=dev)
        env_sharp = (1.0 - sky_grad) * c_gnd + sky_grad * c_sky + softbox * c_box
        ibl_rad = env_sharp * torch.pow(1.0 - r3, 1.45) + c_blr * torch.pow(r3, 1.15)
        ibl_energy = (1.15 * torch.pow(1.0 - r3, 1.55) + 0.10 * (1.0 - r3)) * (1.0 - m3) + (0.36 + 0.95 * torch.pow(1.0 - r3, 0.70)) * m3
        spec_ibl = fresnel_ibl * ibl_rad * ibl_energy * (0.28 + 0.72 * ao3)

        # True Cook-Torrance Trowbridge-Reitz GGX direct microfacet specular
        a = r_t * r_t
        a2 = torch.clamp(a * a, min=1e-4)
        k = (r_t + 1.0).square() * 0.125
        d1 = a2 / (math.pi * (ndoth1 * ndoth1 * (a2 - 1.0) + 1.0).square() + 1e-6)
        d2 = a2 / (math.pi * (ndoth2 * ndoth2 * (a2 - 1.0) + 1.0).square() + 1e-6)
        vis1 = 0.25 / ((ndotl1 * (1.0 - k) + k) * (ndotv * (1.0 - k) + k) + 1e-4)
        vis2 = 0.25 / ((ndotl2 * (1.0 - k) + k) * (ndotv * (1.0 - k) + k) + 1e-4)
        f1 = f0 + (1.0 - f0) * ((1.0 - vdoth1) ** 5)
        f2 = f0 + (1.0 - f0) * ((1.0 - vdoth2) ** 5)
        spec_direct = (
            f1 * (d1 * vis1 * ndotl1 * 2.10).unsqueeze(-1)
            + f2 * (d2 * vis2 * ndotl2 * 0.75).unsqueeze(-1)
        ) * (0.35 + 0.65 * ao3)

        diffuse = diff_col * (0.24 + 0.86 * ndotl1.unsqueeze(-1) + 0.26 * ndotl2.unsqueeze(-1)) * (0.28 + 0.72 * ao3)
        col = diffuse + spec_ibl + spec_direct
        # Smooth ACES-like highlight roll-off
        col = col / (1.0 + 0.26 * col)
        out_u8 = torch.clamp(torch.pow(torch.clamp(col, 0.0, 1.0), 1.0 / 2.2) * 255.0 + 0.5, 0.0, 255.0).to(torch.uint8)
        return out_u8.cpu().numpy()

    L1 = np.array([0.48, 0.55, 0.68], np.float32); L1 /= np.linalg.norm(L1)
    L2 = np.array([-0.62, -0.25, 0.74], np.float32); L2 /= np.linalg.norm(L2)
    V = np.array([0.0, 0.0, 1.0], np.float32)
    H1 = L1 + V; H1 /= np.linalg.norm(H1)
    H2 = L2 + V; H2 /= np.linalg.norm(H2)

    ndotl1 = np.clip((N * L1[None, None, :]).sum(-1), 0.0, 1.0)
    ndotl2 = np.clip((N * L2[None, None, :]).sum(-1), 0.0, 1.0)
    ndoth1 = np.clip((N * H1[None, None, :]).sum(-1), 0.0, 1.0)
    ndoth2 = np.clip((N * H2[None, None, :]).sum(-1), 0.0, 1.0)
    ndotv = np.clip(N[..., 2], 0.02, 1.0)
    vdoth1 = float(np.clip(np.dot(V, H1), 0.0, 1.0))
    vdoth2 = float(np.clip(np.dot(V, H2), 0.0, 1.0))

    m3 = met[..., None]
    r = np.clip(rough, 0.06, 0.98)
    r3 = r[..., None]
    ao3 = ao[..., None]

    # Metal/dielectric energy split & Lagarde roughness-attenuated Fresnel IBL
    diff_col = alb_lin * (1.0 - m3)
    f0 = 0.04 * (1.0 - m3) + alb_lin * m3
    fresnel_ibl = f0 + (np.maximum(1.0 - r3, f0) - f0) * ((1.0 - ndotv) ** 5)[..., None]

    rx = 2.0 * ndotv * N[..., 0]
    ry = 2.0 * ndotv * N[..., 1]
    sky_grad = np.clip(0.5 + 0.5 * ry, 0.0, 1.0)[..., None]
    softbox = np.clip(1.0 - np.hypot(rx - 0.35, ry - 0.45) * 1.15, 0.0, 1.0)[..., None] ** 2
    env_sharp = (
        (1.0 - sky_grad) * np.array([0.12, 0.13, 0.16], np.float32)
        + sky_grad * np.array([0.66, 0.71, 0.80], np.float32)
        + softbox * np.array([2.25, 2.15, 1.98], np.float32)
    )
    env_blur = np.array([0.36, 0.37, 0.40], np.float32)[None, None, :]
    ibl_rad = env_sharp * ((1.0 - r3) ** 1.45) + env_blur * (r3 ** 1.15)
    ibl_energy = (1.15 * ((1.0 - r3) ** 1.55) + 0.10 * (1.0 - r3)) * (1.0 - m3) + (0.36 + 0.95 * ((1.0 - r3) ** 0.70)) * m3
    spec_ibl = fresnel_ibl * ibl_rad * ibl_energy * (0.28 + 0.72 * ao3)

    # True Cook-Torrance Trowbridge-Reitz GGX direct microfacet specular
    a = r * r
    a2 = np.maximum(a * a, 1e-4)
    k = ((r + 1.0) ** 2) * 0.125
    d1 = a2 / (math.pi * ((ndoth1 * ndoth1 * (a2 - 1.0) + 1.0) ** 2) + 1e-6)
    d2 = a2 / (math.pi * ((ndoth2 * ndoth2 * (a2 - 1.0) + 1.0) ** 2) + 1e-6)
    vis1 = 0.25 / ((ndotl1 * (1.0 - k) + k) * (ndotv * (1.0 - k) + k) + 1e-4)
    vis2 = 0.25 / ((ndotl2 * (1.0 - k) + k) * (ndotv * (1.0 - k) + k) + 1e-4)
    f1 = f0 + (1.0 - f0) * ((1.0 - vdoth1) ** 5)
    f2 = f0 + (1.0 - f0) * ((1.0 - vdoth2) ** 5)
    spec_direct = (
        f1 * (d1 * vis1 * ndotl1 * 2.10)[..., None]
        + f2 * (d2 * vis2 * ndotl2 * 0.75)[..., None]
    ) * (0.35 + 0.65 * ao3)

    diffuse = diff_col * (0.24 + 0.86 * ndotl1[..., None] + 0.26 * ndotl2[..., None]) * (0.28 + 0.72 * ao3)
    col = diffuse + spec_ibl + spec_direct
    col = col / (1.0 + 0.26 * col)
    return np.clip(np.power(np.clip(col, 0.0, 1.0), 1.0 / 2.2) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def synthesize_material(
    material_id: str,
    out_dir: Path | str,
    size: int = 512,
    seed: int | None = None,
    wear: float = 0.0,
) -> Path:
    """Synthesize a complete seamless 15-map PBR material set into `out_dir`.

    Args:
        material_id: Built-in material ID (see `list_materials()`).
        out_dir: Destination directory.
        size: Square output resolution in pixels (default: 512).
        seed: Optional deterministic seed override.
        wear: Physical weathering intensity in `[0.0, 1.0]` (`0.0`=fresh, `0.5`=lived-in, `1.0`=weathered).
    """
    catalog_map = {did: (idx, desc, mat_kind, tint) for idx, (did, desc, mat_kind, tint) in enumerate(gd.DAILY_CATALOG)}
    if material_id not in catalog_map:
        raise KeyError(f"Unknown material_id '{material_id}'. Use list_materials() to see available IDs.")
    idx, _desc, mat_kind, tint = catalog_map[material_id]
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    H = W = size
    s_base = (2000 + idx * 41) if seed is None else seed
    w_amt = float(np.clip(wear, 0.0, 1.0))

    h0, slope_scale, cav0, base_r, ridge0 = gd.build_daily_detail(material_id, H, W, seed=s_base)
    h_n = dp.normalize_height(h0, clip_pct=0.5)
    N = gd._height_to_normal_periodic(h0, slope=slope_scale)
    g_kind = "mineral" if mat_kind in ("concrete", "ceramic") else ("brushed" if mat_kind in ("metal", "carbon") else "cast")
    # Keep micro-grain subtle on flat polished slabs (marble/terrazzo/Damascus/glass)
    g_scale_fac = float(np.clip(slope_scale / 10.0, 0.25, 1.0))
    g_str = (0.032 if mat_kind in ("ceramic", "carbon") else 0.055) * g_scale_fac
    N = ng.node_micro_grain({"circular": True}, {"normal": N}, kind=g_kind, scale=36.0, strength=g_str + 0.045 * w_amt, seed=s_base + 1000)["normal"]
    N = ng.node_normal_clamp_slope({"circular": True}, {"normal": N}, max_angle_deg=58.0)["normal"]

    # Relief-calibrated HBAO: flat slabs keep ao_mean ~0.92-0.97, deep reliefs keep crisp crevices (~0.62-0.80)
    ao_hs = float(np.clip(slope_scale * 0.68, 1.0, 14.0))
    ao_str = float(np.clip(0.55 + 0.042 * slope_scale, 0.55, 1.22))
    hbao = ng.node_horizon_bent_normal_ao(
        {"circular": True}, {"height": h_n, "normal": N},
        strength=ao_str, height_scale=ao_hs, ao_floor=0.18,
    )
    ao = hbao["ao"]
    curv = ng.node_curvature_multiscale({"circular": True}, {"normal": N}, gain=2.8)
    alb_u8, alb_lin, met_u8, r_mod = _rich_albedo_and_metal(material_id, mat_kind, tint, h_n, cav0, ridge0, ao, seed=s_base + 2000)

    r_micro = 0.025 * gd._periodic_fbm(H, W, s_base + 3000, (8, 24, 48))
    rough = np.clip(base_r + r_mod + 0.16 * (1.0 - ao) + r_micro, 0.06, 0.94).astype(np.float32)
    met_f = met_u8.astype(np.float32) / 255.0

    if w_amt > 1e-4:
        w_macro = np.clip(0.5 + 0.42 * gd._periodic_fbm(H, W, s_base + 7777, (4, 8, 16)), 0.0, 1.0)
        w_fine = np.clip(0.5 + 0.45 * gd._periodic_fbm(H, W, s_base + 8888, (16, 32, 64)), 0.0, 1.0)
        valley = np.clip(0.55 * (1.0 - ao) + 0.45 * np.clip(1.0 - h_n, 0.0, 1.0), 0.0, 1.0)
        # Thresholded organic tarnish/patina mask with fine-grain edge breakup (no airbrushed halos)
        raw_patina = valley * 0.80 + w_macro * 0.48 + w_fine * 0.42 - (1.02 - 0.74 * w_amt)
        patina = np.clip(raw_patina * (2.2 + 1.5 * w_amt), 0.0, 0.94)
        patina = (patina * patina * (3.0 - 2.0 * patina)).astype(np.float32)

        edge_scuff = np.clip(curv["convex"] * (0.5 + 0.9 * w_fine) * w_amt * 1.4, 0.0, 0.90)

        is_cu = any(k in material_id for k in ("copper", "bronze", "brass")) or mat_kind in ("copper", "bronze", "brass")
        is_metal = (mat_kind in ("metal", "copper", "bronze", "brass", "gold", "steel", "iron")) or (float(met_f.mean()) > 0.35)
        if is_metal:
            # Deep oxide tarnish + authentic muted verdigris patina (Cu/Br) or dark rust-umber (Fe/steel)
            oxide_dark = np.array([0.072, 0.052, 0.040], dtype=np.float32)[None, None, :]
            patina_accent = np.array([0.095, 0.285, 0.235] if is_cu else [0.21, 0.082, 0.036], dtype=np.float32)[None, None, :]
            # At medium wear (0.45): mostly dark oxide tarnish with subtle verdigris pockets;
            # at heavy wear (0.90): rich blue-green verdigris crust settling across valleys
            accent_w = np.clip((w_macro * 0.55 + valley * 0.45 - (0.58 - 0.36 * w_amt)) * 2.6, 0.0, 1.0)[..., None] * (w_amt ** 0.75)
            tarnish_col = oxide_dark * (1.0 - accent_w) + patina_accent * accent_w
            alb_lin = alb_lin * (1.0 - patina[..., None] * 0.90) + tarnish_col * (patina[..., None] * 0.90)
            # Subtle warm polish on worn high crests
            crest_col = np.clip(alb_lin * 1.22 + 0.04, 0.0, 0.94)
            alb_lin = alb_lin * (1.0 - edge_scuff[..., None] * 0.32) + crest_col * (edge_scuff[..., None] * 0.32)
            # Crisp metallic stripping on oxidized crust (dielectric patina/oxide)
            ox_cut = np.clip((patina - 0.28) / 0.18, 0.0, 1.0)
            ox_cut = ox_cut * ox_cut * (3.0 - 2.0 * ox_cut)
            met_f = np.clip(met_f * (1.0 - ox_cut * 0.96), 0.0, 1.0).astype(np.float32)
            rough = np.clip(rough * (1.0 - patina) + np.clip(0.78 + 0.14 * w_fine, 0.68, 0.95) * patina - 0.12 * edge_scuff * met_f, 0.08, 0.97).astype(np.float32)
        else:
            grime_col = np.array([0.065, 0.060, 0.052], dtype=np.float32)[None, None, :]
            scuff_col = np.clip(alb_lin * 1.28 + 0.08, 0.0, 0.90)
            alb_lin = alb_lin * (1.0 - patina[..., None] * 0.72) + grime_col * (patina[..., None] * 0.72)
            alb_lin = alb_lin * (1.0 - edge_scuff[..., None] * 0.42) + scuff_col * (edge_scuff[..., None] * 0.42)
            rough = np.clip(rough + 0.32 * patina + 0.16 * edge_scuff + 0.06 * w_amt * w_fine, 0.08, 0.97).astype(np.float32)

        alb_u8 = np.clip(np.power(np.clip(alb_lin, 0.0, 1.0), 1.0 / 2.2) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        met_u8 = np.clip(met_f * 255.0 + 0.5, 0, 255).astype(np.uint8)

    rough = ng.node_roughness_specular_aa({"circular": True}, {"roughness": rough, "normal": N}, strength=0.25, max_boost=0.08)["roughness"]
    prev_u8 = render_studio_preview(alb_lin, N, rough, met_f, ao)

    aniso = ng.node_anisotropy_flow({"circular": True}, {"height": h_n, "ao": ao, "roughness": rough})
    thick = ng.node_thickness_subsurface({"circular": True}, {"height": h_n})
    rough_u8 = np.clip(rough * 255.0 + 0.5, 0, 255).astype(np.uint8)
    orm_u8 = np.stack([hbao["ao_u8"], rough_u8, met_u8], axis=-1)

    save_maps_parallel([
        (d / "albedo_delit.png", alb_u8),
        (d / "preview.png", prev_u8),
        (d / "height.png", np.clip(h_n * 255.0 + 0.5, 0, 255).astype(np.uint8)),
        (d / "height16.png", np.clip(h_n * 65535.0 + 0.5, 0, 65535).astype(np.uint16)),
        (d / "normal.png", dp.normal_to_u8(N)),
        (d / "bent_normal.png", hbao["bent_normal_u8"]),
        (d / "ao.png", hbao["ao_u8"]),
        (d / "roughness.png", rough_u8),
        (d / "metallic.png", met_u8),
        (d / "orm.png", orm_u8),
        (d / "curvature.png", curv["curvature_u8"]),
        (d / "cavity.png", curv["cavity_u8"]),
        (d / "edge_wear.png", np.clip(curv["convex"] * 255.0 + 0.5, 0, 255).astype(np.uint8)),
        (d / "anisotropy.png", aniso["anisotropy_u8"]),
        (d / "specular.png", aniso["specular_u8"]),
        (d / "thickness.png", thick["thickness_u8"]),
    ])
    return d


def synthesize_all(
    out_dir: Path | str,
    size: int = 512,
    materials: list[str] | None = None,
    wear: float = 0.0,
) -> list[Path]:
    """Synthesize all (or selected) seamless materials into `out_dir/<material_id>/`."""
    root = Path(out_dir)
    ids = materials if materials is not None else [did for did, *_ in gd.DAILY_CATALOG]
    return [synthesize_material(mid, root / mid, size=size, wear=wear) for mid in ids]


def main():
    t0 = time.time()
    out_root = Path("./output/pbr")
    paths = synthesize_all(out_root, size=512)
    print(f"Synthesized {len(paths)} seamless PBR materials in {time.time() - t0:.1f}s -> {out_root}")


if __name__ == "__main__":
    main()
