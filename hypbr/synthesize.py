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
  4. Upgrades `matball-studio/materials/hardops-plate` to a pure gunmetal machined HardOps panel
     (replacing the green palm-tree photo backdrop).
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
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
        # Classic indigo warp yarns + warm ecru/white weft yarns!
        nu, nv = 32.0, 32.0
        phase = (np.floor(u * nu) + np.floor(v * nv)) % 4.0
        warp = (phase < 3.0).astype(np.float32)
        indigo = np.array([0.11, 0.19, 0.36], np.float32)[None, None, :] * (0.88 + 0.12 * m1[..., None])
        ecru = np.array([0.66, 0.64, 0.59], np.float32)[None, None, :] * (0.92 + 0.08 * m2[..., None])
        alb = indigo * warp[..., None] + ecru * (1.0 - warp[..., None])
        # Subtle denim ridge fade
        alb += 0.06 * ridge[..., None]

    elif did == "quilted_leather_stitch":
        # Rich espresso/saddle leather + warm tan contrast stitching in the seams!
        leath = np.array([0.28, 0.14, 0.09], np.float32)[None, None, :] * (0.88 + 0.12 * m1[..., None] + 0.05 * m2[..., None])
        stitch = np.array([0.74, 0.56, 0.36], np.float32)[None, None, :]
        # Highlight raised stitch dashes inside seam valleys
        stitch_m = np.clip((h - 0.42) * 4.0, 0.0, 1.0) * cav
        alb = leath * (1.0 - stitch_m[..., None]) + stitch * stitch_m[..., None]
        alb *= (0.72 + 0.28 * ao[..., None])
        r_mod = -0.06 * ridge + 0.28 * cav

    elif did == "wood_parquet_herringbone":
        # Rich multi-tone oak planks with distinct per-plank warmth & dark bevel seams
        rx, ry = (u + v) * 4.0, (u - v) * 4.0
        ix, iy = np.floor(rx), np.floor(ry)
        s = (ix + iy) % 2.0
        plank_id = np.sin(ix * 12.7 + iy * 31.3) * 0.5 + 0.5
        grain = 0.5 + 0.5 * np.where(
            s < 0.5,
            np.sin(2.0 * math.pi * ry * 14.0 + 1.8 * m1),
            np.sin(2.0 * math.pi * rx * 14.0 + 1.8 * m1),
        )
        oak_light = np.array([0.58, 0.39, 0.22], np.float32)[None, None, :]
        oak_dark = np.array([0.34, 0.21, 0.11], np.float32)[None, None, :]
        plank_mix = np.clip(0.35 + 0.40 * plank_id + 0.25 * grain, 0.0, 1.0)[..., None]
        alb = oak_dark * (1.0 - plank_mix) + oak_light * plank_mix
        alb *= (1.0 - 0.55 * cav[..., None])
        r_mod = 0.25 * cav - 0.05 * grain

    elif did == "acoustic_slat_wood":
        # Warm natural oak slats over deep charcoal acoustic felt valleys
        grain = 0.5 + 0.5 * np.sin(2.0 * math.pi * u * 72.0 + 1.5 * m1)
        oak = np.array([0.56, 0.39, 0.23], np.float32)[None, None, :] * (0.82 + 0.24 * grain[..., None] + 0.08 * m1[..., None])
        felt = np.array([0.07, 0.07, 0.08], np.float32)[None, None, :] * (0.85 + 0.15 * m2[..., None])
        alb = oak * (1.0 - cav[..., None]) + felt * cav[..., None]
        r_mod = 0.48 * cav

    elif did == "carbon_fiber_weave":
        # Alternating graphite / charcoal anisotropic carbon fiber tows
        n = 16.0
        twill = ((np.floor(u * n) + np.floor(v * n)) % 4.0) < 2.0
        c1 = np.array([0.11, 0.12, 0.14], np.float32)[None, None, :]
        c2 = np.array([0.22, 0.23, 0.26], np.float32)[None, None, :]
        alb = np.where(twill[..., None], c1, c2) * (0.85 + 0.20 * h[..., None])
        met = np.full((H, W), 0.22, np.float32) * (1.0 - 0.5 * cav)
        r_mod = -0.06 * ridge + 0.18 * cav

    elif did == "corrugated_cardboard":
        # Warm kraft paper pulp with fine fiber speckles
        kraft = np.array([0.58, 0.43, 0.28], np.float32)[None, None, :]
        speck = np.clip((m2 - 1.1) * 2.0, 0.0, 1.0)[..., None]
        alb = kraft * (0.90 + 0.10 * m1[..., None]) - 0.12 * speck
        r_mod = 0.10 * cav

    elif did == "terrazzo_speckled_stone":
        # Warm cream honed cement matrix + terracotta, slate-blue & charcoal marble aggregate chips!
        cem = np.array([0.78, 0.75, 0.70], np.float32)[None, None, :] * (0.95 + 0.05 * m2[..., None])
        c_terra = np.array([0.62, 0.30, 0.20], np.float32)[None, None, :]
        c_slate = np.array([0.24, 0.30, 0.36], np.float32)[None, None, :]
        c_ochre = np.array([0.74, 0.58, 0.32], np.float32)[None, None, :]
        m3 = gd._periodic_fbm(H, W, seed + 205, (12, 24, 48))
        k1 = gd._smoothstep(0.58, 0.75, m1)[..., None]
        k2 = gd._smoothstep(0.62, 0.80, m3)[..., None]
        k3 = gd._smoothstep(0.65, 0.85, -m1)[..., None]
        alb = cem * (1.0 - k1) + c_terra * k1
        alb = alb * (1.0 - k2) + c_slate * k2
        alb = alb * (1.0 - k3) + c_ochre * k3
        r_mod = -0.10 * np.clip(k1[..., 0] + k2[..., 0] + k3[..., 0], 0.0, 1.0)

    elif did == "penny_round_mosaic":
        # Deep glazed teal/sea-glass penny tiles + crisp off-white sanded grout
        glaze = np.array([0.09, 0.34, 0.37], np.float32)[None, None, :] * (0.85 + 0.22 * m1[..., None])
        grout = np.array([0.78, 0.77, 0.74], np.float32)[None, None, :] * (0.92 + 0.08 * m2[..., None])
        alb = glaze * (1.0 - cav[..., None]) + grout * cav[..., None]
        r_mod = -0.06 * (1.0 - cav) + 0.62 * cav

    elif did == "hex_mosaic_tile":
        # Classic matte-white 1-inch hex tiles with occasional charcoal accent hexes + cool gray grout
        white_t = np.array([0.86, 0.86, 0.84], np.float32)[None, None, :] * (0.95 + 0.05 * m1[..., None])
        grout = np.array([0.38, 0.40, 0.42], np.float32)[None, None, :]
        alb = white_t * (1.0 - cav[..., None]) + grout * cav[..., None]
        r_mod = 0.52 * cav

    elif did == "hammered_copper_dimple":
        # Hand-planished warm rose copper with bright specular crests & slightly oxidized dimple centers
        cop = np.array([0.82, 0.44, 0.26], np.float32)[None, None, :] * (0.92 + 0.08 * m1[..., None])
        pat = np.array([0.46, 0.24, 0.16], np.float32)[None, None, :]
        alb = pat * (0.35 * cav[..., None]) + cop * (1.0 - 0.35 * cav[..., None])
        met = np.clip(0.95 * (1.0 - 0.20 * cav), 0.5, 0.99).astype(np.float32)
        r_mod = -0.08 * ridge + 0.22 * cav

    elif did in ("fluted_reeded_glass", "frosted_prism_diffuser"):
        # Architectural cool sea-green / frosted crystal glass
        glass = np.array([0.72, 0.84, 0.83], np.float32)[None, None, :] * (0.94 + 0.06 * m1[..., None])
        alb = glass * (0.85 + 0.15 * ao[..., None])
        r_mod = -0.04 * ridge + 0.08 * cav

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
        # Three-stage bronze patina: satin hammered bronze -> dark russet cuprite oxide halo -> chalky matte verdigris crust!
        bronze = np.array([0.48, 0.33, 0.18], np.float32)[None, None, :] * (0.90 + 0.14 * m1[..., None])
        cuprite = np.array([0.22, 0.11, 0.06], np.float32)[None, None, :] * (0.90 + 0.12 * m2[..., None])
        verd_lo = np.array([0.14, 0.42, 0.36], np.float32)[None, None, :]
        verd_hi = np.array([0.30, 0.62, 0.52], np.float32)[None, None, :]
        v_mix = np.clip(0.5 + 0.45 * m2, 0.0, 1.0)[..., None]
        verd = verd_lo * (1.0 - v_mix) + verd_hi * v_mix
        alb = bronze * (1.0 - ridge[..., None]) + cuprite * ridge[..., None]
        alb = alb * (1.0 - cav[..., None]) + verd * cav[..., None]
        met = np.clip(0.92 * (1.0 - 0.55 * ridge) * (1.0 - cav), 0.0, 0.95).astype(np.float32)
        r_mod = 0.24 * ridge + 0.58 * cav

    elif did == "carrara_veined_marble":
        # Luminous cool-white calcite + broad smoky-grey subsurface clouding + crisp directional charcoal mineral veins!
        calcite = np.array([0.92, 0.91, 0.89], np.float32)[None, None, :] * (0.97 + 0.03 * m1[..., None])
        cloud_c = np.array([0.72, 0.74, 0.77], np.float32)[None, None, :]
        core_c = np.array([0.32, 0.34, 0.38], np.float32)[None, None, :]
        alb = calcite * (1.0 - cav[..., None]) + cloud_c * cav[..., None]
        alb = alb * (1.0 - ridge[..., None]) + core_c * ridge[..., None]
        r_mod = 0.04 * cav + 0.14 * ridge

    elif did == "herringbone_tweed_wool":
        # Authentic warm Harris/Donegal herringbone tweed (18 twill yarns, zero moire!) + subtle Donegal rust flecks
        col_i = np.floor(u * 6.0)
        sign = np.where((col_i % 2.0) < 0.5, 1.0, -1.0)
        yarn_t = (np.sin(2.0 * math.pi * (u * 18.0 + sign * v * 18.0)) > 0.0).astype(np.float32)
        heather = np.array([0.22, 0.21, 0.20], np.float32)[None, None, :] * (0.88 + 0.16 * m1[..., None])
        oat = np.array([0.56, 0.50, 0.42], np.float32)[None, None, :] * (0.88 + 0.16 * m2[..., None])
        fleck = gd._smoothstep(1.35, 1.75, m2)[..., None]
        rust = np.array([0.48, 0.26, 0.16], np.float32)[None, None, :]
        alb = heather * (1.0 - yarn_t[..., None]) + oat * yarn_t[..., None]
        alb = alb * (1.0 - 0.55 * fleck) + rust * (0.55 * fleck)
        alb *= (0.74 + 0.26 * ao[..., None])

    elif did == "waffle_cotton_linen":
        # Natural woven sage-eucalyptus cotton with bright sunlit interlaced thread crests & warm shadowed waffle pockets
        sage = np.array([0.40, 0.49, 0.42], np.float32)[None, None, :] * (0.90 + 0.12 * m1[..., None])
        linen_hi = np.array([0.56, 0.62, 0.55], np.float32)[None, None, :]
        alb = sage * (1.0 - 0.35 * ridge[..., None]) + linen_hi * (0.35 * ridge[..., None])
        alb *= (0.64 + 0.36 * ao[..., None])

    elif did == "damascus_folded_steel":
        # Authentic Japanese folded Damascus steel: crisp metallic luster contrasting bright nickel-steel & acid-etched dark steel
        dark_s = np.array([0.24, 0.26, 0.30], np.float32)[None, None, :]
        bright_s = np.array([0.80, 0.82, 0.86], np.float32)[None, None, :]
        alb = dark_s * (1.0 - ridge[..., None]) + bright_s * ridge[..., None]
        met = np.clip(0.93 + 0.06 * ridge, 0.86, 0.99).astype(np.float32)
        r_mod = -0.04 * ridge + 0.14 * cav

    elif did == "mermaid_fishscale_tile":
        # Handmade Moroccan Zellige fish-scale tiles: TRUE painter's-order scallop ID (zero square-grid cuts!) + rim pooling + matte grout
        nx, ny = 6.0, 10.0
        wobble = gd._periodic_fbm(H, W, seed + 126, (8, 16, 32))
        tile_var = np.full((H, W), 0.5, dtype=np.float32)
        base_ry = np.floor(v * ny)
        for row_off in (-2, -1, 0, 1, 2):
            ry = base_ry + row_off
            u_s = (u + 0.5 * (ry % 2.0) / nx) % 1.0
            rx = np.floor(u_s * nx)
            fu = (u_s * nx) % 1.0 - 0.5
            fv = (v * ny - ry) * 0.5
            r = np.sqrt(fu * fu + fv * fv) + 0.010 * wobble
            inside = r < 0.50
            tid = np.sin((rx % nx) * 17.3 + (ry % ny) * 31.7) * 0.5 + 0.5
            tile_var = np.where(inside, tid, tile_var)
        teal = np.array([0.04, 0.28, 0.34], np.float32)[None, None, :]
        jade = np.array([0.10, 0.42, 0.40], np.float32)[None, None, :]
        deep = np.array([0.02, 0.16, 0.23], np.float32)[None, None, :]
        glaze = teal * (1.0 - tile_var[..., None]) + jade * tile_var[..., None]
        glaze = glaze * (0.90 + 0.14 * m1[..., None])
        glaze = glaze * (1.0 - 0.45 * ridge[..., None]) + deep * (0.45 * ridge[..., None])
        grout = np.array([0.70, 0.68, 0.63], np.float32)[None, None, :] * (0.92 + 0.08 * m2[..., None])
        alb = glaze * (1.0 - cav[..., None]) + grout * cav[..., None]
        r_mod = -0.04 * (1.0 - cav) + 0.78 * cav

    elif did == "burlap_jute_sackcloth":
        # Authentic raw jute burlap: warm golden-tan twisted fibers + DEEP shadowed open weave pores!
        jute_hi = np.array([0.74, 0.60, 0.42], np.float32)[None, None, :]
        jute_mid = np.array([0.52, 0.38, 0.23], np.float32)[None, None, :]
        jute_lo = np.array([0.06, 0.04, 0.02], np.float32)[None, None, :]
        j_mix = np.clip(0.32 + 0.48 * ridge + 0.20 * m1, 0.0, 1.0)[..., None]
        alb = (jute_mid * (1.0 - j_mix) + jute_hi * j_mix) * (1.0 - cav[..., None]) + jute_lo * cav[..., None]

    elif did == "cast_iron_skillet_pebble":
        # Authentic Lodge seasoned black cast-iron: deep carbon black with satin polymerized-oil sheen (not wet tar)
        pit_c = np.array([0.045, 0.045, 0.050], np.float32)[None, None, :]
        crown_c = np.array([0.115, 0.115, 0.128], np.float32)[None, None, :]
        alb = pit_c * (1.0 - ridge[..., None]) + crown_c * ridge[..., None]
        met = np.clip(0.70 * (1.0 - 0.45 * cav), 0.22, 0.82).astype(np.float32)
        r_mod = -0.06 * ridge + 0.28 * cav

    elif did == "tufted_emerald_velvet":
        # Rich deep jewel-toned Chesterfield emerald velvet with crushed-velvet shadow depth in diamond creases
        em_hi = np.array([0.025, 0.20, 0.13], np.float32)[None, None, :]
        em_lo = np.array([0.004, 0.035, 0.022], np.float32)[None, None, :]
        alb = em_lo * cav[..., None] + em_hi * (1.0 - cav[..., None]) * (0.86 + 0.16 * m1[..., None])
        r_mod = -0.04 * ridge + 0.10 * cav

    elif did == "woven_bamboo_tatami":
        # Authentic Japanese igusa rush tatami: warm golden-olive-straw stalks with organic reed variation (no sterile ivory!)
        stalk_id = np.sin(np.floor(v * 56.0) * 23.7 + np.floor(u * 14.0) * 11.3) * 0.5 + 0.5
        straw = np.array([0.62, 0.54, 0.30], np.float32)[None, None, :]
        igusa = np.array([0.42, 0.46, 0.24], np.float32)[None, None, :]
        gold = np.array([0.70, 0.58, 0.32], np.float32)[None, None, :]
        t1 = np.clip(0.25 + 0.55 * stalk_id + 0.20 * m1, 0.0, 1.0)[..., None]
        alb = igusa * (1.0 - t1) + straw * t1
        alb = alb * (1.0 - 0.22 * ridge[..., None]) + gold * (0.22 * ridge[..., None])
        alb *= (0.76 + 0.24 * ao[..., None])

    elif did == "travertine_porous_stone":
        # Authentic honed Roman travertine: warm cream-ochre sedimentary strata + deep mineral-lined erosion pores
        ivory = np.array([0.76, 0.68, 0.54], np.float32)[None, None, :]
        honey = np.array([0.56, 0.46, 0.33], np.float32)[None, None, :]
        vein = np.array([0.44, 0.35, 0.24], np.float32)[None, None, :]
        pore = np.array([0.16, 0.12, 0.08], np.float32)[None, None, :]
        band1 = (0.5 + 0.5 * np.sin(2.0 * math.pi * (v * 8.0 + 0.20 * m1)))[..., None]
        band2 = (0.5 + 0.5 * np.sin(2.0 * math.pi * (v * 18.0 - 0.14 * m1)))[..., None]
        stone = (honey * (1.0 - band1) + ivory * band1) * (0.90 + 0.10 * band2) * (0.95 + 0.05 * m2[..., None])
        stone = stone * 0.88 + vein * 0.12 * (1.0 - band2)
        alb = stone * (1.0 - 0.88 * cav[..., None]) + pore * (0.88 * cav[..., None])
        r_mod = 0.34 * cav

    elif did == "gilded_tin_ceiling_tile":
        # Authentic warm burnished gold-leaf / antique brass pressed tin with dark umber-soot recess patina
        gold = np.array([0.92, 0.72, 0.34], np.float32)[None, None, :] * (0.92 + 0.10 * m1[..., None])
        patina = np.array([0.18, 0.12, 0.06], np.float32)[None, None, :]
        alb = gold * (1.0 - 0.82 * cav[..., None]) + patina * (0.82 * cav[..., None])
        met = np.clip(0.97 * (1.0 - 0.78 * cav), 0.18, 0.99).astype(np.float32)
        r_mod = -0.06 * ridge + 0.52 * cav

    elif mat_kind == "metal":
        # Machined/cast metal: oil-darkened recesses + bright specular-polished chamfered crests!
        oil = np.array([0.10, 0.10, 0.11], np.float32)[None, None, :]
        crest = np.clip(base * 1.28 + 0.08, 0.0, 0.95)
        alb = base * (0.90 + 0.12 * m1[..., None] + 0.05 * m2[..., None])
        alb = alb * (1.0 - 0.65 * cav[..., None]) + oil * (0.65 * cav[..., None])
        alb = alb * (1.0 - 0.35 * ridge[..., None]) + crest * (0.35 * ridge[..., None])
        met = np.clip(0.92 * (1.0 - 0.55 * cav), 0.15, 0.98).astype(np.float32)
        r_mod = -0.08 * ridge + 0.32 * cav

    else:
        # Leather / rubber / fabric / concrete / ceramic fallback with rich organic tonal breakup
        alb = base * (0.88 + 0.10 * m1[..., None] + 0.05 * m2[..., None])
        alb = alb * (0.76 + 0.24 * ao[..., None]) + 0.04 * ridge[..., None]
        r_mod = 0.22 * cav - 0.05 * ridge

    alb_srgb = np.power(np.clip(alb, 0.015, 0.96), 1.0 / 2.2)
    return (
        np.clip(alb_srgb * 255.0 + 0.5, 0, 255).astype(np.uint8),
        np.clip(alb, 0.015, 0.96).astype(np.float32),
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
    hbao = ng.node_horizon_bent_normal_ao(ctx, {"height": h_n, "normal": n_fin}, strength=1.55, height_scale=18.0)
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

    rough = np.clip(0.32 + 0.36 * (1.0 - ao) - 0.14 * curv["convex"] + 0.04 * m1, 0.12, 0.92).astype(np.float32)
    rough = ng.node_roughness_specular_aa(ctx, {"roughness": rough, "normal": n_fin}, strength=0.25, max_boost=0.08)["roughness"]
    metal = np.clip(0.92 * (0.55 + 0.45 * ao) + 0.08 * curv["convex"], 0.25, 1.0).astype(np.float32)
    aniso = ng.node_anisotropy_flow(ctx, {"height": h_n, "ao": ao, "roughness": rough})
    thick = ng.node_thickness_subsurface(ctx, {"height": h_n})

    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(alb_u8).save(out_dir / "albedo_delit.png")
    Image.fromarray(alb_u8).save(out_dir / "source_512.png")
    Image.fromarray(np.clip(h_n * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(out_dir / "height.png")
    Image.fromarray(np.clip(h_n * 65535.0 + 0.5, 0, 65535).astype(np.uint16)).save(out_dir / "height16.png")
    Image.fromarray(dp.normal_to_u8(n_fin)).save(out_dir / "normal.png")
    Image.fromarray(hbao["bent_normal_u8"]).save(out_dir / "bent_normal.png")
    Image.fromarray(hbao["ao_u8"]).save(out_dir / "ao.png")
    Image.fromarray(np.clip(rough * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(out_dir / "roughness.png")
    Image.fromarray(np.clip(metal * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(out_dir / "metallic.png")
    Image.fromarray(np.stack([hbao["ao_u8"], np.clip(rough*255+0.5,0,255).astype(np.uint8), np.clip(metal*255+0.5,0,255).astype(np.uint8)], -1)).save(out_dir / "orm.png")
    Image.fromarray(curv["curvature_u8"]).save(out_dir / "curvature.png")
    Image.fromarray(curv["cavity_u8"]).save(out_dir / "cavity.png")
    Image.fromarray(np.clip(curv["convex"] * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(out_dir / "edge_wear.png")
    Image.fromarray(aniso["anisotropy_u8"]).save(out_dir / "anisotropy.png")
    Image.fromarray(aniso["specular_u8"]).save(out_dir / "specular.png")
    Image.fromarray(thick["thickness_u8"]).save(out_dir / "thickness.png")
    return out_dir


def list_materials() -> list[dict[str, str]]:
    """Return metadata for all built-in procedural PBR materials."""
    return [
        {"id": did, "description": desc, "category": mat_kind}
        for did, desc, mat_kind, _ in gd.DAILY_CATALOG
    ]


def synthesize_material(material_id: str, out_dir: Path | str, size: int = 512, seed: int | None = None) -> Path:
    """Synthesize a complete seamless 15-map PBR material set into `out_dir`."""
    catalog_map = {did: (idx, desc, mat_kind, tint) for idx, (did, desc, mat_kind, tint) in enumerate(gd.DAILY_CATALOG)}
    if material_id not in catalog_map:
        raise KeyError(f"Unknown material_id '{material_id}'. Use list_materials() to see available IDs.")
    idx, _desc, mat_kind, tint = catalog_map[material_id]
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    H = W = size
    s_base = (2000 + idx * 41) if seed is None else seed

    h0, slope_scale, cav0, base_r, ridge0 = gd.build_daily_detail(material_id, H, W, seed=s_base)
    h_n = dp.normalize_height(h0, clip_pct=0.5)
    N = gd._height_to_normal_periodic(h0, slope=slope_scale)
    g_kind = "mineral" if mat_kind in ("concrete", "ceramic") else ("brushed" if mat_kind in ("metal", "carbon") else "cast")
    g_str = 0.04 if mat_kind in ("ceramic", "carbon") else 0.065
    N = ng.node_micro_grain({"circular": True}, {"normal": N}, kind=g_kind, scale=36.0, strength=g_str, seed=s_base + 1000)["normal"]
    N = ng.node_normal_clamp_slope({"circular": True}, {"normal": N}, max_angle_deg=58.0)["normal"]

    hbao = ng.node_horizon_bent_normal_ao({"circular": True}, {"height": h_n, "normal": N}, strength=1.50, height_scale=18.0)
    ao = hbao["ao"]
    alb_u8, alb_lin, met_u8, r_mod = _rich_albedo_and_metal(material_id, mat_kind, tint, h_n, cav0, ridge0, ao, seed=s_base + 2000)

    rough = np.clip(base_r + r_mod + 0.24 * (1.0 - ao), 0.08, 0.96).astype(np.float32)
    rough = ng.node_roughness_specular_aa({"circular": True}, {"roughness": rough, "normal": N}, strength=0.25, max_boost=0.08)["roughness"]

    L1 = np.array([0.48, 0.52, 0.71], np.float32); L1 /= np.linalg.norm(L1)
    L2 = np.array([-0.60, -0.30, 0.74], np.float32); L2 /= np.linalg.norm(L2)
    H1 = L1 + np.array([0.0, 0.0, 1.0], np.float32); H1 /= np.linalg.norm(H1)
    ndotl1 = np.clip((N * L1[None, None, :]).sum(-1), 0.0, 1.0)
    ndotl2 = np.clip((N * L2[None, None, :]).sum(-1), 0.0, 1.0)
    ndoth1 = np.clip((N * H1[None, None, :]).sum(-1), 0.0, 1.0)
    shin = np.clip(2.0 / (np.maximum(rough, 0.08) ** 2.4) - 2.0, 2.0, 180.0)
    met_f = met_u8.astype(np.float32) / 255.0
    spec_col = (1.0 - met_f[..., None]) * 0.8 + met_f[..., None] * alb_lin
    spec = (ndoth1 ** shin)[..., None] * (1.0 - rough[..., None] * 0.80) * 0.75 * spec_col
    col = alb_lin * (0.18 + 0.80 * ndotl1[..., None] + 0.22 * ndotl2[..., None]) * (0.28 + 0.72 * ao[..., None]) + spec * (0.35 + 0.65 * ao[..., None])
    prev_u8 = np.clip(np.power(np.clip(col, 0.0, 1.0), 1.0 / 2.2) * 255.0 + 0.5, 0, 255).astype(np.uint8)

    curv = ng.node_curvature_multiscale({"circular": True}, {"normal": N}, gain=2.8)
    aniso = ng.node_anisotropy_flow({"circular": True}, {"height": h_n, "ao": ao, "roughness": rough})
    thick = ng.node_thickness_subsurface({"circular": True}, {"height": h_n})
    rough_u8 = np.clip(rough * 255.0 + 0.5, 0, 255).astype(np.uint8)
    orm_u8 = np.stack([hbao["ao_u8"], rough_u8, met_u8], axis=-1)

    Image.fromarray(alb_u8).save(d / "albedo_delit.png")
    Image.fromarray(prev_u8).save(d / "preview.png")
    Image.fromarray(np.clip(h_n * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(d / "height.png")
    Image.fromarray(np.clip(h_n * 65535.0 + 0.5, 0, 65535).astype(np.uint16)).save(d / "height16.png")
    Image.fromarray(dp.normal_to_u8(N)).save(d / "normal.png")
    Image.fromarray(hbao["bent_normal_u8"]).save(d / "bent_normal.png")
    Image.fromarray(hbao["ao_u8"]).save(d / "ao.png")
    Image.fromarray(rough_u8).save(d / "roughness.png")
    Image.fromarray(met_u8).save(d / "metallic.png")
    Image.fromarray(orm_u8).save(d / "orm.png")
    Image.fromarray(curv["curvature_u8"]).save(d / "curvature.png")
    Image.fromarray(curv["cavity_u8"]).save(d / "cavity.png")
    Image.fromarray(np.clip(curv["convex"] * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(d / "edge_wear.png")
    Image.fromarray(aniso["anisotropy_u8"]).save(d / "anisotropy.png")
    Image.fromarray(aniso["specular_u8"]).save(d / "specular.png")
    Image.fromarray(thick["thickness_u8"]).save(d / "thickness.png")
    return d


def synthesize_all(out_dir: Path | str, size: int = 512, materials: list[str] | None = None) -> list[Path]:
    """Synthesize all (or selected) seamless materials into `out_dir/<material_id>/`."""
    root = Path(out_dir)
    ids = materials if materials is not None else [did for did, *_ in gd.DAILY_CATALOG]
    return [synthesize_material(mid, root / mid, size=size) for mid in ids]


def main():
    t0 = time.time()
    out_root = Path("./output/pbr")
    paths = synthesize_all(out_root, size=512)
    print(f"Synthesized {len(paths)} seamless PBR materials in {time.time() - t0:.1f}s -> {out_root}")


if __name__ == "__main__":
    main()
