"""Full-catalog seamlessness, PBR range, curvature-sign, and resolution-invariance regression tests."""
from pathlib import Path

import numpy as np
from PIL import Image

from hypbr import blend as bl
from hypbr import normal_graph as ng
from hypbr import procedural as proc
from hypbr import synthesize as syn
from hypbr import verify as ver

OPEN_VOID_METALS = {
    "chainlink_wire_mesh",
    "expanded_metal_mesh",
    "street_storm_grate",
    "hvac_slotted_vent",
    "stamped_vent_slots_stagger",
    "drip_tray_sieve_grate",
    "stamped_louver_micro",
    "speaker_mesh_hex",
    "circular_perf_sheet",
}

SPECIAL_MIXED_METALS = {
    "gilded_tin_ceiling_tile",
    "cast_iron_skillet_pebble",
    "pcb_solder_mask_traces",
    "damascus_folded_steel",
    "hammered_copper_dimple",
    "chainlink_fence_galvanized",
}


def test_full_catalog_seamless_and_pbr_ranges(tmp_path: Path):
    """Every built-in procedural material + hardops_plate passes verification with zero warnings."""
    catalog_kinds = {item[0]: item[2] for item in proc.DAILY_CATALOG}
    all_ids = [item[0] for item in proc.DAILY_CATALOG] + ["hardops_plate"]
    failures = []
    warnings = []
    range_issues = []

    for did in all_ids:
        out = tmp_path / did
        if did == "hardops_plate":
            syn.synthesize_hardops_plate(out, size=256)
        else:
            syn.synthesize_material(did, out, size=256)
        rep = ver.verify_material(out)
        if not rep["passed"]:
            failures.append((did, rep["errors"]))
        if rep["warnings"]:
            warnings.append((did, rep["warnings"]))

        ao = np.array(Image.open(out / "ao.png"), np.float32) / 255.0
        rgh = np.array(Image.open(out / "roughness.png"), np.float32) / 255.0
        alb = np.array(Image.open(out / "albedo_delit.png"), np.float32)
        met = np.array(Image.open(out / "metallic.png"), np.float32) / 255.0
        nrm = np.array(Image.open(out / "normal.png"), np.float32) / 127.5 - 1.0
        slope_sin = np.sqrt(nrm[..., 0] ** 2 + nrm[..., 1] ** 2)
        thk = np.array(Image.open(out / "thickness.png"), np.float32) / 255.0

        if ao.mean() < 0.45:
            range_issues.append((did, f"ao_mean={ao.mean():.3f} < 0.45"))
        if rgh.std() < 0.015:
            range_issues.append((did, f"roughness_std={rgh.std():.4f} < 0.015"))
        if met.mean() < 0.25 and float(np.percentile(alb, 99)) > 243.0:
            range_issues.append((did, f"dielectric_p99={np.percentile(alb, 99):.1f} > 243"))
        if float(np.median(slope_sin)) > 0.82:
            range_issues.append((did, f"median_slope_sin={np.median(slope_sin):.3f} > 0.82"))
        if float(thk.min()) < 0.08:
            range_issues.append((did, f"thickness_min={thk.min():.3f} < 0.08"))

        kind = catalog_kinds.get(did, "metal")
        if did in OPEN_VOID_METALS:
            if float(met.min()) > 0.05 or float(met.max()) < 0.90:
                range_issues.append((did, f"open_void_metal=[{met.min():.2f},{met.max():.2f}]"))
        elif kind == "metal" and did not in SPECIAL_MIXED_METALS and did != "hardops_plate":
            if float(met.min()) < 0.80:
                range_issues.append((did, f"bare_metal_min={met.min():.3f} < 0.80"))

    assert not failures, f"Verification failures: {failures}"
    assert not warnings, f"Verification warnings: {warnings}"
    assert not range_issues, f"PBR range issues: {range_issues}"


def test_curvature_sign_orientation():
    """Convex dome peaks must produce positive curvature/convexity; concave troughs must reduce cavity."""
    H = W = 128
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r2 = ((xx - 64.0) / 14.0) ** 2 + ((yy - 64.0) / 14.0) ** 2
    h_dome = np.exp(-r2).astype(np.float32)
    N_dome = proc._height_to_normal_periodic(h_dome, slope=10.0)
    curv = ng.node_curvature_multiscale({"circular": True}, {"normal": N_dome}, gain=3.0)

    assert curv["curvature"][64, 64] > 0.55, f"dome peak curvature={curv['curvature'][64, 64]:.3f}"
    assert curv["convex"][64, 64] > 0.10, f"dome peak convex={curv['convex'][64, 64]:.3f}"
    assert curv["concave"][64, 64] == 0.0, f"dome peak concave={curv['concave'][64, 64]:.3f}"
    assert curv["curvature"][64, 84] < 0.50, f"dome ring curvature={curv['curvature'][64, 84]:.3f}"
    assert curv["concave"][64, 84] > 0.0, f"dome ring concave={curv['concave'][64, 84]:.3f}"
    assert curv["cavity"][64, 84] < curv["cavity"][64, 64]


def test_resolution_invariance_multimap(tmp_path: Path):
    """All spatial PBR maps stay UV-consistent across 256px and 512px synthesis."""
    samples = [
        "carrara_veined_marble",
        "hex_mosaic_tile",
        "diamond_tread_plate",
        "wood_parquet_herringbone",
        "cast_iron_skillet_pebble",
        "hardops_plate",
    ]
    maps_to_check = (
        "ao.png",
        "roughness.png",
        "curvature.png",
        "cavity.png",
        "edge_wear.png",
        "thickness.png",
        "anisotropy.png",
        "normal.png",
    )

    def _stat(p: Path, name: str) -> float:
        arr = np.array(Image.open(p), np.float32) / 255.0
        if name == "normal.png":
            n = arr * 2.0 - 1.0
            return float(np.mean(np.sqrt(n[..., 0] ** 2 + n[..., 1] ** 2)))
        if name == "anisotropy.png":
            return float(arr[..., 2].mean())
        return float(arr.mean())

    for did in samples:
        d256 = tmp_path / f"{did}_256"
        d512 = tmp_path / f"{did}_512"
        if did == "hardops_plate":
            syn.synthesize_hardops_plate(d256, size=256)
            syn.synthesize_hardops_plate(d512, size=512)
        else:
            syn.synthesize_material(did, d256, size=256)
            syn.synthesize_material(did, d512, size=512)
        for sz, d in ((256, d256), (512, d512)):
            rep = ver.verify_material(d)
            assert rep["passed"], f"{did}@{sz} failed: {rep['errors']}"
            assert not rep["warnings"], f"{did}@{sz} warnings: {rep['warnings']}"
        for m in maps_to_check:
            v256 = _stat(d256 / m, m)
            v512 = _stat(d512 / m, m)
            assert abs(v256 - v512) < 0.045, f"{did} {m} drifted: 256={v256:.3f} vs 512={v512:.3f}"


def test_weathering_and_blending_quality(tmp_path: Path):
    """Weathered metals/dielectrics and height/cavity material blends stay seamless with zero warnings."""
    d_cu = tmp_path / "cu_wear"
    d_tile = tmp_path / "tile_wear"
    d_tread = tmp_path / "tread"
    d_tile0 = tmp_path / "tile0"
    d_cu0 = tmp_path / "cu0"
    d_iron0 = tmp_path / "iron0"

    syn.synthesize_material("hammered_copper_dimple", d_cu, size=256, wear=0.45)
    syn.synthesize_material("subway_metro_tile", d_tile, size=256, wear=0.45)
    syn.synthesize_material("diamond_tread_plate", d_tread, size=256)
    syn.synthesize_material("subway_metro_tile", d_tile0, size=256)
    syn.synthesize_material("hammered_copper_dimple", d_cu0, size=256)
    syn.synthesize_material("cast_iron_skillet_pebble", d_iron0, size=256)

    d_bh = tmp_path / "blend_height"
    d_bc = tmp_path / "blend_cavity"
    bl.blend_materials(d_tread, d_tile0, d_bh, factor=0.5, mode="height")
    bl.blend_materials(d_cu0, d_iron0, d_bc, factor=0.45, mode="cavity")

    for d in (d_cu, d_tile, d_bh, d_bc):
        rep = ver.verify_material(d)
        assert rep["passed"], f"{d.name} failed: {rep['errors']}"
        assert not rep["warnings"], f"{d.name} warnings: {rep['warnings']}"

    alb_tile = np.array(Image.open(d_tile / "albedo_delit.png"), np.float32)
    assert float(np.percentile(alb_tile, 99)) <= 243.0
