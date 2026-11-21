"""Full-catalog seamlessness, PBR range, and resolution-invariance regression tests."""
from pathlib import Path

import numpy as np
from PIL import Image

from hypbr import procedural as proc
from hypbr import synthesize as syn
from hypbr import verify as ver


def test_full_catalog_seamless_and_pbr_ranges(tmp_path: Path):
    """Every built-in procedural material + hardops_plate passes verification with zero warnings."""
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

        if ao.mean() < 0.45:
            range_issues.append((did, f"ao_mean={ao.mean():.3f} < 0.45"))
        if rgh.std() < 0.015:
            range_issues.append((did, f"roughness_std={rgh.std():.4f} < 0.015"))
        if met.mean() < 0.25 and float(np.percentile(alb, 99)) > 243.0:
            range_issues.append((did, f"dielectric_p99={np.percentile(alb, 99):.1f} > 243"))

    assert not failures, f"Verification failures: {failures}"
    assert not warnings, f"Verification warnings: {warnings}"
    assert not range_issues, f"PBR range issues: {range_issues}"


def test_resolution_invariance_256_vs_512(tmp_path: Path):
    """HBAO and normal relief stay consistent across 256px and 512px synthesis."""
    samples = [
        "carrara_veined_marble",
        "hex_mosaic_tile",
        "wood_parquet_herringbone",
        "pcb_solder_mask_traces",
        "cast_iron_skillet_pebble",
    ]
    for did in samples:
        d256 = tmp_path / f"{did}_256"
        d512 = tmp_path / f"{did}_512"
        syn.synthesize_material(did, d256, size=256)
        syn.synthesize_material(did, d512, size=512)
        for sz, d in ((256, d256), (512, d512)):
            rep = ver.verify_material(d)
            assert rep["passed"], f"{did}@{sz} failed: {rep['errors']}"
            assert not rep["warnings"], f"{did}@{sz} warnings: {rep['warnings']}"
        ao256 = float(np.array(Image.open(d256 / "ao.png"), np.float32).mean() / 255.0)
        ao512 = float(np.array(Image.open(d512 / "ao.png"), np.float32).mean() / 255.0)
        assert abs(ao256 - ao512) < 0.04, f"{did} AO drifted: 256={ao256:.3f} vs 512={ao512:.3f}"
