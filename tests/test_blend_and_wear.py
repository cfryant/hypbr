"""Tests for physical weathering (`wear`), HeightLerp material blending, and HTML studio preview."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from hypbr import (
    blend_materials,
    render_preview_html,
    synthesize_material,
    verify_material,
)


def test_wear_weathering_is_seamless_and_modulates_roughness(tmp_path: Path) -> None:
    fresh = synthesize_material("hammered_copper_dimple", tmp_path / "fresh", size=128, wear=0.0)
    worn = synthesize_material("hammered_copper_dimple", tmp_path / "worn", size=128, wear=0.85)

    assert verify_material(fresh)["passed"] is True
    assert verify_material(worn)["passed"] is True

    r_fresh = np.asarray(Image.open(fresh / "roughness.png"), dtype=np.float32).mean()
    r_worn = np.asarray(Image.open(worn / "roughness.png"), dtype=np.float32).mean()
    assert r_worn > r_fresh


def test_height_lerp_and_cavity_blend_pass_verify(tmp_path: Path) -> None:
    mat_a = synthesize_material("hammered_copper_dimple", tmp_path / "a", size=128)
    mat_b = synthesize_material("venetian_plaster_trowel", tmp_path / "b", size=128)

    for mode in ("height", "cavity", "linear"):
        out = blend_materials(mat_a, mat_b, tmp_path / f"blend_{mode}", factor=0.45, contrast=0.18, mode=mode)
        assert (out / "blend_mask.png").exists()
        report = verify_material(out)
        assert report["passed"] is True, report


def test_preview_html_renders_cleanly() -> None:
    html = render_preview_html("damascus_folded_steel")
    assert "MeshPhysicalMaterial" in html
    assert "damascus_folded_steel" in html
    assert "bent_normal.png" in html
