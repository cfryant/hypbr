"""Tests for `hypbr.verify.verify_material` seamlessness and physical PBR QA checks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from hypbr import synthesize_material, verify_material


def test_verify_passes_on_synthesized_material(tmp_path: Path) -> None:
    out = synthesize_material("patinated_verdigris_bronze", tmp_path / "bronze", size=128)
    report = verify_material(out)
    assert report["passed"] is True, report
    assert report["checks"]["orm_max_channel_diff"] == 0
    assert report["checks"]["normal"]["positive_z_fraction"] == 1.0
    assert report["checks"]["seamless"]["height"]["seam_ratio"] < 2.0


def test_verify_catches_broken_seam_and_bad_orm(tmp_path: Path) -> None:
    out = synthesize_material("damascus_folded_steel", tmp_path / "broken", size=64)

    # Inject a hard non-seamless linear gradient into height.png
    grad = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    Image.fromarray(grad, mode="L").save(out / "height.png")

    # Corrupt ORM channel
    orm = np.asarray(Image.open(out / "orm.png")).copy()
    orm[..., 1] = 255 - orm[..., 1]
    Image.fromarray(orm, mode="RGB").save(out / "orm.png")

    report = verify_material(out)
    assert report["passed"] is False
    assert any("height wrap seam" in e for e in report["errors"])
    assert any("orm.png channels diverge" in e for e in report["errors"])
