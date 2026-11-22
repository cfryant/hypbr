"""Smoke tests for hypbr procedural synthesis and node graph registry."""
from pathlib import Path

from PIL import Image

from hypbr import PRESETS, list_materials, synthesize_material

EXPECTED_MAPS = {
    "albedo_delit.png",
    "anisotropy.png",
    "ao.png",
    "bent_normal.png",
    "cavity.png",
    "curvature.png",
    "edge_wear.png",
    "height.png",
    "height16.png",
    "metallic.png",
    "normal.png",
    "orm.png",
    "preview.png",
    "roughness.png",
    "specular.png",
    "thickness.png",
}


def test_catalog_and_presets_present() -> None:
    mats = list_materials()
    assert len(mats) > 0
    assert "v13-default" in PRESETS


def test_synthesize_single_material_seamless(tmp_path: Path) -> None:
    out_dir = synthesize_material("damascus_folded_steel", tmp_path / "damascus", size=128)
    produced = {p.name for p in out_dir.glob("*.png")}
    assert EXPECTED_MAPS.issubset(produced)
    assert "opacity.png" not in produced

    h16 = Image.open(out_dir / "height16.png")
    assert h16.size == (128, 128)
    assert h16.mode in ("I;16", "I")

    orm = Image.open(out_dir / "orm.png")
    assert orm.size == (128, 128)
    assert orm.mode == "RGB"


def test_gpu_cpu_parity_hbao_and_blur() -> None:
    import numpy as np
    import torch
    from hypbr import depthpro_pbr as dp
    from hypbr import normal_graph as ng

    h = np.linspace(0.0, 1.0, 512 * 512, dtype=np.float32).reshape(512, 512)
    h = np.sin(h * 12.0).astype(np.float32)
    g_blur = dp.gaussian_blur_np(h, sigma=3.0, circular=True)
    g_hbao = ng.node_horizon_bent_normal_ao({"circular": True}, {"height": h})

    old_dev = dp.DEVICE
    try:
        dp.DEVICE = torch.device("cpu")
        c_blur = dp.gaussian_blur_np(h, sigma=3.0, circular=True)
        c_hbao = ng.node_horizon_bent_normal_ao({"circular": True}, {"height": h})
    finally:
        dp.DEVICE = old_dev

    assert np.allclose(g_blur, c_blur, atol=1e-4)
    assert np.allclose(g_hbao["ao"], c_hbao["ao"], atol=1e-4)
    assert np.allclose(g_hbao["bent_normal"], c_hbao["bent_normal"], atol=1e-4)


def test_gpu_cpu_parity_periodic_fbm_and_preview() -> None:
    import numpy as np
    import torch
    from hypbr import depthpro_pbr as dp
    from hypbr import procedural as gd
    from hypbr.synthesize import render_studio_preview

    g_fbm = gd._periodic_fbm(512, 512, 1234, (4, 8, 16, 32))
    alb = np.full((512, 512, 3), 0.55, dtype=np.float32)
    N = np.zeros((512, 512, 3), dtype=np.float32)
    N[..., 2] = 1.0
    rgh = np.full((512, 512), 0.32, dtype=np.float32)
    met = np.full((512, 512), 0.85, dtype=np.float32)
    ao = np.full((512, 512), 0.90, dtype=np.float32)
    g_prev = render_studio_preview(alb, N, rgh, met, ao)

    old_dev = dp.DEVICE
    try:
        dp.DEVICE = torch.device("cpu")
        c_fbm = gd._periodic_fbm(512, 512, 1234, (4, 8, 16, 32))
        c_prev = render_studio_preview(alb, N, rgh, met, ao)
    finally:
        dp.DEVICE = old_dev

    assert np.allclose(g_fbm, c_fbm, atol=1e-3)
    assert int(np.max(np.abs(g_prev.astype(int) - c_prev.astype(int)))) <= 1


def test_transparency_modes_auto_on_off(tmp_path: Path) -> None:
    import json
    from hypbr import blend_materials, export_material

    # 1. auto on transparent material (chainlink) -> writes opacity.png + MASK glTF
    fence_auto = synthesize_material("chainlink_fence_galvanized", tmp_path / "fence_auto", size=128, transparency="auto")
    op = Image.open(fence_auto / "opacity.png")
    assert op.mode == "L"
    assert op.getextrema()[0] < 10 and op.getextrema()[1] > 245
    g_auto = export_material(fence_auto, tmp_path / "gltf_fence_auto", target="gltf", transparency="auto")
    doc_auto = json.loads((g_auto / "material.gltf").read_text())
    assert doc_auto["materials"][0]["alphaMode"] == "MASK"
    assert doc_auto["materials"][0]["doubleSided"] is True
    assert Image.open(g_auto / "baseColor.png").mode == "RGBA"

    # 2. off on transparent material (chainlink) -> no opacity.png + OPAQUE RGB glTF
    fence_off = synthesize_material("chainlink_fence_galvanized", tmp_path / "fence_off", size=128, transparency="off")
    assert not (fence_off / "opacity.png").exists()
    g_off = export_material(fence_auto, tmp_path / "gltf_fence_off", target="gltf", transparency="off")
    doc_off = json.loads((g_off / "material.gltf").read_text())
    assert "alphaMode" not in doc_off["materials"][0]
    assert "doubleSided" not in doc_off["materials"][0]
    assert Image.open(g_off / "baseColor.png").mode == "RGB"
    assert not (g_off / "opacity.png").exists()

    # 3. auto vs on on opaque material (bronze)
    br_auto = synthesize_material("patinated_verdigris_bronze", tmp_path / "br_auto", size=128, transparency="auto")
    assert not (br_auto / "opacity.png").exists()
    br_on = synthesize_material("patinated_verdigris_bronze", tmp_path / "br_on", size=128, transparency="on")
    assert (br_on / "opacity.png").exists()
    # Even if all-white opacity.png exists, auto export keeps glTF OPAQUE (no >0.5% holes)
    g_br_auto = export_material(br_on, tmp_path / "gltf_br_auto", target="gltf", transparency="auto")
    doc_br_auto = json.loads((g_br_auto / "material.gltf").read_text())
    assert "alphaMode" not in doc_br_auto["materials"][0]
    assert Image.open(g_br_auto / "baseColor.png").mode == "RGB"
    # Forced transparency="on" export sets MASK
    g_br_on = export_material(br_auto, tmp_path / "gltf_br_on", target="gltf", transparency="on")
    doc_br_on = json.loads((g_br_on / "material.gltf").read_text())
    assert doc_br_on["materials"][0]["alphaMode"] == "MASK"

    # 4. blend in auto mode: opaque + opaque -> no opacity.png; fence + opaque -> opacity.png; off -> no opacity.png
    bl_opaque = blend_materials(br_auto, br_on, tmp_path / "bl_opaque", transparency="auto")
    assert not (bl_opaque / "opacity.png").exists()
    bl_fence = blend_materials(fence_auto, br_auto, tmp_path / "bl_fence", factor=0.3, transparency="auto")
    assert (bl_fence / "opacity.png").exists()
    bl_fence_off = blend_materials(fence_auto, br_auto, tmp_path / "bl_fence_off", factor=0.3, transparency="off")
    assert not (bl_fence_off / "opacity.png").exists()



def test_emissive_modes_auto_on_off_and_no_opacity(tmp_path: Path) -> None:
    import json
    import numpy as np
    from hypbr import blend_materials, export_material
    from hypbr import procedural as gd

    # 1. All 6 new emissive materials emit emissive.png in auto mode and NEVER emit opacity.png
    for mid in gd.EMISSIVE_MATERIALS:
        d = synthesize_material(mid, tmp_path / f"em_{mid}", size=128, emissive="auto", transparency="auto")
        assert (d / "emissive.png").exists(), f"{mid} missing emissive.png"
        assert not (d / "opacity.png").exists(), f"{mid} unexpectedly emitted opacity.png"
        em_arr = np.asarray(Image.open(d / "emissive.png").convert("RGB"), dtype=np.uint8)
        assert em_arr.max() > 100, f"{mid} emissive too dim"

    # 2. off on emissive material -> no emissive.png and no emissive slot in glTF
    lava_auto = tmp_path / "em_cooling_lava_cracks"
    lava_off = synthesize_material("cooling_lava_cracks", tmp_path / "lava_off", size=128, emissive="off")
    assert not (lava_off / "emissive.png").exists()
    assert not (lava_off / "opacity.png").exists()
    g_off = export_material(lava_auto, tmp_path / "gltf_lava_off", target="gltf", emissive="off")
    doc_off = json.loads((g_off / "material.gltf").read_text())
    assert "emissiveTexture" not in doc_off["materials"][0]
    assert "emissiveFactor" not in doc_off["materials"][0]
    assert not (g_off / "emissive.png").exists()

    # 3. auto vs on on opaque non-emissive material (bronze)
    br_auto = synthesize_material("patinated_verdigris_bronze", tmp_path / "br_auto", size=128, emissive="auto")
    assert not (br_auto / "emissive.png").exists()
    br_on = synthesize_material("patinated_verdigris_bronze", tmp_path / "br_on", size=128, emissive="on")
    assert (br_on / "emissive.png").exists()
    assert np.asarray(Image.open(br_on / "emissive.png")).max() == 0
    # Even if black emissive.png exists, auto export keeps glTF non-emissive
    g_br_auto = export_material(br_on, tmp_path / "gltf_br_auto", target="gltf", emissive="auto")
    doc_br_auto = json.loads((g_br_auto / "material.gltf").read_text())
    assert "emissiveTexture" not in doc_br_auto["materials"][0]
    assert not (g_br_auto / "emissive.png").exists()
    # Forced emissive="on" export writes black emissive.png + emissiveTexture
    g_br_on = export_material(br_auto, tmp_path / "gltf_br_on", target="gltf", emissive="on")
    doc_br_on = json.loads((g_br_on / "material.gltf").read_text())
    assert "emissiveTexture" in doc_br_on["materials"][0]
    assert (g_br_on / "emissive.png").exists()
    assert np.asarray(Image.open(g_br_on / "emissive.png")).max() == 0

    # 4. blend in auto mode: opaque + opaque -> no emissive.png; lava + opaque -> emissive.png; off -> no emissive.png
    bl_plain = blend_materials(br_auto, br_on, tmp_path / "bl_plain", emissive="auto")
    assert not (bl_plain / "emissive.png").exists()
    bl_lava = blend_materials(lava_auto, br_auto, tmp_path / "bl_lava", factor=0.35, emissive="auto")
    assert (bl_lava / "emissive.png").exists()
    assert not (bl_lava / "opacity.png").exists()
    bl_lava_off = blend_materials(lava_auto, br_auto, tmp_path / "bl_lava_off", factor=0.35, emissive="off")
    assert not (bl_lava_off / "emissive.png").exists()
