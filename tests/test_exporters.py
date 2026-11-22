"""Tests for Unity, Unity HDRP, Godot, Unreal, Blender, Three.js, glTF, Vulkan, and MaterialX exporters."""
from __future__ import annotations

import ast
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

from hypbr import SUPPORTED_TARGETS, export_material, synthesize_material


def test_all_engine_exporters(tmp_path: Path) -> None:
    src = synthesize_material("damascus_folded_steel", tmp_path / "src", size=64)
    src_normal = np.asarray(Image.open(src / "normal.png").convert("RGB"))
    src_rough = np.asarray(Image.open(src / "roughness.png").convert("L"))
    src_metal = np.asarray(Image.open(src / "metallic.png").convert("L"))
    src_ao = np.asarray(Image.open(src / "ao.png").convert("L"))

    for target in SUPPORTED_TARGETS:
        out = export_material(src, tmp_path / "out" / target, target=target, name="damascus")
        assert out.exists()

    # Unity URP: MetallicSmoothness RGBA (R=Metallic, A=255-Roughness)
    ms = np.asarray(Image.open(tmp_path / "out" / "unity" / "MetallicSmoothness.png"))
    assert ms.shape == (64, 64, 4)
    assert np.array_equal(ms[..., 0], src_metal)
    assert np.array_equal(ms[..., 3], (255 - src_rough).astype(np.uint8))

    # Unity HDRP: MaskMap RGBA (R=Metallic, G=AO, A=255-Roughness)
    mm = np.asarray(Image.open(tmp_path / "out" / "unity-hdrp" / "MaskMap.png"))
    assert mm.shape == (64, 64, 4)
    assert np.array_equal(mm[..., 0], src_metal)
    assert np.array_equal(mm[..., 1], src_ao)
    assert np.array_equal(mm[..., 3], (255 - src_rough).astype(np.uint8))

    # Godot 4: StandardMaterial3D .tres
    tres = (tmp_path / "out" / "godot" / "material.tres").read_text()
    assert 'type="StandardMaterial3D"' in tres
    assert "heightmap_deep_parallax = true" in tres

    # Unreal Engine 5: DirectX -Y green-channel flip
    ue_nrm = np.asarray(Image.open(tmp_path / "out" / "unreal" / "T_Normal_DX.png"))
    assert np.array_equal(ue_nrm[..., 0], src_normal[..., 0])
    assert np.array_equal(ue_nrm[..., 1], (255 - src_normal[..., 1]).astype(np.uint8))
    assert np.array_equal(ue_nrm[..., 2], src_normal[..., 2])

    # Blender 4: valid Python script
    blender_src = (tmp_path / "out" / "blender" / "import_blender.py").read_text()
    ast.parse(blender_src)
    assert "ShaderNodeBsdfPrincipled" in blender_src

    # Three.js: ES module
    three_js = (tmp_path / "out" / "threejs" / "material.js").read_text()
    assert "MeshPhysicalMaterial" in three_js
    assert "SRGBColorSpace" in three_js

    # glTF 2.0: valid JSON with PBR metallicRoughness
    gltf = json.loads((tmp_path / "out" / "gltf" / "material.gltf").read_text())
    assert gltf["asset"]["version"] == "2.0"
    assert gltf["materials"][0]["name"] == "damascus"

    # Vulkan: valid manifest + RG BC5 source normal (B=0)
    vk = json.loads((tmp_path / "out" / "vulkan" / "vulkan_manifest.json").read_text())
    assert vk["bindings"][0]["vk_format"] == "VK_FORMAT_R8G8B8A8_SRGB"
    vk_rg = np.asarray(Image.open(tmp_path / "out" / "vulkan" / "normal_rg_bc5_src.png"))
    assert np.array_equal(vk_rg[..., 0], src_normal[..., 0])
    assert np.array_equal(vk_rg[..., 1], src_normal[..., 1])
    assert int(vk_rg[..., 2].max()) == 0

    # MaterialX 1.38: valid XML
    root = ET.fromstring((tmp_path / "out" / "materialx" / "material.mtlx").read_text())
    assert root.tag == "materialx"
    assert root.attrib["version"] == "1.38"

    # Single-file binary GLB: valid glTF 2.0 header + JSON + BIN chunks
    import struct

    glb_bytes = (tmp_path / "out" / "glb" / "damascus.glb").read_bytes()
    magic, version, total_len = struct.unpack_from("<III", glb_bytes, 0)
    assert magic == 0x46546C67
    assert version == 2
    assert total_len == len(glb_bytes)
    j_len, j_type = struct.unpack_from("<II", glb_bytes, 12)
    assert j_type == 0x4E4F534A
    glb_doc = json.loads(glb_bytes[20 : 20 + j_len].decode("utf-8"))
    assert glb_doc["materials"][0]["name"] == "damascus"
    assert len(glb_doc["images"]) == 3

    # OpenUSD ASCII (.usda): UsdPreviewSurface with proper normal scale/bias
    usda = (tmp_path / "out" / "usd" / "material.usda").read_text()
    assert "#usda 1.0" in usda
    assert "UsdPreviewSurface" in usda
    assert "float4 inputs:scale = (2, 2, 2, 1)" in usda



def test_emissive_exporters_and_strength(tmp_path: Path) -> None:
    import struct

    # Non-emissive source must have no emissive slot in any target
    opaque_src = synthesize_material("damascus_folded_steel", tmp_path / "opaque_src", size=64)
    g_non = export_material(opaque_src, tmp_path / "out_non_gltf", target="gltf", name="damascus")
    doc_non = json.loads((g_non / "material.gltf").read_text())
    assert "emissiveTexture" not in doc_non["materials"][0]
    assert "emissiveFactor" not in doc_non["materials"][0]
    assert not (g_non / "emissive.png").exists()
    assert "emissiveMap" not in (export_material(opaque_src, tmp_path / "out_non_three", target="threejs", name="damascus") / "material.js").read_text()

    # Emissive source across all targets that handle opacity/emissive
    em_src = synthesize_material("scifi_panel_emissive", tmp_path / "em_src", size=64)
    assert (em_src / "emissive.png").exists()
    assert not (em_src / "opacity.png").exists()

    for target in SUPPORTED_TARGETS:
        out = export_material(em_src, tmp_path / "em_out" / target, target=target, name="scifi", emissive_strength=2.5)
        assert out.exists()

    # Unity URP + HDRP
    u_urp = json.loads((tmp_path / "em_out" / "unity" / "unity_material.json").read_text())
    assert u_urp["emission"] is True
    assert u_urp["emissiveStrength"] == 2.5
    assert "EmissionMap" in u_urp["textures"]
    assert (tmp_path / "em_out" / "unity" / "EmissionMap.png").exists()
    u_hdrp = json.loads((tmp_path / "em_out" / "unity-hdrp" / "unity_material.json").read_text())
    assert u_hdrp["emission"] is True
    assert (tmp_path / "em_out" / "unity-hdrp" / "EmissionMap.png").exists()

    # Godot
    tres = (tmp_path / "em_out" / "godot" / "material.tres").read_text()
    assert "emission_enabled = true" in tres
    assert "emission_energy_multiplier = 2.5000" in tres
    assert (tmp_path / "em_out" / "godot" / "emissive.png").exists()

    # Unreal
    ue = json.loads((tmp_path / "em_out" / "unreal" / "ue5_material.json").read_text())
    assert ue["emissiveStrength"] == 2.5
    assert "EmissiveColor" in ue["textures"]
    assert (tmp_path / "em_out" / "unreal" / "T_Emissive.png").exists()

    # Blender
    bl_py = (tmp_path / "em_out" / "blender" / "import_blender.py").read_text()
    ast.parse(bl_py)
    assert "Emission Color" in bl_py
    assert "2.5000" in bl_py
    assert (tmp_path / "em_out" / "blender" / "emissive.png").exists()

    # Three.js
    three_js = (tmp_path / "em_out" / "threejs" / "material.js").read_text()
    assert "emissiveMap:" in three_js
    assert "emissiveIntensity: 2.5000" in three_js
    assert (tmp_path / "em_out" / "threejs" / "emissive.png").exists()

    # glTF (strength=1.0 omits KHR_materials_emissive_strength; strength=2.5 includes it)
    g1 = export_material(em_src, tmp_path / "em_gltf_1", target="gltf", name="scifi", emissive_strength=1.0)
    doc1 = json.loads((g1 / "material.gltf").read_text())
    assert "emissiveTexture" in doc1["materials"][0]
    assert doc1["materials"][0]["emissiveFactor"] == [1.0, 1.0, 1.0]
    assert "extensionsUsed" not in doc1

    doc2 = json.loads((tmp_path / "em_out" / "gltf" / "material.gltf").read_text())
    assert "KHR_materials_emissive_strength" in doc2["extensionsUsed"]
    assert doc2["materials"][0]["extensions"]["KHR_materials_emissive_strength"]["emissiveStrength"] == 2.5

    # GLB
    glb_bytes = (tmp_path / "em_out" / "glb" / "scifi.glb").read_bytes()
    j_len, _ = struct.unpack_from("<II", glb_bytes, 12)
    glb_doc = json.loads(glb_bytes[20 : 20 + j_len].decode("utf-8"))
    assert len(glb_doc["images"]) == 4
    assert "emissiveTexture" in glb_doc["materials"][0]
    assert "KHR_materials_emissive_strength" in glb_doc["extensionsUsed"]
    assert glb_doc["materials"][0]["extensions"]["KHR_materials_emissive_strength"]["emissiveStrength"] == 2.5

    # MaterialX
    mtlx_txt = (tmp_path / "em_out" / "materialx" / "material.mtlx").read_text()
    ET.fromstring(mtlx_txt)
    assert "emission_color" in mtlx_txt
    assert (tmp_path / "em_out" / "materialx" / "emissive.png").exists()

    # USD
    usda = (tmp_path / "em_out" / "usd" / "material.usda").read_text()
    assert "inputs:emissiveColor.connect" in usda
    assert "TexEmissive" in usda
    assert (tmp_path / "em_out" / "usd" / "emissive.png").exists()
