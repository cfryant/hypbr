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
