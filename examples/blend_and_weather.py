"""Example: physical weathering (`wear`), HeightLerp material blending, QA verification, and GLB/USD export."""
from pathlib import Path

from hypbr import blend_materials, export_material, synthesize_material, verify_material


def main() -> None:
    root = Path("./output/examples_blend")

    # 1. Synthesize a weathered copper base and a plaster mortar overlay
    copper = synthesize_material("hammered_copper_dimple", root / "copper_worn", size=512, wear=0.55)
    plaster = synthesize_material("venetian_plaster_trowel", root / "plaster", size=512, wear=0.25)

    # 2. Blend plaster into crevices/valleys of the copper using physically based cavity HeightLerp
    blended = blend_materials(copper, plaster, root / "aged_copper_plaster", factor=0.42, contrast=0.16, mode="cavity")

    # 3. Verify seamlessness, normal unit-length/+Z, ORM consistency, and physical bounds
    report = verify_material(blended)
    assert report["passed"], f"QA failed: {report['errors']}"
    print("QA passed:", report["checks"]["seamless"]["height"], report["checks"]["normal"])

    # 4. Export as a single-file binary .glb sphere and OpenUSD .usda
    export_material(blended, root / "exports" / "glb", target="glb", name="aged_copper_plaster")
    export_material(blended, root / "exports" / "usd", target="usd", name="aged_copper_plaster")
    print("Exported GLB + USD to:", root / "exports")


if __name__ == "__main__":
    main()
