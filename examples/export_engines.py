"""Example: Synthesize a material and export it to Unity, Godot, Unreal, Blender, Three.js, glTF, Vulkan, and MaterialX."""
from pathlib import Path

from hypbr import SUPPORTED_TARGETS, export_material, synthesize_material


def main() -> None:
    mat_id = "verdigris_patina_bronze"
    raw_dir = synthesize_material(mat_id, Path("./output/raw") / mat_id, size=512)

    for target in SUPPORTED_TARGETS:
        out = export_material(raw_dir, Path("./output/engines") / mat_id / target, target=target, name=mat_id)
        files = sorted(p.name for p in out.iterdir())
        print(f"[{target:10s}] -> {out} ({', '.join(files)})")


if __name__ == "__main__":
    main()
