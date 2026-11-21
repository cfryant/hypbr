"""Example: Synthesize seamless 15-map PBR texture sets in Python."""
from pathlib import Path
from hypbr import list_materials, synthesize_material


def main() -> None:
    out_root = Path("./output/examples")
    samples = [
        "verdigris_patina_bronze",
        "damascus_folded_steel",
        "moroccan_fishscale_tile",
        "emerald_tufted_velvet",
    ]

    print("Sample built-in materials:")
    for item in list_materials()[:6]:
        print(f"  - {item['id']} ({item['category']})")

    for mat_id in samples:
        out_dir = synthesize_material(mat_id, out_root / mat_id, size=512)
        maps = sorted(p.name for p in out_dir.glob("*.png"))
        print(f"Synthesized {mat_id} -> {out_dir} ({len(maps)} maps)")


if __name__ == "__main__":
    main()
