"""Example: Inspect and customize a 47-node PBR graph preset."""
import json
from pathlib import Path
from hypbr import PRESETS


def main() -> None:
    preset_name = "v13-default"
    graph = PRESETS[preset_name]

    print(f"Preset '{preset_name}' has {len(graph['nodes'])} nodes:")
    for node in graph["nodes"]:
        print(f"  [{node['id']}] type={node['type']}")

    out_path = Path("./output/custom_graph.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(graph, indent=2))
    print(f"\nExported editable graph -> {out_path}")
    print(f"Run on any macro photo with:\n  hypbr photo path/to/surface.jpg --graph {out_path} --tileable --out ./output/pbr")


if __name__ == "__main__":
    main()
