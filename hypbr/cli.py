"""Command-line interface for hypbr."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import normal_graph as ng
from .synthesize import list_materials, synthesize_all, synthesize_hardops_plate, synthesize_material


def cmd_list(_args: argparse.Namespace) -> int:
    for item in list_materials():
        print(f"{item['id']:30s}  [{item['category']:8s}]  {item['description']}")
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    t0 = time.time()
    if args.material == "all":
        paths = synthesize_all(args.out, size=args.size)
        print(f"Synthesized {len(paths)} seamless 15-map PBR sets in {time.time() - t0:.1f}s -> {args.out}")
    elif args.material == "hardops_plate":
        out = synthesize_hardops_plate(args.out / "hardops_plate", size=args.size)
        print(f"Synthesized hardops_plate in {time.time() - t0:.2f}s -> {out}")
    else:
        out = synthesize_material(args.material, args.out / args.material, size=args.size, seed=args.seed)
        print(f"Synthesized {args.material} (15 maps, {args.size}x{args.size}) in {time.time() - t0:.2f}s -> {out}")
    return 0


def cmd_dump_preset(args: argparse.Namespace) -> int:
    if args.preset not in ng.PRESETS:
        print(f"Unknown preset '{args.preset}'. Available: {', '.join(sorted(ng.PRESETS))}", file=sys.stderr)
        return 1
    print(json.dumps(ng.PRESETS[args.preset], indent=2))
    return 0


def cmd_photo(args: argparse.Namespace) -> int:
    argv = [str(args.image), "--out", str(args.out), "--preset", args.preset]
    if args.graph:
        argv += ["--graph", str(args.graph)]
    if args.tileable:
        argv.append("--tileable")
    if args.dx:
        argv.append("--dx")
    old_argv = sys.argv
    try:
        sys.argv = ["hypbr-photo", *argv]
        ng.main()
    finally:
        sys.argv = old_argv
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hypbr",
        description="Hybrid Neural + Exact Periodic Procedural PBR Material Synthesizer (15 calibrated PBR maps).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List built-in procedural materials")
    p_list.set_defaults(func=cmd_list)

    p_synth = sub.add_parser("synth", help="Synthesize seamless procedural 15-map PBR materials")
    p_synth.add_argument("material", help="Material ID (see `hypbr list`), `hardops_plate`, or `all`")
    p_synth.add_argument("--out", type=Path, default=Path("./output/pbr"), help="Output directory")
    p_synth.add_argument("--size", type=int, default=512, help="Square texture resolution (default: 512)")
    p_synth.add_argument("--seed", type=int, default=None, help="Optional deterministic seed override")
    p_synth.set_defaults(func=cmd_synth)

    p_dump = sub.add_parser("dump-preset", help="Dump a built-in 47-node PBR graph preset as JSON")
    p_dump.add_argument("preset", nargs="?", default="v13-default", help="Preset name (default: v13-default)")
    p_dump.set_defaults(func=cmd_dump_preset)

    p_photo = sub.add_parser("photo", help="Convert a macro photograph to 15 PBR maps via Depth Pro + DeepBump + 47-node graph")
    p_photo.add_argument("image", type=Path, help="Path to input photograph")
    p_photo.add_argument("--out", type=Path, default=Path("./output/photo_pbr"), help="Output directory")
    p_photo.add_argument("--preset", default="v13-default", help="Graph preset name")
    p_photo.add_argument("--graph", type=Path, default=None, help="Optional custom JSON graph file")
    p_photo.add_argument("--tileable", action="store_true", help="Enable circular-padded seamless evaluation")
    p_photo.add_argument("--dx", action="store_true", help="Output DirectX (-Y) normals instead of OpenGL (+Y)")
    p_photo.set_defaults(func=cmd_photo)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
