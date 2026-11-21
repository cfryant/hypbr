"""hypbr — Hybrid Neural + Exact Periodic Procedural PBR Material Synthesizer."""
from __future__ import annotations

from .normal_graph import PRESETS, run_graph
from .synthesize import (
    list_materials,
    synthesize_all,
    synthesize_hardops_plate,
    synthesize_material,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "PRESETS",
    "list_materials",
    "run_graph",
    "synthesize_all",
    "synthesize_hardops_plate",
    "synthesize_material",
]
