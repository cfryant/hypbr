"""hypbr — Hybrid Neural + Exact Periodic Procedural PBR Material Synthesizer."""
from __future__ import annotations

from .exporters import SUPPORTED_TARGETS, export_material
from .normal_graph import PRESETS, run_graph
from .synthesize import (
    list_materials,
    synthesize_all,
    synthesize_hardops_plate,
    synthesize_material,
)
from .verify import verify_material
from .weightcode import (
    apply_depthpro_weightcode,
    spectral_filter_3x3_kernel,
    tileable_4quad_blend,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "PRESETS",
    "SUPPORTED_TARGETS",
    "apply_depthpro_weightcode",
    "export_material",
    "list_materials",
    "run_graph",
    "spectral_filter_3x3_kernel",
    "synthesize_all",
    "synthesize_hardops_plate",
    "synthesize_material",
    "tileable_4quad_blend",
    "verify_material",
]
