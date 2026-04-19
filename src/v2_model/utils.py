"""V2 utils — forward-import Dale primitives + circular helpers from src/utils.py.

Per v4 spec §Critical files — reuse:
  - `src/utils.py::ExcitatoryLinear`, `::InhibitoryGain`, circular helpers.

Keeping a thin re-export module in `src/v2_model/` so call sites inside the v2
tree read as `from src.v2_model.utils import ExcitatoryLinear` rather than
reaching back into `src/utils.py` — this makes the v2 dependency surface
explicit and lets us swap implementations later without a global grep.
"""

from __future__ import annotations

from src.utils import (
    ExcitatoryLinear,
    InhibitoryGain,
    circular_distance,
    circular_distance_abs,
    circular_gaussian,
    circular_gaussian_fwhm,
    make_circular_gaussian_kernel,
    rectified_softplus,
    shifted_softplus,
)

__all__ = [
    "ExcitatoryLinear",
    "InhibitoryGain",
    "circular_distance",
    "circular_distance_abs",
    "circular_gaussian",
    "circular_gaussian_fwhm",
    "make_circular_gaussian_kernel",
    "rectified_softplus",
    "shifted_softplus",
]
