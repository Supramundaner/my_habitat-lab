"""Pure metric helpers for ObjectNav evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional


def _artifact_distance(value: float) -> Optional[float]:
    """Return a standards-compliant JSON representation of a distance."""

    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class NavigationMetrics:
    success: bool
    spl: float
    geodesic_distance_to_target: float
    optimal_geodesic_distance: float
    path_length: float
    success_threshold: float

    @property
    def sr(self) -> float:
        return float(self.success)

    @property
    def reachable(self) -> bool:
        """Whether both geodesic queries produced finite path lengths."""

        return math.isfinite(self.optimal_geodesic_distance) and math.isfinite(
            self.geodesic_distance_to_target
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sr": self.sr,
            "spl": self.spl,
            "success": self.success,
            "reachable": self.reachable,
            "geodesic_distance_to_target": _artifact_distance(
                self.geodesic_distance_to_target
            ),
            "optimal_geodesic_distance": _artifact_distance(
                self.optimal_geodesic_distance
            ),
            "path_length": _artifact_distance(self.path_length),
            "success_threshold": self.success_threshold,
        }


def compute_navigation_metrics(
    *,
    optimal_distance: float,
    distance_to_target: float,
    path_length: float,
    success_threshold: float,
) -> NavigationMetrics:
    """Compute SR and SPL with explicit handling for invalid/unreachable paths.

    An episode with an unreachable start or final pose is not counted as a
    success.  This fixes the historical behavior that returned ``SR=SPL=1``
    when the optimal path was infinite.
    """

    values = {
        "optimal_distance": float(optimal_distance),
        "distance_to_target": float(distance_to_target),
        "path_length": float(path_length),
        "success_threshold": float(success_threshold),
    }
    if not math.isfinite(values["success_threshold"]) or values[
        "success_threshold"
    ] <= 0:
        raise ValueError("success_threshold must be a positive finite number")
    if values["path_length"] < 0 or math.isnan(values["path_length"]):
        raise ValueError("path_length must be non-negative")
    if math.isfinite(values["optimal_distance"]) and values[
        "optimal_distance"
    ] < 0:
        raise ValueError("optimal_distance must be non-negative")
    if math.isfinite(values["distance_to_target"]) and values[
        "distance_to_target"
    ] < 0:
        raise ValueError("distance_to_target must be non-negative")

    reachable = math.isfinite(values["optimal_distance"]) and math.isfinite(
        values["distance_to_target"]
    )
    success = reachable and (
        values["distance_to_target"] <= values["success_threshold"]
    )

    spl = 0.0
    if success:
        denominator = max(values["optimal_distance"], values["path_length"])
        spl = 1.0 if denominator == 0 else values["optimal_distance"] / denominator
        # Floating point and noisy path accounting should never yield SPL > 1.
        spl = max(0.0, min(1.0, spl))

    return NavigationMetrics(
        success=success,
        spl=spl,
        geodesic_distance_to_target=values["distance_to_target"],
        optimal_geodesic_distance=values["optimal_distance"],
        path_length=values["path_length"],
        success_threshold=values["success_threshold"],
    )
