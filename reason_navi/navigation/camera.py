"""Dependency-free camera geometry used by ObjectNav perception."""

from __future__ import annotations

import math
from typing import Dict


def pinhole_intrinsics(
    width: int, height: int, horizontal_fov_degrees: float
) -> Dict[str, float]:
    """Return pixel intrinsics for a pinhole camera with square pixels."""

    if isinstance(width, bool) or isinstance(height, bool):
        raise ValueError("camera width and height must be positive integers")
    if not isinstance(width, int) or not isinstance(height, int):
        raise ValueError("camera width and height must be positive integers")
    if width <= 0 or height <= 0:
        raise ValueError("camera width and height must be positive integers")
    try:
        hfov = float(horizontal_fov_degrees)
    except (TypeError, ValueError) as exc:
        raise ValueError("horizontal FOV must be between 0 and 180 degrees") from exc
    if not math.isfinite(hfov) or not 0.0 < hfov < 180.0:
        raise ValueError("horizontal FOV must be between 0 and 180 degrees")

    focal_length = width / (2.0 * math.tan(math.radians(hfov) / 2.0))
    return {
        "fx": focal_length,
        "fy": focal_length,
        "cx": (width - 1.0) / 2.0,
        "cy": (height - 1.0) / 2.0,
    }
