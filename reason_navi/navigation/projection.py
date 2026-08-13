"""Projection contract shared by offline preprocessing and online navigation.

The two-stage pipeline creates its wall mask from an offline orthographic
render, then creates a second top-down render in the online simulator.  A
plain image resize is only valid when those renders cover the same world-space
bounds.  This module validates that invariant without importing Habitat-Sim.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence, Tuple


class ProjectionError(ValueError):
    """Raised when offline and online top-down projections are incompatible."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProjectionError(f"{name} must be an object")
    return value


def _vector2(value: Any, name: str) -> Tuple[float, float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise ProjectionError(f"{name} must contain two finite numbers")
    try:
        result = (float(value[0]), float(value[1]))
    except (TypeError, ValueError) as exc:
        raise ProjectionError(f"{name} must contain two finite numbers") from exc
    if not all(math.isfinite(item) for item in result):
        raise ProjectionError(f"{name} must contain two finite numbers")
    return result


def _positive_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProjectionError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(result) or result <= 0:
        raise ProjectionError(f"{name} must be a positive finite number")
    return result


def _scene_identity(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProjectionError(f"{name} must be a non-empty scene path")
    return value.replace("\\", "/").rstrip("/").split("/")[-1]


def _selected_floor(value: Any, name: str) -> Tuple[int, float, float, float]:
    floor = _mapping(value, name)
    try:
        index = int(floor["index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProjectionError(f"{name}.index must be an integer") from exc
    minimum = _positive_or_finite_float(floor.get("min"), f"{name}.min")
    maximum = _positive_or_finite_float(floor.get("max"), f"{name}.max")
    mean = _positive_or_finite_float(floor.get("mean"), f"{name}.mean")
    if minimum > maximum or not minimum <= mean <= maximum:
        raise ProjectionError(f"{name} must satisfy min <= mean <= max")
    return index, minimum, maximum, mean


def _positive_or_finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProjectionError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ProjectionError(f"{name} must be a finite number")
    return result


def validate_projection_compatibility(
    offline_document: Mapping[str, Any],
    runtime_metadata: Mapping[str, Any],
) -> None:
    """Require both renders to cover the same world-space rectangle.

    Resolution may differ (the historical pipeline uses 2048 and 4096 square
    renders), so comparison is performed in meters.  The tolerance allows for
    at most two source pixels of numeric/rendering drift.
    """

    offline = _mapping(
        offline_document.get("unprojected_coords"),
        "map_metadata.unprojected_coords",
    )
    runtime_bounds = _mapping(
        runtime_metadata.get("map_bounds"), "runtime_topdown.map_bounds"
    )
    offline_summary = _mapping(
        offline_document.get("topdown_metadata"),
        "map_metadata.topdown_metadata",
    )

    offline_scene = _mapping(
        offline_document.get("scene_info"), "map_metadata.scene_info"
    )
    offline_scene_id = _scene_identity(
        offline_scene.get("scene_path"), "map_metadata.scene_info.scene_path"
    )
    runtime_scene_id = _scene_identity(
        runtime_metadata.get("scene_path"), "runtime_topdown.scene_path"
    )
    if offline_scene_id != runtime_scene_id:
        raise ProjectionError(
            "offline wall-mask and online runtime refer to different scenes "
            f"({offline_scene_id!r} != {runtime_scene_id!r})"
        )

    offline_floor = _selected_floor(
        offline_summary.get("selected_floor"),
        "map_metadata.topdown_metadata.selected_floor",
    )
    runtime_floor = _selected_floor(
        runtime_metadata.get("selected_floor"),
        "runtime_topdown.selected_floor",
    )
    floor_tolerance = 0.5
    if offline_floor[0] != runtime_floor[0] or any(
        abs(expected - actual) > floor_tolerance
        for expected, actual in zip(offline_floor[1:], runtime_floor[1:])
    ):
        raise ProjectionError(
            "offline wall-mask and online runtime refer to different floors "
            f"(offline index/min/max/mean={offline_floor}, "
            f"runtime={runtime_floor})"
        )

    offline_spacing = _positive_float(
        offline_summary.get("spacing_in_meters_per_pixel"),
        "map_metadata.topdown_metadata.spacing_in_meters_per_pixel",
    )
    runtime_spacing = _positive_float(
        runtime_metadata.get("spacing"), "runtime_topdown.spacing"
    )
    tolerance = max(1e-4, 2.0 * offline_spacing, 2.0 * runtime_spacing)

    fields = ("top_left", "bottom_right", "view_range")
    mismatches = []
    for field in fields:
        expected = _vector2(
            offline.get(field), f"map_metadata.unprojected_coords.{field}"
        )
        actual = _vector2(
            runtime_bounds.get(field), f"runtime_topdown.map_bounds.{field}"
        )
        if any(
            abs(left - right) > tolerance
            for left, right in zip(expected, actual)
        ):
            mismatches.append(
                f"{field}: offline={expected}, runtime={actual}"
            )

    if mismatches:
        detail = "; ".join(mismatches)
        raise ProjectionError(
            "offline wall-mask and online top-down projections cover different "
            f"world bounds ({detail}). Keep custom_ortho_scale/target_coverage "
            "consistent instead of resizing the mask across projections."
        )


def runtime_projection_options(
    offline_document: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Derive the exact online orthographic scale from offline metadata."""

    offline = _mapping(
        offline_document.get("unprojected_coords"),
        "map_metadata.unprojected_coords",
    )
    view_range = _vector2(
        offline.get("view_range"),
        "map_metadata.unprojected_coords.view_range",
    )
    if view_range[0] <= 0 or view_range[1] <= 0:
        raise ProjectionError(
            "map_metadata.unprojected_coords.view_range must be positive"
        )
    if not math.isclose(view_range[0], view_range[1], rel_tol=1e-6, abs_tol=1e-6):
        raise ProjectionError(
            "canonical two-stage ObjectNav requires a square top-down projection"
        )
    return {
        "custom_ortho_scale": 1.0 / view_range[1],
        "expected_center": list(
            _vector2(
                offline.get("center"),
                "map_metadata.unprojected_coords.center",
            )
        ),
    }
