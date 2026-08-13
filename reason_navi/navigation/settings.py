"""Validated navigation tuning for the canonical controller."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"navigation.{name} must be positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"navigation.{name} must be positive") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"navigation.{name} must be positive")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"navigation.{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class NavigationSettings:
    a_star_interval: int
    a_star_epsilon: float
    a_star_max_iterations: int
    unknown_region_distance: float
    intermediate_distance: float
    detection_switch_distance: float
    final_stop_threshold: float
    min_search_radius: float
    max_search_radius: float
    forward_distance: float
    max_iterations: int
    detected_target_max_iterations: int

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "NavigationSettings":
        raw = config.get("navigation", {})
        if not isinstance(raw, Mapping):
            raise ValueError("navigation must be an object")
        return cls(
            a_star_interval=_positive_int(
                raw.get("a_star_interval", 5), "a_star_interval"
            ),
            a_star_epsilon=_positive_float(
                raw.get("a_star_epsilon", 0.01), "a_star_epsilon"
            ),
            a_star_max_iterations=_positive_int(
                raw.get("a_star_max_iterations", 20_000_000),
                "a_star_max_iterations",
            ),
            unknown_region_distance=_positive_float(
                raw.get("unknown_region_distance", 5.0),
                "unknown_region_distance",
            ),
            intermediate_distance=_positive_float(
                raw.get("waypoint_distance", 1.5), "waypoint_distance"
            ),
            detection_switch_distance=_positive_float(
                raw.get("detection_switch_distance", 1.5),
                "detection_switch_distance",
            ),
            final_stop_threshold=_positive_float(
                raw.get("destination_distance", 0.8),
                "destination_distance",
            ),
            min_search_radius=_positive_float(
                raw.get("min_search_radius", 0.2), "min_search_radius"
            ),
            max_search_radius=_positive_float(
                raw.get("max_search_radius", 10.0), "max_search_radius"
            ),
            forward_distance=_positive_float(
                raw.get("forward_distance", 0.25), "forward_distance"
            ),
            max_iterations=_positive_int(
                raw.get("max_iterations", 500), "max_iterations"
            ),
            detected_target_max_iterations=_positive_int(
                raw.get("detected_target_max_iterations", 50),
                "detected_target_max_iterations",
            ),
        )

    def __post_init__(self) -> None:
        if self.min_search_radius > self.max_search_radius:
            raise ValueError(
                "navigation.min_search_radius must not exceed max_search_radius"
            )
