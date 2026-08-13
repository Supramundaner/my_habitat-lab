"""Local, loopback-only visual model clients and servers.

Exports are loaded lazily so importing the client package does not require every
optional model server dependency to be installed in the caller's environment.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .detections import ObjectDetections
    from .grounding_dino import GroundingDINO, GroundingDINOClient
    from .sam import MobileSAM, MobileSAMClient
    from .yolov7 import YOLOv7, YOLOv7Client


_EXPORT_MODULES = {
    "GroundingDINOClient": ".grounding_dino",
    "GroundingDINO": ".grounding_dino",
    "MobileSAMClient": ".sam",
    "MobileSAM": ".sam",
    "ObjectDetections": ".detections",
    "YOLOv7Client": ".yolov7",
    "YOLOv7": ".yolov7",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
