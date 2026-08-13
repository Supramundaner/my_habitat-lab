from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


@contextmanager
def load_controller_module():
    """Load the controller without importing its Habitat/Torch dependencies."""

    module_name = "reason_navi.navigation._controller_contract_test"
    numpy = types.ModuleType("numpy")
    numpy.ndarray = object

    stubs = {"numpy": numpy}
    for relative_name, exported_name in (
        ("simulator", "HabitatSimulator"),
        ("video", "VideoComposer"),
        ("vfh_star", "VFHStar"),
        ("object_detector", "ObjectDetector"),
    ):
        stub = types.ModuleType(f"reason_navi.navigation.{relative_name}")
        setattr(stub, exported_name, type(exported_name, (), {}))
        stubs[stub.__name__] = stub

    camera = types.ModuleType("reason_navi.navigation.camera")
    camera.pinhole_intrinsics = lambda *args, **kwargs: {}
    stubs[camera.__name__] = camera

    settings = types.ModuleType("reason_navi.navigation.settings")
    settings.NavigationSettings = type("NavigationSettings", (), {})
    stubs[settings.__name__] = settings

    utils = types.ModuleType("reason_navi.navigation.utils")
    for name in (
        "slerp",
        "quaternion_to_direction_yaw",
        "quaternion_from_euler",
        "euler_from_quaternion",
        "get_device",
        "use_mixed_precision",
        "to_numpy",
    ):
        setattr(utils, name, lambda *args, **kwargs: None)
    stubs[utils.__name__] = utils

    source_path = (
        Path(__file__).resolve().parents[1]
        / "reason_navi"
        / "navigation"
        / "controller.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            yield module
        finally:
            sys.modules.pop(module_name, None)


class _Depth:
    shape = (4, 4)


class _Simulator:
    def get_observation(self):
        return {"rgb": object(), "depth": _Depth()}


class _Detector:
    def detect_and_get_target_coords(self, *args, **kwargs):
        return None


class _OccupancyMap:
    grid_map = None


class _VFH:
    def update_target(self, target):
        self.target = target


def make_processor(module):
    processor = module.NavigationController.__new__(module.NavigationController)
    processor.simulator = _Simulator()
    processor.object_detector = _Detector()
    processor.occupancy_map = _OccupancyMap()
    processor.nav_config = SimpleNamespace(final_stop_threshold=0.8)
    processor._get_camera_params = lambda shape: {}
    processor._plan_a_star_path = lambda start, target: {"path": [target]}
    return processor


class NavigationControllerResultContractTest(unittest.TestCase):
    def test_failed_rotation_search_is_not_reported_as_target_found(self) -> None:
        with load_controller_module() as module:
            processor = make_processor(module)
            processor._calculate_path_distance_to_target = lambda *args: 0.1
            processor._execute_rotation_search = lambda *args: {
                "success": False,
                "reason": "target_not_found_in_rotation_search",
                "message": "not found",
            }

            result = processor._execute_object_detection_phase(
                [0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                "chair",
                _VFH(),
                None,
                [1.0, 1.0],
            )

            self.assertFalse(result["success"])
            self.assertTrue(result["failed"])
            self.assertEqual(
                result["reason"], "target_not_found_in_rotation_search"
            )

    def test_vfh_failure_is_propagated_from_detection_phase(self) -> None:
        with load_controller_module() as module:
            processor = make_processor(module)
            processor._calculate_path_distance_to_target = lambda *args: 2.0
            processor._execute_vfh_navigation = lambda *args: {
                "success": False,
                "failed": True,
                "reason": "no_feasible_path",
            }

            result = processor._execute_object_detection_phase(
                [0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                "chair",
                _VFH(),
                None,
                [1.0, 1.0],
            )

            self.assertFalse(result["success"])
            self.assertTrue(result["failed"])
            self.assertEqual(result["reason"], "no_feasible_path")

    def test_adjusted_prediction_is_returned_under_controller_state_key(self) -> None:
        with load_controller_module() as module:
            processor = make_processor(module)
            processor._calculate_path_distance_to_target = lambda *args: 2.0
            processor._execute_vfh_navigation = lambda *args: {
                "success": False,
                "failed": False,
                "prev_direction": 0.5,
            }

            result = processor._execute_object_detection_phase(
                [0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                "chair",
                _VFH(),
                None,
                [1.0, 1.0],
            )

            self.assertEqual(result["target_pos_2d"], [1.0, 1.0])
            self.assertNotIn("original_target_pos", result)


if __name__ == "__main__":
    unittest.main()
