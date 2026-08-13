from __future__ import annotations

import unittest

from reason_navi.navigation.projection import (
    ProjectionError,
    runtime_projection_options,
    validate_projection_compatibility,
)


def offline() -> dict:
    return {
        "topdown_metadata": {
            "spacing_in_meters_per_pixel": 0.01,
            "selected_floor": {"index": 1, "min": 2, "max": 3, "mean": 2.5},
        },
        "scene_info": {"scene_path": "/offline/root/scene.basis.glb"},
        "unprojected_coords": {
            "top_left": [-5, -5],
            "bottom_right": [5, 5],
            "view_range": [10, 10],
            "center": [0, 0],
        },
    }


def runtime() -> dict:
    return {
        "scene_path": "/runtime/root/scene.basis.glb",
        "selected_floor": {"index": 1, "min": 2.1, "max": 3.1, "mean": 2.6},
        "map_bounds": {
            "top_left": [-5, -5],
            "bottom_right": [5, 5],
            "view_range": [10, 10],
        },
        "spacing": 0.005,
    }


class ProjectionContractTest(unittest.TestCase):
    def test_different_resolutions_are_allowed_for_same_world_bounds(self) -> None:
        validate_projection_compatibility(offline(), runtime())

    def test_world_bound_shift_is_rejected(self) -> None:
        online = runtime()
        online["map_bounds"]["top_left"] = [-4, -5]
        with self.assertRaisesRegex(ProjectionError, "different world bounds"):
            validate_projection_compatibility(offline(), online)

    def test_missing_projection_fields_are_rejected(self) -> None:
        with self.assertRaisesRegex(ProjectionError, "unprojected_coords"):
            validate_projection_compatibility({}, runtime())

    def test_different_scene_is_rejected(self) -> None:
        online = runtime()
        online["scene_path"] = "/runtime/root/other.basis.glb"
        with self.assertRaisesRegex(ProjectionError, "different scenes"):
            validate_projection_compatibility(offline(), online)

    def test_different_floor_is_rejected(self) -> None:
        online = runtime()
        online["selected_floor"] = {
            "index": 2,
            "min": 5,
            "max": 6,
            "mean": 5.5,
        }
        with self.assertRaisesRegex(ProjectionError, "different floors"):
            validate_projection_compatibility(offline(), online)

    def test_different_floor_clipping_extent_is_rejected(self) -> None:
        online = runtime()
        online["selected_floor"] = {
            "index": 1,
            "min": 1.0,
            "max": 4.0,
            "mean": 2.5,
        }
        with self.assertRaisesRegex(ProjectionError, "different floors"):
            validate_projection_compatibility(offline(), online)

    def test_runtime_scale_is_derived_from_offline_world_range(self) -> None:
        options = runtime_projection_options(offline())
        self.assertEqual(options["custom_ortho_scale"], 0.1)
        self.assertEqual(options["expected_center"], [0.0, 0.0])

    def test_non_square_offline_projection_is_rejected(self) -> None:
        document = offline()
        document["unprojected_coords"]["view_range"] = [10, 5]
        with self.assertRaisesRegex(ProjectionError, "square"):
            runtime_projection_options(document)


if __name__ == "__main__":
    unittest.main()
