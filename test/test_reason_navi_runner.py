"""Dependency-light tests for the shared navigation runner."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from reason_navi.navigation.contracts import ContractError
from reason_navi.navigation.runner import (
    NavigationDependencies,
    run_navigation,
)
from reason_navi.navigation.projection import ProjectionError


def canonical_request() -> Dict[str, Any]:
    return {
        "agent_state": {
            "position": [1, 2, 3],
            "rotation": [0, 0, 0, 1],
        },
        "target_info": {"coordinate": [4, 5], "name": "chair"},
        "wall_mask": "embedded.png",
        "map_metadata": "metadata.json",
    }


def request_with_artifacts(root: Path) -> Dict[str, Any]:
    request = canonical_request()
    mask_path = root / "embedded.png"
    mask_path.touch()
    metadata_path = root / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "topdown_metadata": {
                    "spacing_in_meters_per_pixel": 0.1,
                    "selected_floor": {
                        "index": 0,
                        "min": 0.0,
                        "max": 0.4,
                        "mean": 0.2,
                    },
                },
                "scene_info": {"scene_path": "/data/scene.basis.glb"},
                "unprojected_coords": {
                    "top_left": [0, 0],
                    "bottom_right": [10, 10],
                    "view_range": [10, 10],
                    "center": [5, 5],
                },
            }
        ),
        encoding="utf-8",
    )
    request["wall_mask"] = str(mask_path)
    request["map_metadata"] = str(metadata_path)
    return request


class FakeRuntime:
    def __init__(
        self,
        root: Path,
        *,
        navigation_error: Optional[Exception] = None,
        composer_close_error: Optional[Exception] = None,
        initialize_error: Optional[Exception] = None,
        mask_success: bool = True,
        projection_offset: float = 0.0,
    ) -> None:
        self.root = root
        self.navigation_error = navigation_error
        self.composer_close_error = composer_close_error
        self.initialize_error = initialize_error
        self.mask_success = mask_success
        self.projection_offset = projection_offset
        self.events = []
        self.written_report = None
        self.initial_state = None
        self.agent_state = None
        self.mask_path = None
        self.clock = iter(
            [datetime(2024, 1, 1), datetime(2024, 1, 1) + timedelta(seconds=2)]
        )

    def dependencies(self) -> NavigationDependencies:
        runtime = self

        class Simulator:
            def setup_scene_and_agent(self, initial_state, agent_state) -> None:
                runtime.events.append("sim.setup")
                runtime.initial_state = initial_state
                runtime.agent_state = agent_state

            def get_robot_state(self):
                runtime.events.append("sim.state")
                return {"position": (6, 7, 8), "rotation": (0, 0, 0, 1)}

            def get_topdown_metadata(self):
                offset = runtime.projection_offset
                return {
                    "scene_path": "/other-root/scene.basis.glb",
                    "selected_floor": {
                        "index": 0,
                        "min": 0.0,
                        "max": 0.4,
                        "mean": 0.2,
                    },
                    "map_bounds": {
                        "top_left": (offset, 0),
                        "bottom_right": (10 + offset, 10),
                        "view_range": (10, 10),
                    },
                    "spacing": 0.05,
                }

            def close(self) -> None:
                runtime.events.append("sim.close")

        class Composer:
            def set_occupancy_map(self, occupancy_map, config) -> None:
                runtime.events.append("composer.set_map")

            def add_frame(self) -> None:
                runtime.events.append("composer.frame")

            def save_and_close(self) -> None:
                runtime.events.append("composer.close")
                if runtime.composer_close_error:
                    raise runtime.composer_close_error

        class OccupancyMap:
            def initialize_from_wall_mask(self, path: str) -> bool:
                runtime.events.append("map.mask")
                runtime.mask_path = path
                return runtime.mask_success

        class Controller:
            def navigate(self, command_group):
                runtime.events.append("controller.navigate")
                if runtime.navigation_error:
                    raise runtime.navigation_error
                return {
                    "completed_actions": ["move_forward"],
                    "collision_action": None,
                    "target_found": True,
                }

            def get_execution_stats(self):
                runtime.events.append("controller.stats")
                return {
                    "total_frames": 16,
                    "total_duration": 2.0,
                    "total_distance": 1.5,
                }

        def initialize(config) -> None:
            runtime.events.append("gpu.init")
            if runtime.initialize_error:
                raise runtime.initialize_error

        def paths(output_dir: str):
            runtime.events.append("paths")
            return {
                "video": str(runtime.root / "video.mp4"),
                "report": str(runtime.root / "execution_report.json"),
            }

        def write_report(path: str, report: Dict[str, Any]) -> None:
            runtime.events.append("report.write")
            runtime.written_report = report

        return NavigationDependencies(
            simulator_factory=lambda config: (
                runtime.events.append("sim.factory") or Simulator()
            ),
            composer_factory=lambda simulator, config, output: (
                runtime.events.append("composer.factory") or Composer()
            ),
            occupancy_map_factory=lambda **kwargs: (
                runtime.events.append("map.factory") or OccupancyMap()
            ),
            controller_factory=lambda simulator, composer, config, occupancy_map: (
                runtime.events.append("controller.factory") or Controller()
            ),
            initialize_gpu=initialize,
            clear_gpu_cache=lambda: runtime.events.append("gpu.clear"),
            validate_config=lambda config: (
                runtime.events.append("config.validate") or True
            ),
            generate_output_paths=paths,
            write_json_report=write_report,
            now=lambda: next(runtime.clock),
        )


class NavigationRunnerTest(unittest.TestCase):
    def test_successful_run_uses_override_and_closes_every_resource(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            override = root / "override.png"
            override.touch()
            override_metadata = root / "override.json"
            override_metadata.write_text(
                (root / "metadata.json").read_text(encoding="utf-8")
                if (root / "metadata.json").exists()
                else json.dumps(
                    {
                        "topdown_metadata": {
                            "spacing_in_meters_per_pixel": 0.1,
                            "selected_floor": {
                                "index": 0,
                                "min": 0.0,
                                "max": 0.4,
                                "mean": 0.2,
                            },
                        },
                        "scene_info": {
                            "scene_path": "/data/scene.basis.glb"
                        },
                        "unprojected_coords": {
                            "top_left": [0, 0],
                            "bottom_right": [10, 10],
                            "view_range": [10, 10],
                            "center": [5, 5],
                        },
                    }
                ),
                encoding="utf-8",
            )
            runtime = FakeRuntime(root)
            result = run_navigation(
                {"output_dir": str(root), "gpu": {"enabled": False}},
                request_with_artifacts(root),
                wall_mask=str(override),
                map_metadata=str(override_metadata),
                dependencies=runtime.dependencies(),
            )

        self.assertEqual(runtime.mask_path, str(override))
        self.assertEqual(runtime.initial_state["position"], [0.0, 0.0])
        self.assertEqual(runtime.agent_state["position"], [1.0, 2.0, 3.0])
        self.assertEqual(result.report["completed_sequence"], ["move_forward"])
        self.assertTrue(result.report["target_found"])
        self.assertEqual(
            runtime.events[-3:], ["composer.close", "sim.close", "gpu.clear"]
        )

    def test_unpaired_artifact_override_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            runtime = FakeRuntime(root)
            with self.assertRaisesRegex(ContractError, "supplied together"):
                run_navigation(
                    {"output_dir": raw_directory},
                    request_with_artifacts(root),
                    wall_mask=str(root / "override.png"),
                    dependencies=runtime.dependencies(),
                )
        self.assertEqual(runtime.events, ["config.validate"])

    def test_report_never_serializes_plaintext_secret(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            runtime = FakeRuntime(root)
            run_navigation(
                {
                    "output_dir": str(root),
                    "gpu": {"enabled": False},
                    "llm": {"api_key": "do-not-copy"},
                },
                request_with_artifacts(root),
                dependencies=runtime.dependencies(),
            )
        self.assertNotIn("api_key", runtime.written_report["config"]["llm"])

    def test_controller_failure_still_closes_resources(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            runtime = FakeRuntime(
                Path(raw_directory),
                navigation_error=RuntimeError("planner failed"),
            )
            with self.assertRaisesRegex(RuntimeError, "planner failed"):
                run_navigation(
                    {"output_dir": raw_directory},
                    request_with_artifacts(Path(raw_directory)),
                    dependencies=runtime.dependencies(),
                )
        self.assertEqual(
            runtime.events[-3:], ["composer.close", "sim.close", "gpu.clear"]
        )

    def test_gpu_cleanup_runs_when_initialization_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            runtime = FakeRuntime(
                Path(raw_directory), initialize_error=RuntimeError("no cuda")
            )
            with self.assertRaisesRegex(RuntimeError, "no cuda"):
                run_navigation(
                    {"output_dir": raw_directory},
                    request_with_artifacts(Path(raw_directory)),
                    dependencies=runtime.dependencies(),
                )
        self.assertEqual(runtime.events[-2:], ["gpu.init", "gpu.clear"])

    def test_later_cleanup_runs_when_composer_close_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            runtime = FakeRuntime(
                Path(raw_directory),
                composer_close_error=RuntimeError("encoder close failed"),
            )
            with self.assertRaisesRegex(RuntimeError, "encoder close failed"):
                run_navigation(
                    {"output_dir": raw_directory},
                    request_with_artifacts(Path(raw_directory)),
                    dependencies=runtime.dependencies(),
                )
        self.assertEqual(
            runtime.events[-3:], ["composer.close", "sim.close", "gpu.clear"]
        )

    def test_legacy_action_is_rejected_before_gpu_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            runtime = FakeRuntime(Path(raw_directory))
            request = canonical_request()
            request["action"] = [{"target": "chair"}]
            with self.assertRaisesRegex(ContractError, "requires target_info"):
                run_navigation(
                    {"output_dir": raw_directory},
                    request,
                    dependencies=runtime.dependencies(),
                )
        self.assertEqual(runtime.events, ["config.validate"])

    def test_missing_wall_mask_is_rejected_before_gpu_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            request = request_with_artifacts(root)
            Path(request["wall_mask"]).unlink()
            runtime = FakeRuntime(root)
            with self.assertRaisesRegex(FileNotFoundError, "wall mask"):
                run_navigation(
                    {"output_dir": raw_directory},
                    request,
                    dependencies=runtime.dependencies(),
                )
        self.assertEqual(runtime.events, ["config.validate"])

    def test_corrupt_wall_mask_fails_after_closing_resources(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            runtime = FakeRuntime(root, mask_success=False)
            with self.assertRaisesRegex(RuntimeError, "wall mask"):
                run_navigation(
                    {"output_dir": raw_directory},
                    request_with_artifacts(root),
                    dependencies=runtime.dependencies(),
                )
        self.assertEqual(
            runtime.events[-3:], ["composer.close", "sim.close", "gpu.clear"]
        )

    def test_projection_mismatch_fails_before_map_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            runtime = FakeRuntime(root, projection_offset=1.0)
            with self.assertRaisesRegex(ProjectionError, "different world bounds"):
                run_navigation(
                    {"output_dir": raw_directory},
                    request_with_artifacts(root),
                    dependencies=runtime.dependencies(),
                )
        self.assertNotIn("map.mask", runtime.events)
        self.assertEqual(runtime.events[-2:], ["sim.close", "gpu.clear"])


if __name__ == "__main__":
    unittest.main()
