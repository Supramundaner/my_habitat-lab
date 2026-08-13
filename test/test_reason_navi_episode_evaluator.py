from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from reason_navi.evaluation.episode import (
    EpisodeEvaluator,
    EvaluatorDependencies,
)


class EpisodeEvaluatorTest(unittest.TestCase):
    def test_pipeline_is_portable_sanitized_and_does_not_change_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            episode_path = root / "scene.json"
            episode_path.write_text(
                json.dumps(
                    {
                        "episodes": [
                            {
                                "episode_id": "7",
                                "object_category": "chair",
                                "start_position": [0, 0, 0],
                                "start_rotation": [0, 0, 0, 1],
                                "scene_id": "scene.glb",
                            }
                        ],
                        "goals_by_category": {
                            "scene.glb_chair": [
                                {
                                    "view_points": [
                                        {
                                            "agent_state": {
                                                "position": [2, 0, 0]
                                            }
                                        }
                                    ]
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            config_path = root / "eval.json"
            config_path.write_text(
                json.dumps(
                    {
                        "episode": {
                            "episode_json_path": "scene.json",
                            "episode_id": "7",
                        },
                        "scene": {
                            "scene_file": "scene.glb",
                            "robot_urdf": "robot.urdf",
                        },
                        "preprocess": {
                            "resolution": 1234,
                            "scene_config": {
                                "target_floor": None,
                                "target_coverage": 0.8,
                                "draw_coordinates": True,
                            },
                            "room_segmentation": {"min_room_area_pixels": 9},
                            "graph_generation": {"seed": 42},
                            "llm_config": {
                                "api_key": "must-not-reach-artifacts",
                                "model": "model",
                            },
                        },
                        "video_generation": {
                            "video": {
                                "fps": 8,
                                "resolution": {"width": 640, "height": 480},
                                "fpv_width": 320,
                                "map_width": 320,
                            },
                            "agent": {
                                "linear_speed": 1,
                                "angular_speed": 90,
                                "sensor_height": 1.3,
                            },
                            "OCCUPANCY_MAP": {
                                "HFOV": 90,
                                "CAMERA_HEIGHT": 1.3,
                                "MIN_H": -1.2,
                                "MAX_H": 0.2,
                            },
                            "simulation": {"enable_physics": True},
                            "custom_future_option": {"keep": True},
                        },
                        "evaluation": {"success_distance_threshold": 0.75},
                        "output_dir": "run",
                    }
                ),
                encoding="utf-8",
            )

            calls = {
                "pipeline_cwd": None,
                "metric": None,
                "navigation": None,
            }

            class Pipeline:
                def __init__(self, path: str) -> None:
                    self.path = Path(path)

                def run(self) -> bool:
                    calls["pipeline_cwd"] = Path.cwd()
                    config = json.loads(self.path.read_text(encoding="utf-8"))
                    output = Path(config["output"]["output_dir"])
                    output.mkdir(parents=True, exist_ok=True)
                    (output / "action.json").write_text(
                        json.dumps(
                            {
                                "agent_state": {
                                    "position": [0, 0, 0],
                                    "rotation": [0, 0, 0, 1],
                                },
                                "target_info": {
                                    "coordinate": [2, 0],
                                    "name": "chair",
                                },
                                "wall_mask": "wall_mask.png",
                                "map_metadata": "metadata.json",
                            }
                        ),
                        encoding="utf-8",
                    )
                    (output / "wall_mask.png").write_bytes(b"fake-png")
                    (output / "metadata.json").write_text(
                        "{}", encoding="utf-8"
                    )
                    return True

            def runner(config, request, **kwargs):
                calls["navigation"] = (config, request, kwargs)
                output = Path(config["output_dir"])
                output.mkdir(parents=True, exist_ok=True)
                report_path = output / "custom-report-name.json"
                report_path.write_text(
                    json.dumps(
                        {
                            "final_agent_state": {"position": [1, 0, 0]},
                            "execution_stats": {"total_distance": 2.0},
                        }
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(
                    video_path=output / "video.mp4", report_path=report_path
                )

            def metric_evaluator(**kwargs):
                calls["metric"] = kwargs
                return {
                    "sr": 1.0,
                    "spl": 0.5,
                    "success": True,
                    "path_length": kwargs["path_length"],
                    "success_threshold": kwargs["success_threshold"],
                }

            original_cwd = Path.cwd()
            evaluator = EpisodeEvaluator(
                str(config_path),
                dependencies=EvaluatorDependencies(
                    preprocessing_factory=Pipeline,
                    runner=runner,
                    metric_evaluator=metric_evaluator,
                ),
            )
            self.assertTrue(evaluator.run_evaluation())
            self.assertEqual(Path.cwd(), original_cwd)
            self.assertEqual(calls["pipeline_cwd"], original_cwd)

            preprocess_config = json.loads(
                (root / "run" / "preprocess_config.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(preprocess_config["resolution"], 1234)
            self.assertEqual(
                preprocess_config["scene_config"]["target_coverage"], 0.8
            )
            self.assertEqual(preprocess_config["graph_generation"]["seed"], 42)
            self.assertNotIn("api_key", preprocess_config["llm_config"])
            self.assertEqual(
                preprocess_config["llm_config"]["api_key_env"], "ARK_API_KEY"
            )

            navigation_config = calls["navigation"][0]
            self.assertEqual(
                navigation_config["custom_future_option"], {"keep": True}
            )
            self.assertNotIn("wall_mask", calls["navigation"][2])
            self.assertEqual(
                calls["navigation"][2]["request_base_dir"],
                (root / "run" / "preprocess").resolve(),
            )
            self.assertEqual(calls["metric"]["success_threshold"], 0.75)

            output_text = (root / "run" / "output.json").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("must-not-reach-artifacts", output_text)
            output = json.loads(output_text)
            self.assertEqual(
                output["evaluation_results"]["object_category"], "chair"
            )

    def test_failed_pipeline_is_reported_and_results_are_still_written(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            (root / "episodes.json").write_text(
                json.dumps(
                    {
                        "episodes": [
                            {
                                "episode_id": "1",
                                "object_category": "chair",
                                "start_position": [0, 0, 0],
                                "start_rotation": [0, 0, 0, 1],
                                "scene_id": "scene",
                            }
                        ],
                        "goals_by_category": {"chair": [{"position": [0, 0, 0]}]},
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "episode": {
                    "episode_json_path": "episodes.json",
                    "episode_id": "1",
                },
                "scene": {"scene_file": "scene", "robot_urdf": "robot"},
                "preprocess": {"llm_config": {"api_key_env": "ARK_API_KEY"}},
                "video_generation": {},
                "evaluation": {},
                "output_dir": "failed-run",
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            class FailedPipeline:
                def __init__(self, path: str) -> None:
                    pass

                def run(self) -> bool:
                    return False

            evaluator = EpisodeEvaluator(
                str(config_path),
                dependencies=EvaluatorDependencies(
                    preprocessing_factory=FailedPipeline,
                    runner=lambda *args, **kwargs: None,
                    metric_evaluator=lambda **kwargs: {},
                ),
            )
            self.assertFalse(evaluator.run_evaluation())
            output_path = root / "failed-run" / "output.json"
            self.assertTrue(output_path.is_file())
            output = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertFalse(output["preprocessing_success"])
            self.assertIn("Preprocessing pipeline failed", output["errors"])

    def _write_episode_lookup_fixture(
        self, root: Path, *, episode_scene: str, configured_scene: str, goals: dict
    ) -> Path:
        (root / "episodes.json").write_text(
            json.dumps(
                {
                    "episodes": [
                        {
                            "episode_id": "1",
                            "scene_id": episode_scene,
                            "object_category": "chair",
                            "start_position": [0, 0, 0],
                            "start_rotation": [0, 0, 0, 1],
                        }
                    ],
                    "goals_by_category": goals,
                }
            ),
            encoding="utf-8",
        )
        config_path = root / "lookup.json"
        config_path.write_text(
            json.dumps(
                {
                    "episode": {
                        "episode_json_path": "episodes.json",
                        "episode_id": "1",
                    },
                    "scene": {
                        "scene_file": configured_scene,
                        "robot_urdf": "robot.urdf",
                    },
                    "preprocess": {"llm_config": {}},
                    "video_generation": {},
                    "evaluation": {},
                    "output_dir": "output",
                }
            ),
            encoding="utf-8",
        )
        return config_path

    def test_episode_scene_must_match_configured_scene(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            config_path = self._write_episode_lookup_fixture(
                root,
                episode_scene="/dataset/a.glb",
                configured_scene="/runtime/b.glb",
                goals={"a.glb_chair": []},
            )
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                EpisodeEvaluator(str(config_path))._load_episode_data()

    def test_ambiguous_suffix_goal_key_is_not_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            config_path = self._write_episode_lookup_fixture(
                root,
                episode_scene="a.glb",
                configured_scene="a.glb",
                goals={"other.glb_chair": [], "third.glb_chair": []},
            )
            with self.assertRaisesRegex(RuntimeError, "No goals found"):
                EpisodeEvaluator(str(config_path))._load_episode_data()


if __name__ == "__main__":
    unittest.main()
