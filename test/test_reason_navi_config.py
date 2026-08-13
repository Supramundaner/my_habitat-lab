from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from reason_navi.navigation.config import (
    ConfigError,
    load_batch_evaluation_config,
    load_evaluation_config,
    load_navigation_config,
    sanitize_secrets,
)


class PortableConfigTest(unittest.TestCase):
    def _write(self, directory: Path, name: str, value: dict) -> Path:
        path = directory / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_navigation_paths_are_environment_expanded_and_file_relative(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            path = self._write(
                directory,
                "video.json",
                {
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
                    "scene": {
                        "scene_file": "${DATA_ROOT}/scene.glb",
                        "robot_urdf": "assets/robot.urdf",
                    },
                    "output_dir": "output",
                },
            )
            config = load_navigation_config(
                path, environ={"DATA_ROOT": str(directory / "data")}
            )
        self.assertEqual(
            config["scene"]["scene_file"],
            str((directory / "data" / "scene.glb").resolve()),
        )
        self.assertEqual(
            config["scene"]["robot_urdf"],
            str((directory / "assets" / "robot.urdf").resolve()),
        )
        self.assertEqual(
            config["output_dir"], str((directory / "output").resolve())
        )

    def test_navigation_config_rejects_invalid_runtime_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            path = self._write(
                directory,
                "video.json",
                {
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
                    "scene": {
                        "scene_file": "scene.glb",
                        "robot_urdf": "robot.urdf",
                    },
                    "output_dir": "output",
                    "OCCUPANCY_MAP": {
                        "HFOV": 180,
                        "CAMERA_HEIGHT": 1.3,
                        "MIN_H": -1.2,
                        "MAX_H": 0.2,
                    },
                    "simulation": {"enable_physics": True},
                },
            )
            with self.assertRaisesRegex(ConfigError, "less than 180"):
                load_navigation_config(path)

    def test_missing_environment_variable_has_an_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            path = self._write(
                Path(raw_directory),
                "video.json",
                {"scene": {"scene_file": "${MISSING}/scene.glb"}},
            )
            with self.assertRaisesRegex(ConfigError, "MISSING"):
                load_navigation_config(path, environ={})

    def test_non_standard_json_numbers_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            path = Path(raw_directory) / "invalid.json"
            path.write_text('{"value": Infinity}', encoding="utf-8")
            from reason_navi.navigation.config import load_json_object

            with self.assertRaisesRegex(ConfigError, "non-standard JSON number"):
                load_json_object(path)

    def test_evaluation_config_drops_legacy_key_and_adds_env_contract(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            path = self._write(
                directory,
                "eval.json",
                {
                    "episode": {"episode_json_path": "episode.json"},
                    "scene": {
                        "scene_file": "scene.glb",
                        "robot_urdf": "robot.urdf",
                    },
                    "preprocess": {"llm_config": {"api_key": "do-not-copy"}},
                },
            )
            config = load_evaluation_config(path)
        self.assertNotIn("api_key", config["preprocess"]["llm_config"])
        self.assertEqual(
            config["preprocess"]["llm_config"]["api_key_env"], "ARK_API_KEY"
        )

    def test_sanitizer_is_recursive_and_non_mutating(self) -> None:
        source = {
            "nested": [
                {
                    "api_key": "secret",
                    "apiKey": "secret-2",
                    "openai_api_key": "secret-3",
                    "client_secret": "secret-4",
                    "Authorization": "Bearer secret-5",
                    "model": "m",
                    "api_key_env": "ARK_API_KEY",
                }
            ]
        }
        sanitized = sanitize_secrets(source)
        self.assertEqual(
            sanitized,
            {
                "nested": [
                    {"model": "m", "api_key_env": "ARK_API_KEY"}
                ]
            },
        )
        self.assertIn("api_key", source["nested"][0])

    def test_batch_task_paths_are_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            path = self._write(
                directory,
                "batch.json",
                {
                    "scene": {"robot_urdf": "robot.urdf"},
                    "preprocess": {
                        "prompts": {
                            "choose_room_prompt": "prompts/room.txt",
                            "choose_node_prompt": "prompts/node.txt",
                        }
                    },
                    "evaluation_tasks": [
                        {
                            "episode_json_path": "episodes.json.gz",
                            "scene_file": "scene.glb",
                            "episode_ids": ["7"],
                        }
                    ],
                },
            )
            config = load_batch_evaluation_config(path)
        task = config["evaluation_tasks"][0]
        self.assertEqual(
            task["episode_json_path"],
            str((directory / "episodes.json.gz").resolve()),
        )
        self.assertEqual(
            task["scene_file"], str((directory / "scene.glb").resolve())
        )
        self.assertEqual(
            config["preprocess"]["prompts"]["choose_room_prompt"],
            str((directory / "prompts" / "room.txt").resolve()),
        )

    def test_batch_requires_work_and_valid_episode_id_list(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            empty_path = self._write(
                directory, "empty.json", {"evaluation_tasks": []}
            )
            with self.assertRaisesRegex(ConfigError, "at least one"):
                load_batch_evaluation_config(empty_path)

            for index, episode_ids in enumerate(("12", [], ["../escape"])):
                path = self._write(
                    directory,
                    f"invalid-{index}.json",
                    {
                        "evaluation_tasks": [
                            {
                                "episode_json_path": "episodes.json.gz",
                                "scene_file": "scene.glb",
                                "episode_ids": episode_ids,
                            }
                        ]
                    },
                )
                with self.subTest(episode_ids=episode_ids):
                    with self.assertRaises(ConfigError):
                        load_batch_evaluation_config(path)


if __name__ == "__main__":
    unittest.main()
