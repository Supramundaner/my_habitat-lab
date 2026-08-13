from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from reason_navi.evaluation.batch import BatchEvaluator, _dataset_label


class BatchEvaluatorContractTest(unittest.TestCase):
    @staticmethod
    def _write_batch_config(root: Path, tasks: list) -> Path:
        config_path = root / "batch.json"
        config_path.write_text(
            json.dumps(
                {
                    "scene": {"robot_urdf": "robot.urdf"},
                    "preprocess": {},
                    "video_generation": {},
                    "evaluation": {},
                    "output_dir": "output",
                    "evaluation_tasks": tasks,
                }
            ),
            encoding="utf-8",
        )
        return config_path

    def test_same_scene_in_different_splits_has_distinct_output_label(self) -> None:
        seen = "/data/val_seen_synonyms/content/Dd4bFSTQ8gi.json"
        unseen = "/data/val_unseen/content/Dd4bFSTQ8gi.json"
        self.assertNotEqual(_dataset_label(seen), _dataset_label(unseen))
        self.assertIn("val_seen_synonyms__Dd4bFSTQ8gi", _dataset_label(seen))
        self.assertIn("val_unseen__Dd4bFSTQ8gi", _dataset_label(unseen))

    def test_compressed_and_plain_paths_have_stable_scene_name(self) -> None:
        plain = _dataset_label("/data/val/content/scene.json")
        compressed = _dataset_label("/data/val/content/scene.json.gz")
        self.assertTrue(plain.startswith("val__scene__"))
        self.assertTrue(compressed.startswith("val__scene__"))

    def test_dataset_label_is_stable_when_data_root_moves(self) -> None:
        laptop = "/Users/me/data/val/content/scene.json.gz"
        remote = "/efs/user/data/val/content/scene.json.gz"
        self.assertEqual(_dataset_label(laptop), _dataset_label(remote))

    def test_duplicate_episode_output_is_rejected_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            task = {
                "episode_json_path": "val/content/scene.json.gz",
                "scene_file": "scene.glb",
                "episode_ids": ["7"],
            }
            config_path = self._write_batch_config(root, [task, task])
            evaluator = BatchEvaluator(
                str(config_path),
                evaluator_factory=lambda path: self.fail(
                    "duplicate batch must fail before evaluator construction"
                ),
            )
            with self.assertRaisesRegex(ValueError, "duplicate batch episode"):
                evaluator.run_batch_evaluation()

    def test_episode_setup_failure_is_checkpointed_and_batch_continues(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            tasks = [
                {
                    "episode_json_path": "first/content/broken.json.gz",
                    "scene_file": "broken.glb",
                    "episode_ids": ["1"],
                },
                {
                    "episode_json_path": "second/content/good.json.gz",
                    "scene_file": "good.glb",
                    "episode_ids": ["2"],
                },
            ]
            config_path = self._write_batch_config(root, tasks)
            broken_key = f"{_dataset_label(str(root / tasks[0]['episode_json_path']))}/1"

            class SetupFailureEvaluator(BatchEvaluator):
                def _create_single_episode_config(
                    self, task, episode_id, output_dir
                ):
                    if episode_id == "1":
                        raise OSError("synthetic config-generation failure")
                    return super()._create_single_episode_config(
                        task, episode_id, output_dir
                    )

            def evaluator_factory(config_file: str):
                class SuccessfulEpisode:
                    def run_evaluation(self) -> bool:
                        partial_path = root / "output" / "batch_output.json"
                        partial = json.loads(partial_path.read_text(encoding="utf-8"))
                        self_test.assertIn(
                            broken_key, partial["pipeline_failed_episodes"]
                        )

                        episode_config = json.loads(
                            Path(config_file).read_text(encoding="utf-8")
                        )
                        output_dir = Path(episode_config["output_dir"])
                        (output_dir / "output.json").write_text(
                            json.dumps(
                                {
                                    "evaluation_results": {
                                        "sr": 1.0,
                                        "spl": 0.75,
                                        "success": True,
                                        "reachable": True,
                                        "geodesic_distance_to_target": 0.1,
                                        "optimal_geodesic_distance": 3.0,
                                        "path_length": 4.0,
                                        "object_category": "chair",
                                    }
                                },
                                allow_nan=False,
                            ),
                            encoding="utf-8",
                        )
                        return True

                return SuccessfulEpisode()

            self_test = self
            evaluator = SetupFailureEvaluator(
                str(config_path), evaluator_factory=evaluator_factory
            )
            self.assertFalse(evaluator.run_batch_evaluation())

            final = json.loads(
                (root / "output" / "batch_output.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(final["batch_summary"]["total_episodes_processed"], 2)
            self.assertEqual(final["batch_summary"]["pipeline_failures"], 1)
            self.assertEqual(len(final["completed_episodes"]), 1)
            self.assertIn(broken_key, final["episode_details"])

    def test_failed_strict_json_write_preserves_previous_summary(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            task = {
                "episode_json_path": "val/content/scene.json.gz",
                "scene_file": "scene.glb",
                "episode_ids": ["7"],
            }
            evaluator = BatchEvaluator(
                str(self._write_batch_config(root, [task]))
            )
            destination = root / "output" / "batch_output.json"
            destination.write_text('{"checkpoint": true}\n', encoding="utf-8")
            evaluator.episode_results = {
                "invalid": {
                    "sr": float("nan"),
                    "spl": 0.0,
                    "success": False,
                }
            }

            with self.assertRaises(ValueError):
                evaluator._save_batch_results()

            self.assertEqual(
                destination.read_text(encoding="utf-8"),
                '{"checkpoint": true}\n',
            )
            self.assertEqual(
                list(destination.parent.glob(".batch_output.json.*.tmp")), []
            )

    def test_legacy_infinite_distances_are_normalized_before_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            task = {
                "episode_json_path": "val/content/scene.json.gz",
                "scene_file": "scene.glb",
                "episode_ids": ["7"],
            }
            evaluator = BatchEvaluator(
                str(self._write_batch_config(root, [task]))
            )
            episode_dir = root / "output" / "legacy"
            episode_dir.mkdir()
            (episode_dir / "output.json").write_text(
                json.dumps(
                    {
                        "evaluation_results": {
                            "sr": 0.0,
                            "spl": 0.0,
                            "success": False,
                            "reachable": False,
                            "geodesic_distance_to_target": float("inf"),
                            "optimal_geodesic_distance": 3.0,
                            "path_length": float("inf"),
                        }
                    }
                ),
                encoding="utf-8",
            )

            result = evaluator._load_episode_results(episode_dir, "legacy/7")
            self.assertIsNotNone(result)
            self.assertIsNone(result["geodesic_distance_to_target"])
            self.assertIsNone(result["path_length"])
            destination = evaluator._save_batch_results()
            serialized = destination.read_text(encoding="utf-8")
            self.assertNotIn("Infinity", serialized)
            self.assertNotIn("NaN", serialized)
            checkpoint = json.loads(serialized)
            self.assertFalse(checkpoint["episode_details"]["legacy/7"]["reachable"])


if __name__ == "__main__":
    unittest.main()
