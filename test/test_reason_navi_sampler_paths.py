from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from reason_navi.evaluation.sampler_paths import resolve_sampler_paths


class SamplerPathsTest(unittest.TestCase):
    def test_explicit_relative_roots_are_persisted_as_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            original = Path.cwd()
            try:
                os.chdir(root)
                paths = resolve_sampler_paths(
                    data_root="data/dataset",
                    scene_root="data/scenes",
                    robot_urdf="data/robot.urdf",
                    dataset_suffix="unused",
                    environ={},
                )
            finally:
                os.chdir(original)
        self.assertEqual(
            Path(paths.dataset.config_path), (root / "data/dataset").resolve()
        )
        self.assertEqual(
            Path(paths.scenes.config_path), (root / "data/scenes").resolve()
        )
        self.assertEqual(
            Path(paths.robot_urdf), (root / "data/robot.urdf").resolve()
        )

    def test_environment_based_paths_remain_portable(self) -> None:
        paths = resolve_sampler_paths(
            data_root=None,
            scene_root=None,
            robot_urdf=None,
            dataset_suffix="datasets/objnav",
            environ={"HABITAT_DATA_ROOT": "/runtime/data"},
        )
        self.assertEqual(
            paths.dataset.config_path,
            "${HABITAT_DATA_ROOT}/datasets/objnav",
        )
        self.assertTrue(paths.robot_urdf.startswith("${HABITAT_DATA_ROOT}"))


if __name__ == "__main__":
    unittest.main()
