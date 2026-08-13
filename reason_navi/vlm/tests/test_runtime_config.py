import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reason_navi.vlm.runtime_config import (
    GROUNDING_DINO_PORT,
    MOBILE_SAM_PORT,
    YOLOV7_PORT,
    required_directory,
    required_file,
)


class RuntimeConfigTest(unittest.TestCase):
    def test_service_ports_are_canonical(self) -> None:
        self.assertEqual(GROUNDING_DINO_PORT, 12181)
        self.assertEqual(MOBILE_SAM_PORT, 12184)
        self.assertEqual(YOLOV7_PORT, 12185)

    def test_required_file_uses_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_file = Path(temp_dir) / "model.pth"
            model_file.touch()
            with patch.dict(os.environ, {"TEST_MODEL_PATH": str(model_file)}):
                resolved = required_file(
                    None,
                    env_var="TEST_MODEL_PATH",
                    cli_flag="--model",
                    description="Test model",
                )
        self.assertEqual(resolved, model_file.resolve())

    def test_explicit_directory_takes_precedence_over_environment(self) -> None:
        with tempfile.TemporaryDirectory() as explicit_dir:
            with tempfile.TemporaryDirectory() as env_dir:
                with patch.dict(os.environ, {"TEST_REPO_PATH": env_dir}):
                    resolved = required_directory(
                        explicit_dir,
                        env_var="TEST_REPO_PATH",
                        cli_flag="--root",
                        description="Test repository",
                    )
        self.assertEqual(resolved, Path(explicit_dir).resolve())

    def test_missing_configuration_is_actionable(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                RuntimeError,
                r"--weights.*TEST_WEIGHTS",
            ):
                required_file(
                    None,
                    env_var="TEST_WEIGHTS",
                    cli_flag="--weights",
                    description="Test weights",
                )

    def test_relative_model_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "absolute path"):
            required_file(
                "relative/model.pth",
                env_var="TEST_WEIGHTS",
                cli_flag="--weights",
                description="Test weights",
            )


if __name__ == "__main__":
    unittest.main()
