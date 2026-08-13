"""Lightweight checks for the shared LLM retry configuration.

The old script claimed that exhausted requests always selected the first room
or node and wrote into a fixed directory under ``/tmp``. That behavior is no
longer canonical: fallback must be explicitly enabled. Network retry behavior
is exercised during integration runs; this module verifies the dependency-free
configuration boundary and is safe to run from any working directory.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reason_navi.preprocessing.config import (
    load_preprocessing_config,
    resolve_llm_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class RetryConfigurationTest(unittest.TestCase):
    def _resolve(self, **overrides):
        llm_config = {
            "api_key_env": "TEST_ARK_KEY",
            "max_tokens": 1000,
            "max_retries": 3,
            **overrides,
        }
        return resolve_llm_config(
            {"llm_config": llm_config},
            environ={"TEST_ARK_KEY": "x"},
        )

    def test_retry_count_is_loaded(self):
        self.assertEqual(self._resolve(max_retries=5).max_retries, 5)

    def test_fallback_is_disabled_by_default(self):
        self.assertFalse(self._resolve().fallback_on_failure)

    def test_fallback_requires_explicit_opt_in(self):
        self.assertTrue(
            self._resolve(fallback_on_failure=True).fallback_on_failure
        )

    def test_checked_in_example_is_root_safe(self):
        with patch.dict(
            os.environ,
            {"HABITAT_DATA_ROOT": str(REPO_ROOT / "data")},
        ):
            config = load_preprocessing_config(
                REPO_ROOT
                / "reason_navi"
                / "preprocessing"
                / "config.example.json"
            )
        settings = resolve_llm_config(
            config,
            environ={config["llm_config"]["api_key_env"]: "x"},
        )

        self.assertEqual(settings.max_retries, 3)
        self.assertFalse(settings.fallback_on_failure)

    def test_temporary_config_does_not_depend_on_process_cwd(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temp_root = Path(temporary_directory)
            config_dir = temp_root / "nested" / "configs"
            config_dir.mkdir(parents=True)
            config_path = config_dir / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "path_resolution": {"relative_to": "config"},
                        "llm_config": {
                            "api_key_env": "TEST_ARK_KEY",
                            "max_retries": 2,
                            "fallback_on_failure": False,
                        },
                        "output": {"output_dir": "artifacts"},
                    }
                ),
                encoding="utf-8",
            )

            config = load_preprocessing_config(config_path)

            self.assertEqual(
                Path(config["output"]["output_dir"]),
                (config_dir / "artifacts").resolve(),
            )


if __name__ == "__main__":
    unittest.main()
