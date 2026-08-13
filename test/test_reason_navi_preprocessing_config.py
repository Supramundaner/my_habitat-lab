import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reason_navi.preprocessing.config import (
    ConfigError,
    load_preprocessing_config,
    normalize_topdown_resolution,
    resolve_llm_config,
    safe_config_dict,
    write_preprocessing_config,
)
from reason_navi.preprocessing.build_navigation_request import (
    build_navigation_request,
)


def _write_config(path: Path, config: dict) -> None:
    path.write_text(json.dumps(config), encoding="utf-8")


class PreprocessingConfigTest(unittest.TestCase):
    def test_topdown_resolution_contract_is_square(self):
        self.assertEqual(normalize_topdown_resolution(2048), (2048, 2048))
        self.assertEqual(normalize_topdown_resolution([512]), (512, 512))
        with self.assertRaisesRegex(ConfigError, "square"):
            normalize_topdown_resolution([1024, 512])

    def test_environment_key_takes_precedence_without_mutating_config(self):
        config = {
            "llm_config": {
                "api_key_env": "TEST_ARK_KEY",
                "api_key": "legacy-secret",
                "base_url": "https://example.invalid/api/v3",
                "model": "test-model",
                "max_tokens": 42,
                "max_retries": 2,
            }
        }

        settings = resolve_llm_config(
            config, environ={"TEST_ARK_KEY": "environment-secret"}
        )

        self.assertEqual(settings.api_key, "environment-secret")
        self.assertEqual(settings.model, "test-model")
        self.assertEqual(settings.max_tokens, 42)
        self.assertEqual(settings.max_retries, 2)
        self.assertFalse(settings.fallback_on_failure)
        self.assertNotIn("environment-secret", repr(settings))
        self.assertEqual(config["llm_config"]["api_key"], "legacy-secret")

    def test_legacy_key_is_runtime_compatible_but_never_serialized(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            secret = "legacy-only-secret"
            config_path = temp_path / "config.json"
            _write_config(
                config_path,
                {
                    "llm_config": {
                        "api_key": secret,
                        "model": "test-model",
                    },
                    "output": {"output_dir": "outputs"},
                },
            )

            config = load_preprocessing_config(config_path)

            self.assertNotIn("api_key", config["llm_config"])
            self.assertNotIn(secret, repr(config))
            self.assertEqual(
                resolve_llm_config(config, environ={}).api_key, secret
            )
            self.assertNotIn(secret, json.dumps(safe_config_dict(config)))

            written_path = write_preprocessing_config(
                config, temp_path / "copy.json"
            )
            self.assertNotIn(
                secret, written_path.read_text(encoding="utf-8")
            )

    def test_paths_can_be_relative_to_config_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "nested"
            config_dir.mkdir()
            config_path = config_dir / "config.json"
            _write_config(
                config_path,
                {
                    "path_resolution": {"relative_to": "config"},
                    "scene_config": {"scene_path": "scenes/example.glb"},
                    "prompts": {
                        "choose_room_prompt": "prompts/room.txt"
                    },
                    "output": {"output_dir": "outputs/run"},
                },
            )

            config = load_preprocessing_config(config_path)

            self.assertEqual(
                config["scene_config"]["scene_path"],
                str((config_dir / "scenes/example.glb").resolve()),
            )
            self.assertEqual(
                config["prompts"]["choose_room_prompt"],
                str((config_dir / "prompts/room.txt").resolve()),
            )
            self.assertEqual(
                config["output"]["output_dir"],
                str((config_dir / "outputs/run").resolve()),
            )

    def test_paths_can_be_relative_to_repo(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_repo = temp_path / "repo"
            config_dir = temp_path / "configs"
            config_dir.mkdir()
            config_path = config_dir / "config.json"
            _write_config(
                config_path,
                {
                    "path_resolution": {"relative_to": "repo"},
                    "scene_config": {"scene_path": "data/example.glb"},
                    "prompts": {
                        "choose_node_prompt": (
                            "reason_navi/preprocessing/prompts/node.txt"
                        )
                    },
                },
            )

            config = load_preprocessing_config(config_path, repo_root=fake_repo)

            self.assertEqual(
                config["scene_config"]["scene_path"],
                str((fake_repo / "data/example.glb").resolve()),
            )
            self.assertEqual(
                config["prompts"]["choose_node_prompt"],
                str(
                    (
                        fake_repo
                        / "reason_navi/preprocessing/prompts/node.txt"
                    ).resolve()
                ),
            )

    def test_unresolved_path_environment_variable_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            _write_config(
                config_path,
                {
                    "scene_config": {
                        "scene_path": "${REASON_NAVI_MISSING_ROOT}/scene.glb"
                    }
                },
            )
            with self.assertRaisesRegex(
                ConfigError, "REASON_NAVI_MISSING_ROOT"
            ):
                load_preprocessing_config(config_path)

    def test_non_standard_json_numbers_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(
                '{"graph_generation": {"pds_radius": NaN}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "non-standard JSON number"):
                load_preprocessing_config(config_path)

    def test_safe_config_removes_nested_secret_spellings(self):
        sanitized = safe_config_dict(
            {
                "headers": {"Authorization": "Bearer secret"},
                "provider": {
                    "apiKey": "secret",
                    "openai_api_key": "secret",
                    "client_secret": "secret",
                    "api_key_env": "ARK_API_KEY",
                },
            }
        )
        self.assertEqual(
            sanitized,
            {
                "headers": {},
                "provider": {"api_key_env": "ARK_API_KEY"},
            },
        )

    def test_missing_api_key_has_actionable_error(self):
        with self.assertRaisesRegex(ConfigError, "TEST_MISSING_KEY"):
            resolve_llm_config(
                {"llm_config": {"api_key_env": "TEST_MISSING_KEY"}},
                environ={},
            )

    def test_llm_fallback_must_be_explicit_boolean(self):
        with self.assertRaisesRegex(ConfigError, "fallback_on_failure"):
            resolve_llm_config(
                {
                    "llm_config": {
                        "api_key_env": "TEST_KEY",
                        "fallback_on_failure": "yes",
                    }
                },
                environ={"TEST_KEY": "secret"},
            )

    def test_checked_in_example_has_no_credentials_or_home_paths(self):
        repo_root = Path(__file__).resolve().parents[1]
        config_path = (
            repo_root
            / "reason_navi"
            / "preprocessing"
            / "config.example.json"
        )
        raw_text = config_path.read_text(encoding="utf-8")
        config = json.loads(raw_text)

        self.assertNotIn("api_key", config["llm_config"])
        self.assertEqual(config["llm_config"]["api_key_env"], "ARK_API_KEY")
        self.assertNotIn("/home/", raw_text)

        with patch.dict(
            os.environ,
            {"HABITAT_DATA_ROOT": str(repo_root / "data")},
        ):
            loaded = load_preprocessing_config(config_path)
        self.assertTrue(
            Path(loaded["prompts"]["choose_room_prompt"]).is_file()
        )
        self.assertTrue(
            Path(loaded["prompts"]["choose_node_prompt"]).is_file()
        )

    def test_action_uses_identity_rotation_and_relative_wall_mask(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "wall_mask.png").touch()
            (output_dir / "metadata.json").write_text(
                "{}", encoding="utf-8"
            )
            (output_dir / "node_selection_log.json").write_text(
                json.dumps(
                    {
                        "selected_node": {
                            "node_id": 7,
                            "world_coordinates": [1.25, -3.5],
                        }
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "scene_config": {
                    "target_coordinate": [0.0, 1.0, 2.0],
                    "rotation": None,
                    "goal_object": "chair",
                }
            }

            build_navigation_request(config, str(output_dir))
            action = json.loads(
                (output_dir / "action.json").read_text(encoding="utf-8")
            )

            self.assertEqual(action["agent_state"]["rotation"], [0, 0, 0, 1])
            self.assertEqual(action["wall_mask"], "wall_mask.png")
            self.assertEqual(action["map_metadata"], "metadata.json")


if __name__ == "__main__":
    unittest.main()
