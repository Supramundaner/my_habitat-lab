"""Portable JSON configuration loading for the ObjectNav pipeline.

The research scripts historically embedded workstation-specific absolute paths
in checked-in JSON.  This module gives the canonical two-stage pipeline one
predictable rule instead:

* ``${ENV_NAME}`` placeholders are expanded explicitly and missing variables
  fail with an actionable error;
* relative paths are resolved from the JSON file that contains them; and
* secrets are removed before a config is copied into generated artifacts.

The implementation deliberately depends only on the Python standard library so
it can be exercised before Habitat-Sim, CUDA, and detector services are set up.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Union,
)


_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_SECRET_FIELD_NAMES = {
    "accesstoken",
    "authtoken",
    "authorization",
    "bearertoken",
    "clientsecret",
    "idtoken",
    "password",
    "privatekey",
    "refreshtoken",
    "secret",
    "secretaccesskey",
    "secretkey",
}


class ConfigError(ValueError):
    """Raised when a runtime JSON config is malformed or not portable."""


def _expand_environment_string(
    value: str, environ: Mapping[str, str], field_name: str
) -> str:
    def replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in environ or not environ[name]:
            raise ConfigError(
                f"{field_name} requires environment variable {name}"
            )
        return environ[name]

    return _ENV_REFERENCE.sub(replace, value)


def expand_environment(
    value: Any,
    *,
    environ: Optional[Mapping[str, str]] = None,
    field_name: str = "config",
) -> Any:
    """Recursively expand ``${NAME}`` references in a JSON-compatible value."""

    runtime_environ = os.environ if environ is None else environ
    if isinstance(value, str):
        return _expand_environment_string(value, runtime_environ, field_name)
    if isinstance(value, list):
        return [
            expand_environment(
                item,
                environ=runtime_environ,
                field_name=f"{field_name}[{index}]",
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        return {
            key: expand_environment(
                item,
                environ=runtime_environ,
                field_name=f"{field_name}.{key}",
            )
            for key, item in value.items()
        }
    return value


def load_json_object(
    path: Union[os.PathLike[str], str],
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Load a JSON object and expand its environment references."""

    source = Path(path).expanduser().resolve(strict=False)

    def reject_nonstandard_number(token: str) -> None:
        raise ValueError(f"non-standard JSON number {token}")

    try:
        with source.open("r", encoding="utf-8") as handle:
            value = json.load(handle, parse_constant=reject_nonstandard_number)
    except FileNotFoundError as exc:
        raise ConfigError(f"JSON file not found: {source}") from exc
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Failed to load JSON config {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"Expected a JSON object in {source}")
    return expand_environment(value, environ=environ)


def _nested_mapping(
    value: MutableMapping[str, Any], keys: Sequence[str]
) -> Optional[MutableMapping[str, Any]]:
    current: Any = value
    for key in keys:
        if not isinstance(current, MutableMapping) or key not in current:
            return None
        current = current[key]
    return current if isinstance(current, MutableMapping) else None


def _resolve_path_field(
    config: MutableMapping[str, Any],
    field_path: Sequence[str],
    *,
    base_dir: Path,
) -> None:
    if not field_path:
        return
    parent = config if len(field_path) == 1 else _nested_mapping(
        config, field_path[:-1]
    )
    key = field_path[-1]
    if parent is None or key not in parent or parent[key] is None:
        return
    value = parent[key]
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{'.'.join(field_path)} must be a filesystem path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    parent[key] = str(path.resolve(strict=False))


def resolve_config_paths(
    config: Mapping[str, Any],
    *,
    base_dir: Path,
    fields: Iterable[Sequence[str]],
) -> Dict[str, Any]:
    """Return a copy with selected path fields made absolute."""

    resolved = copy.deepcopy(dict(config))
    for field_path in fields:
        _resolve_path_field(resolved, field_path, base_dir=base_dir)
    return resolved


def sanitize_secrets(value: Any) -> Any:
    """Deep-copy a JSON-compatible value while dropping plaintext secrets."""

    if isinstance(value, Mapping):
        return {
            key: sanitize_secrets(item)
            for key, item in value.items()
            if not _is_secret_field(str(key))
        }
    if isinstance(value, list):
        return [sanitize_secrets(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_secrets(item) for item in value]
    return copy.deepcopy(value)


def _is_secret_field(value: str) -> bool:
    """Recognize common snake/camel/header spellings of credential fields."""

    compact = re.sub(r"[^a-z0-9]+", "", value.lower())
    return compact.endswith("apikey") or compact in _SECRET_FIELD_NAMES


def load_navigation_config(
    path: Union[os.PathLike[str], str],
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Load a portable online-navigation configuration."""

    source = Path(path).expanduser().resolve(strict=False)
    config = load_json_object(source, environ=environ)
    resolved = resolve_config_paths(
        config,
        base_dir=source.parent,
        fields=(
            ("scene", "scene_file"),
            ("scene", "robot_urdf"),
            ("output_dir",),
        ),
    )
    validate_navigation_config(resolved)
    return resolved


def _require_positive_number(config: Mapping[str, Any], path: str) -> None:
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ConfigError(f"{path} is required")
        current = current[part]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise ConfigError(f"{path} must be a positive number")
    if not math.isfinite(float(current)) or float(current) <= 0:
        raise ConfigError(f"{path} must be a positive number")


def _value_at(config: Mapping[str, Any], path: str) -> Any:
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ConfigError(f"{path} is required")
        current = current[part]
    return current


def _require_positive_int(config: Mapping[str, Any], path: str) -> None:
    value = _value_at(config, path)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{path} must be a positive integer")


def _require_finite_number(config: Mapping[str, Any], path: str) -> None:
    value = _value_at(config, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be a finite number")
    if not math.isfinite(float(value)):
        raise ConfigError(f"{path} must be a finite number")


def validate_navigation_config(config: Mapping[str, Any]) -> None:
    """Validate fields required before constructing GPU/native components."""

    for path in (
        "video.fps",
        "agent.linear_speed",
        "agent.angular_speed",
        "agent.sensor_height",
        "OCCUPANCY_MAP.HFOV",
        "OCCUPANCY_MAP.CAMERA_HEIGHT",
    ):
        _require_positive_number(config, path)
    for path in (
        "video.resolution.width",
        "video.resolution.height",
        "video.fpv_width",
        "video.map_width",
    ):
        _require_positive_int(config, path)
    for path in ("OCCUPANCY_MAP.MIN_H", "OCCUPANCY_MAP.MAX_H"):
        _require_finite_number(config, path)
    for path in ("scene.scene_file", "scene.robot_urdf", "output_dir"):
        current: Any = config
        for part in path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                raise ConfigError(f"{path} is required")
            current = current[part]
        if not isinstance(current, str) or not current.strip():
            raise ConfigError(f"{path} must be a filesystem path")
    hfov = float(config["OCCUPANCY_MAP"]["HFOV"])
    if hfov >= 180:
        raise ConfigError("OCCUPANCY_MAP.HFOV must be less than 180 degrees")
    if config["OCCUPANCY_MAP"]["MIN_H"] >= config["OCCUPANCY_MAP"]["MAX_H"]:
        raise ConfigError("OCCUPANCY_MAP.MIN_H must be less than MAX_H")
    video = config["video"]
    if video["fpv_width"] + video["map_width"] != video["resolution"]["width"]:
        raise ConfigError(
            "video.resolution.width must equal video.fpv_width + video.map_width"
        )
    simulation = _value_at(config, "simulation.enable_physics")
    if not isinstance(simulation, bool):
        raise ConfigError("simulation.enable_physics must be a boolean")


def load_evaluation_config(
    path: Union[os.PathLike[str], str],
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Load and sanitize a single-episode two-stage evaluation config."""

    source = Path(path).expanduser().resolve(strict=False)
    config = load_json_object(source, environ=environ)
    resolved = resolve_config_paths(
        config,
        base_dir=source.parent,
        fields=(
            ("episode", "episode_json_path"),
            ("scene", "scene_file"),
            ("scene", "robot_urdf"),
            ("output_dir",),
            ("preprocess", "prompts", "choose_room_prompt"),
            ("preprocess", "prompts", "choose_node_prompt"),
        ),
    )
    # A legacy key may still exist in an old eval config.  Never carry it into
    # preprocess_config.json or output.json; migrate it to the environment
    # contract instead of silently leaving the generated config unusable.
    sanitized = sanitize_secrets(resolved)
    preprocess = sanitized.get("preprocess")
    if isinstance(preprocess, MutableMapping):
        llm_config = preprocess.get("llm_config")
        if isinstance(llm_config, MutableMapping):
            llm_config.setdefault("api_key_env", "ARK_API_KEY")
    return sanitized


def load_batch_evaluation_config(
    path: Union[os.PathLike[str], str],
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Load a portable batch config and resolve every task's data paths."""

    source = Path(path).expanduser().resolve(strict=False)
    config = load_json_object(source, environ=environ)
    resolved = resolve_config_paths(
        config,
        base_dir=source.parent,
        fields=(
            ("scene", "robot_urdf"),
            ("output_dir",),
            ("preprocess", "prompts", "choose_room_prompt"),
            ("preprocess", "prompts", "choose_node_prompt"),
        ),
    )
    tasks = resolved.get("evaluation_tasks")
    if not isinstance(tasks, list):
        raise ConfigError("evaluation_tasks must be a list")
    if not tasks:
        raise ConfigError("evaluation_tasks must contain at least one task")
    for index, task in enumerate(tasks):
        if not isinstance(task, MutableMapping):
            raise ConfigError(f"evaluation_tasks[{index}] must be an object")
        for key in ("episode_json_path", "scene_file"):
            if key not in task:
                raise ConfigError(f"evaluation_tasks[{index}].{key} is required")
            value = task[key]
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(
                    f"evaluation_tasks[{index}].{key} must be a filesystem path"
                )
            task_path = Path(value).expanduser()
            if not task_path.is_absolute():
                task_path = source.parent / task_path
            task[key] = str(task_path.resolve(strict=False))
        episode_ids = task.get("episode_ids")
        if not isinstance(episode_ids, list) or not episode_ids:
            raise ConfigError(
                f"evaluation_tasks[{index}].episode_ids must be a non-empty list"
            )
        for episode_index, episode_id in enumerate(episode_ids):
            if isinstance(episode_id, bool) or not isinstance(
                episode_id, (str, int)
            ):
                raise ConfigError(
                    f"evaluation_tasks[{index}].episode_ids[{episode_index}] "
                    "must be a string or integer"
                )
            normalized_id = str(episode_id)
            if not normalized_id.strip():
                raise ConfigError(
                    f"evaluation_tasks[{index}].episode_ids[{episode_index}] "
                    "must not be empty"
                )
            if (
                normalized_id in {".", ".."}
                or "/" in normalized_id
                or "\\" in normalized_id
                or Path(normalized_id).is_absolute()
            ):
                raise ConfigError(
                    f"evaluation_tasks[{index}].episode_ids[{episode_index}] "
                    "must be a safe path component"
                )
    return sanitize_secrets(resolved)
