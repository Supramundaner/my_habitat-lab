"""Configuration loading for the canonical two-stage preprocessing pipeline.

The JSON configuration is intentionally kept safe to serialize. API keys are
resolved only at runtime and never inserted into the configuration mapping.
"""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple, Union


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_API_KEY_ENV = "ARK_API_KEY"
DEFAULT_MODEL = "seed-1-6-250615"
_ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENV_REFERENCE_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(ValueError):
    """Raised when a preprocessing configuration is invalid."""


class PreprocessingConfig(dict):
    """A dict-compatible runtime config with non-serializable metadata.

    A legacy plaintext key is retained only in this object's private runtime
    state. It is deliberately absent from the dict, so ``json.dump(config)``
    cannot copy it into generated artifacts.
    """

    def __init__(
        self,
        data: Mapping[str, Any],
        *,
        source_path: Path,
        legacy_api_key: Optional[str] = None,
    ) -> None:
        super().__init__(data)
        self._source_path = source_path
        self._legacy_api_key = legacy_api_key

    @property
    def source_path(self) -> Path:
        """Absolute path of the JSON file used to create this config."""

        return self._source_path


@dataclass(frozen=True)
class LLMSettings:
    """Validated runtime settings for both LLM selection stages."""

    api_key: str = field(repr=False)
    base_url: Optional[str]
    model: str
    max_tokens: int
    max_retries: int
    api_key_env: str
    fallback_on_failure: bool


def _required_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"llm_config.{field_name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"llm_config.{field_name} must be a positive integer"
        ) from exc
    if parsed <= 0:
        raise ConfigError(f"llm_config.{field_name} must be a positive integer")
    return parsed


def normalize_topdown_resolution(value: Any) -> Tuple[int, int]:
    """Validate the square projection required by the online map contract."""

    if isinstance(value, bool):
        raise ConfigError("resolution must be a positive integer or square pair")
    if isinstance(value, int):
        dimensions = (value, value)
    elif isinstance(value, (list, tuple)) and len(value) in (1, 2):
        raw_dimensions = (value[0], value[0]) if len(value) == 1 else value
        if any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in raw_dimensions
        ):
            raise ConfigError(
                "resolution must contain positive integer dimensions"
            )
        dimensions = (int(raw_dimensions[0]), int(raw_dimensions[1]))
    else:
        raise ConfigError("resolution must be a positive integer or square pair")

    if dimensions[0] <= 0 or dimensions[1] <= 0:
        raise ConfigError("resolution dimensions must be positive")
    if dimensions[0] != dimensions[1]:
        raise ConfigError(
            "canonical two-stage ObjectNav requires a square top-down resolution"
        )
    return dimensions


def resolve_llm_config(
    config: Mapping[str, Any],
    environ: Optional[Mapping[str, str]] = None,
) -> LLMSettings:
    """Resolve and validate shared LLM settings without mutating ``config``.

    ``llm_config.api_key_env`` selects the environment variable and defaults
    to ``ARK_API_KEY``. An environment value takes precedence over the legacy
    plaintext ``llm_config.api_key`` field. Configurations loaded through
    :func:`load_preprocessing_config` keep a legacy key only in private runtime
    state so it remains compatible but cannot be serialized accidentally.
    """

    raw_settings = config.get("llm_config")
    if not isinstance(raw_settings, Mapping):
        raise ConfigError("llm_config must be a JSON object")

    api_key_env = raw_settings.get("api_key_env", DEFAULT_API_KEY_ENV)
    if not isinstance(api_key_env, str) or not _ENV_NAME_PATTERN.fullmatch(
        api_key_env
    ):
        raise ConfigError(
            "llm_config.api_key_env must be a valid environment variable name"
        )

    runtime_environ = os.environ if environ is None else environ
    api_key = runtime_environ.get(api_key_env, "").strip()
    if not api_key:
        legacy_key = getattr(config, "_legacy_api_key", None)
        if legacy_key is None:
            legacy_key = raw_settings.get("api_key")
        if isinstance(legacy_key, str):
            api_key = legacy_key.strip()

    if not api_key:
        raise ConfigError(
            f"LLM API key is missing; set environment variable {api_key_env}"
        )

    base_url = raw_settings.get("base_url")
    if base_url is not None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ConfigError("llm_config.base_url must be a non-empty string")
        base_url = base_url.strip()

    model = raw_settings.get("model", DEFAULT_MODEL)
    if not isinstance(model, str) or not model.strip():
        raise ConfigError("llm_config.model must be a non-empty string")

    fallback_on_failure = raw_settings.get("fallback_on_failure", False)
    if not isinstance(fallback_on_failure, bool):
        raise ConfigError("llm_config.fallback_on_failure must be a boolean")

    return LLMSettings(
        api_key=api_key,
        base_url=base_url,
        model=model.strip(),
        max_tokens=_required_positive_int(
            raw_settings.get("max_tokens", 1000), "max_tokens"
        ),
        max_retries=_required_positive_int(
            raw_settings.get("max_retries", 3), "max_retries"
        ),
        api_key_env=api_key_env,
        fallback_on_failure=fallback_on_failure,
    )


def _resolve_path(
    value: Any,
    *,
    base_dir: Path,
) -> Any:
    if not isinstance(value, str) or not value.strip():
        return value

    missing = [
        name
        for name in _ENV_REFERENCE_PATTERN.findall(value)
        if not os.environ.get(name)
    ]
    if missing:
        raise ConfigError(
            "path requires environment variable(s): " + ", ".join(missing)
        )
    expanded_value = os.path.expandvars(value)
    if "$" in expanded_value:
        raise ConfigError(f"path contains an unresolved environment variable: {value}")
    expanded = Path(expanded_value).expanduser()
    if not expanded.is_absolute():
        expanded = base_dir / expanded
    return str(expanded.resolve(strict=False))


def _resolve_known_paths(
    config: MutableMapping[str, Any],
    *,
    config_path: Path,
    repo_root: Path,
) -> None:
    path_options = config.get("path_resolution", {})
    if path_options is None:
        path_options = {}
    if not isinstance(path_options, Mapping):
        raise ConfigError("path_resolution must be a JSON object")

    relative_to = path_options.get("relative_to", "config")
    if relative_to == "config":
        base_dir = config_path.parent
    elif relative_to == "repo":
        base_dir = repo_root
    else:
        raise ConfigError(
            "path_resolution.relative_to must be either 'config' or 'repo'"
        )

    scene_config = config.get("scene_config")
    if isinstance(scene_config, MutableMapping) and "scene_path" in scene_config:
        scene_config["scene_path"] = _resolve_path(
            scene_config["scene_path"], base_dir=base_dir
        )

    prompts = config.get("prompts")
    if isinstance(prompts, MutableMapping):
        for prompt_name in ("choose_room_prompt", "choose_node_prompt"):
            if prompt_name in prompts:
                prompts[prompt_name] = _resolve_path(
                    prompts[prompt_name], base_dir=base_dir
                )

    output = config.get("output")
    if isinstance(output, MutableMapping) and "output_dir" in output:
        output["output_dir"] = _resolve_path(
            output["output_dir"], base_dir=base_dir
        )


def safe_config_dict(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a deep-copied config that is safe to write to disk."""

    secret_names = {
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

    def is_secret_field(value: str) -> bool:
        compact = re.sub(r"[^a-z0-9]+", "", value.lower())
        return compact.endswith("apikey") or compact in secret_names

    def sanitize(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: sanitize(item)
                for key, item in value.items()
                if not is_secret_field(str(key))
            }
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if isinstance(value, tuple):
            return [sanitize(item) for item in value]
        return copy.deepcopy(value)

    return sanitize(dict(config))


def load_preprocessing_config(
    config_path: Union[os.PathLike[str], str],
    *,
    repo_root: Optional[Union[os.PathLike[str], str]] = None,
) -> PreprocessingConfig:
    """Load, sanitize, and resolve paths in a preprocessing JSON config."""

    source_path = Path(config_path).expanduser().resolve(strict=False)
    def reject_nonstandard_number(token: str) -> None:
        raise ValueError(f"non-standard JSON number {token}")

    try:
        with source_path.open("r", encoding="utf-8") as config_file:
            raw_config = json.load(
                config_file, parse_constant=reject_nonstandard_number
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Failed to load config {source_path}: {exc}") from exc

    if not isinstance(raw_config, dict):
        raise ConfigError("Preprocessing config must contain a JSON object")

    llm_config = raw_config.get("llm_config")
    legacy_api_key = None
    if isinstance(llm_config, MutableMapping):
        legacy_api_key = llm_config.pop("api_key", None)
        if legacy_api_key is not None and not isinstance(legacy_api_key, str):
            raise ConfigError("llm_config.api_key must be a string")

    resolved_repo_root = (
        REPO_ROOT
        if repo_root is None
        else Path(repo_root).expanduser().resolve(strict=False)
    )
    _resolve_known_paths(
        raw_config,
        config_path=source_path,
        repo_root=resolved_repo_root,
    )
    return PreprocessingConfig(
        raw_config,
        source_path=source_path,
        legacy_api_key=legacy_api_key,
    )


def write_preprocessing_config(
    config: Mapping[str, Any],
    output_path: Union[os.PathLike[str], str],
) -> Path:
    """Write a sanitized preprocessing config and return its absolute path."""

    destination = Path(output_path).expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as config_file:
        json.dump(
            safe_config_dict(config),
            config_file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        config_file.write("\n")
    return destination
