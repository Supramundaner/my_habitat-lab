"""Runtime configuration shared by the local VLM services."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


GROUNDING_DINO_PORT = 12181
MOBILE_SAM_PORT = 12184
YOLOV7_PORT = 12185


def required_file(
    value: Optional[str],
    *,
    env_var: str,
    cli_flag: str,
    description: str,
) -> Path:
    """Resolve a required file supplied explicitly or through an environment variable."""
    return _required_path(
        value,
        env_var=env_var,
        cli_flag=cli_flag,
        description=description,
        expect_directory=False,
    )


def required_directory(
    value: Optional[str],
    *,
    env_var: str,
    cli_flag: str,
    description: str,
) -> Path:
    """Resolve a required directory from an explicit value or environment variable."""
    return _required_path(
        value,
        env_var=env_var,
        cli_flag=cli_flag,
        description=description,
        expect_directory=True,
    )


def _required_path(
    value: Optional[str],
    *,
    env_var: str,
    cli_flag: str,
    description: str,
    expect_directory: bool,
) -> Path:
    configured_value = value or os.environ.get(env_var)
    if not configured_value:
        raise RuntimeError(
            f"{description} is required. Pass {cli_flag} or set {env_var} "
            "to an absolute path."
        )

    path = Path(configured_value).expanduser()
    if not path.is_absolute():
        raise ValueError(
            f"{description} must be an absolute path: {path}. "
            f"Update {env_var} or {cli_flag}."
        )
    expected_type = "directory" if expect_directory else "file"
    exists_with_expected_type = (
        path.is_dir() if expect_directory else path.is_file()
    )
    if not exists_with_expected_type:
        raise FileNotFoundError(
            f"{description} {expected_type} does not exist: {path}. "
            f"Update {env_var} or {cli_flag}."
        )

    return path.resolve()
