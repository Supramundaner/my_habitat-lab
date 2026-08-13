"""Portable path handling shared by the evaluation data samplers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Optional


HABITAT_DATA_ROOT_ENV = "HABITAT_DATA_ROOT"
HABITAT_DATA_ROOT_REFERENCE = "${HABITAT_DATA_ROOT}"


@dataclass(frozen=True)
class PortableRoot:
    """A filesystem root plus the spelling to persist in generated JSON."""

    runtime_path: Path
    config_path: str

    def child_for_config(self, child: Path) -> str:
        """Return ``child`` relative to this root using portable separators."""

        try:
            relative = child.relative_to(self.runtime_path)
        except ValueError as exc:
            raise ValueError(
                f"{child} is outside configured root {self.runtime_path}"
            ) from exc
        return str(
            PurePosixPath(self.config_path)
            / PurePosixPath(relative.as_posix())
        )


@dataclass(frozen=True)
class SamplerPaths:
    dataset: PortableRoot
    scenes: PortableRoot
    robot_urdf: str


def _expand_data_root_reference(
    value: str,
    *,
    habitat_data_root: Optional[str],
    option_name: str,
) -> Path:
    if HABITAT_DATA_ROOT_REFERENCE in value:
        if not habitat_data_root:
            raise ValueError(
                f"{option_name} references {HABITAT_DATA_ROOT_REFERENCE}, but "
                f"{HABITAT_DATA_ROOT_ENV} is not set"
            )
        value = value.replace(
            HABITAT_DATA_ROOT_REFERENCE, habitat_data_root
        )
    return Path(value).expanduser()


def _resolve_root(
    cli_value: Optional[str],
    *,
    default_suffix: str,
    habitat_data_root: Optional[str],
    option_name: str,
) -> PortableRoot:
    if cli_value:
        runtime_path = _expand_data_root_reference(
            cli_value,
            habitat_data_root=habitat_data_root,
            option_name=option_name,
        ).resolve(strict=False)
        return PortableRoot(
            runtime_path=runtime_path,
            # Explicit relative CLI inputs are anchored to invocation cwd,
            # while generated JSON may be written elsewhere. Persist the
            # resolved absolute spelling so the output cannot change meaning.
            config_path=(
                cli_value.rstrip("/")
                if HABITAT_DATA_ROOT_REFERENCE in cli_value
                else str(runtime_path)
            ),
        )
    if not habitat_data_root:
        raise ValueError(
            f"Set {HABITAT_DATA_ROOT_ENV} or pass {option_name} explicitly"
        )
    return PortableRoot(
        runtime_path=Path(habitat_data_root).expanduser() / default_suffix,
        config_path=f"{HABITAT_DATA_ROOT_REFERENCE}/{default_suffix}",
    )


def resolve_sampler_paths(
    *,
    data_root: Optional[str],
    scene_root: Optional[str],
    robot_urdf: Optional[str],
    dataset_suffix: str,
    environ: Optional[Mapping[str, str]] = None,
) -> SamplerPaths:
    """Resolve sampler inputs without reading environment at import time.

    Environment-derived inputs retain ``${HABITAT_DATA_ROOT}`` in generated
    configs. Explicit command-line values are preserved instead.
    """

    runtime_environ = os.environ if environ is None else environ
    habitat_data_root = runtime_environ.get(HABITAT_DATA_ROOT_ENV)
    dataset = _resolve_root(
        data_root,
        default_suffix=dataset_suffix,
        habitat_data_root=habitat_data_root,
        option_name="--data-root",
    )
    scenes = _resolve_root(
        scene_root,
        default_suffix="versioned_data/hm3d-0.2/hm3d/val",
        habitat_data_root=habitat_data_root,
        option_name="--scene-root",
    )

    if robot_urdf:
        # Validate an environment reference now while preserving its portable
        # spelling in the generated JSON.
        runtime_robot_urdf = _expand_data_root_reference(
            robot_urdf,
            habitat_data_root=habitat_data_root,
            option_name="--robot-urdf",
        )
        config_robot_urdf = (
            robot_urdf
            if HABITAT_DATA_ROOT_REFERENCE in robot_urdf
            else str(runtime_robot_urdf.resolve(strict=False))
        )
    elif habitat_data_root:
        config_robot_urdf = (
            f"{HABITAT_DATA_ROOT_REFERENCE}/robots/hab_fetch/robots/"
            "hab_fetch.urdf"
        )
    else:
        raise ValueError(
            f"Set {HABITAT_DATA_ROOT_ENV} or pass --robot-urdf explicitly"
        )

    return SamplerPaths(
        dataset=dataset,
        scenes=scenes,
        robot_urdf=config_robot_urdf,
    )


def validate_input_directories(paths: SamplerPaths) -> None:
    """Fail before sampling with an actionable missing-directory message."""

    for label, directory in (
        ("dataset", paths.dataset.runtime_path),
        ("scene", paths.scenes.runtime_path),
    ):
        if not directory.is_dir():
            raise ValueError(f"{label} directory does not exist: {directory}")
