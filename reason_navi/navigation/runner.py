"""Application service for executing the canonical ObjectNav runtime.

Both the command line entrypoint and the evaluation pipeline call this module.
Heavy Habitat dependencies are imported lazily so the orchestration can be
unit-tested with fakes on machines that do not have Habitat-Sim installed.
"""

from __future__ import annotations

import copy
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from .contracts import ContractError, NavigationRequest, RequestFormat
from .config import (
    load_json_object,
    sanitize_secrets,
    validate_navigation_config,
)
from .projection import (
    runtime_projection_options,
    validate_projection_compatibility,
)


def _as_list(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


@dataclass(frozen=True)
class NavigationDependencies:
    """Factories and side effects used by :func:`run_navigation`."""

    simulator_factory: Callable[[Dict[str, Any]], Any]
    composer_factory: Callable[[Any, Dict[str, Any], str], Any]
    occupancy_map_factory: Callable[..., Any]
    controller_factory: Callable[[Any, Any, Dict[str, Any], Any], Any]
    initialize_gpu: Callable[[Dict[str, Any]], None]
    clear_gpu_cache: Callable[[], None]
    validate_config: Callable[[Dict[str, Any]], bool]
    generate_output_paths: Callable[[str], Dict[str, str]]
    write_json_report: Callable[[str, Dict[str, Any]], None]
    now: Callable[[], datetime] = datetime.now


@dataclass(frozen=True)
class NavigationResult:
    """Successful runtime result returned to CLI and evaluators."""

    video_path: Path
    report_path: Path
    report: Mapping[str, Any]


def default_dependencies() -> NavigationDependencies:
    """Load the production implementation only when a real run starts."""

    from .controller import NavigationController
    from .occupancy_map import OccupancyMapBuilder
    from .simulator import HabitatSimulator
    from .utils import (
        clear_gpu_cache,
        generate_output_paths,
        initialize_gpu,
        write_json_report,
    )
    from .video import VideoComposer

    return NavigationDependencies(
        simulator_factory=HabitatSimulator,
        composer_factory=VideoComposer,
        occupancy_map_factory=OccupancyMapBuilder,
        controller_factory=NavigationController,
        initialize_gpu=initialize_gpu,
        clear_gpu_cache=clear_gpu_cache,
        validate_config=lambda config: (
            validate_navigation_config(config) is None
        ),
        generate_output_paths=generate_output_paths,
        write_json_report=write_json_report,
    )


def run_navigation(
    config: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    wall_mask: Optional[str] = None,
    map_metadata: Optional[str] = None,
    output_dir: Optional[str] = None,
    request_base_dir: Optional[Path] = None,
    dependencies: Optional[NavigationDependencies] = None,
) -> NavigationResult:
    """Execute one navigation request and write its video/report artifacts.

    Args:
        config: Loaded navigation runtime configuration.
        request: Canonical request loaded from ``action.json``.
        wall_mask: Optional explicit wall-mask override. It must be supplied
            together with ``map_metadata`` so the image cannot be detached
            from the world-space projection that produced it.
        map_metadata: Projection metadata paired with ``wall_mask``.
        output_dir: Optional output directory override.
        request_base_dir: Directory used to resolve relative artifact paths
            embedded in the navigation request.
        dependencies: Test seam for simulator, planner, writer, and clock.

    Raises:
        ContractError: If the action document is malformed.
        ValueError: If the navigation configuration is invalid.
        Any exception raised by a production component after resources have
            been closed in the ``finally`` block.
    """

    deps = dependencies or default_dependencies()
    runtime_config: Dict[str, Any] = copy.deepcopy(dict(config))
    if output_dir is not None:
        runtime_config["output_dir"] = output_dir

    if not deps.validate_config(runtime_config):
        raise ValueError("navigation configuration is invalid")
    if not runtime_config.get("output_dir"):
        raise ValueError("navigation configuration requires output_dir")

    if (wall_mask is None) != (map_metadata is None):
        raise ContractError(
            "wall_mask and map_metadata overrides must be supplied together"
        )
    normalized_request = NavigationRequest.from_mapping(
        request,
        wall_mask_override=wall_mask,
        map_metadata_override=map_metadata,
        base_dir=request_base_dir,
    )
    if normalized_request.input_format is not RequestFormat.TARGET_INFO:
        raise ContractError(
            "the canonical two-stage ObjectNav runtime requires target_info; "
            f"received legacy {normalized_request.input_format.value} format"
        )
    wall_mask_path = normalized_request.wall_mask
    if wall_mask_path is None:
        raise ContractError(
            "the canonical two-stage ObjectNav runtime requires wall_mask"
        )
    if not wall_mask_path.is_file():
        raise FileNotFoundError(f"wall mask does not exist: {wall_mask_path}")

    if normalized_request.map_metadata is None:
        raise ContractError(
            "the canonical two-stage ObjectNav runtime requires map_metadata"
        )
    if not normalized_request.map_metadata.is_file():
        raise FileNotFoundError(
            "top-down projection metadata does not exist: "
            f"{normalized_request.map_metadata}"
        )
    map_metadata = load_json_object(normalized_request.map_metadata)
    runtime_config["topdown_projection"] = dict(
        runtime_projection_options(map_metadata)
    )
    initial_state, agent_state = normalized_request.simulator_states()

    with ExitStack() as resources:
        # Register cleanup before initialization so a partially initialized GPU
        # context is still released if a later factory fails.
        resources.callback(deps.clear_gpu_cache)
        deps.initialize_gpu(runtime_config)
        paths = deps.generate_output_paths(str(runtime_config["output_dir"]))

        simulator = deps.simulator_factory(runtime_config)
        resources.callback(simulator.close)
        simulator.setup_scene_and_agent(initial_state, agent_state)
        runtime_metadata = simulator.get_topdown_metadata()
        if runtime_metadata is None:
            raise RuntimeError(
                "online simulator did not provide top-down projection metadata"
            )
        validate_projection_compatibility(map_metadata, runtime_metadata)

        composer = deps.composer_factory(
            simulator, runtime_config, paths["video"]
        )
        resources.callback(composer.save_and_close)
        use_gpu = runtime_config.get("gpu", {}).get("enabled", False)
        occupancy_map = deps.occupancy_map_factory(
            use_gpu=use_gpu, config=runtime_config
        )
        composer.set_occupancy_map(occupancy_map, runtime_config)

        if not occupancy_map.initialize_from_wall_mask(str(wall_mask_path)):
            raise RuntimeError(
                f"failed to initialize occupancy map from wall mask: {wall_mask_path}"
            )

        controller = deps.controller_factory(
            simulator, composer, runtime_config, occupancy_map
        )
        composer.add_frame()

        start_time = deps.now()
        completed_actions = []
        collision_action = None
        target_found = False
        for command_group in normalized_request.command_groups:
            report_data = controller.navigate(dict(command_group))
            completed_actions.extend(report_data.get("completed_actions", []))
            collision_action = report_data.get("collision_action")
            target_found = bool(report_data.get("target_found", False))
            if collision_action is not None or target_found:
                break
        end_time = deps.now()

        final_state = simulator.get_robot_state()
        execution_stats = controller.get_execution_stats()
        execution_time = (end_time - start_time).total_seconds()
        report: Dict[str, Any] = {
            "execution_info": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "execution_time_seconds": execution_time,
                "total_frames": execution_stats["total_frames"],
                "video_duration_seconds": execution_stats["total_duration"],
            },
            "config": sanitize_secrets(runtime_config),
            "final_agent_state": {
                "position": _as_list(final_state["position"]),
                "rotation": _as_list(final_state["rotation"]),
            },
            "navigation_request": sanitize_secrets(
                dict(normalized_request.raw)
            ),
            "command_groups": [
                dict(group) for group in normalized_request.command_groups
            ],
            # Transitional report aliases retained for existing consumers.
            "original_actions_input": sanitize_secrets(
                dict(normalized_request.raw)
            ),
            "original_sequence": [
                dict(group) for group in normalized_request.command_groups
            ],
            "completed_sequence": completed_actions,
            "collision_at_action": collision_action,
            "target_found": target_found,
            "execution_stats": execution_stats,
        }
        deps.write_json_report(paths["report"], report)
        return NavigationResult(
            video_path=Path(paths["video"]),
            report_path=Path(paths["report"]),
            report=report,
        )
