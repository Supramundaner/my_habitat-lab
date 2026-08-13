"""Single-episode evaluator for the canonical two-stage ObjectNav pipeline."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import logging
import math
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence


# These must be set before Habitat-Sim is imported.  Habitat is loaded lazily
# below, which keeps config/orchestration tests runnable on a CPU-only laptop.
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
os.environ.setdefault("GLOG_logtostderr", "0")
os.environ.setdefault("GLOG_stderrthreshold", "3")
os.environ.setdefault("GLOG_v", "0")

logging.getLogger("habitat").setLevel(logging.ERROR)
logging.getLogger("habitat_sim").setLevel(logging.ERROR)


EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
REPOSITORY_ROOT = PROJECT_ROOT.parent
PREPROCESSING_ROOT = PROJECT_ROOT / "preprocessing"
if str(REPOSITORY_ROOT) not in sys.path:
    # Support direct script execution without making internal modules depend
    # on the caller's cwd or a generic top-level ``src`` package.
    sys.path.insert(0, str(REPOSITORY_ROOT))

from reason_navi.navigation.config import (
    load_evaluation_config,
    load_json_object,
    load_navigation_config,
    sanitize_secrets,
)
from reason_navi.navigation.metrics import compute_navigation_metrics
from reason_navi.navigation.runner import run_navigation
from reason_navi.preprocessing.config import write_preprocessing_config


def _default_preprocessing_factory(config_path: str) -> Any:
    """Import the Habitat/OpenCV preprocessing stack only when it is run."""

    from reason_navi.preprocessing.pipeline import PreprocessingPipeline

    return PreprocessingPipeline(config_path)


def _goal_positions(goals: Sequence[Mapping[str, Any]]) -> List[List[float]]:
    view_points: List[List[float]] = []
    for goal in goals:
        for view_point in goal.get("view_points", []):
            try:
                position = view_point["agent_state"]["position"]
            except (KeyError, TypeError):
                continue
            view_points.append(list(position))
    if not view_points:
        view_points = [
            list(goal["position"])
            for goal in goals
            if isinstance(goal, Mapping) and "position" in goal
        ]
    if not view_points:
        raise ValueError("No target viewpoints or goal positions were found")
    return view_points


def evaluate_geodesic_navigation(
    *,
    scene_file: str,
    start_position: Sequence[float],
    final_position: Sequence[float],
    goals: Sequence[Mapping[str, Any]],
    path_length: float,
    success_threshold: float,
) -> Dict[str, Any]:
    """Calculate geodesic SR/SPL using a short-lived Habitat-Sim instance."""

    try:
        import habitat_sim
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Habitat-Sim and NumPy are required for geodesic evaluation"
        ) from exc

    view_points = [np.asarray(point, dtype=float) for point in _goal_positions(goals)]
    start = np.asarray(start_position, dtype=float)
    final = np.asarray(final_position, dtype=float)

    simulator = None
    try:
        simulator_config = habitat_sim.SimulatorConfiguration()
        simulator_config.scene_id = scene_file
        agent_config = habitat_sim.AgentConfiguration()
        simulator = habitat_sim.Simulator(
            habitat_sim.Configuration(simulator_config, [agent_config])
        )
        if not simulator.pathfinder.is_loaded:
            navmesh_settings = habitat_sim.NavMeshSettings()
            navmesh_settings.set_defaults()
            if not simulator.recompute_navmesh(
                simulator.pathfinder, navmesh_settings
            ):
                raise RuntimeError(
                    "Failed to load or compute a navmesh for geodesic evaluation"
                )

        start_path = habitat_sim.MultiGoalShortestPath()
        start_path.requested_start = start
        start_path.requested_ends = view_points
        simulator.pathfinder.find_path(start_path)

        final_path = habitat_sim.MultiGoalShortestPath()
        final_path.requested_start = final
        final_path.requested_ends = view_points
        simulator.pathfinder.find_path(final_path)

        return compute_navigation_metrics(
            optimal_distance=start_path.geodesic_distance,
            distance_to_target=final_path.geodesic_distance,
            path_length=path_length,
            success_threshold=success_threshold,
        ).as_dict()
    finally:
        if simulator is not None:
            simulator.close()


@dataclass(frozen=True)
class EvaluatorDependencies:
    """Side-effect boundaries used by :class:`EpisodeEvaluator`."""

    preprocessing_factory: Callable[[str], Any] = (
        _default_preprocessing_factory
    )
    runner: Callable[..., Any] = run_navigation
    metric_evaluator: Callable[..., Dict[str, Any]] = evaluate_geodesic_navigation


def _dataset_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".gz"):
        name = name[:-3]
    if name.endswith(".json"):
        name = name[:-5]
    return name


def _scene_identity(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("scene id must be a non-empty string")
    normalized = value.replace("\\", "/").rstrip("/")
    return normalized.split("/")[-1]


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a JSON object")
    return value


class EpisodeEvaluator:
    """Orchestrate preprocessing, navigation, and SR/SPL evaluation."""

    def __init__(
        self,
        config_path: str,
        *,
        dependencies: Optional[EvaluatorDependencies] = None,
    ) -> None:
        self.config_path = Path(config_path).expanduser().resolve(strict=False)
        self.config = load_evaluation_config(self.config_path)
        self.dependencies = dependencies or EvaluatorDependencies()
        self.project_root = PROJECT_ROOT
        self.preprocessing_dir = PREPROCESSING_ROOT

        episode_config = _require_mapping(self.config.get("episode"), "episode")
        episode_path = Path(str(episode_config.get("episode_json_path", "")))
        if not episode_path.name:
            raise ValueError("episode.episode_json_path is required")
        episode_id = str(episode_config.get("episode_id", ""))
        if not episode_id:
            raise ValueError("episode.episode_id is required")
        self.scene_id = _dataset_stem(episode_path)

        configured_output = self.config.get("output_dir")
        self.output_dir = (
            Path(str(configured_output))
            if configured_output
            else EVAL_DIR / "output" / self.scene_id / episode_id
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")

        self.results: Dict[str, Any] = {
            "config": sanitize_secrets(self.config),
            "episode_data": None,
            "preprocessing_success": False,
            "navigation_success": False,
            # Transitional alias retained in output.json.
            "video_generation_success": False,
            "evaluation_results": None,
            "errors": [],
        }

    def _load_episode_data(self) -> Dict[str, Any]:
        episode_config = _require_mapping(self.config["episode"], "episode")
        episode_json_path = Path(str(episode_config["episode_json_path"]))
        episode_id = str(episode_config["episode_id"])

        try:
            opener = gzip.open if episode_json_path.suffix == ".gz" else open
            with opener(episode_json_path, "rt", encoding="utf-8") as handle:
                dataset = json.load(handle)

            target_episode = next(
                (
                    episode
                    for episode in dataset["episodes"]
                    if str(episode["episode_id"]) == episode_id
                ),
                None,
            )
            if target_episode is None:
                raise ValueError(
                    f"Episode {episode_id} not found in {episode_json_path}"
                )

            configured_scene = _require_mapping(
                self.config.get("scene"), "scene"
            ).get("scene_file")
            episode_scene = target_episode.get("scene_id")
            if _scene_identity(episode_scene) != _scene_identity(configured_scene):
                raise ValueError(
                    "episode scene does not match configured simulator scene: "
                    f"{_scene_identity(episode_scene)!r} != "
                    f"{_scene_identity(configured_scene)!r}"
                )

            object_category = target_episode["object_category"]
            goals_by_category = dataset.get("goals_by_category", {})
            if not isinstance(goals_by_category, Mapping):
                raise ValueError("goals_by_category must be a JSON object")
            goals_key = f"{_scene_identity(episode_scene)}_{object_category}"
            if goals_key not in goals_by_category:
                # Early single-scene artifacts sometimes used the bare
                # category. It is safe only when it is the sole goal entry.
                if list(goals_by_category) == [object_category]:
                    goals_key = object_category
                else:
                    goals_key = None
            if goals_key is None:
                raise ValueError(
                    f"No goals found for object category {object_category}"
                )

            goals = goals_by_category[goals_key]
            if not isinstance(goals, list):
                raise ValueError(f"Goals for {goals_key} must be a list")
            episode_data = {
                "episode": target_episode,
                "goals": goals,
                "object_category": object_category,
                "start_position": target_episode["start_position"],
                "start_rotation": target_episode["start_rotation"],
                "scene_id": target_episode["scene_id"],
            }
            print(
                f"Loaded episode {episode_id}: {object_category} "
                f"({len(episode_data['goals'])} goals)"
            )
            return episode_data
        except Exception as exc:
            message = f"Failed to load episode data: {exc}"
            self.results["errors"].append(message)
            raise RuntimeError(message) from exc

    def _create_preprocessing_config(
        self, episode_data: Mapping[str, Any]
    ) -> Path:
        try:
            configured = _require_mapping(
                self.config.get("preprocess"), "preprocess"
            )
            preprocessing_config = copy.deepcopy(dict(configured))

            scene_config = preprocessing_config.setdefault("scene_config", {})
            if not isinstance(scene_config, dict):
                raise ValueError("preprocess.scene_config must be a JSON object")
            scene_config.update(
                {
                    "scene_path": self.config["scene"]["scene_file"],
                    "target_coordinate": episode_data["start_position"],
                    "goal_object": episode_data["object_category"],
                    "rotation": episode_data["start_rotation"],
                }
            )

            prompts = preprocessing_config.setdefault("prompts", {})
            if not isinstance(prompts, dict):
                raise ValueError("preprocess.prompts must be a JSON object")
            prompts.setdefault(
                "choose_room_prompt",
                str(PREPROCESSING_ROOT / "prompts" / "select_room.txt"),
            )
            prompts.setdefault(
                "choose_node_prompt",
                str(
                    PREPROCESSING_ROOT
                    / "prompts"
                    / "select_navigation_node.txt"
                ),
            )
            preprocessing_config["output"] = {
                "output_dir": str(self.output_dir / "preprocess")
            }

            destination = self.output_dir / "preprocess_config.json"
            write_preprocessing_config(
                sanitize_secrets(preprocessing_config), destination
            )
            print(f"Created preprocessing config: {destination}")
            return destination
        except Exception as exc:
            message = f"Failed to create preprocessing config: {exc}"
            self.results["errors"].append(message)
            raise RuntimeError(message) from exc

    def _run_preprocessing(self, preprocessing_config_path: Path) -> bool:
        try:
            pipeline = self.dependencies.preprocessing_factory(
                str(preprocessing_config_path.resolve())
            )
            if pipeline.run():
                print("Preprocessing completed")
                return True
            message = "Preprocessing pipeline failed"
        except Exception as exc:
            message = f"Failed to run preprocessing: {exc}"
            traceback.print_exc()
        self.results["errors"].append(message)
        print(message)
        return False

    def _create_navigation_assets(self) -> Dict[str, Path]:
        try:
            configured = _require_mapping(
                self.config.get("video_generation"), "video_generation"
            )
            navigation_config = copy.deepcopy(dict(configured))
            navigation_config["scene"] = copy.deepcopy(self.config["scene"])
            navigation_config["output_dir"] = str(self.output_dir / "video")

            config_path = self.output_dir / "video_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with config_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    sanitize_secrets(navigation_config),
                    handle,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                handle.write("\n")

            request_path = self.output_dir / "preprocess" / "action.json"
            wall_mask_path = self.output_dir / "preprocess" / "wall_mask.png"
            if not request_path.is_file():
                raise FileNotFoundError(
                    f"Navigation request not found: {request_path}"
                )
            if not wall_mask_path.is_file():
                raise FileNotFoundError(
                    f"Wall-mask file not found: {wall_mask_path}"
                )
            return {
                "config_path": config_path,
                "request_path": request_path,
                "wall_mask_path": wall_mask_path,
            }
        except Exception as exc:
            message = f"Failed to create navigation assets: {exc}"
            self.results["errors"].append(message)
            raise RuntimeError(message) from exc

    def _run_navigation(self, assets: Mapping[str, Path]) -> bool:
        try:
            config_path = assets["config_path"].resolve()
            request_path = assets["request_path"].resolve()
            result = self.dependencies.runner(
                load_navigation_config(config_path),
                load_json_object(request_path),
                request_base_dir=request_path.parent,
            )
            navigation_artifacts = {
                "video": str(result.video_path),
                "report": str(result.report_path),
            }
            self.results["navigation_artifacts"] = navigation_artifacts
            # Transitional alias retained in output.json.
            self.results["video_artifacts"] = navigation_artifacts
            print("Navigation completed")
            return True
        except Exception as exc:
            message = f"Failed to run navigation: {exc}"
            self.results["errors"].append(message)
            traceback.print_exc()
            return False

    def _evaluate_navigation_success(
        self, episode_data: Mapping[str, Any]
    ) -> Dict[str, Any]:
        try:
            artifacts = _require_mapping(
                self.results.get("navigation_artifacts"),
                "navigation_artifacts",
            )
            report_path = Path(str(artifacts["report"]))
            report = load_json_object(report_path)
            final_position = report["final_agent_state"]["position"]
            start_position = episode_data["start_position"]
            fallback_distance = math.dist(start_position, final_position)
            path_length = float(
                report["execution_stats"].get(
                    "total_distance", fallback_distance
                )
            )
            evaluation = _require_mapping(
                self.config.get("evaluation", {}), "evaluation"
            )
            threshold = float(
                evaluation.get("success_distance_threshold", 0.25)
            )

            metrics = dict(
                self.dependencies.metric_evaluator(
                    scene_file=self.config["scene"]["scene_file"],
                    start_position=start_position,
                    final_position=final_position,
                    goals=episode_data["goals"],
                    path_length=path_length,
                    success_threshold=threshold,
                )
            )
            metrics["object_category"] = episode_data["object_category"]
            print(
                "Evaluation completed: "
                f"SR={metrics['sr']:.0f}, SPL={metrics['spl']:.3f}"
            )
            return metrics
        except Exception as exc:
            message = f"Failed to evaluate navigation success: {exc}"
            self.results["errors"].append(message)
            traceback.print_exc()
            return {
                "sr": 0.0,
                "spl": 0.0,
                "success": False,
                "error": message,
                "object_category": episode_data.get(
                    "object_category", "unknown"
                ),
            }

    def _save_results(self) -> None:
        output_path = self.output_dir / "output.json"
        temporary_path = output_path.with_name(f".{output_path.name}.tmp")
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with temporary_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    sanitize_secrets(self.results),
                    handle,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, output_path)
            print(f"Results saved: {output_path}")
        except Exception:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            raise

    def run_evaluation(self) -> bool:
        """Run all stages; navigation failure is a valid measured result."""

        try:
            episode_data = self._load_episode_data()
            self.results["episode_data"] = episode_data

            preprocessing_path = self._create_preprocessing_config(
                episode_data
            )
            self.results["preprocessing_success"] = self._run_preprocessing(
                preprocessing_path
            )
            if not self.results["preprocessing_success"]:
                return False

            navigation_assets = self._create_navigation_assets()
            navigation_success = self._run_navigation(navigation_assets)
            self.results["navigation_success"] = navigation_success
            self.results["video_generation_success"] = navigation_success
            if not navigation_success:
                return False

            metrics = self._evaluate_navigation_success(episode_data)
            self.results["evaluation_results"] = metrics
            return "error" not in metrics
        except Exception as exc:
            message = f"Evaluation failed: {exc}"
            self.results["errors"].append(message)
            traceback.print_exc()
            return False
        finally:
            self._save_results()


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one canonical two-stage ObjectNav episode"
    )
    parser.add_argument("config", help="Path to an evaluation JSON config")
    parser.add_argument(
        "--verbose", action="store_true", help="Print setup tracebacks"
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_arguments(argv)
    try:
        evaluator = EpisodeEvaluator(args.config)
        return 0 if evaluator.run_evaluation() else 1
    except Exception as exc:
        print(f"Evaluation setup failed: {exc}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
