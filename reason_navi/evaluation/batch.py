"""Batch driver for the canonical two-stage ObjectNav evaluator."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence


EVAL_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EVAL_DIR.parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from reason_navi.evaluation.episode import EpisodeEvaluator
from reason_navi.navigation.config import (
    load_batch_evaluation_config,
    sanitize_secrets,
)


def _dataset_label(episode_json_path: str) -> str:
    """Return a readable, collision-resistant label for a dataset shard."""

    path = Path(episode_json_path)
    name = path.name
    if name.endswith(".gz"):
        name = name[:-3]
    if name.endswith(".json"):
        name = name[:-5]
    parts = path.parts
    parent_label = ""
    if "content" in parts:
        content_index = len(parts) - 1 - list(reversed(parts)).index("content")
        if content_index > 0:
            parent_label = parts[content_index - 1]
    elif path.parent.name:
        parent_label = path.parent.name
    label = f"{parent_label}__{name}" if parent_label else name
    # Hash the logical shard identity, not its absolute deployment root, so a
    # dataset copied from a laptop to /efs keeps the same output key.
    logical_identity = f"{parent_label}/{name}" if parent_label else name
    digest = hashlib.sha1(logical_identity.encode("utf-8")).hexdigest()[:8]
    return f"{label}__{digest}"


def _finite_number_or_none(value: Any) -> Optional[float]:
    """Normalize a possibly legacy JSON distance without emitting NaN/Infinity."""

    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _write_json_atomic(destination: Path, document: Mapping[str, Any]) -> None:
    """Atomically replace ``destination`` with a strict JSON document."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(destination.parent),
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                document,
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(destination))
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


class BatchEvaluator:
    """Run a list of scene/episode pairs and aggregate their metrics."""

    def __init__(
        self,
        batch_config_path: str,
        *,
        evaluator_factory: Callable[[str], Any] = EpisodeEvaluator,
    ) -> None:
        self.batch_config_path = (
            Path(batch_config_path).expanduser().resolve(strict=False)
        )
        self.config = load_batch_evaluation_config(self.batch_config_path)
        self.evaluator_factory = evaluator_factory
        configured_output = self.config.get("output_dir")
        self.base_output_dir = (
            Path(str(configured_output))
            if configured_output
            else EVAL_DIR / "output"
        )
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.results_summary: Dict[str, list] = {"completed": [], "failed": []}
        self.episode_results: Dict[str, Dict[str, Any]] = {}

    def _create_single_episode_config(
        self,
        task: Mapping[str, Any],
        episode_id: str,
        output_dir: Path,
    ) -> Dict[str, Any]:
        scene = self.config.get("scene", {})
        if not isinstance(scene, Mapping) or "robot_urdf" not in scene:
            raise ValueError("scene.robot_urdf is required")
        return sanitize_secrets(
            {
                "preprocess": copy.deepcopy(self.config["preprocess"]),
                "video_generation": copy.deepcopy(
                    self.config["video_generation"]
                ),
                "evaluation": copy.deepcopy(self.config["evaluation"]),
                "output_dir": str(output_dir.resolve()),
                "episode": {
                    "episode_json_path": task["episode_json_path"],
                    "episode_id": str(episode_id),
                },
                "scene": {
                    "scene_file": task["scene_file"],
                    "robot_urdf": scene["robot_urdf"],
                },
            }
        )

    def _load_episode_results(
        self, output_dir: Path, episode_key: str
    ) -> Optional[Dict[str, Any]]:
        output_file = output_dir / "output.json"
        if not output_file.is_file():
            return None
        try:
            with output_file.open("r", encoding="utf-8") as handle:
                document = json.load(handle)
            metrics = document.get("evaluation_results")
            if not isinstance(metrics, Mapping):
                return None
            geodesic_distance = _finite_number_or_none(
                metrics.get("geodesic_distance_to_target")
            )
            optimal_distance = _finite_number_or_none(
                metrics.get("optimal_geodesic_distance")
            )
            path_length = _finite_number_or_none(metrics.get("path_length"))
            sr = float(metrics.get("sr", 0.0))
            spl = float(metrics.get("spl", 0.0))
            if not math.isfinite(sr) or not math.isfinite(spl):
                raise ValueError("episode metrics must contain finite SR and SPL")
            explicit_reachable = metrics.get("reachable")
            reachable = (
                geodesic_distance is not None and optimal_distance is not None
            )
            if explicit_reachable is not None and (
                not isinstance(explicit_reachable, bool)
                or explicit_reachable != reachable
            ):
                raise ValueError(
                    "episode reachable flag is inconsistent with geodesic distances"
                )
            result = {
                "sr": sr,
                "spl": spl,
                "success": bool(metrics.get("success", False)),
                "reachable": reachable,
                "geodesic_distance_to_target": geodesic_distance,
                "optimal_geodesic_distance": optimal_distance,
                "path_length": path_length,
                "object_category": metrics.get("object_category", "unknown"),
            }
            self.episode_results[episode_key] = result
            return result
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"Could not load results for {episode_key}: {exc}")
            return None

    def _record_pipeline_failure(self, episode_key: str, message: str) -> None:
        result = self.episode_results.setdefault(
            episode_key,
            {
                "sr": 0.0,
                "spl": 0.0,
                "success": False,
                "reachable": False,
                "geodesic_distance_to_target": None,
                "optimal_geodesic_distance": None,
                "path_length": None,
                "object_category": "unknown",
                "pipeline_error": message,
            },
        )
        result["pipeline_error"] = message

    def _save_batch_results(self) -> Path:
        successful = [
            key for key, value in self.episode_results.items() if value["success"]
        ]
        unsuccessful = [
            key for key, value in self.episode_results.items() if not value["success"]
        ]
        total = len(self.episode_results)
        output = {
            "batch_summary": {
                "total_episodes_processed": total,
                "navigation_successes": len(successful),
                "navigation_failures": len(unsuccessful),
                # Preserve the historical summary keys for downstream readers.
                "succeeded": len(successful),
                "failed": len(unsuccessful),
                "pipeline_failures": len(self.results_summary["failed"]),
                "overall_sr": (
                    sum(value["sr"] for value in self.episode_results.values())
                    / total
                    if total
                    else 0.0
                ),
                "overall_spl": (
                    sum(value["spl"] for value in self.episode_results.values())
                    / total
                    if total
                    else 0.0
                ),
            },
            "successful_episodes": successful,
            "unsuccessful_episodes": unsuccessful,
            "failed_episodes": unsuccessful,
            "completed_episodes": self.results_summary["completed"],
            "pipeline_failed_episodes": self.results_summary["failed"],
            "episode_details": self.episode_results,
            "config_used": sanitize_secrets(self.config),
        }
        destination = self.base_output_dir / "batch_output.json"
        _write_json_atomic(destination, output)
        print(f"Batch results saved: {destination}")
        return destination

    def run_batch_evaluation(self) -> bool:
        tasks = self.config.get("evaluation_tasks", [])
        total = sum(len(task.get("episode_ids", [])) for task in tasks)
        processed = 0
        scheduled_keys = set()

        for task in tasks:
            scene_name = _dataset_label(task["episode_json_path"])
            for raw_episode_id in task["episode_ids"]:
                episode_key = f"{scene_name}/{raw_episode_id}"
                if episode_key in scheduled_keys:
                    raise ValueError(
                        f"duplicate batch episode would overwrite output: {episode_key}"
                    )
                scheduled_keys.add(episode_key)

        for task in tasks:
            scene_name = _dataset_label(task["episode_json_path"])

            for raw_episode_id in task.get("episode_ids", []):
                processed += 1
                episode_id = str(raw_episode_id)
                episode_key = f"{scene_name}/{episode_id}"
                output_dir = self.base_output_dir / scene_name / episode_id
                print(f"[{processed}/{total}] Evaluating {episode_key}")
                try:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    config_path = output_dir / "eval_config.json"
                    config_data = self._create_single_episode_config(
                        task, episode_id, output_dir
                    )
                    _write_json_atomic(config_path, config_data)
                    completed = bool(
                        self.evaluator_factory(str(config_path)).run_evaluation()
                    )
                    result = self._load_episode_results(output_dir, episode_key)
                    if completed and result is not None:
                        self.results_summary["completed"].append(episode_key)
                    else:
                        message = "episode pipeline did not produce valid metrics"
                        self.results_summary["failed"].append(episode_key)
                        self._record_pipeline_failure(episode_key, message)
                except Exception as exc:
                    traceback.print_exc()
                    self.results_summary["failed"].append(episode_key)
                    self._record_pipeline_failure(episode_key, str(exc))
                finally:
                    # Checkpoint after every attempted episode so an interruption
                    # cannot discard the rest of a long batch's completed work.
                    self._save_batch_results()

        # Preserve the historical behavior of producing an empty summary when a
        # valid batch contains no episodes.
        if processed == 0:
            self._save_batch_results()
        print(
            f"Batch complete: {len(self.results_summary['completed'])} completed, "
            f"{len(self.results_summary['failed'])} pipeline failures"
        )
        return not self.results_summary["failed"]


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a batch of canonical two-stage ObjectNav episodes"
    )
    parser.add_argument("config", help="Path to a batch evaluation JSON config")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_arguments(argv)
    try:
        return 0 if BatchEvaluator(args.config).run_batch_evaluation() else 1
    except Exception as exc:
        print(f"Batch evaluator setup failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
