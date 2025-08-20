import os
import sys
import json
import shutil
import traceback
from pathlib import Path
from typing import Dict, Any

# Ensure the local run_eval module can be imported
# This assumes batch_eval.py and run_eval.py are in the same directory.
try:
    from run_eval import EpisodeEvaluator
except ImportError:
    print("Fatal: Could not import EpisodeEvaluator from run_eval.py.")
    print("Please ensure batch_eval.py and run_eval.py are in the same directory.")
    sys.exit(1)

class BatchEvaluator:
    """Orchestrates running evaluations for multiple episodes across multiple scenes."""

    def __init__(self, batch_config_path: str):
        """
        Initialize the batch evaluator with a batch configuration file.

        Args:
            batch_config_path: Path to the JSON configuration file for the batch run.
        """
        self.batch_config_path = Path(batch_config_path)
        if not self.batch_config_path.exists():
            raise FileNotFoundError(f"Batch config file not found: {self.batch_config_path}")

        self.config = self._load_config()
        self.project_root = Path(__file__).resolve().parent.parent
        self.base_output_dir = self.project_root / "eval" / "output"
        self.results_summary = {"succeeded": [], "failed": []}

    def _load_config(self) -> Dict[str, Any]:
        """Loads the batch JSON configuration file."""
        print(f"Loading batch configuration from: {self.batch_config_path}")
        with open(self.batch_config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _create_single_episode_config(self, task: Dict[str, Any], episode_id: str) -> Dict[str, Any]:
        """
        Dynamically creates a configuration dictionary for a single episode.

        Args:
            task: The task dictionary from the batch config, containing scene info.
            episode_id: The specific episode ID to run.

        Returns:
            A dictionary formatted for the EpisodeEvaluator.
        """
        # Start with the common settings
        single_config = {
            "preprocess": self.config["preprocess"],
            "video_generation": self.config["video_generation"],
            "evaluation": self.config["evaluation"]
        }
        
        # Add the episode-specific information
        single_config["episode"] = {
            "episode_json_path": task["episode_json_path"],
            "episode_id": episode_id
        }

        # Add the scene-specific information
        single_config["scene"] = {
            "scene_file": task["scene_file"],
            "robot_urdf": self.config["scene"]["robot_urdf"]
        }
        
        return single_config

    def run_batch_evaluation(self):
        """Executes the evaluation for all tasks and episodes in the config."""
        tasks = self.config.get("evaluation_tasks", [])
        total_episodes = sum(len(task.get("episode_ids", [])) for task in tasks)
        
        print("\n" + "🚀" * 30)
        print(f"  STARTING BATCH EVALUATION: {len(tasks)} tasks, {total_episodes} total episodes.")
        print("🚀" * 30 + "\n")

        processed_count = 0
        for task in tasks:
            scene_json_path = Path(task["episode_json_path"])
            scene_stem = scene_json_path.stem

            for episode_id in task.get("episode_ids", []):
                processed_count += 1
                print("\n" + "=" * 80)
                print(f"Processing Episode {processed_count}/{total_episodes}: Scene='{scene_stem}', Episode ID='{episode_id}'")
                print("=" * 80)

                # 1. Define final output directory and create it
                final_output_dir = self.base_output_dir / scene_stem / episode_id
                final_output_dir.mkdir(parents=True, exist_ok=True)

                # 2. Create the temporary single-episode config
                temp_config_data = self._create_single_episode_config(task, episode_id)
                temp_config_path = final_output_dir / f"temp_eval_config_{episode_id}.json"
                
                with open(temp_config_path, 'w', encoding='utf-8') as f:
                    json.dump(temp_config_data, f, indent=4)
                
                # The EpisodeEvaluator will write to `eval/output/<scene_stem>`
                intermediate_output_dir = self.base_output_dir / scene_stem
                
                try:
                    # 3. Instantiate and run the single episode evaluator
                    evaluator = EpisodeEvaluator(str(temp_config_path))
                    
                    # The EpisodeEvaluator's output dir will be the intermediate one.
                    # We need to ensure it's not the same as our final one to avoid recursion.
                    if evaluator.output_dir.resolve() == final_output_dir.resolve():
                        raise SystemExit(f"Logic Error: Intermediate path {evaluator.output_dir} is the same as final path {final_output_dir}. Aborting.")

                    # Ensure the intermediate directory doesn't already exist from a failed previous run
                    if intermediate_output_dir.exists() and intermediate_output_dir.is_dir():
                         if any(intermediate_output_dir.iterdir()):
                            print(f"Warning: Cleaning up unexpected files in intermediate directory: {intermediate_output_dir}")
                            shutil.rmtree(intermediate_output_dir)

                    success = evaluator.run_evaluation()

                    # 4. Move the results from the intermediate path to the final path
                    if intermediate_output_dir.exists():
                        print(f"Moving results from '{intermediate_output_dir}' to '{final_output_dir}'...")
                        # We rename the directory, which is an atomic operation
                        os.rename(intermediate_output_dir, final_output_dir)
                    else:
                        print(f"Warning: Evaluator finished but no output directory found at '{intermediate_output_dir}'")


                    if success:
                        print(f"✅ SUCCESS: Episode '{episode_id}' completed successfully.")
                        self.results_summary["succeeded"].append(f"{scene_stem}/{episode_id}")
                    else:
                        print(f"❌ FAILED: Episode '{episode_id}' finished with errors (check logs).")
                        self.results_summary["failed"].append(f"{scene_stem}/{episode_id}")

                except Exception as e:
                    print(f"💥 CRITICAL FAILURE: An unhandled exception occurred for episode '{episode_id}'.")
                    traceback.print_exc()
                    self.results_summary["failed"].append(f"{scene_stem}/{episode_id}")
                finally:
                    # 5. Clean up temporary config file
                    if temp_config_path.exists():
                        temp_config_path.unlink()
                    # If the rename failed for some reason and the intermediate dir still exists, clean it.
                    if intermediate_output_dir.exists() and not any(intermediate_output_dir.iterdir()):
                        intermediate_output_dir.rmdir()


        self._print_summary()

    def _print_summary(self):
        """Prints a final summary of the batch evaluation results."""
        print("\n" + "📊" * 30)
        print("          BATCH EVALUATION SUMMARY")
        print("📊" * 30)

        num_succeeded = len(self.results_summary["succeeded"])
        num_failed = len(self.results_summary["failed"])
        total = num_succeeded + num_failed

        print(f"\nTotal Episodes Processed: {total}")
        print(f"  ✅ Succeeded: {num_succeeded}")
        print(f"  ❌ Failed:    {num_failed}")

        if self.results_summary["succeeded"]:
            print("\nSuccessful Episodes:")
            for item in self.results_summary["succeeded"]:
                print(f"  - {item}")

        if self.results_summary["failed"]:
            print("\nFailed Episodes:")
            for item in self.results_summary["failed"]:
                print(f"  - {item}")
        
        print("\nBatch run complete.")


def main():
    """Main entry point for the batch evaluation script."""
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(f"Usage: python {sys.argv[0]} <path_to_batch_eval_config.json>")
        sys.exit(1)

    config_path = sys.argv[1]
    try:
        batch_runner = BatchEvaluator(config_path)
        batch_runner.run_batch_evaluation()
    except Exception as e:
        print(f"\nFatal error during batch evaluator setup: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()