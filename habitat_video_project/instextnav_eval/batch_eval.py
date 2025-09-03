import traceback
import os
import sys

# Suppress verbose logging from habitat-sim - MUST be set before any habitat imports
os.environ['GLOG_minloglevel'] = '3'  # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL
os.environ['MAGNUM_LOG'] = 'quiet'
os.environ['HABITAT_SIM_LOG'] = 'quiet'
# Additional environment variables to suppress C++ logging
os.environ['GLOG_logtostderr'] = '0'
os.environ['GLOG_stderrthreshold'] = '3'
os.environ['GLOG_v'] = '0'

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure Python logging
import logging
logging.getLogger("habitat").setLevel(logging.ERROR)
logging.getLogger("habitat_sim").setLevel(logging.ERROR)

# Ensure the local run_eval module can be imported
# This assumes batch_eval.py and run_eval.py are in the same directory.
try:
    from run_eval import EpisodeEvaluator
except ImportError:
    print("Fatal: Could not import EpisodeEvaluator from run_eval.py.")
    print("Please ensure batch_eval.py and run_eval.py are in the same directory.")
    sys.exit(1)

class BatchEvaluator:
    """Orchestrates running evaluations for multiple episodes across multiple scenes for image instance navigation."""

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
        self.base_output_dir = self.project_root / "instextnav_eval" / "output"
        self.results_summary = {"succeeded": [], "failed": []}
        self.episode_results = {}  # Store detailed results for each episode

    def _load_config(self) -> Dict[str, Any]:
        """Loads the batch JSON configuration file."""
        print(f"Loading batch configuration from: {self.batch_config_path}")
        with open(self.batch_config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _create_single_episode_config(self, task: Dict[str, Any], episode_id: str, output_dir: Path) -> Dict[str, Any]:
        """
        Dynamically creates a configuration dictionary for a single episode.

        Args:
            task: The task dictionary from the batch config, containing scene info.
            episode_id: The specific episode ID to run.
            output_dir: The desired output directory for this episode.

        Returns:
            A dictionary formatted for the EpisodeEvaluator.
        """
        # Start with the common settings
        single_config = {
            "preprocess": self.config["preprocess"],
            "video_generation": self.config["video_generation"],
            "evaluation": self.config["evaluation"],
            "output_dir": str(output_dir)  # Override the output directory
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

    def _load_episode_results(self, output_dir: Path, episode_key: str) -> Optional[Dict[str, Any]]:
        """Load evaluation results from an episode's output.json file."""
        try:
            output_file = output_dir / "output.json"
            if output_file.exists():
                with open(output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "evaluation_results" in data:
                        results = data["evaluation_results"]
                        return {
                            "success": results.get("success", False),
                            "sr": results.get("sr", 0.0),
                            "spl": results.get("spl", 0.0),
                            "min_distance_to_target": results.get("geodesic_distance_to_target", results.get("min_distance_to_target", float('inf'))),
                            "episode_key": episode_key
                        }
            return None
        except Exception as e:
            print(f"Warning: Could not load results for {episode_key}: {e}")
            return None

    def _save_batch_results(self):
        """Save detailed batch results to batch_output.json."""
        try:
            # Calculate overall statistics
            successful_episodes = []
            failed_episodes = []
            total_sr = 0.0
            total_spl = 0.0
            total_episodes = len(self.episode_results)

            for episode_key, results in self.episode_results.items():
                if results["success"]:
                    successful_episodes.append(episode_key)
                else:
                    failed_episodes.append(episode_key)
                
                total_sr += results["sr"]
                total_spl += results["spl"]

            # Calculate averages
            num_successful = len(successful_episodes)
            avg_sr = total_sr / total_episodes if total_episodes > 0 else 0.0
            avg_spl = total_spl / total_episodes if total_episodes > 0 else 0.0

            # Create batch results dictionary
            batch_results = {
                "batch_summary": {
                    "total_episodes_processed": total_episodes,
                    "succeeded": num_successful,
                    "failed": len(failed_episodes),
                    "overall_sr": avg_sr,
                    "overall_spl": avg_spl
                },
                "successful_episodes": successful_episodes,
                "failed_episodes": failed_episodes,
                "episode_details": self.episode_results,
                "config_used": self.config
            }

            # Save to batch_output.json
            batch_output_file = self.base_output_dir / "batch_output.json"
            batch_output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(batch_output_file, 'w', encoding='utf-8') as f:
                json.dump(batch_results, f, indent=2)
            
            print(f"\n📊 Batch results saved to: {batch_output_file}")

        except Exception as e:
            print(f"Warning: Could not save batch results: {e}")

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
                temp_config_data = self._create_single_episode_config(task, episode_id, final_output_dir)
                temp_config_path = final_output_dir / f"temp_eval_config_{episode_id}.json"
                
                with open(temp_config_path, 'w', encoding='utf-8') as f:
                    json.dump(temp_config_data, f, indent=2)
                
                episode_key = f"{scene_stem}_episode_{episode_id}"
                
                try:
                    # 3. Run the single episode evaluation
                    print(f"🔄 Running evaluation for {episode_key}...")
                    evaluator = EpisodeEvaluator(str(temp_config_path))
                    success = evaluator.run_evaluation()
                    
                    # 4. Load and store results
                    episode_results = self._load_episode_results(final_output_dir, episode_key)
                    if episode_results:
                        self.episode_results[episode_key] = episode_results
                        if success:
                            self.results_summary["succeeded"].append(episode_key)
                            print(f"✅ {episode_key} completed successfully")
                        else:
                            self.results_summary["failed"].append(episode_key)
                            print(f"❌ {episode_key} failed")
                    else:
                        self.results_summary["failed"].append(episode_key)
                        self.episode_results[episode_key] = {
                            "success": False,
                            "sr": 0.0,
                            "spl": 0.0,
                            "min_distance_to_target": float('inf'),
                            "episode_key": episode_key,
                            "error": "Could not load results"
                        }
                        print(f"❌ {episode_key} failed - could not load results")

                except Exception as e:
                    error_msg = f"Episode {episode_key} failed with error: {str(e)}"
                    print(f"❌ {error_msg}")
                    self.results_summary["failed"].append(episode_key)
                    self.episode_results[episode_key] = {
                        "success": False,
                        "sr": 0.0,
                        "spl": 0.0,
                        "min_distance_to_target": float('inf'),
                        "episode_key": episode_key,
                        "error": str(e)
                    }
                finally:
                    # Clean up temporary config file
                    try:
                        if temp_config_path.exists():
                            temp_config_path.unlink()
                    except Exception:
                        pass

        self._save_batch_results()
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

        # Calculate and display overall metrics
        if self.episode_results:
            successful_results = [r for r in self.episode_results.values() if r["success"]]
            if successful_results:
                avg_sr = sum(r["sr"] for r in self.episode_results.values()) / len(self.episode_results)
                avg_spl = sum(r["spl"] for r in self.episode_results.values()) / len(self.episode_results)
                print(f"\nOverall Metrics:")
                print(f"  📈 Success Rate (SR): {avg_sr:.3f}")
                print(f"  🎯 SPL: {avg_spl:.3f}")

        if self.results_summary["succeeded"]:
            print("\nSuccessful Episodes:")
            for item in self.results_summary["succeeded"]:
                if item in self.episode_results:
                    results = self.episode_results[item]
                    print(f"  ✅ {item} - SR: {results['sr']:.3f}, SPL: {results['spl']:.3f}, Dist: {results['min_distance_to_target']:.3f}m")
                else:
                    print(f"  ✅ {item}")

        if self.results_summary["failed"]:
            print("\nFailed Episodes:")
            for item in self.results_summary["failed"]:
                if item in self.episode_results:
                    results = self.episode_results[item]
                    error_info = f" - {results.get('error', 'Unknown error')}" if 'error' in results else ""
                    print(f"  ❌ {item}{error_info}")
                else:
                    print(f"  ❌ {item}")
        
        print("\nBatch run complete.")
        print(f"📁 Detailed results saved to: {self.base_output_dir / 'batch_output.json'}")


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
