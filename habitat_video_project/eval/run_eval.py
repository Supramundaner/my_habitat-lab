#!/usr/bin/env python3
"""
Automated evaluation script for single episode navigation tasks.
Integrates the full pipeline: preprocessing -> action generation -> video generation -> evaluation.

Usage:
    python run_eval.py eval_config.json

The script will:
1. Load episode data and extract target object and start position
2. Generate preprocessing config and run preprocessing pipeline
3. Generate video generation config and run video generation
4. Evaluate success based on distance to target viewpoints
5. Output results to eval/output/<episode_scene_id>/ directory
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import gzip
import traceback

# Add paths for imports
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))

try:
    import habitat_sim
    from habitat.utils.visualizations import maps
    from habitat_sim import Simulator as Sim
    from habitat.tasks.nav.nav import TopDownMap
except ImportError as e:
    print(f"Warning: Could not import habitat modules: {e}")
    print("Some evaluation features may not work properly.")


class EpisodeEvaluator:
    """Main evaluation orchestrator for single episode navigation tasks."""
    
    def __init__(self, config_path: str):
        """Initialize evaluator with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.project_root = Path(__file__).parent.parent.absolute()
        self.preprocess_dir = self.project_root.parent / "preprocess"
        
        # Extract scene identifier for output directory
        episode_json_path = Path(self.config['episode']['episode_json_path'])
        self.scene_id = episode_json_path.stem
        
        # Setup output directory
        self.output_dir = self.project_root / "eval" / "output" / self.scene_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results
        self.results = {
            "config": self.config,
            "episode_data": None,
            "preprocessing_success": False,
            "video_generation_success": False,
            "evaluation_results": None,
            "errors": []
        }
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration from {self.config_path}: {e}")
    
    def _load_episode_data(self) -> Dict[str, Any]:
        """Load and extract episode data from JSON file."""
        episode_json_path = self.config['episode']['episode_json_path']
        episode_id = self.config['episode']['episode_id']
        
        try:
            # Handle both regular JSON and gzipped JSON
            if episode_json_path.endswith('.gz'):
                with gzip.open(episode_json_path, 'rt', encoding='utf-8') as f:
                    episode_data = json.load(f)
            else:
                with open(episode_json_path, 'r', encoding='utf-8') as f:
                    episode_data = json.load(f)
            
            # Find the specific episode
            target_episode = None
            for episode in episode_data['episodes']:
                if str(episode['episode_id']) == str(episode_id):
                    target_episode = episode
                    break
            
            if target_episode is None:
                raise ValueError(f"Episode {episode_id} not found in {episode_json_path}")
            
            # Extract goals for the object category
            object_category = target_episode['object_category']
            goals_key = None
            
            # Find the correct goals key (may have scene prefix)
            for key in episode_data['goals_by_category'].keys():
                if key.endswith(object_category) or key == object_category:
                    goals_key = key
                    break
            
            if goals_key is None:
                raise ValueError(f"No goals found for object category: {object_category}")
            
            goals = episode_data['goals_by_category'][goals_key]
            
            episode_info = {
                "episode": target_episode,
                "goals": goals,
                "object_category": object_category,
                "start_position": target_episode['start_position'],
                "start_rotation": target_episode['start_rotation'],
                "scene_id": target_episode['scene_id']
            }
            
            print(f"✓ Loaded episode {episode_id}")
            print(f"  Object category: {object_category}")
            print(f"  Start position: {target_episode['start_position']}")
            print(f"  Number of goals: {len(goals)}")
            
            return episode_info
            
        except Exception as e:
            error_msg = f"Failed to load episode data: {e}"
            self.results["errors"].append(error_msg)
            raise RuntimeError(error_msg)
    
    def _create_preprocess_config(self, episode_data: Dict[str, Any]) -> Path:
        """Create preprocessing configuration file."""
        try:
            # Create preprocessing config based on episode data
            preprocess_config = {
                "scene_config": {
                    "scene_path": self.config['scene']['scene_file'],
                    "target_floor": self.config['preprocess']['scene_config']['target_floor'],
                    "target_coordinate": episode_data['start_position'],
                    "goal_object": episode_data['object_category'],
                    "rotation": self.config['preprocess']['scene_config']['rotation'],
                    "custom_ortho_scale": self.config['preprocess']['scene_config']['custom_ortho_scale'],
                    "target_coverage": self.config['preprocess']['scene_config']['target_coverage'],
                    "draw_coordinates": self.config['preprocess']['scene_config']['draw_coordinates']
                },
                "room_segmentation": self.config['preprocess']['room_segmentation'],
                "graph_generation": self.config['preprocess']['graph_generation'],
                "llm_config": self.config['preprocess']['llm_config'],
                "prompts": self.config['preprocess']['prompts'],
                "output": {
                    "output_dir": str(self.output_dir / "preprocess")
                }
            }
            
            # Save preprocessing config
            preprocess_config_path = self.output_dir / "preprocess_config.json"
            with open(preprocess_config_path, 'w', encoding='utf-8') as f:
                json.dump(preprocess_config, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Created preprocessing config: {preprocess_config_path}")
            return preprocess_config_path
            
        except Exception as e:
            error_msg = f"Failed to create preprocessing config: {e}"
            self.results["errors"].append(error_msg)
            raise RuntimeError(error_msg)
    
    def _run_preprocessing(self, preprocess_config_path: Path) -> bool:
        """Run preprocessing pipeline."""
        try:
            print("\\n" + "="*60)
            print("RUNNING PREPROCESSING PIPELINE")
            print("="*60)
            
            # Change to preprocessing directory
            original_cwd = os.getcwd()
            os.chdir(self.preprocess_dir)
            
            try:
                # Run preprocessing
                cmd = [sys.executable, "main_workflow.py", str(preprocess_config_path)]
                print(f"Running: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                if result.returncode == 0:
                    print("✓ Preprocessing completed successfully")
                    
                    # Check if action.json was generated
                    action_json_path = self.output_dir / "preprocess" / "action.json"
                    if action_json_path.exists():
                        print(f"✓ Action file generated: {action_json_path}")
                        return True
                    else:
                        error_msg = "Preprocessing completed but action.json not found"
                        self.results["errors"].append(error_msg)
                        print(f"✗ {error_msg}")
                        return False
                else:
                    error_msg = f"Preprocessing failed with return code {result.returncode}"
                    if result.stderr:
                        error_msg += f": {result.stderr}"
                    self.results["errors"].append(error_msg)
                    print(f"✗ {error_msg}")
                    return False
                    
            finally:
                os.chdir(original_cwd)
                
        except subprocess.TimeoutExpired:
            error_msg = "Preprocessing timed out after 5 minutes"
            self.results["errors"].append(error_msg)
            print(f"✗ {error_msg}")
            return False
        except Exception as e:
            error_msg = f"Failed to run preprocessing: {e}"
            self.results["errors"].append(error_msg)
            print(f"✗ {error_msg}")
            return False
    
    def _create_video_config(self, episode_data: Dict[str, Any]) -> Tuple[Path, Path]:
        """Create video generation configuration and action files."""
        try:
            # Create video generation config
            video_config = {
                "video": self.config['video_generation']['video'],
                "agent": self.config['video_generation']['agent'],
                "scene": self.config['scene'],
                "simulation": self.config['video_generation']['simulation'],
                "output_dir": str(self.output_dir / "video"),
                "gpu": self.config['video_generation']['gpu'],
                "OCCUPANCY_MAP": self.config['video_generation']['OCCUPANCY_MAP'],
                "vfh": self.config['video_generation']['vfh'],
                "navigation": self.config['video_generation']['navigation'],
                "object_detection": self.config['video_generation']['object_detection']
            }
            
            # Save video config
            video_config_path = self.output_dir / "video_config.json"
            with open(video_config_path, 'w', encoding='utf-8') as f:
                json.dump(video_config, f, indent=2, ensure_ascii=False)
            
            # Copy action.json from preprocessing output
            preprocess_action_path = self.output_dir / "preprocess" / "action.json"
            video_action_path = self.output_dir / "action.json"
            
            if preprocess_action_path.exists():
                shutil.copy2(preprocess_action_path, video_action_path)
                print(f"✓ Created video config: {video_config_path}")
                print(f"✓ Copied action file: {video_action_path}")
                return video_config_path, video_action_path
            else:
                raise FileNotFoundError(f"Action file not found: {preprocess_action_path}")
                
        except Exception as e:
            error_msg = f"Failed to create video config: {e}"
            self.results["errors"].append(error_msg)
            raise RuntimeError(error_msg)
    
    def _run_video_generation(self, video_config_path: Path, video_action_path: Path) -> bool:
        """Run video generation."""
        try:
            print("\\n" + "="*60)
            print("RUNNING VIDEO GENERATION")
            print("="*60)
            
            # Change to video project directory
            original_cwd = os.getcwd()
            os.chdir(self.project_root)
            
            try:
                # Run video generation
                cmd = [
                    sys.executable, "main.py",
                    "--config", str(video_config_path),
                    "--actions", str(video_action_path),
                    "--no-histogram"
                ]
                print(f"Running: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout
                )
                
                if result.returncode == 0:
                    print("✓ Video generation completed successfully")
                    
                    # Check if video was generated
                    video_dir = self.output_dir / "video"
                    video_files = list(video_dir.glob("*.mp4"))
                    if video_files:
                        print(f"✓ Video file generated: {video_files[0]}")
                        # Copy video to main output directory
                        shutil.copy2(video_files[0], self.output_dir / "output.mp4")
                        return True
                    else:
                        error_msg = "Video generation completed but no video file found"
                        self.results["errors"].append(error_msg)
                        print(f"✗ {error_msg}")
                        return False
                else:
                    error_msg = f"Video generation failed with return code {result.returncode}"
                    if result.stderr:
                        error_msg += f": {result.stderr}"
                    self.results["errors"].append(error_msg)
                    print(f"✗ {error_msg}")
                    return False
                    
            finally:
                os.chdir(original_cwd)
                
        except subprocess.TimeoutExpired:
            error_msg = "Video generation timed out after 10 minutes"
            self.results["errors"].append(error_msg)
            print(f"✗ {error_msg}")
            return False
        except Exception as e:
            error_msg = f"Failed to run video generation: {e}"
            self.results["errors"].append(error_msg)
            print(f"✗ {error_msg}")
            return False
    
    def _evaluate_navigation_success(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate navigation success based on distance to target viewpoints."""
        try:
            print("\\n" + "="*60)
            print("EVALUATING NAVIGATION SUCCESS")
            print("="*60)
            
            # Load video generation report to get final agent position
            video_report_path = self.output_dir / "video" / "execution_report.json"
            if not video_report_path.exists():
                raise FileNotFoundError(f"Video report not found: {video_report_path}")
            
            with open(video_report_path, 'r') as f:
                video_report = json.load(f)
            
            final_position = np.array(video_report['final_agent_state']['position'])
            start_position = np.array(episode_data['start_position'])
            
            # Extract viewpoints from goals
            view_points = []
            for goal in episode_data['goals']:
                for view_point in goal.get('view_points', []):
                    view_points.append(np.array(view_point['agent_state']['position']))
            
            if not view_points:
                print("⚠ No viewpoints found in goals, using goal positions instead")
                for goal in episode_data['goals']:
                    if 'position' in goal:
                        view_points.append(np.array(goal['position']))
            
            if not view_points:
                raise ValueError("No target viewpoints or positions found in goals")
            
            print(f"Final agent position: {final_position}")
            print(f"Number of target viewpoints: {len(view_points)}")
            
            # Calculate distances to all viewpoints
            distances = [np.linalg.norm(final_position - vp) for vp in view_points]
            min_distance = min(distances)
            
            # Success criteria
            success_threshold = self.config['evaluation']['success_distance_threshold']
            success = min_distance <= success_threshold
            
            # Calculate SPL (Success weighted by Path Length)
            # This requires habitat-sim for pathfinding - simplified version here
            episode_distance = video_report['execution_info'].get('total_distance', 0)
            if episode_distance == 0:
                episode_distance = np.linalg.norm(final_position - start_position)
            
            # Simplified SPL calculation (assumes direct path distance)
            optimal_distance = min(distances)  # Simplified - should use pathfinder
            if success and optimal_distance > 0:
                spl = optimal_distance / max(optimal_distance, episode_distance)
            else:
                spl = 0.0
            
            evaluation_results = {
                "success": success,
                "min_distance_to_target": min_distance,
                "success_threshold": success_threshold,
                "spl": spl,
                "final_position": final_position.tolist(),
                "start_position": start_position.tolist(),
                "target_viewpoints": [vp.tolist() for vp in view_points],
                "episode_distance": episode_distance,
                "object_category": episode_data['object_category']
            }
            
            print(f"✓ Navigation Success: {success}")
            print(f"  Distance to target: {min_distance:.3f}m (threshold: {success_threshold}m)")
            print(f"  SPL: {spl:.3f}")
            
            return evaluation_results
            
        except Exception as e:
            error_msg = f"Failed to evaluate navigation success: {e}"
            self.results["errors"].append(error_msg)
            print(f"✗ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "min_distance_to_target": float('inf'),
                "success_threshold": self.config['evaluation']['success_distance_threshold'],
                "spl": 0.0
            }
    
    def _save_results(self):
        """Save final evaluation results."""
        try:
            output_json_path = self.output_dir / "output.json"
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            print(f"✓ Results saved to: {output_json_path}")
        except Exception as e:
            print(f"✗ Failed to save results: {e}")
    
    def run_evaluation(self) -> bool:
        """Run the complete evaluation pipeline."""
        try:
            print("\\n" + "🚀" + "="*58 + "🚀")
            print("🎯 STARTING AUTOMATED EPISODE EVALUATION 🎯")
            print("🚀" + "="*58 + "🚀")
            print(f"Episode: {self.config['episode']['episode_id']}")
            print(f"Output directory: {self.output_dir}")
            
            # Step 1: Load episode data
            print("\\n1. Loading episode data...")
            episode_data = self._load_episode_data()
            self.results["episode_data"] = episode_data
            
            # Step 2: Create preprocessing config and run preprocessing
            print("\\n2. Running preprocessing...")
            preprocess_config_path = self._create_preprocess_config(episode_data)
            preprocessing_success = self._run_preprocessing(preprocess_config_path)
            self.results["preprocessing_success"] = preprocessing_success
            
            if not preprocessing_success:
                print("✗ Preprocessing failed, stopping evaluation")
                return False
            
            # Step 3: Create video config and run video generation
            print("\\n3. Running video generation...")
            video_config_path, video_action_path = self._create_video_config(episode_data)
            video_success = self._run_video_generation(video_config_path, video_action_path)
            self.results["video_generation_success"] = video_success
            
            if not video_success:
                print("✗ Video generation failed, stopping evaluation")
                return False
            
            # Step 4: Evaluate navigation success
            print("\\n4. Evaluating navigation success...")
            evaluation_results = self._evaluate_navigation_success(episode_data)
            self.results["evaluation_results"] = evaluation_results
            
            # Step 5: Save results
            print("\\n5. Saving results...")
            self._save_results()
            
            print("\\n" + "🎉" + "="*58 + "🎉")
            print("🎯 EVALUATION COMPLETED SUCCESSFULLY! 🎯")
            print("🎉" + "="*58 + "🎉")
            
            # Print summary
            success = evaluation_results.get("success", False)
            distance = evaluation_results.get("min_distance_to_target", float('inf'))
            spl = evaluation_results.get("spl", 0.0)
            
            print(f"\\nEvaluation Summary:")
            print(f"  Episode ID: {self.config['episode']['episode_id']}")
            print(f"  Object Category: {episode_data['object_category']}")
            print(f"  Success: {'✓' if success else '✗'}")
            print(f"  Distance to target: {distance:.3f}m")
            print(f"  SPL: {spl:.3f}")
            print(f"\\nOutput files:")
            print(f"  Video: {self.output_dir}/output.mp4")
            print(f"  Results: {self.output_dir}/output.json")
            
            return True
            
        except Exception as e:
            error_msg = f"Evaluation failed with error: {e}"
            print(f"\\n✗ {error_msg}")
            self.results["errors"].append(error_msg)
            self._save_results()
            if "--verbose" in sys.argv:
                traceback.print_exc()
            return False


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python run_eval.py <eval_config.json> [--verbose]")
        print("\\nExample:")
        print("  python run_eval.py eval_config_template.json")
        print("\\nThe script will automatically:")
        print("  1. Extract episode data and create preprocessing config")
        print("  2. Run preprocessing to generate actions")
        print("  3. Run video generation with the actions")
        print("  4. Evaluate navigation success")
        print("  5. Save results and video to output directory")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        evaluator = EpisodeEvaluator(config_path)
        success = evaluator.run_evaluation()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\\n\\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\nFatal error during evaluation: {e}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
