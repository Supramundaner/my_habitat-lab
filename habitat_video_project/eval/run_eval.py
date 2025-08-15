#!/usr/bin/env python3
"""
Automated evaluation script for single episode navigation tasks.
Integrates the full pipeline: preprocessing -> action generation -> video generation -> evaluation.

This version directly imports and executes the logic from main_workflow.py and main.py
instead of using subprocess calls, allowing for better integration and debugging.

Usage:
    python run_eval.py eval_config.json
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import gzip
import traceback
from datetime import datetime

# --- Start of Integrated Imports ---

# Add paths for local imports
current_dir = Path(__file__).parent.absolute()
# Assuming run_eval.py is in a structure like: project/video/run_eval.py
# and main.py is in project/video/
# and main_workflow.py is in project/preprocess/
project_root = current_dir.parent
preprocess_root = project_root.parent / "preprocess"
video_root = project_root

sys.path.insert(0, str(preprocess_root))
sys.path.insert(0, str(video_root))

# Imports from main_workflow.py
try:
    from main_workflow import WorkflowOrchestrator
except ImportError as e:
    print(f"Fatal: Could not import WorkflowOrchestrator from main_workflow.py: {e}")
    print("Please ensure main_workflow.py is in the correct path (preprocess/).")
    sys.exit(1)

# Imports from main.py
try:
    from src.simulator import HabitatSimulator
    from src.video_composer import VideoComposer
    from src.action_processor import ActionProcessor
    from src.map_builder import OccupancyMapBuilder
    from src.utils import (
        load_json_config,
        write_json_report,
        generate_output_paths,
        validate_config,
        initialize_gpu,
        clear_gpu_cache
    )
except ImportError as e:
    print(f"Fatal: Could not import modules for video generation: {e}")
    print("Please ensure the src/ directory and its contents are in the video project root.")
    sys.exit(1)

# Imports for evaluation part
try:
    import habitat_sim
    from habitat.utils.visualizations import maps
    from habitat_sim import Simulator as Sim
    from habitat.tasks.nav.nav import TopDownMap
except ImportError as e:
    print(f"Warning: Could not import habitat modules: {e}")
    print("Some evaluation features may not work properly.")

# --- End of Integrated Imports ---


class EpisodeEvaluator:
    """Main evaluation orchestrator for single episode navigation tasks."""

    def __init__(self, config_path: str):
        """Initialize evaluator with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        # self.project_root = Path(__file__).parent.parent.absolute()
        self.project_root = video_root
        self.preprocess_dir = preprocess_root

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
            if episode_json_path.endswith('.gz'):
                with gzip.open(episode_json_path, 'rt', encoding='utf-8') as f:
                    episode_data = json.load(f)
            else:
                with open(episode_json_path, 'r', encoding='utf-8') as f:
                    episode_data = json.load(f)

            target_episode = next((ep for ep in episode_data['episodes'] if str(ep['episode_id']) == str(episode_id)), None)

            if target_episode is None:
                raise ValueError(f"Episode {episode_id} not found in {episode_json_path}")

            object_category = target_episode['object_category']
            goals_key = next((key for key in episode_data['goals_by_category'] if key.endswith(object_category) or key == object_category), None)

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
            preprocess_config = {
                "scene_config": {
                    "scene_path": self.config['scene']['scene_file'],
                    "target_floor": self.config['preprocess']['scene_config']['target_floor'],
                    "target_coordinate": episode_data['start_position'],
                    "goal_object": episode_data['object_category'],
                    "rotation": episode_data['start_rotation'],
                    "custom_ortho_scale": self.config['preprocess']['scene_config']['custom_ortho_scale'],
                    "target_coverage": self.config['preprocess']['scene_config']['target_coverage'],
                    "draw_coordinates": self.config['preprocess']['scene_config']['draw_coordinates']
                },
                "room_segmentation": self.config['preprocess']['room_segmentation'],
                "graph_generation": self.config['preprocess']['graph_generation'],
                "llm_config": self.config['preprocess']['llm_config'],
                "prompts": {
                    "choose_room_prompt": str(self.preprocess_dir / "prompts" / "choose_a_room.txt"),
                    "choose_node_prompt": str(self.preprocess_dir / "prompts" / "choose_a_node.txt")
                },
                "output": {
                    "output_dir": str(self.output_dir / "preprocess")
                }
            }

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
        """Run preprocessing pipeline by direct import."""
        original_cwd = os.getcwd()
        try:
            print("\n" + "="*60)
            print("RUNNING PREPROCESSING PIPELINE (IN-PROCESS)")
            print("="*60)

            # Preprocessing scripts often use relative paths, so we change directory
            os.chdir(self.preprocess_dir)

            orchestrator = WorkflowOrchestrator(str(preprocess_config_path))
            success = orchestrator.run_workflow()

            if success:
                print("✓ Preprocessing completed successfully")
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
                error_msg = "Preprocessing workflow failed."
                if orchestrator.output_data.get("errors"):
                    error_msg += f" Last error: {orchestrator.output_data['errors'][-1]}"
                self.results["errors"].append(error_msg)
                print(f"✗ {error_msg}")
                return False

        except Exception as e:
            error_msg = f"Failed to run preprocessing: {e}"
            self.results["errors"].append(error_msg)
            print(f"✗ {error_msg}")
            traceback.print_exc()
            return False
        finally:
            os.chdir(original_cwd)

    def _create_video_config(self, episode_data: Dict[str, Any]) -> Tuple[Path, Path]:
        """Create video generation configuration and action files."""
        try:
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

            video_config_path = self.output_dir / "video_config.json"
            with open(video_config_path, 'w', encoding='utf-8') as f:
                json.dump(video_config, f, indent=2, ensure_ascii=False)

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
    
    def run_video_generation_from_args(self, args: Dict[str, Any]) -> bool:
        """
        Executes the video generation logic from main.py, adapted to take a dict.
        """
        simulator = None
        composer = None
        try:
            print("=" * 60)
            print("Habitat视频生成器 - 重构版本 (In-Process Call)")
            print("=" * 60)
            
            # 1. Load config and actions
            print("1. 加载配置文件...")
            config = load_json_config(args['config'])
            
            print("1.5. 初始化GPU设置...")
            initialize_gpu(config)
            
            if not validate_config(config):
                raise ValueError("配置文件验证失败")
            
            print("2. 加载动作序列...")
            actions = load_json_config(args['actions'])
            
            if 'action' in actions:
                print("检测到新的动作序列格式（包含target参数）")
                action_sequences = [actions['action'][0]]
                print(f"测试第一个action对象，目标: {action_sequences[0].get('target', 'None')}")
            else:
                print("使用旧的动作序列格式")
                action_sequences = [{'sequence': actions['sequence']}]
            
            if args.get('output_dir'):
                config['output_dir'] = args['output_dir']
            
            # 3. Generate paths
            print("3. 生成输出路径...")
            paths = generate_output_paths(config['output_dir'])
            print(f"   视频输出: {paths['video']}")
            print(f"   报告输出: {paths['report']}")
            
            # 4. Initialize components
            print("4. 初始化模拟器...")
            simulator = HabitatSimulator(config)
            
            print("5. 设置场景和智能体...")
            agent_state = actions.get('agent_state')
            initial_state = actions.get('initial_state')
            if agent_state is None and initial_state is None:
                raise ValueError("必须提供 'agent_state' 或 'initial_state'")
            if initial_state is None:
                initial_state = {"position": [0.0, 0.0], "rotation": 0.0}
                print("警告: 没有提供initial_state，使用默认值")
            simulator.setup_scene_and_agent(initial_state, agent_state)
            
            print("6. 初始化视频合成器...")
            composer = VideoComposer(simulator, config, paths['video'])
            
            print("7. 初始化占用地图构建器...")
            use_gpu = config.get('gpu', {}).get('enabled', False)
            map_builder = OccupancyMapBuilder(use_gpu=use_gpu, config=config)
            composer.set_map_builder(map_builder, config)
            
            print("8. 初始化动作处理器...")
            processor = ActionProcessor(simulator, composer, config, map_builder)
            
            print("9. 添加初始帧...")
            composer.add_frame()
            
            # 10. Execute actions
            print("10. 执行动作序列...")
            start_time = datetime.now()
            all_completed_actions = []
            all_collision_action = None
            
            for i, action_data in enumerate(action_sequences):
                print(f"\n执行动作组 {i+1}/{len(action_sequences)}")
                report_data = processor.execute_sequence(action_data)
                all_completed_actions.extend(report_data['completed_actions'])
                if report_data['collision_action']:
                    all_collision_action = report_data['collision_action']
                    break
                if report_data.get('target_found', False):
                    print("目标物体已找到，停止执行")
                    break
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # 11. Generate report
            print("11. 生成执行报告...")
            final_state = simulator.get_robot_state()
            execution_stats = processor.get_execution_stats()
            full_report = {
                'execution_info': {
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'execution_time_seconds': execution_time,
                    'total_frames': execution_stats['total_frames'],
                    'video_duration_seconds': execution_stats['total_duration']
                },
                'config': config,
                'final_agent_state': {
                    'position': final_state['position'].tolist(),
                    'rotation': final_state['rotation'].tolist()
                },
                'original_sequence': action_sequences,
                'completed_sequence': all_completed_actions,
                'collision_at_action': all_collision_action,
                'execution_stats': execution_stats
            }
            write_json_report(paths['report'], full_report)
            
            # 12. Summary
            print("\n" + "=" * 60)
            print("执行完成!")
            print(f"  视频: {paths['video']}\n  报告: {paths['report']}")
            return True
            
        except Exception as e:
            print(f"\n执行过程中发生错误: {e}")
            traceback.print_exc()
            return False
            
        finally:
            print("\n清理资源...")
            if composer:
                composer.save_and_close()
            if simulator:
                simulator.close()
            clear_gpu_cache()
            print("清理完成")


    def _run_video_generation(self, video_config_path: Path, video_action_path: Path) -> bool:
        """Run video generation by direct import."""
        original_cwd = os.getcwd()
        try:
            print("\n" + "="*60)
            print("RUNNING VIDEO GENERATION (IN-PROCESS)")
            print("="*60)

            # The video generation script also uses relative paths
            os.chdir(self.project_root)

            # Prepare arguments as a dictionary
            args = {
                'config': str(video_config_path),
                'actions': str(video_action_path),
                'output_dir': None, # Let config file decide, can be overridden
                'verbose': False, # Can be set to True for more debug info
                'show_histogram': False # from --no-histogram flag
            }

            success = self.run_video_generation_from_args(args)

            if success:
                print("✓ Video generation completed successfully")
                video_dir = self.output_dir / "video"
                video_files = list(video_dir.glob("*.mp4"))
                if video_files:
                    print(f"✓ Video file generated: {video_files[0]}")
                    shutil.copy2(video_files[0], self.output_dir / "output.mp4")
                    return True
                else:
                    error_msg = "Video generation completed but no video file found"
                    self.results["errors"].append(error_msg)
                    print(f"✗ {error_msg}")
                    return False
            else:
                error_msg = "Video generation failed."
                self.results["errors"].append(error_msg)
                print(f"✗ {error_msg}")
                return False

        except Exception as e:
            error_msg = f"Failed to run video generation: {e}"
            self.results["errors"].append(error_msg)
            print(f"✗ {error_msg}")
            traceback.print_exc()
            return False
        finally:
            os.chdir(original_cwd)

    def _evaluate_navigation_success(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate navigation success based on distance to target viewpoints."""
        try:
            print("\n" + "="*60)
            print("EVALUATING NAVIGATION SUCCESS")
            print("="*60)

            video_report_path = self.output_dir / "video" / "execution_report.json"
            if not video_report_path.exists():
                raise FileNotFoundError(f"Video report not found: {video_report_path}")

            with open(video_report_path, 'r') as f:
                video_report = json.load(f)

            final_position = np.array(video_report['final_agent_state']['position'])
            start_position = np.array(episode_data['start_position'])

            view_points = [np.array(vp['agent_state']['position']) for goal in episode_data['goals'] for vp in goal.get('view_points', [])]
            if not view_points:
                print("⚠ No viewpoints found in goals, using goal positions instead")
                view_points = [np.array(goal['position']) for goal in episode_data['goals'] if 'position' in goal]
            if not view_points:
                raise ValueError("No target viewpoints or positions found in goals")

            print(f"Final agent position: {final_position}")
            print(f"Number of target viewpoints: {len(view_points)}")

            distances = [np.linalg.norm(final_position - vp) for vp in view_points]
            min_distance = min(distances) if distances else float('inf')

            success_threshold = self.config['evaluation']['success_distance_threshold']
            success = min_distance <= success_threshold

            episode_distance = video_report['execution_info'].get('total_distance', np.linalg.norm(final_position - start_position))
            
            # Simplified SPL - proper SPL requires a pathfinder for optimal_distance
            optimal_distance = np.linalg.norm(view_points[np.argmin(distances)] - start_position) if distances else float('inf')
            spl = (optimal_distance / max(optimal_distance, episode_distance)) if success and optimal_distance > 0 else 0.0

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
            return {"success": False, "error": error_msg, "min_distance_to_target": float('inf'), "spl": 0.0}

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
            print("\n" + "🚀" + "="*58 + "🚀")
            print("🎯 STARTING AUTOMATED EPISODE EVALUATION 🎯")
            print("🚀" + "="*58 + "🚀")
            print(f"Episode: {self.config['episode']['episode_id']}")
            print(f"Output directory: {self.output_dir}")

            print("\n1. Loading episode data...")
            episode_data = self._load_episode_data()
            self.results["episode_data"] = episode_data

            print("\n2. Running preprocessing...")
            preprocess_config_path = self._create_preprocess_config(episode_data)
            self.results["preprocessing_success"] = self._run_preprocessing(preprocess_config_path)
            if not self.results["preprocessing_success"]:
                print("✗ Preprocessing failed, stopping evaluation")
                return False

            print("\n3. Running video generation...")
            video_config_path, video_action_path = self._create_video_config(episode_data)
            self.results["video_generation_success"] = self._run_video_generation(video_config_path, video_action_path)
            if not self.results["video_generation_success"]:
                print("✗ Video generation failed, stopping evaluation")
                return False

            print("\n4. Evaluating navigation success...")
            self.results["evaluation_results"] = self._evaluate_navigation_success(episode_data)

            print("\n5. Saving results...")
            self._save_results()

            print("\n" + "🎉" + "="*58 + "🎉")
            print("🎯 EVALUATION COMPLETED SUCCESSFULLY! 🎯")
            print("🎉" + "="*58 + "🎉")

            eval_res = self.results["evaluation_results"]
            print(f"\nEvaluation Summary:")
            print(f"  Episode ID: {self.config['episode']['episode_id']}")
            print(f"  Object Category: {episode_data['object_category']}")
            print(f"  Success: {'✓' if eval_res.get('success') else '✗'}")
            print(f"  Distance to target: {eval_res.get('min_distance_to_target', float('inf')):.3f}m")
            print(f"  SPL: {eval_res.get('spl', 0.0):.3f}")
            print(f"\nOutput files:")
            print(f"  Video: {self.output_dir}/output.mp4")
            print(f"  Results: {self.output_dir}/output.json")
            return True

        except Exception as e:
            error_msg = f"Evaluation failed with error: {e}"
            print(f"\n✗ {error_msg}")
            self.results["errors"].append(error_msg)
            if "--verbose" in sys.argv:
                traceback.print_exc()
            return False
        finally:
            self._save_results()

def main():
    """Main entry point."""
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print("Usage: python run_eval.py <eval_config.json> [--verbose]")
        print("\nExample:")
        print("  python run_eval.py eval_config_template.json")
        print("\nThe script will automatically:")
        print("  1. Extract episode data and create preprocessing config")
        print("  2. Run preprocessing to generate actions (in-process)")
        print("  3. Run video generation with the actions (in-process)")
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
        print("\n\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error during evaluation setup: {e}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()