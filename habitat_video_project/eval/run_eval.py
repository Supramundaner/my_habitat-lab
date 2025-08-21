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
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import gzip
import traceback
from datetime import datetime

# Configure Python logging
import logging
logging.getLogger("habitat").setLevel(logging.ERROR)
logging.getLogger("habitat_sim").setLevel(logging.ERROR)

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
        # Additional logging suppression
        logging.getLogger().setLevel(logging.ERROR)
        logging.getLogger("habitat").setLevel(logging.ERROR)
        logging.getLogger("habitat_sim").setLevel(logging.ERROR)
        logging.getLogger("magnum").setLevel(logging.ERROR)
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.project_root = project_root
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
            opener = gzip.open if episode_json_path.endswith('.gz') else open
            with opener(episode_json_path, 'rt', encoding='utf-8') as f:
                episode_data = json.load(f)

            target_episode = next((ep for ep in episode_data['episodes'] if str(ep['episode_id']) == str(episode_id)), None)
            if target_episode is None:
                raise ValueError(f"Episode {episode_id} not found in {episode_json_path}")

            object_category = target_episode['object_category']
            goals_key = next((key for key in episode_data.get('goals_by_category', {}) if key.endswith(object_category) or key == object_category), None)
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
            os.chdir(self.preprocess_dir)
            orchestrator = WorkflowOrchestrator(str(preprocess_config_path))
            success = orchestrator.run_workflow()
            if success:
                print("✓ Preprocessing completed successfully")
                return True
            else:
                error_msg = "Preprocessing workflow failed."
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

    def _create_video_generation_assets(self) -> Dict[str, Path]:
        """Creates config and identifies paths needed for video generation."""
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
            wall_mask_path = self.output_dir / "preprocess" / "wall_mask.png"

            if not preprocess_action_path.exists():
                raise FileNotFoundError(f"Action file not found: {preprocess_action_path}")
            if not wall_mask_path.exists():
                raise FileNotFoundError(f"Wall mask file not found: {wall_mask_path}")

            print(f"✓ Created video config: {video_config_path}")
            print(f"✓ Found action file: {preprocess_action_path}")
            print(f"✓ Found wall mask: {wall_mask_path}")

            return {
                "config_path": video_config_path,
                "actions_path": preprocess_action_path,
                "wall_mask_path": wall_mask_path
            }
        except Exception as e:
            error_msg = f"Failed to create video generation assets: {e}"
            self.results["errors"].append(error_msg)
            raise RuntimeError(error_msg)

    def _run_video_generation_in_process(self, args: Dict[str, Any]) -> bool:
        """
        Executes the video generation logic from main.py, adapted to take a dict.
        This is a direct copy and adaptation of the main() function from the new main.py.
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
            
            # THIS IS THE KEY CHANGE: Handling the new action format
            if 'target_info' in actions:
                print("检测到新的动作序列格式（包含target_info和wall_mask）")
                wall_mask_path = args.get('wall_mask')
                # Create a simplified action_sequences structure for main.py's loop
                action_sequences = [{
                    'target_info': actions['target_info'],
                    'wall_mask_path': wall_mask_path
                }]
                actions['wall_mask_path'] = wall_mask_path
            else:
                # Fallback for old formats, though not expected in this workflow
                print("警告：未检测到 'target_info'，尝试旧格式")
                if 'action' in actions:
                    action_sequences = [actions['action'][0]]
                else:
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
            simulator.setup_scene_and_agent(initial_state, agent_state)
            
            print("6. 初始化视频合成器...")
            composer = VideoComposer(simulator, config, paths['video'])
            
            print("7. 初始化占用地图构建器...")
            use_gpu = config.get('gpu', {}).get('enabled', False)
            map_builder = OccupancyMapBuilder(use_gpu=use_gpu, config=config)
            composer.set_map_builder(map_builder, config)
            
            # 7.5 Initialize map from wall mask
            wall_mask_path = args.get('wall_mask')
            if wall_mask_path and os.path.exists(wall_mask_path):
                print(f"7.5. 使用 wall mask 初始化占用地图: {wall_mask_path}")
                map_builder.initialize_from_wall_mask(wall_mask_path)
            else:
                print(f"7.5. 未提供或未找到 wall mask，使用默认的空地图")
            
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
                'original_actions_input': actions,
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

    def _run_video_generation(self, assets: Dict[str, Path]) -> bool:
        """Run video generation by calling the in-process function."""
        original_cwd = os.getcwd()
        try:
            print("\n" + "="*60)
            print("RUNNING VIDEO GENERATION (IN-PROCESS)")
            print("="*60)
            os.chdir(self.project_root)

            args = {
                'config': str(assets['config_path']),
                'actions': str(assets['actions_path']),
                'wall_mask': str(assets['wall_mask_path']), # Crucial new argument
                'output_dir': None,
                'verbose': "--verbose" in sys.argv
            }
            success = self._run_video_generation_in_process(args)

            if success:
                print("✓ Video generation completed successfully")
                return True
            else:
                error_msg = "Video generation failed."
                self.results["errors"].append(error_msg)
                print(f"✗ {error_msg}")
                return False
        except Exception as e:
            error_msg = f"Failed to run video generation: {e}"
            self.results["errors"].append(error_msg)
            traceback.print_exc()
            return False
        finally:
            os.chdir(original_cwd)

    def _evaluate_navigation_success(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate navigation success based on geodesic distance to target viewpoints.
        (This part remains unchanged as per the instructions).
        """
        if not habitat_sim or not np:
            print("✗ Evaluation skipped: habitat_sim or numpy not installed.")
            return {"sr": -1.0, "spl": -1.0, "success": False, "error": "Missing dependencies"}
        
        sim = None
        try:
            print("\n" + "="*60)
            print("EVALUATING NAVIGATION SUCCESS (GEODESIC METHOD)")
            print("="*60)

            video_report_path = self.output_dir / "video" / "execution_report.json"
            if not video_report_path.exists():
                raise FileNotFoundError(f"Video report not found: {video_report_path}")

            with open(video_report_path, 'r') as f:
                video_report = json.load(f)

            final_position = np.array(video_report['final_agent_state']['position'])
            start_position = np.array(episode_data['start_position'])
            episode_cum_distance = video_report['execution_stats'].get('total_distance', np.linalg.norm(final_position - start_position))
            
            view_points = [np.array(vp['agent_state']['position']) for goal in episode_data['goals'] for vp in goal.get('view_points', [])]
            if not view_points:
                print("⚠ No viewpoints found in goals, using goal positions instead")
                view_points = [np.array(goal['position']) for goal in episode_data['goals'] if 'position' in goal]
            if not view_points:
                raise ValueError("No target viewpoints or positions found in goals")
            
            scene_file = self.config['scene']['scene_file']
            success_threshold = self.config['evaluation'].get('success_distance_threshold', 0.25)

            # Suppress additional logging before creating simulator
            logging.getLogger().setLevel(logging.CRITICAL)
            
            sim_settings = {"scene": scene_file, "default_agent": 0, "sensor_height": 1.5, "width": 128, "height": 128}
            sim_cfg = habitat_sim.SimulatorConfiguration()
            sim_cfg.scene_id = sim_settings["scene"]
            agent_cfg = habitat_sim.AgentConfiguration()
            cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
            sim = habitat_sim.Simulator(cfg)
            path_finder = sim.pathfinder
            
            path_start = habitat_sim.MultiGoalShortestPath()
            path_start.requested_start = start_position
            path_start.requested_ends = view_points
            path_finder.find_path(path_start)
            start_end_geo_distance = path_start.geodesic_distance

            path_agent = habitat_sim.MultiGoalShortestPath()
            path_agent.requested_start = final_position
            path_agent.requested_ends = view_points
            path_finder.find_path(path_agent)
            agent_end_geo_distance = path_agent.geodesic_distance
                
            sr = 1.0 if agent_end_geo_distance <= success_threshold and agent_end_geo_distance != np.inf else 0.0
            spl = 0.0
            if sr > 0:
                spl = start_end_geo_distance / max(start_end_geo_distance, episode_cum_distance)

            evaluation_results = {
                "sr": sr, "spl": spl, "success": bool(sr),
                "geodesic_distance_to_target": agent_end_geo_distance,
                "optimal_geodesic_distance": start_end_geo_distance,
                "path_length": episode_cum_distance,
                "success_threshold": success_threshold
            }
            print(f"✓ Evaluation Complete: SR={sr}, SPL={spl:.3f}, Dist={agent_end_geo_distance:.3f}m")
            return evaluation_results
        except Exception as e:
            error_msg = f"Failed to evaluate navigation success: {e}"
            self.results["errors"].append(error_msg)
            print(f"✗ {error_msg}")
            traceback.print_exc()
            return {"sr": 0.0, "spl": 0.0, "success": False, "error": error_msg}
        finally:
            if sim:
                sim.close()

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
            print("🎯 STARTING AUTOMATED EPISODE EVALUATION (REFACTORED) 🎯")
            print("🚀" + "="*58 + "🚀")

            print("\n1. Loading episode data...")
            episode_data = self._load_episode_data()
            self.results["episode_data"] = episode_data

            print("\n2. Running preprocessing...")
            preprocess_config_path = self._create_preprocess_config(episode_data)
            self.results["preprocessing_success"] = self._run_preprocessing(preprocess_config_path)
            if not self.results["preprocessing_success"]:
                print("✗ Preprocessing failed, stopping evaluation.")
                return False

            print("\n3. Running video generation...")
            video_assets = self._create_video_generation_assets()
            self.results["video_generation_success"] = self._run_video_generation(video_assets)
            if not self.results["video_generation_success"]:
                print("✗ Video generation failed, stopping evaluation.")
                return False

            print("\n4. Evaluating navigation success...")
            self.results["evaluation_results"] = self._evaluate_navigation_success(episode_data)

            print("\n" + "🎉" + "="*58 + "🎉")
            print("🎯 EVALUATION COMPLETED SUCCESSFULLY! 🎯")
            print("🎉" + "="*58 + "🎉")

            eval_res = self.results["evaluation_results"]
            print(f"\nEvaluation Summary:")
            print(f"  Success (SR): {'✓' if eval_res.get('sr', 0.0) > 0 else '✗'} ({eval_res.get('sr', 0.0):.1f})")
            print(f"  SPL: {eval_res.get('spl', 0.0):.3f}")
            print(f"  Output: {self.output_dir}")
            return True

        except Exception as e:
            error_msg = f"Evaluation failed with error: {e}"
            print(f"\n✗ {error_msg}")
            self.results["errors"].append(error_msg)
            traceback.print_exc()
            return False
        finally:
            self._save_results()

def main():
    """Main entry point."""
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print("Usage: python run_eval.py <eval_config.json> [--verbose]")
        sys.exit(1)

    config_path = sys.argv[1]
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    try:
        evaluator = EpisodeEvaluator(config_path)
        success = evaluator.run_evaluation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nFatal error during evaluation setup: {e}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()