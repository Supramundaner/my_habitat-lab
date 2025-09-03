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
import importlib.util
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
# Current file: habitat-lab/habitat_video_project/instextnav_eval/run_eval.py
# Target: habitat-lab/instextnav_preprocess/main.py
project_root = current_dir.parent  # habitat_video_project
habitat_lab_root = project_root.parent  # habitat-lab
instextnav_preprocess_root = habitat_lab_root / "instextnav_preprocess"
video_root = project_root

# Insert instextnav_preprocess path FIRST to ensure we import from the correct main.py
sys.path.insert(0, str(instextnav_preprocess_root))
sys.path.insert(1, str(video_root))

# Imports from main.py (instextnav_preprocess)
try:
    # First check if the main.py file exists
    main_py_path = instextnav_preprocess_root / "main.py"
    if not main_py_path.exists():
        raise ImportError(f"main.py not found at {main_py_path}")
    
    print(f"✓ Found main.py at: {main_py_path}")
    
    # Try to import the specific module using importlib
    import importlib.util
    spec = importlib.util.spec_from_file_location("instextnav_main", str(main_py_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {main_py_path}")
    
    instextnav_main = importlib.util.module_from_spec(spec)
    
    # Add to sys.modules to avoid import conflicts
    sys.modules["instextnav_main"] = instextnav_main
    
    # Execute the module
    spec.loader.exec_module(instextnav_main)
    
    # Get TextNavigationOrchestrator class (TextNav only)
    if hasattr(instextnav_main, 'TextNavigationOrchestrator'):
        NavigationOrchestrator = instextnav_main.TextNavigationOrchestrator
        print("✓ Successfully imported TextNavigationOrchestrator from instextnav_preprocess/main.py")
    else:
        available_classes = [attr for attr in dir(instextnav_main) if not attr.startswith('_')]
        raise ImportError(f"TextNavigationOrchestrator not found in main.py. Available classes: {available_classes}")
    
except Exception as e:
    print(f"Fatal: Could not import navigation orchestrator from main.py: {e}")
    print(f"Checked path: {main_py_path}")
    print(f"InstExtNav preprocess directory contents:")
    if instextnav_preprocess_root.exists():
        for item in instextnav_preprocess_root.iterdir():
            print(f"  - {item.name}")
    else:
        print(f"  Directory does not exist: {instextnav_preprocess_root}")
    print("Please ensure main.py is in the correct path (instextnav_preprocess/) and contains TextNavigationOrchestrator class.")
    traceback.print_exc()
    sys.exit(1)

# Imports from main.py (video generation)
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
    """Main evaluation orchestrator for single episode image instance navigation tasks."""

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
        self.instextnav_preprocess_dir = instextnav_preprocess_root

        # Extract scene identifier for output directory
        episode_json_path = Path(self.config['episode']['episode_json_path'])
        self.scene_id = episode_json_path.stem

        # Setup output directory (always save to instextnav_eval)
        if 'output_dir' in self.config:
            self.output_dir = Path(self.config['output_dir'])
        else:
            # Use instextnav_eval instead of instextnav_eval
            self.output_dir = self.project_root / "instextnav_eval" / "output" / self.scene_id
        
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ Output directory created/verified: {self.output_dir}")
        except Exception as e:
            print(f"✗ Failed to create output directory {self.output_dir}: {e}")
            raise RuntimeError(f"Cannot create output directory: {e}")

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
                config = json.load(f)
            return config
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration from {self.config_path}: {e}")

    def _load_episode_data(self) -> Dict[str, Any]:
        """Load and extract episode data from JSON file with support for new dataset structure."""
        episode_json_path = self.config['episode']['episode_json_path']
        episode_id = self.config['episode']['episode_id']
        scene_file = self.config['scene']['scene_file']

        try:
            # Extract scene name from scene file path
            scene_name = self._extract_scene_name(scene_file)
            print(f"🔍 Extracted scene name: {scene_name}")
            
            # Try the new dataset structure first
            if self._try_load_from_new_structure(episode_json_path, episode_id, scene_name):
                return self.episode_info
            
            # Fallback to old structure
            print("📂 Trying legacy dataset structure...")
            return self._load_from_legacy_structure(episode_json_path, episode_id)
            
        except Exception as e:
            error_msg = f"Failed to load episode data: {e}"
            self.results["errors"].append(error_msg)
            raise RuntimeError(error_msg)

    def _extract_scene_name(self, scene_file: str) -> str:
        """Extract scene name from scene file path."""
        # Example: /path/to/4ok3usBNeis.basis.glb -> 4ok3usBNeis
        scene_filename = Path(scene_file).name
        if scene_filename.endswith('.basis.glb'):
            return scene_filename.replace('.basis.glb', '')
        elif scene_filename.endswith('.glb'):
            return scene_filename.replace('.glb', '')
        else:
            # Try to extract from directory structure
            # e.g., "/path/00877-4ok3usBNeis/4ok3usBNeis.basis.glb"
            parts = scene_file.split('/')
            for part in parts:
                if '-' in part and len(part.split('-')) == 2:
                    return part.split('-')[1]
            return scene_filename

    def _try_load_from_new_structure(self, base_path: str, episode_id: str, scene_name: str) -> bool:
        """Try to load episode data from new dataset structure."""
        try:
            # New structure: /base_path/content/{scene_name}.json
            dataset_dir = Path(base_path).parent if Path(base_path).is_file() else Path(base_path)
            scene_json_path = dataset_dir / "content" / f"{scene_name}.json"
            
            print(f"🔍 Trying new structure: {scene_json_path}")
            
            if not scene_json_path.exists():
                print(f"⚠️ Scene file not found: {scene_json_path}")
                return False
            
            # Load scene-specific episode data
            with open(scene_json_path, 'r', encoding='utf-8') as f:
                episode_data = json.load(f)
            
            print(f"✓ Loaded scene data from: {scene_json_path}")
            
            # Find the target episode
            target_episode = next((ep for ep in episode_data['episodes'] if str(ep['episode_id']) == str(episode_id)), None)
            if target_episode is None:
                print(f"⚠️ Episode {episode_id} not found in {scene_json_path}")
                return False

            # Extract episode information
            goal_object_id = target_episode['goal_object_id']
            goal_image_id = target_episode['goal_image_id']
            object_category = target_episode['object_category']
            
            # Find the corresponding goal by object_id
            goal_key = None
            for key, goal in episode_data.get('goals', {}).items():
                if str(goal['object_id']) == str(goal_object_id):
                    goal_key = key
                    break
            
            if goal_key is None:
                raise ValueError(f"Goal object with ID {goal_object_id} not found in goals data")

            goal_data = episode_data['goals'][goal_key]
            
            # Get the specific image goal
            if goal_image_id >= len(goal_data['image_goals']):
                raise ValueError(f"Image goal with ID {goal_image_id} not found for object {goal_object_id}")
            
            image_goal = goal_data['image_goals'][goal_image_id]

            self.episode_info = {
                "episode": target_episode,
                "goal_data": goal_data,
                "image_goal": image_goal,
                "object_category": object_category,
                "start_position": target_episode['start_position'],
                "start_rotation": target_episode['start_rotation'],
                "scene_id": target_episode['scene_id'],
                "goal_object_id": goal_object_id,
                "goal_image_id": goal_image_id
            }

            print(f"✓ Loaded episode {episode_id} from new structure")
            print(f"  Object category: {object_category}")
            print(f"  Goal object: {goal_data['object_name']} (ID: {goal_object_id})")
            print(f"  Goal image ID: {goal_image_id}")
            print(f"  Start position: {target_episode['start_position']}")
            print(f"  Number of viewpoints: {len(goal_data.get('view_points', []))}")
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to load from new structure: {e}")
            return False

    def _load_from_legacy_structure(self, episode_json_path: str, episode_id: str) -> Dict[str, Any]:
        """Load episode data from legacy dataset structure."""
        opener = gzip.open if episode_json_path.endswith('.gz') else open
        with opener(episode_json_path, 'rt', encoding='utf-8') as f:
            episode_data = json.load(f)

        target_episode = next((ep for ep in episode_data['episodes'] if str(ep['episode_id']) == str(episode_id)), None)
        if target_episode is None:
            raise ValueError(f"Episode with ID {episode_id} not found in {episode_json_path}")

        # For image instance navigation, we need to get the specific goal object and image
        goal_object_id = target_episode['goal_object_id']
        goal_image_id = target_episode['goal_image_id']
        object_category = target_episode['object_category']
        
        # Find the corresponding goal by object_id
        goal_key = None
        for key, goal in episode_data.get('goals', {}).items():
            if str(goal['object_id']) == str(goal_object_id):
                goal_key = key
                break
        
        if goal_key is None:
            raise ValueError(f"Goal object with ID {goal_object_id} not found in goals data")

        goal_data = episode_data['goals'][goal_key]
        
        # Get the specific image goal
        if goal_image_id >= len(goal_data['image_goals']):
            raise ValueError(f"Image goal with ID {goal_image_id} not found for object {goal_object_id}")
        
        image_goal = goal_data['image_goals'][goal_image_id]

        episode_info = {
            "episode": target_episode,
            "goal_data": goal_data,
            "image_goal": image_goal,
            "object_category": object_category,
            "start_position": target_episode['start_position'],
            "start_rotation": target_episode['start_rotation'],
            "scene_id": target_episode['scene_id'],
            "goal_object_id": goal_object_id,
            "goal_image_id": goal_image_id
        }

        print(f"✓ Loaded episode {episode_id} from legacy structure")
        print(f"  Object category: {object_category}")
        print(f"  Goal object: {goal_data['object_name']} (ID: {goal_object_id})")
        print(f"  Goal image ID: {goal_image_id}")
        print(f"  Start position: {target_episode['start_position']}")
        print(f"  Number of viewpoints: {len(goal_data.get('view_points', []))}")
        return episode_info

    def _create_preprocess_config(self, episode_data: Dict[str, Any]) -> Path:
        """Create preprocessing configuration file for instextnav_preprocess."""
        try:
            # Create instextnav_preprocess compatible config
            instextnav_config = {
                "scene_config": {
                    "scene_path": self.config['scene']['scene_file'],
                    "episodes_file": episode_data['episode']['scene_id'].replace('.basis.glb', ''),
                    "episode_id": int(episode_data['episode']['episode_id']),
                    "custom_ortho_scale": self.config['preprocess']['scene_config'].get('custom_ortho_scale'),
                    "target_coverage": self.config['preprocess']['scene_config'].get('target_coverage', 0.9),
                    "draw_coordinates": self.config['preprocess']['scene_config'].get('draw_coordinates', False),
                    "use_text_nav": True  # Always use TextNav in this evaluation system
                },
                "val_text_path": self.config['preprocess'].get('val_text_path', '/home/yaoaa/habitat-lab/data/datasets/instancenav/val/val_text.json'),
                "output": {
                    "output_dir": str(self.output_dir / "instextnav_preprocess_output")
                },
                "resolution": self.config['preprocess']['resolution'],
                "room_segmentation": self.config['preprocess']['room_segmentation'],
                "graph_generation": self.config['preprocess']['graph_generation'],
                "llm_config": self.config['preprocess']['llm_config'],
                "prompts": self.config['preprocess']['prompts']
            }
            
            # Fix episodes_file to point to the actual scene-specific JSON file
            scene_name = self._extract_scene_name(self.config['scene']['scene_file'])
            episode_json_base = self.config['episode']['episode_json_path']
            
            # Construct the path to the scene-specific JSON file
            if episode_json_base.endswith('/'):
                scene_json_path = os.path.join(episode_json_base, 'content', f'{scene_name}.json')
            else:
                # Legacy single file path
                scene_json_path = episode_json_base
            
            instextnav_config["scene_config"]["episodes_file"] = scene_json_path
            print(f"✓ Using episode file: {scene_json_path}")
            
            # Save temporary config file
            temp_config_path = self.output_dir / "temp_instextnav_preprocess_config.json"
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                json.dump(instextnav_config, f, indent=2)
            
            print(f"✓ Created instextnav_preprocess config: {temp_config_path}")
            return temp_config_path
        except Exception as e:
            error_msg = f"Failed to create preprocess config: {e}"
            self.results["errors"].append(error_msg)
            raise RuntimeError(error_msg)

    def _run_preprocessing(self, preprocess_config_path: Path) -> bool:
        """Run preprocessing pipeline by direct import."""
        original_cwd = os.getcwd()
        try:
            os.chdir(self.instextnav_preprocess_dir)
            print(f"🔄 Running instextnav_preprocess from: {self.instextnav_preprocess_dir}")
            
            # Initialize the orchestrator and run workflow
            orchestrator = NavigationOrchestrator(str(preprocess_config_path))
            success = orchestrator.run_workflow()
            
            if success:
                print("✓ InstExtNav preprocessing completed successfully")
                self.results["preprocessing_success"] = True
                
                # Check for action.json output
                action_json_path = Path(orchestrator.output_dir) / "action.json"
                if action_json_path.exists():
                    # Copy action.json to our output directory for video generation
                    dest_action_path = self.output_dir / "action.json"
                    shutil.copy2(action_json_path, dest_action_path)
                    print(f"✓ Action.json copied to: {dest_action_path}")
                else:
                    print("⚠ Warning: action.json not found in preprocessing output")
                
                return True
            else:
                print("✗ InstExtNav preprocessing failed")
                self.results["errors"].append("InstExtNav preprocessing pipeline failed")
                return False
                
        except Exception as e:
            error_msg = f"InstExtNav preprocessing error: {str(e)}"
            print(f"✗ {error_msg}")
            self.results["errors"].append(error_msg)
            return False
        finally:
            os.chdir(original_cwd)

    def _create_video_generation_assets(self) -> Dict[str, Path]:
        """Creates config and identifies paths needed for video generation."""
        try:
            # Check for action.json file from preprocessing
            action_json_path = self.output_dir / "action.json"
            if not action_json_path.exists():
                raise FileNotFoundError(f"action.json not found at {action_json_path}")
            
            # Create video generation config that matches the expected format
            video_config = {
                "scene": {
                    "scene_file": self.config['scene']['scene_file'],
                    "robot_urdf": self.config['scene']['robot_urdf']
                },
                "output_dir": str(self.output_dir / "video_output"),
                **self.config['video_generation']
            }
            
            # Debug: Print GPU config in video_config
            print(f"Debug: Creating video config with GPU: {video_config.get('gpu', {})}")
            
            video_config_path = self.output_dir / "video_generation_config.json"
            with open(video_config_path, 'w', encoding='utf-8') as f:
                json.dump(video_config, f, indent=2)
            
            print(f"✓ Video generation config created: {video_config_path}")
            
            return {
                "config_path": video_config_path,
                "action_json_path": action_json_path,
                "output_dir": Path(video_config['output_dir']),
                "wall_mask_path": self.output_dir / "instextnav_preprocess_output" / "wall_mask.png"
            }
        except Exception as e:
            error_msg = f"Failed to create video generation assets: {e}"
            self.results["errors"].append(error_msg)
            raise RuntimeError(error_msg)

    def _run_video_generation_in_process(self, args: Dict[str, Any]) -> bool:
        """
        Executes the video generation logic adapted from habitat_video_project/main.py.
        """
        simulator = None
        composer = None
        try:
            print("🎬 Starting video generation...")
            
            # Load and validate configuration
            config = load_json_config(args['config_path'])
            
            # Debug GPU configuration
            print(f"Debug: GPU config in video_config: {config.get('gpu', {})}")
            
            # Initialize GPU settings (simplified like eval/run_eval.py)
            print("🔧 初始化GPU设置...")
            initialize_gpu(config)
            
            # Validate config (but handle missing keys gracefully)
            try:
                validate_config(config)
            except Exception as e:
                print(f"Configuration validation warning: {e}")
            
            # Load actions from action.json
            print("🎯 Loading navigation actions...")
            actions_data = load_json_config(args['action_json_path'])
            
            # Handle different action file formats
            if 'action' in actions_data:
                # Old format with action parameter
                action_sequences = [actions_data['action'][0]]
            elif 'target_info' in actions_data:
                # New format with target_info
                action_sequences = [actions_data]
            else:
                # Direct sequence format
                action_sequences = [{'sequence': actions_data.get('sequence', [])}]
            
            # Generate output paths
            output_paths = generate_output_paths(config['output_dir'])
            
            # Initialize components
            print("📋 Initializing simulation components...")
            simulator = HabitatSimulator(config)
            
            # Setup scene and agent
            print("🏠 Setting up scene and agent...")
            agent_state = actions_data.get('agent_state', None)
            initial_state = actions_data.get('initial_state', None)
            
            if agent_state is None and initial_state is None:
                # Create default states if not provided
                initial_state = {
                    "position": [0.0, 0.0],
                    "rotation": 0.0
                }
                print("Warning: Using default initial state")
            
            simulator.setup_scene_and_agent(initial_state, agent_state)
            
            # Initialize video composer
            print("🎬 Initializing video composer...")
            composer = VideoComposer(simulator, config, output_paths['video'])
            
            # Initialize map builder
            print("🗺️ Initializing map builder...")
            use_gpu = config.get('gpu', {}).get('enabled', False)
            map_builder = OccupancyMapBuilder(use_gpu=use_gpu, config=config)
            composer.set_map_builder(map_builder, config)
            
            # Initialize map from wall mask (if available from preprocessing)
            # Check for wall mask in instextnav_preprocess output
            preprocess_output_dir = self.output_dir / "instextnav_preprocess_output"
            wall_mask_path = preprocess_output_dir / "wall_mask.png"
            if wall_mask_path.exists():
                print(f"🗺️ 使用 wall mask 初始化占用地图: {wall_mask_path}")
                map_builder.initialize_from_wall_mask(str(wall_mask_path))
            else:
                print(f"🗺️ 未找到 wall mask，使用默认的空地图")
            
            # Initialize action processor
            print("⚡ Initializing action processor...")
            processor = ActionProcessor(simulator, composer, config, map_builder)
            
            # Add initial frame
            print("📸 Adding initial frame...")
            composer.add_frame()
            
            # Execute action sequences
            print("� Executing navigation actions...")
            all_completed_actions = []
            
            for i, action_data in enumerate(action_sequences):
                print(f"\nExecuting action group {i+1}/{len(action_sequences)}")
                if 'target' in action_data:
                    print(f"Target object: {action_data['target']}")
                
                report_data = processor.execute_sequence(action_data)
                all_completed_actions.extend(report_data['completed_actions'])
                
                if report_data.get('collision_action'):
                    print("Collision detected, stopping execution")
                    break
                
                if report_data.get('target_found', False):
                    print("Target found, stopping execution")
                    break
            
            # Finalize video
            print("🎞️ Finalizing video...")
            composer.save_and_close()
            
            # Generate execution report
            print("📊 Generating execution report...")
            agent_state = simulator.agent.get_state()
            # Convert quaternion properly - habitat-sim quaternion format
            quat = agent_state.rotation
            final_state = {
                'position': agent_state.position.tolist(),
                'rotation': [quat.x, quat.y, quat.z, quat.w]
            }
            
            # Get accurate execution stats from processor (like eval/run_eval.py)
            execution_stats = processor.get_execution_stats()
            
            full_report = {
                'timestamp': datetime.now().isoformat(),
                'video_info': {
                    'path': str(output_paths['video']),
                    'total_frames': execution_stats['total_frames'],
                    'video_duration_seconds': execution_stats['total_duration']
                },
                'config': config,
                'final_agent_state': final_state,
                'original_sequence': action_sequences,
                'completed_sequence': all_completed_actions,
                'execution_stats': execution_stats
            }
            
            write_json_report(output_paths['report'], full_report)
            print(f"📊 Report saved to: {output_paths['report']}")
            
            print(f"✅ Video generation completed successfully!")
            print(f"📹 Video saved to: {output_paths['video']}")
            
            return True
            
        except Exception as e:
            error_msg = f"Video generation failed: {str(e)}"
            print(f"❌ {error_msg}")
            if '--verbose' in sys.argv:
                traceback.print_exc()
            return False
        finally:
            # Cleanup
            if simulator:
                simulator.close()
            if composer and hasattr(composer, 'cleanup'):
                composer.cleanup()
            
            # Clear GPU cache (simplified like eval/run_eval.py)
            clear_gpu_cache()

    def _run_video_generation(self, assets: Dict[str, Path]) -> bool:
        """Run video generation by calling the in-process function."""
        original_cwd = os.getcwd()
        try:
            os.chdir(video_root)
            print(f"🔄 Running video generation from: {video_root}")
            
            # Create arguments dict for video generation
            video_args = {
                'config_path': str(assets['config_path']),
                'action_json_path': str(assets['action_json_path']),
                'output_dir': str(assets['output_dir'])
            }
            
            success = self._run_video_generation_in_process(video_args)
            
            if success:
                print("✓ Video generation completed successfully")
                self.results["video_generation_success"] = True
                return True
            else:
                print("✗ Video generation failed")
                self.results["errors"].append("Video generation pipeline failed")
                return False
                
        except Exception as e:
            error_msg = f"Video generation error: {str(e)}"
            print(f"✗ {error_msg}")
            self.results["errors"].append(error_msg)
            return False
        finally:
            os.chdir(original_cwd)

    def _evaluate_navigation_success(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate navigation success based on geodesic distance to target viewpoints.
        For image instance navigation, the goal is a specific image goal, not any object of the category.
        """
        if not habitat_sim or not np:
            return {"success": False, "sr": 0.0, "spl": 0.0, "error": "Required modules not available"}
        
        sim = None
        try:
            print("EVALUATING NAVIGATION SUCCESS (GEODESIC METHOD)")
            print("="*60)

            # The execution report is saved in the video output directory as execution_report.json
            video_report_path = self.output_dir / "video_output" / "execution_report.json"
            if not video_report_path.exists():
                raise FileNotFoundError(f"Video report not found: {video_report_path}")

            with open(video_report_path, 'r') as f:
                video_report = json.load(f)

            final_position = np.array(video_report['final_agent_state']['position'])
            start_position = np.array(episode_data['start_position'])
            episode_cum_distance = video_report['execution_stats'].get('total_distance', np.linalg.norm(final_position - start_position))
            
            # 调试信息：检查距离计算
            euclidean_distance = np.linalg.norm(final_position - start_position)
            print(f"� Distance Debug Info:")
            print(f"  Euclidean distance (start->end): {euclidean_distance:.3f}m")
            print(f"  Cumulative distance traveled: {episode_cum_distance:.3f}m")
            
            # Get viewpoints for the unique goal object
            goal_data = episode_data['goal_data']
            view_points = [np.array(vp['agent_state']['position']) for vp in goal_data.get('view_points', [])]
            
            if not view_points:
                print("⚠ No viewpoints found in goal_data, using goal positions instead")
                view_points = [np.array(goal_data['position'])] if 'position' in goal_data else []
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
                
            if start_end_geo_distance == np.inf:
                sr = 1
                spl = 1
            elif agent_end_geo_distance == np.inf:
                sr = 0
                spl = 0
            else:
                sr = agent_end_geo_distance <= success_threshold
                spl = sr * start_end_geo_distance / max(start_end_geo_distance, episode_cum_distance)

            evaluation_results = {
                "sr": sr, "spl": spl, "success": bool(sr),
                "geodesic_distance_to_target": agent_end_geo_distance,
                "optimal_geodesic_distance": start_end_geo_distance,
                "path_length": episode_cum_distance,
                "success_threshold": success_threshold,
                "final_position": final_position.tolist(),
                "total_viewpoints_checked": len(view_points)
            }
            
            print(f"✓ Evaluation Complete: SR={sr}, SPL={spl:.3f}, Dist={agent_end_geo_distance:.3f}m")
            return evaluation_results
            
        except Exception as e:
            error_msg = f"Evaluation failed: {str(e)}"
            print(f"✗ {error_msg}")
            return {"success": False, "sr": 0.0, "spl": 0.0, "error": error_msg}
        finally:
            if sim:
                sim.close()
            


    def _save_results(self):
        """Save final evaluation results."""
        try:
            results_path = self.output_dir / "evaluation_results.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2)
            print(f"✓ Results saved to: {results_path}")
            
            # Also save to output.json for consistency
            output_path = self.output_dir / "output.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2)
            print(f"✓ Output saved to: {output_path}")
            
        except Exception as e:
            print(f"✗ Failed to save results: {e}")
            


    def run_evaluation(self) -> bool:
        """Run the complete evaluation pipeline."""
        try:
            print("\n" + "🔍" * 30)
            print("  STARTING IMAGE INSTANCE NAVIGATION EVALUATION")
            print("🔍" * 30)
            
            # Step 1: Load episode data
            print("\n📖 Loading episode data...")
            episode_data = self._load_episode_data()
            self.results["episode_data"] = episode_data
            
            # Step 2: Run preprocessing
            print("\n⚙️ Running instextnav_preprocess pipeline...")
            preprocess_config_path = self._create_preprocess_config(episode_data)
            
            if not self._run_preprocessing(preprocess_config_path):
                print("❌ Preprocessing failed, stopping evaluation")
                return False
            
            # Step 3: Create video generation assets
            print("\n🎬 Preparing video generation...")
            video_assets = self._create_video_generation_assets()
            
            # Step 4: Run video generation
            if not self._run_video_generation(video_assets):
                print("❌ Video generation failed, stopping evaluation")
                return False
            
            # Step 5: Evaluate navigation success
            print("\n📊 Evaluating navigation success...")
            evaluation_results = self._evaluate_navigation_success(episode_data)
            self.results["evaluation_results"] = evaluation_results
            
            # Step 6: Save final results
            self._save_results()
            
            success = evaluation_results.get("success", False)
            if success:
                print("\n🎉 EVALUATION COMPLETED SUCCESSFULLY! 🎉")
            else:
                print("\n💔 EVALUATION COMPLETED - NAVIGATION FAILED")
            
            return success
            
        except Exception as e:
            error_msg = f"Evaluation pipeline error: {str(e)}"
            print(f"\n❌ {error_msg}")
            self.results["errors"].append(error_msg)
            self._save_results()
            return False
        finally:
            # Cleanup temporary files
            temp_files = [
                self.output_dir / "temp_instextnav_preprocess_config.json",
                self.output_dir / "video_generation_config.json"
            ]
            for temp_file in temp_files:
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except Exception:
                        pass
            
 


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
