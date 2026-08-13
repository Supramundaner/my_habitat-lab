#!/usr/bin/env python3
"""
Data Sampler for OVON HM3D Episodes

This script samples episodes from three different OVON HM3D datasets:
- val_seen: Default 4 episodes per scene
- val_seen_synonyms: Default 3 episodes per scene  
- val_unseen: Default 3 episodes per scene

Output format matches batch_episodes.json structure.

Usage from the repository root:
    python3 -m reason_navi.evaluation.sample_episodes --seed 42
"""

import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional

if __package__:
    from .sampler_paths import (
        SamplerPaths,
        resolve_sampler_paths,
        validate_input_directories,
    )
else:
    from sampler_paths import (  # type: ignore
        SamplerPaths,
        resolve_sampler_paths,
        validate_input_directories,
    )


class OVONDataSampler:
    def __init__(self, paths: SamplerPaths, seed: int = 0):
        self.paths = paths
        self.seed = seed
        self.data_root = paths.dataset.runtime_path
        self.scene_root = paths.scenes.runtime_path
        
        # Define dataset paths
        self.datasets = {
            'val_seen': self.data_root / 'val_seen' / 'content',
            'val_seen_synonyms': self.data_root / 'val_seen_synonyms' / 'content', 
            'val_unseen': self.data_root / 'val_unseen' / 'content'
        }
        
        # Default sampling counts
        self.default_sample_counts = {
            'val_seen': 4,
            'val_seen_synonyms': 3,
            'val_unseen': 3
        }
    
    def get_scene_files(self) -> Dict[str, Path]:
        """
        Map scene names to their full .glb file paths
        """
        scene_files = {}
        for scene_dir in sorted(self.scene_root.iterdir()):
            if scene_dir.is_dir():
                # Extract scene ID from directory name (e.g., "00824-Dd4bFSTQ8gi" -> "Dd4bFSTQ8gi")
                scene_id = scene_dir.name.split('-', 1)[1] if '-' in scene_dir.name else scene_dir.name
                
                # Find the .glb file
                glb_files = sorted(scene_dir.glob('*.glb'))
                if glb_files:
                    scene_files[scene_id] = glb_files[0]
                    
        return scene_files
    
    def load_episodes_from_file(self, episode_file: Path) -> List[Dict]:
        """
        Load episodes from a JSON file
        """
        try:
            with open(episode_file, 'r') as f:
                data = json.load(f)
                return data.get('episodes', [])
        except Exception as e:
            print(f"Error loading {episode_file}: {e}")
            return []
    
    def sample_episodes_from_dataset(self, dataset_name: str, sample_count: int) -> List[Dict]:
        """
        Sample episodes from a specific dataset using random sampling
        """
        dataset_path = self.datasets[dataset_name]
        evaluation_tasks = []
        scene_files = self.get_scene_files()
        
        # Get all episode files and sort them for consistent ordering
        episode_files = sorted(list(dataset_path.glob('*.json')))
        print(f"Found {len(episode_files)} scene files in {dataset_name}")
        
        for episode_file in episode_files:
            # Extract scene name from filename (e.g., "Dd4bFSTQ8gi.json" -> "Dd4bFSTQ8gi")
            scene_name = episode_file.stem
            
            # Load episodes from file
            episodes = self.load_episodes_from_file(episode_file)
            if not episodes:
                print(f"No episodes found in {episode_file}")
                continue
                
            # Sample episodes randomly
            if len(episodes) <= sample_count:
                sampled_episodes = episodes
                print(f"Taking all {len(episodes)} episodes from {scene_name} (requested {sample_count})")
            else:
                sampled_episodes = random.sample(episodes, sample_count)
                print(f"Randomly sampled {sample_count} episodes from {len(episodes)} in {scene_name}")
            
            # Get scene file path
            if scene_name not in scene_files:
                print(f"Warning: Scene file not found for {scene_name}")
                continue
                
            scene_file = scene_files[scene_name]
            
            # Create evaluation task
            evaluation_task = {
                "episode_json_path": self.paths.dataset.child_for_config(
                    episode_file
                ),
                "scene_file": self.paths.scenes.child_for_config(scene_file),
                "episode_ids": [ep['episode_id'] for ep in sampled_episodes]
            }
            
            evaluation_tasks.append(evaluation_task)
            
        print(f"Created {len(evaluation_tasks)} evaluation tasks for {dataset_name}")
        return evaluation_tasks
    
    def sample_all_datasets(self, sample_counts: Dict[str, int] = None) -> Dict:
        """
        Sample episodes from all datasets
        """
        if sample_counts is None:
            sample_counts = self.default_sample_counts.copy()
            
        all_evaluation_tasks = []
        
        for dataset_name in self.datasets.keys():
            count = sample_counts.get(dataset_name, self.default_sample_counts[dataset_name])
            print(f"\n=== Sampling from {dataset_name} (count: {count}) ===")
            
            tasks = self.sample_episodes_from_dataset(dataset_name, count)
            all_evaluation_tasks.extend(tasks)
        
        # Create the output structure matching batch_episodes.json format
        output = {
            "evaluation_tasks": all_evaluation_tasks,
            "scene": {
                "robot_urdf": self.paths.robot_urdf
            },
            "preprocess": {
                "resolution": 2048,
                "scene_config": {
                    "target_floor": None,
                    "target_coordinate": None,
                    "goal_object": None,
                    "rotation": None,
                    "custom_ortho_scale": None,
                    "target_coverage": 0.9,
                    "draw_coordinates": False
                },
                "room_segmentation": {
                    "morph_closing_width_meters": 0.01,
                    "seed_min_distance_from_wall_meters": None,
                    "min_room_area_pixels": 15000
                },
                "graph_generation": {
                    "pds_radius": 0.5,
                    "max_attempts": 30,
                    "seed": self.seed,
                    "node_radius_pixels": 8
                },
                "llm_config": {
                    "api_key_env": "ARK_API_KEY",
                    "base_url": "https://ark.ap-southeast.bytepluses.com/api/v3",
                    "model": "seed-1-6-250615",
                    "max_tokens": 35000
                }
            },
            "video_generation": {
                "video": {
                    "fps": 8,
                    "resolution": {
                        "width": 2048,
                        "height": 1024
                    },
                    "fpv_width": 1024,
                    "map_width": 1024
                },
                "agent": {
                    "linear_speed": 2.0,
                    "angular_speed": 240.0,
                    "sensor_height": 1.31,
                    "time_steps_num": 15,
                    "min_displacement": 0.3,
                    "max_time_steps": 50
                },
                "simulation": {
                    "enable_physics": True
                },
                "gpu": {
                    "enabled": True,
                    "device": "cuda:0",
                    "mixed_precision": False,
                    "memory_efficient": True,
                    "max_chunk_size": 50000
                },
                "OCCUPANCY_MAP": {
                    "MIN_DEPTH": 0.2,
                    "MAX_DEPTH": 8.0,
                    "CAMERA_HEIGHT": 1.31,
                    "MIN_H": -1.2,
                    "MAX_H": 0.2,
                    "HFOV": 90
                },
                "vfh": {
                    "mu1": 5.0,
                    "mu2": 2.0,
                    "mu3": 2.0,
                    "mu1_prime": 5.0,
                    "mu2_prime": 1.0,
                    "mu3_prime": 1.0,
                    "lambda": 0.8,
                    "robot_radius": 0.14,
                    "sensor_range": 0.8,
                    "search_depth": 5
                },
                "navigation": {
                    "waypoint_distance": 1.2,
                    "destination_distance": 1.0
                },
                "object_detection": {
                    "enabled": True,
                    "grounding_dino_port": 12181,
                    "mobile_sam_port": 12184,
                    "detection_threshold": 0.4,
                    "max_detection_distance": 10.0
                }
            },
            "evaluation": {
                "success_distance_threshold": 0.25
            }
        }
        
        return output


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Sample episodes from OVON HM3D datasets')
    parser.add_argument('--data-root', 
                      default=None,
                      help=('Root path to OVON HM3D datasets; defaults to '
                            '$HABITAT_DATA_ROOT/datasets/ovon/hm3d'))
    parser.add_argument('--scene-root',
                      default=None,
                      help=('Root path to HM3D scene files; defaults to '
                            '$HABITAT_DATA_ROOT/versioned_data/hm3d-0.2/hm3d/val'))
    parser.add_argument('--robot-urdf',
                      default=None,
                      help=('Robot URDF written to the config; defaults to '
                            '${HABITAT_DATA_ROOT}/robots/hab_fetch/robots/'
                            'hab_fetch.urdf'))
    parser.add_argument('--output', '-o',
                      default='batch_episodes.json',
                      help='Output JSON file name')
    parser.add_argument('--val-seen-count', 
                      type=int, default=4,
                      help='Number of episodes to sample from val_seen per scene')
    parser.add_argument('--val-seen-synonyms-count',
                      type=int, default=3, 
                      help='Number of episodes to sample from val_seen_synonyms per scene')
    parser.add_argument('--val-unseen-count',
                      type=int, default=3,
                      help='Number of episodes to sample from val_unseen per scene')
    parser.add_argument('--seed',
                      type=int, default=42,
                      help='Random seed for reproducible sampling')
    
    args = parser.parse_args(argv)

    try:
        paths = resolve_sampler_paths(
            data_root=args.data_root,
            scene_root=args.scene_root,
            robot_urdf=args.robot_urdf,
            dataset_suffix='datasets/ovon/hm3d',
        )
        validate_input_directories(paths)
    except ValueError as exc:
        parser.error(str(exc))
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    # Create sampler
    sampler = OVONDataSampler(paths, seed=args.seed)
    
    # Define sample counts
    sample_counts = {
        'val_seen': args.val_seen_count,
        'val_seen_synonyms': args.val_seen_synonyms_count, 
        'val_unseen': args.val_unseen_count
    }
    
    print("=== OVON HM3D Data Sampler ===")
    print(f"Data root: {paths.dataset.runtime_path}")
    print(f"Scene root: {paths.scenes.runtime_path}")
    print(f"Sample counts: {sample_counts}")
    print(f"Output file: {args.output}")
    
    # Sample episodes
    result = sampler.sample_all_datasets(sample_counts)
    
    # Save to file
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=4, allow_nan=False)
    
    print(f"\n=== Summary ===")
    print(f"Total evaluation tasks: {len(result['evaluation_tasks'])}")
    total_episodes = sum(len(task['episode_ids']) for task in result['evaluation_tasks'])
    print(f"Total episodes sampled: {total_episodes}")
    print(f"Output saved to: {args.output}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
