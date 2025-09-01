#!/usr/bin/env python3
"""
Script to generate batch configuration files from instance_imagenav_hm3d_v3 episodes.
This script reads all episode JSON files and creates separate batch_config.json files 
for each scene with all its episodes.
"""

import json
import os
import glob
from pathlib import Path

def load_template_config():
    """Load the template batch configuration."""
    template_path = "/home/yaoaa/habitat-lab/habitat_video_project/insnav_eval/batch_config_example.json"
    with open(template_path, 'r') as f:
        return json.load(f)

def find_scene_file(scene_id, scenes_dir):
    """Find the .basis.glb file for a given scene ID."""
    scene_pattern = f"*-{scene_id}/{scene_id}.basis.glb"
    scene_files = glob.glob(os.path.join(scenes_dir, scene_pattern))
    if scene_files:
        return scene_files[0]
    return None

def extract_scene_id_from_filename(json_filename):
    """Extract scene ID from JSON filename (e.g., '4ok3usBNeis.json' -> '4ok3usBNeis')."""
    return Path(json_filename).stem

def load_episodes_from_json(json_path):
    """Load episodes from a JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data.get('episodes', [])

def create_batch_config(template_config, scene_file, episode_json_path, episode_ids):
    """Create a batch configuration for a specific scene and episodes."""
    config = template_config.copy()
    
    # Update the evaluation tasks
    config['evaluation_tasks'] = [{
        "scene_file": scene_file,
        "episode_json_path": episode_json_path,
        "episode_ids": episode_ids
    }]
    
    return config

def main():
    # Paths
    episodes_dir = "/home/yaoaa/habitat-lab/data/datasets/instance_imagenav_hm3d_v3/val/content"
    scenes_dir = "/home/yaoaa/habitat-lab/data/versioned_data/hm3d-0.2/hm3d/val"
    output_file = "/home/yaoaa/habitat-lab/batch_config_all_episodes.json"
    
    # Load template configuration
    template_config = load_template_config()
    
    # Initialize evaluation tasks list
    all_evaluation_tasks = []
    
    # Process each episode JSON file
    json_files = glob.glob(os.path.join(episodes_dir, "*.json"))
    
    print(f"Found {len(json_files)} episode files to process...")
    total_episodes = 0
    
    for json_file in json_files:
        scene_id = extract_scene_id_from_filename(json_file)
        print(f"\nProcessing scene: {scene_id}")
        
        # Find corresponding scene file
        scene_file = find_scene_file(scene_id, scenes_dir)
        if not scene_file:
            print(f"  Warning: Scene file not found for {scene_id}, skipping...")
            continue
            
        print(f"  Found scene file: {scene_file}")
        
        # Load episodes from JSON
        try:
            episodes = load_episodes_from_json(json_file)
            if not episodes:
                print(f"  Warning: No episodes found in {json_file}, skipping...")
                continue
                
            print(f"  Found {len(episodes)} episodes")
            total_episodes += len(episodes)
            
            # Extract episode IDs
            episode_ids = [ep['episode_id'] for ep in episodes]
            
            # Create evaluation task for this scene
            evaluation_task = {
                "scene_file": scene_file,
                "episode_json_path": json_file,
                "episode_ids": episode_ids
            }
            
            all_evaluation_tasks.append(evaluation_task)
            print(f"  Added evaluation task with {len(episode_ids)} episodes")
            
        except Exception as e:
            print(f"  Error processing {json_file}: {str(e)}")
            continue
    
    # Create final batch configuration with all tasks
    final_config = template_config.copy()
    final_config['evaluation_tasks'] = all_evaluation_tasks
    
    # Save the combined batch configuration
    with open(output_file, 'w') as f:
        json.dump(final_config, f, indent=4)
    
    print(f"\nBatch configuration generation completed!")
    print(f"Output file: {output_file}")
    print(f"Total scenes processed: {len(all_evaluation_tasks)}")
    print(f"Total episodes: {total_episodes}")

if __name__ == "__main__":
    main()
