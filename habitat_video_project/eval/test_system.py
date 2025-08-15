#!/usr/bin/env python3
"""
Test script for the evaluation system.
Performs basic validation of the configuration and dependencies.
"""

import os
import sys
import json
from pathlib import Path


def test_dependencies():
    """Test if all required dependencies are available."""
    print("Testing dependencies...")
    
    try:
        import numpy as np
        print("✓ numpy")
    except ImportError:
        print("✗ numpy - required for calculations")
        return False
    
    try:
        import habitat_sim
        print("✓ habitat_sim")
    except ImportError:
        print("✗ habitat_sim - required for simulation")
        return False
    
    try:
        from habitat.utils.visualizations import maps
        print("✓ habitat visualization utils")
    except ImportError:
        print("✗ habitat visualization utils - required for maps")
        return False
    
    return True


def test_file_structure():
    """Test if all required files and directories exist."""
    print("\\nTesting file structure...")
    
    eval_dir = Path(__file__).parent
    project_root = eval_dir.parent
    habitat_root = project_root.parent
    
    required_files = [
        eval_dir / "run_eval.py",
        eval_dir / "eval_config_template.json",
        eval_dir / "eval_config_episode_427.json",
        project_root / "main.py",
        habitat_root / "preprocess" / "main_workflow.py"
    ]
    
    all_exist = True
    for file_path in required_files:
        if file_path.exists():
            print(f"✓ {file_path.name}")
        else:
            print(f"✗ {file_path} - missing")
            all_exist = False
    
    return all_exist


def test_config_validity():
    """Test if the configuration files are valid JSON."""
    print("\\nTesting configuration files...")
    
    eval_dir = Path(__file__).parent
    config_files = [
        eval_dir / "eval_config_template.json",
        eval_dir / "eval_config_episode_427.json"
    ]
    
    all_valid = True
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"✓ {config_file.name} - valid JSON")
            
            # Test required sections
            required_sections = ['episode', 'scene', 'preprocess', 'video_generation', 'evaluation']
            for section in required_sections:
                if section in config:
                    print(f"  ✓ {section} section present")
                else:
                    print(f"  ✗ {section} section missing")
                    all_valid = False
                    
        except json.JSONDecodeError as e:
            print(f"✗ {config_file.name} - invalid JSON: {e}")
            all_valid = False
        except FileNotFoundError:
            print(f"✗ {config_file.name} - file not found")
            all_valid = False
    
    return all_valid


def test_scene_files():
    """Test if scene files exist."""
    print("\\nTesting scene files...")
    
    eval_dir = Path(__file__).parent
    config_file = eval_dir / "eval_config_episode_427.json"
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        scene_file = Path(config['scene']['scene_file'])
        robot_urdf = Path(config['scene']['robot_urdf'])
        episode_json = Path(config['episode']['episode_json_path'])
        
        files_to_check = [
            (scene_file, "Scene file"),
            (robot_urdf, "Robot URDF"),
            (episode_json, "Episode JSON")
        ]
        
        all_exist = True
        for file_path, description in files_to_check:
            if file_path.exists():
                print(f"✓ {description}: {file_path.name}")
            else:
                print(f"✗ {description}: {file_path} - not found")
                all_exist = False
        
        return all_exist
        
    except Exception as e:
        print(f"✗ Error checking scene files: {e}")
        return False


def test_episode_data():
    """Test if episode data can be loaded."""
    print("\\nTesting episode data...")
    
    eval_dir = Path(__file__).parent
    config_file = eval_dir / "eval_config_episode_427.json"
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        episode_json_path = config['episode']['episode_json_path']
        episode_id = config['episode']['episode_id']
        
        # Load episode data
        if episode_json_path.endswith('.gz'):
            import gzip
            with gzip.open(episode_json_path, 'rt', encoding='utf-8') as f:
                episode_data = json.load(f)
        else:
            with open(episode_json_path, 'r', encoding='utf-8') as f:
                episode_data = json.load(f)
        
        # Find target episode
        target_episode = None
        for episode in episode_data['episodes']:
            if str(episode['episode_id']) == str(episode_id):
                target_episode = episode
                break
        
        if target_episode:
            print(f"✓ Episode {episode_id} found")
            print(f"  Object category: {target_episode.get('object_category', 'N/A')}")
            print(f"  Start position: {target_episode.get('start_position', 'N/A')}")
            
            # Check if goals exist for this category
            object_category = target_episode.get('object_category')
            if object_category:
                goals_found = False
                for key in episode_data.get('goals_by_category', {}).keys():
                    if key.endswith(object_category) or key == object_category:
                        goals = episode_data['goals_by_category'][key]
                        print(f"  ✓ Found {len(goals)} goals for category '{object_category}'")
                        goals_found = True
                        break
                
                if not goals_found:
                    print(f"  ✗ No goals found for category '{object_category}'")
                    return False
            
            return True
        else:
            print(f"✗ Episode {episode_id} not found")
            return False
            
    except Exception as e:
        print(f"✗ Error loading episode data: {e}")
        return False


def main():
    """Run all tests."""
    print("🔍 Testing Evaluation System")
    print("="*50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("File Structure", test_file_structure), 
        ("Configuration Files", test_config_validity),
        ("Scene Files", test_scene_files),
        ("Episode Data", test_episode_data)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\\n{test_name}:")
        print("-" * len(test_name))
        passed = test_func()
        if not passed:
            all_passed = False
    
    print("\\n" + "="*50)
    if all_passed:
        print("🎉 All tests passed! The evaluation system is ready to use.")
        print("\\nTo run a single episode evaluation:")
        print("  python run_eval.py eval_config_episode_427.json")
        print("\\nTo run batch evaluation:")
        print("  python batch_eval.py batch_episodes_example.json")
    else:
        print("❌ Some tests failed. Please fix the issues before running evaluations.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
