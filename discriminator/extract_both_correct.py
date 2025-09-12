#!/usr/bin/env python3
"""
Extract Both Models Correct Episodes

This script extracts all episodes where both model1 and model2 are correct (sr=true)
using the same logic as controversial episode extraction.

Usage:
    python extract_both_correct.py [--config_path /path/to/config.json] [--output_file both_correct_episodes.json]
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def load_batch_results(model_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load batch results from individual episode directories"""
    results = {}
    
    # Walk through all scene directories
    for scene_dir in model_dir.iterdir():
        if scene_dir.is_dir() and scene_dir.name != "batch_output.json":
            scene_id = scene_dir.name
            
            # Walk through all episode directories in this scene
            for episode_dir in scene_dir.iterdir():
                if episode_dir.is_dir() and episode_dir.name.isdigit():
                    episode_id = episode_dir.name
                    episode_key = f"{scene_id}/{episode_id}"
                    
                    # Check if output.json exists
                    output_file = episode_dir / "output.json"
                    if output_file.exists():
                        try:
                            with open(output_file, 'r') as f:
                                data = json.load(f)
                            
                            # Extract evaluation results
                            eval_results = data.get('evaluation_results')
                            if eval_results is None:
                                print(f"Warning: No evaluation_results found for {episode_key}")
                                continue
                            
                            # Get target object name from action.json (more reliable)
                            target_object_name = 'unknown'
                            preprocess_dir = episode_dir / "preprocess"
                            action_file = preprocess_dir / "action.json"
                            
                            if action_file.exists():
                                try:
                                    with open(action_file, 'r') as f:
                                        action_data = json.load(f)
                                    target_object_name = action_data.get('target_info', {}).get('name', 'unknown')
                                except (json.JSONDecodeError, KeyError) as e:
                                    print(f"Warning: Could not load target name from {action_file}: {e}")
                                    # Fallback to output.json
                                    target_object_name = data.get('object_category', 'unknown')
                            else:
                                print(f"Warning: action.json not found for {episode_key}, using fallback")
                                # Fallback to output.json
                                target_object_name = data.get('object_category', 'unknown')
                            
                            results[episode_key] = {
                                'sr': eval_results.get('sr', False),
                                'spl': eval_results.get('spl', 0.0),
                                'success': eval_results.get('success', False),
                                'geodesic_distance_to_target': eval_results.get('geodesic_distance_to_target', 0.0),
                                'path_length': eval_results.get('path_length', 0.0),
                                'object_category': target_object_name
                            }
                        except (json.JSONDecodeError, KeyError) as e:
                            print(f"Warning: Could not load results for {episode_key}: {e}")
                            continue
    
    return results


def extract_both_correct_episodes(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract episodes where both models are correct"""
    
    model1_dir = Path(config['model_paths']['model1_output'])
    model2_dir = Path(config['model_paths']['model2_output'])
    
    print("Loading model results...")
    model1_results = load_batch_results(model1_dir)
    model2_results = load_batch_results(model2_dir)
    
    # Find common episodes
    common_episodes = set(model1_results.keys()) & set(model2_results.keys())
    print(f"Found {len(common_episodes)} common episodes")
    
    both_correct = []
    
    for episode_key in common_episodes:
        m1_result = model1_results[episode_key]
        m2_result = model2_results[episode_key]
        
        m1_success = m1_result.get('success', False)
        m2_success = m2_result.get('success', False)
        
        # Check if both models are correct (sr=true)
        if m1_success and m2_success:
            episode_data = {
                'episode_key': episode_key,
                'scene_id': episode_key.split('/')[0],
                'episode_id': episode_key.split('/')[1],
                'model1_success': m1_success,
                'model1_sr': m1_result.get('sr', False),
                'model1_spl': m1_result.get('spl', 0.0),
                'model1_path_length': m1_result.get('path_length', 0.0),
                'model2_success': m2_success,
                'model2_sr': m2_result.get('sr', False),
                'model2_spl': m2_result.get('spl', 0.0),
                'model2_path_length': m2_result.get('path_length', 0.0),
                'object_category': m1_result.get('object_category', 'unknown'),
                'agreement_type': 'both_success'
            }
            both_correct.append(episode_data)
    
    print(f"Found {len(both_correct)} episodes where both models are correct")
    return both_correct


def save_results(both_correct_episodes: List[Dict[str, Any]], output_file: str):
    """Save both correct episodes to file"""
    
    # Create summary statistics
    total_episodes = len(both_correct_episodes)
    
    # Group by object category
    category_stats = {}
    for episode in both_correct_episodes:
        category = episode['object_category']
        if category not in category_stats:
            category_stats[category] = []
        category_stats[category].append(episode)
    
    # Calculate average SPL for each model
    if total_episodes > 0:
        avg_model1_spl = sum(ep['model1_spl'] for ep in both_correct_episodes) / total_episodes
        avg_model2_spl = sum(ep['model2_spl'] for ep in both_correct_episodes) / total_episodes
        avg_model1_path_length = sum(ep['model1_path_length'] for ep in both_correct_episodes) / total_episodes
        avg_model2_path_length = sum(ep['model2_path_length'] for ep in both_correct_episodes) / total_episodes
    else:
        avg_model1_spl = avg_model2_spl = avg_model1_path_length = avg_model2_path_length = 0
    
    output_data = {
        "summary": {
            "total_both_correct_episodes": total_episodes,
            "average_model1_spl": avg_model1_spl,
            "average_model2_spl": avg_model2_spl,
            "average_model1_path_length": avg_model1_path_length,
            "average_model2_path_length": avg_model2_path_length,
            "categories_count": len(category_stats),
            "category_breakdown": {cat: len(episodes) for cat, episodes in category_stats.items()}
        },
        "both_correct_episodes": both_correct_episodes
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("BOTH MODELS CORRECT - SUMMARY")
    print("="*60)
    print(f"Total Episodes (Both Correct): {total_episodes}")
    print(f"Average Model1 SPL: {avg_model1_spl:.4f}")
    print(f"Average Model2 SPL: {avg_model2_spl:.4f}")
    print(f"Average Model1 Path Length: {avg_model1_path_length:.2f}")
    print(f"Average Model2 Path Length: {avg_model2_path_length:.2f}")
    print(f"Number of Object Categories: {len(category_stats)}")
    
    print("\nTop 10 Object Categories:")
    sorted_categories = sorted(category_stats.items(), key=lambda x: len(x[1]), reverse=True)
    for i, (category, episodes) in enumerate(sorted_categories[:10], 1):
        percentage = len(episodes) / total_episodes * 100
        print(f"  {i:2d}. {category}: {len(episodes)} episodes ({percentage:.1f}%)")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Extract episodes where both models are correct")
    parser.add_argument("--config_path", default="discriminator_config.json", 
                       help="Path to configuration file")
    parser.add_argument("--output_file", default="both_correct_episodes.json",
                       help="Output file for both correct episodes")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config_path)
    
    # Extract both correct episodes
    both_correct_episodes = extract_both_correct_episodes(config)
    
    # Save results
    save_results(both_correct_episodes, args.output_file)
    
    return len(both_correct_episodes)


if __name__ == "__main__":
    total_found = main()
    print(f"\n✅ Extraction completed. Found {total_found} episodes where both models are correct.")
