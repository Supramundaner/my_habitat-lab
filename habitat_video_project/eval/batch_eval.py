#!/usr/bin/env python3
"""
Batch evaluation script for multiple episodes.
Generates individual config files and runs evaluations for multiple episodes.

Usage:
    python batch_eval.py episodes_list.json

where episodes_list.json contains:
{
    "base_config": "eval_config_template.json",
    "episodes": [
        {
            "episode_json_path": "/path/to/episode1.json",
            "episode_id": "427",
            "scene_file": "/path/to/scene1.glb"
        },
        {
            "episode_json_path": "/path/to/episode2.json", 
            "episode_id": "256",
            "scene_file": "/path/to/scene2.glb"
        }
    ]
}
"""

import json
import sys
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import concurrent.futures
from datetime import datetime


class BatchEvaluator:
    """Batch evaluator for multiple episodes."""
    
    def __init__(self, episodes_config_path: str):
        """Initialize batch evaluator."""
        self.episodes_config_path = Path(episodes_config_path)
        self.episodes_config = self._load_episodes_config()
        self.eval_dir = Path(__file__).parent
        self.results = []
        
    def _load_episodes_config(self) -> Dict[str, Any]:
        """Load episodes configuration."""
        with open(self.episodes_config_path, 'r') as f:
            return json.load(f)
    
    def _create_episode_config(self, episode_info: Dict[str, Any]) -> Path:
        """Create individual episode config based on template."""
        # Load base config
        base_config_path = self.eval_dir / self.episodes_config['base_config']
        with open(base_config_path, 'r') as f:
            config = json.load(f)
        
        # Update episode-specific information
        config['episode']['episode_json_path'] = episode_info['episode_json_path']
        config['episode']['episode_id'] = episode_info['episode_id']
        config['scene']['scene_file'] = episode_info['scene_file']
        
        # Update robot_urdf if provided
        if 'robot_urdf' in episode_info:
            config['scene']['robot_urdf'] = episode_info['robot_urdf']
        
        # Create config filename
        episode_json_name = Path(episode_info['episode_json_path']).stem
        episode_id = episode_info['episode_id']
        config_filename = f"eval_config_{episode_json_name}_{episode_id}.json"
        config_path = self.eval_dir / config_filename
        
        # Save config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config_path
    
    def _run_single_evaluation(self, episode_info: Dict[str, Any]) -> Dict[str, Any]:
        """Run evaluation for a single episode."""
        try:
            print(f"\\n{'='*60}")
            print(f"Evaluating Episode {episode_info['episode_id']}")
            print(f"Scene: {Path(episode_info['scene_file']).name}")
            print(f"{'='*60}")
            
            # Create episode config
            config_path = self._create_episode_config(episode_info)
            
            # Run evaluation
            cmd = [sys.executable, "run_eval.py", str(config_path)]
            print(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=self.eval_dir,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout per episode
            )
            
            # Parse results
            episode_result = {
                "episode_info": episode_info,
                "config_path": str(config_path),
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timestamp": datetime.now().isoformat()
            }
            
            if result.returncode == 0:
                # Try to load evaluation results
                episode_json_name = Path(episode_info['episode_json_path']).stem
                output_dir = self.eval_dir / "output" / episode_json_name
                output_json = output_dir / "output.json"
                
                if output_json.exists():
                    with open(output_json, 'r') as f:
                        eval_results = json.load(f)
                    episode_result["evaluation_results"] = eval_results.get("evaluation_results", {})
                
                print(f"✓ Episode {episode_info['episode_id']} completed successfully")
            else:
                print(f"✗ Episode {episode_info['episode_id']} failed")
                if result.stderr:
                    print(f"Error: {result.stderr}")
            
            return episode_result
            
        except subprocess.TimeoutExpired:
            print(f"✗ Episode {episode_info['episode_id']} timed out")
            return {
                "episode_info": episode_info,
                "success": False,
                "error": "Timeout after 30 minutes",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"✗ Episode {episode_info['episode_id']} failed with error: {e}")
            return {
                "episode_info": episode_info,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def run_sequential(self) -> List[Dict[str, Any]]:
        """Run evaluations sequentially."""
        print(f"\\n🚀 Starting batch evaluation of {len(self.episodes_config['episodes'])} episodes")
        print("Running sequentially...")
        
        results = []
        for i, episode_info in enumerate(self.episodes_config['episodes']):
            print(f"\\n[{i+1}/{len(self.episodes_config['episodes'])}]")
            result = self._run_single_evaluation(episode_info)
            results.append(result)
        
        return results
    
    def run_parallel(self, max_workers: int = 2) -> List[Dict[str, Any]]:
        """Run evaluations in parallel."""
        print(f"\\n🚀 Starting batch evaluation of {len(self.episodes_config['episodes'])} episodes")
        print(f"Running in parallel with {max_workers} workers...")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._run_single_evaluation, episode_info): episode_info
                for episode_info in self.episodes_config['episodes']
            }
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
        
        return results
    
    def generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics."""
        total_episodes = len(results)
        successful_runs = sum(1 for r in results if r['success'])
        
        # Navigation success statistics
        nav_successes = []
        nav_spls = []
        category_stats = {}
        
        for result in results:
            if result['success'] and 'evaluation_results' in result:
                eval_res = result['evaluation_results']
                nav_success = eval_res.get('success', False)
                nav_spl = eval_res.get('spl', 0.0)
                category = eval_res.get('object_category', 'unknown')
                
                nav_successes.append(nav_success)
                nav_spls.append(nav_spl)
                
                if category not in category_stats:
                    category_stats[category] = {'successes': [], 'spls': []}
                
                category_stats[category]['successes'].append(nav_success)
                category_stats[category]['spls'].append(nav_spl)
        
        summary = {
            "total_episodes": total_episodes,
            "successful_runs": successful_runs,
            "run_success_rate": successful_runs / total_episodes if total_episodes > 0 else 0,
            "navigation_success_rate": sum(nav_successes) / len(nav_successes) if nav_successes else 0,
            "average_spl": sum(nav_spls) / len(nav_spls) if nav_spls else 0,
            "category_statistics": {}
        }
        
        # Category-wise statistics
        for category, stats in category_stats.items():
            summary["category_statistics"][category] = {
                "count": len(stats['successes']),
                "success_rate": sum(stats['successes']) / len(stats['successes']),
                "average_spl": sum(stats['spls']) / len(stats['spls'])
            }
        
        return summary
    
    def save_results(self, results: List[Dict[str, Any]]):
        """Save batch evaluation results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.eval_dir / f"batch_results_{timestamp}.json"
        
        summary = self.generate_summary(results)
        
        output_data = {
            "summary": summary,
            "episodes_config": self.episodes_config,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\\n📊 Batch evaluation results saved to: {output_path}")
        
        # Print summary
        print("\\n" + "="*60)
        print("BATCH EVALUATION SUMMARY")
        print("="*60)
        print(f"Total episodes: {summary['total_episodes']}")
        print(f"Successful runs: {summary['successful_runs']} ({summary['run_success_rate']:.1%})")
        print(f"Navigation success rate: {summary['navigation_success_rate']:.1%}")
        print(f"Average SPL: {summary['average_spl']:.3f}")
        
        if summary['category_statistics']:
            print("\\nBy category:")
            for category, stats in summary['category_statistics'].items():
                print(f"  {category}: SR={stats['success_rate']:.1%}, SPL={stats['average_spl']:.3f} ({stats['count']} episodes)")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python batch_eval.py <episodes_config.json> [--parallel] [--workers N]")
        print("\\nExample episodes_config.json:")
        example = {
            "base_config": "eval_config_template.json",
            "episodes": [
                {
                    "episode_json_path": "/path/to/episode1.json",
                    "episode_id": "427",
                    "scene_file": "/path/to/scene1.glb"
                },
                {
                    "episode_json_path": "/path/to/episode2.json", 
                    "episode_id": "256",
                    "scene_file": "/path/to/scene2.glb"
                }
            ]
        }
        print(json.dumps(example, indent=2))
        sys.exit(1)
    
    episodes_config_path = sys.argv[1]
    parallel = "--parallel" in sys.argv
    
    # Parse workers argument
    max_workers = 2
    if "--workers" in sys.argv:
        workers_idx = sys.argv.index("--workers")
        if workers_idx + 1 < len(sys.argv):
            max_workers = int(sys.argv[workers_idx + 1])
    
    if not os.path.exists(episodes_config_path):
        print(f"Error: Episodes configuration file not found: {episodes_config_path}")
        sys.exit(1)
    
    try:
        evaluator = BatchEvaluator(episodes_config_path)
        
        if parallel:
            results = evaluator.run_parallel(max_workers)
        else:
            results = evaluator.run_sequential()
        
        evaluator.save_results(results)
        
        # Success if at least some runs completed
        success = any(r['success'] for r in results)
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\\n\\nBatch evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\nFatal error during batch evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
