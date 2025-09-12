#!/usr/bin/env python3
"""
Final Evaluation Script

This script combines the original evaluation results from both models with the discriminated results
to compute overall success rates and SPL metrics.

Usage:
    python final_evaluation.py [--config_path /path/to/config.json] [--discrimination_results /path/to/discrimination_results.json]
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


@dataclass
class FinalEpisodeResult:
    """Final episode result after discrimination"""
    scene_id: str
    episode_id: str
    original_model1_sr: bool
    original_model1_spl: float
    original_model2_sr: bool
    original_model2_spl: float
    final_sr: bool
    final_spl: float
    source: str  # "model1_original", "model2_original", "discriminated_model1", "discriminated_model2"
    was_controversial: bool


class FinalEvaluator:
    """Combines original results with discrimination results"""
    
    def __init__(self, config: Dict[str, Any], discrimination_results_path: str = None):
        self.config = config
        self.model1_dir = Path(config['model_paths']['model1_output'])
        self.model2_dir = Path(config['model_paths']['model2_output'])
        self.discriminator_output_dir = Path(config['output_config']['discriminator_output'])
        
        # Load controversial episodes from file
        self.controversial_episodes_file = Path(config['output_config']['controversial_episodes_file'])
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
    def load_batch_results(self, model_dir: Path) -> Dict[str, Dict[str, Any]]:
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
                                        # Fallback to output.json
                                        target_object_name = data.get('object_category', 'unknown')
                                else:
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
                                self.logger.warning(f"Could not load results for {episode_key}: {e}")
                                continue
        
        return results
    
    def load_controversial_episodes(self) -> Dict[str, Dict[str, Any]]:
        """Load controversial episodes from file"""
        if not self.controversial_episodes_file.exists():
            self.logger.warning(f"Controversial episodes file not found: {self.controversial_episodes_file}")
            return {}
            
        try:
            with open(self.controversial_episodes_file, 'r') as f:
                data = json.load(f)
            
            controversial_dict = {}
            # Handle different file formats
            if isinstance(data, list):
                episodes_list = data
            elif isinstance(data, dict) and 'controversial_episodes' in data:
                episodes_list = data['controversial_episodes']
            else:
                self.logger.error(f"Unexpected file format in {self.controversial_episodes_file}")
                return {}
            
            for episode in episodes_list:
                episode_key = f"{episode['scene_id']}/{episode['episode_id']}"
                controversial_dict[episode_key] = episode
            
            self.logger.info(f"Loaded {len(controversial_dict)} controversial episodes")
            return controversial_dict
            
        except Exception as e:
            self.logger.error(f"Error loading controversial episodes: {e}")
            return {}
    
    def load_discrimination_result(self, scene_id: str, episode_id: str) -> Dict[str, Any]:
        """Load discrimination result for a specific episode"""
        discrimination_file = self.discriminator_output_dir / scene_id / episode_id / "discrimination_result.json"
        
        if discrimination_file.exists():
            try:
                with open(discrimination_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load discrimination result for {scene_id}/{episode_id}: {e}")
        
        return None
    
    def combine_results(self) -> Tuple[List[FinalEpisodeResult], Dict[str, Any]]:
        """Combine original results with discrimination results using new logic"""
        self.logger.info("Loading original results...")
        
        model1_results = self.load_batch_results(self.model1_dir)
        model2_results = self.load_batch_results(self.model2_dir)
        
        self.logger.info("Loading controversial episodes...")
        controversial_episodes = self.load_controversial_episodes()
        
        # Find all episodes from both models
        all_episodes = set(model1_results.keys()) | set(model2_results.keys())
        common_episodes = set(model1_results.keys()) & set(model2_results.keys())
        
        self.logger.info(f"Total episodes: {len(all_episodes)}")
        self.logger.info(f"Common episodes: {len(common_episodes)}")
        self.logger.info(f"Controversial episodes: {len(controversial_episodes)}")
        
        final_results = []
        discriminated_count = 0
        discrimination_correct_count = 0  # New counter for correct discriminations
        
        for episode_key in all_episodes:
            scene_id, episode_id = episode_key.split('/')
            
            # Get original results
            model1_data = model1_results.get(episode_key)
            model2_data = model2_results.get(episode_key)
            
            # Default values
            m1_sr = model1_data.get('sr', False) if model1_data else False
            m1_spl = model1_data.get('spl', 0.0) if model1_data else 0.0
            m2_sr = model2_data.get('sr', False) if model2_data else False
            m2_spl = model2_data.get('spl', 0.0) if model2_data else 0.0
            
            # Determine final result based on new logic
            if episode_key in controversial_episodes:
                # This is a controversial episode
                was_controversial = True
                
                # Try to load discrimination result
                discrimination_result = self.load_discrimination_result(scene_id, episode_id)
                
                if discrimination_result and discrimination_result.get('decision'):
                    # We have a discrimination result
                    decision = discrimination_result.get('decision')
                    discriminated_count += 1
                    
                    # Check if discrimination is correct
                    # Correct decision means choosing the model with better performance
                    # Priority: success rate first, then SPL if success rates are equal
                    is_discrimination_correct = False
                    
                    if m1_sr != m2_sr:
                        # Different success rates - correct choice is the successful one
                        if (m1_sr and decision == "Model 1") or (m2_sr and decision == "Model 2"):
                            is_discrimination_correct = True
                    else:
                        # Same success rates - choose based on SPL
                        if (m1_spl > m2_spl and decision == "Model 1") or (m2_spl > m1_spl and decision == "Model 2"):
                            is_discrimination_correct = True
                        elif m1_spl == m2_spl:
                            # Equal performance - any decision is reasonable
                            is_discrimination_correct = True
                    
                    if is_discrimination_correct:
                        discrimination_correct_count += 1
                    
                    if decision == "Model 1":
                        final_sr = m1_sr
                        final_spl = m1_spl
                        source = "discriminated_model1"
                    elif decision == "Model 2":
                        final_sr = m2_sr
                        final_spl = m2_spl
                        source = "discriminated_model2"
                    else:
                        # Unclear decision, default to model2
                        final_sr = m2_sr
                        final_spl = m2_spl
                        source = "controversial_no_clear_decision_default_model2"
                else:
                    # No discrimination result yet, default to model2
                    final_sr = m2_sr
                    final_spl = m2_spl
                    source = "controversial_no_discrimination_default_model2"
                    
            else:
                # Not controversial episode
                was_controversial = False
                
                if model1_data and model2_data:
                    # Both models have results, they should agree (or this would be controversial)
                    # Check if they actually agree
                    if m1_sr == m2_sr:
                        final_sr = m1_sr
                        
                        if m1_sr and m2_sr:  # Both successful (SR=True)
                            # Try to get discrimination result to choose which model's SPL to use
                            discrimination_result = self.load_discrimination_result(scene_id, episode_id)
                            
                            if discrimination_result and discrimination_result.get('decision'):
                                decision = discrimination_result.get('decision')
                                discriminated_count += 1
                                
                                # Check if discrimination is correct (both are successful, so compare SPL)
                                if (m1_spl > m2_spl and decision == "Model 1") or (m2_spl > m1_spl and decision == "Model 2"):
                                    discrimination_correct_count += 1
                                elif m1_spl == m2_spl:
                                    # Equal SPL - any decision is reasonable
                                    discrimination_correct_count += 1
                                
                                if decision == "Model 1":
                                    final_spl = m1_spl
                                    source = "both_successful_discriminated_model1"
                                elif decision == "Model 2":
                                    final_spl = m2_spl
                                    source = "both_successful_discriminated_model2"
                                else:
                                    # Unclear decision, default to model2
                                    final_spl = m2_spl
                                    source = "both_successful_unclear_decision_default_model2"
                            else:
                                # No discrimination result, default to model2
                                final_spl = m2_spl
                                source = "both_successful_no_discrimination_default_model2"
                        else:
                            # Both failed (SR=False), SPL should be 0.0
                            final_spl = 0.0
                            source = "both_models_failed"
                    else:
                        # They disagree but not in controversial list - this shouldn't happen
                        # Default to model2
                        final_sr = m2_sr
                        final_spl = m2_spl
                        source = "both_models_disagree_not_in_controversial_default_model2"
                        self.logger.warning(f"Episode {episode_key} has disagreeing models but not in controversial list")
                        
                elif model1_data and not model2_data:
                    # Only model1 has results
                    final_sr = m1_sr
                    final_spl = m1_spl
                    source = "model1_only"
                    
                elif model2_data and not model1_data:
                    # Only model2 has results
                    final_sr = m2_sr
                    final_spl = m2_spl
                    source = "model2_only"
                    
                else:
                    # Neither model has results (shouldn't happen)
                    final_sr = False
                    final_spl = 0.0
                    source = "no_results"
                    self.logger.warning(f"Episode {episode_key} has no results from either model")
            
            final_results.append(FinalEpisodeResult(
                scene_id=scene_id,
                episode_id=episode_id,
                original_model1_sr=m1_sr,
                original_model1_spl=m1_spl,
                original_model2_sr=m2_sr,
                original_model2_spl=m2_spl,
                final_sr=final_sr,
                final_spl=final_spl,
                source=source,
                was_controversial=was_controversial
            ))
        
        # Generate summary statistics
        summary = self.generate_summary(final_results, model1_results, model2_results, 
                                       controversial_episodes, discriminated_count, discrimination_correct_count)
        
        return final_results, summary
    
    def generate_summary(self, final_results: List[FinalEpisodeResult], 
                        model1_results: Dict[str, Any], 
                        model2_results: Dict[str, Any], 
                        controversial_episodes: Dict[str, Any],
                        discriminated_count: int,
                        discrimination_correct_count: int) -> Dict[str, Any]:
        """Generate comprehensive summary statistics"""
        
        total_episodes = len(final_results)
        final_successes = sum(1 for r in final_results if r.final_sr)
        final_sr = final_successes / total_episodes if total_episodes > 0 else 0
        
        total_spl = sum(r.final_spl for r in final_results)
        final_spl = total_spl / total_episodes if total_episodes > 0 else 0
        
        controversial_count = len(controversial_episodes)
        controversial_episodes_in_results = sum(1 for r in final_results if r.was_controversial)
        
        # Original model statistics (for common episodes only)
        common_episodes = [r for r in final_results if r.original_model1_sr is not None and r.original_model2_sr is not None]
        common_count = len(common_episodes)
        
        if common_count > 0:
            model1_original_sr = sum(1 for r in common_episodes if r.original_model1_sr) / common_count
            model1_original_spl = sum(r.original_model1_spl for r in common_episodes) / common_count
            model2_original_sr = sum(1 for r in common_episodes if r.original_model2_sr) / common_count
            model2_original_spl = sum(r.original_model2_spl for r in common_episodes) / common_count
        else:
            model1_original_sr = model1_original_spl = model2_original_sr = model2_original_spl = 0
        
        # Source breakdown
        source_counts = {}
        for result in final_results:
            source = result.source
            if source not in source_counts:
                source_counts[source] = 0
            source_counts[source] += 1
        
        # Calculate discrimination completion rate
        discrimination_completion_rate = discriminated_count / controversial_count if controversial_count > 0 else 0
        
        summary = {
            "final_evaluation": {
                "total_episodes": total_episodes,
                "successful_episodes": final_successes,
                "overall_sr": final_sr,
                "overall_spl": final_spl
            },
            "original_model_performance": {
                "common_episodes": common_count,
                "model1": {
                    "sr": model1_original_sr,
                    "spl": model1_original_spl
                },
                "model2": {
                    "sr": model2_original_sr,
                    "spl": model2_original_spl
                }
            },
            "discrimination_impact": {
                "controversial_episodes_total": controversial_count,
                "controversial_episodes_in_results": controversial_episodes_in_results,
                "discriminated_episodes": discriminated_count,
                "discrimination_correct_episodes": discrimination_correct_count,
                "discrimination_accuracy": discrimination_correct_count / discriminated_count if discriminated_count > 0 else 0,
                "discrimination_completion_rate": discrimination_completion_rate,
                "controversy_rate": controversial_count / common_count if common_count > 0 else 0
            },
            "episode_sources": source_counts
        }
        
        return summary
    
    def save_results(self, final_results: List[FinalEpisodeResult], summary: Dict[str, Any]):
        """Save final evaluation results"""
        output_file = self.config['output_config']['final_results_file']
        
        output_data = {
            "summary": summary,
            "detailed_results": []
        }
        
        for result in final_results:
            output_data["detailed_results"].append({
                "episode_key": f"{result.scene_id}/{result.episode_id}",
                "scene_id": result.scene_id,
                "episode_id": result.episode_id,
                "original_model1": {
                    "sr": result.original_model1_sr,
                    "spl": result.original_model1_spl
                },
                "original_model2": {
                    "sr": result.original_model2_sr,
                    "spl": result.original_model2_spl
                },
                "final_result": {
                    "sr": result.final_sr,
                    "spl": result.final_spl
                },
                "source": result.source,
                "was_controversial": result.was_controversial
            })
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        self.logger.info(f"Results saved to: {output_file}")
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print a formatted summary"""
        print("\n" + "="*80)
        print("🎯 FINAL EVALUATION SUMMARY")
        print("="*80)
        
        final_eval = summary["final_evaluation"]
        print(f"📊 Overall Results:")
        print(f"   Total Episodes: {final_eval['total_episodes']}")
        print(f"   Final Success Rate (SR): {final_eval['overall_sr']:.4f} ({final_eval['successful_episodes']}/{final_eval['total_episodes']})")
        print(f"   Final SPL: {final_eval['overall_spl']:.4f}")
        
        print("\n" + "-"*80)
        print("📈 ORIGINAL MODEL PERFORMANCE (Common Episodes)")
        print("-"*80)
        
        orig = summary["original_model_performance"]
        print(f"Common Episodes: {orig['common_episodes']}")
        print(f"Model 1 - SR: {orig['model1']['sr']:.4f}, SPL: {orig['model1']['spl']:.4f}")
        print(f"Model 2 - SR: {orig['model2']['sr']:.4f}, SPL: {orig['model2']['spl']:.4f}")
        
        print("\n" + "-"*80)
        print("🤖 DISCRIMINATION ANALYSIS")
        print("-"*80)
        
        discrim = summary["discrimination_impact"]
        print(f"Controversial Episodes (Total): {discrim['controversial_episodes_total']}")
        print(f"Controversial Episodes (In Results): {discrim['controversial_episodes_in_results']}")
        print(f"Already Discriminated: {discrim['discriminated_episodes']}")
        print(f"Discrimination Correct: {discrim['discrimination_correct_episodes']}")
        print(f"Discrimination Accuracy: {discrim['discrimination_accuracy']:.2%} ({discrim['discrimination_correct_episodes']}/{discrim['discriminated_episodes']})")
        print(f"Discrimination Completion Rate: {discrim['discrimination_completion_rate']:.2%}")
        print(f"Controversy Rate: {discrim['controversy_rate']:.2%}")
        
        print("\n" + "-"*80)
        print("📋 EPISODE SOURCE BREAKDOWN")
        print("-"*80)
        
        # Group related sources for better readability
        source_groups = {
            "Controversial Episodes": [
                "discriminated_model1", "discriminated_model2", 
                "controversial_no_clear_decision_default_model2",
                "controversial_no_discrimination_default_model2"
            ],
            "Both Models Successful": [
                "both_successful_discriminated_model1", "both_successful_discriminated_model2",
                "both_successful_unclear_decision_default_model2", 
                "both_successful_no_discrimination_default_model2"
            ],
            "Agreement Cases": [
                "both_models_agree", "both_models_failed"
            ],
            "Single Model Results": [
                "model1_only", "model2_only"
            ],
            "Edge Cases": [
                "both_models_disagree_not_in_controversial_default_model2", "no_results"
            ]
        }
        
        for group_name, source_list in source_groups.items():
            group_total = sum(summary["episode_sources"].get(source, 0) for source in source_list)
            if group_total > 0:
                print(f"\n{group_name}: {group_total}")
                for source in source_list:
                    count = summary["episode_sources"].get(source, 0)
                    if count > 0:
                        percentage = count / final_eval['total_episodes'] * 100
                        print(f"   {source}: {count} ({percentage:.1f}%)")
        
        # Show any other sources not in groups
        grouped_sources = set()
        for source_list in source_groups.values():
            grouped_sources.update(source_list)
        
        other_sources = set(summary["episode_sources"].keys()) - grouped_sources
        if other_sources:
            print(f"\nOther Sources:")
            for source in other_sources:
                count = summary["episode_sources"][source]
                percentage = count / final_eval['total_episodes'] * 100
                print(f"   {source}: {count} ({percentage:.1f}%)")
        
        print("\n" + "="*80)
    
    def run_evaluation(self):
        """Run the complete final evaluation"""
        self.logger.info("Starting final evaluation...")
        
        final_results, summary = self.combine_results()
        
        self.save_results(final_results, summary)
        self.print_summary(summary)
        
        return final_results, summary


def main():
    parser = argparse.ArgumentParser(description="Final Evaluation Script")
    parser.add_argument("--config_path", default="discriminator_config.json", 
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config_path)
    
    evaluator = FinalEvaluator(config=config)
    
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
