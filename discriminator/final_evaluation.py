#!/usr/bin/env python3
"""
Final Evaluation Script

This script combines the original evaluation results from both models with the discriminated results
to compute overall success rates and SPL metrics.

Usage:
    python final_evaluation.py --model1_dir /path/to/model1/output --model2_dir /path/to/model2/output --discrimination_results /path/to/discrimination_results.json
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


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
    
    def __init__(self, model1_dir: str, model2_dir: str, discrimination_results_path: str):
        self.model1_dir = Path(model1_dir)
        self.model2_dir = Path(model2_dir)
        self.discrimination_results_path = Path(discrimination_results_path)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
    def load_batch_results(self, model_dir: Path) -> Dict[str, Dict[str, Any]]:
        """Load batch results from model directory"""
        batch_file = model_dir / "batch_output.json"
        if not batch_file.exists():
            raise FileNotFoundError(f"Batch output file not found: {batch_file}")
            
        with open(batch_file, 'r') as f:
            data = json.load(f)
        
        return data.get('episode_details', {})
    
    def load_discrimination_results(self) -> Dict[str, Any]:
        """Load discrimination results"""
        if not self.discrimination_results_path.exists():
            raise FileNotFoundError(f"Discrimination results not found: {self.discrimination_results_path}")
            
        with open(self.discrimination_results_path, 'r') as f:
            return json.load(f)
    
    def combine_results(self) -> Tuple[List[FinalEpisodeResult], Dict[str, Any]]:
        """Combine original results with discrimination results"""
        self.logger.info("Loading original results...")
        
        model1_results = self.load_batch_results(self.model1_dir)
        model2_results = self.load_batch_results(self.model2_dir)
        
        self.logger.info("Loading discrimination results...")
        discrimination_data = self.load_discrimination_results()
        
        # Create lookup for discriminated episodes
        discriminated_episodes = {}
        for result in discrimination_data.get('discriminated_results', []):
            key = f"{result['scene_id']}/{result['episode_id']}"
            discriminated_episodes[key] = result
        
        # Find all episodes from both models
        all_episodes = set(model1_results.keys()) | set(model2_results.keys())
        common_episodes = set(model1_results.keys()) & set(model2_results.keys())
        
        self.logger.info(f"Total episodes: {len(all_episodes)}")
        self.logger.info(f"Common episodes: {len(common_episodes)}")
        self.logger.info(f"Discriminated episodes: {len(discriminated_episodes)}")
        
        final_results = []
        
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
            
            # Check if this episode was discriminated
            if episode_key in discriminated_episodes:
                discrimination_result = discriminated_episodes[episode_key]
                decision = discrimination_result.get('decision')
                
                if decision == "Model 1":
                    final_sr = m1_sr
                    final_spl = m1_spl
                    source = "discriminated_model1"
                elif decision == "Model 2":
                    final_sr = m2_sr
                    final_spl = m2_spl
                    source = "discriminated_model2"
                else:
                    # No clear decision, use model1 as default (or could use other logic)
                    final_sr = m1_sr
                    final_spl = m1_spl
                    source = "no_decision_default_model1"
                
                was_controversial = True
                
            else:
                # Episode was not controversial, both models agreed
                if model1_data and model2_data:
                    # Both models have results, they should agree
                    final_sr = m1_sr  # They should be the same
                    final_spl = m1_spl  # They should be the same
                    source = "both_models_agree"
                elif model1_data:
                    # Only model1 has results
                    final_sr = m1_sr
                    final_spl = m1_spl
                    source = "model1_only"
                elif model2_data:
                    # Only model2 has results
                    final_sr = m2_sr
                    final_spl = m2_spl
                    source = "model2_only"
                else:
                    # Neither model has results (shouldn't happen)
                    final_sr = False
                    final_spl = 0.0
                    source = "no_results"
                
                was_controversial = False
            
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
        summary = self.generate_summary(final_results, model1_results, model2_results, discrimination_data)
        
        return final_results, summary
    
    def generate_summary(self, final_results: List[FinalEpisodeResult], 
                        model1_results: Dict[str, Any], 
                        model2_results: Dict[str, Any], 
                        discrimination_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary statistics"""
        
        total_episodes = len(final_results)
        final_successes = sum(1 for r in final_results if r.final_sr)
        final_sr = final_successes / total_episodes if total_episodes > 0 else 0
        
        total_spl = sum(r.final_spl for r in final_results)
        final_spl = total_spl / total_episodes if total_episodes > 0 else 0
        
        controversial_episodes = sum(1 for r in final_results if r.was_controversial)
        
        # Original model statistics
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
        
        # Discrimination impact
        discrimination_changes = 0
        for result in final_results:
            if result.was_controversial:
                if result.source == "discriminated_model1" and not result.original_model1_sr:
                    discrimination_changes += 1
                elif result.source == "discriminated_model2" and not result.original_model2_sr:
                    discrimination_changes += 1
        
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
                "controversial_episodes": controversial_episodes,
                "discrimination_changes": discrimination_changes,
                "controversy_rate": controversial_episodes / common_count if common_count > 0 else 0
            },
            "episode_sources": source_counts,
            "discrimination_summary": discrimination_data.get('summary', {})
        }
        
        return summary
    
    def save_results(self, final_results: List[FinalEpisodeResult], summary: Dict[str, Any], output_path: str):
        """Save final evaluation results"""
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
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        self.logger.info(f"Results saved to: {output_path}")
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print a formatted summary"""
        print("\n" + "="*60)
        print("FINAL EVALUATION SUMMARY")
        print("="*60)
        
        final_eval = summary["final_evaluation"]
        print(f"Total Episodes: {final_eval['total_episodes']}")
        print(f"Final Success Rate (SR): {final_eval['overall_sr']:.3f} ({final_eval['successful_episodes']}/{final_eval['total_episodes']})")
        print(f"Final SPL: {final_eval['overall_spl']:.3f}")
        
        print("\n" + "-"*40)
        print("ORIGINAL MODEL PERFORMANCE")
        print("-"*40)
        
        orig = summary["original_model_performance"]
        print(f"Common Episodes: {orig['common_episodes']}")
        print(f"Model 1 - SR: {orig['model1']['sr']:.3f}, SPL: {orig['model1']['spl']:.3f}")
        print(f"Model 2 - SR: {orig['model2']['sr']:.3f}, SPL: {orig['model2']['spl']:.3f}")
        
        print("\n" + "-"*40)
        print("DISCRIMINATION IMPACT")
        print("-"*40)
        
        discrim = summary["discrimination_impact"]
        print(f"Controversial Episodes: {discrim['controversial_episodes']}")
        print(f"Controversy Rate: {discrim['controversy_rate']:.3f}")
        print(f"Episodes Changed by Discrimination: {discrim['discrimination_changes']}")
        
        print("\n" + "-"*40)
        print("EPISODE SOURCE BREAKDOWN")
        print("-"*40)
        
        for source, count in summary["episode_sources"].items():
            percentage = count / final_eval['total_episodes'] * 100
            print(f"{source}: {count} ({percentage:.1f}%)")
        
        if "discrimination_summary" in summary and summary["discrimination_summary"]:
            print("\n" + "-"*40)
            print("DISCRIMINATION DETAILS")
            print("-"*40)
            
            disc_sum = summary["discrimination_summary"]
            print(f"Model 1 Wins: {disc_sum.get('model1_wins', 0)} ({disc_sum.get('model1_win_rate', 0):.1%})")
            print(f"Model 2 Wins: {disc_sum.get('model2_wins', 0)} ({disc_sum.get('model2_win_rate', 0):.1%})")
            print(f"No Decision: {disc_sum.get('no_decision', 0)}")
            print(f"High Confidence Decisions: {disc_sum.get('high_confidence_decisions', 0)}")
    
    def run_evaluation(self, output_path: str = "final_evaluation_results.json"):
        """Run the complete final evaluation"""
        self.logger.info("Starting final evaluation...")
        
        final_results, summary = self.combine_results()
        
        self.save_results(final_results, summary, output_path)
        self.print_summary(summary)
        
        return final_results, summary


def main():
    parser = argparse.ArgumentParser(description="Final Evaluation Script")
    parser.add_argument("--model1_dir", required=True, help="Path to Model 1 output directory")
    parser.add_argument("--model2_dir", required=True, help="Path to Model 2 output directory") 
    parser.add_argument("--discrimination_results", required=True, help="Path to discrimination results JSON file")
    parser.add_argument("--output", default="final_evaluation_results.json", help="Output file path")
    
    args = parser.parse_args()
    
    evaluator = FinalEvaluator(
        model1_dir=args.model1_dir,
        model2_dir=args.model2_dir,
        discrimination_results_path=args.discrimination_results
    )
    
    evaluator.run_evaluation(args.output)


if __name__ == "__main__":
    main()
