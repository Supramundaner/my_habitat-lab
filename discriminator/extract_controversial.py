#!/usr/bin/env python3
"""
Extract Controversial Episodes

This utility script extracts and analyzes controversial episodes between two models
without running the full discrimination process. Useful for initial analysis and debugging.

Usage:
    python extract_controversial.py --model1_dir /path/to/model1/output --model2_dir /path/to/model2/output
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd


class ControversyExtractor:
    """Extract and analyze controversial episodes"""
    
    def __init__(self, model1_dir: str, model2_dir: str):
        self.model1_dir = Path(model1_dir)
        self.model2_dir = Path(model2_dir)
        
    def load_batch_results(self, model_dir: Path) -> Dict[str, Dict[str, Any]]:
        """Load batch results from model directory"""
        batch_file = model_dir / "batch_output.json"
        if not batch_file.exists():
            raise FileNotFoundError(f"Batch output file not found: {batch_file}")
            
        with open(batch_file, 'r') as f:
            data = json.load(f)
        
        return data.get('episode_details', {})
    
    def extract_controversies(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Extract controversial episodes and generate statistics"""
        print("Loading model results...")
        
        model1_results = self.load_batch_results(self.model1_dir)
        model2_results = self.load_batch_results(self.model2_dir)
        
        # Find common episodes
        common_episodes = set(model1_results.keys()) & set(model2_results.keys())
        print(f"Common episodes: {len(common_episodes)}")
        
        controversial = []
        agreements = []
        
        for episode_key in common_episodes:
            m1_result = model1_results[episode_key]
            m2_result = model2_results[episode_key]
            
            m1_success = m1_result.get('success', False)
            m2_success = m2_result.get('success', False)
            
            episode_data = {
                'episode_key': episode_key,
                'scene_id': episode_key.split('/')[0],
                'episode_id': episode_key.split('/')[1],
                'model1_success': m1_success,
                'model1_sr': m1_result.get('sr', False),
                'model1_spl': m1_result.get('spl', 0.0),
                'model1_geodesic_distance': m1_result.get('geodesic_distance_to_target', 0.0),
                'model1_path_length': m1_result.get('path_length', 0.0),
                'model2_success': m2_success,
                'model2_sr': m2_result.get('sr', False),
                'model2_spl': m2_result.get('spl', 0.0),
                'model2_geodesic_distance': m2_result.get('geodesic_distance_to_target', 0.0),
                'model2_path_length': m2_result.get('path_length', 0.0),
                'object_category': m1_result.get('object_category', 'unknown')
            }
            
            if m1_success != m2_success:
                episode_data['controversy_type'] = 'model1_success' if m1_success else 'model2_success'
                controversial.append(episode_data)
            else:
                episode_data['agreement_type'] = 'both_success' if m1_success else 'both_failure'
                agreements.append(episode_data)
        
        # Generate statistics
        stats = self.generate_statistics(controversial, agreements, model1_results, model2_results)
        
        return controversial, stats
    
    def generate_statistics(self, controversial: List[Dict[str, Any]], 
                          agreements: List[Dict[str, Any]],
                          model1_results: Dict[str, Any], 
                          model2_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive statistics"""
        
        total_common = len(controversial) + len(agreements)
        
        # Controversy statistics
        model1_only_success = len([c for c in controversial if c['controversy_type'] == 'model1_success'])
        model2_only_success = len([c for c in controversial if c['controversy_type'] == 'model2_success'])
        
        # Agreement statistics
        both_success = len([a for a in agreements if a['agreement_type'] == 'both_success'])
        both_failure = len([a for a in agreements if a['agreement_type'] == 'both_failure'])
        
        # Overall model performance on common episodes
        common_episodes = controversial + agreements
        
        m1_total_success = sum(1 for ep in common_episodes if ep['model1_success'])
        m2_total_success = sum(1 for ep in common_episodes if ep['model2_success'])
        
        m1_total_spl = sum(ep['model1_spl'] for ep in common_episodes) / len(common_episodes)
        m2_total_spl = sum(ep['model2_spl'] for ep in common_episodes) / len(common_episodes)
        
        # Object category analysis
        category_controversy = {}
        for episode in controversial:
            category = episode['object_category']
            if category not in category_controversy:
                category_controversy[category] = {'model1_wins': 0, 'model2_wins': 0, 'total': 0}
            
            category_controversy[category]['total'] += 1
            if episode['controversy_type'] == 'model1_success':
                category_controversy[category]['model1_wins'] += 1
            else:
                category_controversy[category]['model2_wins'] += 1
        
        stats = {
            'overall': {
                'total_common_episodes': total_common,
                'controversial_episodes': len(controversial),
                'agreement_episodes': len(agreements),
                'controversy_rate': len(controversial) / total_common if total_common > 0 else 0,
                'agreement_rate': len(agreements) / total_common if total_common > 0 else 0
            },
            'controversy_breakdown': {
                'model1_only_success': model1_only_success,
                'model2_only_success': model2_only_success,
                'model1_advantage': model1_only_success - model2_only_success
            },
            'agreement_breakdown': {
                'both_success': both_success,
                'both_failure': both_failure
            },
            'model_performance': {
                'model1': {
                    'success_count': m1_total_success,
                    'success_rate': m1_total_success / total_common if total_common > 0 else 0,
                    'average_spl': m1_total_spl
                },
                'model2': {
                    'success_count': m2_total_success,
                    'success_rate': m2_total_success / total_common if total_common > 0 else 0,
                    'average_spl': m2_total_spl
                }
            },
            'category_analysis': category_controversy,
            'total_episodes_per_model': {
                'model1': len(model1_results),
                'model2': len(model2_results)
            }
        }
        
        return stats
    
    def save_results(self, controversial: List[Dict[str, Any]], stats: Dict[str, Any], output_dir: str):
        """Save extraction results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save controversial episodes
        with open(output_path / "controversial_episodes.json", 'w') as f:
            json.dump(controversial, f, indent=2)
        
        # Save statistics
        with open(output_path / "controversy_statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save CSV for easy analysis
        if controversial:
            df = pd.DataFrame(controversial)
            df.to_csv(output_path / "controversial_episodes.csv", index=False)
        
        print(f"Results saved to: {output_path}")
    
    def print_summary(self, stats: Dict[str, Any]):
        """Print formatted summary"""
        print("\n" + "="*60)
        print("CONTROVERSY ANALYSIS SUMMARY")
        print("="*60)
        
        overall = stats['overall']
        print(f"Total Common Episodes: {overall['total_common_episodes']}")
        print(f"Controversial Episodes: {overall['controversial_episodes']} ({overall['controversy_rate']:.1%})")
        print(f"Agreement Episodes: {overall['agreement_episodes']} ({overall['agreement_rate']:.1%})")
        
        print("\n" + "-"*40)
        print("CONTROVERSY BREAKDOWN")
        print("-"*40)
        
        controversy = stats['controversy_breakdown']
        print(f"Model 1 Only Success: {controversy['model1_only_success']}")
        print(f"Model 2 Only Success: {controversy['model2_only_success']}")
        print(f"Model 1 Advantage: {controversy['model1_advantage']:+d}")
        
        print("\n" + "-"*40)
        print("AGREEMENT BREAKDOWN")
        print("-"*40)
        
        agreement = stats['agreement_breakdown']
        print(f"Both Models Success: {agreement['both_success']}")
        print(f"Both Models Failure: {agreement['both_failure']}")
        
        print("\n" + "-"*40)
        print("MODEL PERFORMANCE ON COMMON EPISODES")
        print("-"*40)
        
        perf = stats['model_performance']
        print(f"Model 1: {perf['model1']['success_count']}/{overall['total_common_episodes']} " +
              f"({perf['model1']['success_rate']:.1%}), SPL: {perf['model1']['average_spl']:.3f}")
        print(f"Model 2: {perf['model2']['success_count']}/{overall['total_common_episodes']} " +
              f"({perf['model2']['success_rate']:.1%}), SPL: {perf['model2']['average_spl']:.3f}")
        
        print("\n" + "-"*40)
        print("OBJECT CATEGORY ANALYSIS")
        print("-"*40)
        
        categories = stats['category_analysis']
        if categories:
            for category, data in sorted(categories.items()):
                total = data['total']
                m1_wins = data['model1_wins']
                m2_wins = data['model2_wins']
                print(f"{category}: {total} controversial ({m1_wins} M1 wins, {m2_wins} M2 wins)")
        else:
            print("No category-specific controversies found")
        
        print("\n" + "-"*40)
        print("DATASET COVERAGE")
        print("-"*40)
        
        coverage = stats['total_episodes_per_model']
        print(f"Model 1 Total Episodes: {coverage['model1']}")
        print(f"Model 2 Total Episodes: {coverage['model2']}")
    
    def run_extraction(self, output_dir: str = "controversy_analysis"):
        """Run the complete extraction process"""
        print("Starting controversy extraction...")
        
        controversial, stats = self.extract_controversies()
        
        self.save_results(controversial, stats, output_dir)
        self.print_summary(stats)
        
        return controversial, stats


def main():
    parser = argparse.ArgumentParser(description="Extract Controversial Episodes")
    parser.add_argument("--model1_dir", required=True, help="Path to Model 1 output directory")
    parser.add_argument("--model2_dir", required=True, help="Path to Model 2 output directory")
    parser.add_argument("--output_dir", default="controversy_analysis", help="Output directory")
    
    args = parser.parse_args()
    
    extractor = ControversyExtractor(
        model1_dir=args.model1_dir,
        model2_dir=args.model2_dir
    )
    
    extractor.run_extraction(args.output_dir)


if __name__ == "__main__":
    main()
