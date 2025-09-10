#!/usr/bin/env python3
"""
Discriminator Pipeline Runner

This script runs the complete discriminator pipeline:
1. Extract controversial episodes
2. Run discrimination analysis
3. Generate final evaluation

Usage:
    python run_discriminator_pipeline.py --model1_dir /path/to/model1 --model2_dir /path/to/model2 [options]
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description} failed!")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Complete Discriminator Pipeline")
    parser.add_argument("--model1_dir", required=True, help="Path to Model 1 output directory")
    parser.add_argument("--model2_dir", required=True, help="Path to Model 2 output directory")
    parser.add_argument("--output_dir", help="Base output directory (default: discriminator_pipeline_TIMESTAMP)")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--skip_extraction", action="store_true", help="Skip controversy extraction step")
    parser.add_argument("--skip_discrimination", action="store_true", help="Skip discrimination step")
    parser.add_argument("--skip_final", action="store_true", help="Skip final evaluation step")
    parser.add_argument("--discrimination_results", help="Path to existing discrimination results (for final eval only)")
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"discriminator_pipeline_{timestamp}")
    
    output_dir.mkdir(exist_ok=True)
    
    print(f"Pipeline output directory: {output_dir}")
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    # Step 1: Extract controversial episodes
    if not args.skip_extraction:
        extract_output = output_dir / "controversy_analysis"
        cmd = [
            sys.executable,
            str(script_dir / "extract_controversial.py"),
            "--model1_dir", args.model1_dir,
            "--model2_dir", args.model2_dir,
            "--output_dir", str(extract_output)
        ]
        
        if not run_command(cmd, "Controversial Episode Extraction"):
            print("Pipeline failed at extraction step!")
            return 1
    
    # Step 2: Run discrimination
    discrimination_results_path = None
    if not args.skip_discrimination:
        discrimination_output = output_dir / "discrimination_results"
        cmd = [
            sys.executable,
            str(script_dir / "discriminator_system.py"),
            "--model1_dir", args.model1_dir,
            "--model2_dir", args.model2_dir,
            "--output_dir", str(discrimination_output)
        ]
        
        if args.config:
            cmd.extend(["--config", args.config])
        
        if not run_command(cmd, "Discrimination Analysis"):
            print("Pipeline failed at discrimination step!")
            return 1
        
        discrimination_results_path = discrimination_output / "discrimination_results.json"
    
    # Step 3: Generate final evaluation
    if not args.skip_final:
        if args.discrimination_results:
            discrimination_results_path = args.discrimination_results
        elif discrimination_results_path is None:
            print("ERROR: No discrimination results available for final evaluation!")
            print("Either run discrimination step or provide --discrimination_results path")
            return 1
        
        final_output = output_dir / "final_evaluation_results.json"
        cmd = [
            sys.executable,
            str(script_dir / "final_evaluation.py"),
            "--model1_dir", args.model1_dir,
            "--model2_dir", args.model2_dir,
            "--discrimination_results", str(discrimination_results_path),
            "--output", str(final_output)
        ]
        
        if not run_command(cmd, "Final Evaluation"):
            print("Pipeline failed at final evaluation step!")
            return 1
    
    # Create summary
    summary_file = output_dir / "pipeline_summary.json"
    summary = {
        "pipeline_run": {
            "timestamp": datetime.now().isoformat(),
            "model1_dir": args.model1_dir,
            "model2_dir": args.model2_dir,
            "output_dir": str(output_dir),
            "config_file": args.config,
            "steps_completed": []
        },
        "outputs": {}
    }
    
    if not args.skip_extraction:
        summary["pipeline_run"]["steps_completed"].append("extraction")
        summary["outputs"]["controversy_analysis"] = str(output_dir / "controversy_analysis")
    
    if not args.skip_discrimination:
        summary["pipeline_run"]["steps_completed"].append("discrimination")
        summary["outputs"]["discrimination_results"] = str(output_dir / "discrimination_results")
    
    if not args.skip_final:
        summary["pipeline_run"]["steps_completed"].append("final_evaluation")
        summary["outputs"]["final_evaluation"] = str(output_dir / "final_evaluation_results.json")
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print('='*60)
    print(f"Output directory: {output_dir}")
    print(f"Summary file: {summary_file}")
    
    # Print quick summary if final evaluation was run
    if not args.skip_final and (output_dir / "final_evaluation_results.json").exists():
        try:
            with open(output_dir / "final_evaluation_results.json", 'r') as f:
                final_data = json.load(f)
            
            final_eval = final_data["summary"]["final_evaluation"]
            print(f"\nFINAL RESULTS:")
            print(f"  Total Episodes: {final_eval['total_episodes']}")
            print(f"  Final Success Rate: {final_eval['overall_sr']:.3f}")
            print(f"  Final SPL: {final_eval['overall_spl']:.3f}")
            
            controversy = final_data["summary"]["discrimination_impact"]
            print(f"  Controversial Episodes: {controversy['controversial_episodes']}")
            print(f"  Controversy Rate: {controversy['controversy_rate']:.3f}")
            
        except Exception as e:
            print(f"Could not parse final results: {e}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
