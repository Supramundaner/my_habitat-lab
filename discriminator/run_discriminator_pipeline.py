#!/usr/bin/env python3
"""
Discriminator Pipeline Runner

This script runs the complete discriminator pipeline:
1. Extract controversial episodes
2. Run discrimination analysis
3. Generate final evaluation

Usage:
    python run_discriminator_pipeline.py [--config_path /path/to/config.json] [options]
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


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
    parser.add_argument("--config_path", default="discriminator_config.json", 
                       help="Path to configuration file")
    parser.add_argument("--skip_extraction", action="store_true", help="Skip controversy extraction step")
    parser.add_argument("--skip_discrimination", action="store_true", help="Skip discrimination step")
    parser.add_argument("--skip_final", action="store_true", help="Skip final evaluation step")
    parser.add_argument("--discrimination_results", help="Path to existing discrimination results (for final eval only)")
    parser.add_argument("--controversial_episodes", help="Path to existing controversial episodes JSON file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config_path)
    
    # Extract paths from config
    model1_dir = config['model_paths']['model1_output']
    model2_dir = config['model_paths']['model2_output']
    output_dir = Path(config['output_config']['discriminator_output'])
    
    output_dir.mkdir(exist_ok=True)
    
    print(f"Pipeline output directory: {output_dir}")
    print(f"Model 1 directory: {model1_dir}")
    print(f"Model 2 directory: {model2_dir}")
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    # Step 1: Extract controversial episodes or use existing file
    if args.controversial_episodes:
        # Use existing controversial episodes file
        controversial_episodes_file = args.controversial_episodes
        print(f"Using existing controversial episodes file: {controversial_episodes_file}")
        
        # Copy the file to the expected location if it's different
        expected_path = config['output_config']['controversial_episodes_file']
        if Path(controversial_episodes_file).resolve() != Path(expected_path).resolve():
            import shutil
            shutil.copy2(controversial_episodes_file, expected_path)
            print(f"Copied controversial episodes file to: {expected_path}")
        
    elif not args.skip_extraction:
        cmd = [
            sys.executable,
            str(script_dir / "extract_controversial.py"),
            "--config_path", args.config_path
        ]
        
        if not run_command(cmd, "Controversial Episode Extraction"):
            print("Pipeline failed at extraction step!")
            return 1
    else:
        # Check if the expected controversial episodes file exists
        expected_path = config['output_config']['controversial_episodes_file']
        if not Path(expected_path).exists():
            print(f"ERROR: No controversial episodes file found at {expected_path}")
            print("Either run extraction step or provide --controversial_episodes path")
            return 1
    
    # Step 2: Run discrimination
    discrimination_results_path = None
    if not args.skip_discrimination:
        cmd = [
            sys.executable,
            str(script_dir / "discriminator_system.py"),
            "--config_path", args.config_path
        ]
        
        # Add controversial episodes file if provided or if using existing
        if args.controversial_episodes:
            cmd.extend(["--controversial_episodes", args.controversial_episodes])
        elif args.skip_extraction:
            # Use the default controversial episodes file location
            expected_path = config['output_config']['controversial_episodes_file']
            cmd.extend(["--controversial_episodes", expected_path])
        
        if not run_command(cmd, "Discrimination Analysis"):
            print("Pipeline failed at discrimination step!")
            return 1
        
        discrimination_results_path = output_dir / "discrimination_results.json"
    
    # Step 3: Generate final evaluation
    if not args.skip_final:
        cmd = [
            sys.executable,
            str(script_dir / "final_evaluation.py"),
            "--config_path", args.config_path
        ]
        
        if args.discrimination_results:
            cmd.extend(["--discrimination_results", args.discrimination_results])
        elif discrimination_results_path is None and not (output_dir / "discrimination_results.json").exists():
            print("ERROR: No discrimination results available for final evaluation!")
            print("Either run discrimination step or provide --discrimination_results path")
            return 1
        
        if not run_command(cmd, "Final Evaluation"):
            print("Pipeline failed at final evaluation step!")
            return 1
    
    # Create summary
    summary_file = output_dir / "pipeline_summary.json"
    summary = {
        "pipeline_run": {
            "timestamp": datetime.now().isoformat(),
            "config_file": args.config_path,
            "model1_dir": model1_dir,
            "model2_dir": model2_dir,
            "output_dir": str(output_dir),
            "steps_completed": []
        },
        "outputs": {}
    }
    
    if not args.skip_extraction:
        summary["pipeline_run"]["steps_completed"].append("extraction")
        summary["outputs"]["controversial_episodes"] = config['output_config']['controversial_episodes_file']
    
    if not args.skip_discrimination:
        summary["pipeline_run"]["steps_completed"].append("discrimination")
        summary["outputs"]["discrimination_results"] = str(output_dir / "discrimination_results.json")
    
    if not args.skip_final:
        summary["pipeline_run"]["steps_completed"].append("final_evaluation")
        summary["outputs"]["final_evaluation"] = config['output_config']['final_results_file']
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print('='*60)
    print(f"Output directory: {output_dir}")
    print(f"Summary file: {summary_file}")
    
    # Print quick summary if final evaluation was run
    final_results_file = config['output_config']['final_results_file']
    if not args.skip_final and Path(final_results_file).exists():
        try:
            with open(final_results_file, 'r') as f:
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
