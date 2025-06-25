#!/usr/bin/env python3
"""
Simple launcher script for the Habitat Map Navigation Project.

This script provides an easy way to start the navigation system
with different configurations and scenes.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

from main_controller import NavigationController


def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(
        description="Launch Habitat Map Navigation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch.py                           # Use default configuration
  python launch.py --scene 17DRP5sb8fy      # Use specific scene
  python launch.py --output custom_images   # Custom output directory
  python launch.py --config my_config.yaml  # Custom configuration
        """
    )
    
    parser.add_argument(
        '--config', 
        default='configs/navigation_config.yaml',
        help='Path to configuration file (default: configs/navigation_config.yaml)'
    )
    
    parser.add_argument(
        '--scene',
        help='Scene ID to load (overrides config file)'
    )
    
    parser.add_argument(
        '--output',
        default='output_images',
        help='Output directory for images (default: output_images)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = current_dir / config_path
    
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = current_dir / output_dir
    
    # Check if config file exists
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        print("Please check the path or create the configuration file.")
        return 1
    
    # Print startup information
    print("="*60)
    print("HABITAT MAP NAVIGATION SYSTEM LAUNCHER")
    print("="*60)
    if args.verbose:
        print(f"Configuration: {config_path}")
        print(f"Output Directory: {output_dir}")
        print(f"Scene Override: {args.scene or 'None (using config)'}")
        print("-"*60)
    
    try:
        # Initialize controller
        print("Initializing navigation controller...")
        controller = NavigationController(
            config_path=str(config_path),
            output_dir=str(output_dir),
            scene_id=args.scene
        )
        
        # Initialize system
        print("Starting Habitat environment...")
        if controller.initialize_system():
            print("System ready! Starting interactive mode...")
            controller.run_interactive_loop()
        else:
            print("ERROR: Failed to initialize system")
            print("\nTroubleshooting tips:")
            print("1. Check that habitat-sim and habitat-lab are properly installed")
            print("2. Verify MP3D dataset is downloaded and configured")
            print("3. Ensure the scene ID exists in your dataset")
            print("4. Check file permissions for data directories")
            return 1
    
    except KeyboardInterrupt:
        print("\nReceived interrupt signal. Shutting down...")
    
    except Exception as e:
        print(f"ERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        if 'controller' in locals():
            controller.cleanup()
        print("System shutdown complete.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
