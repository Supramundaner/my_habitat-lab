#!/usr/bin/env python3
"""
Launcher script for Habitat Lab Navigation System

This script provides multiple ways to start the navigation system:
- Enhanced terminal interface (recommended)
- Basic interactive interface
- System tests
- Dataset downloader
"""

import sys
import os
import argparse
from pathlib import Path

# Add src directory to path
workspace_path = Path(__file__).parent
src_path = workspace_path / "src"
sys.path.append(str(src_path))


def run_enhanced_terminal():
    """Run the enhanced terminal interface"""
    try:
        from enhanced_terminal_interface import main as enhanced_main
        enhanced_main()
    except ImportError as e:
        print(f"❌ Failed to import enhanced terminal interface: {e}")
        print("💡 Falling back to basic interface...")
        run_basic_interface()


def run_basic_interface():
    """Run the basic interactive interface"""
    try:
        from interactive_navigation import main as basic_main
        basic_main()
    except ImportError as e:
        print(f"❌ Failed to import basic interface: {e}")
        sys.exit(1)


def run_tests():
    """Run system tests"""
    test_file = workspace_path / "test_system.py"
    if test_file.exists():
        os.system(f"python {test_file}")
    else:
        print("❌ Test file not found")


def run_dataset_downloader():
    """Run dataset downloader"""
    download_file = workspace_path / "download_datasets.py"
    if download_file.exists():
        os.system(f"python {download_file}")
    else:
        print("❌ Dataset downloader not found")


def show_banner():
    """Show startup banner"""
    print("="*80)
    print("🏠 HABITAT LAB COORDINATE NAVIGATION SYSTEM")
    print("   Multi-Interface Launcher")
    print("="*80)
    print()


def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description="Habitat Lab Navigation System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch.py                    # Start enhanced terminal interface
  python launch.py --basic            # Start basic interface
  python launch.py --test             # Run system tests
  python launch.py --download         # Download datasets
        """
    )
    
    parser.add_argument(
        '--basic', '-b',
        action='store_true',
        help='Use basic interactive interface instead of enhanced'
    )
    
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Run system tests'
    )
    
    parser.add_argument(
        '--download', '-d',
        action='store_true',
        help='Run dataset downloader'
    )
    
    parser.add_argument(
        '--list-interfaces', '-l',
        action='store_true',
        help='List available interfaces'
    )
    
    args = parser.parse_args()
    
    show_banner()
    
    if args.list_interfaces:
        print("📋 Available Interfaces:")
        print("  1. Enhanced Terminal Interface (default) - Advanced features with colors")
        print("  2. Basic Interactive Interface - Simple command-line interface")
        print("  3. System Tests - Automated testing suite")
        print("  4. Dataset Downloader - Download required datasets")
        return
    
    if args.test:
        print("🧪 Running system tests...")
        run_tests()
    elif args.download:
        print("📥 Starting dataset downloader...")
        run_dataset_downloader()
    elif args.basic:
        print("🚀 Starting basic interactive interface...")
        run_basic_interface()
    else:
        print("🚀 Starting enhanced terminal interface...")
        run_enhanced_terminal()


if __name__ == "__main__":
    main()
