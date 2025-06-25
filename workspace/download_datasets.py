#!/usr/bin/env python3
"""
Dataset Download Script for Habitat Lab Navigation

This script downloads the required datasets for the coordinate navigation system.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command: list, description: str) -> bool:
    """Run a command and return success status"""
    print(f"🔄 {description}")
    print(f"   Command: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success!")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed!")
        print(f"   Error: {e.stderr.strip()}")
        return False
    except FileNotFoundError:
        print(f"❌ {description} - Command not found!")
        print("   Please ensure habitat-sim is installed: pip install habitat-sim")
        return False


def check_existing_datasets(data_path: str) -> dict:
    """Check which datasets are already available"""
    datasets_status = {}
    scene_datasets_path = os.path.join(data_path, "scene_datasets")
    
    if not os.path.exists(scene_datasets_path):
        return datasets_status
    
    # Check for different datasets
    dataset_checks = [
        ("habitat-test-scenes", "habitat-test-scenes"),
        ("mp3d", "mp3d"),
        ("replica_cad", "replica_cad"),
        ("hm3d", "hm3d"),
    ]
    
    for dataset_name, dataset_dir in dataset_checks:
        dataset_path = os.path.join(scene_datasets_path, dataset_dir)
        if os.path.exists(dataset_path):
            # Count scenes
            scene_count = 0
            if dataset_name == "habitat-test-scenes":
                scene_count = len([f for f in os.listdir(dataset_path) 
                                 if f.endswith('.glb') or f.endswith('.ply')])
            else:
                scene_count = len([d for d in os.listdir(dataset_path) 
                                 if os.path.isdir(os.path.join(dataset_path, d))])
            
            datasets_status[dataset_name] = scene_count
    
    return datasets_status


def download_habitat_test_scenes(data_path: str) -> bool:
    """Download Habitat test scenes (recommended for development)"""
    command = [
        sys.executable, "-m", "habitat_sim.utils.datasets_download",
        "--uids", "habitat_test_scenes",
        "--data-path", data_path,
        "--no-replace"
    ]
    
    return run_command(command, "Downloading Habitat Test Scenes")


def download_matterport3d(data_path: str) -> bool:
    """Download Matterport3D dataset (requires academic license)"""
    print("⚠️  Note: Matterport3D requires academic license and registration!")
    print("   Visit: https://niessner.github.io/Matterport/")
    
    command = [
        sys.executable, "-m", "habitat_sim.utils.datasets_download",
        "--uids", "mp3d",
        "--data-path", data_path,
        "--no-replace"
    ]
    
    return run_command(command, "Downloading Matterport3D Dataset")


def download_hm3d_minival(data_path: str) -> bool:
    """Download HM3D minival dataset"""
    command = [
        sys.executable, "-m", "habitat_sim.utils.datasets_download",
        "--uids", "hm3d_minival",
        "--data-path", data_path,
        "--no-replace"
    ]
    
    return run_command(command, "Downloading HM3D Minival Dataset")


def download_replica_cad(data_path: str) -> bool:
    """Download Replica CAD dataset"""
    command = [
        sys.executable, "-m", "habitat_sim.utils.datasets_download",
        "--uids", "replica_cad_dataset",
        "--data-path", data_path,
        "--no-replace"
    ]
    
    return run_command(command, "Downloading Replica CAD Dataset")


def interactive_download(data_path: str):
    """Interactive dataset download"""
    print("🏠 HABITAT LAB DATASET DOWNLOADER")
    print("="*50)
    
    # Check existing datasets
    existing = check_existing_datasets(data_path)
    if existing:
        print("📋 Existing datasets found:")
        for dataset, count in existing.items():
            print(f"   {dataset}: {count} scenes")
        print()
    
    print("Available datasets to download:")
    print("1. Habitat Test Scenes (recommended for development)")
    print("2. Matterport3D (requires academic license)")
    print("3. HM3D Minival")
    print("4. Replica CAD")
    print("5. Download all available")
    print("6. Exit")
    
    while True:
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == '1':
            download_habitat_test_scenes(data_path)
            break
        elif choice == '2':
            download_matterport3d(data_path)
            break
        elif choice == '3':
            download_hm3d_minival(data_path)
            break
        elif choice == '4':
            download_replica_cad(data_path)
            break
        elif choice == '5':
            print("🔄 Downloading all available datasets...")
            download_habitat_test_scenes(data_path)
            download_hm3d_minival(data_path)
            download_replica_cad(data_path)
            # Note: Matterport3D usually requires manual setup
            print("ℹ️  Skipping Matterport3D (requires manual setup)")
            break
        elif choice == '6':
            print("👋 Exiting...")
            return
        else:
            print("❌ Invalid choice. Please select 1-6.")


def main():
    """Main function"""
    print("🏠 Habitat Lab Dataset Downloader")
    print("="*40)
    
    # Determine data path
    workspace_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(workspace_path), "data")
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    
    print(f"📂 Data directory: {data_path}")
    
    # Create data directory if it doesn't exist
    os.makedirs(data_path, exist_ok=True)
    
    # Check current status
    existing = check_existing_datasets(data_path)
    
    if not existing:
        print("📭 No datasets found. Starting download process...")
        interactive_download(data_path)
    else:
        print("📋 Current datasets:")
        for dataset, count in existing.items():
            print(f"   ✅ {dataset}: {count} scenes")
        
        print("\nOptions:")
        print("1. Download additional datasets")
        print("2. Re-download existing datasets")
        print("3. Exit")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            interactive_download(data_path)
        elif choice == '2':
            print("Re-downloading will overwrite existing datasets.")
            confirm = input("Continue? (y/N): ").strip().lower()
            if confirm == 'y':
                interactive_download(data_path)
        elif choice == '3':
            print("👋 Exiting...")
        else:
            print("❌ Invalid choice.")
    
    # Final status check
    final_status = check_existing_datasets(data_path)
    if final_status:
        print("\n✅ Dataset download complete!")
        print("📋 Available datasets:")
        for dataset, count in final_status.items():
            print(f"   {dataset}: {count} scenes")
        print(f"\n🚀 You can now run the navigation system:")
        print(f"   cd {workspace_path}")
        print(f"   python interactive_navigation.py")
    else:
        print("\n❌ No datasets were successfully downloaded.")
        print("Please check your internet connection and habitat-sim installation.")


if __name__ == "__main__":
    main()
