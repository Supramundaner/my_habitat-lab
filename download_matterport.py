#!/usr/bin/env python3
"""
Download Matterport3D Dataset for Habitat

This script downloads the Matterport3D dataset required for coordinate navigation.
You need to have access to the Matterport3D dataset through academic license.
"""

import os
import subprocess
import sys
from pathlib import Path


def download_matterport_dataset(data_path: str = None):
    """
    Download Matterport3D dataset using habitat_sim utilities
    
    Args:
        data_path: Path where to download the data (default: ./data)
    """
    if data_path is None:
        data_path = os.path.join(os.getcwd(), "data")
    
    data_path = Path(data_path)
    data_path.mkdir(exist_ok=True)
    
    print(f"Downloading Matterport3D dataset to: {data_path}")
    print("This requires academic access to Matterport3D dataset...")
    
    try:
        # Download using habitat_sim dataset downloader
        cmd = [
            sys.executable, 
            "-m", 
            "habitat_sim.utils.datasets_download",
            "--uids", "mp3d",
            "--data-path", str(data_path),
            "--no-replace"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Download completed successfully!")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        print(f"Error output: {e.stderr}")
        print("\nAlternative download methods:")
        print("1. Manual download from Matterport3D official website")
        print("2. Use academic license to access the dataset")
        print("3. Download sample scenes for testing:")
        
        # Try to download test scenes instead
        try_download_test_scenes(data_path)
        
    except FileNotFoundError:
        print("habitat_sim not found. Please install habitat-sim first:")
        print("pip install habitat-sim")


def try_download_test_scenes(data_path: str):
    """
    Try to download test scenes as alternative
    """
    print("\nTrying to download test scenes instead...")
    
    try:
        cmd = [
            sys.executable,
            "-m", 
            "habitat_sim.utils.datasets_download",
            "--uids", "habitat_test_scenes",
            "--data-path", str(data_path),
            "--no-replace"
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Test scenes downloaded successfully!")
        print("You can use these for testing the navigation system.")
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to download test scenes: {e}")
        print("Please check your internet connection and habitat-sim installation.")


def check_existing_data(data_path: str = None):
    """
    Check if Matterport data already exists
    """
    if data_path is None:
        data_path = os.path.join(os.getcwd(), "data")
    
    mp3d_path = os.path.join(data_path, "scene_datasets", "mp3d")
    test_scenes_path = os.path.join(data_path, "scene_datasets", "habitat-test-scenes")
    
    print("Checking existing data...")
    
    if os.path.exists(mp3d_path):
        scenes = [d for d in os.listdir(mp3d_path) if os.path.isdir(os.path.join(mp3d_path, d))]
        print(f"Found Matterport3D data with {len(scenes)} scenes:")
        for scene in scenes[:5]:  # Show first 5
            print(f"  - {scene}")
        if len(scenes) > 5:
            print(f"  ... and {len(scenes) - 5} more scenes")
        return True
    
    elif os.path.exists(test_scenes_path):
        scenes = [d for d in os.listdir(test_scenes_path) if os.path.isdir(os.path.join(test_scenes_path, d))]
        print(f"Found test scenes with {len(scenes)} scenes:")
        for scene in scenes:
            print(f"  - {scene}")
        return True
    
    else:
        print("No Matterport or test scene data found.")
        return False


def setup_alternative_datasets():
    """
    Setup alternative datasets if Matterport is not available
    """
    print("\nSetting up alternative datasets...")
    
    data_path = os.path.join(os.getcwd(), "data")
    
    datasets_to_try = [
        ("habitat_test_scenes", "Habitat Test Scenes"),
        ("replica_cad_dataset", "Replica CAD Dataset"),
        ("hm3d_minival", "HM3D Minival Dataset")
    ]
    
    for dataset_id, dataset_name in datasets_to_try:
        try:
            print(f"\nTrying to download {dataset_name}...")
            cmd = [
                sys.executable,
                "-m",
                "habitat_sim.utils.datasets_download", 
                "--uids", dataset_id,
                "--data-path", data_path,
                "--no-replace"
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"Successfully downloaded {dataset_name}")
            return True
            
        except subprocess.CalledProcessError:
            print(f"Failed to download {dataset_name}")
            continue
    
    print("Could not download any alternative datasets.")
    return False


def main():
    """Main function"""
    print("=== Habitat Matterport Dataset Downloader ===")
    
    # Check command line arguments
    data_path = None
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    
    # Check existing data first
    if check_existing_data(data_path):
        print("\nDataset already exists. You can start using the navigation system!")
        return
    
    # Try to download Matterport
    print("\nDownloading datasets...")
    download_matterport_dataset(data_path)
    
    # If that fails, try alternatives
    if not check_existing_data(data_path):
        print("\nMatterport download failed. Trying alternative datasets...")
        setup_alternative_datasets()
    
    # Final check
    if check_existing_data(data_path):
        print("\n✅ Dataset setup complete!")
        print("You can now run the coordinate navigation system:")
        print("python examples/coordinate_navigation.py")
    else:
        print("\n❌ Failed to setup any datasets.")
        print("Please check the following:")
        print("1. Internet connection")
        print("2. habitat-sim installation")
        print("3. Academic access to Matterport3D dataset")


if __name__ == "__main__":
    main()
