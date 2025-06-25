#!/usr/bin/env python3
"""
Enhanced Launch Script for Habitat Lab Navigation

This script launches the enhanced coordinate navigation system with all new features:
- HM3D dataset prioritization
- Enhanced coordinate system with grid overlay
- User-selectable starting positions
"""

import os
import sys

def main():
    """Main launch function"""
    print("🏠 Enhanced Habitat Lab Coordinate Navigation System")
    print("="*60)
    
    # Check if this is the first run
    workspace_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(workspace_path), "data")
    scene_datasets_path = os.path.join(data_path, "scene_datasets")
    
    # Check for datasets
    has_datasets = os.path.exists(scene_datasets_path) and os.listdir(scene_datasets_path)
    
    if not has_datasets:
        print("📋 First time setup detected!")
        print("🔽 You need to download datasets before using the system.")
        print("\nOptions:")
        print("1. Download HM3D dataset (recommended for navigation research)")
        print("2. Download test scenes (good for development)")
        print("3. Manual dataset setup")
        print("4. Exit")
        
        while True:
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '1':
                print("🔄 Launching HM3D dataset download...")
                os.system(f"cd {workspace_path} && python download_datasets.py")
                break
            elif choice == '2':
                print("🔄 Launching test scenes download...")
                os.system(f"cd {workspace_path} && python download_datasets.py")
                break
            elif choice == '3':
                print("📖 Manual setup instructions:")
                print("   Run: python download_datasets.py")
                print("   Or visit: https://aihabitat.org/datasets/hm3d/")
                return
            elif choice == '4':
                print("👋 Exiting...")
                return
            else:
                print("❌ Invalid choice. Please select 1-4.")
    
    # Launch the enhanced interface
    print("\n🚀 Launching Enhanced Terminal Interface...")
    print("✨ New Features:")
    print("   • Enhanced coordinate system with grid overlay")
    print("   • HM3D dataset prioritization")
    print("   • Interactive position selection")
    print("   • Improved visualization")
    print()
    
    # Add src directory to path and launch
    sys.path.append(os.path.join(workspace_path, 'src'))
    
    try:
        from enhanced_terminal_interface import EnhancedTerminalInterface
        
        # Create and run interface
        interface = EnhancedTerminalInterface()
        
        # Initialize navigation system
        if interface.initialize_system():
            print("🎯 Enhanced coordinate navigation system ready!")
            print("💡 The system will first show you a coordinate map to choose your starting position.")
            print("🗺️  Look for gray areas (navigable) when selecting coordinates.")
            print()
            
            # Run main loop
            interface.run()
        else:
            print("❌ Failed to initialize navigation system")
            print("💡 Try running: python download_datasets.py")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Try running the test script first: python test_enhanced_coordinate_system.py")

if __name__ == "__main__":
    main()
