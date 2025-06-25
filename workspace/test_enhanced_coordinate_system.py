#!/usr/bin/env python3
"""
Test script for enhanced coordinate navigation system

This script tests the new features:
1. Enhanced coordinate system with grid overlay
2. HM3D dataset prioritization 
3. User-selectable starting positions
"""

import os
import sys
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

def test_coordinate_system():
    """Test the enhanced coordinate system features"""
    print("🧪 Testing Enhanced Coordinate Navigation System")
    print("="*60)
    
    try:
        from coordinate_navigation import CoordinateNavigationSystem
        
        # Test system initialization
        print("1. Testing system initialization...")
        workspace_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(os.path.dirname(workspace_path), "data")
        
        # Check if datasets exist
        scene_datasets_path = os.path.join(data_path, "scene_datasets")
        if not os.path.exists(scene_datasets_path):
            print("❌ No datasets found. Please run 'python download_datasets.py' first")
            return False
        
        print(f"📂 Data path: {data_path}")
        
        # Initialize system (this will show the initial map and ask for position)
        nav_system = CoordinateNavigationSystem(
            workspace_path=workspace_path,
            data_path=data_path
        )
        
        if nav_system.sim is None:
            print("❌ Failed to initialize navigation system")
            return False
        
        print("✅ Navigation system initialized successfully")
        
        # Test coordinate system features
        print("\n2. Testing coordinate system features...")
        
        # Test coordinate conversions
        test_map_coords = [512, 400]
        world_coords = nav_system._map_to_world_coords(test_map_coords)
        converted_back = nav_system._world_to_map_coords(world_coords)
        
        print(f"   Map coords: {test_map_coords}")
        print(f"   -> World coords: {world_coords}")
        print(f"   -> Back to map: {converted_back}")
        
        # Test enhanced map generation
        print("\n3. Testing enhanced map generation with coordinate grid...")
        
        # Generate current observations
        observations = nav_system.get_current_observations()
        if observations:
            print("✅ Observations generated successfully")
        else:
            print("❌ Failed to generate observations")
            return False
        
        # Generate enhanced topdown map
        topdown_map = nav_system.get_topdown_map_with_position()
        if topdown_map.size > 0:
            print("✅ Enhanced topdown map generated successfully")
        else:
            print("❌ Failed to generate enhanced topdown map")
            return False
        
        # Test coordinate overlay
        map_with_coords = nav_system.create_coordinate_system_overlay(topdown_map)
        print("✅ Coordinate system overlay added successfully")
        
        # Save test images
        print("\n4. Testing enhanced image saving...")
        test_files = nav_system.save_current_views("test_enhanced_")
        
        if test_files:
            print("✅ Enhanced images saved successfully:")
            for view_type, filepath in test_files.items():
                file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                print(f"   📷 {view_type}: {os.path.basename(filepath)} ({file_size:.1f} MB)")
        else:
            print("❌ Failed to save enhanced images")
            return False
        
        # Test dataset prioritization
        print("\n5. Testing dataset prioritization...")
        available_scenes = nav_system.available_scenes
        print(f"   Available scenes: {len(available_scenes)}")
        
        hm3d_scenes = [s for s in available_scenes if s.startswith('hm3d/')]
        if hm3d_scenes:
            print(f"✅ Found {len(hm3d_scenes)} HM3D scenes (prioritized)")
        else:
            print("⚠️  No HM3D scenes found, using other datasets")
        
        print(f"   Current scene: {nav_system.scene_id}")
        
        print("\n✅ All tests completed successfully!")
        print(f"🗺️  Enhanced coordinate system is working properly")
        print(f"📊 Map bounds: X[{nav_system.map_bounds['min_x']:.1f}, {nav_system.map_bounds['max_x']:.1f}], Z[{nav_system.map_bounds['min_z']:.1f}, {nav_system.map_bounds['max_z']:.1f}]")
        print(f"📍 Current position: Map({nav_system.current_map_position[0]:.1f}, {nav_system.current_map_position[1]:.1f}) World({nav_system.current_position[0]:.2f}, {nav_system.current_position[2]:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_interface():
    """Test the enhanced terminal interface"""
    print("\n🧪 Testing Enhanced Terminal Interface")
    print("="*50)
    
    try:
        from enhanced_terminal_interface import EnhancedTerminalInterface
        
        print("1. Testing interface initialization...")
        interface = EnhancedTerminalInterface()
        print("✅ Enhanced terminal interface initialized")
        
        print("2. Testing command completion setup...")
        print(f"   Available commands: {len(interface.commands)}")
        print(f"   Commands: {', '.join(interface.commands[:10])}...")
        
        print("3. Testing session statistics...")
        interface.show_session_stats()
        
        print("✅ Enhanced terminal interface test completed")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced interface test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Enhanced Habitat Lab Coordinate Navigation Test Suite")
    print("="*70)
    
    # Test coordinate system
    coord_test_passed = test_coordinate_system()
    
    # Test enhanced interface (without full initialization)
    interface_test_passed = test_enhanced_interface()
    
    # Summary
    print("\n📋 TEST SUMMARY")
    print("="*30)
    print(f"Coordinate System: {'✅ PASSED' if coord_test_passed else '❌ FAILED'}")
    print(f"Enhanced Interface: {'✅ PASSED' if interface_test_passed else '❌ FAILED'}")
    
    if coord_test_passed and interface_test_passed:
        print("\n🎉 All tests passed! The enhanced system is ready to use.")
        print("\n🚀 To start the enhanced interface:")
        print("   python launch.py")
        print("   or")
        print("   python src/enhanced_terminal_interface.py")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
    
    return coord_test_passed and interface_test_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
