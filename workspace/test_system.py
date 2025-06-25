#!/usr/bin/env python3
"""
Test script for Habitat Lab Coordinate Navigation System

This script runs automated tests to verify system functionality.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from coordinate_navigation import CoordinateNavigationSystem
import time


def test_system_initialization():
    """Test system initialization"""
    print("🔧 Testing system initialization...")
    
    workspace_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(workspace_path), "data")
    
    nav_system = CoordinateNavigationSystem(
        workspace_path=workspace_path,
        data_path=data_path
    )
    
    if nav_system.sim is None:
        print("❌ System initialization failed")
        return False, nav_system
    
    print(f"✅ System initialized successfully with scene: {nav_system.scene_id}")
    return True, nav_system


def test_coordinate_conversion(nav_system):
    """Test coordinate conversion functionality"""
    print("\n🔧 Testing coordinate conversion...")
    
    # Test map to world conversion
    test_map_coords = [512, 256]  # Center-ish coordinates
    world_coords = nav_system._map_to_world_coords(test_map_coords)
    converted_back = nav_system._world_to_map_coords(world_coords)
    
    print(f"   Map coords: {test_map_coords}")
    print(f"   World coords: [{world_coords[0]:.2f}, {world_coords[1]:.2f}, {world_coords[2]:.2f}]")
    print(f"   Converted back: [{converted_back[0]:.1f}, {converted_back[1]:.1f}]")
    
    # Check if conversion is reasonably accurate
    diff_x = abs(test_map_coords[0] - converted_back[0])
    diff_y = abs(test_map_coords[1] - converted_back[1])
    
    if diff_x < 1.0 and diff_y < 1.0:
        print("✅ Coordinate conversion test passed")
        return True
    else:
        print(f"❌ Coordinate conversion test failed - difference too large: ({diff_x:.2f}, {diff_y:.2f})")
        return False


def test_navigation(nav_system):
    """Test navigation functionality"""
    print("\n🔧 Testing navigation...")
    
    initial_pos = nav_system.get_current_position()
    print(f"   Initial position: {initial_pos}")
    
    # Try to move to a different position
    target_map_x = initial_pos["map_position"][0] + 50
    target_map_y = initial_pos["map_position"][1] + 50
    
    print(f"   Attempting to move to map coordinates: ({target_map_x:.1f}, {target_map_y:.1f})")
    
    success = nav_system.move_to_map_coordinate(target_map_x, target_map_y)
    
    if success:
        new_pos = nav_system.get_current_position()
        print(f"   New position: {new_pos}")
        print("✅ Navigation test passed")
        return True
    else:
        print("⚠️  Navigation test - position not navigable (this is normal)")
        # Try a position closer to current location
        target_map_x = initial_pos["map_position"][0] + 10
        target_map_y = initial_pos["map_position"][1] + 10
        success = nav_system.move_to_map_coordinate(target_map_x, target_map_y)
        
        if success:
            print("✅ Navigation test passed (alternative position)")
            return True
        else:
            print("❌ Navigation test failed - no navigable positions found")
            return False


def test_view_rotation(nav_system):
    """Test view rotation functionality"""
    print("\n🔧 Testing view rotation...")
    
    initial_rotation = nav_system.current_rotation.copy()
    print(f"   Initial rotation: {initial_rotation}")
    
    # Test rotation
    nav_system.adjust_view_angle(45, 0)
    new_rotation = nav_system.current_rotation.copy()
    
    print(f"   New rotation: {new_rotation}")
    
    # Check if rotation changed
    if initial_rotation != new_rotation:
        print("✅ View rotation test passed")
        return True
    else:
        print("❌ View rotation test failed - rotation unchanged")
        return False


def test_observations(nav_system):
    """Test observation collection"""
    print("\n🔧 Testing observations...")
    
    observations = nav_system.get_current_observations()
    
    expected_sensors = ["color_sensor", "depth_sensor", "third_person_sensor"]
    missing_sensors = []
    
    for sensor in expected_sensors:
        if sensor not in observations:
            missing_sensors.append(sensor)
        else:
            shape = observations[sensor].shape
            print(f"   {sensor}: {shape}")
    
    if not missing_sensors:
        print("✅ Observations test passed")
        return True
    else:
        print(f"❌ Observations test failed - missing sensors: {missing_sensors}")
        return False


def test_topdown_map(nav_system):
    """Test topdown map generation"""
    print("\n🔧 Testing topdown map generation...")
    
    topdown_map = nav_system.get_topdown_map_with_position()
    
    if topdown_map.size > 0:
        print(f"   Topdown map shape: {topdown_map.shape}")
        print("✅ Topdown map test passed")
        return True
    else:
        print("❌ Topdown map test failed - empty map")
        return False


def test_image_saving(nav_system):
    """Test image saving functionality"""
    print("\n🔧 Testing image saving...")
    
    saved_files = nav_system.save_current_views("test_")
    
    print(f"   Images saved to: {nav_system.images_path}")
    
    all_files_exist = True
    for view_type, filepath in saved_files.items():
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            print(f"   {view_type}: {os.path.basename(filepath)} ({file_size} bytes)")
        else:
            print(f"   ❌ Missing file: {filepath}")
            all_files_exist = False
    
    if all_files_exist and saved_files:
        print("✅ Image saving test passed")
        return True
    else:
        print("❌ Image saving test failed")
        return False


def run_all_tests():
    """Run all tests"""
    print("🧪 Starting Habitat Lab Coordinate Navigation System Tests")
    print("="*60)
    
    # Initialize system
    success, nav_system = test_system_initialization()
    if not success:
        print("\n❌ Tests aborted - system initialization failed")
        print("💡 Please run 'python download_datasets.py' to download datasets")
        return
    
    # Run tests
    test_results = []
    
    test_results.append(("Coordinate Conversion", test_coordinate_conversion(nav_system)))
    test_results.append(("Navigation", test_navigation(nav_system)))
    test_results.append(("View Rotation", test_view_rotation(nav_system)))
    test_results.append(("Observations", test_observations(nav_system)))
    test_results.append(("Topdown Map", test_topdown_map(nav_system)))
    test_results.append(("Image Saving", test_image_saving(nav_system)))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("-"*60)
    print(f"Total: {passed + failed} tests, {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 All tests passed! System is ready to use.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the issues above.")
    
    # Cleanup
    nav_system.close()
    
    print(f"\n🚀 To start using the system:")
    print(f"   python src/interactive_navigation.py")


if __name__ == "__main__":
    run_all_tests()
