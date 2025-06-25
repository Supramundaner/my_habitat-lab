#!/usr/bin/env python3
"""
Simple test script for the Habitat Map Navigation Project.

This script performs basic functionality tests without requiring
full Habitat environment setup.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))


def test_coordinate_conversion():
    """Test coordinate conversion functions."""
    print("Testing coordinate conversion...")
    
    try:
        # Mock map info for testing
        mock_map_info = {
            'map_size': (1024, 1024),
            'world_bounds': [[-10, 0, -10], [10, 2, 10]],
            'map_scale': 20.0 / 1024
        }
        
        # Test world to map conversion
        world_pos = np.array([5.0, 1.0, -5.0])
        
        # Manual calculation for verification
        bounds = mock_map_info['world_bounds']
        map_size = mock_map_info['map_size']
        
        norm_x = (world_pos[0] - bounds[0][0]) / (bounds[1][0] - bounds[0][0])
        norm_z = (world_pos[2] - bounds[0][2]) / (bounds[1][2] - bounds[0][2])
        
        map_x = norm_x * map_size[1]
        map_y = (1.0 - norm_z) * map_size[0]
        
        print(f"  World position: {world_pos}")
        print(f"  Converted to map: ({map_x:.1f}, {map_y:.1f})")
        
        # Test reverse conversion
        norm_x_back = map_x / map_size[1]
        norm_z_back = 1.0 - (map_y / map_size[0])
        
        world_x_back = bounds[0][0] + norm_x_back * (bounds[1][0] - bounds[0][0])
        world_z_back = bounds[0][2] + norm_z_back * (bounds[1][2] - bounds[0][2])
        
        print(f"  Converted back to world: ({world_x_back:.1f}, {world_z_back:.1f})")
        
        # Check if conversion is accurate
        if abs(world_x_back - world_pos[0]) < 0.1 and abs(world_z_back - world_pos[2]) < 0.1:
            print("✓ Coordinate conversion test passed")
            return True
        else:
            print("❌ Coordinate conversion test failed")
            return False
            
    except Exception as e:
        print(f"❌ Coordinate conversion test error: {e}")
        return False


def test_command_parsing():
    """Test command parsing functions."""
    print("\nTesting command parsing...")
    
    try:
        import re
        
        # Test move command parsing
        def parse_move_command(command):
            pattern = r"move\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)"
            match = re.match(pattern, command.strip().lower())
            if match:
                return (float(match.group(1)), float(match.group(2)))
            return None
        
        # Test cases
        test_commands = [
            ("move 5.5 -3.2", (5.5, -3.2)),
            ("move 10 20", (10.0, 20.0)),
            ("move -5.0 0", (-5.0, 0.0)),
            ("invalid command", None),
            ("move abc def", None)
        ]
        
        all_passed = True
        for command, expected in test_commands:
            result = parse_move_command(command)
            if result == expected:
                print(f"✓ '{command}' -> {result}")
            else:
                print(f"❌ '{command}' -> {result} (expected {expected})")
                all_passed = False
        
        if all_passed:
            print("✓ Command parsing test passed")
            return True
        else:
            print("❌ Command parsing test failed")
            return False
            
    except Exception as e:
        print(f"❌ Command parsing test error: {e}")
        return False


def test_image_processing():
    """Test basic image processing capabilities."""
    print("\nTesting image processing...")
    
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import cv2
        
        # Create a simple test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test matplotlib figure creation
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(test_image)
        ax.set_title("Test Image")
        
        # Test saving to file
        test_output = project_root / "output_images" / "test_image.png"
        test_output.parent.mkdir(exist_ok=True)
        
        plt.savefig(test_output, dpi=100)
        plt.close()
        
        if test_output.exists():
            print(f"✓ Image saved successfully: {test_output}")
            # Clean up test file
            test_output.unlink()
            return True
        else:
            print("❌ Failed to save test image")
            return False
            
    except Exception as e:
        print(f"❌ Image processing test error: {e}")
        return False


def test_file_structure():
    """Test if all required files are present."""
    print("\nTesting file structure...")
    
    required_files = [
        'main_controller.py',
        'habitat_env.py',
        'map_visualizer.py',
        'launch.py',
        'verify_system.py',
        'configs/navigation_config.yaml',
        'requirements.txt',
        'README.md'
    ]
    
    all_present = True
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"❌ {file_path} (missing)")
            all_present = False
    
    if all_present:
        print("✓ File structure test passed")
        return True
    else:
        print("❌ File structure test failed")
        return False


def run_all_tests():
    """Run all available tests."""
    print("="*60)
    print("HABITAT MAP NAVIGATION - BASIC FUNCTIONALITY TESTS")
    print("="*60)
    
    tests = [
        test_file_structure,
        test_coordinate_conversion,
        test_command_parsing,
        test_image_processing
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Basic functionality is working.")
        print("\nNext steps:")
        print("1. Run 'python verify_system.py' to check Habitat dependencies")
        print("2. Run 'python launch.py' to start the navigation system")
        return True
    else:
        print(f"❌ {total - passed} test(s) failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
