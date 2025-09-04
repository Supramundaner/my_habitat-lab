#!/usr/bin/env python3
"""
Test script for multi-point selection functionality.
This script performs basic validation of the multi-point workflow without running the full pipeline.
"""

import os
import json
import sys
import numpy as np
import cv2
from typing import Dict, Any, List

def test_config_validation():
    """Test configuration validation."""
    print("🧪 Testing configuration validation...")
    
    try:
        from multi_point_utils import validate_k_points_config
        
        # Test valid config
        config = {"k_points": 3}
        k_points = validate_k_points_config(config)
        assert k_points == 3, f"Expected 3, got {k_points}"
        
        # Test default config
        config = {}
        k_points = validate_k_points_config(config)
        assert k_points == 1, f"Expected 1, got {k_points}"
        
        # Test invalid config
        try:
            config = {"k_points": 0}
            validate_k_points_config(config)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        
        print("✅ Configuration validation tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation tests failed: {e}")
        return False

def test_marker_colors():
    """Test marker color generation."""
    print("🧪 Testing marker color generation...")
    
    try:
        from multi_point_utils import get_marker_color
        
        config = {
            "multi_point_config": {
                "marker_colors": [[0, 0, 255], [255, 0, 0], [0, 255, 0]]
            }
        }
        
        # Test color cycling
        color0 = get_marker_color(0, config)
        color1 = get_marker_color(1, config)
        color2 = get_marker_color(2, config)
        color3 = get_marker_color(3, config)  # Should cycle back to first color
        
        assert color0 == (0, 0, 255), f"Expected (0, 0, 255), got {color0}"
        assert color1 == (255, 0, 0), f"Expected (255, 0, 0), got {color1}"
        assert color2 == (0, 255, 0), f"Expected (0, 255, 0), got {color2}"
        assert color3 == color0, f"Expected {color0}, got {color3}"
        
        print("✅ Marker color generation tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Marker color generation tests failed: {e}")
        return False

def test_node_marking():
    """Test node marking functionality."""
    print("🧪 Testing node marking functionality...")
    
    try:
        from multi_point_utils import mark_selected_node
        
        # Create a test image
        test_image = np.ones((400, 400, 3), dtype=np.uint8) * 255  # White background
        
        # Create mock node result
        node_result = {
            "selected_node": {
                "pixel_coordinates": [200, 200],
                "node_id": 15
            }
        }
        
        config = {
            "multi_point_config": {
                "marker_colors": [[0, 0, 255]],
                "marker_size": 30,
                "marker_thickness": 4,
                "show_iteration_numbers": True
            }
        }
        
        # Test marking
        marked_image = mark_selected_node(test_image, node_result, 0, config)
        
        # Verify the image was modified (should not be all white anymore)
        assert not np.array_equal(test_image, marked_image), "Image should have been modified"
        
        # Verify the marked pixel is different from white
        marked_pixel = marked_image[200, 200]
        assert not np.array_equal(marked_pixel, [255, 255, 255]), "Marked pixel should not be white"
        
        print("✅ Node marking functionality tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Node marking functionality tests failed: {e}")
        return False

def test_prompt_generation():
    """Test prompt generation for iterations."""
    print("🧪 Testing prompt generation...")
    
    try:
        from multi_point_utils import generate_iteration_prompt_addition
        
        # Test first iteration (should return empty string)
        prompt_addition = generate_iteration_prompt_addition(0, [], "tv", "room")
        assert prompt_addition == "", "First iteration should return empty string"
        
        # Test subsequent iteration
        selected_results = [
            {
                "room_result": {
                    "llm_response": {"selected_room": 1}
                },
                "node_result": {
                    "llm_response": {"selected_node_id": 15}
                }
            }
        ]
        
        prompt_addition = generate_iteration_prompt_addition(1, selected_results, "tv", "room")
        assert len(prompt_addition) > 0, "Should generate non-empty prompt addition"
        assert "IMPORTANT CONTEXT" in prompt_addition, "Should contain important context"
        assert "tv" in prompt_addition, "Should contain goal object"
        
        print("✅ Prompt generation tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Prompt generation tests failed: {e}")
        return False

def test_config_file():
    """Test the updated configuration file."""
    print("🧪 Testing configuration file...")
    
    try:
        config_path = "input_config.json"
        if not os.path.exists(config_path):
            print(f"⚠️  Config file not found: {config_path}")
            return False
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Check for required fields
        assert "k_points" in config, "k_points field missing"
        assert "multi_point_config" in config, "multi_point_config field missing"
        
        k_points = config["k_points"]
        assert isinstance(k_points, int) and k_points > 0, f"Invalid k_points: {k_points}"
        
        multi_config = config["multi_point_config"]
        assert "marker_colors" in multi_config, "marker_colors missing"
        assert "marker_size" in multi_config, "marker_size missing"
        assert "marker_thickness" in multi_config, "marker_thickness missing"
        
        print("✅ Configuration file tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration file tests failed: {e}")
        return False

def test_workflow_imports():
    """Test that all workflow modules can be imported."""
    print("🧪 Testing workflow module imports...")
    
    try:
        from main_workflow import WorkflowOrchestrator
        from multi_point_utils import validate_k_points_config, mark_selected_node
        from step_3_llm_room_selection import select_room_with_llm
        from step_5_node_selection import select_navigation_node  
        from step_6_path_planning import path_planning_step
        
        print("✅ All workflow modules imported successfully")
        return True
        
    except Exception as e:
        print(f"❌ Workflow module import tests failed: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("🚀 Starting Multi-Point Selection Tests")
    print("="*60)
    
    tests = [
        test_config_validation,
        test_marker_colors,
        test_node_marking,
        test_prompt_generation,
        test_config_file,
        test_workflow_imports
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            print()  # Add spacing between tests
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            print()
    
    print("="*60)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Multi-point selection implementation looks good.")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
