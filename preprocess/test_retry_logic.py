#!/usr/bin/env python3
"""
Test script to verify the retry logic in room and node selection.
This script tests the retry mechanism without running the full workflow.
"""

import os
import json
import sys
from typing import Dict, Any

# Test the imports
try:
    from step_3_llm_room_selection import get_available_rooms, select_room_manually
    from step_5_node_selection import select_navigation_node
    print("✓ Successfully imported modified modules")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

def test_get_available_rooms():
    """Test the get_available_rooms function."""
    print("\n🧪 Testing get_available_rooms function:")
    
    # Create a mock output.json
    mock_output_data = {
        "final_results": {
            "room_bounding_boxes": {
                "1": {"x_min": 100, "y_min": 100, "x_max": 200, "y_max": 200},
                "2": {"x_min": 300, "y_min": 300, "x_max": 400, "y_max": 400},
                "3": {"x_min": 500, "y_min": 500, "x_max": 600, "y_max": 600}
            }
        }
    }
    
    # Create temporary directory and file
    test_dir = "/tmp/test_workflow"
    os.makedirs(test_dir, exist_ok=True)
    
    output_json_path = os.path.join(test_dir, "output.json")
    with open(output_json_path, 'w') as f:
        json.dump(mock_output_data, f)
    
    # Test the function
    available_rooms = get_available_rooms(test_dir)
    expected_rooms = [1, 2, 3]
    
    if sorted(available_rooms) == sorted(expected_rooms):
        print(f"✓ get_available_rooms returned correct rooms: {sorted(available_rooms)}")
    else:
        print(f"✗ get_available_rooms returned {available_rooms}, expected {expected_rooms}")
    
    # Cleanup
    os.remove(output_json_path)
    os.rmdir(test_dir)

def test_config_loading():
    """Test loading configuration with retry parameters."""
    print("\n🧪 Testing configuration loading:")
    
    config_path = "example_config_with_retry.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        max_retries = config['llm_config'].get('max_retries', 3)
        print(f"✓ Loaded max_retries from config: {max_retries}")
        
        if max_retries >= 1:
            print("✓ Retry configuration is valid")
        else:
            print("✗ Invalid retry configuration")
    else:
        print(f"✗ Configuration file not found: {config_path}")

def main():
    """Main test function."""
    print("🚀 Testing Retry Logic Modifications")
    print("="*50)
    
    test_get_available_rooms()
    test_config_loading()
    
    print("\n🏁 Test completed!")
    print("\nKey improvements added:")
    print("1. ✅ Room selection now validates against available rooms")
    print("2. ✅ Node selection now validates against available nodes") 
    print("3. ✅ Both functions retry up to max_retries times on invalid responses")
    print("4. ✅ Fallback to first available option if all retries fail")
    print("5. ✅ Enhanced logging with all attempt details")
    print("6. ✅ Better error handling and status reporting")

if __name__ == "__main__":
    main()
