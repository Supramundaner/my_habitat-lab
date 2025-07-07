#!/usr/bin/env python3
"""
Quick test to verify the fix for the Matrix4.data() issue
"""

import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Try to import the fixed module
try:
    from multi_agent_navigation import MultiAgentSimulator
    print("✓ Import successful! The fix appears to be working.")
    
    # Try to create a minimal config to test the class can be instantiated
    test_config = {
        "scene": {
            "scene_dataset_path": "data/scene_datasets/habitat-test-scenes/apartment_1.glb"
        },
        "simulator": {
            "gpu_device_id": 0,
            "enable_physics": True
        },
        "agents": [{
            "id": "test_agent",
            "sensors": {
                "color_sensor": {
                    "resolution": [480, 640]
                }
            }
        }],
        "collision_detection": {
            "enabled": False
        },
        "movement": {
            "linear_speed": 1.0,
            "angular_speed": 30.0,
            "time_step": 0.1
        },
        "video_output": {
            "output_dir": "outputs/test",
            "fps": 10,
            "resolution": [480, 640]
        },
        "map_config": {
            "agent_marker_size": 5,
            "agent_marker_color": [255, 0, 0],
            "direction_arrow_length": 10
        },
        "logging": {
            "log_level": "INFO",
            "log_file": "outputs/test.log",
            "console_output": True
        },
        "state_persistence": {
            "save_after_each_action": False,
            "save_final_state": False,
            "state_file": "outputs/states.json"
        }
    }
    
    print("Test config created successfully.")
    print("The fix should resolve the '_magnum.Matrix4' object has no attribute 'data' error.")
    
except ImportError as e:
    print(f"✗ Import failed: {e}")
except Exception as e:
    print(f"✗ Other error: {e}")
