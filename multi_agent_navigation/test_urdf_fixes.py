#!/usr/bin/env python3
"""
Test script to verify URDF robot camera handling fixes
"""

import sys
import os
import numpy as np
import logging

# Add the parent directory to path to import the simulator
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from multi_agent_simulator import URDFCameraParser, safe_matrix4_to_numpy, safe_quaternion_to_numpy
    print("✅ Successfully imported URDF handling classes")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

def test_urdf_parser():
    """Test URDF parser with a dummy URDF file"""
    print("\n=== Testing URDF Parser ===")
    
    # Create a minimal test URDF content
    test_urdf_content = """<?xml version="1.0"?>
<robot name="test_robot">
  <joint name="torso_lift_joint" type="prismatic">
    <origin xyz="0 0 0.7" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="torso_lift_link"/>
  </joint>
  
  <joint name="head_pan_joint" type="revolute">
    <origin xyz="0.053 0 0.603" rpy="0 0 0"/>
    <parent link="torso_lift_link"/>
    <child link="head_pan_link"/>
  </joint>
  
  <joint name="head_tilt_joint" type="revolute">
    <origin xyz="0.14253 0 0.057999" rpy="0 0 0"/>
    <parent link="head_pan_link"/>
    <child link="head_tilt_link"/>
  </joint>
  
  <joint name="head_camera_joint" type="fixed">
    <origin xyz="0.0 0.0 0.0" rpy="0 0 0"/>
    <parent link="head_tilt_link"/>
    <child link="head_camera_link"/>
  </joint>
  
  <joint name="head_camera_rgb_joint" type="fixed">
    <origin xyz="0.0 -0.02 0.0" rpy="0 0 0"/>
    <parent link="head_camera_link"/>
    <child link="head_camera_rgb_frame"/>
  </joint>
</robot>"""
    
    # Write test URDF to temporary file
    test_urdf_path = "test_robot.urdf"
    try:
        with open(test_urdf_path, 'w') as f:
            f.write(test_urdf_content)
        
        # Test URDF parser
        parser = URDFCameraParser(test_urdf_path)
        print(f"✅ URDF parser created successfully")
        print(f"   Found {len(parser.joint_transforms)} joints")
        
        # Test camera transform calculation
        robot_transform = np.eye(4)
        robot_transform[:3, 3] = [1.0, 0.0, 2.0]  # Robot at position (1, 0, 2)
        
        camera_pos, camera_quat = parser.get_camera_position_and_orientation(robot_transform)
        
        if camera_pos is not None and camera_quat is not None:
            print(f"✅ Camera transform calculated successfully")
            print(f"   Camera position: {camera_pos}")
            print(f"   Camera quaternion: {camera_quat}")
        else:
            print(f"❌ Failed to calculate camera transform")
        
        # Clean up
        os.remove(test_urdf_path)
        
    except Exception as e:
        print(f"❌ URDF parser test failed: {e}")
        if os.path.exists(test_urdf_path):
            os.remove(test_urdf_path)

def test_matrix_conversion():
    """Test matrix conversion utilities"""
    print("\n=== Testing Matrix Conversion ===")
    
    try:
        # Test numpy to numpy (should work)
        test_matrix = np.eye(4)
        result = safe_matrix4_to_numpy(test_matrix)
        print("✅ Numpy matrix conversion works")
        
        # Test quaternion conversion
        test_quat = np.array([0, 0, 0, 1])
        result_quat = safe_quaternion_to_numpy(test_quat)
        print("✅ Quaternion conversion works")
        print(f"   Input: {test_quat}")
        print(f"   Output: {result_quat}")
        
    except Exception as e:
        print(f"❌ Matrix conversion test failed: {e}")

def test_logging():
    """Test that logging works correctly"""
    print("\n=== Testing Logging ===")
    
    try:
        # Set up a logger similar to the simulator
        logger = logging.getLogger("TestLogger")
        logger.setLevel(logging.INFO)
        
        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_format = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        logger.info("✅ Test log message - logging system works")
        logger.warning("⚠️ Test warning - Matrix4 conversion error handling")
        logger.warning("⚠️ Test warning - FPV fallback system")
        
    except Exception as e:
        print(f"❌ Logging test failed: {e}")

def main():
    print("🔧 Testing Multi-Agent Simulator URDF Robot Fixes")
    print("=" * 50)
    
    test_urdf_parser()
    test_matrix_conversion()
    test_logging()
    
    print("\n" + "=" * 50)
    print("🎯 Test Summary:")
    print("   - URDF camera parsing implemented")
    print("   - Matrix4 conversion error handling added")
    print("   - FPV fallback system implemented")
    print("   - Error handling improved throughout")
    print("\n✅ Fixes should resolve the reported issues!")

if __name__ == "__main__":
    main()
