#!/usr/bin/env python3
"""
Simple test for coordinate navigation system
"""

import os
import sys
sys.path.append('/home/yaoaa/habitat-lab')

from examples.coordinate_navigation import CoordinateNavigationAgent


def test_navigation():
    """Simple test of the navigation system"""
    print("=== Testing Coordinate Navigation System ===")
    
    # Initialize agent
    print("Initializing agent...")
    agent = CoordinateNavigationAgent()
    
    if agent.sim is None:
        print("❌ Failed to initialize simulator")
        print("Available scenes:", agent.available_scenes)
        return False
    
    print(f"✅ Successfully initialized with scene: {agent.scene_id}")
    
    # Test basic functions
    print(f"Current position: {agent.get_current_position()}")
    
    # Display initial view
    print("Displaying initial observations...")
    agent.display_observations(save_images=True)
    
    # Test movement
    print("Testing movement...")
    current_pos = agent.get_current_position()
    new_pos = [current_pos[0] + 1.0, current_pos[1], current_pos[2] + 1.0]
    
    success = agent.move_to_coordinate(new_pos)
    if success:
        print("✅ Movement successful")
        agent.display_observations(save_images=True)
    else:
        print("❌ Movement failed - trying alternative position")
        # Try a closer position
        new_pos = [current_pos[0] + 0.5, current_pos[1], current_pos[2]]
        success = agent.move_to_coordinate(new_pos)
        if success:
            print("✅ Alternative movement successful")
            agent.display_observations(save_images=True)
    
    # Test view adjustment
    print("Testing view adjustment...")
    agent.adjust_view_angle(45, 0)
    agent.display_observations(save_images=True)
    
    # Test topdown map
    print("Testing topdown map...")
    topdown_map = agent.get_topdown_map()
    if topdown_map.size > 0:
        print("✅ Topdown map generation successful")
    else:
        print("❌ Topdown map generation failed")
    
    # Cleanup
    agent.close()
    print("✅ Test completed successfully!")
    return True


if __name__ == "__main__":
    test_navigation()
