#!/usr/bin/env python3
"""
Interactive Coordinate Navigation Demo

This demo shows how to use the coordinate navigation system interactively.
Run this script to start the interactive navigation interface.
"""

import sys
import os
sys.path.append('/home/yaoaa/habitat-lab')

from examples.coordinate_navigation import CoordinateNavigationAgent
import numpy as np


def demo_coordinate_navigation():
    """Demo of coordinate navigation with predefined movements"""
    print("=== Coordinate Navigation Demo ===")
    
    # Initialize agent
    agent = CoordinateNavigationAgent()
    
    if agent.sim is None:
        print("❌ Failed to initialize simulator")
        return
    
    print(f"✅ Loaded scene: {agent.scene_id}")
    
    # Show available scenes
    print(f"\nAvailable scenes ({len(agent.available_scenes)}):")
    for i, scene in enumerate(agent.available_scenes[:5]):  # Show first 5
        print(f"  {i+1}. {scene}")
    if len(agent.available_scenes) > 5:
        print(f"  ... and {len(agent.available_scenes) - 5} more")
    
    # Show initial state
    print(f"\nInitial position: {agent.get_current_position()}")
    print("Displaying initial view...")
    agent.display_observations()
    
    # Demo coordinated movements
    print("\n=== Demo Movements ===")
    
    # Get some navigable points around the current position
    current_pos = np.array(agent.get_current_position())
    
    # Try different directions
    directions = [
        ([1.0, 0.0, 0.0], "Moving East"),
        ([0.0, 0.0, 1.0], "Moving North"), 
        ([-1.0, 0.0, 0.0], "Moving West"),
        ([0.0, 0.0, -1.0], "Moving South"),
    ]
    
    for direction, description in directions:
        print(f"\n{description}...")
        target_pos = current_pos + 0.5 * np.array(direction)
        target_pos[1] = current_pos[1]  # Keep same height
        
        success = agent.move_to_coordinate(target_pos.tolist())
        if success:
            print("✅ Movement successful!")
            agent.display_observations()
            current_pos = np.array(agent.get_current_position())
            break
        else:
            print("❌ Position not navigable, trying next direction...")
    
    # Demo view rotation
    print("\n=== Demo View Rotation ===")
    for angle in [45, 90, -45, -90]:
        print(f"Rotating view by {angle} degrees...")
        agent.adjust_view_angle(angle, 0)
        agent.display_observations()
    
    # Clean up
    agent.close()
    print("\n✅ Demo completed!")


def run_interactive_mode():
    """Run the full interactive navigation system"""
    from examples.coordinate_navigation import interactive_navigation
    interactive_navigation()


def main():
    """Main menu"""
    while True:
        print("\n=== Habitat Coordinate Navigation System ===")
        print("1. Run interactive navigation")
        print("2. Run demo with predefined movements")
        print("3. List available scenes")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            print("\nStarting interactive navigation...")
            run_interactive_mode()
        
        elif choice == '2':
            print("\nStarting demo...")
            demo_coordinate_navigation()
        
        elif choice == '3':
            print("\nListing available scenes...")
            agent = CoordinateNavigationAgent()
            if agent.available_scenes:
                print(f"Found {len(agent.available_scenes)} scenes:")
                for i, scene in enumerate(agent.available_scenes):
                    print(f"  {i+1}. {scene}")
            else:
                print("No scenes found. Please run download script first.")
            agent.close()
        
        elif choice == '4':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please select 1-4.")


if __name__ == "__main__":
    main()
