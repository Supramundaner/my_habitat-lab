#!/usr/bin/env python3
"""
Simple interactive test for coordinate navigation
"""

import sys
import os
sys.path.append('/home/yaoaa/habitat-lab')

from examples.coordinate_navigation import CoordinateNavigationAgent


def simple_interactive_test():
    """Simple interactive test"""
    print("=== Simple Interactive Navigation Test ===")
    
    # Initialize agent
    agent = CoordinateNavigationAgent()
    
    if agent.sim is None:
        print("❌ Failed to initialize simulator")
        return
    
    print(f"✅ Loaded scene: {agent.scene_id}")
    print(f"Current position: {agent.get_current_position()}")
    
    # Show initial view
    print("Displaying initial view...")
    agent.display_observations()
    
    while True:
        print("\nCommands:")
        print("1 - View current position")
        print("2 - Display observations")
        print("3 - Move to a nearby position")
        print("4 - Rotate view 45 degrees")
        print("5 - List available scenes")
        print("q - Quit")
        
        choice = input("\nEnter choice: ").strip().lower()
        
        if choice == 'q':
            break
        elif choice == '1':
            pos = agent.get_current_position()
            print(f"Current position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
        elif choice == '2':
            agent.display_observations()
        elif choice == '3':
            # Try to move to a nearby position
            current_pos = agent.get_current_position()
            new_pos = [current_pos[0] + 0.5, current_pos[1], current_pos[2]]
            print(f"Trying to move to: {new_pos}")
            success = agent.move_to_coordinate(new_pos)
            if success:
                print("✅ Movement successful!")
                agent.display_observations()
            else:
                print("❌ Movement failed - position not navigable")
        elif choice == '4':
            print("Rotating view by 45 degrees...")
            agent.adjust_view_angle(45, 0)
            agent.display_observations()
        elif choice == '5':
            agent.list_available_scenes()
        else:
            print("Invalid choice. Please select 1-5 or q.")
    
    agent.close()
    print("✅ Test completed!")


if __name__ == "__main__":
    simple_interactive_test()
