"""
Main Controller for Habitat Map Navigation Project

This module provides the main control loop and user interface for the 
map-based navigation system. It coordinates between the Habitat environment
and map visualization components.
"""

import os
import sys
import re
import numpy as np
from typing import Optional, Dict, Any, Tuple
import cv2

# Local imports
from habitat_env import HabitatEnvironment
from map_visualizer import MapVisualizer, create_third_person_view


class NavigationController:
    """
    Main controller class that manages the navigation system.
    
    This class handles:
    - User command parsing and validation
    - Coordination between environment and visualization
    - Image generation and saving
    - Interactive command loop
    """
    
    def __init__(self, config_path: str, output_dir: str = "output_images", scene_id: str = None):
        """
        Initialize the navigation controller.
        
        Args:
            config_path: Path to Habitat configuration file
            output_dir: Directory to save output images
            scene_id: Optional scene ID to load
        """
        self.config_path = config_path
        self.output_dir = output_dir
        self.scene_id = scene_id
        
        # Initialize components
        self.habitat_env: Optional[HabitatEnvironment] = None
        self.map_visualizer: Optional[MapVisualizer] = None
        
        # State tracking
        self.step_count = 0
        self.is_initialized = False
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print("Navigation Controller initialized")
        print(f"Output directory: {output_dir}")
    
    def initialize_system(self) -> bool:
        """
        Initialize the Habitat environment and map visualizer.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            print("Initializing Habitat environment...")
            
            # Initialize Habitat environment
            self.habitat_env = HabitatEnvironment(self.config_path, self.scene_id)
            
            if not self.habitat_env.start_environment():
                print("Failed to start Habitat environment")
                return False
            
            # Get initial map data
            print("Loading map data...")
            map_data = self.habitat_env.get_top_down_map()
            
            if map_data is None:
                print("Failed to load map data")
                return False
            
            # Initialize map visualizer
            self.map_visualizer = MapVisualizer(map_data, self.habitat_env.map_info)
            
            # Generate initial images
            print("Generating initial images...")
            self._generate_images("init")
            
            self.is_initialized = True
            print("System initialization complete!")
            
            return True
            
        except Exception as e:
            print(f"Error during initialization: {e}")
            return False
    
    def _generate_images(self, prefix: str) -> bool:
        """
        Generate and save the current set of images (FPV, TPV, Map).
        
        Args:
            prefix: Prefix for the image filenames
            
        Returns:
            bool: True if images generated successfully, False otherwise
        """
        if not self.is_initialized or not self.habitat_env or not self.map_visualizer:
            print("System not properly initialized")
            return False
        
        try:
            # Get current observations and agent state
            observations = self.habitat_env.get_observations()
            agent_state = self.habitat_env.get_agent_state()
            agent_state['step_count'] = self.step_count
            
            # Extract RGB and depth images
            rgb_image = None
            depth_image = None
            
            if observations:
                if 'rgb' in observations:
                    rgb_image = observations['rgb']
                    # Convert from RGB to BGR for OpenCV compatibility
                    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                
                if 'depth' in observations:
                    depth_image = observations['depth']
                    if len(depth_image.shape) == 3:
                        depth_image = depth_image[:, :, 0]  # Take first channel if multi-channel
            
            # Generate image filenames
            fpv_filename = os.path.join(self.output_dir, f"{prefix}_fpv.png")
            tpv_filename = os.path.join(self.output_dir, f"{prefix}_tpv.png")
            map_filename = os.path.join(self.output_dir, f"{prefix}_map.png")
            composite_filename = os.path.join(self.output_dir, f"{prefix}_composite.png")
            
            # Save first-person view
            if rgb_image is not None:
                cv2.imwrite(fpv_filename, rgb_image)
                print(f"Saved first-person view: {fpv_filename}")
            else:
                print("No RGB image available for first-person view")
            
            # Create and save third-person view (simulated)
            if rgb_image is not None:
                tpv_image = create_third_person_view(
                    agent_state['position'], 
                    cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB),  # Convert back to RGB
                    self.habitat_env.map_info['world_bounds']
                )
                cv2.imwrite(tpv_filename, cv2.cvtColor(tpv_image, cv2.COLOR_RGB2BGR))
                print(f"Saved third-person view: {tpv_filename}")
            
            # Generate and save map view
            map_title = f"Navigation Map - Step {self.step_count}"
            if self.map_visualizer.generate_map_image(agent_state, map_filename, map_title):
                print(f"Saved map view: {map_filename}")
            
            # Generate composite view
            composite_title = f"Navigation View - Step {self.step_count}"
            rgb_for_composite = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB) if rgb_image is not None else None
            
            if self.map_visualizer.generate_comparative_view(
                agent_state, rgb_for_composite, depth_image, 
                composite_filename, composite_title
            ):
                print(f"Saved composite view: {composite_filename}")
            
            return True
            
        except Exception as e:
            print(f"Error generating images: {e}")
            return False
    
    def _parse_move_command(self, command: str) -> Optional[Tuple[float, float]]:
        """
        Parse move command and extract coordinates.
        
        Args:
            command: User input command
            
        Returns:
            Optional[Tuple[float, float]]: Parsed coordinates or None if invalid
        """
        # Match pattern: move <x> <y>
        pattern = r"move\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)"
        match = re.match(pattern, command.strip().lower())
        
        if match:
            try:
                x = float(match.group(1))
                y = float(match.group(2))
                return (x, y)
            except ValueError:
                return None
        
        return None
    
    def _parse_turn_command(self, command: str) -> Optional[Tuple[str, float]]:
        """
        Parse turn command and extract direction and degrees.
        
        Args:
            command: User input command
            
        Returns:
            Optional[Tuple[str, float]]: (direction, degrees) or None if invalid
        """
        # Match pattern: turn <left|right> <degrees>
        pattern = r"turn\s+(left|right)\s+(-?\d+\.?\d*)"
        match = re.match(pattern, command.strip().lower())
        
        if match:
            try:
                direction = match.group(1)
                degrees = float(match.group(2))
                return (direction, degrees)
            except ValueError:
                return None
        
        return None
    
    def _parse_look_command(self, command: str) -> Optional[Tuple[str, float]]:
        """
        Parse look command and extract direction and degrees.
        
        Args:
            command: User input command
            
        Returns:
            Optional[Tuple[str, float]]: (direction, degrees) or None if invalid
        """
        # Match pattern: look <up|down> <degrees>
        pattern = r"look\s+(up|down)\s+(-?\d+\.?\d*)"
        match = re.match(pattern, command.strip().lower())
        
        if match:
            try:
                direction = match.group(1)
                degrees = float(match.group(2))
                return (direction, degrees)
            except ValueError:
                return None
        
        return None
    
    def _execute_move_command(self, map_x: float, map_y: float) -> bool:
        """
        Execute move command by converting map coordinates to world coordinates.
        
        Args:
            map_x: X coordinate on the map
            map_y: Y coordinate on the map
            
        Returns:
            bool: True if move successful, False otherwise
        """
        if not self.habitat_env:
            return False
        
        try:
            # Convert map coordinates to world coordinates
            world_pos = self.habitat_env.map_to_world_coordinates(map_x, map_y)
            
            print(f"Moving to map coordinates ({map_x}, {map_y}) -> world coordinates {world_pos}")
            
            # Execute movement
            success = self.habitat_env.move_to_position(world_pos)
            
            if success:
                self.step_count += 1
                self._generate_images(f"step_{self.step_count:02d}")
                
                agent_state = self.habitat_env.get_agent_state()
                print(f"Movement successful! New position: {agent_state['position']}")
            else:
                print("Movement failed - target position may not be navigable")
            
            return success
            
        except Exception as e:
            print(f"Error executing move command: {e}")
            return False
    
    def _execute_turn_command(self, direction: str, degrees: float) -> bool:
        """
        Execute turn command.
        
        Args:
            direction: 'left' or 'right'
            degrees: Degrees to turn
            
        Returns:
            bool: True if turn successful, False otherwise
        """
        if not self.habitat_env:
            return False
        
        try:
            success = self.habitat_env.turn_agent(direction, degrees)
            
            if success:
                self.step_count += 1
                self._generate_images(f"step_{self.step_count:02d}")
                
                agent_state = self.habitat_env.get_agent_state()
                print(f"Turn successful! New yaw: {agent_state['yaw_degrees']:.1f}°")
            else:
                print("Turn failed")
            
            return success
            
        except Exception as e:
            print(f"Error executing turn command: {e}")
            return False
    
    def _execute_look_command(self, direction: str, degrees: float) -> bool:
        """
        Execute look command (camera pitch adjustment).
        
        Args:
            direction: 'up' or 'down'
            degrees: Degrees to adjust pitch
            
        Returns:
            bool: True if look successful, False otherwise
        """
        if not self.habitat_env:
            return False
        
        try:
            success = self.habitat_env.look_vertical(direction, degrees)
            
            if success:
                self.step_count += 1
                self._generate_images(f"step_{self.step_count:02d}")
                
                agent_state = self.habitat_env.get_agent_state()
                print(f"Look successful! New pitch: {agent_state['camera_pitch_degrees']:.1f}°")
            else:
                print("Look adjustment failed")
            
            return success
            
        except Exception as e:
            print(f"Error executing look command: {e}")
            return False
    
    def print_help(self):
        """Print available commands and usage instructions."""
        help_text = """
=== MAP NAVIGATION COMMANDS ===

Available Commands:
  move <x> <y>           - Move agent to map coordinates (x, y)
                          Example: move 5.2 -3.8

  turn <direction> <degrees>  - Turn agent left or right
                              Example: turn right 45
                              Example: turn left 90

  look <direction> <degrees>  - Adjust camera pitch up or down
                              Example: look up 30
                              Example: look down 20

  help                   - Show this help message
  quit / exit           - Exit the program

Instructions:
1. Use 'move' command with map coordinates to navigate
2. Map coordinates are displayed on the grid overlay
3. Red marker shows current agent position and orientation
4. Images are automatically saved after each command
5. Check the output_images/ folder for generated images

Current Status:
- Step Count: %d
- Output Directory: %s
- System Status: %s
        """ % (
            self.step_count,
            self.output_dir,
            "Ready" if self.is_initialized else "Not Initialized"
        )
        
        print(help_text)
    
    def run_interactive_loop(self):
        """
        Run the main interactive command loop.
        
        This method handles user input and executes commands until the user exits.
        """
        if not self.is_initialized:
            print("System not initialized. Run initialize_system() first.")
            return
        
        print("\n" + "="*60)
        print("HABITAT MAP NAVIGATION SYSTEM")
        print("="*60)
        print("System ready! Type 'help' for available commands, 'quit' to exit.")
        
        # Display initial agent state
        if self.habitat_env:
            agent_state = self.habitat_env.get_agent_state()
            print(f"\nInitial Agent State:")
            print(f"  Position: ({agent_state['position'][0]:.2f}, {agent_state['position'][2]:.2f})")
            print(f"  Yaw: {agent_state['yaw_degrees']:.1f}°")
            print(f"  Images saved with prefix 'init'")
        
        while True:
            try:
                # Get user input
                command = input("\n> ").strip()
                
                if not command:
                    continue
                
                # Parse and execute commands
                if command.lower() in ['quit', 'exit']:
                    print("Exiting navigation system...")
                    break
                
                elif command.lower() == 'help':
                    self.print_help()
                
                elif command.lower().startswith('move'):
                    coords = self._parse_move_command(command)
                    if coords:
                        map_x, map_y = coords
                        self._execute_move_command(map_x, map_y)
                    else:
                        print("Invalid move command. Usage: move <x> <y>")
                        print("Example: move 5.2 -3.8")
                
                elif command.lower().startswith('turn'):
                    turn_params = self._parse_turn_command(command)
                    if turn_params:
                        direction, degrees = turn_params
                        self._execute_turn_command(direction, degrees)
                    else:
                        print("Invalid turn command. Usage: turn <left|right> <degrees>")
                        print("Example: turn right 45")
                
                elif command.lower().startswith('look'):
                    look_params = self._parse_look_command(command)
                    if look_params:
                        direction, degrees = look_params
                        self._execute_look_command(direction, degrees)
                    else:
                        print("Invalid look command. Usage: look <up|down> <degrees>")
                        print("Example: look up 30")
                
                else:
                    print(f"Unknown command: {command}")
                    print("Type 'help' for available commands.")
                
            except KeyboardInterrupt:
                print("\nReceived Ctrl+C. Exiting...")
                break
            except Exception as e:
                print(f"Error processing command: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        if self.habitat_env:
            self.habitat_env.cleanup()
        print("Navigation controller cleaned up")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


def main():
    """
    Main function to run the navigation system.
    
    This function can be used for testing or as a standalone entry point.
    """
    # Configuration
    config_path = os.path.join(os.path.dirname(__file__), "configs", "navigation_config.yaml")
    output_dir = os.path.join(os.path.dirname(__file__), "output_images")
    
    # You can specify a different scene here
    scene_id = None  # Use default from config, or specify like "17DRP5sb8fy"
    
    print("Starting Habitat Map Navigation System...")
    
    # Initialize controller
    controller = NavigationController(config_path, output_dir, scene_id)
    
    try:
        # Initialize system
        if controller.initialize_system():
            # Run interactive loop
            controller.run_interactive_loop()
        else:
            print("Failed to initialize system")
    
    finally:
        # Cleanup
        controller.cleanup()


if __name__ == "__main__":
    main()
