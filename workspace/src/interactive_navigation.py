#!/usr/bin/env python3
"""
Interactive Coordinate Navigation Interface

This module provides an interactive command-line interface for the 
Habitat Lab coordinate navigation system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from coordinate_navigation import CoordinateNavigationSystem
import numpy as np
import re


class InteractiveNavigationInterface:
    """Interactive interface for coordinate navigation"""
    
    def __init__(self):
        self.workspace_path = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(os.path.dirname(self.workspace_path), "data")
        self.nav_system = None
        self.running = True
        
    def initialize_system(self, scene_id: str = None):
        """Initialize the navigation system"""
        print("Initializing navigation system...")
        
        self.nav_system = CoordinateNavigationSystem(
            scene_id=scene_id,
            workspace_path=self.workspace_path,
            data_path=self.data_path
        )
        
        if self.nav_system.sim is None:
            print("❌ Failed to initialize navigation system")
            return False
        
        print(f"✅ Successfully initialized with scene: {self.nav_system.scene_id}")
        return True
    
    def show_help(self):
        """Display help information"""
        print("\n" + "="*60)
        print("HABITAT LAB COORDINATE NAVIGATION - COMMAND HELP")
        print("="*60)
        print("\n📍 COORDINATE COMMANDS:")
        print("  move <x> <y>        - Move to map coordinates (x, y)")
        print("  goto <x> <y>        - Same as move")
        print("  position            - Show current position")
        print("  bounds              - Show map coordinate bounds")
        print("\n👁️  VIEW COMMANDS:")
        print("  rotate <yaw> [pitch] - Rotate view (degrees)")
        print("  turn <angle>         - Turn by angle (degrees)")
        print("  view                 - Display current views")
        print("  show                 - Same as view")
        print("\n🗺️  SCENE COMMANDS:")
        print("  scenes              - List available scenes")
        print("  switch <scene_id>   - Switch to different scene")
        print("  reload              - Reload current scene")
        print("\n💾 UTILITY COMMANDS:")
        print("  save [prefix]       - Save current images")
        print("  info                - Show system information")
        print("  help                - Show this help")
        print("  quit, exit          - Exit program")
        print("\n📋 EXAMPLES:")
        print("  move 512 256        - Move to center-left of map")
        print("  rotate 90           - Turn right 90 degrees")
        print("  switch habitat-test-scenes/van-gogh-room")
        print("  save exploration_   - Save with prefix")
        print("="*60)
    
    def show_system_info(self):
        """Display system information"""
        if self.nav_system is None:
            print("❌ Navigation system not initialized")
            return
        
        print("\n" + "="*50)
        print("SYSTEM INFORMATION")
        print("="*50)
        print(f"Current Scene: {self.nav_system.scene_id}")
        print(f"Map Resolution: {self.nav_system.map_resolution}x{self.nav_system.map_resolution}")
        print(f"Map Scale: {self.nav_system.map_scale} meters/pixel")
        print(f"Camera Height: {self.nav_system.camera_height} meters")
        print(f"Field of View: {self.nav_system.fov} degrees")
        
        bounds = self.nav_system.map_bounds
        print(f"\nMap Bounds (World Coordinates):")
        print(f"  X: [{bounds['min_x']:.2f}, {bounds['max_x']:.2f}] meters")
        print(f"  Z: [{bounds['min_z']:.2f}, {bounds['max_z']:.2f}] meters")
        
        current_pos = self.nav_system.get_current_position()
        print(f"\nCurrent Position:")
        print(f"  World: [{current_pos['world_position'][0]:.2f}, "
              f"{current_pos['world_position'][1]:.2f}, "
              f"{current_pos['world_position'][2]:.2f}] meters")
        print(f"  Map: [{current_pos['map_position'][0]:.1f}, "
              f"{current_pos['map_position'][1]:.1f}] pixels")
        
        print(f"\nImages Directory: {self.nav_system.images_path}")
        print(f"Available Scenes: {len(self.nav_system.available_scenes)}")
        print("="*50)
    
    def parse_coordinate_command(self, command: str) -> tuple:
        """Parse coordinate commands like 'move 100 200' or 'goto 150 300'"""
        parts = command.strip().split()
        if len(parts) < 3:
            return None, None, "Invalid format. Use: move <x> <y>"
        
        try:
            x = float(parts[1])
            y = float(parts[2])
            return x, y, None
        except ValueError:
            return None, None, "Invalid coordinates. Use numeric values."
    
    def parse_rotation_command(self, command: str) -> tuple:
        """Parse rotation commands like 'rotate 45' or 'rotate 45 10'"""
        parts = command.strip().split()
        if len(parts) < 2:
            return None, None, "Invalid format. Use: rotate <yaw> [pitch]"
        
        try:
            yaw = float(parts[1])
            pitch = float(parts[2]) if len(parts) > 2 else 0.0
            return yaw, pitch, None
        except ValueError:
            return None, None, "Invalid angles. Use numeric values."
    
    def execute_move_command(self, x: float, y: float):
        """Execute movement command"""
        if self.nav_system is None:
            print("❌ Navigation system not initialized")
            return
        
        print(f"🚶 Moving to map coordinates ({x:.1f}, {y:.1f})...")
        
        # Validate coordinates
        if not (0 <= x <= self.nav_system.map_resolution and 
                0 <= y <= self.nav_system.map_resolution):
            print(f"⚠️  Warning: Coordinates outside map bounds "
                  f"[0, {self.nav_system.map_resolution}]")
        
        success = self.nav_system.move_to_map_coordinate(x, y)
        
        if success:
            print("✅ Movement successful!")
            # Automatically display new view
            self.nav_system.display_current_state()
        else:
            print("❌ Movement failed - position not navigable")
            print("💡 Try coordinates closer to navigable areas")
    
    def execute_rotation_command(self, yaw: float, pitch: float = 0.0):
        """Execute rotation command"""
        if self.nav_system is None:
            print("❌ Navigation system not initialized")
            return
        
        print(f"🔄 Rotating view - Yaw: {yaw}°, Pitch: {pitch}°")
        self.nav_system.adjust_view_angle(yaw, pitch)
        
        # Automatically display new view
        self.nav_system.display_current_state()
    
    def process_command(self, command: str):
        """Process user command"""
        command = command.strip().lower()
        
        if not command:
            return
        
        # Navigation commands
        if command.startswith(('move ', 'goto ')):
            x, y, error = self.parse_coordinate_command(command)
            if error:
                print(f"❌ {error}")
            else:
                self.execute_move_command(x, y)
        
        # Rotation commands
        elif command.startswith('rotate '):
            yaw, pitch, error = self.parse_rotation_command(command)
            if error:
                print(f"❌ {error}")
            else:
                self.execute_rotation_command(yaw, pitch)
        
        elif command.startswith('turn '):
            parts = command.split()
            if len(parts) < 2:
                print("❌ Invalid format. Use: turn <angle>")
            else:
                try:
                    angle = float(parts[1])
                    self.execute_rotation_command(angle, 0)
                except ValueError:
                    print("❌ Invalid angle. Use numeric value.")
        
        # View commands
        elif command in ['view', 'show']:
            if self.nav_system is None:
                print("❌ Navigation system not initialized")
            else:
                self.nav_system.display_current_state()
        
        # Position commands
        elif command == 'position':
            if self.nav_system is None:
                print("❌ Navigation system not initialized")
            else:
                pos = self.nav_system.get_current_position()
                print(f"📍 Current Position:")
                print(f"   World: [{pos['world_position'][0]:.2f}, "
                      f"{pos['world_position'][1]:.2f}, "
                      f"{pos['world_position'][2]:.2f}] meters")
                print(f"   Map: [{pos['map_position'][0]:.1f}, "
                      f"{pos['map_position'][1]:.1f}] pixels")
        
        elif command == 'bounds':
            if self.nav_system is None:
                print("❌ Navigation system not initialized")
            else:
                bounds = self.nav_system.map_bounds
                print(f"🗺️  Map Coordinate Bounds:")
                print(f"   Resolution: {self.nav_system.map_resolution}x{self.nav_system.map_resolution} pixels")
                print(f"   World X: [{bounds['min_x']:.2f}, {bounds['max_x']:.2f}] meters")
                print(f"   World Z: [{bounds['min_z']:.2f}, {bounds['max_z']:.2f}] meters")
        
        # Scene commands
        elif command == 'scenes':
            if self.nav_system is None:
                print("❌ Navigation system not initialized")
            else:
                self.nav_system.list_available_scenes()
        
        elif command.startswith('switch '):
            if self.nav_system is None:
                print("❌ Navigation system not initialized")
            else:
                scene_id = command[7:].strip()
                print(f"🔄 Switching to scene: {scene_id}")
                self.nav_system.switch_scene(scene_id)
        
        elif command == 'reload':
            if self.nav_system is None:
                print("❌ Navigation system not initialized")
            else:
                current_scene = self.nav_system.scene_id
                print(f"🔄 Reloading scene: {current_scene}")
                self.nav_system.switch_scene(current_scene)
        
        # Utility commands
        elif command.startswith('save'):
            if self.nav_system is None:
                print("❌ Navigation system not initialized")
            else:
                parts = command.split()
                prefix = parts[1] if len(parts) > 1 else ""
                print("💾 Saving current images...")
                saved_files = self.nav_system.save_current_views(prefix)
                print(f"✅ Images saved to: {self.nav_system.images_path}")
                for view_type, filepath in saved_files.items():
                    print(f"   {view_type}: {os.path.basename(filepath)}")
        
        elif command == 'info':
            self.show_system_info()
        
        elif command == 'help':
            self.show_help()
        
        elif command in ['quit', 'exit']:
            self.running = False
            print("👋 Goodbye!")
        
        else:
            print(f"❓ Unknown command: '{command}'. Type 'help' for available commands.")
    
    def run(self):
        """Run the interactive interface"""
        print("🏠 Welcome to Habitat Lab Coordinate Navigation System!")
        print("Type 'help' for available commands or 'quit' to exit.")
        
        # Initialize system
        if not self.initialize_system():
            return
        
        # Show initial state
        print("\n📋 Displaying initial state...")
        self.nav_system.display_current_state()
        
        # Main interaction loop
        while self.running:
            try:
                command = input("\n🤖 Enter command: ").strip()
                if command:
                    self.process_command(command)
            except KeyboardInterrupt:
                print("\n\n👋 Exiting...")
                break
            except EOFError:
                print("\n\n👋 Exiting...")
                break
        
        # Cleanup
        if self.nav_system:
            self.nav_system.close()


def main():
    """Main function"""
    interface = InteractiveNavigationInterface()
    interface.run()


if __name__ == "__main__":
    main()
