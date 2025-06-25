#!/usr/bin/env python3
"""
Enhanced Terminal Interface for Habitat Lab Coordinate Navigation System

This module provides an advanced command-line interface with improved features:
- Colored output and emojis
- Command history and auto-completion
- Progress indicators
- Advanced error handling
- Interactive help system
- Session management
"""

import sys
import os
import time
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import readline
import atexit

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from coordinate_navigation import CoordinateNavigationSystem


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ProgressBar:
    """Simple progress bar for operations"""
    
    def __init__(self, total: int, width: int = 40):
        self.total = total
        self.width = width
        self.current = 0
    
    def update(self, current: int):
        self.current = current
        percent = current / self.total
        filled = int(self.width * percent)
        bar = '█' * filled + '░' * (self.width - filled)
        print(f'\r{Colors.CYAN}[{bar}] {percent*100:.1f}%{Colors.ENDC}', end='', flush=True)
    
    def finish(self):
        print()


class EnhancedTerminalInterface:
    """Enhanced terminal interface with advanced features"""
    
    def __init__(self):
        self.workspace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = os.path.join(os.path.dirname(self.workspace_path), "data")
        self.nav_system = None
        self.running = True
        self.session_start = datetime.now()
        self.command_history = []
        self.session_stats = {
            'commands_executed': 0,
            'movements': 0,
            'rotations': 0,
            'images_saved': 0,
            'scenes_switched': 0
        }
        
        # Setup command completion
        self.commands = [
            'move', 'goto', 'position', 'bounds', 'rotate', 'turn', 'view', 'show',
            'scenes', 'switch', 'reload', 'save', 'info', 'help', 'quit', 'exit',
            'history', 'stats', 'clear', 'demo', 'explore', 'random', 'center'
        ]
        
        self.setup_readline()
        self.print_banner()
    
    def setup_readline(self):
        """Setup readline for command history and completion"""
        def completer(text, state):
            options = [cmd for cmd in self.commands if cmd.startswith(text)]
            if state < len(options):
                return options[state]
            else:
                return None
        
        readline.parse_and_bind("tab: complete")
        readline.set_completer(completer)
        
        # Load command history
        histfile = os.path.join(self.workspace_path, '.command_history')
        try:
            readline.read_history_file(histfile)
        except FileNotFoundError:
            pass
        
        # Save history on exit
        atexit.register(readline.write_history_file, histfile)
    
    def print_banner(self):
        """Print startup banner"""
        print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}")
        print("  🏠 HABITAT LAB COORDINATE NAVIGATION SYSTEM")
        print("     Enhanced Terminal Interface v2.0")
        print(f"{Colors.ENDC}{Colors.HEADER}{'='*80}{Colors.ENDC}")
        print()
        print(f"{Colors.CYAN}🚀 Features:{Colors.ENDC}")
        print("  • Advanced command completion and history")
        print("  • Colored output and progress indicators")
        print("  • Session statistics and exploration tools")
        print("  • Interactive demos and guided exploration")
        print()
        print(f"{Colors.WARNING}💡 Type 'help' for commands or 'demo' for a guided tour{Colors.ENDC}")
        print()
    
    def print_colored(self, text: str, color: str = Colors.ENDC):
        """Print colored text"""
        print(f"{color}{text}{Colors.ENDC}")
    
    def print_success(self, text: str):
        """Print success message"""
        self.print_colored(f"✅ {text}", Colors.GREEN)
    
    def print_error(self, text: str):
        """Print error message"""
        self.print_colored(f"❌ {text}", Colors.FAIL)
    
    def print_warning(self, text: str):
        """Print warning message"""
        self.print_colored(f"⚠️  {text}", Colors.WARNING)
    
    def print_info(self, text: str):
        """Print info message"""
        self.print_colored(f"ℹ️  {text}", Colors.CYAN)
    
    def initialize_system(self, scene_id: Optional[str] = None) -> bool:
        """Initialize the navigation system with progress indicator"""
        self.print_info("Initializing navigation system...")
        
        # Simulate progress for better UX
        progress = ProgressBar(10)
        for i in range(11):
            progress.update(i)
            time.sleep(0.1)
        progress.finish()
        
        try:
            self.nav_system = CoordinateNavigationSystem(
                scene_id=scene_id,
                workspace_path=self.workspace_path,
                data_path=self.data_path
            )
            
            if self.nav_system.sim is None:
                self.print_error("Failed to initialize navigation system")
                self.print_warning("Please run 'python download_datasets.py' to download datasets")
                return False
            
            self.print_success(f"Successfully initialized with scene: {self.nav_system.scene_id}")
            return True
            
        except Exception as e:
            self.print_error(f"Initialization failed: {str(e)}")
            return False
    
    def show_enhanced_help(self):
        """Display enhanced help information"""
        print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}HABITAT LAB COORDINATE NAVIGATION - COMMAND REFERENCE{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
        
        sections = [
            ("📍 NAVIGATION", [
                ("move <x> <y>", "Move to map coordinates (e.g., move 512 256)"),
                ("goto <x> <y>", "Same as move command"),
                ("position", "Show current position in both coordinate systems"),
                ("bounds", "Display map coordinate bounds and limits"),
                ("center", "Move to center of the map"),
                ("random", "Move to a random navigable position")
            ]),
            ("👁️  VIEW CONTROL", [
                ("rotate <yaw> [pitch]", "Rotate view by degrees (e.g., rotate 90 15)"),
                ("turn <angle>", "Quick turn by specified angle"),
                ("view", "Display current views and observations"),
                ("show", "Same as view command")
            ]),
            ("🗺️  SCENE MANAGEMENT", [
                ("scenes", "List all available scenes with details"),
                ("switch <scene_id>", "Switch to different scene"),
                ("reload", "Reload current scene"),
                ("info", "Show detailed system information")
            ]),
            ("💾 DATA & EXPORT", [
                ("save [prefix]", "Save current images with optional prefix"),
                ("explore", "Start guided exploration mode"),
                ("demo", "Run interactive demonstration")
            ]),
            ("🔧 SYSTEM UTILITIES", [
                ("history", "Show command history"),
                ("stats", "Display session statistics"),
                ("clear", "Clear terminal screen"),
                ("help", "Show this help information"),
                ("quit/exit", "Exit the program")
            ])
        ]
        
        for section_title, commands in sections:
            print(f"\n{Colors.CYAN}{section_title}:{Colors.ENDC}")
            for cmd, desc in commands:
                print(f"  {Colors.GREEN}{cmd:<20}{Colors.ENDC} - {desc}")
        
        print(f"\n{Colors.WARNING}📋 EXAMPLES:{Colors.ENDC}")
        examples = [
            "move 512 256                    # Move to center-left of map",
            "rotate 90                      # Turn right 90 degrees",
            "switch habitat-test-scenes/van-gogh-room  # Change scene",
            "save exploration_              # Save with custom prefix",
            "demo                           # Start interactive demo"
        ]
        
        for example in examples:
            print(f"  {Colors.BLUE}{example}{Colors.ENDC}")
        
        print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
    
    def show_session_stats(self):
        """Display session statistics"""
        duration = datetime.now() - self.session_start
        hours, remainder = divmod(int(duration.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\n{Colors.CYAN}📊 SESSION STATISTICS{Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*40}{Colors.ENDC}")
        print(f"Session Duration: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"Commands Executed: {self.session_stats['commands_executed']}")
        print(f"Movements: {self.session_stats['movements']}")
        print(f"View Rotations: {self.session_stats['rotations']}")
        print(f"Images Saved: {self.session_stats['images_saved']}")
        print(f"Scene Switches: {self.session_stats['scenes_switched']}")
        
        if self.nav_system:
            pos = self.nav_system.get_current_position()
            print(f"Current Scene: {self.nav_system.scene_id}")
            print(f"Current Position: ({pos['map_position'][0]:.1f}, {pos['map_position'][1]:.1f})")
        
        print(f"{Colors.CYAN}{'='*40}{Colors.ENDC}")
    
    def show_command_history(self):
        """Display command history"""
        if not self.command_history:
            self.print_info("No commands in history")
            return
        
        print(f"\n{Colors.CYAN}📜 COMMAND HISTORY (last 10):{Colors.ENDC}")
        print(f"{Colors.CYAN}{'-'*40}{Colors.ENDC}")
        
        recent_commands = self.command_history[-10:]
        for i, (timestamp, command) in enumerate(recent_commands, 1):
            time_str = timestamp.strftime("%H:%M:%S")
            print(f"{Colors.BLUE}{i:2d}.{Colors.ENDC} [{time_str}] {command}")
    
    def run_demo(self):
        """Run interactive demonstration"""
        if not self.nav_system:
            self.print_error("Navigation system not initialized")
            return
        
        self.print_info("🎬 Starting interactive demonstration...")
        print(f"\n{Colors.WARNING}This demo will move the agent and rotate the view.{Colors.ENDC}")
        
        demo_steps = [
            ("Moving to center of map", lambda: self.execute_move_command(512, 512)),
            ("Rotating view 90 degrees", lambda: self.execute_rotation_command(90, 0)),
            ("Moving to corner", lambda: self.execute_move_command(256, 256)),
            ("Looking around", lambda: self.execute_rotation_command(180, 15)),
            ("Returning to center", lambda: self.execute_move_command(512, 512))
        ]
        
        for i, (description, action) in enumerate(demo_steps, 1):
            input(f"\nPress Enter to continue with step {i}: {description}")
            action()
            time.sleep(1)
        
        self.print_success("Demo completed! You can now explore on your own.")
    
    def explore_mode(self):
        """Start guided exploration mode"""
        if not self.nav_system:
            self.print_error("Navigation system not initialized")
            return
        
        self.print_info("🗺️  Starting guided exploration mode...")
        
        # Generate some interesting points
        center = self.nav_system.map_resolution // 2
        exploration_points = [
            (center, center, "Center of the map"),
            (center - 200, center - 200, "Northwest area"),
            (center + 200, center - 200, "Northeast area"),
            (center + 200, center + 200, "Southeast area"),
            (center - 200, center + 200, "Southwest area")
        ]
        
        print(f"\n{Colors.CYAN}Suggested exploration points:{Colors.ENDC}")
        for i, (x, y, desc) in enumerate(exploration_points, 1):
            print(f"  {i}. {desc} - Map coords: ({x}, {y})")
        
        while True:
            try:
                choice = input(f"\n{Colors.WARNING}Choose point (1-{len(exploration_points)}) or 'q' to quit: {Colors.ENDC}")
                if choice.lower() == 'q':
                    break
                
                idx = int(choice) - 1
                if 0 <= idx < len(exploration_points):
                    x, y, desc = exploration_points[idx]
                    print(f"\n🚶 Exploring: {desc}")
                    self.execute_move_command(x, y)
                    
                    # Rotate for better view
                    self.execute_rotation_command(45, 0)
                    
                    # Ask if user wants to save
                    save = input(f"{Colors.WARNING}Save images of this location? (y/n): {Colors.ENDC}")
                    if save.lower() == 'y':
                        prefix = f"explore_{idx+1}_"
                        self.execute_save_command(prefix)
                else:
                    self.print_error("Invalid choice")
                    
            except ValueError:
                self.print_error("Please enter a valid number")
            except KeyboardInterrupt:
                break
        
        self.print_info("Exploration mode ended")
    
    def parse_coordinate_command(self, command: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """Parse coordinate commands with enhanced error handling"""
        parts = command.strip().split()
        if len(parts) < 3:
            return None, None, "Invalid format. Use: move <x> <y>"
        
        try:
            x = float(parts[1])
            y = float(parts[2])
            
            # Validate coordinates
            max_coord = self.nav_system.map_resolution if self.nav_system else 1024
            if not (0 <= x <= max_coord and 0 <= y <= max_coord):
                return x, y, f"Coordinates outside valid range [0, {max_coord}]"
            
            return x, y, None
        except ValueError:
            return None, None, "Invalid coordinates. Use numeric values."
    
    def execute_move_command(self, x: float, y: float):
        """Execute movement command with enhanced feedback"""
        if not self.nav_system:
            self.print_error("Navigation system not initialized")
            return
        
        print(f"\n🚶 Moving to map coordinates ({x:.1f}, {y:.1f})...")
        
        # Show progress animation
        progress = ProgressBar(5, 30)
        for i in range(6):
            progress.update(i)
            time.sleep(0.1)
        progress.finish()
        
        success = self.nav_system.move_to_map_coordinate(x, y)
        
        if success:
            self.print_success("Movement successful!")
            self.session_stats['movements'] += 1
            
            # Show new position
            pos = self.nav_system.get_current_position()
            print(f"📍 New position: Map({pos['map_position'][0]:.1f}, {pos['map_position'][1]:.1f}) "
                  f"World({pos['world_position'][0]:.2f}, {pos['world_position'][2]:.2f})")
            
            # Automatically display new view
            self.nav_system.display_current_state()
        else:
            self.print_error("Movement failed - position not navigable")
            self.print_warning("Try coordinates closer to navigable areas (gray areas on topdown map)")
    
    def execute_rotation_command(self, yaw: float, pitch: float = 0.0):
        """Execute rotation command with enhanced feedback"""
        if not self.nav_system:
            self.print_error("Navigation system not initialized")
            return
        
        print(f"🔄 Rotating view - Yaw: {yaw}°, Pitch: {pitch}°")
        self.nav_system.adjust_view_angle(yaw, pitch)
        self.session_stats['rotations'] += 1
        
        # Show rotation feedback
        total_rotation = abs(yaw) + abs(pitch)
        if total_rotation > 90:
            print("🌀 Large rotation applied")
        elif total_rotation > 45:
            print("↻ Medium rotation applied")
        else:
            print("↺ Small rotation applied")
        
        # Automatically display new view
        self.nav_system.display_current_state()
    
    def execute_save_command(self, prefix: str = ""):
        """Execute save command with enhanced feedback"""
        if not self.nav_system:
            self.print_error("Navigation system not initialized")
            return
        
        print("💾 Saving current images...")
        
        # Show progress
        progress = ProgressBar(3, 25)
        progress.update(1)
        
        saved_files = self.nav_system.save_current_views(prefix)
        progress.update(2)
        
        self.session_stats['images_saved'] += len(saved_files)
        progress.update(3)
        progress.finish()
        
        self.print_success(f"Images saved to: {self.nav_system.images_path}")
        for view_type, filepath in saved_files.items():
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            print(f"  📷 {view_type}: {os.path.basename(filepath)} ({file_size:.1f} MB)")
    
    def process_command(self, command: str):
        """Process user command with enhanced error handling"""
        original_command = command.strip()
        command = command.strip().lower()
        
        if not command:
            return
        
        # Add to history
        self.command_history.append((datetime.now(), original_command))
        self.session_stats['commands_executed'] += 1
        
        try:
            # Navigation commands
            if command.startswith(('move ', 'goto ')):
                x, y, error = self.parse_coordinate_command(command)
                if error:
                    self.print_error(error)
                else:
                    self.execute_move_command(x, y)
            
            elif command == 'center':
                if self.nav_system:
                    center = self.nav_system.map_resolution // 2
                    self.execute_move_command(center, center)
                else:
                    self.print_error("Navigation system not initialized")
            
            elif command == 'random':
                if self.nav_system:
                    import random
                    max_coord = self.nav_system.map_resolution
                    x = random.randint(100, max_coord - 100)
                    y = random.randint(100, max_coord - 100)
                    print(f"🎲 Attempting random position: ({x}, {y})")
                    self.execute_move_command(x, y)
                else:
                    self.print_error("Navigation system not initialized")
            
            # Rotation commands
            elif command.startswith('rotate '):
                parts = command.split()
                if len(parts) < 2:
                    self.print_error("Invalid format. Use: rotate <yaw> [pitch]")
                else:
                    try:
                        yaw = float(parts[1])
                        pitch = float(parts[2]) if len(parts) > 2 else 0.0
                        self.execute_rotation_command(yaw, pitch)
                    except ValueError:
                        self.print_error("Invalid angles. Use numeric values.")
            
            elif command.startswith('turn '):
                parts = command.split()
                if len(parts) < 2:
                    self.print_error("Invalid format. Use: turn <angle>")
                else:
                    try:
                        angle = float(parts[1])
                        self.execute_rotation_command(angle, 0)
                    except ValueError:
                        self.print_error("Invalid angle. Use numeric value.")
            
            # View commands
            elif command in ['view', 'show']:
                if self.nav_system:
                    self.nav_system.display_current_state()
                else:
                    self.print_error("Navigation system not initialized")
            
            # Position commands
            elif command == 'position':
                if self.nav_system:
                    pos = self.nav_system.get_current_position()
                    print(f"\n📍 Current Position:")
                    print(f"   🌍 World: [{pos['world_position'][0]:.2f}, "
                          f"{pos['world_position'][1]:.2f}, "
                          f"{pos['world_position'][2]:.2f}] meters")
                    print(f"   🗺️  Map: [{pos['map_position'][0]:.1f}, "
                          f"{pos['map_position'][1]:.1f}] pixels")
                else:
                    self.print_error("Navigation system not initialized")
            
            elif command == 'bounds':
                if self.nav_system:
                    bounds = self.nav_system.map_bounds
                    print(f"\n🗺️  Map Information:")
                    print(f"   Resolution: {self.nav_system.map_resolution}x{self.nav_system.map_resolution} pixels")
                    print(f"   Scale: {self.nav_system.map_scale} meters/pixel")
                    print(f"   World Bounds X: [{bounds['min_x']:.2f}, {bounds['max_x']:.2f}] meters")
                    print(f"   World Bounds Z: [{bounds['min_z']:.2f}, {bounds['max_z']:.2f}] meters")
                else:
                    self.print_error("Navigation system not initialized")
            
            # Scene commands
            elif command == 'scenes':
                if self.nav_system:
                    self.nav_system.list_available_scenes()
                else:
                    self.print_error("Navigation system not initialized")
            
            elif command.startswith('switch '):
                if self.nav_system:
                    scene_id = original_command[7:].strip()  # Use original case
                    print(f"🔄 Switching to scene: {scene_id}")
                    
                    # Show progress
                    progress = ProgressBar(8)
                    for i in range(9):
                        progress.update(i)
                        time.sleep(0.2)
                    progress.finish()
                    
                    success = self.nav_system.switch_scene(scene_id)
                    if success:
                        self.session_stats['scenes_switched'] += 1
                        self.print_success(f"Switched to scene: {scene_id}")
                    else:
                        self.print_error("Failed to switch scene")
                else:
                    self.print_error("Navigation system not initialized")
            
            elif command == 'reload':
                if self.nav_system:
                    current_scene = self.nav_system.scene_id
                    print(f"🔄 Reloading scene: {current_scene}")
                    self.nav_system.switch_scene(current_scene)
                else:
                    self.print_error("Navigation system not initialized")
            
            # Utility commands
            elif command.startswith('save'):
                parts = original_command.split()  # Use original case for prefix
                prefix = parts[1] if len(parts) > 1 else ""
                self.execute_save_command(prefix)
            
            elif command == 'info':
                if self.nav_system:
                    self.show_system_info()
                else:
                    self.print_error("Navigation system not initialized")
            
            elif command == 'help':
                self.show_enhanced_help()
            
            elif command == 'history':
                self.show_command_history()
            
            elif command == 'stats':
                self.show_session_stats()
            
            elif command == 'clear':
                os.system('clear' if os.name == 'posix' else 'cls')
                self.print_banner()
            
            elif command == 'demo':
                self.run_demo()
            
            elif command == 'explore':
                self.explore_mode()
            
            elif command in ['quit', 'exit']:
                self.running = False
                print(f"\n{Colors.GREEN}👋 Session Summary:{Colors.ENDC}")
                self.show_session_stats()
                print(f"\n{Colors.CYAN}Thank you for using Habitat Lab Navigation System!{Colors.ENDC}")
            
            else:
                self.print_error(f"Unknown command: '{original_command}'")
                self.print_info("Type 'help' for available commands")
        
        except Exception as e:
            self.print_error(f"Command execution failed: {str(e)}")
            self.print_info("Type 'help' for available commands")
    
    def show_system_info(self):
        """Display enhanced system information"""
        if not self.nav_system:
            self.print_error("Navigation system not initialized")
            return
        
        print(f"\n{Colors.CYAN}{'='*60}{Colors.ENDC}")
        print(f"{Colors.CYAN}{Colors.BOLD}SYSTEM INFORMATION{Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}")
        
        # Scene information
        print(f"\n{Colors.GREEN}🏠 Scene Information:{Colors.ENDC}")
        print(f"   Current Scene: {self.nav_system.scene_id}")
        print(f"   Available Scenes: {len(self.nav_system.available_scenes)}")
        
        # Map configuration
        print(f"\n{Colors.GREEN}🗺️  Map Configuration:{Colors.ENDC}")
        print(f"   Resolution: {self.nav_system.map_resolution}x{self.nav_system.map_resolution} pixels")
        print(f"   Scale: {self.nav_system.map_scale} meters/pixel")
        print(f"   Camera Height: {self.nav_system.camera_height} meters")
        print(f"   Field of View: {self.nav_system.fov} degrees")
        
        # World bounds
        bounds = self.nav_system.map_bounds
        print(f"\n{Colors.GREEN}🌍 World Coordinate Bounds:{Colors.ENDC}")
        print(f"   X Range: [{bounds['min_x']:.2f}, {bounds['max_x']:.2f}] meters")
        print(f"   Z Range: [{bounds['min_z']:.2f}, {bounds['max_z']:.2f}] meters")
        print(f"   Total Area: {(bounds['max_x']-bounds['min_x'])*(bounds['max_z']-bounds['min_z']):.1f} m²")
        
        # Current position
        current_pos = self.nav_system.get_current_position()
        print(f"\n{Colors.GREEN}📍 Current Position:{Colors.ENDC}")
        print(f"   World: [{current_pos['world_position'][0]:.2f}, "
              f"{current_pos['world_position'][1]:.2f}, "
              f"{current_pos['world_position'][2]:.2f}] meters")
        print(f"   Map: [{current_pos['map_position'][0]:.1f}, "
              f"{current_pos['map_position'][1]:.1f}] pixels")
        
        # File paths
        print(f"\n{Colors.GREEN}📁 Directories:{Colors.ENDC}")
        print(f"   Workspace: {self.workspace_path}")
        print(f"   Data: {self.data_path}")
        print(f"   Images: {self.nav_system.images_path}")
        
        print(f"\n{Colors.CYAN}{'='*60}{Colors.ENDC}")
    
    def run(self):
        """Run the enhanced terminal interface"""
        # Initialize system
        if not self.initialize_system():
            return
        
        # Show initial state
        self.print_info("Displaying initial state...")
        self.nav_system.display_current_state()
        
        print(f"\n{Colors.WARNING}🎯 Ready for commands! Try 'demo' for a guided tour or 'help' for all commands.{Colors.ENDC}")
        
        # Main interaction loop
        while self.running:
            try:
                prompt = f"\n{Colors.BOLD}{Colors.BLUE}🤖 Command >{Colors.ENDC} "
                command = input(prompt).strip()
                if command:
                    self.process_command(command)
            except KeyboardInterrupt:
                print(f"\n\n{Colors.WARNING}Use 'quit' or 'exit' to close properly{Colors.ENDC}")
            except EOFError:
                self.running = False
                print(f"\n\n{Colors.CYAN}👋 Goodbye!{Colors.ENDC}")
        
        # Cleanup
        if self.nav_system:
            self.nav_system.close()


def main():
    """Main function"""
    try:
        interface = EnhancedTerminalInterface()
        interface.run()
    except Exception as e:
        print(f"{Colors.FAIL}Fatal error: {str(e)}{Colors.ENDC}")
        sys.exit(1)


if __name__ == "__main__":
    main()
