#!/usr/bin/env python3
"""
Habitat Lab: Coordinate-based Navigation with Matterport Dataset

This program allows coordinate-based navigation through Matterport scenes using:
1. Global topdown view as reference map
2. First-person and third-person view rendering
3. Coordinate-based movement commands
4. View angle adjustment commands
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Dict, List, Tuple, Optional, Any
import math
import json
import random

import habitat
import habitat_sim
from habitat.config.default_structured_configs import (
    TopDownMapMeasurementConfig,
    FogOfWarConfig,
)
from habitat.utils.visualizations import maps
from habitat_sim.utils import viz_utils as vut
from habitat.core.simulator import Observations

# Quiet the Habitat simulator logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"


class CoordinateNavigationAgent:
    """
    Agent for coordinate-based navigation in Matterport scenes
    """
    
    def __init__(self, scene_id: str = None, data_path: str = None):
        """
        Initialize the navigation agent
        
        Args:
            scene_id: Specific scene ID to load (if None, will list available scenes)
            data_path: Path to habitat data directory
        """
        self.data_path = data_path or os.path.join(os.getcwd(), "data")
        self.scene_id = scene_id
        self.env = None
        self.sim = None
        self.current_position = [0.0, 0.0, 0.0]
        self.current_rotation = [0.0, 0.0, 0.0, 1.0]  # quaternion
        self.available_scenes = []
        
        # Camera parameters
        self.camera_height = 1.5  # meters
        self.fov = 90  # degrees
        
        self._setup_environment()
    
    def _setup_environment(self):
        """Setup Habitat environment with Matterport dataset"""
        # Check if any dataset exists
        scene_datasets_path = os.path.join(self.data_path, "scene_datasets")
        if not os.path.exists(scene_datasets_path):
            print(f"Scene datasets directory not found at {scene_datasets_path}")
            print("Please run the download script first:")
            print("python download_matterport.py")
            return
        
        # List available scenes
        self.available_scenes = self._get_available_scenes()
        if not self.available_scenes:
            print("No scenes found! Available datasets:")
            scene_datasets_path = os.path.join(self.data_path, "scene_datasets")
            if os.path.exists(scene_datasets_path):
                for item in os.listdir(scene_datasets_path):
                    print(f"  - {item}")
            return
        
        # Select scene
        if self.scene_id is None:
            self.scene_id = random.choice(self.available_scenes)
            print(f"Randomly selected scene: {self.scene_id}")
        
        # Create configuration
        config = self._create_config()
        
        # Initialize environment
        try:
            self.sim = habitat_sim.Simulator(config)
            print(f"Successfully loaded scene: {self.scene_id}")
            
            # Initialize random position
            self._initialize_random_position()
            
        except Exception as e:
            print(f"Error initializing simulator: {e}")
            return
    
    def _get_available_scenes(self) -> List[str]:
        """Get list of available scenes from multiple datasets"""
        scenes = []
        scene_datasets_path = os.path.join(self.data_path, "scene_datasets")
        
        # Check different dataset types
        dataset_paths = [
            ("mp3d", "mp3d"),  # Matterport3D
            ("habitat-test-scenes", "habitat-test-scenes"),  # Test scenes
            ("replica_cad", "replica_cad"),  # Replica CAD
            ("hm3d", "hm3d"),  # HM3D
        ]
        
        for dataset_name, dataset_dir in dataset_paths:
            dataset_path = os.path.join(scene_datasets_path, dataset_dir)
            if os.path.exists(dataset_path):
                # For habitat-test-scenes, files are directly in the directory
                if dataset_name == "habitat-test-scenes":
                    for item in os.listdir(dataset_path):
                        if item.endswith('.glb') or item.endswith('.ply'):
                            scene_name = item.replace('.glb', '').replace('.mesh.ply', '').replace('.ply', '')
                            scenes.append(f"{dataset_name}/{scene_name}")
                else:
                    # For other datasets, check subdirectories
                    for item in os.listdir(dataset_path):
                        scene_path = os.path.join(dataset_path, item)
                        if os.path.isdir(scene_path):
                            # Check if scene has required files
                            glb_file = os.path.join(scene_path, f"{item}.glb")
                            ply_file = os.path.join(scene_path, f"{item}.ply")
                            if os.path.exists(glb_file) or os.path.exists(ply_file):
                                scenes.append(f"{dataset_name}/{item}")
        
        return scenes
    
    def _create_config(self) -> habitat_sim.Configuration:
        """Create Habitat-Sim configuration"""
        # Parse scene path
        dataset_name, scene_id = self.scene_id.split('/', 1)
        
        # Determine scene file path
        if dataset_name == "habitat-test-scenes":
            # Files are directly in the dataset directory
            scene_base_path = os.path.join(self.data_path, "scene_datasets", dataset_name)
            scene_file = None
            for ext in [".glb", ".mesh.ply", ".ply"]:
                potential_file = os.path.join(scene_base_path, f"{scene_id}{ext}")
                if os.path.exists(potential_file):
                    scene_file = potential_file
                    break
        else:
            # Files are in subdirectories
            scene_base_path = os.path.join(
                self.data_path, 
                "scene_datasets", 
                dataset_name, 
                scene_id
            )
            scene_file = None
            for ext in [".glb", ".ply"]:
                potential_file = os.path.join(scene_base_path, f"{scene_id}{ext}")
                if os.path.exists(potential_file):
                    scene_file = potential_file
                    break
        
        if scene_file is None:
            raise ValueError(f"Scene file not found for {self.scene_id}")
        
        print(f"Loading scene file: {scene_file}")
        
        # Create settings
        settings = {
            "scene": scene_file,
            "default_agent": 0,
            "sensor_height": self.camera_height,
            "width": 512,
            "height": 512,
            "hfov": self.fov,
            "color_sensor": True,
            "depth_sensor": True,
            "semantic_sensor": False,
            "seed": 1,
            "enable_physics": False,
        }
        
        # Create configuration
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = settings["scene"]
        sim_cfg.enable_physics = settings["enable_physics"]
        
        # Create sensor specifications
        sensors = []
        
        # RGB sensor
        color_sensor_spec = habitat_sim.CameraSensorSpec()
        color_sensor_spec.uuid = "color_sensor"
        color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_sensor_spec.resolution = [settings["height"], settings["width"]]
        color_sensor_spec.postition = [0.0, settings["sensor_height"], 0.0]
        color_sensor_spec.hfov = np.radians(settings["hfov"])
        sensors.append(color_sensor_spec)
        
        # Depth sensor
        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = [settings["height"], settings["width"]]
        depth_sensor_spec.postition = [0.0, settings["sensor_height"], 0.0]
        depth_sensor_spec.hfov = np.radians(settings["hfov"])
        sensors.append(depth_sensor_spec)
        
        # Third person view sensor (higher up and looking down)
        third_person_spec = habitat_sim.CameraSensorSpec()
        third_person_spec.uuid = "third_person_sensor"
        third_person_spec.sensor_type = habitat_sim.SensorType.COLOR
        third_person_spec.resolution = [settings["height"], settings["width"]]
        third_person_spec.postition = [0.0, settings["sensor_height"] + 2.0, -1.0]
        third_person_spec.orientation = [-0.3, 0.0, 0.0]  # Look down
        sensors.append(third_person_spec)
        
        # Agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensors
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.1)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=5.0)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=5.0)
            ),
        }
        
        return habitat_sim.Configuration(sim_cfg, [agent_cfg])
    
    def _initialize_random_position(self):
        """Initialize agent at a random navigable position"""
        if self.sim is None:
            return
        
        # Get navigable points
        navigable_point = self.sim.pathfinder.get_random_navigable_point()
        
        # Set initial position
        self.current_position = [navigable_point[0], self.camera_height, navigable_point[2]]
        
        # Random rotation (quaternion format: [x, y, z, w])
        yaw = random.uniform(0, 2*np.pi)
        self.current_rotation = [0.0, np.sin(yaw/2), 0.0, np.cos(yaw/2)]
        
        # Set agent state
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array(self.current_position)
        agent_state.rotation = np.array(self.current_rotation)
        self.sim.get_agent(0).set_state(agent_state)
        
        print(f"Initialized at position: {self.current_position}")
    
    def get_current_observations(self) -> Dict[str, np.ndarray]:
        """Get current observations from all sensors"""
        if self.sim is None:
            return {}
        
        observations = self.sim.get_sensor_observations()
        return observations
    
    def get_topdown_map(self) -> np.ndarray:
        """Get topdown map of the current scene"""
        if self.sim is None:
            return np.array([])
        
        # Generate topdown map
        top_down_map = maps.get_topdown_map_from_sim(
            self.sim, map_resolution=1024, draw_border=True
        )
        
        # Recolor map for better visualization
        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )
        top_down_map = recolor_map[top_down_map]
        
        return top_down_map
    
    def move_to_coordinate(self, target_position: List[float]) -> bool:
        """
        Move agent to specified coordinates
        
        Args:
            target_position: [x, y, z] coordinates
            
        Returns:
            bool: True if movement successful
        """
        if self.sim is None:
            return False
        
        # Validate if position is navigable
        target_position = np.array(target_position)
        target_position[1] = self.camera_height  # Maintain camera height
        
        if not self.sim.pathfinder.is_navigable(target_position):
            print(f"Target position {target_position} is not navigable!")
            return False
        
        # Calculate direction to target
        current_pos = np.array(self.current_position)
        direction = target_position - current_pos
        direction[1] = 0  # Ignore vertical component for rotation
        
        if np.linalg.norm(direction) > 0:
            # Calculate rotation to face target (quaternion format)
            angle = np.arctan2(direction[2], direction[0])
            rotation = [0.0, np.sin(angle/2), 0.0, np.cos(angle/2)]
            
            # Update agent state
            agent_state = habitat_sim.AgentState()
            agent_state.position = target_position
            agent_state.rotation = np.array(rotation)
            self.sim.get_agent(0).set_state(agent_state)
            
            # Update internal state
            self.current_position = target_position.tolist()
            self.current_rotation = rotation
            
            print(f"Moved to position: {self.current_position}")
            return True
        
        return False
    
    def adjust_view_angle(self, yaw_delta: float = 0.0, pitch_delta: float = 0.0):
        """
        Adjust viewing angle
        
        Args:
            yaw_delta: Change in yaw angle (degrees)
            pitch_delta: Change in pitch angle (degrees)
        """
        if self.sim is None:
            return
        
        # Convert to radians
        yaw_rad = np.radians(yaw_delta)
        pitch_rad = np.radians(pitch_delta)
        
        # Get current rotation quaternion
        current_rotation = self.sim.get_agent(0).get_state().rotation
        
        # Create rotation quaternions for yaw and pitch
        yaw_quat = [0.0, np.sin(yaw_rad/2), 0.0, np.cos(yaw_rad/2)]
        
        # For simplicity, just update yaw for now
        # In a full implementation, you'd want proper quaternion multiplication
        new_rotation = yaw_quat
        
        # Update agent rotation
        agent_state = self.sim.get_agent(0).get_state()
        agent_state.rotation = np.array(new_rotation)
        self.sim.get_agent(0).set_state(agent_state)
        
        self.current_rotation = new_rotation
        print(f"Adjusted view angle - Yaw: {yaw_delta}°, Pitch: {pitch_delta}°")
    
    def display_observations(self, save_images: bool = False):
        """Display current observations"""
        observations = self.get_current_observations()
        
        if not observations:
            print("No observations available")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Current View - Scene: {self.scene_id}", fontsize=16)
        
        # Create safe filename
        safe_scene_id = self.scene_id.replace('/', '_')
        
        # First person RGB
        if "color_sensor" in observations:
            axes[0, 0].imshow(observations["color_sensor"])
            axes[0, 0].set_title("First Person View (RGB)")
            axes[0, 0].axis('off')
            
            if save_images:
                plt.imsave(f"first_person_rgb_{safe_scene_id}.png", observations["color_sensor"])
        
        # Depth view
        if "depth_sensor" in observations:
            depth_img = observations["depth_sensor"]
            axes[0, 1].imshow(depth_img, cmap='viridis')
            axes[0, 1].set_title("Depth View")
            axes[0, 1].axis('off')
            
            if save_images:
                plt.imsave(f"depth_view_{safe_scene_id}.png", depth_img, cmap='viridis')
        
        # Third person view
        if "third_person_sensor" in observations:
            axes[1, 0].imshow(observations["third_person_sensor"])
            axes[1, 0].set_title("Third Person View")
            axes[1, 0].axis('off')
            
            if save_images:
                plt.imsave(f"third_person_view_{safe_scene_id}.png", observations["third_person_sensor"])
        
        # Topdown map
        topdown_map = self.get_topdown_map()
        if topdown_map.size > 0:
            axes[1, 1].imshow(topdown_map)
            axes[1, 1].set_title("Topdown Map")
            axes[1, 1].axis('off')
            
            if save_images:
                plt.imsave(f"topdown_map_{safe_scene_id}.png", topdown_map)
        
        plt.tight_layout()
        plt.show()
    
    def list_available_scenes(self):
        """List all available scenes"""
        print("Available Matterport scenes:")
        for i, scene in enumerate(self.available_scenes):
            print(f"{i+1}. {scene}")
    
    def switch_scene(self, scene_id: str):
        """Switch to a different scene"""
        if scene_id in self.available_scenes:
            self.scene_id = scene_id
            if self.sim:
                self.sim.close()
            self._setup_environment()
        else:
            print(f"Scene {scene_id} not found in available scenes")
    
    def get_current_position(self) -> List[float]:
        """Get current agent position"""
        if isinstance(self.current_position, list):
            return self.current_position.copy()
        else:
            return self.current_position.tolist()
    
    def close(self):
        """Clean up resources"""
        if self.sim:
            self.sim.close()


def interactive_navigation():
    """Interactive navigation interface"""
    print("=== Habitat Coordinate Navigation ===")
    print("Loading navigation agent...")
    
    # Initialize agent
    agent = CoordinateNavigationAgent()
    
    if agent.sim is None:
        print("Failed to initialize agent. Please check Matterport dataset installation.")
        return
    
    print("\nNavigation Commands:")
    print("1. 'move x y z' - Move to coordinates (x, y, z)")
    print("2. 'rotate yaw pitch' - Adjust view angle (degrees)")
    print("3. 'view' - Display current observations")
    print("4. 'position' - Show current position")
    print("5. 'scenes' - List available scenes")
    print("6. 'switch scene_id' - Switch to different scene")
    print("7. 'help' - Show this help")
    print("8. 'quit' - Exit program")
    
    # Show initial view
    print(f"\nCurrent scene: {agent.scene_id}")
    print(f"Initial position: {agent.get_current_position()}")
    agent.display_observations()
    
    try:
        while True:
            command = input("\nEnter command: ").strip().lower()
            print(f"Debug: Received command: '{command}'")  # Debug line
            
            if command == 'quit':
                break
            elif command == 'help':
                print("\nNavigation Commands:")
                print("1. 'move x y z' - Move to coordinates")
                print("2. 'rotate yaw pitch' - Adjust view angle")
                print("3. 'view' - Display current observations")
                print("4. 'position' - Show current position")
                print("5. 'scenes' - List available scenes")
                print("6. 'switch scene_id' - Switch to different scene")
                print("7. 'quit' - Exit program")
            
            elif command == 'view':
                agent.display_observations()
            
            elif command == 'position':
                pos = agent.get_current_position()
                print(f"Current position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
            
            elif command == 'scenes':
                agent.list_available_scenes()
            
            elif command.startswith('move'):
                try:
                    parts = command.split()
                    if len(parts) == 4:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        success = agent.move_to_coordinate([x, y, z])
                        if success:
                            agent.display_observations()
                    else:
                        print("Usage: move x y z")
                except ValueError:
                    print("Invalid coordinates. Usage: move x y z")
            
            elif command.startswith('rotate'):
                try:
                    parts = command.split()
                    if len(parts) == 3:
                        yaw, pitch = float(parts[1]), float(parts[2])
                        agent.adjust_view_angle(yaw, pitch)
                        agent.display_observations()
                    else:
                        print("Usage: rotate yaw_degrees pitch_degrees")
                except ValueError:
                    print("Invalid angles. Usage: rotate yaw pitch")
            
            elif command.startswith('switch'):
                try:
                    parts = command.split()
                    if len(parts) == 2:
                        scene_id = parts[1]
                        agent.switch_scene(scene_id)
                        if agent.sim:
                            agent.display_observations()
                    else:
                        print("Usage: switch scene_id")
                except Exception as e:
                    print(f"Error switching scene: {e}")
            
            else:
                print("Unknown command. Type 'help' for available commands.")
    
    except KeyboardInterrupt:
        print("\nExiting...")
    
    finally:
        agent.close()


if __name__ == "__main__":
    interactive_navigation()
