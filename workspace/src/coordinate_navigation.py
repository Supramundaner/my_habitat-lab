#!/usr/bin/env python3
"""
Habitat Lab Coordinate Navigation System

This module provides coordinate-based navigation functionality for Habitat Lab
using Matterport3D dataset with 2D map coordinate system.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import cv2
from typing import Dict, List, Tuple, Optional, Any
import math
import json
import random
import time
from pathlib import Path

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


class CoordinateNavigationSystem:
    """
    Advanced coordinate-based navigation system for Habitat Lab
    with 2D map coordinate input and 3D world coordinate conversion
    """
    
    def __init__(self, scene_id: str = None, data_path: str = None, workspace_path: str = None):
        """
        Initialize the navigation system
        
        Args:
            scene_id: Specific scene ID to load
            data_path: Path to habitat data directory
            workspace_path: Path to workspace directory for saving images
        """
        self.workspace_path = workspace_path or os.path.dirname(os.path.abspath(__file__))
        self.data_path = data_path or os.path.join(os.path.dirname(self.workspace_path), "data")
        self.images_path = os.path.join(self.workspace_path, "images")
        
        # Create images directory if it doesn't exist
        os.makedirs(self.images_path, exist_ok=True)
        
        self.scene_id = scene_id
        self.sim = None
        self.current_position = [0.0, 0.0, 0.0]  # 3D world coordinates
        self.current_rotation = [0.0, 0.0, 0.0, 1.0]  # quaternion
        self.current_map_position = [0.0, 0.0]  # 2D map coordinates
        self.available_scenes = []
        
        # Map and coordinate system parameters
        self.map_resolution = 1024
        self.map_scale = 0.1  # meters per pixel
        self.camera_height = 1.5  # meters
        self.fov = 90  # degrees
        
        # Coordinate system bounds (will be set after loading scene)
        self.map_bounds = {"min_x": -10, "max_x": 10, "min_z": -10, "max_z": 10}
        
        # Initialize system
        self._setup_environment()
        
    def _setup_environment(self):
        """Setup Habitat environment with dataset detection"""
        # Check for available datasets
        scene_datasets_path = os.path.join(self.data_path, "scene_datasets")
        if not os.path.exists(scene_datasets_path):
            print(f"Scene datasets directory not found at {scene_datasets_path}")
            self._suggest_dataset_download()
            return
        
        # Get available scenes
        self.available_scenes = self._get_available_scenes()
        if not self.available_scenes:
            print("No scenes found!")
            self._suggest_dataset_download()
            return
        
        # Select scene
        if self.scene_id is None:
            self.scene_id = random.choice(self.available_scenes)
            print(f"Randomly selected scene: {self.scene_id}")
        
        # Initialize simulator
        try:
            config = self._create_config()
            self.sim = habitat_sim.Simulator(config)
            print(f"Successfully loaded scene: {self.scene_id}")
            
            # Initialize map bounds and position
            self._initialize_map_system()
            self._initialize_random_position()
            
        except Exception as e:
            print(f"Error initializing simulator: {e}")
            return
    
    def _get_available_scenes(self) -> List[str]:
        """Get list of available scenes from multiple datasets"""
        scenes = []
        scene_datasets_path = os.path.join(self.data_path, "scene_datasets")
        
        dataset_paths = [
            ("mp3d", "mp3d"),
            ("habitat-test-scenes", "habitat-test-scenes"),
            ("replica_cad", "replica_cad"),
            ("hm3d", "hm3d"),
        ]
        
        for dataset_name, dataset_dir in dataset_paths:
            dataset_path = os.path.join(scene_datasets_path, dataset_dir)
            if os.path.exists(dataset_path):
                if dataset_name == "habitat-test-scenes":
                    # Files are directly in the directory
                    for item in os.listdir(dataset_path):
                        if item.endswith('.glb') or item.endswith('.ply'):
                            scene_name = item.replace('.glb', '').replace('.mesh.ply', '').replace('.ply', '')
                            scenes.append(f"{dataset_name}/{scene_name}")
                else:
                    # Check subdirectories
                    for item in os.listdir(dataset_path):
                        scene_path = os.path.join(dataset_path, item)
                        if os.path.isdir(scene_path):
                            glb_file = os.path.join(scene_path, f"{item}.glb")
                            ply_file = os.path.join(scene_path, f"{item}.ply")
                            if os.path.exists(glb_file) or os.path.exists(ply_file):
                                scenes.append(f"{dataset_name}/{item}")
        
        return scenes
    
    def _suggest_dataset_download(self):
        """Suggest how to download datasets"""
        print("\nTo use this system, you need to download datasets first:")
        print("1. For test scenes (recommended for development):")
        print("   python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path ./data")
        print("\n2. For Matterport3D (requires academic license):")
        print("   python -m habitat_sim.utils.datasets_download --uids mp3d --data-path ./data")
        print("\n3. For HM3D minival:")
        print("   python -m habitat_sim.utils.datasets_download --uids hm3d_minival --data-path ./data")
    
    def _create_config(self) -> habitat_sim.Configuration:
        """Create Habitat-Sim configuration"""
        dataset_name, scene_id = self.scene_id.split('/', 1)
        
        # Determine scene file path
        if dataset_name == "habitat-test-scenes":
            scene_base_path = os.path.join(self.data_path, "scene_datasets", dataset_name)
            scene_file = None
            for ext in [".glb", ".mesh.ply", ".ply"]:
                potential_file = os.path.join(scene_base_path, f"{scene_id}{ext}")
                if os.path.exists(potential_file):
                    scene_file = potential_file
                    break
        else:
            scene_base_path = os.path.join(self.data_path, "scene_datasets", dataset_name, scene_id)
            scene_file = None
            for ext in [".glb", ".ply"]:
                potential_file = os.path.join(scene_base_path, f"{scene_id}{ext}")
                if os.path.exists(potential_file):
                    scene_file = potential_file
                    break
        
        if scene_file is None:
            raise ValueError(f"Scene file not found for {self.scene_id}")
        
        print(f"Loading scene file: {scene_file}")
        
        # Create simulator configuration
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = scene_file
        sim_cfg.enable_physics = False
        
        # Create sensor specifications
        sensors = []
        
        # RGB sensor (first person)
        color_sensor = habitat_sim.CameraSensorSpec()
        color_sensor.uuid = "color_sensor"
        color_sensor.sensor_type = habitat_sim.SensorType.COLOR
        color_sensor.resolution = [512, 512]
        color_sensor.position = [0.0, self.camera_height, 0.0]
        color_sensor.hfov = np.radians(self.fov)
        sensors.append(color_sensor)
        
        # Depth sensor
        depth_sensor = habitat_sim.CameraSensorSpec()
        depth_sensor.uuid = "depth_sensor"
        depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor.resolution = [512, 512]
        depth_sensor.position = [0.0, self.camera_height, 0.0]
        depth_sensor.hfov = np.radians(self.fov)
        sensors.append(depth_sensor)
        
        # Third person sensor (behind and above)
        third_person_sensor = habitat_sim.CameraSensorSpec()
        third_person_sensor.uuid = "third_person_sensor"
        third_person_sensor.sensor_type = habitat_sim.SensorType.COLOR
        third_person_sensor.resolution = [512, 512]
        third_person_sensor.position = [0.0, self.camera_height + 1.5, -2.0]
        third_person_sensor.orientation = [-0.2, 0.0, 0.0]  # Look down slightly
        sensors.append(third_person_sensor)
        
        # Agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensors
        
        return habitat_sim.Configuration(sim_cfg, [agent_cfg])
    
    def _initialize_map_system(self):
        """Initialize the map coordinate system"""
        if self.sim is None:
            return
        
        # Get scene bounds from pathfinder
        pathfinder = self.sim.pathfinder
        if pathfinder.is_loaded:
            bounds = pathfinder.get_bounds()
            self.map_bounds = {
                "min_x": bounds[0][0],
                "max_x": bounds[1][0], 
                "min_z": bounds[0][2],
                "max_z": bounds[1][2]
            }
            print(f"Map bounds: X[{self.map_bounds['min_x']:.2f}, {self.map_bounds['max_x']:.2f}], "
                  f"Z[{self.map_bounds['min_z']:.2f}, {self.map_bounds['max_z']:.2f}]")
        else:
            print("Warning: Pathfinder not loaded, using default bounds")
    
    def _initialize_random_position(self):
        """Initialize agent at a random navigable position"""
        if self.sim is None:
            return
        
        # Get random navigable point
        navigable_point = self.sim.pathfinder.get_random_navigable_point()
        self.current_position = [navigable_point[0], self.camera_height, navigable_point[2]]
        
        # Convert to map coordinates
        self.current_map_position = self._world_to_map_coords(self.current_position)
        
        # Random rotation
        yaw = random.uniform(0, 2*np.pi)
        self.current_rotation = [0.0, np.sin(yaw/2), 0.0, np.cos(yaw/2)]
        
        # Set agent state
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array(self.current_position)
        agent_state.rotation = np.array(self.current_rotation)
        self.sim.get_agent(0).set_state(agent_state)
        
        print(f"Initialized at world position: {self.current_position}")
        print(f"Map coordinates: {self.current_map_position}")
    
    def _world_to_map_coords(self, world_pos: List[float]) -> List[float]:
        """Convert 3D world coordinates to 2D map coordinates"""
        # Normalize to [0, 1] based on scene bounds
        x_norm = (world_pos[0] - self.map_bounds["min_x"]) / (self.map_bounds["max_x"] - self.map_bounds["min_x"])
        z_norm = (world_pos[2] - self.map_bounds["min_z"]) / (self.map_bounds["max_z"] - self.map_bounds["min_z"])
        
        # Convert to map pixel coordinates
        map_x = x_norm * self.map_resolution
        map_y = (1 - z_norm) * self.map_resolution  # Flip Y axis for display
        
        return [map_x, map_y]
    
    def _map_to_world_coords(self, map_pos: List[float]) -> List[float]:
        """Convert 2D map coordinates to 3D world coordinates"""
        # Normalize map coordinates to [0, 1]
        x_norm = map_pos[0] / self.map_resolution
        z_norm = 1 - (map_pos[1] / self.map_resolution)  # Flip Y axis
        
        # Convert to world coordinates
        world_x = self.map_bounds["min_x"] + x_norm * (self.map_bounds["max_x"] - self.map_bounds["min_x"])
        world_z = self.map_bounds["min_z"] + z_norm * (self.map_bounds["max_z"] - self.map_bounds["min_z"])
        
        # Get appropriate height from pathfinder
        test_point = np.array([world_x, self.camera_height, world_z])
        if self.sim.pathfinder.is_navigable(test_point):
            world_y = self.camera_height
        else:
            # Try to find nearest navigable point
            nearest_point = self.sim.pathfinder.get_random_navigable_point()
            world_y = nearest_point[1]
        
        return [world_x, world_y, world_z]
    
    def move_to_map_coordinate(self, map_x: float, map_y: float) -> bool:
        """
        Move agent to specified map coordinates
        
        Args:
            map_x: X coordinate in map pixel space
            map_y: Y coordinate in map pixel space
            
        Returns:
            bool: True if movement successful
        """
        if self.sim is None:
            return False
        
        # Convert map coordinates to world coordinates
        target_world_pos = self._map_to_world_coords([map_x, map_y])
        target_world_pos = np.array(target_world_pos)
        
        # Check if position is navigable
        if not self.sim.pathfinder.is_navigable(target_world_pos):
            print(f"Target position {target_world_pos} is not navigable!")
            return False
        
        # Calculate direction for rotation
        current_pos = np.array(self.current_position)
        direction = target_world_pos - current_pos
        direction[1] = 0  # Ignore Y component
        
        if np.linalg.norm(direction) > 0:
            # Calculate rotation to face movement direction
            angle = np.arctan2(direction[2], direction[0])
            rotation = [0.0, np.sin(angle/2), 0.0, np.cos(angle/2)]
            
            # Update agent state
            agent_state = habitat_sim.AgentState()
            agent_state.position = target_world_pos
            agent_state.rotation = np.array(rotation)
            self.sim.get_agent(0).set_state(agent_state)
            
            # Update internal state
            self.current_position = target_world_pos.tolist()
            self.current_rotation = rotation
            self.current_map_position = [map_x, map_y]
            
            print(f"Moved to world position: {self.current_position}")
            print(f"Map coordinates: {self.current_map_position}")
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
        
        # Simple yaw rotation (around Y-axis)
        current_rotation = np.array(self.current_rotation)
        yaw_quat = [0.0, np.sin(yaw_rad/2), 0.0, np.cos(yaw_rad/2)]
        
        # Update rotation (simplified - in practice you'd want proper quaternion multiplication)
        new_rotation = yaw_quat
        
        # Update agent state
        agent_state = self.sim.get_agent(0).get_state()
        agent_state.rotation = np.array(new_rotation)
        self.sim.get_agent(0).set_state(agent_state)
        
        self.current_rotation = new_rotation
        print(f"Adjusted view angle - Yaw: {yaw_delta}°, Pitch: {pitch_delta}°")
    
    def get_current_observations(self) -> Dict[str, np.ndarray]:
        """Get current observations from all sensors"""
        if self.sim is None:
            return {}
        
        observations = self.sim.get_sensor_observations()
        return observations
    
    def get_topdown_map_with_position(self) -> np.ndarray:
        """Get topdown map with current position marked"""
        if self.sim is None:
            return np.array([])
        
        # Generate base topdown map
        top_down_map = maps.get_topdown_map_from_sim(
            self.sim, map_resolution=self.map_resolution, draw_border=True
        )
        
        # Recolor map
        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )
        colored_map = recolor_map[top_down_map]
        
        # Add current position marker
        map_pos = self.current_map_position
        if 0 <= map_pos[0] < self.map_resolution and 0 <= map_pos[1] < self.map_resolution:
            # Draw a red circle for current position
            cv2.circle(colored_map, 
                      (int(map_pos[0]), int(map_pos[1])), 
                      10, (255, 0, 0), -1)
            
            # Draw direction indicator
            rotation_angle = self._get_rotation_angle()
            end_x = int(map_pos[0] + 20 * np.cos(rotation_angle))
            end_y = int(map_pos[1] + 20 * np.sin(rotation_angle))
            cv2.arrowedLine(colored_map,
                           (int(map_pos[0]), int(map_pos[1])),
                           (end_x, end_y),
                           (0, 0, 255), 3)
        
        return colored_map
    
    def _get_rotation_angle(self) -> float:
        """Get current rotation angle in radians"""
        # Extract yaw from quaternion (simplified)
        quat = self.current_rotation
        yaw = 2 * np.arctan2(quat[1], quat[3])
        return yaw
    
    def create_coordinate_system_overlay(self, image: np.ndarray) -> np.ndarray:
        """Add coordinate system overlay to image"""
        img_with_overlay = image.copy()
        height, width = img_with_overlay.shape[:2]
        
        # Add coordinate system text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 2
        
        # Add map bounds information
        text_lines = [
            f"Map Bounds:",
            f"X: [{self.map_bounds['min_x']:.1f}, {self.map_bounds['max_x']:.1f}]",
            f"Z: [{self.map_bounds['min_z']:.1f}, {self.map_bounds['max_z']:.1f}]",
            f"Current Map Pos: [{self.current_map_position[0]:.1f}, {self.current_map_position[1]:.1f}]"
        ]
        
        for i, line in enumerate(text_lines):
            y_pos = 30 + i * 25
            cv2.putText(img_with_overlay, line, (10, y_pos), 
                       font, font_scale, color, thickness)
        
        return img_with_overlay
    
    def save_current_views(self, prefix: str = "") -> Dict[str, str]:
        """Save current observations to files"""
        observations = self.get_current_observations()
        saved_files = {}
        
        timestamp = int(time.time())
        scene_name = self.scene_id.replace('/', '_')
        
        # Save first person view
        if "color_sensor" in observations:
            filename = f"{prefix}first_person_{scene_name}_{timestamp}.png"
            filepath = os.path.join(self.images_path, filename)
            cv2.imwrite(filepath, cv2.cvtColor(observations["color_sensor"], cv2.COLOR_RGB2BGR))
            saved_files["first_person"] = filepath
        
        # Save third person view
        if "third_person_sensor" in observations:
            filename = f"{prefix}third_person_{scene_name}_{timestamp}.png"
            filepath = os.path.join(self.images_path, filename)
            cv2.imwrite(filepath, cv2.cvtColor(observations["third_person_sensor"], cv2.COLOR_RGB2BGR))
            saved_files["third_person"] = filepath
        
        # Save topdown map with position
        topdown_map = self.get_topdown_map_with_position()
        if topdown_map.size > 0:
            # Add coordinate system overlay
            map_with_coords = self.create_coordinate_system_overlay(topdown_map)
            filename = f"{prefix}topdown_map_{scene_name}_{timestamp}.png"
            filepath = os.path.join(self.images_path, filename)
            cv2.imwrite(filepath, cv2.cvtColor(map_with_coords, cv2.COLOR_RGB2BGR))
            saved_files["topdown_map"] = filepath
        
        return saved_files
    
    def display_current_state(self, save_images: bool = True):
        """Display current state with all views"""
        observations = self.get_current_observations()
        
        if not observations:
            print("No observations available")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Current State - Scene: {self.scene_id}\n"
                    f"World Pos: [{self.current_position[0]:.2f}, {self.current_position[1]:.2f}, {self.current_position[2]:.2f}]\n"
                    f"Map Pos: [{self.current_map_position[0]:.1f}, {self.current_map_position[1]:.1f}]", 
                    fontsize=14)
        
        # First person view
        if "color_sensor" in observations:
            axes[0, 0].imshow(observations["color_sensor"])
            axes[0, 0].set_title("First Person View")
            axes[0, 0].axis('off')
        
        # Third person view
        if "third_person_sensor" in observations:
            axes[0, 1].imshow(observations["third_person_sensor"])
            axes[0, 1].set_title("Third Person View")
            axes[0, 1].axis('off')
        
        # Depth view
        if "depth_sensor" in observations:
            depth_img = observations["depth_sensor"]
            axes[1, 0].imshow(depth_img, cmap='viridis')
            axes[1, 0].set_title("Depth View")
            axes[1, 0].axis('off')
        
        # Topdown map with position
        topdown_map = self.get_topdown_map_with_position()
        if topdown_map.size > 0:
            axes[1, 1].imshow(topdown_map)
            axes[1, 1].set_title("Topdown Map (Red=Position, Blue=Direction)")
            axes[1, 1].axis('off')
            
            # Add coordinate system information
            axes[1, 1].text(0.02, 0.98, 
                           f"Map Bounds:\nX: [{self.map_bounds['min_x']:.1f}, {self.map_bounds['max_x']:.1f}]\n"
                           f"Z: [{self.map_bounds['min_z']:.1f}, {self.map_bounds['max_z']:.1f}]\n"
                           f"Current: [{self.current_map_position[0]:.1f}, {self.current_map_position[1]:.1f}]",
                           transform=axes[1, 1].transAxes, fontsize=8,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # Save images if requested
        if save_images:
            saved_files = self.save_current_views()
            print(f"Images saved to: {self.images_path}")
            for view_type, filepath in saved_files.items():
                print(f"  {view_type}: {os.path.basename(filepath)}")
    
    def list_available_scenes(self):
        """List all available scenes"""
        print(f"Available scenes ({len(self.available_scenes)}):")
        for i, scene in enumerate(self.available_scenes):
            print(f"  {i+1}. {scene}")
    
    def switch_scene(self, scene_id: str):
        """Switch to a different scene"""
        if scene_id in self.available_scenes:
            self.scene_id = scene_id
            if self.sim:
                self.sim.close()
            self._setup_environment()
        else:
            print(f"Scene {scene_id} not found in available scenes")
    
    def get_current_position(self) -> Dict[str, List[float]]:
        """Get current position in both coordinate systems"""
        return {
            "world_position": self.current_position.copy(),
            "map_position": self.current_map_position.copy()
        }
    
    def close(self):
        """Clean up resources"""
        if self.sim:
            self.sim.close()


def main():
    """Main function for testing the system"""
    print("=== Habitat Lab Coordinate Navigation System ===")
    
    # Initialize workspace path
    workspace_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(workspace_path), "data")
    
    # Initialize navigation system
    nav_system = CoordinateNavigationSystem(
        workspace_path=workspace_path,
        data_path=data_path
    )
    
    if nav_system.sim is None:
        print("Failed to initialize navigation system")
        return
    
    # Display initial state
    print("\n=== Initial State ===")
    nav_system.display_current_state()
    
    # Example navigation
    print("\n=== Navigation Test ===")
    current_pos = nav_system.get_current_position()
    print(f"Current position: {current_pos}")
    
    # Test movement to center of map
    center_x = nav_system.map_resolution // 2
    center_y = nav_system.map_resolution // 2
    
    print(f"Attempting to move to map center: ({center_x}, {center_y})")
    success = nav_system.move_to_map_coordinate(center_x, center_y)
    
    if success:
        print("Movement successful!")
        nav_system.display_current_state()
    else:
        print("Movement failed, trying alternative position...")
        # Try a different position
        alt_x = nav_system.current_map_position[0] + 50
        alt_y = nav_system.current_map_position[1] + 50
        success = nav_system.move_to_map_coordinate(alt_x, alt_y)
        if success:
            nav_system.display_current_state()
    
    # Test view adjustment
    print("\n=== View Adjustment Test ===")
    nav_system.adjust_view_angle(45, 0)
    nav_system.display_current_state()
    
    # Clean up
    nav_system.close()
    print("System closed successfully")


if __name__ == "__main__":
    main()
