"""
Habitat Environment Manager for Map Navigation Project

This module handles all interactions with the Habitat-sim environment,
including scene loading, agent initialization, movement commands, and sensor data retrieval.
"""

import os
import sys
import numpy as np
import quaternion
from typing import Optional, Tuple, Dict, Any, List
import random
import logging

# Add habitat-lab to path
habitat_lab_path = os.path.join(os.path.dirname(__file__), '..', 'habitat-lab')
if habitat_lab_path not in sys.path:
    sys.path.insert(0, habitat_lab_path)

import habitat
from habitat.config.default import get_agent_config
from habitat.core.simulator import Observations
from habitat.utils.visualizations import maps
from habitat_sim.utils.common import quat_from_angle_axis, quat_to_angle_axis

# Suppress Habitat logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"
logging.getLogger("habitat").setLevel(logging.WARNING)


class HabitatEnvironment:
    """
    Manages the Habitat simulation environment for map-based navigation.
    
    This class provides high-level interfaces for:
    - Loading MP3D scenes
    - Managing agent state (position, orientation, camera angles)
    - Converting between world coordinates and map coordinates
    - Executing navigation commands
    """
    
    def __init__(self, config_path: str, scene_id: str = None):
        """
        Initialize the Habitat environment.
        
        Args:
            config_path: Path to the Habitat configuration file
            scene_id: Optional scene ID to load. If None, uses config default.
        """
        self.config_path = config_path
        self.config = habitat.get_config(config_path)
        
        # Override scene if provided
        if scene_id:
            with habitat.config.read_write(self.config):
                self.config.habitat.dataset.scenes = [scene_id]
        
        # Initialize environment variables
        self.env: Optional[habitat.Env] = None
        self.current_position: Optional[np.ndarray] = None
        self.current_rotation: Optional[np.ndarray] = None
        self.map_info: Optional[Dict] = None
        self.step_count = 0
        
        # Camera control (pitch angle for look up/down)
        self.camera_pitch = 0.0  # Initial pitch angle in radians
        
        print(f"Habitat Environment initialized with config: {config_path}")
        if scene_id:
            print(f"Scene override: {scene_id}")
    
    def start_environment(self) -> bool:
        """
        Start the Habitat environment and initialize the agent.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Create dataset
            dataset = habitat.make_dataset(
                id_dataset=self.config.habitat.dataset.type,
                config=self.config.habitat.dataset
            )
            
            # Create environment
            self.env = habitat.Env(config=self.config, dataset=dataset)
            
            # Reset environment to get initial observations
            observations = self.env.reset()
            
            # Get initial agent state
            agent_state = self.env.sim.get_agent_state()
            self.current_position = agent_state.position.copy()
            self.current_rotation = agent_state.rotation.copy()
            
            # Initialize map information
            self._initialize_map_info()
            
            print(f"Environment started successfully!")
            print(f"Initial position: {self.current_position}")
            print(f"Initial rotation: {self.current_rotation}")
            print(f"Scene: {self.env.current_episode.scene_id}")
            
            return True
            
        except Exception as e:
            print(f"Error starting environment: {e}")
            return False
    
    def _initialize_map_info(self):
        """Initialize map coordinate system information."""
        if not self.env:
            return
            
        # Get the top-down map to establish coordinate system
        top_down_map = maps.get_topdown_map_from_sim(
            self.env.sim, map_resolution=1024
        )
        
        # Get map boundaries from the simulator
        bounds = self.env.sim.pathfinder.get_bounds()
        
        self.map_info = {
            'map_size': top_down_map.shape[:2],  # (height, width)
            'world_bounds': bounds,  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
            'map_resolution': 1024,
            'map_scale': None  # Will be calculated
        }
        
        # Calculate map scale (world units per pixel)
        world_width = bounds[1][0] - bounds[0][0]  # max_x - min_x
        world_height = bounds[1][2] - bounds[0][2]  # max_z - min_z
        
        map_width = self.map_info['map_size'][1]
        map_height = self.map_info['map_size'][0]
        
        scale_x = world_width / map_width
        scale_z = world_height / map_height
        
        # Use the same scale for both dimensions to maintain aspect ratio
        self.map_info['map_scale'] = max(scale_x, scale_z)
        
        print(f"Map initialized: size={self.map_info['map_size']}, "
              f"world_bounds={bounds}, scale={self.map_info['map_scale']:.4f}")
    
    def world_to_map_coordinates(self, world_pos: np.ndarray) -> Tuple[float, float]:
        """
        Convert 3D world coordinates to 2D map coordinates.
        
        Args:
            world_pos: 3D position in world coordinates [x, y, z]
            
        Returns:
            Tuple[float, float]: 2D map coordinates (x_map, y_map)
        """
        if not self.map_info:
            raise RuntimeError("Map info not initialized")
        
        bounds = self.map_info['world_bounds']
        map_size = self.map_info['map_size']
        scale = self.map_info['map_scale']
        
        # Convert world coordinates to map pixel coordinates
        # Note: In Habitat, X is right, Z is forward, Y is up
        # In map, we typically use X as horizontal, Y as vertical
        
        # Normalize to [0, 1] range
        norm_x = (world_pos[0] - bounds[0][0]) / (bounds[1][0] - bounds[0][0])
        norm_z = (world_pos[2] - bounds[0][2]) / (bounds[1][2] - bounds[0][2])
        
        # Convert to map coordinates
        # Note: Map Y-axis is typically inverted (0 at top)
        map_x = norm_x * map_size[1]
        map_y = (1.0 - norm_z) * map_size[0]  # Invert Y-axis
        
        return map_x, map_y
    
    def map_to_world_coordinates(self, map_x: float, map_y: float) -> np.ndarray:
        """
        Convert 2D map coordinates to 3D world coordinates.
        
        Args:
            map_x: X coordinate on the map
            map_y: Y coordinate on the map
            
        Returns:
            np.ndarray: 3D world position [x, y, z]
        """
        if not self.map_info:
            raise RuntimeError("Map info not initialized")
        
        bounds = self.map_info['world_bounds']
        map_size = self.map_info['map_size']
        
        # Normalize map coordinates to [0, 1]
        norm_x = map_x / map_size[1]
        norm_z = 1.0 - (map_y / map_size[0])  # Invert Y-axis
        
        # Convert to world coordinates
        world_x = bounds[0][0] + norm_x * (bounds[1][0] - bounds[0][0])
        world_z = bounds[0][2] + norm_z * (bounds[1][2] - bounds[0][2])
        
        # Get the appropriate Y (height) for this position
        world_y = self._get_navigable_height(world_x, world_z)
        
        return np.array([world_x, world_y, world_z])
    
    def _get_navigable_height(self, x: float, z: float) -> float:
        """
        Get the navigable height (Y coordinate) for a given X,Z position.
        
        Args:
            x: World X coordinate
            z: World Z coordinate
            
        Returns:
            float: Navigable Y coordinate
        """
        if not self.env:
            return 0.0
        
        # Use pathfinder to get the floor height at this position
        try:
            # Try to snap to navigable surface
            test_point = np.array([x, 0.0, z])
            snapped_point = self.env.sim.pathfinder.snap_point(test_point)
            
            if self.env.sim.pathfinder.is_navigable(snapped_point):
                return snapped_point[1]
            else:
                # If not navigable, try different heights
                for height in [0.0, 0.1, 0.2, 0.5, 1.0]:
                    test_point = np.array([x, height, z])
                    if self.env.sim.pathfinder.is_navigable(test_point):
                        return height
                
                # If still not navigable, return current agent height
                return self.current_position[1] if self.current_position is not None else 0.0
        except:
            # Fallback to current agent height
            return self.current_position[1] if self.current_position is not None else 0.0
    
    def move_to_position(self, target_pos: np.ndarray) -> bool:
        """
        Move agent to the target position using pathfinding.
        
        Args:
            target_pos: Target 3D world position [x, y, z]
            
        Returns:
            bool: True if movement successful, False otherwise
        """
        if not self.env:
            print("Environment not initialized")
            return False
        
        try:
            # Validate target position
            if not self.env.sim.pathfinder.is_navigable(target_pos):
                # Try to snap to nearest navigable point
                snapped_pos = self.env.sim.pathfinder.snap_point(target_pos)
                if self.env.sim.pathfinder.is_navigable(snapped_pos):
                    target_pos = snapped_pos
                    print(f"Target position snapped to navigable surface: {target_pos}")
                else:
                    print(f"Target position is not navigable: {target_pos}")
                    return False
            
            # Calculate direction to target for final orientation
            direction = target_pos - self.current_position
            direction[1] = 0  # Ignore Y component for horizontal direction
            direction = direction / np.linalg.norm(direction)
            
            # Calculate target rotation (yaw angle)
            target_yaw = np.arctan2(direction[0], direction[2])  # X, Z components
            target_rotation = quat_from_angle_axis(target_yaw, np.array([0, 1, 0]))
            
            # Set agent state directly
            agent_state = self.env.sim.get_agent_state()
            agent_state.position = target_pos
            agent_state.rotation = target_rotation
            
            self.env.sim.set_agent_state(agent_state)
            
            # Update internal state
            self.current_position = target_pos.copy()
            self.current_rotation = target_rotation.copy()
            
            print(f"Agent moved to position: {target_pos}")
            return True
            
        except Exception as e:
            print(f"Error moving to position: {e}")
            return False
    
    def turn_agent(self, direction: str, degrees: float) -> bool:
        """
        Turn the agent left or right by specified degrees.
        
        Args:
            direction: 'left' or 'right'
            degrees: Degrees to turn
            
        Returns:
            bool: True if turn successful, False otherwise
        """
        if not self.env:
            print("Environment not initialized")
            return False
        
        try:
            # Convert degrees to radians
            radians = np.radians(degrees)
            if direction.lower() == 'left':
                radians = -radians  # Left is negative rotation
            
            # Get current rotation and apply turn
            current_yaw = quat_to_angle_axis(self.current_rotation)[0]
            new_yaw = current_yaw + radians
            
            # Create new rotation quaternion
            new_rotation = quat_from_angle_axis(new_yaw, np.array([0, 1, 0]))
            
            # Apply rotation
            agent_state = self.env.sim.get_agent_state()
            agent_state.rotation = new_rotation
            self.env.sim.set_agent_state(agent_state)
            
            # Update internal state
            self.current_rotation = new_rotation.copy()
            
            print(f"Agent turned {direction} by {degrees} degrees")
            return True
            
        except Exception as e:
            print(f"Error turning agent: {e}")
            return False
    
    def look_vertical(self, direction: str, degrees: float) -> bool:
        """
        Adjust camera pitch (look up or down).
        
        Note: This adjusts the camera sensor orientation, not the agent body.
        
        Args:
            direction: 'up' or 'down'
            degrees: Degrees to adjust pitch
            
        Returns:
            bool: True if adjustment successful, False otherwise
        """
        try:
            # Convert degrees to radians
            radians = np.radians(degrees)
            if direction.lower() == 'down':
                radians = -radians  # Down is negative pitch
            
            # Update camera pitch
            self.camera_pitch += radians
            
            # Clamp pitch to reasonable range (-90 to +90 degrees)
            max_pitch = np.radians(90)
            self.camera_pitch = np.clip(self.camera_pitch, -max_pitch, max_pitch)
            
            # Update camera sensor orientation
            # Note: This is a simplified implementation
            # In a full implementation, you might need to modify the sensor configuration
            
            print(f"Camera pitched {direction} by {degrees} degrees (total pitch: {np.degrees(self.camera_pitch):.1f}°)")
            return True
            
        except Exception as e:
            print(f"Error adjusting camera pitch: {e}")
            return False
    
    def get_observations(self) -> Optional[Observations]:
        """
        Get current sensor observations from the environment.
        
        Returns:
            Observations: Current sensor data (RGB, depth, etc.)
        """
        if not self.env:
            return None
        
        try:
            # Get observations without stepping the environment
            observations = self.env.sim.get_sensor_observations()
            return observations
        except Exception as e:
            print(f"Error getting observations: {e}")
            return None
    
    def get_agent_state(self) -> Dict[str, Any]:
        """
        Get current agent state information.
        
        Returns:
            Dict with agent position, rotation, and other state info
        """
        if not self.env:
            return {}
        
        agent_state = self.env.sim.get_agent_state()
        
        # Convert quaternion to Euler angles for readability
        yaw_angle = quat_to_angle_axis(agent_state.rotation)[0]
        
        return {
            'position': agent_state.position,
            'rotation_quat': agent_state.rotation,
            'yaw_degrees': np.degrees(yaw_angle),
            'camera_pitch_degrees': np.degrees(self.camera_pitch),
            'step_count': self.step_count
        }
    
    def get_top_down_map(self) -> Optional[np.ndarray]:
        """
        Get the top-down map of the current scene.
        
        Returns:
            np.ndarray: Top-down map image
        """
        if not self.env:
            return None
        
        try:
            # Get colorized top-down map from simulator
            top_down_map = maps.get_topdown_map_from_sim(
                self.env.sim, map_resolution=1024
            )
            
            # Convert to RGB format for visualization
            if len(top_down_map.shape) == 2:
                # If grayscale, convert to RGB
                recolor_map = np.array([
                    [255, 255, 255],  # Navigable -> White
                    [128, 128, 128],  # Non-navigable -> Gray  
                    [0, 0, 0]         # Border -> Black
                ], dtype=np.uint8)
                top_down_map = recolor_map[top_down_map]
            
            return top_down_map
            
        except Exception as e:
            print(f"Error getting top-down map: {e}")
            return None
    
    def cleanup(self):
        """Clean up the environment resources."""
        if self.env:
            self.env.close()
            self.env = None
        print("Environment cleaned up")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
