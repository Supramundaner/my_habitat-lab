"""
Map Visualizer for Habitat Navigation Project

This module handles all map visualization functionality including:
- Loading and displaying MP3D top-down maps
- Drawing coordinate grids and agent markers
- Converting between coordinate systems
- Generating composite visualization images
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import Tuple, Optional, Dict, Any
import cv2

# Add habitat-lab to path
habitat_lab_path = os.path.join(os.path.dirname(__file__), '..', 'habitat-lab')
if habitat_lab_path not in sys.path:
    sys.path.insert(0, habitat_lab_path)


class MapVisualizer:
    """
    Handles visualization of top-down maps with agent position and coordinate grid.
    
    This class provides functionality to:
    - Display colorized MP3D top-down maps
    - Overlay coordinate grids with black lines and labels
    - Mark agent position and orientation
    - Generate publication-ready map images
    """
    
    def __init__(self, map_data: np.ndarray, map_info: Dict[str, Any]):
        """
        Initialize the map visualizer.
        
        Args:
            map_data: Top-down map image data (H x W x 3)
            map_info: Dictionary containing map metadata (bounds, scale, etc.)
        """
        self.map_data = map_data
        self.map_info = map_info
        
        # Grid configuration
        self.grid_spacing_pixels = 50  # Grid spacing in pixels
        self.grid_color = 'black'
        self.grid_linewidth = 0.8
        self.grid_alpha = 0.8
        
        # Agent marker configuration
        self.agent_radius = 8  # Agent marker radius in pixels
        self.agent_color = 'red'
        self.agent_arrow_length = 15  # Length of direction arrow
        self.agent_arrow_width = 3
        
        print(f"Map visualizer initialized with map size: {map_data.shape}")
    
    def create_coordinate_grid(self, fig, ax) -> None:
        """
        Draw coordinate grid on the map with black lines and labels.
        
        Args:
            fig: Matplotlib figure object
            ax: Matplotlib axes object
        """
        height, width = self.map_data.shape[:2]
        
        # Calculate grid lines
        x_positions = np.arange(0, width, self.grid_spacing_pixels)
        y_positions = np.arange(0, height, self.grid_spacing_pixels)
        
        # Draw vertical grid lines
        for x in x_positions:
            ax.axvline(x=x, color=self.grid_color, linewidth=self.grid_linewidth, 
                      alpha=self.grid_alpha, linestyle='-')
        
        # Draw horizontal grid lines
        for y in y_positions:
            ax.axhline(y=y, color=self.grid_color, linewidth=self.grid_linewidth,
                      alpha=self.grid_alpha, linestyle='-')
        
        # Add coordinate labels
        self._add_coordinate_labels(ax, x_positions, y_positions)
    
    def _add_coordinate_labels(self, ax, x_positions: np.ndarray, y_positions: np.ndarray) -> None:
        """
        Add coordinate labels to the grid.
        
        Args:
            ax: Matplotlib axes object
            x_positions: X positions for vertical grid lines
            y_positions: Y positions for horizontal grid lines
        """
        # Add X-axis labels (bottom of map)
        for i, x in enumerate(x_positions[::2]):  # Every other line to avoid crowding
            world_coord = self._pixel_to_world_coord(x, 0)
            label = f"{world_coord[0]:.1f}"
            ax.text(x, self.map_data.shape[0] - 5, label, 
                   ha='center', va='top', fontsize=8, 
                   color=self.grid_color, weight='bold')
        
        # Add Y-axis labels (left side of map)
        for i, y in enumerate(y_positions[::2]):  # Every other line to avoid crowding
            world_coord = self._pixel_to_world_coord(0, y)
            label = f"{world_coord[1]:.1f}"
            ax.text(5, y, label, 
                   ha='left', va='center', fontsize=8,
                   color=self.grid_color, weight='bold')
        
        # Add axis labels
        ax.set_xlabel('World X Coordinate', fontsize=10, color=self.grid_color, weight='bold')
        ax.set_ylabel('World Z Coordinate', fontsize=10, color=self.grid_color, weight='bold')
    
    def _pixel_to_world_coord(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """
        Convert pixel coordinates to world coordinates for labeling.
        
        Args:
            pixel_x: X coordinate in pixels
            pixel_y: Y coordinate in pixels
            
        Returns:
            Tuple[float, float]: World coordinates (x, z)
        """
        if not self.map_info:
            return (pixel_x, pixel_y)
        
        bounds = self.map_info['world_bounds']
        map_size = self.map_info['map_size']
        
        # Normalize to [0, 1]
        norm_x = pixel_x / map_size[1]
        norm_y = pixel_y / map_size[0]
        
        # Convert to world coordinates
        world_x = bounds[0][0] + norm_x * (bounds[1][0] - bounds[0][0])
        world_z = bounds[0][2] + (1.0 - norm_y) * (bounds[1][2] - bounds[0][2])  # Invert Y
        
        return world_x, world_z
    
    def draw_agent_marker(self, ax, agent_pos_pixels: Tuple[float, float], 
                         agent_yaw_radians: float) -> None:
        """
        Draw agent position and orientation marker on the map.
        
        Args:
            ax: Matplotlib axes object
            agent_pos_pixels: Agent position in pixel coordinates (x, y)
            agent_yaw_radians: Agent yaw angle in radians
        """
        x, y = agent_pos_pixels
        
        # Draw agent circle
        circle = plt.Circle((x, y), self.agent_radius, 
                          color=self.agent_color, alpha=0.7, zorder=10)
        ax.add_patch(circle)
        
        # Draw direction arrow
        arrow_end_x = x + self.agent_arrow_length * np.sin(agent_yaw_radians)
        arrow_end_y = y - self.agent_arrow_length * np.cos(agent_yaw_radians)  # Negative because Y is inverted
        
        ax.arrow(x, y, arrow_end_x - x, arrow_end_y - y,
                head_width=self.agent_radius*0.8, head_length=self.agent_radius*0.8,
                fc=self.agent_color, ec=self.agent_color, 
                linewidth=self.agent_arrow_width, zorder=11)
        
        # Add agent label
        ax.text(x, y - self.agent_radius - 15, 'AGENT', 
               ha='center', va='top', fontsize=9,
               color=self.agent_color, weight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    def world_to_pixel_coordinates(self, world_pos: np.ndarray) -> Tuple[float, float]:
        """
        Convert 3D world coordinates to 2D pixel coordinates on the map.
        
        Args:
            world_pos: 3D world position [x, y, z]
            
        Returns:
            Tuple[float, float]: Pixel coordinates (x, y)
        """
        if not self.map_info:
            return (0, 0)
        
        bounds = self.map_info['world_bounds']
        map_size = self.map_info['map_size']
        
        # Normalize to [0, 1] range
        norm_x = (world_pos[0] - bounds[0][0]) / (bounds[1][0] - bounds[0][0])
        norm_z = (world_pos[2] - bounds[0][2]) / (bounds[1][2] - bounds[0][2])
        
        # Convert to pixel coordinates
        pixel_x = norm_x * map_size[1]
        pixel_y = (1.0 - norm_z) * map_size[0]  # Invert Y-axis
        
        return pixel_x, pixel_y
    
    def generate_map_image(self, agent_state: Dict[str, Any], 
                          output_path: str, title: str = "Navigation Map") -> bool:
        """
        Generate a complete map visualization with grid, agent marker, and labels.
        
        Args:
            agent_state: Dictionary containing agent position and orientation
            output_path: Path to save the generated image
            title: Title for the map image
            
        Returns:
            bool: True if image generated successfully, False otherwise
        """
        try:
            # Create figure with appropriate size
            fig_width = 12
            fig_height = 12 * (self.map_data.shape[0] / self.map_data.shape[1])
            
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            
            # Display the map
            ax.imshow(self.map_data, origin='upper')
            
            # Draw coordinate grid
            self.create_coordinate_grid(fig, ax)
            
            # Convert agent world position to pixel coordinates
            agent_world_pos = agent_state['position']
            agent_pixel_pos = self.world_to_pixel_coordinates(agent_world_pos)
            
            # Draw agent marker
            agent_yaw = np.radians(agent_state['yaw_degrees'])
            self.draw_agent_marker(ax, agent_pixel_pos, agent_yaw)
            
            # Set title and remove axes ticks
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            ax.set_xlim(0, self.map_data.shape[1])
            ax.set_ylim(self.map_data.shape[0], 0)  # Invert Y-axis
            
            # Remove default ticks but keep the coordinate labels we added
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add metadata text
            metadata_text = (
                f"Agent Position: ({agent_world_pos[0]:.2f}, {agent_world_pos[2]:.2f})\n"
                f"Agent Yaw: {agent_state['yaw_degrees']:.1f}°\n"
                f"Step: {agent_state.get('step_count', 0)}"
            )
            
            ax.text(0.02, 0.98, metadata_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Map image saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error generating map image: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return False
    
    def generate_comparative_view(self, agent_state: Dict[str, Any], 
                                rgb_image: np.ndarray, depth_image: np.ndarray,
                                output_path: str, title: str = "Navigation View") -> bool:
        """
        Generate a composite image showing map, RGB, and depth views.
        
        Args:
            agent_state: Dictionary containing agent state information
            rgb_image: RGB sensor image
            depth_image: Depth sensor image
            output_path: Path to save the composite image
            title: Title for the composite image
            
        Returns:
            bool: True if image generated successfully, False otherwise
        """
        try:
            # Create figure with 3 subplots
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            
            # Map view (left panel)
            ax1.imshow(self.map_data, origin='upper')
            self.create_coordinate_grid(fig, ax1)
            
            agent_world_pos = agent_state['position']
            agent_pixel_pos = self.world_to_pixel_coordinates(agent_world_pos)
            agent_yaw = np.radians(agent_state['yaw_degrees'])
            self.draw_agent_marker(ax1, agent_pixel_pos, agent_yaw)
            
            ax1.set_title('Top-Down Map', fontsize=12, fontweight='bold')
            ax1.set_xlim(0, self.map_data.shape[1])
            ax1.set_ylim(self.map_data.shape[0], 0)
            ax1.set_xticks([])
            ax1.set_yticks([])
            
            # RGB view (middle panel)
            if rgb_image is not None:
                ax2.imshow(rgb_image)
                ax2.set_title('First-Person View (RGB)', fontsize=12, fontweight='bold')
            else:
                ax2.text(0.5, 0.5, 'RGB Image\nNot Available', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('First-Person View (RGB)', fontsize=12, fontweight='bold')
            ax2.set_xticks([])
            ax2.set_yticks([])
            
            # Depth view (right panel)
            if depth_image is not None:
                # Normalize depth for visualization
                depth_vis = depth_image.copy()
                depth_vis = np.clip(depth_vis, 0, 10)  # Clip to 10 meters
                depth_vis = (depth_vis / 10.0 * 255).astype(np.uint8)
                ax3.imshow(depth_vis, cmap='viridis')
                ax3.set_title('Depth View', fontsize=12, fontweight='bold')
            else:
                ax3.text(0.5, 0.5, 'Depth Image\nNot Available', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Depth View', fontsize=12, fontweight='bold')
            ax3.set_xticks([])
            ax3.set_yticks([])
            
            # Add overall title
            fig.suptitle(title, fontsize=16, fontweight='bold')
            
            # Add metadata
            metadata_text = (
                f"Position: ({agent_world_pos[0]:.2f}, {agent_world_pos[2]:.2f}) | "
                f"Yaw: {agent_state['yaw_degrees']:.1f}° | "
                f"Pitch: {agent_state.get('camera_pitch_degrees', 0):.1f}° | "
                f"Step: {agent_state.get('step_count', 0)}"
            )
            
            fig.text(0.5, 0.02, metadata_text, ha='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
            
            # Save the figure
            plt.tight_layout()
            plt.subplots_adjust(top=0.9, bottom=0.1)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Composite view saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error generating composite view: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return False
    
    def update_map_data(self, new_map_data: np.ndarray, new_map_info: Dict[str, Any]) -> None:
        """
        Update the map data and information.
        
        Args:
            new_map_data: New map image data
            new_map_info: New map metadata
        """
        self.map_data = new_map_data
        self.map_info = new_map_info
        print(f"Map data updated, new size: {new_map_data.shape}")


def create_third_person_view(agent_pos: np.ndarray, rgb_image: np.ndarray, 
                           scene_bounds: np.ndarray) -> np.ndarray:
    """
    Create a simulated third-person view of the agent.
    
    This is a placeholder implementation that creates a simple third-person
    visualization. In a full implementation, this would render a proper
    third-person camera view from the simulation.
    
    Args:
        agent_pos: Agent 3D position
        rgb_image: First-person RGB image
        scene_bounds: Scene boundary information
        
    Returns:
        np.ndarray: Third-person view image
    """
    # For now, return a processed version of the RGB image with overlays
    if rgb_image is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create a copy of the RGB image
    tpv_image = rgb_image.copy()
    
    # Add simple overlay text to indicate this is a "third-person" view
    cv2.putText(tpv_image, "Third-Person View (Simulated)", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(tpv_image, f"Agent Pos: ({agent_pos[0]:.1f}, {agent_pos[2]:.1f})", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return tpv_image
