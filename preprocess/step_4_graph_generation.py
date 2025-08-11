"""
Step 4: Navigation graph generation using Poisson Disk Sampling (PDS) algorithm.
Creates a graph of navigation nodes distributed across the walkable area.
"""

import os
import cv2
import json
import numpy as np
import random
import math
from typing import Dict, Any, List, Tuple, Optional

class PoissonDiskSampling:
    """Poisson Disk Sampling implementation for generating evenly distributed points."""
    
    def __init__(self, width: int, height: int, radius: float, max_attempts: int = 30):
        """
        Initialize PDS sampler.
        
        Args:
            width: Image width
            height: Image height  
            radius: Minimum distance between points (in pixels)
            max_attempts: Maximum attempts to place a new point around existing points
        """
        self.width = width
        self.height = height
        self.radius = radius
        self.max_attempts = max_attempts
        
        # Grid for efficient neighbor lookup
        self.cell_size = radius / math.sqrt(2)
        self.grid_width = int(math.ceil(width / self.cell_size))
        self.grid_height = int(math.ceil(height / self.cell_size))
        self.grid = [[None for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        
        self.points = []
        self.active_list = []
    
    def _get_grid_coords(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        grid_x = int(x / self.cell_size)
        grid_y = int(y / self.cell_size)
        return grid_x, grid_y
    
    def _is_valid_point(self, x: float, y: float, walkable_mask: np.ndarray) -> bool:
        """Check if a point is valid (within bounds and on walkable area)."""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        
        # Check if point is on walkable area (white pixels in mask)
        if walkable_mask[int(y), int(x)] == 0:
            return False
        
        # Check minimum distance to existing points
        grid_x, grid_y = self._get_grid_coords(x, y)
        
        # Check neighboring grid cells
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                neighbor_x = grid_x + dx
                neighbor_y = grid_y + dy
                
                if (0 <= neighbor_x < self.grid_width and 
                    0 <= neighbor_y < self.grid_height and 
                    self.grid[neighbor_y][neighbor_x] is not None):
                    
                    existing_point = self.grid[neighbor_y][neighbor_x]
                    distance = math.sqrt((x - existing_point[0])**2 + (y - existing_point[1])**2)
                    
                    if distance < self.radius:
                        return False
        
        return True
    
    def _add_point(self, x: float, y: float) -> int:
        """Add a valid point to the sampling."""
        point_id = len(self.points)
        point = (x, y, point_id)
        
        self.points.append(point)
        self.active_list.append(point)
        
        # Add to grid
        grid_x, grid_y = self._get_grid_coords(x, y)
        self.grid[grid_y][grid_x] = point
        
        return point_id
    
    def sample(self, walkable_mask: np.ndarray, initial_point: Optional[Tuple[float, float]] = None) -> List[Tuple[float, float, int]]:
        """
        Generate Poisson disk sampling points.
        
        Args:
            walkable_mask: Binary mask (255 for walkable, 0 for walls)
            initial_point: Optional initial point to start sampling
            
        Returns:
            List of (x, y, point_id) tuples
        """
        print(f"🎯 Starting Poisson Disk Sampling:")
        print(f"  - Image size: {self.width}x{self.height}")
        print(f"  - Sampling radius: {self.radius:.1f} pixels")
        print(f"  - Max attempts per point: {self.max_attempts}")
        
        # Initialize with random point or provided initial point
        if initial_point:
            x, y = initial_point
            if self._is_valid_point(x, y, walkable_mask):
                self._add_point(x, y)
            else:
                print(f"⚠️ Initial point ({x}, {y}) is not valid, using random point")
                initial_point = None
        
        if not initial_point:
            # Find a random valid starting point
            attempts = 0
            while attempts < 1000:
                x = random.uniform(0, self.width - 1)
                y = random.uniform(0, self.height - 1)
                
                if self._is_valid_point(x, y, walkable_mask):
                    self._add_point(x, y)
                    break
                attempts += 1
            
            if not self.points:
                raise RuntimeError("Could not find valid starting point for PDS")
        
        # Generate points using PDS algorithm
        while self.active_list:
            # Choose random point from active list
            active_index = random.randint(0, len(self.active_list) - 1)
            active_point = self.active_list[active_index]
            
            found_valid_point = False
            
            # Try to generate new points around the active point
            for _ in range(self.max_attempts):
                # Generate random point in annulus between radius and 2*radius
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(self.radius, 2 * self.radius)
                
                new_x = active_point[0] + distance * math.cos(angle)
                new_y = active_point[1] + distance * math.sin(angle)
                
                if self._is_valid_point(new_x, new_y, walkable_mask):
                    self._add_point(new_x, new_y)
                    found_valid_point = True
                    break
            
            # If no valid point found, remove from active list
            if not found_valid_point:
                self.active_list.pop(active_index)
        
        print(f"✓ Generated {len(self.points)} points using PDS")
        return self.points

def generate_navigation_graph(topdown_path: str, wall_mask_path: str, metadata_path: str,
                            config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Generate navigation graph using Poisson Disk Sampling.
    
    Args:
        topdown_path: Path to topdown view image
        wall_mask_path: Path to wall mask image
        metadata_path: Path to metadata JSON file
        config: Configuration dictionary
        output_dir: Output directory path
        
    Returns:
        Dictionary with generated files and results
    """
    print(f"📁 Loading files for graph generation:")
    print(f"  - Topdown: {topdown_path}")
    print(f"  - Wall mask: {wall_mask_path}")
    print(f"  - Metadata: {metadata_path}")
    
    # Load images and metadata
    topdown_image = cv2.imread(topdown_path)
    wall_mask = cv2.imread(wall_mask_path, cv2.IMREAD_GRAYSCALE)
    
    if topdown_image is None:
        raise FileNotFoundError(f"Topdown image not found: {topdown_path}")
    if wall_mask is None:
        raise FileNotFoundError(f"Wall mask not found: {wall_mask_path}")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    spacing_in_meters_per_pixel = metadata['topdown_metadata']['spacing_in_meters_per_pixel']
    
    print(f"✓ Images loaded, size: {topdown_image.shape[1]}x{topdown_image.shape[0]}")
    print(f"📏 Pixel spacing: {spacing_in_meters_per_pixel:.6f} m/pixel")
    
    # Get graph generation parameters
    graph_config = config['graph_generation']
    pds_radius_meters = graph_config.get('pds_radius', 1.0)
    max_attempts = graph_config.get('max_attempts', 10000)
    node_radius_pixels = graph_config.get('node_radius_pixels', 8)
    
    # Convert PDS radius from meters to pixels
    pds_radius_pixels = pds_radius_meters / spacing_in_meters_per_pixel
    
    print(f"🔧 Graph generation parameters:")
    print(f"  - PDS radius: {pds_radius_meters}m ({pds_radius_pixels:.1f} pixels)")
    print(f"  - Max attempts: {max_attempts}")
    print(f"  - Node visualization radius: {node_radius_pixels} pixels")
    
    # Initialize PDS sampler
    height, width = wall_mask.shape
    sampler = PoissonDiskSampling(
        width=width,
        height=height,
        radius=pds_radius_pixels,
        max_attempts=30
    )
    
    # Generate points
    points = sampler.sample(wall_mask)
    
    if not points:
        raise RuntimeError("No valid points generated by PDS")
    
    print(f"✓ Generated {len(points)} navigation nodes")
    
    # Convert points to world coordinates and create node data
    nodes_data = []
    unprojected_coords = metadata.get('unprojected_coords', {})
    
    if unprojected_coords:
        tl_x, tl_z = unprojected_coords['top_left']
        tr_x, _ = unprojected_coords['top_right']
        _, bl_z = unprojected_coords['bottom_left']
        
        for x_pixel, y_pixel, node_id in points:
            # Convert pixel coordinates to world coordinates
            fx = x_pixel / width
            fz = y_pixel / height
            
            world_x = tl_x + fx * (tr_x - tl_x)
            world_z = tl_z + fz * (bl_z - tl_z)
            
            node_data = {
                "node_id": int(node_id),
                "pixel_coordinates": [float(x_pixel), float(y_pixel)],
                "world_coordinates": [float(world_x), float(world_z)]
            }
            nodes_data.append(node_data)
    else:
        print("⚠️ No unprojected coordinates found, using pixel coordinates only")
        for x_pixel, y_pixel, node_id in points:
            node_data = {
                "node_id": int(node_id),
                "pixel_coordinates": [float(x_pixel), float(y_pixel)],
                "world_coordinates": None
            }
            nodes_data.append(node_data)
    
    # Create visualization: graph overlaid on topdown view
    graph_image = topdown_image.copy()
    
    # Draw nodes
    for i, (x, y, node_id) in enumerate(points):
        # Draw node circle
        cv2.circle(graph_image, (int(x), int(y)), node_radius_pixels, (0, 255, 0), -1)  # Green filled circle
        cv2.circle(graph_image, (int(x), int(y)), node_radius_pixels, (0, 0, 0), 2)     # Black border
        
        # Draw node number
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text = str(node_id)
        
        # Calculate text size for centering
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = int(x - text_width // 2)
        text_y = int(y + text_height // 2)
        
        # Draw text with white color
        cv2.putText(graph_image, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
    
    # Save graph visualization
    graph_with_topdown_path = os.path.join(output_dir, "graph_with_topdown_view.png")
    cv2.imwrite(graph_with_topdown_path, graph_image)
    print(f"✓ Graph visualization saved to: {graph_with_topdown_path}")
    
    # Save nodes data
    nodes_json_path = os.path.join(output_dir, "navigation_nodes.json")
    nodes_json_data = {
        "parameters": {
            "pds_radius_meters": pds_radius_meters,
            "pds_radius_pixels": pds_radius_pixels,
            "spacing_in_meters_per_pixel": spacing_in_meters_per_pixel,
            "max_attempts": max_attempts,
            "total_nodes": len(points)
        },
        "nodes": nodes_data
    }
    
    with open(nodes_json_path, 'w', encoding='utf-8') as f:
        json.dump(nodes_json_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Navigation nodes data saved to: {nodes_json_path}")
    
    return {
        "generated_files": {
            "graph_with_topdown_view": graph_with_topdown_path,
            "navigation_nodes": nodes_json_path
        },
        "results": {
            "total_nodes": len(points),
            "pds_radius_meters": pds_radius_meters,
            "pds_radius_pixels": pds_radius_pixels,
            "nodes_data": nodes_data
        }
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 5:
        print("Usage: python step_4_graph_generation.py <topdown_path> <wall_mask_path> <metadata_path> <config_path>")
        sys.exit(1)
    
    topdown_path = sys.argv[1]
    wall_mask_path = sys.argv[2]
    metadata_path = sys.argv[3]
    config_path = sys.argv[4]
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    result = generate_navigation_graph(topdown_path, wall_mask_path, metadata_path, config, output_dir)
    print("Step 4 completed:", result)
