"""
Step 5: Navigation graph generation using Poisson Disk Sampling (PDS) algorithm.
Creates a graph of navigation nodes distributed across the walkable area.
Adapted from the object-goal navigation version.
"""

import os
import cv2
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy.ndimage import label, binary_dilation, binary_erosion

def apply_wall_padding(wall_mask: np.ndarray, padding_meters: float, 
                      spacing_in_meters_per_pixel: float) -> np.ndarray:
    """
    Apply padding to wall mask to ensure navigation points are at a minimum distance from walls.
    
    Args:
        wall_mask: Original wall mask (255 for walkable, 0 for walls)
        padding_meters: Minimum distance from walls in meters
        spacing_in_meters_per_pixel: Meters per pixel
        
    Returns:
        Processed wall mask with reduced walkable area
    """
    print(f"🛡️ Applying wall padding:")
    print(f"  - Padding distance: {padding_meters}m")
    print(f"  - Pixel spacing: {spacing_in_meters_per_pixel:.6f} m/pixel")
    
    # Calculate padding in pixels
    padding_pixels = int(np.ceil(padding_meters / spacing_in_meters_per_pixel))
    print(f"  - Padding pixels: {padding_pixels} pixels")
    
    if padding_pixels == 0:
        return wall_mask
    
    # Convert to binary mask (True for walkable, False for walls)
    binary_walkable = wall_mask > 127
    
    # Apply erosion to shrink walkable area
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * padding_pixels + 1, 2 * padding_pixels + 1))
    eroded_walkable = binary_erosion(binary_walkable, structure=kernel)
    
    # Convert back to original format
    padded_mask = np.where(eroded_walkable, 255, 0).astype(np.uint8)
    
    # Calculate statistics
    original_pixels = np.sum(binary_walkable)
    final_pixels = np.sum(eroded_walkable)
    reduction = (1 - final_pixels / original_pixels) * 100 if original_pixels > 0 else 0
    
    print(f"  - Walkable area reduced by: {reduction:.1f}%")
    
    return padded_mask

class PoissonDiskSampling:
    """Poisson Disk Sampling implementation for generating well-distributed points."""
    
    def __init__(self, width: int, height: int, radius: float, max_attempts: int = 30):
        self.width = width
        self.height = height
        self.radius = radius
        self.max_attempts = max_attempts
        
        # Grid for fast neighbor lookup
        self.cell_size = radius / np.sqrt(2)
        self.grid_width = int(np.ceil(width / self.cell_size))
        self.grid_height = int(np.ceil(height / self.cell_size))
        self.grid = [[-1 for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        
        self.points = []
        self.active_list = []
    
    def _get_grid_coords(self, x: float, y: float) -> Tuple[int, int]:
        grid_x = int(x / self.cell_size)
        grid_y = int(y / self.cell_size)
        return grid_x, grid_y
    
    def _is_valid_point(self, x: float, y: float, walkable_mask: np.ndarray) -> bool:
        # Check bounds
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        
        # Check if point is in walkable area
        if walkable_mask[int(y), int(x)] == 0:
            return False
        
        # Check distance to existing points
        grid_x, grid_y = self._get_grid_coords(x, y)
        
        # Check surrounding cells
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                gx, gy = grid_x + dx, grid_y + dy
                if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                    point_idx = self.grid[gy][gx]
                    if point_idx >= 0:
                        px, py = self.points[point_idx][:2]
                        dist = np.sqrt((x - px)**2 + (y - py)**2)
                        if dist < self.radius:
                            return False
        return True
    
    def sample(self, walkable_mask: np.ndarray, initial_point: Optional[Tuple[float, float]] = None) -> List[Tuple[float, float, int]]:
        """
        Sample points using Poisson Disk Sampling.
        
        Args:
            walkable_mask: Binary mask (255 for walkable, 0 for walls)
            initial_point: Optional starting point
            
        Returns:
            List of (x, y, point_id) tuples
        """
        # Find initial point if not provided
        if initial_point is None:
            # Find a random valid point
            attempts = 0
            while attempts < 1000:
                x = np.random.uniform(0, self.width)
                y = np.random.uniform(0, self.height)
                if self._is_valid_point(x, y, walkable_mask):
                    initial_point = (x, y)
                    break
                attempts += 1
            
            if initial_point is None:
                print("Could not find valid initial point")
                return []
        
        # Add initial point
        self.points = [(*initial_point, 0)]
        self.active_list = [0]
        
        grid_x, grid_y = self._get_grid_coords(initial_point[0], initial_point[1])
        self.grid[grid_y][grid_x] = 0
        
        point_id = 1
        
        # Main sampling loop
        while self.active_list:
            # Choose random active point
            active_idx = np.random.randint(len(self.active_list))
            current_idx = self.active_list[active_idx]
            current_x, current_y = self.points[current_idx][:2]
            
            found_valid_point = False
            
            # Try to generate new points around current point
            for _ in range(self.max_attempts):
                # Random point in annulus
                angle = np.random.uniform(0, 2 * np.pi)
                distance = np.random.uniform(self.radius, 2 * self.radius)
                
                new_x = current_x + distance * np.cos(angle)
                new_y = current_y + distance * np.sin(angle)
                
                if self._is_valid_point(new_x, new_y, walkable_mask):
                    # Add new point
                    self.points.append((new_x, new_y, point_id))
                    self.active_list.append(point_id)
                    
                    grid_x, grid_y = self._get_grid_coords(new_x, new_y)
                    self.grid[grid_y][grid_x] = point_id
                    
                    point_id += 1
                    found_valid_point = True
                    break
            
            # Remove current point from active list if no valid point found
            if not found_valid_point:
                self.active_list.pop(active_idx)
        
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
    
    # Get parameters
    spacing = metadata['topdown_metadata']['spacing_in_meters_per_pixel']
    graph_config = config['graph_generation']
    
    node_spacing_m = graph_config['node_spacing_m']
    wall_padding_m = graph_config['wall_padding_m']
    
    print(f"🔧 Graph generation parameters:")
    print(f"  - Node spacing: {node_spacing_m} m")
    print(f"  - Wall padding: {wall_padding_m} m")
    print(f"  - Pixel spacing: {spacing:.6f} m/pixel")
    
    # Convert node spacing to pixels
    node_radius_pixels = node_spacing_m / (2 * spacing)  # Radius for PDS
    
    print(f"  - Node radius in pixels: {node_radius_pixels:.2f}")
    
    # Apply wall padding
    padded_mask = apply_wall_padding(wall_mask, wall_padding_m, spacing)
    
    # Save padded mask for debugging
    padded_mask_path = os.path.join(output_dir, "padded_wall_mask.png")
    cv2.imwrite(padded_mask_path, padded_mask)
    
    # Generate navigation points using PDS
    height, width = wall_mask.shape
    pds = PoissonDiskSampling(width, height, node_radius_pixels)
    points = pds.sample(padded_mask)
    
    if len(points) == 0:
        raise RuntimeError("No navigation points generated")
    
    # Convert points to world coordinates
    corner_coords = metadata['unprojected_coords']
    
    # Extract coordinates from the 2D arrays [x, z]
    tl_x, tl_z = corner_coords['top_left'][0], corner_coords['top_left'][1]
    br_x, br_z = corner_coords['bottom_right'][0], corner_coords['bottom_right'][1]
    
    world_width = br_x - tl_x
    world_height = br_z - tl_z
    
    print(f"🌍 World coordinate conversion:")
    print(f"  - Top-left: ({tl_x:.3f}, {tl_z:.3f})")
    print(f"  - Bottom-right: ({br_x:.3f}, {br_z:.3f})")
    print(f"  - World dimensions: {world_width:.3f}m x {world_height:.3f}m")
    
    nodes_data = []
    for x, y, point_id in points:
        # Convert to world coordinates
        world_x = tl_x + (x / width) * world_width
        world_z = tl_z + (y / height) * world_height
        
        node_data = {
            "node_id": int(point_id),
            "pixel_coordinates": {"x": float(x), "y": float(y)},
            "world_coordinates": [float(world_x), float(world_z)]
        }
        nodes_data.append(node_data)
    
    # Create visualization
    graph_image = topdown_image.copy()
    
    # Draw nodes
    for node in nodes_data:
        x, y = int(node["pixel_coordinates"]["x"]), int(node["pixel_coordinates"]["y"])
        node_id = node["node_id"]
        
        # Draw node circle
        cv2.circle(graph_image, (x, y), 8, (0, 0, 255), -1)  # Red filled circle
        
        # Draw node ID
        cv2.putText(graph_image, str(node_id), (x + 10, y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Save graph visualization
    graph_path = os.path.join(output_dir, "graph_with_topdown.png")
    cv2.imwrite(graph_path, graph_image)
    print(f"✓ Graph visualization saved to: {graph_path}")
    
    # Save nodes data
    graph_data = {
        "total_nodes": len(nodes_data),
        "nodes": nodes_data,
        "generation_parameters": {
            "node_spacing_m": node_spacing_m,
            "wall_padding_m": wall_padding_m,
            "node_radius_pixels": node_radius_pixels,
            "spacing_in_meters_per_pixel": spacing
        }
    }
    
    nodes_path = os.path.join(output_dir, "navigation_nodes.json")
    with open(nodes_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Navigation nodes data saved to: {nodes_path}")
    
    print(f"📊 Graph generation completed:")
    print(f"  - Total nodes: {len(nodes_data)}")
    
    return {
        "generated_files": {
            "graph_image": graph_path,
            "nodes_data": nodes_path,
            "padded_mask": padded_mask_path
        },
        "results": {
            "total_nodes": len(nodes_data),
            "node_spacing_m": node_spacing_m,
            "wall_padding_m": wall_padding_m,
            "generation_success": True
        }
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 5:
        print("Usage: python step_5_graph_generation.py <topdown_path> <wall_mask_path> <metadata_path> <config_json>")
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
    print("Step 5 completed:", result)
