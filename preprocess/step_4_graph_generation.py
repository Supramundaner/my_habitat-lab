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
from scipy.ndimage import label, binary_dilation, binary_erosion
from sklearn.cluster import DBSCAN
from typing import Dict, Any, List, Tuple, Optional


def apply_wall_padding(wall_mask: np.ndarray, padding_meters: float, 
                      spacing_in_meters_per_pixel: float) -> np.ndarray:
    """
    对wall_mask应用padding，使生成的导航点距离墙壁至少有指定距离。
    
    Args:
        wall_mask: 原始墙壁mask (255为可行走区域，0为墙壁)
        padding_meters: 距离墙壁的最小距离（米）
        spacing_in_meters_per_pixel: 每像素对应的米数
        
    Returns:
        处理后的wall_mask，可行走区域被缩小了padding距离
    """
    print(f"🛡️ 应用墙壁padding:")
    print(f"  - Padding距离: {padding_meters}m")
    print(f"  - 像素间距: {spacing_in_meters_per_pixel:.6f} m/pixel")
    
    # 计算padding对应的像素数
    padding_pixels = int(np.ceil(padding_meters / spacing_in_meters_per_pixel))
    print(f"  - Padding像素数: {padding_pixels} pixels")
    
    if padding_pixels == 0:
        print("  - Padding像素数为0，跳过处理")
        return wall_mask
    
    # 转换为二值mask (True为可行走，False为墙壁)
    binary_walkable = wall_mask > 127
    
    # 统计原始可行走区域
    original_walkable_pixels = np.sum(binary_walkable)
    
    # 使用binary_erosion来缩小可行走区域
    # 创建圆形结构元素以获得更自然的padding效果
    from scipy.ndimage import generate_binary_structure
    
    if padding_pixels <= 3:
        # 对于小的padding，使用简单的结构元素
        structure = generate_binary_structure(2, 1)  # 4-连通
        iterations = padding_pixels
    else:
        # 对于大的padding，使用圆形结构元素
        y, x = np.ogrid[-padding_pixels:padding_pixels+1, -padding_pixels:padding_pixels+1]
        structure = x*x + y*y <= padding_pixels*padding_pixels
        iterations = 1
    
    # 应用腐蚀操作来缩小可行走区域
    eroded_walkable = binary_erosion(binary_walkable, structure=structure, iterations=iterations)
    
    # 转换回原始格式 (255为可行走，0为墙壁)
    padded_mask = np.where(eroded_walkable, 255, 0).astype(np.uint8)
    
    # 统计处理后的可行走区域
    final_walkable_pixels = np.sum(eroded_walkable)
    reduction_percentage = (1 - final_walkable_pixels / original_walkable_pixels) * 100
    
    print(f"  - 原始可行走像素: {original_walkable_pixels}")
    print(f"  - 处理后可行走像素: {final_walkable_pixels}")
    print(f"  - 可行走区域减少: {reduction_percentage:.1f}%")
    
    return padded_mask


class ConnectedComponentsAnalyzer:
    """Analyze connected components in walkable areas."""
    
    def __init__(self, walkable_mask: np.ndarray, min_component_area: int = 100):
        """
        Initialize component analyzer.
        
        Args:
            walkable_mask: Binary mask (255 for walkable, 0 for walls)
            min_component_area: Minimum area (pixels) for a component to be considered
        """
        self.walkable_mask = walkable_mask
        self.min_component_area = min_component_area
        self.height, self.width = walkable_mask.shape
        
    def find_connected_components(self) -> List[Dict[str, Any]]:
        """
        Find all connected components in the walkable area.
        
        Returns:
            List of component dictionaries with area, bbox, and mask
        """
        print(f"🔍 Analyzing connected components...")
        
        # Convert to binary mask (0 or 1)
        binary_mask = (self.walkable_mask > 127).astype(np.uint8)
        
        # Find connected components
        labeled_array, num_features = label(binary_mask)
        
        components = []
        for component_id in range(1, num_features + 1):
            # Get component mask
            component_mask = (labeled_array == component_id)
            area = np.sum(component_mask)
            
            # Skip small components
            if area < self.min_component_area:
                continue
                
            # Get bounding box
            coords = np.where(component_mask)
            min_y, max_y = coords[0].min(), coords[0].max()
            min_x, max_x = coords[1].min(), coords[1].max()
            
            # Calculate centroid
            centroid_y = coords[0].mean()
            centroid_x = coords[1].mean()
            
            component_info = {
                "id": len(components),
                "original_id": component_id,
                "area": area,
                "bbox": (min_x, min_y, max_x, max_y),
                "centroid": (centroid_x, centroid_y),
                "mask": component_mask
            }
            components.append(component_info)
        
        print(f"✓ Found {len(components)} connected components")
        for i, comp in enumerate(components):
            print(f"  Component {i}: area={comp['area']} pixels, bbox={comp['bbox']}")
        
        return components


class MultiRegionPoissonDiskSampling:
    """Enhanced PDS that handles multiple disconnected regions."""
    
    def __init__(self, width: int, height: int, radius: float, max_attempts: int = 30):
        """
        Initialize multi-region PDS sampler.
        
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
        
        self.all_points = []
        self.global_point_counter = 0


class PoissonDiskSampling:
    """Single-region Poisson Disk Sampling implementation."""
    
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
    
    def sample_region(self, walkable_mask: np.ndarray, region_mask: np.ndarray, 
                     initial_point: Optional[Tuple[float, float]] = None, 
                     point_id_offset: int = 0) -> List[Tuple[float, float, int]]:
        """
        Generate PDS points within a specific region.
        
        Args:
            walkable_mask: Full walkable area mask
            region_mask: Mask for this specific region
            initial_point: Optional initial point
            point_id_offset: Offset for point IDs to ensure global uniqueness
            
        Returns:
            List of (x, y, point_id) tuples
        """
        # Reset internal state
        self.points = []
        self.active_list = []
        self.grid = [[None for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        
        print(f"🎯 Sampling region (offset={point_id_offset}):")
        region_area = np.sum(region_mask)
        print(f"  - Region area: {region_area} pixels")
        print(f"  - Sampling radius: {self.radius:.1f} pixels")
        
        # Find initial point in this region
        if initial_point:
            x, y = initial_point
            if self._is_valid_point_in_region(x, y, walkable_mask, region_mask):
                self._add_point(x, y, point_id_offset)
            else:
                initial_point = None
        
        if not initial_point:
            # Find random valid starting point within the region
            region_coords = np.where(region_mask)
            if len(region_coords[0]) == 0:
                print("⚠️ No valid pixels in region mask")
                return []
            
            attempts = 0
            while attempts < 1000:
                # Choose random point from region
                idx = random.randint(0, len(region_coords[0]) - 1)
                y = region_coords[0][idx]
                x = region_coords[1][idx]
                
                if self._is_valid_point_in_region(x, y, walkable_mask, region_mask):
                    self._add_point(x, y, point_id_offset)
                    break
                attempts += 1
            
            if not self.points:
                print("⚠️ Could not find valid starting point in region")
                return []
        
        # Generate points using PDS algorithm within this region
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
                
                if self._is_valid_point_in_region(new_x, new_y, walkable_mask, region_mask):
                    self._add_point(new_x, new_y, point_id_offset)
                    found_valid_point = True
                    break
            
            # If no valid point found, remove from active list
            if not found_valid_point:
                self.active_list.pop(active_index)
        
        print(f"✓ Generated {len(self.points)} points in region")
        return self.points
    
    def _is_valid_point_in_region(self, x: float, y: float, walkable_mask: np.ndarray, 
                                 region_mask: np.ndarray) -> bool:
        """Check if a point is valid within a specific region."""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        
        int_x, int_y = int(x), int(y)
        
        # Check if point is within the specific region
        if not region_mask[int_y, int_x]:
            return False
        
        # Check if point is on walkable area
        if walkable_mask[int_y, int_x] == 0:
            return False
        
        # Check minimum distance to existing points in this region
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
    
    def _add_point(self, x: float, y: float, point_id_offset: int) -> int:
        """Add a valid point to the sampling with ID offset."""
        point_id = len(self.points) + point_id_offset
        point = (x, y, point_id)
        
        self.points.append(point)
        self.active_list.append(point)
        
        # Add to grid
        grid_x, grid_y = self._get_grid_coords(x, y)
        self.grid[grid_y][grid_x] = point
        
        return point_id

def sample_multiple_regions(walkable_mask: np.ndarray, radius: float, 
                          min_component_area: int = 100, max_attempts: int = 30) -> List[Tuple[float, float, int]]:
    """
    Sample points across multiple disconnected regions.
    
    Args:
        walkable_mask: Binary mask (255 for walkable, 0 for walls)
        radius: PDS radius in pixels
        min_component_area: Minimum area for a region to be sampled
        max_attempts: Max attempts per point
        
    Returns:
        List of (x, y, point_id) tuples from all regions
    """
    height, width = walkable_mask.shape
    
    # Step 1: Find connected components
    analyzer = ConnectedComponentsAnalyzer(walkable_mask, min_component_area)
    components = analyzer.find_connected_components()
    
    if not components:
        raise RuntimeError("No valid connected components found")
    
    # Step 2: Sample each region independently
    all_points = []
    point_id_offset = 0
    
    for comp_idx, component in enumerate(components):
        print(f"\n🎯 Processing component {comp_idx + 1}/{len(components)}")
        
        # Create PDS sampler for this region
        sampler = PoissonDiskSampling(
            width=width,
            height=height, 
            radius=radius,
            max_attempts=max_attempts
        )
        
        # Sample points in this region
        region_points = sampler.sample_region(
            walkable_mask=walkable_mask,
            region_mask=component['mask'],
            point_id_offset=point_id_offset
        )
        
        all_points.extend(region_points)
        point_id_offset += len(region_points)
        
        print(f"✓ Component {comp_idx}: {len(region_points)} points added")
    
    print(f"\n🎉 Multi-region sampling completed:")
    print(f"  - Total components: {len(components)}")
    print(f"  - Total points: {len(all_points)}")
    
    return all_points


def generate_navigation_graph(topdown_path: str, wall_mask_path: str, metadata_path: str,
                            config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Generate navigation graph using multi-region Poisson Disk Sampling.
    
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
    min_component_area = graph_config.get('min_component_area', 100)
    enable_multi_region = graph_config.get('enable_multi_region', True)
    wall_padding_meters = graph_config.get('wall_padding_meters', 0.1)
    
    # Convert PDS radius from meters to pixels
    pds_radius_pixels = pds_radius_meters / spacing_in_meters_per_pixel
    
    print(f"🔧 Graph generation parameters:")
    print(f"  - PDS radius: {pds_radius_meters}m ({pds_radius_pixels:.1f} pixels)")
    print(f"  - Max attempts: {max_attempts}")
    print(f"  - Node visualization radius: {node_radius_pixels} pixels")
    print(f"  - Min component area: {min_component_area} pixels")
    print(f"  - Multi-region sampling: {enable_multi_region}")
    print(f"  - Wall padding: {wall_padding_meters}m")
    
    # Apply wall padding to ensure minimum distance from walls
    if wall_padding_meters > 0:
        wall_mask = apply_wall_padding(wall_mask, wall_padding_meters, spacing_in_meters_per_pixel)
        print(f"✓ Wall padding applied")
    
    # Generate points using appropriate method
    height, width = wall_mask.shape
    
    if enable_multi_region:
        # Use multi-region sampling
        points = sample_multiple_regions(
            walkable_mask=wall_mask,
            radius=pds_radius_pixels,
            min_component_area=min_component_area,
            max_attempts=30
        )
    else:
        # Use traditional single-region PDS
        sampler = PoissonDiskSampling(
            width=width,
            height=height,
            radius=pds_radius_pixels,
            max_attempts=30
        )
        points = sampler.sample_region(wall_mask, wall_mask > 127)
    
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
            "min_component_area": min_component_area,
            "enable_multi_region": enable_multi_region,
            "wall_padding_meters": wall_padding_meters,
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
