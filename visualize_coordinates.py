#!/usr/bin/env python3
"""
Visualize 2D coordinates (X, Z) on a floor topdown view.

This script renders a topdown view of a specified floor in a 3D scene
and marks the given 2D coordinates on the image.

Usage examples:
    # Single coordinate
    python visualize_coordinates.py --scene data/versioned_data/hm3d-0.2/hm3d/val/00890-6s7QHgap2fW/6s7QHgap2fW.basis.glb --floor 0 --coordinates "1.5,2.3"
    
    # Multiple coordinates
    python visualize_coordinates.py --scene data/versioned_data/hm3d-0.2/hm3d/val/00890-6s7QHgap2fW/6s7QHgap2fW.basis.glb --floor 0 --coordinates "1.5,2.3" "4.2,-1.8" "-0.5,3.1"
    
    # Using config file
    python visualize_coordinates.py --config coordinate_config.json
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Import the topdown renderer from preprocess/current_topdown.py
try:
    from preprocess.current_topdown import render_topdown_view
except Exception as e:
    # Fallback: add repo root to sys.path explicitly and retry
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    from preprocess.current_topdown import render_topdown_view


def parse_coordinates(coord_strings: List[str]) -> List[Tuple[float, float]]:
    """
    Parse coordinate strings into (x, z) tuples.
    
    Args:
        coord_strings: List of coordinate strings in format "x,z"
        
    Returns:
        List of (x, z) coordinate tuples
    """
    coordinates = []
    for coord_str in coord_strings:
        try:
            parts = coord_str.strip().split(',')
            if len(parts) != 2:
                print(f"Warning: Invalid coordinate format '{coord_str}', expected 'x,z'")
                continue
            x, z = float(parts[0].strip()), float(parts[1].strip())
            coordinates.append((x, z))
        except ValueError as e:
            print(f"Warning: Could not parse coordinate '{coord_str}': {e}")
            continue
    return coordinates


def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def world_to_pixel(x: float, z: float, coords: Dict) -> Tuple[int, int]:
    """
    Project world (x, z) coordinates to image pixel coordinates using
    unprojected corner coords produced by render_topdown_view.
    
    Args:
        x, z: World coordinates
        coords: Unprojected coordinate info from render_topdown_view
        
    Returns:
        (pixel_x, pixel_y) tuple
    """
    tl_x, tl_z = coords['top_left']
    tr_x, _ = coords['top_right']
    _, bl_z = coords['bottom_left']
    img_w, img_h = coords['image_size']

    fx = (x - tl_x) / (tr_x - tl_x)
    fz = (z - tl_z) / (bl_z - tl_z)
    px = int(round(fx * img_w))
    py = int(round(fz * img_h))
    return px, py


def draw_coordinates(
    base_image: np.ndarray,
    coords: Dict,
    target_coords: List[Tuple[float, float]],
    show_indices: bool = True,
    point_radius: int = 8,
    colors: Optional[List[Tuple[int, int, int]]] = None
) -> Image.Image:
    """
    Draw coordinate points on the topdown image.
    
    Args:
        base_image: Base topdown image (numpy array)
        coords: Unprojected coordinate info from render_topdown_view
        target_coords: List of (x, z) coordinates to mark
        show_indices: Whether to show coordinate indices
        point_radius: Radius of coordinate markers in pixels
        colors: Optional list of RGB colors for each coordinate
        
    Returns:
        PIL Image with coordinates marked
    """
    # Convert to PIL RGB image
    if base_image.shape[2] == 4:
        pil_img = Image.fromarray(base_image[..., :3], "RGB")
    else:
        pil_img = Image.fromarray(base_image, "RGB")
    
    draw = ImageDraw.Draw(pil_img)

    # Load fonts
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 12)
    except IOError:
        font = font_small = ImageFont.load_default()

    # Default colors if not provided
    if colors is None:
        default_colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 165, 0),  # Orange
            (128, 0, 128),  # Purple
            (255, 192, 203),# Pink
            (0, 128, 0),    # Dark Green
        ]
        colors = [default_colors[i % len(default_colors)] for i in range(len(target_coords))]

    # Draw coordinate points
    visible_count = 0
    for idx, ((x, z), color) in enumerate(zip(target_coords, colors)):
        px, py = world_to_pixel(x, z, coords)
        
        # Check if point is within image bounds
        if 0 <= px < pil_img.width and 0 <= py < pil_img.height:
            # Draw filled circle with black border
            draw.ellipse(
                [px - point_radius, py - point_radius, px + point_radius, py + point_radius],
                fill=color,
                outline=(0, 0, 0),
                width=2
            )
            
            # Draw coordinate label
            label = f"({x:.2f}, {z:.2f})"
            if show_indices:
                label = f"{idx}: {label}"
            
            # Position label to avoid overlapping with marker
            label_x = px + point_radius + 4
            label_y = py - point_radius - 2
            
            # Draw label background for better readability
            bbox = draw.textbbox((label_x, label_y), label, font=font_small)
            draw.rectangle(
                [bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1],
                fill=(0, 0, 0, 128)
            )
            draw.text((label_x, label_y), label, fill=(255, 255, 255), font=font_small)
            
            visible_count += 1
        else:
            print(f"Warning: Coordinate {idx} ({x:.2f}, {z:.2f}) is outside image bounds")

    # Draw legend/info box
    info_lines = [
        f"Coordinates: {len(target_coords)} total, {visible_count} visible",
        f"Image bounds: X[{coords['top_left'][0]:.2f}, {coords['top_right'][0]:.2f}]",
        f"              Z[{coords['top_left'][1]:.2f}, {coords['bottom_left'][1]:.2f}]"
    ]
    
    # Calculate legend box size
    max_width = 0
    total_height = 0
    for line in info_lines:
        bbox = draw.textbbox((0, 0), line, font=font_small)
        max_width = max(max_width, bbox[2] - bbox[0])
        total_height += bbox[3] - bbox[1] + 2
    
    # Draw legend background
    legend_x, legend_y = 10, 10
    draw.rectangle(
        [legend_x, legend_y, legend_x + max_width + 16, legend_y + total_height + 8],
        fill=(0, 0, 0, 160),
        outline=(255, 255, 255),
        width=1
    )
    
    # Draw legend text
    y_offset = legend_y + 4
    for line in info_lines:
        draw.text((legend_x + 8, y_offset), line, fill=(255, 255, 255), font=font_small)
        bbox = draw.textbbox((0, 0), line, font=font_small)
        y_offset += bbox[3] - bbox[1] + 2

    return pil_img


def infer_scene_id(scene_path: str) -> str:
    """Infer scene ID from scene file path."""
    # Prefer directory name if structured as .../<scene_id>/<scene_file>
    dirname = os.path.basename(os.path.dirname(scene_path))
    if dirname and dirname != os.path.basename(scene_path):
        return dirname
    # Else use file stem
    return os.path.splitext(os.path.basename(scene_path))[0]


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 2D coordinates on a floor topdown view.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single coordinate
    python visualize_coordinates.py --scene scene.glb --floor 0 --coordinates "1.5,2.3"
    
    # Multiple coordinates  
    python visualize_coordinates.py --scene scene.glb --floor 0 --coordinates "1.5,2.3" "4.2,-1.8" "-0.5,3.1"
    
    # Using Y coordinate to auto-select floor
    python visualize_coordinates.py --scene scene.glb --floor_y -1.0 --coordinates "1.5,2.3"
    
    # Using config file
    python visualize_coordinates.py --config coordinate_config.json
        """
    )
    
    # Config file option
    parser.add_argument("--config", help="Path to JSON config file")
    
    # Direct parameters
    parser.add_argument("--scene", default =  "/home/yaoaa/habitat-lab/data/versioned_data/hm3d-0.2/hm3d/val/00853-5cdEh9F2hJL/5cdEh9F2hJL.basis.glb",help="Path to the scene .glb file")
    parser.add_argument("--floor", type=int, default=0,help="Floor index (0-based)")
    parser.add_argument("--floor_y", type=float, help="Y coordinate to auto-select floor")
    parser.add_argument("--coordinates", nargs='+', help="Coordinates in 'x,z' format")
    parser.add_argument("--output", help="Output image path (default: coordinates_visualization/<scene_id>_coordinates.png)")
    
    # Rendering options
    parser.add_argument("--resolution", type=int, default=2048, help="Render resolution (default: 2048)")
    parser.add_argument("--custom_ortho_scale", type=float, help="Custom orthographic scale")
    parser.add_argument("--target_coverage", type=float, default=0.9, help="Coverage ratio for auto ortho scale")
    parser.add_argument("--draw_coordinates_grid", action="store_true", help="Draw coordinate grid on image")
    
    # Visualization options
    parser.add_argument("--show_indices", action="store_true", default=True, help="Show coordinate indices")
    parser.add_argument("--marker_radius", type=int, default=8, help="Radius of coordinate markers")
    
    args = parser.parse_args()
    
    # Load config if provided
    config = {}
    if args.config:
        if not os.path.exists(args.config):
            print(f"Config file not found: {args.config}")
            sys.exit(1)
        config = load_config(args.config)
    
    # Extract parameters (command line args override config)
    scene_path = args.scene or config.get("scene_config", {}).get("scene_path")
    floor_index = args.floor if args.floor is not None else config.get("scene_config", {}).get("target_floor")
    floor_y = args.floor_y if args.floor_y is not None else config.get("scene_config", {}).get("target_coordinate")
    coordinates_str = args.coordinates or config.get("coordinates", [])
    resolution = args.resolution or config.get("resolution", 2048)
    
    if not scene_path:
        print("Error: Scene path must be provided via --scene or config file")
        sys.exit(1)
    
    if not os.path.exists(scene_path):
        print(f"Error: Scene file not found: {scene_path}")
        sys.exit(1)
    
    if floor_index is None and floor_y is None:
        print("Error: Either --floor or --floor_y must be specified")
        sys.exit(1)
        
    if not coordinates_str:
        print("Error: No coordinates provided")
        sys.exit(1)
    
    # Parse coordinates
    target_coords = parse_coordinates(coordinates_str)
    if not target_coords:
        print("Error: No valid coordinates found")
        sys.exit(1)
    
    print(f"Parsed {len(target_coords)} coordinates: {target_coords}")
    
    # Determine target floor
    if floor_y is not None:
        target_floor = [0.0, floor_y, 0.0]  # Use Y coordinate to select floor
        print(f"Using Y coordinate {floor_y} to auto-select floor")
    else:
        target_floor = floor_index
        print(f"Using floor index {floor_index}")
    
    # Render topdown view
    print("Rendering topdown view...")
    try:
        img, coords, meta = render_topdown_view(
            scene_path,
            target_floor=target_floor,
            custom_ortho_scale=args.custom_ortho_scale,
            target_coverage=args.target_coverage,
            draw_coordinates=args.draw_coordinates_grid,
            resolution=[resolution, resolution]
        )
    except Exception as e:
        print(f"Error rendering topdown view: {e}")
        sys.exit(1)
    
    if img is None or coords is None:
        print("Failed to render topdown view")
        sys.exit(1)
    
    print("Rendering completed successfully")
    
    # Draw coordinates on the image
    print(f"Drawing {len(target_coords)} coordinates on the image...")
    annotated_img = draw_coordinates(
        img,
        coords,
        target_coords,
        show_indices=args.show_indices,
        point_radius=args.marker_radius
    )
    
    # Save result
    scene_id = infer_scene_id(scene_path)
    out_dir = os.path.join(os.getcwd(), "coordinates_visualization")
    os.makedirs(out_dir, exist_ok=True)
    
    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join(out_dir, f"{scene_id}_coordinates.png")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    
    annotated_img.save(out_path)
    print(f"Saved visualization: {out_path}")
    
    # Print summary
    print("\n=== Visualization Summary ===")
    print(f"Scene: {scene_path}")
    print(f"Floor: {target_floor}")
    print(f"Coordinates visualized: {len(target_coords)}")
    print(f"Image resolution: {resolution}x{resolution}")
    print(f"Output saved to: {out_path}")


if __name__ == "__main__":
    main()
