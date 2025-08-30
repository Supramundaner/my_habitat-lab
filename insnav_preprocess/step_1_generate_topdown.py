"""
Step 1: Generate topdown view and metadata using the current_topdown.py functionality.
This is adapted from the object-goal navigation version.
"""

import os
import cv2
import json
import numpy as np
from typing import Dict, Any, Tuple, Optional

# Import the render function from current_topdown.py
from current_topdown import render_topdown_view

def make_json_serializable(obj):
    """
    Convert numpy types and other non-JSON-serializable types to JSON-serializable types.
    """
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def generate_topdown_view(scene_path: str, target_floor, 
                         custom_ortho_scale: Optional[float] = None,
                         target_coverage: float = 0.9,
                         draw_coordinates: bool = False,
                         resolution: Tuple[int, int] = (2048, 2048),
                         output_dir: str = None) -> Dict[str, Any]:
    """
    Generate topdown view and metadata.
    
    Args:
        scene_path: Path to scene file
        target_floor: Floor level - can be either an integer (floor index) or a 3-element list/tuple (world coordinate [x,y,z])
        custom_ortho_scale: Custom orthographic scale
        target_coverage: Target coverage for auto-scaling
        draw_coordinates: Whether to draw coordinate system
        resolution: Image resolution
        output_dir: Output directory path
        
    Returns:
        Dictionary with generated files and results
    """
    print(f"📁 Scene path: {scene_path}")
    print(f"📏 Resolution: {resolution}")
    print(f"� Target floor/coordinate: {target_floor}")
    
    # Check if scene file exists
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene file not found: {scene_path}")
    
    # Generate topdown view using existing function
    result_image, unprojected_coords, meta_data = render_topdown_view(
        scene_path,
        target_floor=target_floor,
        custom_ortho_scale=custom_ortho_scale,
        target_coverage=target_coverage,
        draw_coordinates=draw_coordinates,
        resolution=list(resolution)
    )
    
    if result_image is None:
        raise RuntimeError("Failed to generate topdown view")
    
    # Save topdown view image
    if output_dir is None:
        topdown_path = "topdown_view.png"
        metadata_path = "metadata.json"
    else:
        topdown_path = os.path.join(output_dir, "topdown_view.png")
        metadata_path = os.path.join(output_dir, "metadata.json")
    
    # Debug: Print image info
    print(f"🔍 Result image shape: {result_image.shape}")
    print(f"🔍 Result image dtype: {result_image.dtype}")
    print(f"🔍 Result image min/max: {result_image.min()}/{result_image.max()}")
    
    if len(result_image.shape) == 3 and result_image.shape[2] == 4:
        # RGBA format - take RGB channels and convert for saving
        result_rgb = result_image[:, :, :3]  # Extract RGB channels
        # Check if it's already in BGR format or needs conversion
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    elif len(result_image.shape) == 3 and result_image.shape[2] == 3:
        # RGB format - convert to BGR for OpenCV saving
        result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    else:
        # Grayscale or other format
        result_bgr = result_image
    
    cv2.imwrite(topdown_path, result_bgr)
    print(f"✓ Topdown view saved to: {topdown_path}")
    
    # Save metadata
    combined_metadata = {
        "topdown_metadata": meta_data,
        "unprojected_coords": unprojected_coords,
        "scene_info": {
            "scene_path": scene_path,
            "target_floor_coordinate": target_floor
        }
    }
    
    # Convert all data to JSON-serializable format
    combined_metadata = make_json_serializable(combined_metadata)
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(combined_metadata, f, indent=2, ensure_ascii=False)
    print(f"✓ Metadata saved to: {metadata_path}")
    
    return {
        "generated_files": {
            "topdown_view": topdown_path,
            "metadata": metadata_path
        },
        "results": {
            "image_size": result_image.shape[:2] if result_image is not None else None,
            "spacing_in_meters_per_pixel": meta_data.get("spacing_in_meters_per_pixel") if meta_data else None
        }
    }

if __name__ == "__main__":
    # Test function
    import sys
    if len(sys.argv) < 3:
        print("Usage: python step_1_generate_topdown.py <scene_path> <target_floor_or_coordinate>")
        print("  target_floor_or_coordinate can be:")
        print("    - An integer (floor index): 0, 1, 2, ...")
        print("    - Three space-separated coordinates: x y z")
        sys.exit(1)
    
    scene_path = sys.argv[1]
    
    # Parse target_floor parameter
    if len(sys.argv) == 3:
        # Single parameter - try to parse as integer first, then as coordinate
        try:
            target_floor = int(sys.argv[2])
        except ValueError:
            target_floor = float(sys.argv[2])  # Single float value
    elif len(sys.argv) == 5:
        # Three coordinates: x y z
        try:
            target_floor = [float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])]
        except ValueError:
            print("Error: Could not parse coordinates as floats")
            sys.exit(1)
    else:
        print("Error: Invalid number of arguments")
        sys.exit(1)
    
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    result = generate_topdown_view(scene_path, target_floor, output_dir=output_dir)
    print("Step 1 completed:", result)
