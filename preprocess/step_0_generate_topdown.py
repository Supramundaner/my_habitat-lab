"""
Step 0: Generate topdown view and metadata using the current_topdown.py functionality.
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

def generate_topdown_view(config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Generate topdown view and metadata.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory path
        
    Returns:
        Dictionary with generated files and results
    """
    scene_config = config['scene_config']
    scene_path = scene_config['scene_path']
    
    # Determine target floor
    if scene_config['target_coordinate'] is not None:
        target_floor = scene_config['target_coordinate']
    else:
        target_floor = scene_config.get('target_floor', 0)
    
    print(f"📁 Scene path: {scene_path}")
    print(f"🏢 Target floor: {target_floor}")
    
    # Check if scene file exists
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene file not found: {scene_path}")
    
    try:
        # Generate topdown view
        result_image, unprojected_coords, meta_data = render_topdown_view(
            scene_path,
            target_floor=target_floor,
            custom_ortho_scale=scene_config.get('custom_ortho_scale'),
            target_coverage=scene_config.get('target_coverage', 0.9),
            draw_coordinates=scene_config.get('draw_coordinates', False)
        )
        
        if result_image is None:
            raise RuntimeError("Failed to generate topdown view")
        
        # Save topdown view image
        topdown_path = os.path.join(output_dir, "topdown_view.png")
        
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
        metadata_path = os.path.join(output_dir, "metadata.json")
        combined_metadata = {
            "topdown_metadata": meta_data,
            "unprojected_coords": unprojected_coords,
            "scene_info": {
                "scene_path": scene_path,
                "target_floor": target_floor
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
        
    except Exception as e:
        raise RuntimeError(f"Failed to generate topdown view: {str(e)}")

if __name__ == "__main__":
    # Test function
    import sys
    if len(sys.argv) != 2:
        print("Usage: python step_0_generate_topdown.py <config_path>")
        sys.exit(1)
        
    with open(sys.argv[1], 'r') as f:
        config = json.load(f)
    
    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    result = generate_topdown_view(config, output_dir)
    print("Step 0 completed:", result)
