"""
Step 2: Generate wall mask from topdown view.
The black areas in the topdown view represent walls, white/colored areas are walkable.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, Tuple

def generate_wall_mask(topdown_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Generate wall mask from topdown view image.
    
    Args:
        topdown_path: Path to the topdown view image
        output_dir: Output directory path
        
    Returns:
        Dictionary with generated files and results
    """
    print(f"📁 Loading topdown view from: {topdown_path}")
    
    if not os.path.exists(topdown_path):
        raise FileNotFoundError(f"Topdown view image not found: {topdown_path}")
    
    # Load the topdown view image
    image = cv2.imread(topdown_path)
    if image is None:
        raise RuntimeError(f"Failed to load image from: {topdown_path}")
    
    print(f"✓ Image loaded, size: {image.shape[1]}x{image.shape[0]}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create wall mask
    # Dark areas (close to black) are walls (value 0 in mask)
    # Bright areas are walkable (value 255 in mask)
    
    # Define threshold for wall detection
    # Pixels darker than this threshold are considered walls
    wall_threshold = 0
    
    # Create binary mask: 255 for walkable, 0 for walls
    _, wall_mask = cv2.threshold(gray, wall_threshold, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to clean up the mask
    # Remove small noise and fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # Close small gaps in walkable areas
    #wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Remove small noise in walls
    #wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Create inverted mask (255 for walls, 0 for walkable)
    wall_mask_inverted = 255 - wall_mask
    
    # Save the wall mask (255 for walkable, 0 for walls)
    wall_mask_path = os.path.join(output_dir, "wall_mask.png")
    cv2.imwrite(wall_mask_path, wall_mask)
    print(f"✓ Wall mask saved to: {wall_mask_path}")
    
    # Calculate statistics
    total_pixels = wall_mask.shape[0] * wall_mask.shape[1]
    walkable_pixels = cv2.countNonZero(wall_mask)
    wall_pixels = total_pixels - walkable_pixels
    walkable_percentage = (walkable_pixels / total_pixels) * 100
    wall_percentage = (wall_pixels / total_pixels) * 100
    
    print(f"📊 Mask statistics:")
    print(f"  - Total pixels: {total_pixels:,}")
    print(f"  - Walkable pixels: {walkable_pixels:,} ({walkable_percentage:.1f}%)")
    print(f"  - Wall pixels: {wall_pixels:,} ({wall_percentage:.1f}%)")
    
    return {
        "generated_files": {
            "wall_mask": wall_mask_path
        },
        "results": {
            "total_pixels": total_pixels,
            "walkable_pixels": int(walkable_pixels),
            "wall_pixels": int(wall_pixels),
            "walkable_percentage": walkable_percentage,
            "wall_percentage": wall_percentage,
            "wall_threshold": wall_threshold
        }
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 3:
        topdown_path = sys.argv[1]
        output_dir = sys.argv[2]
        os.makedirs(output_dir, exist_ok=True)
        result = generate_wall_mask(topdown_path, output_dir)
        print("Step 2 completed:", result)
    else:
        print("Usage: python step_2_generate_wall_mask.py <topdown_path> <output_dir>")
        sys.exit(1)
