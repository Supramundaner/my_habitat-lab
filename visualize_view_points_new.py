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
except Exception as e:  # pragma: no cover - robust import fallback
    # Fallback: add repo root to sys.path explicitly and retry
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    from preprocess.current_topdown import render_topdown_view

def load_episodes(episodes_path: str) -> Dict:
    with open(episodes_path, "r") as f:
        return json.load(f)

def find_object_entry(data: Dict, object_id: int) -> Optional[Dict]:
    """
    Locate the object entry in the episodes JSON by object_id.
    Returns the object entry if found, else None.
    """
    for category, objects in data.get("goals_by_category", {}).items():
        for obj in objects:
            if obj.get("object_id") == object_id:
                return obj
    return None

def extract_viewpoint_positions(obj_entry: Dict) -> List[Tuple[float, float, float, Optional[float]]]:
    """
    Extract viewpoint positions (x, y, z) and optional IoU from the object entry.
    Returns list of tuples (x, y, z, iou|None).
    """
    results = []
    for vp in obj_entry.get("view_points", []):
        agent_state = vp.get("agent_state", {})
        position = agent_state.get("position")
        iou = vp.get("iou")
        if position and len(position) == 3:
            results.append((position[0], position[1], position[2], iou))
    return results

def world_to_pixel(x: float, z: float, coords: Dict) -> Tuple[int, int]:
    """
    Project world (x, z) to image pixel using unprojected corner coords produced by render_topdown_view.
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

def draw_points(
    base_image: np.ndarray,
    coords: Dict,
    object_pos: Tuple[float, float, float],
    vps: List[Tuple[float, float, float, Optional[float]]],
    label_object: str,
    show_indices: bool = False,
    point_radius: int = 6,
) -> Image.Image:
    """
    Draw the object position and viewpoint positions on the image.
    """
    pil_img = Image.fromarray(base_image[..., :3], "RGB") if base_image.shape[2] == 4 else Image.fromarray(base_image, "RGB")
    draw = ImageDraw.Draw(pil_img)

    # Fonts
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 12)
    except IOError:
        font = font_small = ImageFont.load_default()

    # Colors
    obj_color = (255, 215, 0)  # gold
    vp_color = (0, 200, 255)   # cyan

    # Draw object position
    ox, oy, oz = object_pos
    opx, opy = world_to_pixel(ox, oz, coords)
    r = point_radius + 2
    draw.ellipse([opx - r, opy - r, opx + r, opy + r], outline=(0, 0, 0), width=3, fill=obj_color)
    draw.text((opx + r + 4, opy - r - 2), f"Object {label_object}", fill=(255, 255, 255), font=font)

    # Draw viewpoints
    for idx, (x, y, z, iou) in enumerate(vps):
        px, py = world_to_pixel(x, z, coords)
        if 0 <= px < pil_img.width and 0 <= py < pil_img.height:
            draw.ellipse([px - point_radius, py - point_radius, px + point_radius, py + point_radius], fill=vp_color, outline=(0, 0, 0), width=2)
            if show_indices:
                draw.text((px + point_radius + 2, py - point_radius - 2), f"{idx}", fill=(255, 255, 255), font=font_small)

    return pil_img

def main():
    parser = argparse.ArgumentParser(description="Visualize viewpoints on a floor topdown view for a specific object.")
    parser.add_argument("--episodes", required=True, help="Path to the episodes JSON file for the scene")
    parser.add_argument("--scene", required=True, help="Path to the scene .glb file")
    parser.add_argument("--object_id", type=int, required=True, help="Target object_id to visualize")
    parser.add_argument("--output", default=None, help="Output image path. Default: viewpoints_visualization/<scene_id>_<object_id>.png")
    parser.add_argument("--show_indices", action="store_true", help="Label viewpoint indices next to markers")
    parser.add_argument("--marker_radius", type=int, default=6, help="Radius of viewpoint markers in pixels")
    args = parser.parse_args()

    data = load_episodes(args.episodes)
    obj_entry = find_object_entry(data, args.object_id)
    if obj_entry is None:
        print(f"Object {args.object_id} not found in {args.episodes}")
        sys.exit(2)

    obj_pos = obj_entry.get("position")
    if not obj_pos or len(obj_pos) != 3:
        print(f"Object {args.object_id} has no valid 'position' in episodes file.")
        sys.exit(3)

    vps = extract_viewpoint_positions(obj_entry)
    if not vps:
        print(f"No viewpoint positions found for {args.object_id}.")

    # Render the floor that contains the object using its Y coordinate
    print("Rendering floor topdown view...")
    img, coords, meta = render_topdown_view(
        args.scene,
        target_floor=[obj_pos[0], obj_pos[1], obj_pos[2]],
        draw_coordinates=False,  # keep raw image to match unprojected coords
    )

    if img is None or coords is None:
        print("Failed to render topdown view.")
        sys.exit(4)

    annotated = draw_points(
        img, coords, tuple(obj_pos), vps, str(args.object_id), show_indices=args.show_indices, point_radius=args.marker_radius
    )

    # Save
    scene_id = os.path.splitext(os.path.basename(args.scene))[0]
    out_dir = os.path.join(os.getcwd(), "viewpoints_visualization")
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.output or os.path.join(out_dir, f"{scene_id}_{args.object_id}.png")
    annotated.save(out_path)
    print(f"Saved visualization: {out_path}")

if __name__ == "__main__":
    main()
