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


def find_object_entry(data: Dict, object_id: str) -> Tuple[Optional[Dict], Optional[str]]:
	"""
	Locate the object entry in the episodes JSON by object_id.
	Returns (object_entry, category_key) if found, else (None, None).
	"""
	goals = data.get("goals_by_category", {})
	for category_key, obj_list in goals.items():
		for obj in obj_list:
			if obj.get("object_id") == object_id:
				return obj, category_key
	return None, None


def extract_viewpoint_positions(obj_entry: Dict) -> List[Tuple[float, float, float, Optional[float]]]:
	"""
	Extract viewpoint positions (x, y, z) and optional IoU from the object entry.
	Returns list of tuples (x, y, z, iou|None).
	"""
	results: List[Tuple[float, float, float, Optional[float]]] = []
	for vp in obj_entry.get("view_points", []):
		pos = None
		iou = vp.get("iou") if isinstance(vp, dict) else None
		if isinstance(vp, dict):
			# Common structure: vp["agent_state"]["position"]
			agent_state = vp.get("agent_state", {}) if isinstance(vp.get("agent_state", {}), dict) else {}
			pos = agent_state.get("position")
			# Fallbacks
			if pos is None:
				pos = vp.get("position")
		if pos and isinstance(pos, (list, tuple)) and len(pos) == 3:
			try:
				x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
				results.append((x, y, z, float(iou) if iou is not None else None))
			except Exception:
				continue
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
	# Work on PIL image in RGB
	if base_image.shape[2] == 4:
		pil_img = Image.fromarray(base_image[..., :3], "RGB")
	else:
		pil_img = Image.fromarray(base_image, "RGB")
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
	label = f"Object {label_object}"
	draw.text((opx + r + 4, opy - r - 2), label, fill=(255, 255, 255), font=font)

	# Compute IoU range if available for color scaling
	ious = [iou for (_, _, _, iou) in vps if iou is not None]
	iou_min = min(ious) if ious else None
	iou_max = max(ious) if ious else None

	def iou_to_color(iou_val: Optional[float]) -> Tuple[int, int, int]:
		if iou_val is None or iou_min is None or iou_max is None or iou_max <= iou_min:
			return vp_color
		# Map iou in [min, max] -> [0,1]
		t = (iou_val - iou_min) / (iou_max - iou_min + 1e-6)
		# Simple gradient: blue (low) -> red (high)
		r_c = int(255 * t)
		g_c = int(120 * (1 - t) + 30)  # keep some green for visibility
		b_c = int(255 * (1 - t))
		return (r_c, g_c, b_c)

	# Draw viewpoints
	visible_count = 0
	for idx, (x, y, z, iou) in enumerate(vps):
		px, py = world_to_pixel(x, z, coords)
		# Check bounds
		if 0 <= px < pil_img.width and 0 <= py < pil_img.height:
			c = iou_to_color(iou)
			draw.ellipse([px - point_radius, py - point_radius, px + point_radius, py + point_radius], fill=c, outline=(0,0,0), width=2)
			if show_indices:
				draw.text((px + point_radius + 2, py - point_radius - 2), f"{idx}", fill=(255, 255, 255), font=font_small)
			visible_count += 1

	# Legend
	legend_lines = [
		f"Viewpoints: {len(vps)} (visible: {visible_count})",
	]
	if iou_min is not None and iou_max is not None:
		legend_lines.append(f"IoU range: {iou_min:.3f} - {iou_max:.3f}")
	legend = "\n".join(legend_lines)
	draw.rectangle([8, 8, 8 + 380, 8 + 48], fill=(0, 0, 0, 160))
	draw.text((14, 14), legend, fill=(255, 255, 255), font=font_small)

	return pil_img


def infer_scene_id(scene_path: str) -> str:
	# Prefer directory name if structured as .../<scene_id>/<scene_file>
	dirname = os.path.basename(os.path.dirname(scene_path))
	if dirname and dirname != os.path.basename(scene_path):
		return dirname
	# Else use file stem
	return os.path.splitext(os.path.basename(scene_path))[0]


def main():
	parser = argparse.ArgumentParser(description="Visualize viewpoints on a floor topdown view for a specific object.")
	parser.add_argument("--episodes", default="/home/yaoaa/habitat-lab/data/datasets/ovon/hm3d/val_seen/content/LT9Jq6dN3Ea.json", help="Path to the episodes JSON file for the scene")
	parser.add_argument("--scene", default="/home/yaoaa/habitat-lab/data/versioned_data/hm3d-0.2/hm3d/val/00862-LT9Jq6dN3Ea/LT9Jq6dN3Ea.basis.glb", help="Path to the scene .glb file")
	parser.add_argument("--object_id", default="laundry machine_743", help="Target object_id to visualize")
	parser.add_argument("--output", default=None, help="Output image path. Default: viewpoints_visualization/<scene_id>_<object_id>.png")
	parser.add_argument("--custom_ortho_scale", type=float, default=None, help="Optional custom ortho scale override")
	parser.add_argument("--target_coverage", type=float, default=0.9, help="Coverage ratio used to compute ortho scale when not set")
	parser.add_argument("--show_indices", action="store_true", help="Label viewpoint indices next to markers")
	parser.add_argument("--marker_radius", type=int, default=6, help="Radius of viewpoint markers in pixels")
	args = parser.parse_args()

	data = load_episodes(args.episodes)
	obj_entry, category_key = find_object_entry(data, args.object_id)
	if obj_entry is None:
		print(f"Object {args.object_id} not found in {args.episodes}")
		sys.exit(2)

	obj_pos = obj_entry.get("position")
	if not obj_pos or len(obj_pos) != 3:
		print(f"Object {args.object_id} has no valid 'position' in episodes file.")
		sys.exit(3)
	obj_pos = (float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2]))

	vps = extract_viewpoint_positions(obj_entry)
	if not vps:
		print(f"No viewpoint positions found for {args.object_id}.")

	# Render the floor that contains the object using its Y coordinate
	print("Rendering floor topdown view...")
	img, coords, meta = render_topdown_view(
		args.scene,
		target_floor=[obj_pos[0], obj_pos[1], obj_pos[2]],
		custom_ortho_scale=args.custom_ortho_scale,
		target_coverage=args.target_coverage,
		draw_coordinates=False,  # keep raw image to match unprojected coords
	)

	if img is None or coords is None:
		print("Failed to render topdown view.")
		sys.exit(4)

	annotated = draw_points(
		img, coords, obj_pos, vps, args.object_id, show_indices=args.show_indices, point_radius=args.marker_radius
	)

	# Save
	scene_id = infer_scene_id(args.scene)
	out_dir = os.path.join(os.getcwd(), "viewpoints_visualization")
	os.makedirs(out_dir, exist_ok=True)
	out_path = args.output or os.path.join(out_dir, f"{scene_id}_{args.object_id}.png")
	annotated.save(out_path)
	print(f"Saved visualization: {out_path}")


if __name__ == "__main__":
	main()

