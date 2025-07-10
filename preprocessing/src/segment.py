import argparse
import torch
import numpy as np
from PIL import Image as PILImage
import os
import json
import cv2

# 确保安装了SAM2和transformers, anndata(用于Hugging Face Hub)
try:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError:
    print("Please install SAM2 dependencies: pip install git+https://github.com/facebookresearch/segment-anything-2.git")
    exit()



# --- 过滤和合并参数 ---
MIN_MASK_AREA = 400
MAX_MASK_AREA = 10000
MERGE_IOU_THRESHOLD = 0.1
MERGE_DISTANCE_THRESHOLD = 10
MERGE_COLOR_SIMILARITY_THRESHOLD = 0.95

# --- 上下文裁剪参数 ---
CROP_PADDING = 80

# --- SAM2 特定参数 ---
POINTS_PER_SIDE = 32
PRED_IOU_THRESH = 0.9
STABILITY_SCORE_THRESH = 0.9
CROP_N_LAYERS = 2
CROP_N_POINTS_DOWNSCALE_FACTOR = 1

# --- 设备参数 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def calculate_iou(mask1, mask2):
    """Calculates IoU for two boolean masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    if intersection == 0:
        return 0.0
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union

def calculate_color_similarity(image, mask1, mask2, bbox1, bbox2):
    """
    Calculates color similarity between two masked regions using HSV histograms.
    This version is more robust by using bounding boxes and the 'mask' parameter.
    """
    # Helper to get a cropped region and its mask
    def get_masked_region(img, mask, bbox):
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)

        if w == 0 or h == 0:
            return None, None
        # Crop both the image and the mask
        img_crop = img[y:y+h, x:x+w]
        mask_crop = mask[y:y+h, x:x+w]
        return img_crop, mask_crop.astype(np.uint8)

    # Get cropped regions for both masks
    region1_img, region1_mask = get_masked_region(image, mask1, bbox1)
    region2_img, region2_mask = get_masked_region(image, mask2, bbox2)

    # Check if regions are valid
    if region1_img is None or region2_img is None:
        return 0.0
    if np.sum(region1_mask) == 0 or np.sum(region2_mask) == 0:
        return 0.0

    # Convert cropped images to HSV
    hsv_region1 = cv2.cvtColor(region1_img, cv2.COLOR_RGB2HSV)
    hsv_region2 = cv2.cvtColor(region2_img, cv2.COLOR_RGB2HSV)
    
    # Calculate H-S二维直方图, 使用 mask 参数来只计算感兴趣的区域
    hist_region1 = cv2.calcHist([hsv_region1], [0, 1], region1_mask, [30, 32], [0, 180, 0, 256])
    hist_region2 = cv2.calcHist([hsv_region2], [0, 1], region2_mask, [30, 32], [0, 180, 0, 256])
    
    cv2.normalize(hist_region1, hist_region1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_region2, hist_region2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    similarity = cv2.compareHist(hist_region1, hist_region2, cv2.HISTCMP_CORREL)
    
    return max(0, similarity)

def merge_masks_robust(image_np, masks, iou_threshold=0.1, distance_threshold=10, color_similarity_threshold=0.7):
    """
    Robustly merges masks based on a graph connectivity approach.
    """
    if not masks:
        return []

    num_masks = len(masks)

    contours = []
    for m in masks:
        # cv2.findContours需要uint8类型的输入
        mask_uint8 = m['segmentation'].astype(np.uint8)
        # RETR_EXTERNAL只找外轮廓，CHAIN_APPROX_SIMPLE压缩轮廓点
        cnts, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 通常只有一个外轮廓，我们取最大的那个
        if cnts:
            contours.append(max(cnts, key=cv2.contourArea))
        else:
            contours.append(None) # 如果mask是空的或无效的

    # Build adjacency matrix based on IoU
    adj = [[False] * num_masks for _ in range(num_masks)]
    for i in range(num_masks):
        for j in range(i + 1, num_masks):
            # --- 【核心改进】 ---
            # 标准1：IOU (检查重叠)
            iou = calculate_iou(masks[i]['segmentation'], masks[j]['segmentation'])
            if iou > iou_threshold:
                adj[i][j] = adj[j][i] = True
                continue # 如果已经满足IOU，就不需要再计算距离了

            # 标准2：距离 (检查近邻)
            contour1 = contours[i]
            contour2 = contours[j]
            
            # 确保两个轮廓都存在
            if contour1 is None or contour2 is None or len(contour1) == 0 or len(contour2) == 0:
                continue

            # 这是一个高效计算距离的近似方法
            # 遍历contour1上的每个点，计算它到contour2的最短距离
            min_dist = float('inf')
            # 为了效率，可以只对轮廓上的一部分点进行采样
            # 这里为了简单，我们遍历所有点
            for pt in contour1.reshape(-1, 2): # 将点转换为 (N, 2) 形状
                dist = cv2.pointPolygonTest(contour2, tuple(float(x) for x in pt), True)
                # dist是带符号的距离，内部为正，外部为负，边界为0
                min_dist = min(min_dist, abs(dist))
            
            if min_dist < distance_threshold:
                # --- 【核心修复】 ---
                # 传递bbox给相似度计算函数
                bbox_i = masks[i]['bbox']
                bbox_j = masks[j]['bbox']
                similarity = calculate_color_similarity(
                    image_np, 
                    masks[i]['segmentation'], 
                    masks[j]['segmentation'],
                    bbox_i,
                    bbox_j
                )
                if similarity > color_similarity_threshold:
                    adj[i][j] = adj[j][i] = True
    

    # Find connected components using DFS
    visited = [False] * num_masks
    merged_masks_final = []
    for i in range(num_masks):
        if not visited[i]:
            component_indices = []
            stack = [i]
            visited[i] = True
            
            while stack:
                u = stack.pop()
                component_indices.append(u)
                for v in range(num_masks):
                    if adj[u][v] and not visited[v]:
                        visited[v] = True
                        stack.append(v)
            
            # Merge all masks in the current component
            if not component_indices: continue
                
            # Initialize with the first mask of the component
            base_mask = masks[component_indices[0]]['segmentation'].copy()
            for k in range(1, len(component_indices)):
                base_mask = np.logical_or(base_mask, masks[component_indices[k]]['segmentation'])
            
            # --- Update Metadata for the new merged mask ---
            ys, xs = np.where(base_mask)
            if len(xs) == 0: continue
            
            # Bbox
            bbox_xywh = [
                int(np.min(xs)), 
                int(np.min(ys)), 
                int(np.max(xs) - np.min(xs)), 
                int(np.max(ys) - np.min(ys))
            ]
            
            # Area
            area = int(base_mask.sum())
            
            # Scores (average them)
            avg_iou = np.mean([masks[k]['predicted_iou'] for k in component_indices])
            avg_stability = np.mean([masks[k]['stability_score'] for k in component_indices])

            merged_masks_final.append({
                'segmentation': base_mask,
                'area': area,
                'bbox': bbox_xywh,
                'predicted_iou': avg_iou,
                'stability_score': avg_stability,
                'component_indices': component_indices # Keep track of original masks
            })
            
    return merged_masks_final

def process_and_save_masks(image_np, masks, base_output_dir, padding_pixels):
    """
    Processes final masks to save contextual crops, binary masks, and metadata.json.
    """
    crops_dir = os.path.join(base_output_dir, 'crops')
    masks_dir = os.path.join(base_output_dir, 'masks')
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    metadata_list = []
    img_h, img_w = image_np.shape[:2]

    print(f"\nProcessing {len(masks)} final masks to generate structured output...")

    for i, mask_data in enumerate(masks):
        segmentation_mask = mask_data['segmentation']
        x, y, w, h = mask_data['bbox']

        # Calculate padded bounding box for contextual crop
        x1_padded = max(0, x - padding_pixels)
        y1_padded = max(0, y - padding_pixels)
        x2_padded = min(img_w, x + w + padding_pixels)
        y2_padded = min(img_h, y + h + padding_pixels)
        
        # Create the contextual crop
        contextual_crop = image_np[y1_padded:y2_padded, x1_padded:x2_padded]

        # Highlight the primary mask on the crop
        highlight_overlay = contextual_crop.copy()
        mask_in_crop_coords = segmentation_mask[y1_padded:y2_padded, x1_padded:x2_padded]
        red_color = np.array([255, 0, 0], dtype=np.uint8)
        highlight_overlay[mask_in_crop_coords] = \
            highlight_overlay[mask_in_crop_coords] * 0.6 + red_color * 0.4
        
        # Define file paths
        crop_filename = f"object_{i:04d}.png"
        mask_filename = f"mask_{i:04d}.png"
        crop_filepath = os.path.join(crops_dir, crop_filename)
        mask_filepath = os.path.join(masks_dir, mask_filename)

        # Save images
        PILImage.fromarray(highlight_overlay.astype(np.uint8)).save(crop_filepath)
        PILImage.fromarray(segmentation_mask.astype(np.uint8) * 255).save(mask_filepath)

        # Assemble metadata for this object
        object_metadata = {
            "id": i,
            "area": mask_data['area'],
            "predicted_iou": float(mask_data['predicted_iou']),
            "stability_score": float(mask_data['stability_score']),
            "bbox_xywh": mask_data['bbox'],
            "crop_image_path": os.path.join('crops', crop_filename),
            "mask_path": os.path.join('masks', mask_filename)
        }
        metadata_list.append(object_metadata)

    # Save the final metadata JSON file
    metadata_filepath = os.path.join(base_output_dir, 'metadata.json')
    with open(metadata_filepath, 'w') as f:
        json.dump(metadata_list, f, indent=4)
    
    print(f"\nSuccessfully generated structured output.")
    print(f"Contextual cropped images saved in: {crops_dir}")
    print(f"Mask files saved in: {masks_dir}")
    print(f"Metadata saved to: {metadata_filepath}")

def segment(base_dir, file_name):
    IMAGE_PATH = os.path.join(base_dir, "data", "top_down", file_name + ".png")
    OUTPUT_DIR = os.path.join(base_dir, "data", "processed", file_name)
    SEGMENTATION_MODEL_PATH = "facebook/sam2-hiera-large"



    print(f"--- Step 0: Setup ---")
    print(f"Using device: {DEVICE}")
    print(f"Loading image: {IMAGE_PATH}")
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image path not found: {IMAGE_PATH}")
        return
    image_np = np.array(PILImage.open(IMAGE_PATH).convert("RGB"))
    
    print(f"Loading SAM2 model: {SEGMENTATION_MODEL_PATH}")
    mask_generator = SAM2AutomaticMaskGenerator.from_pretrained(
        SEGMENTATION_MODEL_PATH,
        points_per_side=POINTS_PER_SIDE,
        pred_iou_thresh=PRED_IOU_THRESH,
        stability_score_thresh=STABILITY_SCORE_THRESH,
        crop_n_layers=CROP_N_LAYERS,
        crop_n_points_downscale_factor=CROP_N_POINTS_DOWNSCALE_FACTOR,
    )

    print(f"\n--- Step 1: Raw Mask Generation ---")
    with torch.no_grad():
        raw_masks = mask_generator.generate(image_np)
    print(f"Generated {len(raw_masks)} raw masks.")
    
    print(f"\n--- Step 2: Merging Masks ---")
    raw_masks = [m for m in raw_masks if m['area'] >= MIN_MASK_AREA and m['area'] <= MAX_MASK_AREA]

    merged_masks = merge_masks_robust(image_np, raw_masks, iou_threshold=MERGE_IOU_THRESHOLD, distance_threshold=MERGE_DISTANCE_THRESHOLD,color_similarity_threshold=MERGE_COLOR_SIMILARITY_THRESHOLD)
    print(f"Resulted in {len(merged_masks)} final masks after merging.")

    print(f"\n--- Step 3: Filtering Small Masks ---")
    final_masks = [m for m in merged_masks if m['area'] >= MIN_MASK_AREA and m['area'] <= MAX_MASK_AREA]
    print(f"Kept {len(final_masks)} masks after filtering.")
    
    # Sort final masks by area for consistent output
    final_masks = sorted(final_masks, key=lambda x: x['area'], reverse=True)

    print(f"\n--- Step 4: Saving Structured Output ---")
    process_and_save_masks(image_np, final_masks, OUTPUT_DIR, padding_pixels=CROP_PADDING)

    print("\nWorkflow completed successfully!")

if __name__ == "__main__":
    segment()