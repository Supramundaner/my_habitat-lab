#!/usr/bin/env python3
"""
Batch sample generation script.
从JSON文件中采样objects，生成对应的房间分割图像，并保存对应关系。
"""

import os
import json
import random
import shutil
import sys
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

# Import workflow modules (delay import to avoid dependency issues)
# from main_workflow import WorkflowOrchestrator


def load_objects_from_json(json_path: str) -> List[Dict[str, Any]]:
    """
    从JSON文件中加载所有objects。
    
    Args:
        json_path: JSON文件路径
        
    Returns:
        objects列表，每个object包含category, id, position等信息
    """
    print(f"📁 Loading objects from: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    objects = []
    goals_by_category = data.get('goals_by_category', {})
    
    for category_key, category_objects in goals_by_category.items():
        # 从category_key中提取scene_name，例如 "4ok3usBNeis.basis.glb_bench" -> "4ok3usBNeis"
        scene_name = category_key.split('.basis.glb_')[0]
        
        for obj in category_objects:
            object_info = {
                'scene_name': scene_name,
                'category_key': category_key,
                'object_category': obj['object_category'],
                'object_id': obj['object_id'],
                'position': obj['position'],
                'view_points': obj.get('view_points', [])
            }
            objects.append(object_info)
    
    print(f"✓ Loaded {len(objects)} objects from {len(goals_by_category)} categories")
    return objects


def sample_objects(objects: List[Dict[str, Any]], num_samples: int = 10) -> List[Dict[str, Any]]:
    """
    从objects列表中随机采样指定数量的objects。
    
    Args:
        objects: objects列表
        num_samples: 采样数量
        
    Returns:
        采样后的objects列表
    """
    if len(objects) < num_samples:
        print(f"⚠️  Available objects ({len(objects)}) < requested samples ({num_samples})")
        print(f"   Using all available objects")
        return objects
    
    sampled = random.sample(objects, num_samples)
    print(f"✓ Sampled {len(sampled)} objects")
    
    # 显示采样的objects信息
    for i, obj in enumerate(sampled):
        print(f"  {i+1}. {obj['object_category']} ({obj['object_id']}) at {obj['position']}")
    
    return sampled


def find_scene_directory(scene_name: str) -> str:
    """
    根据scene_name找到实际的场景目录名（包含序号前缀）。
    
    Args:
        scene_name: 从JSON中提取的场景名称，如 "4ok3usBNeis"
        
    Returns:
        实际的目录名，如 "00877-4ok3usBNeis"
    """
    import glob
    
    # 在val目录中查找包含scene_name的目录
    val_dir = "/home/awangas/my_habitat-lab/data/versioned_data/hm3d-0.2/hm3d/val"
    pattern = os.path.join(val_dir, f"*{scene_name}")
    
    matching_dirs = glob.glob(pattern)
    
    if not matching_dirs:
        raise FileNotFoundError(f"No scene directory found for scene_name: {scene_name}")
    
    if len(matching_dirs) > 1:
        print(f"⚠️  Multiple directories found for {scene_name}: {matching_dirs}")
        print(f"   Using the first one: {matching_dirs[0]}")
    
    # 返回目录名（不包含完整路径）
    full_dir_name = os.path.basename(matching_dirs[0])
    return full_dir_name


def create_config_for_object(base_config_path: str, obj: Dict[str, Any], 
                           output_config_path: str) -> str:
    """
    为特定object创建配置文件。
    
    Args:
        base_config_path: 基础配置文件路径
        obj: object信息
        output_config_path: 输出配置文件路径
        
    Returns:
        创建的配置文件路径
    """
    # 加载基础配置
    with open(base_config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 修改scene_path和target_coordinate
    scene_name = obj['scene_name']
    
    # 找到实际的场景目录名（包含序号前缀）
    try:
        actual_dir_name = find_scene_directory(scene_name)
        scene_path = f"data/versioned_data/hm3d-0.2/hm3d/val/{actual_dir_name}/{scene_name}.basis.glb"
        print(f"  📁 Scene directory: {actual_dir_name}")
        print(f"  📄 Scene file: {scene_name}.basis.glb")
    except FileNotFoundError as e:
        print(f"  ❌ {e}")
        # 回退到原来的路径（可能会失败，但至少保持一致性）
        scene_path = f"data/versioned_data/hm3d-0.2/hm3d/val/{scene_name}/{scene_name}.basis.glb"
    
    config['scene_config']['scene_path'] = scene_path
    config['scene_config']['target_coordinate'] = obj['position']
    config['scene_config']['goal_object'] = obj['object_category']
    
    # 保存配置文件
    with open(output_config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    return output_config_path


def world_to_pixel(x: float, z: float, coords: Dict) -> Tuple[int, int]:
    """
    将世界坐标(x, z)投影到图像像素坐标。
    
    Args:
        x, z: 世界坐标
        coords: 由render_topdown_view返回的坐标信息
        
    Returns:
        (px, py): 像素坐标
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


def extract_viewpoint_positions(obj: Dict[str, Any]) -> List[Tuple[float, float, float, Optional[float]]]:
    """
    从object数据中提取viewpoint位置和IoU信息。
    
    Args:
        obj: object数据
        
    Returns:
        List of (x, y, z, iou) tuples
    """
    results = []
    for vp in obj.get("view_points", []):
        pos = None
        iou = vp.get("iou") if isinstance(vp, dict) else None
        if isinstance(vp, dict):
            agent_state = vp.get("agent_state", {}) if isinstance(vp.get("agent_state", {}), dict) else {}
            pos = agent_state.get("position")
            if pos is None:
                pos = vp.get("position")
        if pos and isinstance(pos, (list, tuple)) and len(pos) == 3:
            try:
                x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
                results.append((x, y, z, float(iou) if iou is not None else None))
            except Exception:
                continue
    return results


def create_viewpoint_visualization(obj: Dict[str, Any], topdown_image_path: str, 
                                 coords: Dict, output_path: str) -> bool:
    """
    创建带有object位置和viewpoints的可视化图像。
    
    Args:
        obj: object数据
        topdown_image_path: 俯视图路径
        coords: 坐标变换信息
        output_path: 输出图像路径
        
    Returns:
        bool: 是否成功创建可视化
    """
    try:
        # Dynamic imports to avoid dependency issues
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        import cv2
        
        # 加载俯视图
        base_image = cv2.imread(topdown_image_path)
        if base_image is None:
            print(f"  ❌ Cannot load topdown image: {topdown_image_path}")
            return False
        
        # 转换为RGB格式
        base_image_rgb = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
        
        # 创建PIL图像
        pil_img = Image.fromarray(base_image_rgb, "RGB")
        draw = ImageDraw.Draw(pil_img)
        
        # 字体设置
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 16)
            font_small = ImageFont.truetype("DejaVuSans.ttf", 12)
        except (IOError, OSError):
            font = font_small = ImageFont.load_default()
        
        # 颜色定义
        obj_color = (255, 215, 0)  # 金色 - object位置
        vp_color = (0, 200, 255)   # 青色 - viewpoints
        
        # 绘制object位置
        obj_pos = obj['position']
        ox, oy, oz = float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2])
        opx, opy = world_to_pixel(ox, oz, coords)
        
        # 检查object位置是否在图像范围内
        if 0 <= opx < pil_img.width and 0 <= opy < pil_img.height:
            r = 8  # object标记半径
            draw.ellipse([opx - r, opy - r, opx + r, opy + r], 
                        outline=(0, 0, 0), width=3, fill=obj_color)
            label = f"Object: {obj['object_category']} ({obj['object_id']})"
            draw.text((opx + r + 4, opy - r - 2), label, fill=(255, 255, 255), font=font)
        
        # 提取viewpoints
        vps = extract_viewpoint_positions(obj)
        
        # 计算IoU范围用于颜色映射
        ious = [iou for (_, _, _, iou) in vps if iou is not None]
        iou_min = min(ious) if ious else None
        iou_max = max(ious) if ious else None
        
        def iou_to_color(iou_val: Optional[float]) -> Tuple[int, int, int]:
            if iou_val is None or iou_min is None or iou_max is None or iou_max <= iou_min:
                return vp_color
            # 将IoU映射到颜色：蓝色(低) -> 红色(高)
            t = (iou_val - iou_min) / (iou_max - iou_min + 1e-6)
            r_c = int(255 * t)
            g_c = int(120 * (1 - t) + 30)
            b_c = int(255 * (1 - t))
            return (r_c, g_c, b_c)
        
        # 绘制viewpoints
        visible_count = 0
        point_radius = 4
        for idx, (x, y, z, iou) in enumerate(vps):
            px, py = world_to_pixel(x, z, coords)
            # 检查是否在图像范围内
            if 0 <= px < pil_img.width and 0 <= py < pil_img.height:
                c = iou_to_color(iou)
                draw.ellipse([px - point_radius, py - point_radius, 
                            px + point_radius, py + point_radius], 
                           fill=c, outline=(0, 0, 0), width=1)
                visible_count += 1
        
        # 添加图例
        legend_lines = [
            f"Object: {obj['object_category']}",
            f"Viewpoints: {len(vps)} (visible: {visible_count})"
        ]
        if iou_min is not None and iou_max is not None:
            legend_lines.append(f"IoU range: {iou_min:.3f} - {iou_max:.3f}")
        
        legend = "\n".join(legend_lines)
        # 绘制半透明背景
        legend_bbox = draw.textbbox((0, 0), legend, font=font_small)
        legend_width = legend_bbox[2] - legend_bbox[0] + 20
        legend_height = legend_bbox[3] - legend_bbox[1] + 20
        
        draw.rectangle([8, 8, 8 + legend_width, 8 + legend_height], 
                      fill=(0, 0, 0), outline=(255, 255, 255))
        draw.text((14, 14), legend, fill=(255, 255, 255), font=font_small)
        
        # 保存图像
        pil_img.save(output_path)
        print(f"  ✓ Viewpoint visualization saved: {os.path.basename(output_path)}")
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to create viewpoint visualization: {e}")
        return False


def run_room_segmentation_for_object(config_path: str, output_dir: str) -> Dict[str, Any]:
    """
    为指定配置运行房间分割流程。
    
    Args:
        config_path: 配置文件路径
        output_dir: 输出目录
        
    Returns:
        运行结果
    """
    try:
        # Dynamic import to avoid dependency issues
        from main_workflow import WorkflowOrchestrator
        orchestrator = WorkflowOrchestrator(config_path)
        # 更新输出目录
        orchestrator.output_dir = output_dir
        orchestrator.config['output']['output_dir'] = output_dir
        
        # 只运行前3个步骤 (topdown, wall_mask, room_segmentation)
        success = (orchestrator.run_step_0() and 
                  orchestrator.run_step_1() and 
                  orchestrator.run_step_2())
        
        if success:
            # 保存输出数据
            orchestrator._save_output()
            return {
                'success': True,
                'output_data': orchestrator.output_data,
                'generated_files': orchestrator.output_data.get('generated_files', {}),
                'results': orchestrator.output_data.get('final_results', {})
            }
        else:
            return {
                'success': False,
                'error': 'Workflow failed',
                'output_data': orchestrator.output_data
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'traceback': str(e)
        }


def batch_generate_samples(json_path: str, base_config_path: str, 
                          output_base_dir: str, num_samples: int = 10) -> Dict[str, Any]:
    """
    批量生成样本。
    
    Args:
        json_path: objects JSON文件路径
        base_config_path: 基础配置文件路径
        output_base_dir: 输出基础目录
        num_samples: 采样数量
        
    Returns:
        批量处理结果
    """
    print("🚀" + "="*60 + "🚀")
    print("🎯 STARTING BATCH SAMPLE GENERATION 🎯")
    print("🚀" + "="*60 + "🚀")
    
    # 创建输出目录
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 加载和采样objects
    objects = load_objects_from_json(json_path)
    sampled_objects = sample_objects(objects, num_samples)
    
    # 批量处理结果
    batch_results = {
        'total_samples': len(sampled_objects),
        'successful_samples': 0,
        'failed_samples': 0,
        'samples': [],
        'summary': {}
    }
    
    # 处理每个object
    for i, obj in enumerate(sampled_objects):
        print(f"\n{'='*80}")
        print(f"处理样本 {i+1}/{len(sampled_objects)}: {obj['object_category']} ({obj['object_id']})")
        print(f"{'='*80}")
        
        # 为每个object创建单独的输出目录
        sample_name = f"{obj['object_category']}_{obj['object_id']}"
        # 清理文件名中的特殊字符
        sample_name = sample_name.replace(' ', '_').replace('/', '_')
        sample_output_dir = os.path.join(output_base_dir, f"sample_{i+1:02d}_{sample_name}")
        os.makedirs(sample_output_dir, exist_ok=True)
        
        # 创建配置文件
        config_path = os.path.join(sample_output_dir, "config.json")
        try:
            create_config_for_object(base_config_path, obj, config_path)
            print(f"✓ Config created: {config_path}")
        except Exception as e:
            print(f"✗ Failed to create config: {e}")
            batch_results['failed_samples'] += 1
            batch_results['samples'].append({
                'sample_id': i + 1,
                'object_info': obj,
                'success': False,
                'error': f"Config creation failed: {e}",
                'output_dir': sample_output_dir
            })
            continue
        
        # 运行房间分割
        print(f"🔄 Running room segmentation...")
        result = run_room_segmentation_for_object(config_path, sample_output_dir)
        
        if result['success']:
            print(f"✅ Sample {i+1} completed successfully")
            batch_results['successful_samples'] += 1
            
            # 保存object信息到输出目录
            object_info_path = os.path.join(sample_output_dir, "object_info.json")
            with open(object_info_path, 'w', encoding='utf-8') as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)
            
            # 创建viewpoint可视化（如果有topdown图像和坐标信息）
            generated_files = result.get('generated_files', {})
            topdown_path = generated_files.get('topdown_view')
            
            if topdown_path and os.path.exists(topdown_path):
                try:
                    # 尝试从metadata中获取坐标信息
                    metadata_path = generated_files.get('metadata')
                    if metadata_path and os.path.exists(metadata_path):
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        
                        unprojected_coords = metadata.get('unprojected_coords')
                        if unprojected_coords:
                            viz_output_path = os.path.join(sample_output_dir, "viewpoint_visualization.png")
                            print(f"🎨 Creating viewpoint visualization...")
                            
                            viz_success = create_viewpoint_visualization(
                                obj, topdown_path, unprojected_coords, viz_output_path
                            )
                            
                            if viz_success:
                                generated_files['viewpoint_visualization'] = viz_output_path
                        else:
                            print(f"  ⚠️  No unprojected_coords found in metadata")
                    else:
                        print(f"  ⚠️  No metadata file found for visualization")
                except Exception as e:
                    print(f"  ❌ Visualization creation failed: {e}")
            else:
                print(f"  ⚠️  No topdown image found for visualization")
            
            batch_results['samples'].append({
                'sample_id': i + 1,
                'object_info': obj,
                'success': True,
                'output_dir': sample_output_dir,
                'generated_files': generated_files,
                'results': result.get('results', {})
            })
        else:
            print(f"❌ Sample {i+1} failed: {result.get('error', 'Unknown error')}")
            batch_results['failed_samples'] += 1
            batch_results['samples'].append({
                'sample_id': i + 1,
                'object_info': obj,
                'success': False,
                'error': result.get('error', 'Unknown error'),
                'output_dir': sample_output_dir
            })
    
    # 生成汇总信息
    batch_results['summary'] = {
        'success_rate': batch_results['successful_samples'] / batch_results['total_samples'] * 100,
        'categories_processed': len(set(obj['object_category'] for obj in sampled_objects)),
        'scenes_processed': len(set(obj['scene_name'] for obj in sampled_objects))
    }
    
    # 保存批量处理结果
    batch_result_path = os.path.join(output_base_dir, "batch_results.json")
    with open(batch_result_path, 'w', encoding='utf-8') as f:
        json.dump(batch_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉" + "="*60 + "🎉")
    print(f"🎯 BATCH PROCESSING COMPLETED 🎯")
    print(f"📊 Results: {batch_results['successful_samples']}/{batch_results['total_samples']} successful")
    print(f"📈 Success rate: {batch_results['summary']['success_rate']:.1f}%")
    print(f"📁 Results saved to: {batch_result_path}")
    print(f"🎉" + "="*60 + "🎉")
    
    return batch_results


def main():
    """主函数"""
    if len(sys.argv) < 4:
        print("Usage: python batch_sample_generation.py <json_path> <base_config_path> <output_dir> [num_samples]")
        print("Example: python preprocess/batch_sample_generation.py data/datasets/hm3d_ovon/val_seen/content/4ok3usBNeis.json preprocess/input_config.json output/batch_samples 10")
        sys.exit(1)
    
    json_path = sys.argv[1]
    base_config_path = sys.argv[2]
    output_dir = sys.argv[3]
    num_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    
    # 检查输入文件是否存在
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        sys.exit(1)
    
    if not os.path.exists(base_config_path):
        print(f"Error: Base config file not found: {base_config_path}")
        sys.exit(1)
    
    # 设置随机种子以保证可重现性
    random.seed(42)
    
    # 运行批量生成
    results = batch_generate_samples(json_path, base_config_path, output_dir, num_samples)
    
    # 返回成功状态
    success_rate = results['summary']['success_rate']
    sys.exit(0 if success_rate > 0 else 1)


if __name__ == "__main__":
    main()
