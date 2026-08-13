"""
HabitatSimulator - 核心模拟器类 (Model)
处理Habitat-Sim的底层交互和状态管理
"""

import numpy as np
import magnum as mn
import habitat_sim
from PIL import Image
from typing import Dict, Tuple, Optional, Any, List # 确保 List 已导入
import math
import os

from .utils import convert_to_magnum_quat, convert_to_numpy_quat
from .topdown import MultiFloorTopdownRenderer


class _AgentStateRestoreError(RuntimeError):
    """Raised when a temporary sensor pose cannot be rolled back safely."""


class HabitatSimulator:
    """封装Habitat-sim相关逻辑的核心模拟器类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化模拟器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.sim = None
        self.agent = None
        self.robot_object = None
        
        # 场景信息
        self.scene_bounds = None
        self.scene_center = None
        self.scene_size = None
        self.base_map_image = None
        
        # 多楼层渲染器
        self.multi_floor_renderer = None
        
        # 坐标转换参数
        self.map_width = None
        self.map_height = None
        self.world_to_map_scale_x = None
        self.world_to_map_scale_z = None
        self.topdown_floor = None
        
        try:
            self._initialize_sim()
        except Exception:
            # Factory construction can fail after Habitat has allocated native
            # resources.  Since the partially built object is never returned
            # to the application service, it must clean itself up here.
            self.close()
            raise
    
    def _initialize_sim(self):
        """配置并创建habitat_sim.Simulator实例"""
        # 后端配置
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = self.config['scene']['scene_file']
        backend_cfg.enable_physics = self.config['simulation']['enable_physics']
        backend_cfg.random_seed = 1
        camera_hfov = float(self.config['OCCUPANCY_MAP']['HFOV'])
        
        # FPV传感器配置
        fpv_sensor_spec = habitat_sim.CameraSensorSpec()
        fpv_sensor_spec.uuid = "color_sensor"
        fpv_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        fpv_sensor_spec.resolution = [512, 512]  # 临时分辨率，后续会调整
        fpv_sensor_spec.position = mn.Vector3(0, 0, 0)  # 不设置高度offset，让传感器在agent位置
        fpv_sensor_spec.hfov = camera_hfov
        
        # 深度传感器配置
        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.resolution = [512, 512]
        depth_sensor_spec.position = mn.Vector3(0, 0, 0)
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.hfov = camera_hfov
        
        # 正交传感器配置（用于生成topdown地图）
        # 注意：ortho_scale将在_generate_topdown_map中根据场景大小动态设置
        ortho_sensor_spec = habitat_sim.CameraSensorSpec()
        ortho_sensor_spec.uuid = "ortho_sensor"
        ortho_sensor_spec.resolution = [4096, 4096]  # 高分辨率用于地图生成
        ortho_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        ortho_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
        ortho_sensor_spec.far = 1000.0
        ortho_sensor_spec.near = 0.01
        ortho_sensor_spec.hfov = 90
        ortho_sensor_spec.ortho_scale = 0.05  # 临时值，将在地图生成时动态更新
        ortho_sensor_spec.clear_color = [0., 0., 0., 0.]
        
        # 智能体配置
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [fpv_sensor_spec, depth_sensor_spec, ortho_sensor_spec]
        
        # 动作空间配置
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "move_backward": habitat_sim.agent.ActionSpec(
                "move_backward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
        }
        
        # 创建模拟器
        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        self.sim = habitat_sim.Simulator(cfg)
        self.agent = self.sim.get_agent(0)
        
        # Ensure the canonical runtime and top-down renderer use a navmesh.
        self._ensure_navmesh_loaded()
        if not self.sim.pathfinder.is_loaded:
            raise RuntimeError(
                "Navigation mesh is unavailable after recomputation"
            )
        
        print("模拟器初始化完成")
    
    def plan_path(self, start_pos: np.ndarray, end_pos: np.ndarray) -> Optional[List[np.ndarray]]:
        """
        使用Habitat的pathfinder规划从起点到终点的路径。
        
        Args:
            start_pos: 起始位置 [x, y, z]
            end_pos: 终点位置 [x, y, z]
            
        Returns:
            一个包含路径上所有航点(waypoint)的列表，如果无路径则返回None。
        """
        path = habitat_sim.ShortestPath()
        path.requested_start = start_pos
        path.requested_end = end_pos
        
        # 尝试找到路径
        found_path = self.sim.pathfinder.find_path(path)
        
        # 检查路径是否有效
        if not found_path or path.geodesic_distance == math.inf:
            print(f"警告: 无法找到从 {start_pos} 到 {end_pos} 的路径。")
            return None
            
        # 路径点少于2个也意味着无效
        if len(path.points) < 2:
            print(f"警告: 路径点过少，无法形成有效路径。")
            return None

        print(f"路径规划成功，共 {len(path.points)} 个航点，总长 {path.geodesic_distance:.2f} 米。")
        
        # 将路径点转换为numpy数组列表返回
        return [np.array(p) for p in path.points]
    
    def _snap_to_navmesh(self, position: np.ndarray, island_index: int = -1) -> Optional[np.ndarray]:
        """
        将一个3D位置点吸附到最近的可导航网格上。
        
        Args:
            position: 期望的3D位置 [x, y, z]。
            island_index: 导航网格岛屿的索引，-1表示在所有岛屿中搜索。
        
        Returns:
            吸附到导航网格上的3D坐标 [x, y, z]，如果无法找到可导航点则返回None。
        """
        if not self.sim.pathfinder.is_loaded:
            print("错误: 导航网格未加载，无法执行吸附操作。")
            return None
        
        snapped_point = self.sim.pathfinder.snap_point(position, island_index=island_index)
        
        # 检查吸附后的点是否真正可导航
        if self.sim.pathfinder.is_navigable(snapped_point):
            snapped_pos_np = np.array(snapped_point, dtype=np.float32)
            distance_moved = np.linalg.norm(snapped_pos_np - position)
            
            # 如果移动距离大于一个很小的值，就打印提示信息
            if distance_moved > 1e-4:
                print(f"提示: 初始位置已吸附到导航网格。")
                print(f"  - 原始位置: {position}")
                print(f"  - 吸附后位置: {snapped_pos_np}")
                print(f"  - 移动距离: {distance_moved:.4f} 米")
            
            return snapped_pos_np
        else:
            print(f"警告: 无法将位置 {position} 吸附到任何可导航点。")
            return None
    
    def _ensure_navmesh_loaded(self):
        """
        确保导航网格已加载，如果没有则重新计算
        Recompute the runtime navmesh when no compatible sidecar is available.
        """
        try:
            if not self.sim.pathfinder.is_loaded:
                print("警告: 导航网格未加载，正在重新计算...")
                navmesh_settings = habitat_sim.NavMeshSettings()
                navmesh_settings.set_defaults()
                
                # Use Habitat defaults consistently with offline top-down
                # preprocessing and SR/SPL evaluation. A different navmesh in
                # any stage changes floors, paths, and geodesic distances.
                
                # 重新计算导航网格
                success = self.sim.recompute_navmesh(self.sim.pathfinder, navmesh_settings)
                
                if success:
                    print("导航网格重新计算成功")
                    # 验证导航网格是否真的加载了
                    if self.sim.pathfinder.is_loaded:
                        navmesh_vertices = np.array(self.sim.pathfinder.build_navmesh_vertices())
                        print(f"导航网格顶点数: {len(navmesh_vertices)}")
                        if len(navmesh_vertices) > 0:
                            print("导航网格验证通过")
                        else:
                            print("警告: 导航网格加载但无顶点")
                    else:
                        print("警告: 导航网格计算成功但未正确加载")
                else:
                    print("错误: 导航网格重新计算失败")
                    print("可能的原因:")
                    print("  1. 场景文件格式不支持")
                    print("  2. 场景中没有可导航的区域")
                    print("  3. 导航网格参数设置不当")
            else:
                print("导航网格已存在，无需重新计算")
                
        except Exception as e:
            print(f"检查/重新计算导航网格时出错: {e}")
            print("提示: 可能需要检查场景文件是否正确或场景是否支持导航")
            # 提供更详细的错误信息
            import traceback
            print("详细错误信息:")
            traceback.print_exc()
    def setup_scene_and_agent(self, initial_state: Dict[str, Any], agent_state: Optional[Dict[str, Any]] = None):
            """
            设置场景和智能体。
            在设置智能体位置前，会先将其吸附到最近的可导航网格点。
            
            Args:
                initial_state: 传统初始状态配置，包含2D position和yaw rotation。
                agent_state: 新的3D初始状态配置，包含3D position和四元数rotation。
            """
            self.print_navmesh_info()
            
            snapped_position = None
            rotation_quat = None
            rotation_yaw = None
            
            # 确定使用哪种格式的初始状态，并进行吸附操作
            if agent_state is not None:
                print("使用3D agent_state格式初始化...")
                initial_position_3d = np.array(agent_state['position'], dtype=np.float32)
                rotation_quat = np.array(agent_state['rotation'], dtype=np.float32)
                
                # --- 新增：吸附到Navmesh ---
                snapped_position = self._snap_to_navmesh(initial_position_3d)
                if snapped_position is None:
                    raise RuntimeError(
                        "Unable to snap the 3-D initial position to the navmesh"
                    )
                
                self._generate_single_floor_topdown_map(snapped_position)
                
            else:
                print("使用传统initial_state格式初始化...")
                position_2d = initial_state['position']
                rotation_yaw = initial_state['rotation']
                
                # --- 修改：使用新的吸附逻辑 ---
                # 构造一个临时的3D点来进行吸附。Y值可以先用0或者场景中心Y值。
                # 我们先生成地图，这样就有场景中心了。
                self._generate_topdown_map() 
                temp_y = self.scene_center[1] if self.scene_center is not None else 0.0
                # 注意：initial_state的position通常是(x, z)
                temp_3d_pos = np.array([position_2d[0], temp_y, position_2d[1]], dtype=np.float32)
                
                snapped_position = self._snap_to_navmesh(temp_3d_pos)
                if snapped_position is None:
                    raise RuntimeError(
                        "Unable to snap the 2-D initial position to the navmesh"
                    )

                rotation_quat = self._yaw_to_quaternion(rotation_yaw)

            # 1. 生成topdown地图 (对于2D情况，已在上面完成)
            # (对于3D情况，已在上面完成)

            # 2. 加载物理机器人
            robot_urdf = self.config['scene']['robot_urdf']
            if not os.path.isfile(robot_urdf):
                raise FileNotFoundError(f"Robot URDF does not exist: {robot_urdf}")
            self._load_physical_robot(robot_urdf, snapped_position)
            
            # 3. 设置初始位置和朝向 (使用吸附后的位置)
            self.set_robot_pose(snapped_position, rotation_quat)
            
            if agent_state:
                print(f"智能体初始化到吸附后3D位置: {snapped_position}, 四元数旋转: {rotation_quat}")
            else:
                print(f"智能体初始化到吸附后位置: {snapped_position}, 朝向: {rotation_yaw}度")
    
    def _generate_topdown_map(self) -> None:
        """
        Generate the runtime top-down map and its world-coordinate metadata.
        使用临时模拟器来确保正确的ortho_scale
        """
        # 确保导航网格已加载
        if not self.sim.pathfinder.is_loaded:
            print("警告: 导航网格未加载，尝试重新计算...")
            self._ensure_navmesh_loaded()
            if not self.sim.pathfinder.is_loaded:
                raise RuntimeError(
                    "Cannot generate the top-down map without a navmesh"
                )
        
        scene_file = self.config['scene']['scene_file']
        
        # Step 1: calculate scene bounds with the active simulator.
        navmesh_vertices = np.array(self.sim.pathfinder.build_navmesh_vertices())
        if len(navmesh_vertices) == 0:
            print("错误: 无法获取导航网格顶点")
            return
        
        #x_min, x_max = navmesh_vertices[:, 0].min(), navmesh_vertices[:, 0].max()
        #z_min, z_max = navmesh_vertices[:, 2].min(), navmesh_vertices[:, 2].max()
        y_min, y_max = navmesh_vertices[:, 1].min(), navmesh_vertices[:, 1].max()
        scene_root_node = self.sim.get_active_scene_graph().get_root_node()
        scene_bb = scene_root_node.cumulative_bb
    
        x_min, x_max = scene_bb.min[0], scene_bb.max[0]
        z_min, z_max = scene_bb.min[2], scene_bb.max[2]
        scene_width = x_max - x_min
        scene_depth = z_max - z_min
        scene_size = max(scene_width, scene_depth)
        
        print(f"场景尺寸: 宽度={scene_width:.2f}m, 深度={scene_depth:.2f}m, 最大维度={scene_size:.2f}m")
        
        # Step 2: calculate an orthographic projection scale.
        target_coverage = 0.9
        optimal_ortho_scale = self._calculate_ortho_scale(scene_size, target_coverage)
        
        print(f"目标覆盖: {target_coverage:.1%}")
        print(f"计算的正交投影比例: {optimal_ortho_scale:.4f}")
        
        # 第三步：保存场景边界信息
        min_bounds = navmesh_vertices.min(axis=0).tolist()
        max_bounds = navmesh_vertices.max(axis=0).tolist()
        self.scene_bounds = (min_bounds, max_bounds)
        
        # 第四步：计算场景中心
        x_center = (x_min + x_max) / 2.0
        z_center = (z_min + z_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        self.scene_center = [x_center, y_center, z_center]
        
        print(f"场景中心: X={x_center:.2f}, Y={y_center:.2f}, Z={z_center:.2f}")
        
        # Step 5: create a dedicated orthographic simulator.
        ortho_sim = self._create_ortho_simulator(scene_file, optimal_ortho_scale)
        if ortho_sim is None:
            print("错误: 无法创建正交投影模拟器")
            return
        
        try:
            # 第六步：获取楼层信息
            ortho_navmesh_vertices = np.array(ortho_sim.pathfinder.build_navmesh_vertices())
            floor_extents = self._get_floor_extents(ortho_navmesh_vertices)
            floor_extents = sorted(floor_extents, key=lambda x: x['mean'])
            
            # Step 7: render each detected floor.
            floor_images = []
            for fext in floor_extents:
                # 过滤当前楼层的navmesh顶点
                mask = (
                    (ortho_navmesh_vertices[:, 1] <= fext['max'] + 0.25) & 
                    (ortho_navmesh_vertices[:, 1] >= fext['min'] - 0.25)
                )
                if mask.any():
                    fcent = np.median(ortho_navmesh_vertices[mask, :], axis=0).tolist()
                else:
                    fcent = [x_center, fext['mean'], z_center]
                
                # Set the orthographic sensor pose for this floor.
                agent_position = [x_center, fcent[1]+1.5, z_center]
                agent_rotation = np.array([-0.7071067, 0.0, 0.0, 0.7071067])  # get_downward_quaternion()
                
                ortho_agent = ortho_sim.get_agent(0)
                new_state = ortho_agent.get_state()
                new_state.position = agent_position
                new_state.rotation = agent_rotation
                new_state.sensor_states = {}
                ortho_agent.set_state(new_state, True)
                
                print(f"楼层 {len(floor_images)+1}: 相机位置 {agent_position}")
                
                # 获取观察 - 使用正交传感器
                obs = ortho_sim.get_sensor_observations()
                if 'ortho_sensor' in obs:
                    floor_images.append(obs['ortho_sensor'])
                elif 'rgba_camera' in obs:  # Compatibility with older sensor naming.
                    floor_images.append(obs['rgba_camera'])
                else:
                    print(f"错误: 未找到正交传感器，可用传感器: {list(obs.keys())}")
                    break
            
            # Step 8: compose the rendered floor images.
            if len(floor_images) > 1:
                floor_images = np.concatenate(floor_images, axis=0)
            elif len(floor_images) == 1:
                floor_images = floor_images[0]
            else:
                print("错误: 未能获取任何楼层图像")
                return
            
            # 第九步：保存为PIL图像
            self.base_map_image = Image.fromarray(floor_images[..., :3], "RGB")
            
            # Step 10: calculate the world-to-map projection metadata.
            self.map_width, self.map_height = self.base_map_image.size
            
            # 计算视野范围 - 基于正交投影参数
            current_view_size = 1.0 / (optimal_ortho_scale)
            view_half_width = current_view_size / 2.0
            view_half_height = view_half_width
            
            # 计算世界坐标覆盖范围
            world_coverage_x = view_half_width * 2
            world_coverage_z = view_half_height * 2
            
            # 保存转换参数
            self.world_to_map_scale_x = self.map_width / world_coverage_x
            self.world_to_map_scale_z = self.map_height / world_coverage_z
            self.view_range = (world_coverage_x, world_coverage_z)
            
            # 计算topdown地图边界（用于与occupancy map同步）
            self.topdown_spacing = world_coverage_x / self.map_width  # 米/像素
            self.topdown_map_bounds = {
                'top_left': (x_center - view_half_width, z_center - view_half_height),
                'bottom_right': (x_center + view_half_width, z_center + view_half_height),
                'view_range': (world_coverage_x, world_coverage_z),
                'image_size': (self.map_width, self.map_height)
            }
            
            print(f"地图尺寸: {self.map_width} x {self.map_height}")
            print(f"视野覆盖范围: {world_coverage_x:.2f}m x {world_coverage_z:.2f}m")
            print(f"坐标缩放因子: X={self.world_to_map_scale_x:.2f}, Z={self.world_to_map_scale_z:.2f}")
            print(f"像素间距: {self.topdown_spacing:.6f} m/pixel")
            
            print("topdown地图生成完成")
            
        finally:
            # 清理正交投影模拟器
            ortho_sim.close()
    
    def _generate_single_floor_topdown_map(self, target_position: np.ndarray) -> None:
        """
        为指定位置所在的楼层生成单独的topdown地图
        使用MultiFloorTopdownRenderer进行正确的多楼层渲染
        
        Args:
            target_position: 目标位置 [x, y, z]，用于确定渲染哪个楼层
        """
        print(f"为位置 {target_position} 生成单楼层topdown地图...")
        
        # 确保导航网格已加载
        if not self.sim.pathfinder.is_loaded:
            print("警告: 导航网格未加载，尝试重新计算...")
            self._ensure_navmesh_loaded()
            if not self.sim.pathfinder.is_loaded:
                raise RuntimeError(
                    "Cannot generate the single-floor map without a navmesh"
                )
        
        scene_file = self.config['scene']['scene_file']
        
        try:
            # 创建多楼层渲染器
            if self.multi_floor_renderer is None:
                self.multi_floor_renderer = MultiFloorTopdownRenderer(scene_file)
                projection = self.config.get('topdown_projection', {})
                if not isinstance(projection, dict):
                    raise ValueError("topdown_projection must be an object")
                self.multi_floor_renderer.analyze_scene(
                    custom_ortho_scale=projection.get('custom_ortho_scale'),
                    target_coverage=projection.get('target_coverage', 0.9),
                )
            
            # 渲染指定位置所在的楼层
            floor_image, unprojected_coords, meta_data = (
                self.multi_floor_renderer.render_floor_by_position(
                    target_position
                )
            )
            
            if floor_image is None:
                raise RuntimeError("Failed to render the requested floor")
            if not meta_data or not meta_data.get("selected_floor"):
                raise RuntimeError(
                    "Top-down renderer did not identify the selected floor"
                )
            self.topdown_floor = dict(meta_data["selected_floor"])
            
            # 保存为PIL图像
            self.base_map_image = Image.fromarray(floor_image[..., :3], "RGB")
            
            # 计算坐标转换参数
            self.map_width, self.map_height = self.base_map_image.size
            
            if unprojected_coords:
                # 计算视野覆盖范围
                view_width_meters, view_height_meters = unprojected_coords['view_range']
                
                # 保存转换参数
                self.world_to_map_scale_x = self.map_width / view_width_meters
                self.world_to_map_scale_z = self.map_height / view_height_meters
                self.view_range = (view_width_meters, view_height_meters)
                
                # 计算topdown地图边界（用于与occupancy map同步）
                self.topdown_spacing = view_width_meters / self.map_width  # 米/像素
                
                tl_x, tl_z = unprojected_coords['top_left']
                br_x, br_z = unprojected_coords['bottom_right']
                center_x, center_z = unprojected_coords['center']
                
                self.topdown_map_bounds = {
                    'top_left': (tl_x, tl_z),
                    'bottom_right': (br_x, br_z),
                    'view_range': (view_width_meters, view_height_meters),
                    'image_size': (self.map_width, self.map_height)
                }
                
                # 更新场景中心信息
                self.scene_center = [center_x, target_position[1], center_z]
                
                print(f"单楼层地图尺寸: {self.map_width} x {self.map_height}")
                print(f"视野覆盖范围: {view_width_meters:.2f}m x {view_height_meters:.2f}m")
                print(f"坐标缩放因子: X={self.world_to_map_scale_x:.2f}, Z={self.world_to_map_scale_z:.2f}")
                print(f"像素间距: {self.topdown_spacing:.6f} m/pixel")
                print(f"场景中心更新为: {self.scene_center}")
                
                print("单楼层topdown地图生成完成")
            else:
                print("警告: 无法获取坐标转换信息")
                
        except Exception as e:
            print(f"生成单楼层topdown地图失败: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError("Failed to generate the single-floor map") from e
    
    def _calculate_ortho_scale(self, scene_size: float, target_coverage: float = 0.9,safety_margin=1) -> float:
        """
        Calculate an orthographic scale that covers the scene bounds.
        
        Args:
            scene_size: 场景最大维度
            target_coverage: 目标覆盖范围
        
        Returns:
            正交投影比例
        """
        #base_scene_size = 20.0
        #base_ortho_scale = 0.05
        desired_view_size = scene_size / target_coverage * safety_margin
        calculated_scale = 1.0 / desired_view_size
        return max(0.01, calculated_scale)
    """
        calculated_scale = (base_ortho_scale * base_scene_size) / (scene_size / target_coverage)
        safety_margin = 1.2
        ortho_scale = calculated_scale / safety_margin
        return max(0.01, min(0.2, ortho_scale))
    """ 
    def _create_ortho_simulator(self, scene_path: str, ortho_scale: float):
        """
        Create the dedicated orthographic Habitat-Sim configuration.
        
        Args:
            scene_path: 场景文件路径
            ortho_scale: 正交投影比例
        
        Returns:
            正交投影模拟器实例
        """
        ortho_sim = None
        try:
            # 后端配置
            backend_cfg = habitat_sim.SimulatorConfiguration()
            backend_cfg.scene_id = scene_path
            
            # Orthographic sensor configuration.
            if habitat_sim.__version__ == "0.1.7":
                sensor_cfg = habitat_sim.SensorSpec()
                sensor_cfg.resolution = [4096, 4096]
                sensor_cfg.sensor_type = habitat_sim.SensorType.COLOR
                sensor_cfg.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
                sensor_cfg.parameters['far'] = '1000'
                sensor_cfg.parameters['near'] = '0.01'
                sensor_cfg.parameters['fov'] = '90'
                sensor_cfg.parameters['ortho_scale'] = str(ortho_scale)
            else:
                sensor_cfg = habitat_sim.CameraSensorSpec()
                sensor_cfg.uuid = "ortho_sensor"  # 确保UUID一致
                sensor_cfg.resolution = [4096, 4096]
                sensor_cfg.sensor_type = habitat_sim.SensorType.COLOR
                sensor_cfg.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
                sensor_cfg.far = 1000.0
                sensor_cfg.near = 0.01
                sensor_cfg.hfov = 90
                sensor_cfg.ortho_scale = ortho_scale
                sensor_cfg.clear_color = [0., 0., 0., 0.]
                sensor_cfg.position = [0.0, 0.0, 0.0] 
            
            # 智能体配置
            agent_cfg = habitat_sim.agent.AgentConfiguration()
            agent_cfg.sensor_specifications = [sensor_cfg]
            
            # 创建配置和模拟器
            cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
            ortho_sim = habitat_sim.Simulator(cfg)
            
            # Ensure a compatible navigation mesh is loaded.
            if not ortho_sim.pathfinder.is_loaded:
                navmesh_settings = habitat_sim.NavMeshSettings()
                navmesh_settings.set_defaults()
                ortho_sim.recompute_navmesh(ortho_sim.pathfinder, navmesh_settings)
            
            print(f"正交投影模拟器创建成功，ortho_scale={ortho_scale:.4f}")
            return ortho_sim
            
        except Exception as e:
            if ortho_sim is not None:
                ortho_sim.close()
            print(f"创建正交投影模拟器失败: {e}")
            return None
    
    def _get_floor_extents(self, navmesh_vertices):
        """
        获取楼层范围信息 - 简化版本的get_floor_navigable_extents
        """
        try:
            y_coords = navmesh_vertices[:, 1]
            y_range = y_coords.max() - y_coords.min()
            y_std = y_coords.std()
            
            print(f"Y坐标范围: {y_coords.min():.2f} 到 {y_coords.max():.2f} (范围: {y_range:.2f}m)")
            print(f"Y坐标标准差: {y_std:.3f}")
            
            # 简单的楼层检测
            if y_range < 1.5 and y_std < 0.3:
                # 单层建筑
                floor_extents = [{
                    'min': y_coords.min().item(),
                    'max': y_coords.max().item(),
                    'mean': y_coords.mean().item()
                }]
                print("检测为单层建筑")
            else:
                # 多层建筑 - 简单分层
                floor_extents = []
                # 使用四分位数进行粗略分层
                percentiles = [0, 33, 66, 100]
                for i in range(len(percentiles) - 1):
                    start_pct = percentiles[i]
                    end_pct = percentiles[i + 1]
                    
                    start_y = np.percentile(y_coords, start_pct)
                    end_y = np.percentile(y_coords, end_pct)
                    
                    if end_y - start_y > 0.5:  # 至少0.5米高才算一层
                        floor_extents.append({
                            'min': start_y.item(),
                            'max': end_y.item(),
                            'mean': ((start_y + end_y) / 2).item()
                        })
                
                print(f"检测为多层建筑，共{len(floor_extents)}层")
            
            for i, fext in enumerate(floor_extents):
                print(f"  楼层 {i+1}: Y范围 [{fext['min']:.2f}, {fext['max']:.2f}], 平均 {fext['mean']:.2f}")
            
            return floor_extents
            
        except Exception as e:
            print(f"楼层检测失败: {e}")
            # 返回默认单楼层
            y_coords = navmesh_vertices[:, 1]
            return [{
                'min': y_coords.min().item(),
                'max': y_coords.max().item(),
                'mean': y_coords.mean().item()
            }]
    
    def _load_physical_robot(self, model_path: str, initial_position: np.ndarray) -> Any:
        """
        加载物理机器人URDF模型
        
        Args:
            model_path: URDF文件路径
            initial_position: 初始3D位置
        
        Returns:
            机器人对象或None
        """
        try:
            # 获取关节对象管理器
            articulated_obj_mgr = self.sim.get_articulated_object_manager()
            
            # 从URDF文件加载机器人
            self.robot_object = articulated_obj_mgr.add_articulated_object_from_urdf(
                filepath=model_path,
                fixed_base=False,  # 允许机器人移动
                global_scale=1.0,
                mass_scale=1.0
            )
            if self.robot_object is None:
                raise RuntimeError("Habitat-Sim returned no articulated object")
            
            # 设置初始位置
            robot_initial_transform = mn.Matrix4.translation(mn.Vector3(
                initial_position[0], initial_position[1], initial_position[2]
            ))
            self.robot_object.transformation = robot_initial_transform
            
            print(f"物理机器人加载成功: {model_path}")
            return self.robot_object
            
        except Exception as exc:
            self.robot_object = None
            raise RuntimeError(
                f"Failed to load physical robot URDF: {model_path}"
            ) from exc
    
    def _convert_2d_to_3d(self, x: float, z: float) -> Optional[np.ndarray]:
        """
        将2D坐标转换为3D坐标，通过navmesh获取Y坐标
        
        Args:
            x: X坐标
            z: Z坐标
        
        Returns:
            3D坐标[x, y, z]或None（如果不可导航）
        """
        y = self.get_navigable_y(x, z)
        if y is not None:
            return np.array([x, y, z], dtype=np.float32)
        return None
    
    def get_navigable_y(self, x: float, z: float) -> Optional[float]:
        """
        获取指定(x,z)位置对应的可导航Y坐标
        
        Args:
            x: X坐标
            z: Z坐标
        
        Returns:
            Y坐标或None（如果不可导航）
        """
        try:
            # 确保导航网格已加载
            if not self.sim.pathfinder.is_loaded:
                print("警告: 导航网格未加载，尝试重新计算...")
                self._ensure_navmesh_loaded()
                if not self.sim.pathfinder.is_loaded:
                    print("错误: 无法加载导航网格")
                    return None
            
            # 获取当前agent的y坐标作为参考
            current_agent_state = self.agent.get_state()
            current_y = current_agent_state.position[1]
            
            # 使用当前agent的y坐标进行测试
            test_point = mn.Vector3(x, current_y, z)
            snapped_point = self.sim.pathfinder.snap_point(test_point)
            
            if self.sim.pathfinder.is_navigable(snapped_point):
                return float(snapped_point.y)
            else:
                # 如果不可导航，直接返回None
                return None
                
        except Exception as e:
            print(f"获取导航高度失败: {e}")
            return None
    """    def check_straight_path_collision(self, start_pos: np.ndarray, end_pos: np.ndarray, 
                                    step_size: float = 0.1) -> bool:
        try:
            # 确保导航网格已加载
            if not self.sim.pathfinder.is_loaded:
                print("警告: 导航网格未加载，尝试重新计算...")
                self._ensure_navmesh_loaded()
                if not self.sim.pathfinder.is_loaded:
                    print("错误: 无法加载导航网格，无法进行碰撞检测")
                    return True  # 保守策略，无法检测时认为会碰撞
            
            # 计算路径方向和总距离
            direction = end_pos - start_pos
            total_distance = np.linalg.norm(direction)
            
            if total_distance == 0:
                return False
            
            # 归一化方向向量
            direction = direction / total_distance
            
            # 计算检测步数
            num_steps = max(1, int(total_distance / step_size))
            
            # 逐步检查路径
            for i in range(1, num_steps + 1):
                # 计算当前检测点
                progress = min(i * step_size, total_distance)
                current_pos = start_pos + direction * progress
                next_pos = start_pos + direction * min((i + 1) * step_size, total_distance)
                
                # 使用pathfinder检查这一步是否可行
                current_point = mn.Vector3(current_pos[0], current_pos[1], current_pos[2])
                next_point = mn.Vector3(next_pos[0], next_pos[1], next_pos[2])
                
                # 尝试从当前点移动到下一点
                filtered_end = self.sim.pathfinder.try_step(current_point, next_point)
                
                # 计算实际移动距离和期望移动距离
                expected_move = np.linalg.norm(next_pos - current_pos)
                actual_move = np.linalg.norm(
                    [filtered_end.x - current_pos[0], 
                     filtered_end.y - current_pos[1], 
                     filtered_end.z - current_pos[2]]
                )
                
                # 如果实际移动距离明显小于期望距离，说明遇到障碍物
                if actual_move + 1e-5 < expected_move:
                    return True
            
            return False
            
        except Exception as e:
            print(f"碰撞检测失败: {e}")
            return True  # 保守策略，检测失败时认为会碰撞"""

    
    def set_robot_pose(self, position: np.ndarray, rotation_quat: np.ndarray):
        """
        设置机器人姿态（同时更新物理机器人和虚拟智能体）
        
        Args:
            position: 3D位置 [x, y, z]
            rotation_quat: 四元数旋转 [x, y, z, w]
        """
        position = np.asarray(position, dtype=np.float32)
        rotation_quat = np.asarray(rotation_quat, dtype=np.float32)
        if position.shape != (3,):
            raise ValueError(f"position must have shape (3,), got {position.shape}")
        if rotation_quat.shape != (4,):
            raise ValueError(
                f"rotation_quat must have shape (4,), got {rotation_quat.shape}"
            )
        if not np.all(np.isfinite(position)):
            raise ValueError("position must contain only finite values")
        if not np.all(np.isfinite(rotation_quat)):
            raise ValueError("rotation_quat must contain only finite values")
        if self.agent is None:
            raise RuntimeError(
                "Cannot set robot pose before the Habitat agent is initialized"
            )

        original_agent_state = None
        original_robot_transform = None
        robot_pose_changed = False
        agent_pose_changed = False

        try:
            # Save both representations before changing either one so a
            # partial update can be rolled back instead of leaving them out of
            # sync.
            original_agent_state = self.agent.get_state()
            if self.robot_object is not None:
                original_robot_transform = self.robot_object.transformation

            # 更新物理机器人
            if self.robot_object is not None:
                pos_vec = mn.Vector3(position[0], position[1], position[2])
                quat = convert_to_magnum_quat(rotation_quat)
                
                # 创建变换矩阵
                rotation_matrix = mn.Matrix4.from_(quat.to_matrix(), mn.Vector3())
                translation_matrix = mn.Matrix4.translation(pos_vec)
                
                # 设置机器人变换
                self.robot_object.transformation = (
                    translation_matrix @ rotation_matrix
                )
                robot_pose_changed = True
            
            # 同步更新虚拟智能体（用于传感器）
            agent_state = habitat_sim.AgentState()
            agent_state.position = position
            agent_state.rotation = rotation_quat
            self.agent.set_state(agent_state)
            agent_pose_changed = True
            
        except Exception as e:
            rollback_errors = []
            if original_agent_state is not None:
                try:
                    self.agent.set_state(original_agent_state)
                except Exception as rollback_error:
                    rollback_errors.append(
                        f"agent rollback failed: {rollback_error}"
                    )
            if robot_pose_changed and original_robot_transform is not None:
                try:
                    self.robot_object.transformation = original_robot_transform
                except Exception as rollback_error:
                    rollback_errors.append(
                        f"robot rollback failed: {rollback_error}"
                    )

            message = "Failed to set robot pose"
            if rollback_errors:
                message += "; " + "; ".join(rollback_errors)
            elif robot_pose_changed or agent_pose_changed:
                message += "; previous pose was restored"
            raise RuntimeError(message) from e
    
    def get_fpv_observation(self) -> np.ndarray:
        """
        获取第一人称视角观察
        
        Returns:
            RGB图像数组
        """
        # 获取机器人状态
        robot_state = self.get_robot_state()
        robot_pos = robot_state['position']
        robot_rot = robot_state['rotation']
        
        # 设置第三人称视角相机（在机器人后方和上方）
        camera_distance = 3.0  # 相机距离机器人的距离
        camera_height = 2.0    # 相机高度
        
        # 计算相机位置（在机器人后方）
        robot_forward = self._get_forward_vector(robot_rot)
        camera_offset = -robot_forward * camera_distance + np.array([0, camera_height, 0])
        camera_pos = robot_pos + camera_offset
        
        # 设置相机朝向机器人
        camera_look_at = robot_pos
        
        # 设置相机参数
        camera_fov = 60.0  # 视野角度
        camera_aspect = 1.0  # 宽高比
        
        # 创建观察
        observation = self.sim.get_observations_at(
            position=camera_pos,
            rotation=robot_rot,
            keep_agent_at_new_pose=False
        )
        
        if 'rgb' in observation:
            return observation['rgb']
        raise RuntimeError("Third-person observation did not contain an rgb frame")
    
    def _get_robot_sensor_observation(self) -> np.ndarray:
        """从物理机器人的传感器位置获取观察"""
        try:
            # 获取机器人当前变换
            robot_transform = self.robot_object.transformation
            robot_position = robot_transform.translation
            robot_rotation = mn.Quaternion.from_matrix(robot_transform.rotation())
            
            # 计算传感器位置（只有物理机器人时才添加传感器高度偏移）
            sensor_offset = mn.Vector3(0, self.config['agent']['sensor_height'], 0)
            sensor_position = robot_position + robot_transform.transform_vector(sensor_offset)
            
            # 临时设置虚拟智能体到传感器位置
            temp_agent_state = habitat_sim.AgentState()
            temp_agent_state.position = np.array([sensor_position.x, sensor_position.y, sensor_position.z])
            temp_agent_state.rotation = convert_to_numpy_quat(robot_rotation)
            
            observations = self._get_observations_at_temporary_agent_state(
                temp_agent_state
            )
            return observations["color_sensor"]

        except _AgentStateRestoreError:
            # Continuing after a failed rollback could make all subsequent
            # navigation operate from the sensor pose rather than robot pose.
            raise
        except Exception as exc:
            raise RuntimeError(
                "Failed to read the RGB observation at the robot sensor pose"
            ) from exc

    def _get_observations_at_temporary_agent_state(
        self, temp_agent_state: habitat_sim.AgentState
    ) -> Dict[str, np.ndarray]:
        """Observe from a temporary pose and always restore the agent state."""
        original_state = self.agent.get_state()
        try:
            self.agent.set_state(temp_agent_state)
            return self.sim.get_sensor_observations()
        finally:
            try:
                self.agent.set_state(original_state)
            except Exception as restore_error:
                raise _AgentStateRestoreError(
                    "Failed to restore the Habitat agent after a temporary "
                    "sensor observation; simulator state is unsafe to continue"
                ) from restore_error
    
    def get_observation(self) -> Dict[str, np.ndarray]:
        """
        获取完整的传感器观测数据，包括RGB和深度
        
        Returns:
            包含 'rgb' 和 'depth' 键的观测字典
        """
        try:
            # 如果有物理机器人，从其传感器位置获取观察
            if self.robot_object is not None:
                return self._get_robot_full_observation()
            else:
                # 否则直接从虚拟智能体获取
                observations = self.sim.get_sensor_observations()
                return {
                    'rgb': observations["color_sensor"],
                    'depth': observations["depth_sensor"]
                }
                
        except _AgentStateRestoreError:
            raise
        except Exception as e:
            raise RuntimeError("Failed to read RGB-D observation") from e
    
    def _get_robot_full_observation(self) -> Dict[str, np.ndarray]:
        """从物理机器人的传感器位置获取完整观察"""
        try:
            # 获取机器人当前变换
            robot_transform = self.robot_object.transformation
            robot_position = robot_transform.translation
            robot_rotation = mn.Quaternion.from_matrix(robot_transform.rotation())
            
            # 计算传感器位置
            sensor_offset = mn.Vector3(0, self.config['agent']['sensor_height'], 0)
            sensor_position = robot_position + robot_transform.transform_vector(sensor_offset)
            
            # 临时设置虚拟智能体到传感器位置
            temp_agent_state = habitat_sim.AgentState()
            temp_agent_state.position = np.array([sensor_position.x, sensor_position.y, sensor_position.z])
            temp_agent_state.rotation = convert_to_numpy_quat(robot_rotation)
            
            observations = self._get_observations_at_temporary_agent_state(
                temp_agent_state
            )

            return {
                'rgb': observations["color_sensor"],
                'depth': observations["depth_sensor"]
            }

        except _AgentStateRestoreError:
            raise
        except Exception as exc:
            raise RuntimeError(
                "Failed to read RGB-D at the physical robot sensor pose"
            ) from exc
    
    def get_robot_state(self) -> Dict[str, np.ndarray]:
        """
        获取机器人当前状态
        
        Returns:
            包含position和rotation的字典
        """
        try:
            if self.robot_object is not None:
                # 从物理机器人获取状态
                transform = self.robot_object.transformation
                position = transform.translation
                rotation_quat = mn.Quaternion.from_matrix(transform.rotation())
                
                return {
                    'position': np.array([position.x, position.y, position.z]),
                    'rotation': convert_to_numpy_quat(rotation_quat)
                }
            else:
                # 从虚拟智能体获取状态
                agent_state = self.agent.get_state()
                return {
                    'position': agent_state.position,
                    'rotation': agent_state.rotation
                }
                
        except Exception as e:
            raise RuntimeError("Failed to read the current robot state") from e
    
    def get_base_map(self) -> Optional[Image.Image]:
        """
        获取基础topdown地图
        
        Returns:
            PIL图像对象
        """
        return self.base_map_image

    def get_topdown_metadata(self) -> Optional[Dict]:
        """
        获取topdown地图的元数据，用于与occupancy map同步坐标系
        
        Returns:
            包含场景中心、地图边界、像素间距等信息的字典
        """
        if not hasattr(self, 'topdown_map_bounds') or not hasattr(self, 'scene_center'):
            return None
            
        return {
            'scene_path': self.config['scene']['scene_file'],
            'selected_floor': self.topdown_floor,
            'scene_center': self.scene_center,
            'map_bounds': self.topdown_map_bounds,
            'spacing': self.topdown_spacing,
            'map_size': (self.map_width, self.map_height)
        }
    
    def world_to_map_coords(self, world_pos: np.ndarray) -> Tuple[int, int]:
        """
        将3D世界坐标转换为2D地图像素坐标
        Convert Habitat world coordinates into canonical top-down pixels.
        
        Args:
            world_pos: 世界坐标 [x, y, z]
        
        Returns:
            地图像素坐标 (map_x, map_y)
        """
        if self.base_map_image is None or not hasattr(self, 'view_range'):
            raise RuntimeError("Top-down map coordinates are not initialized")
        
        # 获取场景中心
        world_center_x = self.scene_center[0]
        world_center_z = self.scene_center[2]
        # 计算相对于场景中心的坐标
        rel_x = world_pos[0] - world_center_x
        rel_z = world_pos[2] - world_center_z
        
        # 计算视野范围
        view_half_width = self.view_range[0] / 2.0
        view_half_height = self.view_range[1] / 2.0
        
        # Convert normalized coordinates to map pixels.
        # 地图中心对应图像中心
        pixel_x = self.map_width / 2 + (rel_x / view_half_width) * (self.map_width / 2)
        # Z轴方向是从上到下增加
        pixel_y = self.map_height / 2 + (rel_z / view_half_height) * (self.map_height / 2)
        
        # 转换为整数像素坐标并确保在范围内
        map_x = max(0, min(int(pixel_x), self.map_width - 1))
        map_y = max(0, min(int(pixel_y), self.map_height - 1))
        
        return (map_x, map_y)
    
    def _yaw_to_quaternion(self, yaw_degrees: float) -> np.ndarray:
        """
        将偏航角（度）转换为四元数
        偏航角是绕Y轴（垂直/高度轴）的旋转
        
        Args:
            yaw_degrees: 偏航角（度）
        
        Returns:
            四元数 [x, y, z, w]
        """
        yaw_rad = math.radians(yaw_degrees)

        # 绕Y轴旋转的四元数
        return np.array([
            0.0,
            math.sin(yaw_rad / 2.0),  # Y分量
            0.0,
            math.cos(yaw_rad / 2.0)
        ], dtype=np.float32)
    
    def close(self):
        """关闭模拟器"""
        if self.sim:
            self.sim.close()
            self.sim = None
            print("模拟器已关闭")
    
    def get_navmesh_info(self) -> Dict[str, Any]:
        """
        获取导航网格信息
        
        Returns:
            包含导航网格状态和统计信息的字典
        """
        try:
            info = {
                'is_loaded': self.sim.pathfinder.is_loaded,
                'navmesh_vertices_count': 0,
                'navigable_area': 0.0,
                'bounds': None
            }
            
            if self.sim.pathfinder.is_loaded:
                # 获取导航网格顶点
                navmesh_vertices = np.array(self.sim.pathfinder.build_navmesh_vertices())
                info['navmesh_vertices_count'] = len(navmesh_vertices)
                
                if len(navmesh_vertices) > 0:
                    # 计算边界
                    min_bounds = navmesh_vertices.min(axis=0)
                    max_bounds = navmesh_vertices.max(axis=0)
                    info['bounds'] = {
                        'min': min_bounds.tolist(),
                        'max': max_bounds.tolist(),
                        'size': (max_bounds - min_bounds).tolist()
                    }
                    
                    # 估算可导航面积（粗略计算）
                    x_range = max_bounds[0] - min_bounds[0]
                    z_range = max_bounds[2] - min_bounds[2]
                    info['navigable_area'] = x_range * z_range
            
            return info
            
        except Exception as e:
            print(f"获取导航网格信息失败: {e}")
            return {
                'is_loaded': False,
                'navmesh_vertices_count': 0,
                'navigable_area': 0.0,
                'bounds': None,
                'error': str(e)
            }
    
    def print_navmesh_info(self):
        """打印导航网格信息"""
        info = self.get_navmesh_info()
        print("=== 导航网格信息 ===")
        print(f"是否已加载: {info['is_loaded']}")
        print(f"顶点数量: {info['navmesh_vertices_count']}")
        print(f"可导航面积: {info['navigable_area']:.2f} 平方米")
        
        if info['bounds']:
            bounds = info['bounds']
            print(f"边界范围:")
            print(f"  X: {bounds['min'][0]:.2f} ~ {bounds['max'][0]:.2f} (宽度: {bounds['size'][0]:.2f}m)")
            print(f"  Y: {bounds['min'][1]:.2f} ~ {bounds['max'][1]:.2f} (高度: {bounds['size'][1]:.2f}m)")
            print(f"  Z: {bounds['min'][2]:.2f} ~ {bounds['max'][2]:.2f} (深度: {bounds['size'][2]:.2f}m)")
        
        if 'error' in info:
            print(f"错误信息: {info['error']}")
        
        print("=" * 25)
    
    def force_recompute_navmesh(self, custom_settings: Optional[Dict[str, Any]] = None) -> bool:
        """
        强制重新计算导航网格
        
        Args:
            custom_settings: 自定义navmesh设置字典，可包含以下键:
                - cell_size: 网格单元大小（米）
                - cell_height: 网格单元高度（米）
                - agent_height: 智能体高度（米）
                - agent_radius: 智能体半径（米）
                - agent_max_climb: 最大攀爬高度（米）
                - agent_max_slope: 最大坡度（度）
        
        Returns:
            是否成功重新计算导航网格
        """
        try:
            print("强制重新计算导航网格...")
            navmesh_settings = habitat_sim.NavMeshSettings()
            navmesh_settings.set_defaults()
            
            # 应用自定义设置
            if custom_settings:
                if 'cell_size' in custom_settings:
                    navmesh_settings.cell_size = custom_settings['cell_size']
                if 'cell_height' in custom_settings:
                    navmesh_settings.cell_height = custom_settings['cell_height']
                if 'agent_height' in custom_settings:
                    navmesh_settings.agent_height = custom_settings['agent_height']+0.1
                if 'agent_radius' in custom_settings:
                    navmesh_settings.agent_radius = custom_settings['agent_radius']
                if 'agent_max_climb' in custom_settings:
                    navmesh_settings.agent_max_climb = custom_settings['agent_max_climb']
                if 'agent_max_slope' in custom_settings:
                    navmesh_settings.agent_max_slope = custom_settings['agent_max_slope']
                
                print("使用自定义导航网格设置:")
                for key, value in custom_settings.items():
                    print(f"  {key}: {value}")
            
            # 重新计算导航网格
            success = self.sim.recompute_navmesh(self.sim.pathfinder, navmesh_settings)
            
            if success:
                print("导航网格强制重新计算成功")
                self.print_navmesh_info()  # 打印新的navmesh信息
                return True
            else:
                print("导航网格强制重新计算失败")
                return False
                
        except Exception as e:
            print(f"强制重新计算导航网格失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_forward_vector(self, rotation_quat: np.ndarray) -> np.ndarray:
        """
        从四元数获取前向向量
        
        Args:
            rotation_quat: 四元数旋转
            
        Returns:
            前向向量 [x, y, z]
        """
        # 使用Magnum四元数转换
        quat = convert_to_magnum_quat(rotation_quat)
        
        # 在Habitat中，-Z轴是前方
        forward_vec = quat.transform_vector(mn.Vector3(0, 0, -1))
        
        return np.array([forward_vec.x, forward_vec.y, forward_vec.z])
