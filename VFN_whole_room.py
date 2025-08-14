import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import heapq
from PIL import Image
# --- 模拟环境和传感器参数 ---

GRID_RESOLUTION = 0.2 # 稍微增大分辨率以加快可视化的绘制

SENSOR_MAX_RANGE = 5.0
SENSOR_FOV = np.deg2rad(90)
NUM_SCAN_RAYS = 45 # 减少光线数量以提高模拟速度

# (MODIFICATION 1: 新增地图状态常量)
MAP_UNKNOWN = -1
MAP_FREE = 0
MAP_OCCUPIED = 1


# --- 算法参数 ---
MU1, MU2, MU3 = 5, 2, 2
MU1_PRIME, MU2_PRIME, MU3_PRIME = 5, 1, 1
LAMBDA = 0.8
ROBOT_RADIUS = 0.14
SENSOR_RANGE_VFH = 0.4 # VFH*算法从记忆地图中提取障碍物的半径
HISTOGRAM_ALPHA = np.deg2rad(5)
SMAX = 16
NUM_HISTOGRAM_BINS = int(2 * np.pi / HISTOGRAM_ALPHA)
DS = ROBOT_RADIUS * 0.5
ALIGNMENT_TOLERANCE = np.deg2rad(17)
DISCRETE_ACTIONS = {"TURN_LEFT_30": np.deg2rad(30), "TURN_RIGHT_30": -np.deg2rad(30)}

# --- 辅助函数 ---
def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def world_to_grid(x, y, resolution):
    col = int(x / resolution)
    row = int(y / resolution)
    return row, col

def grid_to_world(row, col, resolution):
    x = (col + 0.5) * resolution
    y = (row + 0.5) * resolution
    return x, y

# (新函数) 从图像文件创建世界地图
def create_world_from_image(image_path, image_pixel_spacing_meters, grid_resolution_meters):
    """
    从一个黑白图像加载地图。

    参数:
    - image_path (str): 图像文件的路径。
    - image_pixel_spacing_meters (float): 原始图像中每个像素代表的真实世界距离（米）。
    - grid_resolution_meters (float): 我们模拟中占据栅格地图的分辨率（米/像素）。

    返回:
    - global_grid (np.array): 模拟用的占据栅格地图。
    - world_size_meters (tuple): (宽, 高)，以米为单位的整个世界的尺寸。
    """
    try:
        # 打开图像并转换为灰度模式
        img = Image.open(image_path).convert('L')
        # 将图像数据转为numpy数组
        img_array = np.array(img)
    except FileNotFoundError:
        print(f"错误：找不到图像文件 '{image_path}'。请确保文件路径正确。")
        exit()

    # 将图像坐标系(左上角为原点)翻转为世界坐标系(左下角为原点)
    img_array = np.flipud(img_array)

    img_height_pixels, img_width_pixels = img_array.shape

    # 1. 根据图像尺寸和spacing计算真实世界的总尺寸
    world_width_meters = img_width_pixels * image_pixel_spacing_meters
    world_height_meters = img_height_pixels * image_pixel_spacing_meters
    world_size_meters = (world_width_meters, world_height_meters)

    # 2. 根据世界总尺寸和栅格分辨率，计算我们的模拟栅格地图的尺寸
    grid_shape = (int(world_height_meters / grid_resolution_meters),
                  int(world_width_meters / grid_resolution_meters))

    # 3. 创建空的栅格地图
    global_grid = np.zeros(grid_shape, dtype=int)

    # 4. 填充栅格地图
    # 遍历我们模拟地图的每一个栅格
    for r in range(grid_shape[0]):
        for c in range(grid_shape[1]):
            # 计算该栅格中心点的真实世界坐标 (x, y)
            x, y = grid_to_world(r, c, grid_resolution_meters)

            # 将真实世界坐标 (x, y) 转换回原始图像的像素坐标
            img_px = int(x / image_pixel_spacing_meters)
            img_py = int(y / image_pixel_spacing_meters)

            # 确保坐标在图像边界内
            img_px = max(0, min(img_width_pixels - 1, img_px))
            img_py = max(0, min(img_height_pixels - 1, img_py))

            # 读取原始图像对应像素的颜色值 (灰度, 0=黑, 255=白)
            pixel_value = img_array[img_py, img_px]

            # 如果像素是黑色或深灰色 (这里用128作为阈值)，则认为是障碍物
            if pixel_value < 128:
                global_grid[r, c] = 1 # 1 代表障碍物

    print(f"成功从 '{image_path}' 加载地图。")
    print(f"真实世界尺寸: {world_size_meters[0]:.2f}m x {world_size_meters[1]:.2f}m")
    print(f"模拟栅格尺寸: {global_grid.shape[1]} x {global_grid.shape[0]} pixels")

    return global_grid, world_size_meters

def create_global_grid(obstacles, world_size, resolution):
    grid_shape = (int(world_size[1] / resolution), int(world_size[0] / resolution))
    grid = np.zeros(grid_shape)
    for r in range(grid_shape[0]):
        for c in range(grid_shape[1]):
            x, y = grid_to_world(r, c, resolution)
            for (ox, oy, orad) in obstacles:
                if np.sqrt((x - ox)**2 + (y - oy)**2) < orad:
                    grid[r, c] = 1
                    break
    return grid

# (MODIFICATION 2: 重构传感器函数，现在它负责更新记忆地图)
def update_memory_map_with_scan(robot, global_grid, memory_map, resolution):
    """
    模拟传感器扫描并更新机器人的内部记忆地图 (Ray Casting)。
    """
    robot_row, robot_col = world_to_grid(robot.x, robot.y, resolution)
    start_angle = robot.theta - SENSOR_FOV / 2
    end_angle = robot.theta + SENSOR_FOV / 2
    
    # 遍历每一条扫描光线
    for ray_angle in np.linspace(start_angle, end_angle, NUM_SCAN_RAYS):
        # 沿着光线方向步进，检查栅格
        for step in np.arange(0, SENSOR_MAX_RANGE, resolution):
            dist_x = step * np.cos(ray_angle)
            dist_y = step * np.sin(ray_angle)
            
            # 计算当前检查点的栅格坐标
            check_row, check_col = world_to_grid(robot.x + dist_x, robot.y + dist_y, resolution)

            # 检查是否越界
            if not (0 <= check_row < memory_map.shape[0] and 0 <= check_col < memory_map.shape[1]):
                break # 光线射出地图边界

            # 查询真实世界地图
            if global_grid[check_row, check_col] == 1:
                # 遇到了障碍物，将此栅格在记忆地图中标记为OCCUPIED
                memory_map[check_row, check_col] = MAP_OCCUPIED
                break # 光线被阻挡，停止这条光线的步进
            else:
                # 没遇到障碍物，是自由空间，在记忆地图中标记为FREE
                memory_map[check_row, check_col] = MAP_FREE

# (MODIFICATION 3: 新增函数，从记忆地图中为VFH*提取局部障碍物)
def get_obstacles_from_memory(robot, memory_map, local_radius, resolution):
    """
    从机器人的记忆地图中，提取其周围指定半径内的所有已知障碍物。
    """
    obstacles = []
    robot_row, robot_col = world_to_grid(robot.x, robot.y, resolution)
    radius_in_grid = int(local_radius / resolution)

    # 遍历机器人周围的局部区域
    for r_offset in range(-radius_in_grid, radius_in_grid + 1):
        for c_offset in range(-radius_in_grid, radius_in_grid + 1):
            r, c = robot_row + r_offset, robot_col + c_offset

            # 检查边界
            if not (0 <= r < memory_map.shape[0] and 0 <= c < memory_map.shape[1]):
                continue
            
            # 如果记忆地图中该位置是障碍物
            if memory_map[r, c] == MAP_OCCUPIED:
                ox, oy = grid_to_world(r, c, resolution)
                # 确保只考虑半径范围内的点
                if np.sqrt((ox - robot.x)**2 + (oy - robot.y)**2) <= local_radius:
                    obstacles.append((ox, oy, resolution / 2)) # 等效半径为栅格的一半
    
    return obstacles


# --- Robot 和 VFHStar 类保持不变 (但VFH*的调用方式会变) ---
class Robot:
    def __init__(self, x, y, theta, speed=1.5):
        self.x, self.y, self.theta = x, y, theta
        self.speed = speed
        self.path = [(x, y)]
        self.prev_selected_direction = self.theta
        self.is_stopped = False
    def rotate(self, direction_change, dt=0.1):
        if self.is_stopped: return
        self.theta = normalize_angle(self.theta + direction_change)
        self.path.append((self.x, self.y))
        self.prev_selected_direction = self.theta
    def walk_forward(self, dt=1):
        if self.is_stopped: return
        self.x += self.speed * np.cos(self.theta) * dt
        self.y += self.speed * np.sin(self.theta) * dt
        self.path.append((self.x, self.y))
class VFHStar:
    def __init__(self, target):
        self.target = target
        self.ax_hist, self.ax_tree = None, None
    def get_best_direction(self, robot, ng, perceived_obstacles):
        # 这个函数本身逻辑不变，但传入的perceived_obstacles来源变了
        primary_candidates = self._get_candidate_directions(robot.x, robot.y, perceived_obstacles)
        if not primary_candidates: return None
        if len(primary_candidates) == 1 or ng == 1:
            costs = [self._cost_g0(c, robot) for c in primary_candidates]
            return primary_candidates[np.argmin(costs)]
        return self._a_star_search(robot, primary_candidates, ng, perceived_obstacles)
    def _get_polar_histogram(self, x, y, obstacles):
        histogram = np.zeros(NUM_HISTOGRAM_BINS)
        for ox, oy, orad in obstacles:
            dx, dy = ox - x, oy - y
            dist = np.sqrt(dx**2 + dy**2)
            if dist < SENSOR_RANGE_VFH:
                angle = np.arctan2(dy, dx)
                gamma = np.arcsin(min(1.0, (ROBOT_RADIUS + orad) / dist))
                start_angle = normalize_angle(angle - gamma)
                end_angle = normalize_angle(angle + gamma)
                start_bin = int(normalize_angle(start_angle) / HISTOGRAM_ALPHA + NUM_HISTOGRAM_BINS/2) % NUM_HISTOGRAM_BINS
                end_bin = int(normalize_angle(end_angle) / HISTOGRAM_ALPHA + NUM_HISTOGRAM_BINS/2) % NUM_HISTOGRAM_BINS
                curr = start_bin
                while curr != end_bin:
                    histogram[curr] = 1.0
                    curr = (curr + 1) % NUM_HISTOGRAM_BINS
                histogram[end_bin] = 1.0
        return histogram
    def _get_candidate_directions(self, x, y, obstacles):
        histogram = self._get_polar_histogram(x, y, obstacles)
        if np.all(histogram == 0):
            target_angle = np.arctan2(self.target[1] - y, self.target[0] - x)
            return [normalize_angle(target_angle)]
        free_bins = np.where(histogram == 0)[0]
        if len(free_bins) == 0: return []
        diffs = np.diff(free_bins); split_indices = np.where(diffs > 1)[0] + 1
        valleys_bins = np.split(free_bins, split_indices)
        if len(valleys_bins) > 1 and free_bins[0] == 0 and free_bins[-1] == NUM_HISTOGRAM_BINS - 1:
            valleys_bins[-1] = np.concatenate((valleys_bins[-1], valleys_bins[0])); valleys_bins.pop(0)
        candidates, target_angle = [], np.arctan2(self.target[1] - y, self.target[0] - x)
        for valley in valleys_bins:
            if len(valley) == 0: continue
            if len(valley) < SMAX: candidates.append(normalize_angle((valley[len(valley) // 2] - NUM_HISTOGRAM_BINS/2) * HISTOGRAM_ALPHA))
            else:
                safe_margin = SMAX // 2
                candidates.append(normalize_angle((valley[safe_margin] - NUM_HISTOGRAM_BINS/2) * HISTOGRAM_ALPHA))
                candidates.append(normalize_angle((valley[-1 - safe_margin] - NUM_HISTOGRAM_BINS/2) * HISTOGRAM_ALPHA))
                target_bin = int(normalize_angle(target_angle) / HISTOGRAM_ALPHA + NUM_HISTOGRAM_BINS/2) % NUM_HISTOGRAM_BINS
                if target_bin in valley: candidates.append(normalize_angle(target_angle))
        return list(set(candidates))
    def _a_star_search(self, robot, primary_candidates, ng, perceived_obstacles):
        open_set = []
        for cand in primary_candidates:
            g = self._cost_g0(cand, robot); h = self._heuristic_h(cand, robot.theta, robot.prev_selected_direction, 1)
            new_pos, new_theta = self._project_robot(robot.x, robot.y, robot.theta, cand)
            heapq.heappush(open_set, (g + h, g, 1, (new_pos[0], new_pos[1], new_theta, cand), [(robot.x, robot.y), new_pos]))
        expanded_nodes = {}
        while open_set:
            f, g, depth, node, path = heapq.heappop(open_set)
            if depth not in expanded_nodes: expanded_nodes[depth] = []
            expanded_nodes[depth].append(path)
            if depth >= ng:
                for pc in primary_candidates:
                    p_pos, _ = self._project_robot(robot.x, robot.y, robot.theta, pc)
                    if np.allclose(p_pos, path[1]): self._visualize_search_tree([(robot.x, robot.y)] + path, expanded_nodes); return pc
            x, y, theta, prev_dir = node
            projected_candidates = self._get_candidate_directions(x, y, perceived_obstacles)
            for cand in projected_candidates:
                new_pos, _ = self._project_robot(x, y, theta, cand)
                is_collision = any(np.sqrt((new_pos[0] - ox)**2 + (new_pos[1] - oy)**2) < ROBOT_RADIUS + orad for ox, oy, orad in perceived_obstacles)
                if is_collision: continue
                new_g = g + self._cost_gi(cand, theta, prev_dir, depth); new_h = self._heuristic_h(cand, theta, prev_dir, depth + 1)
                _, new_theta = self._project_robot(x, y, theta, cand)
                heapq.heappush(open_set, (new_g + new_h, new_g, depth + 1, (new_pos[0], new_pos[1], new_theta, cand), path + [new_pos]))
        costs = [self._cost_g0(c, robot) for c in primary_candidates]; best_dir = primary_candidates[np.argmin(costs)]
        self._visualize_search_tree([(robot.x, robot.y)], {}); return best_dir
    def _project_robot(self, x, y, theta, cand_dir): new_theta = normalize_angle(cand_dir); return (x + DS * np.cos(new_theta), y + DS * np.sin(new_theta)), new_theta
    def _delta(self, a1, a2): return normalize_angle(a1 - a2)
    def _cost_g0(self, c0, robot): target_angle = np.arctan2(self.target[1] - robot.y, self.target[0] - robot.x); return MU1 * abs(self._delta(c0, target_angle)) + MU2 * abs(self._delta(c0, robot.theta)) + MU3 * abs(self._delta(c0, robot.prev_selected_direction))
    def _cost_gi(self, ci, theta_i, ci_minus_1, i): target_angle = np.arctan2(self.target[1] - 0, self.target[0] - 0); return (LAMBDA**i) * (MU1_PRIME * abs(self._delta(ci, target_angle)) + MU2_PRIME * abs(self._delta(ci, theta_i)) + MU3_PRIME * abs(self._delta(ci, ci_minus_1)))
    def _heuristic_h(self, c, theta, prev_dir, depth): target_angle = np.arctan2(self.target[1] - 0, self.target[0] - 0); h=0; [h := h + (LAMBDA**i) * (MU2_PRIME * abs(self._delta(target_angle, theta)) + MU3_PRIME * abs(self._delta(target_angle, prev_dir))) for i in range(depth, depth + 3)]; return h
    def set_visualization_axes(self, ax_main, ax_hist, ax_tree=None): # <--- 修改这里
        """
        Sets the matplotlib axes for visualization.
        ax_tree is now optional.
        """
        self.ax_main, self.ax_hist, self.ax_tree = ax_main, ax_hist, ax_tree
    def _visualize_polar_histogram(self, x, y, obstacles):
        if not self.ax_hist: return
        self.ax_hist.clear(); histogram = self._get_polar_histogram(x, y, obstacles)
        angles = np.linspace(-np.pi, np.pi, NUM_HISTOGRAM_BINS, endpoint=False)
        self.ax_hist.bar(angles, histogram, width=HISTOGRAM_ALPHA, color='blue', alpha=0.7, align='center')
        self.ax_hist.set_theta_zero_location('E'); self.ax_hist.set_title("Polar Histogram (From Memory)")
    def _visualize_search_tree(self, best_path, expanded_nodes):
        if not self.ax_tree: return
        self.ax_tree.clear()
        for depth, paths in expanded_nodes.items(): [self.ax_tree.plot(*zip(*p), 'o-', c='gray', alpha=0.3, ms=3) for p in paths]
        if best_path and len(best_path) > 1: self.ax_tree.plot(*zip(*best_path), 'o-', c='black', lw=2, ms=5)
        self.ax_tree.set_title("A* Search Tree"); self.ax_tree.axis('equal')

IMAGE_PATH = '/Users/wanganbang/Documents/Code/MultiAgentNav/data/sample_1/floorplan_mask.png'  # <--- 将 'map.png' 替换为您的地图文件名
IMAGE_PIXEL_SPACING = 0.02  # 每个图像像素代表0.1米 (10厘米)
GRID_RESOLUTION = 0.05     # 模拟栅格的分辨率设为5厘米/像素

# (MODIFICATION): 从图像生成全局地图和世界尺寸
# 注意：旧的 OBSTACLES_DEFINITION 和 WORLD_SIZE 被这些所取代
GLOBAL_GRID, WORLD_SIZE = create_world_from_image(IMAGE_PATH, IMAGE_PIXEL_SPACING, GRID_RESOLUTION)
GLOBAL_GRID_SHAPE = GLOBAL_GRID.shape

# (MODIFICATION): 根据新的世界尺寸来设置目标点
# 确保目标点在世界边界内并且不是一个障碍物
WAYPOINTS = [
    (WORLD_SIZE[0] * 0.2, WORLD_SIZE[1] * 0.15), # 路点 0
    (WORLD_SIZE[0] * 0.2, WORLD_SIZE[1] * 0.45), # 路点 1
    (WORLD_SIZE[0] * 0.25, WORLD_SIZE[1] * 0.6), # 路点 2
    (WORLD_SIZE[0] * 0.45, WORLD_SIZE[1] * 0.6), # 最终目标 (路点 3)
    (WORLD_SIZE[0] * 0.5, WORLD_SIZE[1] * 0.7)  # 最终目标 (路点 3)
]

# (新增): 初始化路径点追踪状态
current_waypoint_index = 0

# 检查路径点列表是否为空
if not WAYPOINTS:
    raise ValueError("错误: 路径点列表 (WAYPOINTS) 不能为空！")

# 初始化机器人
robot = Robot(WORLD_SIZE[0] * 0.1, WORLD_SIZE[1] * 0.1, np.pi / 4, speed=0.25)

# (核心修改): VFH*的初始目标是序列中的第一个路径点
vfh_star = VFHStar(WAYPOINTS[0])

# 初始化机器人的记忆地图
robot_memory_map = np.full(GLOBAL_GRID_SHAPE, MAP_UNKNOWN, dtype=int)

# --- 可视化 ---
# (几乎不变, 只需要更新xlim和ylim)
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 3)
ax_main = fig.add_subplot(gs[:, 0:2])
ax_hist = fig.add_subplot(gs[0, 2], polar=True)
ax_global_map = fig.add_subplot(gs[1, 2])

vfh_star.set_visualization_axes(ax_main, ax_hist, None)

map_image = None

def init():
    global map_image

    # --- 创建一个通用的“真实世界障碍物”图层 ---
    # 这个图层将在两个地图中使用，以保证视觉风格统一
    # RGBA格式: [R, G, B, Alpha]
    true_map_obstacles_layer = np.zeros((*GLOBAL_GRID_SHAPE, 4))
    # 将障碍物位置设置为半透明的深灰色
    true_map_obstacles_layer[GLOBAL_GRID == 1] = [0.2, 0.2, 0.2, 0.8] # 稍微加深不透明度以示区分

    # --- 1. 设置主地图 (ax_main) ---
    ax_main.set_xlim(0, WORLD_SIZE[0])
    ax_main.set_ylim(0, WORLD_SIZE[1])
    ax_main.grid(True)
    ax_main.set_title(f"Robot's View: Heading to Waypoint {current_waypoint_index}")
    
    # 绘制背景：真实障碍物位置 (作为参考)
    ax_main.imshow(true_map_obstacles_layer, extent=(0, WORLD_SIZE[0], 0, WORLD_SIZE[1]), origin='lower')
    
    # 绘制前景：机器人的记忆地图 (将在update中更新)
    map_image = ax_main.imshow(np.zeros((*GLOBAL_GRID_SHAPE, 4)), 
                               extent=(0, WORLD_SIZE[0], 0, WORLD_SIZE[1]), 
                               origin='lower', alpha=0.6) # 记忆地图可以稍微透明

    # 绘制路径点
    if WAYPOINTS:
        wp_x, wp_y = zip(*WAYPOINTS)
        ax_main.plot(wp_x, wp_y, 'o--', color='orange', alpha=0.8, label="Waypoints Path")
        ax_main.plot(WAYPOINTS[-1][0], WAYPOINTS[-1][1], 'g*', ms=15, label="Final Target")
    ax_main.legend()


    # --- 2. (核心修改) 设置右下角的全局地图 (ax_global_map) ---
    ax_global_map.set_title("Global Map Overview")
    ax_global_map.set_xlim(0, WORLD_SIZE[0]) # 确保坐标系一致
    ax_global_map.set_ylim(0, WORLD_SIZE[1])
    ax_global_map.set_aspect('equal', adjustable='box') # 保持长宽比
    ax_global_map.set_xticks([])
    ax_global_map.set_yticks([])

    # 绘制背景：直接使用上面创建的同一个“真实世界障碍物”图层
    ax_global_map.imshow(true_map_obstacles_layer, extent=(0, WORLD_SIZE[0], 0, WORLD_SIZE[1]), origin='lower')
    
    # 同样绘制路径点，提供完整的上下文
    if WAYPOINTS:
        wp_x, wp_y = zip(*WAYPOINTS)
        ax_global_map.plot(wp_x, wp_y, 'o--', color='orange', alpha=0.8, lw=1)
        ax_global_map.plot(WAYPOINTS[-1][0], WAYPOINTS[-1][1], 'g*', ms=10) # 稍微小一点的星星
    
    return []

# --- 核心修改: 更新主循环 ---
# --- 核心修改: 更新主循环以实现路点跟踪 ---
def update(frame):
    global robot_memory_map, map_image, current_waypoint_index
    
    if robot.is_stopped: return []

    # (新逻辑) --- 路径点切换的核心逻辑 ---
    # 1. 检查是否已经完成所有路径点
    if current_waypoint_index >= len(WAYPOINTS):
        # 这种情况理论上在前一帧的末尾就会处理，但作为安全检查保留
        return []

    # 2. 获取当前目标点
    current_target = WAYPOINTS[current_waypoint_index]

    # 3. 计算机器人到当前目标点的距离
    dist_to_target = np.sqrt((robot.x - current_target[0])**2 + (robot.y - current_target[1])**2)

    # 4. 根据是否为最后一个点，确定切换阈值
    is_last_waypoint = (current_waypoint_index == len(WAYPOINTS) - 1)
    distance_threshold = 0.5 if is_last_waypoint else 1.0

    # 5. 检查是否已"到达"当前目标点
    if dist_to_target < distance_threshold:
        print(f"到达路径点 {current_waypoint_index}: {current_target}")
        current_waypoint_index += 1 # 索引指向下一个路径点

        # 检查是否所有路径点都已完成
        if current_waypoint_index >= len(WAYPOINTS):
            print("成功完成所有路径点！")
            robot.is_stopped = True
            ani.event_source.stop()
            ax_main.set_title("Mission Accomplished!", color='green')
            return []
        else:
            # 如果未完成，则更新VFH*算法的新目标
            new_target = WAYPOINTS[current_waypoint_index]
            vfh_star.target = new_target
            print(f"--> 新目标点 {current_waypoint_index}: {new_target}")
            ax_main.set_title(f"Heading to Waypoint {current_waypoint_index}")

    # --- (原有VFH*和机器人移动逻辑几乎不变) ---
    # 这个逻辑现在会自动使用我们上面可能已经更新过的 vfh_star.target

    # 1. 用当前扫描更新机器人的永久记忆地图
    update_memory_map_with_scan(robot, GLOBAL_GRID, robot_memory_map, GRID_RESOLUTION)

    # 2. 从记忆地图中提取局部障碍物给VFH*用
    perceived_obstacles = get_obstacles_from_memory(robot, robot_memory_map, SENSOR_RANGE_VFH, GRID_RESOLUTION)

    # 3. VFH* 使用提取出的局部障碍物进行规划
    ng = 5
    ideal_continuous_direction = vfh_star.get_best_direction(robot, ng, perceived_obstacles)

    if ideal_continuous_direction is None:
        print("Robot is TRAPPED! No path found.")
        robot.is_stopped = True
        ax_main.set_title("Robot Trapped!", color='red')
        return []

    # 4. 决策与行动 (逻辑不变)
    direction_error = normalize_angle(ideal_continuous_direction - robot.theta)
    if abs(direction_error) >= ALIGNMENT_TOLERANCE:
        best_action, min_err = None, float('inf')
        for name, val in {**DISCRETE_ACTIONS, "NO_TURN": 0.0}.items():
            err = abs(normalize_angle(ideal_continuous_direction - normalize_angle(robot.theta + val)))
            if err < min_err: min_err, best_action = err, val
        robot.rotate(best_action)
    else:
        robot.walk_forward()

    # --- (可视化更新逻辑不变) ---
    for line in ax_main.lines:
        if line.get_label() not in ['Waypoints Path', 'Final Target']: line.remove()
    for patch in ax_main.patches: patch.remove()
    
    for line in ax_global_map.lines:
        # 不要移除初始化时绘制的路径点连线
        if line.get_linestyle() != '--': line.remove() 
    for patch in ax_global_map.patches: patch.remove()

    # 2. 更新主地图的记忆图层 (这部分逻辑不变)
    color_map = np.zeros((*GLOBAL_GRID_SHAPE, 3))
    color_map[robot_memory_map == MAP_FREE] = [0.8, 0.8, 0.8]
    color_map[robot_memory_map == MAP_OCCUPIED] = [1, 0, 0]
    map_image.set_data(color_map)
    alpha_map = np.ones(GLOBAL_GRID_SHAPE) 
    alpha_map[robot_memory_map == MAP_UNKNOWN] = 0
    map_image.set_alpha(alpha_map)
    
    # 3. 绘制机器人当前的路径、位置和传感器
    path_x, path_y = zip(*robot.path)
    
    # 在主地图上绘制
    ax_main.plot(path_x, path_y, 'b-', label='_nolegend_')
    ax_main.add_patch(patches.Circle((robot.x, robot.y), ROBOT_RADIUS, fc='cyan'))
    ax_main.arrow(robot.x, robot.y, 1.5 * np.cos(robot.theta), 1.5 * np.sin(robot.theta), head_width=0.3, fc='k', ec='k')
    sensor_cone = patches.Wedge((robot.x, robot.y), SENSOR_MAX_RANGE, np.rad2deg(robot.theta - SENSOR_FOV / 2), np.rad2deg(robot.theta + SENSOR_FOV / 2), fc='yellow', alpha=0.15)
    ax_main.add_patch(sensor_cone)

    # (核心修改) 在全局地图上同样绘制路径和位置
    ax_global_map.plot(path_x, path_y, 'b-', lw=1.5) # 使用稍粗的线以便观察
    ax_global_map.add_patch(patches.Circle((robot.x, robot.y), ROBOT_RADIUS, fc='cyan'))

    # 4. 更新极坐标直方图 (这部分逻辑不变)
    vfh_star._visualize_polar_histogram(robot.x, robot.y, perceived_obstacles)
    
    return []

ani = FuncAnimation(fig, update, frames=10000, init_func=init, blit=False, interval=100, repeat=False)
plt.tight_layout()
plt.show()