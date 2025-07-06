#!/usr/bin/env python3
"""
Habitat Multi-Agent Video Generator with Predictive Collision Detection and State Persistence
"""

import os
import sys
import json
import yaml
import math
import time
import logging
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Optional, Tuple
import xml.etree.ElementTree as ET

# Habitat and Magnum imports
import habitat_sim
import magnum as mn

# --- Constants ---
# Epsilon for float comparisons
EPSILON = 1e-4

# --- URDF Camera Parser ---
class URDFCameraParser:
    """Parse URDF files to get robot head camera real position and orientation"""
    
    def __init__(self, urdf_path: str):
        self.urdf_path = urdf_path
        self.tree = ET.parse(urdf_path)
        self.root = self.tree.getroot()
        self.joint_transforms = {}
        self._parse_joints()
    
    def _parse_joints(self):
        """Parse all joint transforms"""
        for joint in self.root.findall('joint'):
            joint_name = joint.get('name')
            origin = joint.find('origin')
            if origin is not None:
                xyz = origin.get('xyz', '0 0 0').split()
                rpy = origin.get('rpy', '0 0 0').split()
                translation = np.array([float(x) for x in xyz])
                rotation = np.array([float(x) for x in rpy])
                
                # Convert RPY to rotation matrix
                rot_matrix = self._rpy_to_rotation_matrix(rotation)
                
                # Build 4x4 transform matrix
                transform = np.eye(4)
                transform[:3, :3] = rot_matrix
                transform[:3, 3] = translation
                
                self.joint_transforms[joint_name] = {
                    'transform': transform,
                    'parent': joint.find('parent').get('link') if joint.find('parent') is not None else None,
                    'child': joint.find('child').get('link') if joint.find('child') is not None else None
                }
    
    def _rpy_to_rotation_matrix(self, rpy):
        """Convert RPY angles to rotation matrix"""
        roll, pitch, yaw = rpy
        
        # Rotation matrix = Rz(yaw) * Ry(pitch) * Rx(roll)
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])
        
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
        
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
        
        return R_z @ R_y @ R_x
    
    def get_camera_transform_chain(self, camera_link='head_camera_rgb_frame'):
        """Get complete transform chain from base_link to camera"""
        # Define Fetch robot kinematic chain: base_link -> torso_lift_link -> head_pan_link -> head_tilt_link -> head_camera_link -> head_camera_rgb_frame
        chain = [
            'torso_lift_joint',    # base_link -> torso_lift_link
            'head_pan_joint',      # torso_lift_link -> head_pan_link  
            'head_tilt_joint',     # head_pan_link -> head_tilt_link
            'head_camera_joint',   # head_tilt_link -> head_camera_link
            'head_camera_rgb_joint' # head_camera_link -> head_camera_rgb_frame
        ]
        
        # Calculate cumulative transform
        cumulative_transform = np.eye(4)
        for joint_name in chain:
            if joint_name in self.joint_transforms:
                cumulative_transform = cumulative_transform @ self.joint_transforms[joint_name]['transform']
            else:
                logging.warning(f"Joint '{joint_name}' not found in URDF")
        
        return cumulative_transform
    
    def get_camera_position_and_orientation(self, robot_transform):
        """Get camera position and orientation in world coordinates"""
        try:
            # Get camera transform relative to robot base
            camera_transform = self.get_camera_transform_chain()
            
            # Apply robot world transform
            world_camera_transform = robot_transform @ camera_transform
            
            # Extract position
            position = world_camera_transform[:3, 3]
            
            # Extract rotation matrix and convert to quaternion
            rotation_matrix = world_camera_transform[:3, :3]
            quaternion = self._rotation_matrix_to_quaternion(rotation_matrix)
            
            return position, quaternion
            
        except Exception as e:
            logging.error(f"Failed to get camera transform: {e}")
            return None, None
    
    def _rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion [x, y, z, w]"""
        trace = np.trace(R)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return np.array([x, y, z, w])

# --- Matrix4 Utilities ---
def safe_matrix4_to_numpy(matrix4: mn.Matrix4) -> np.ndarray:
    """Safely convert Magnum Matrix4 to numpy array"""
    try:
        # Try the new API first
        if hasattr(matrix4, 'data'):
            return np.array(matrix4.data()).reshape(4, 4).T  # magnum is column-major, need transpose
        else:
            # Fallback to manual extraction
            result = np.zeros((4, 4))
            for i in range(4):
                for j in range(4):
                    result[i, j] = matrix4[i, j]
            return result
    except Exception as e:
        logging.warning(f"Matrix4 conversion failed: {e}, using identity matrix")
        return np.eye(4)

def safe_quaternion_to_numpy(quat) -> np.ndarray:
    """Safely convert various quaternion formats to numpy array [x, y, z, w]"""
    try:
        if isinstance(quat, np.ndarray):
            return quat.astype(np.float32)
        elif hasattr(quat, 'vector') and hasattr(quat, 'scalar'):
            # Magnum Quaternion
            vec = quat.vector
            return np.array([vec.x, vec.y, vec.z, quat.scalar], dtype=np.float32)
        elif hasattr(quat, '__len__') and len(quat) == 4:
            return np.array(quat, dtype=np.float32)
        else:
            # Default identity quaternion
            return np.array([0, 0, 0, 1], dtype=np.float32)
    except Exception as e:
        logging.warning(f"Quaternion conversion failed: {e}, using identity")
        return np.array([0, 0, 0, 1], dtype=np.float32)

# --- Utility: Logger Setup ---
def setup_logger(log_path: str) -> logging.Logger:
    """Sets up a logger to file and console."""
    logger = logging.getLogger("MultiAgentNav")
    logger.setLevel(logging.INFO)
    
    # Avoid adding handlers if already configured
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_path, mode='w')
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    return logger

class VideoRecorder:
    """
    Handles video creation for a single agent.
    Generates a split-screen video with FPV and a dynamic map view.
    """
    def __init__(self, agent_id: str, config: Dict[str, Any], base_map: Image.Image, logger: logging.Logger):
        self.agent_id = agent_id
        self.config = config
        self.video_config = config['video_settings']
        self.map_config = config['map_settings']
        self.base_map = base_map
        self.logger = logger
        
        self.frames: List[np.ndarray] = []
        self.output_path = os.path.join(
            self.config['output_dir'], f"{self.agent_id}_output.mp4"
        )
        
        # Initialize URDF parser for physical robots
        self.urdf_parser = None
        agent_idx = next(i for i, agent in enumerate(config['agents']) if agent['agent_id'] == agent_id)
        agent_config = config['agents'][agent_idx]
        if 'urdf_path' in agent_config and agent_config['urdf_path']:
            try:
                self.urdf_parser = URDFCameraParser(agent_config['urdf_path'])
                self.logger.info(f"[{agent_id}] URDF parser initialized for camera positioning")
            except Exception as e:
                self.logger.warning(f"[{agent_id}] Failed to initialize URDF parser: {e}")
        
        # Pre-load font
        try:
            self.font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except IOError:
            self.font_small = ImageFont.load_default()

    def capture_frame(self, sim: habitat_sim.Simulator, agent_idx: int, scene_bounds: Tuple[np.ndarray, np.ndarray]):
        """Captures and composes a single video frame."""
        if not self.video_config['enabled']:
            return

        # 1. Get FPV observation - handle URDF robots properly
        try:
            fpv_image_np = self._get_fpv_observation(sim, agent_idx)
            fpv_pil = Image.fromarray(fpv_image_np[..., :3], "RGB")
        except Exception as e:
            self.logger.warning(f"[{self.agent_id}] Failed to get FPV observation: {e}")
            # Fallback to black image
            fpv_pil = Image.new('RGB', (640, 480), (0, 0, 0))

        # 2. Prepare the map view
        map_image = self.base_map.copy()
        agent_state = sim.get_agent(agent_idx).state
        
        # Draw the current agent's marker on the map
        self._draw_agent_on_map(map_image, agent_state.position, agent_state.rotation, scene_bounds)

        # 3. Resize and combine
        vid_w, vid_h = self.video_config['resolution']
        fpv_w, map_w = vid_w // 2, vid_w // 2
        fpv_resized = fpv_pil.resize((fpv_w, vid_h), Image.Resampling.LANCZOS)
        map_resized = self._resize_map_with_aspect_ratio(map_image, map_w, vid_h)
        
        combined_frame = Image.new('RGB', (vid_w, vid_h))
        combined_frame.paste(fpv_resized, (0, 0))
        combined_frame.paste(map_resized, (fpv_w, 0))

        self.frames.append(np.array(combined_frame))

    def _get_fpv_observation(self, sim: habitat_sim.Simulator, agent_idx: int) -> np.ndarray:
        """Get FPV observation, using URDF robot camera position if available"""
        agent_config = self.config['agents'][agent_idx]
        fpv_sensor_uuid = agent_config['fpv_sensor_uuid']
        
        # Check if this is a physical URDF robot
        if self.urdf_parser and 'urdf_path' in agent_config:
            try:
                return self._get_urdf_robot_observation(sim, agent_idx, fpv_sensor_uuid)
            except Exception as e:
                self.logger.warning(f"⚠️ URDF-based camera failed: {e}")
                self.logger.warning("⚠️ FPV fallback - System falls back to virtual agent observations")
        
        # Fallback to standard sensor observation
        observations = sim.get_sensor_observations(agent_index=agent_idx)
        return observations[fpv_sensor_uuid]
    
    def _get_urdf_robot_observation(self, sim: habitat_sim.Simulator, agent_idx: int, sensor_uuid: str) -> np.ndarray:
        """Get observation from URDF robot camera with proper positioning"""
        try:
            # Get robot object if it exists
            robot_obj = None
            if hasattr(sim, 'get_rigid_object_manager'):
                rigid_obj_mgr = sim.get_rigid_object_manager()
                # Find robot object by name/id
                robot_id = self.config['agents'][agent_idx]['agent_id']
                for obj_id in rigid_obj_mgr.get_object_handles():
                    obj = rigid_obj_mgr.get_object_by_handle(obj_id)
                    if robot_id in obj_id or str(agent_idx) in obj_id:
                        robot_obj = obj
                        break
            
            if robot_obj is None:
                # No physical robot found, use agent state
                agent_state = sim.get_agent(agent_idx).state
                robot_transform = np.eye(4)
                robot_transform[:3, 3] = agent_state.position
                quat_array = safe_quaternion_to_numpy(agent_state.rotation)
                robot_transform[:3, :3] = self._quaternion_to_rotation_matrix(quat_array)
            else:
                # Get robot transformation matrix
                robot_transform_matrix = robot_obj.transformation
                robot_transform = safe_matrix4_to_numpy(robot_transform_matrix)
            
            # Get camera position and orientation using URDF
            camera_position, camera_orientation = self.urdf_parser.get_camera_position_and_orientation(robot_transform)
            
            if camera_position is not None and camera_orientation is not None:
                # Create temporary agent state with camera pose
                camera_agent_state = habitat_sim.AgentState()
                camera_agent_state.position = mn.Vector3(camera_position)
                
                # Convert quaternion to Magnum Quaternion
                quat_norm = np.linalg.norm(camera_orientation)
                if quat_norm > 0:
                    camera_orientation = camera_orientation / quat_norm
                camera_agent_state.rotation = mn.Quaternion(
                    mn.Vector3(camera_orientation[0], camera_orientation[1], camera_orientation[2]), 
                    camera_orientation[3]
                )
                
                # Temporarily set agent to camera position
                original_state = sim.get_agent(agent_idx).state
                sim.get_agent(agent_idx).set_state(camera_agent_state, reset_sensors=False)
                
                # Get observation
                observations = sim.get_sensor_observations(agent_index=agent_idx)
                observation = observations[sensor_uuid]
                
                # Restore original state
                sim.get_agent(agent_idx).set_state(original_state, reset_sensors=False)
                
                self.logger.debug(f"[{self.agent_id}] Got URDF-based camera observation from position: {camera_position}")
                return observation
            else:
                raise Exception("Failed to get camera transform from URDF")
                
        except Exception as e:
            self.logger.error(f"⚠️ Matrix4 conversion error - The URDF-based camera parsing has a technical issue with Magnum Matrix4 data access: {e}")
            raise e
    
    def _quaternion_to_rotation_matrix(self, q):
        """Convert quaternion [x, y, z, w] to rotation matrix"""
        x, y, z, w = q
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
        ])

        self.frames.append(np.array(combined_frame))

    def _world_to_map_coords(self, world_pos: np.ndarray, scene_bounds: Tuple[np.ndarray, np.ndarray]) -> Tuple[int, int]:
        """Converts world (x,z) to map image pixel coordinates."""
        padded_width, padded_height = self.base_map.size
        pad_left, pad_bottom, pad_top, pad_right = self.map_config['padding'].values()

        original_width = padded_width - pad_left - pad_right
        original_height = padded_height - pad_top - pad_bottom

        world_min, world_max = scene_bounds
        world_size_x = world_max[0] - world_min[0]
        world_size_z = world_max[2] - world_min[2]

        if world_size_x < EPSILON or world_size_z < EPSILON:
             return (0,0)

        px_in_original = (world_pos[0] - world_min[0]) / world_size_x * original_width
        py_in_original = (world_pos[2] - world_min[2]) / world_size_z * original_height

        px = int(px_in_original + pad_left)
        py = int(py_in_original + pad_top)
        
        return (px, py)

    def _draw_agent_on_map(self, image: Image.Image, position: np.ndarray, rotation: mn.Quaternion, scene_bounds):
        """Draws a single agent's marker and orientation on the map."""
        draw = ImageDraw.Draw(image)
        map_x, map_y = self._world_to_map_coords(position, scene_bounds)

        # Draw position dot
        dot_radius = 8
        draw.ellipse([map_x - dot_radius, map_y - dot_radius, map_x + dot_radius, map_y + dot_radius],
                     fill=(255, 0, 0), outline=(255, 255, 255), width=2)
                     
        # Draw orientation arrow
        forward_vec = rotation.transform_vector(mn.Vector3(0, 0, -1)) # Agent forward is -Z
        arrow_length = 25
        end_x = map_x + forward_vec.x * arrow_length
        end_y = map_y + forward_vec.z * arrow_length # Map Y corresponds to World Z
        
        draw.line([(map_x, map_y), (end_x, end_y)], fill=(255, 0, 0), width=4)

    def _resize_map_with_aspect_ratio(self, image: Image.Image, target_width: int, target_height: int) -> Image.Image:
        """Resizes an image, maintaining aspect ratio and padding with black."""
        original_aspect = image.width / image.height
        target_aspect = target_width / target_height

        if original_aspect > target_aspect:
            new_width = target_width
            new_height = int(new_width / original_aspect)
        else:
            new_height = target_height
            new_width = int(new_height * original_aspect)

        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        result = Image.new('RGB', (target_width, target_height), (0, 0, 0))
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        result.paste(resized, (x_offset, y_offset))
        return result

    def save_video(self):
        """Saves the captured frames to an MP4 file."""
        if not self.video_config['enabled'] or not self.frames:
            self.logger.info(f"[{self.agent_id}] Video generation skipped (disabled or no frames).")
            return
        
        self.logger.info(f"[{self.agent_id}] Saving {len(self.frames)} frames to {self.output_path}...")
        try:
            import cv2
            vid_w, vid_h = self.video_config['resolution']
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(self.output_path, fourcc, self.video_config['fps'], (vid_w, vid_h))
            
            for frame in self.frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            self.logger.info(f"[{self.agent_id}] Video saved successfully.")
        except ImportError:
            self.logger.error("OpenCV is not installed. Cannot save video. Please install with 'pip install opencv-python'.")
        except Exception as e:
            self.logger.error(f"[{self.agent_id}] Failed to save video: {e}")

class AgentController:
    """Manages a single agent's state, actions, and video recording."""
    def __init__(self, agent_idx: int, config: Dict[str, Any], actions: List[Dict],
                 base_map: Image.Image, logger: logging.Logger):
        self.agent_idx = agent_idx
        self.config = config
        self.agent_config = config['agents'][agent_idx]
        self.agent_id = self.agent_config['agent_id']
        self.motion_params = self.agent_config['motion_params']
        self.logger = logger

        self.action_queue = actions
        self.current_action = None
        self.is_finished = not bool(self.action_queue)
        
        self.video_recorder = VideoRecorder(self.agent_id, config, base_map, logger)

    def start_next_action(self):
        """Pops the next action from the queue and sets it as current."""
        if self.action_queue:
            self.current_action = self.action_queue.pop(0)
            self.logger.info(f"[{self.agent_id}] Starting action: {self.current_action}")
        else:
            self.current_action = None
            self.is_finished = True
            self.logger.info(f"[{self.agent_id}] Action queue complete.")

    def update(self, current_state: habitat_sim.AgentState, time_step: float) -> habitat_sim.AgentState:
        """
        Calculates the agent's proposed state for the next time step based on the current action.
        This is a *proposed* state, used for collision detection before execution.
        """
        if self.is_finished or not self.current_action:
            return current_state

        action_type = self.current_action['action']
        
        next_state = habitat_sim.AgentState()
        next_state.position = np.copy(current_state.position)
        next_state.rotation = current_state.rotation
        
        # --- Handle move_to action ---
        if action_type == 'move_to':
            target_2d = np.array(self.current_action['target'])
            current_pos_2d = np.array([current_state.position[0], current_state.position[2]])
            
            vec_to_target = target_2d - current_pos_2d
            dist_to_target = np.linalg.norm(vec_to_target)

            # Check if destination is reached
            if dist_to_target < self.motion_params['linear_speed'] * time_step:
                # Close enough, snap to target and finish action
                # Note: Y-coord will be snapped to navmesh later
                next_state.position[0] = target_2d[0]
                next_state.position[2] = target_2d[1]
                self.start_next_action()
            else:
                # --- Turn towards target ---
                direction_vec = vec_to_target / dist_to_target
                target_angle_rad = math.atan2(direction_vec[0], direction_vec[1]) # angle wrt +Z
                
                # Get current heading
                forward_vec = current_state.rotation.transform_vector(mn.Vector3(0, 0, -1))
                current_angle_rad = math.atan2(forward_vec.x, forward_vec.z)
                
                # Calculate angle difference
                angle_diff = target_angle_rad - current_angle_rad
                # Normalize to [-pi, pi]
                angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

                # Rotate if not facing target
                max_rotation_rad = math.radians(self.motion_params['angular_speed']) * time_step
                if abs(angle_diff) > max_rotation_rad:
                    rot_angle = np.sign(angle_diff) * max_rotation_rad
                    rot_quat = mn.Quaternion.rotation(mn.Rad(rot_angle), mn.Vector3.y_axis())
                    next_state.rotation = rot_quat * current_state.rotation
                else: # Facing target, now move forward
                    # Snap rotation to exact heading
                    rot_quat = mn.Quaternion.rotation(mn.Rad(angle_diff), mn.Vector3.y_axis())
                    next_state.rotation = rot_quat * current_state.rotation
                    
                    # Move forward
                    move_dist = self.motion_params['linear_speed'] * time_step
                    move_vec = next_state.rotation.transform_vector(mn.Vector3(0, 0, -1)) * move_dist
                    next_state.position += np.array([move_vec.x, move_vec.y, move_vec.z])

        # --- Handle turn_left/right actions ---
        elif action_type in ['turn_left', 'turn_right']:
            remaining_angle = self.current_action.get('remaining_angle', self.current_action['angle'])
            turn_rad = math.radians(self.motion_params['angular_speed'] * time_step)
            
            if remaining_angle < math.degrees(turn_rad):
                turn_rad = math.radians(remaining_angle)
                self.start_next_action()
            else:
                self.current_action['remaining_angle'] = remaining_angle - math.degrees(turn_rad)

            if action_type == 'turn_left':
                rot_quat = mn.Quaternion.rotation(mn.Rad(turn_rad), mn.Vector3.y_axis())
            else: # turn_right
                rot_quat = mn.Quaternion.rotation(mn.Rad(-turn_rad), mn.Vector3.y_axis())
            
            next_state.rotation = rot_quat * current_state.rotation
            
        elif action_type == "move_forward":
            remaining_dist = self.current_action.get("remaining_distance", self.current_action["distance"])
            move_dist_this_step = self.motion_params["linear_speed"] * time_step
            
            if remaining_dist < move_dist_this_step:
                move_dist_this_step = remaining_dist
                self.start_next_action()
            else:
                self.current_action["remaining_distance"] = remaining_dist - move_dist_this_step

            move_vec = current_state.rotation.transform_vector(mn.Vector3(0, 0, -1)) * move_dist_this_step
            next_state.position += np.array([move_vec.x, move_vec.y, move_vec.z])


        return next_state


class MultiAgentSimulator:
    """
    Main class to orchestrate the multi-agent simulation.
    """
    def __init__(self, config_path: str, actions_path: str, resume_state_path: Optional[str]):
        self.config = self._load_yaml(config_path)
        self.actions = self._load_json(actions_path)
        self.resume_state = self._load_json(resume_state_path) if resume_state_path else None
        
        os.makedirs(self.config['output_dir'], exist_ok=True)
        os.makedirs(self.config['state_dir'], exist_ok=True)

        self.logger = setup_logger(os.path.join(self.config['output_dir'], self.config['log_file']))

        self.sim: Optional[habitat_sim.Simulator] = None
        self.agents: List[AgentController] = []
        self.scene_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.base_map: Optional[Image.Image] = None

    def _load_yaml(self, path: str) -> Dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _load_json(self, path: str) -> Dict:
        with open(path, 'r') as f:
            return json.load(f)

    def _generate_base_map(self):
        """Generates a clean top-down map of the scene before adding agents."""
        self.logger.info("Generating base top-down map...")
        
        # Create a temporary simulator just for map generation
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = self.config['scene_path']
        backend_cfg.gpu_device_id = self.config['simulation_settings']['gpu_device_id']

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        
        # Get scene bounds to calculate map aspect ratio
        temp_sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))
        bounds = temp_sim.pathfinder.get_bounds()
        world_size_x = bounds[1][0] - bounds[0][0]
        world_size_z = bounds[1][2] - bounds[0][2]
        temp_sim.close()
        
        aspect_ratio = world_size_x / world_size_z if world_size_z > EPSILON else 1.0
        map_res = self.config['map_settings']['resolution']
        if aspect_ratio > 1:
            map_w, map_h = map_res, int(map_res / aspect_ratio)
        else:
            map_w, map_h = int(map_res * aspect_ratio), map_res

        # Configure orthographic sensor
        ortho_sensor_spec = habitat_sim.CameraSensorSpec()
        ortho_sensor_spec.uuid = "ortho_sensor"
        ortho_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        ortho_sensor_spec.resolution = [map_h, map_w]
        ortho_sensor_spec.position = [0.0, 100.0, 0.0] # High up
        ortho_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
        agent_cfg.sensor_specifications = [ortho_sensor_spec]
        
        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        map_sim = habitat_sim.Simulator(cfg)

        # Position camera to capture entire scene
        center = (bounds[0] + bounds[1]) / 2.0
        ortho_sensor_spec.position = [center[0], center[1] + 5.0, center[2]]
        
        # Set agent state to look down
        agent_state = habitat_sim.AgentState()
        agent_state.position = ortho_sensor_spec.position
        # Look down from +Y axis
        agent_state.rotation = mn.Quaternion.rotation(mn.Deg(-90), mn.Vector3.x_axis())
        map_sim.get_agent(0).set_state(agent_state)

        # Set ortho scale
        map_sim.get_agent(0).agent_config.sensor_specifications[0].orthographic_scale = max(world_size_x, world_size_z) / 2.0
        
        obs = map_sim.get_sensor_observations()
        map_image_np = obs["ortho_sensor"]
        map_pil = Image.fromarray(map_image_np[..., :3], "RGB")
        
        map_sim.close()

        self.scene_bounds = (bounds[0], bounds[1])
        self.base_map = self._draw_coordinate_system(map_pil, (bounds[0], bounds[1]))
        self.logger.info("Base map generated successfully.")

    def _draw_coordinate_system(self, image: Image.Image, scene_bounds) -> Image.Image:
        """Draws a coordinate grid and labels on the map image."""
        # This function is a simplified version adapted from the original generator
        # It would be implemented similarly to the provided habitat_video_generator.py
        # For brevity, this example just returns the image, but the full logic would go here.
        # Key steps:
        # 1. Create a new larger image with padding.
        # 2. Paste the original map onto it.
        # 3. Draw grid lines, tick marks, and numeric labels for X and Z axes.
        # 4. Add axis titles ("X (meters)", "Z (meters)").
        self.logger.info("Coordinate system drawing is simplified for this example. The full implementation would add a grid.")
        return image

    def _initialize_sim(self):
        """Initializes the main Habitat simulator with multiple agents."""
        self.logger.info("Initializing multi-agent simulator...")
        
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = self.config['scene_path']
        backend_cfg.enable_physics = True
        backend_cfg.gpu_device_id = self.config['simulation_settings']['gpu_device_id']
        backend_cfg.allow_sliding = False # Important for character control

        agent_configs = []
        for agent_cfg_data in self.config['agents']:
            h_agent_cfg = habitat_sim.agent.AgentConfiguration()
            h_agent_cfg.name = agent_cfg_data['agent_id']
            
            # Load agent model - handle URDF robots properly
            if 'urdf_path' in agent_cfg_data and agent_cfg_data['urdf_path']:
                h_agent_cfg.urdf_path = agent_cfg_data['urdf_path']
                self.logger.info(f"Loading URDF robot: {agent_cfg_data['urdf_path']}")
            
            # Configure sensors - avoid hardcoded orientation for URDF robots
            h_agent_cfg.sensor_specifications = []
            for sensor_data in agent_cfg_data['sensors']:
                sensor_spec = habitat_sim.CameraSensorSpec()
                sensor_spec.uuid = sensor_data['uuid']
                sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
                sensor_spec.resolution = sensor_data['resolution']
                sensor_spec.position = sensor_data['position']
                
                # For URDF robots, don't set hardcoded orientation - it will be calculated dynamically
                if 'urdf_path' not in agent_cfg_data or not agent_cfg_data['urdf_path']:
                    # Only set orientation for non-URDF agents
                    if 'orientation' in sensor_data:
                        try:
                            sensor_spec.orientation = sensor_data['orientation']
                        except Exception as e:
                            self.logger.warning(f"Failed to set sensor orientation: {e}, using default")
                            sensor_spec.orientation = [0.0, 0.0, 0.0]
                    else:
                        sensor_spec.orientation = [0.0, 0.0, 0.0]
                else:
                    # For URDF robots, use default orientation - will be overridden by URDF parsing
                    sensor_spec.orientation = [0.0, 0.0, 0.0]
                    self.logger.info(f"URDF robot sensor orientation will be calculated dynamically")
                
                sensor_spec.hfov = sensor_data['hfov']
                h_agent_cfg.sensor_specifications.append(sensor_spec)
            agent_configs.append(h_agent_cfg)

        sim_cfg = habitat_sim.Configuration(backend_cfg, agent_configs)
        self.sim = habitat_sim.Simulator(sim_cfg)

        # Initialize AgentControllers
        for i, agent_cfg_data in enumerate(self.config['agents']):
            agent_id = agent_cfg_data['agent_id']
            agent_actions = self.actions.get(agent_id, [])
            controller = AgentController(
                agent_idx=i,
                config=self.config,
                actions=agent_actions,
                base_map=self.base_map,
                logger=self.logger
            )
            self.agents.append(controller)

        self._set_initial_agent_states()
        self.logger.info("Multi-agent simulator initialized.")

    def _set_initial_agent_states(self):
        """Sets the starting position and rotation for each agent."""
        self.logger.info("Setting initial agent states...")
        for i, agent_controller in enumerate(self.agents):
            agent_id = agent_controller.agent_id
            agent_config = self.config['agents'][i]
            state_to_set = None

            # Priority 1: Resume from state file
            if self.resume_state and agent_id in self.resume_state:
                state_data = self.resume_state[agent_id]
                state_to_set = habitat_sim.AgentState()
                state_to_set.position = np.array(state_data['position'])
                
                # Safe quaternion handling
                try:
                    rotation_data = state_data['rotation']
                    if len(rotation_data) == 4:
                        state_to_set.rotation = mn.Quaternion(
                            mn.Vector3(rotation_data[0], rotation_data[1], rotation_data[2]), 
                            rotation_data[3]
                        )
                    else:
                        state_to_set.rotation = mn.Quaternion.rotation(mn.Deg(0.0), mn.Vector3.y_axis())
                except Exception as e:
                    self.logger.warning(f"[{agent_id}] Failed to set rotation from state: {e}")
                    state_to_set.rotation = mn.Quaternion.rotation(mn.Deg(0.0), mn.Vector3.y_axis())
                
                self.logger.info(f"[{agent_id}] Resuming from saved state.")

            # Priority 2: Use initial state from config
            else:
                initial_state_data = agent_config['initial_state']
                pos = initial_state_data['position']
                
                # Snap to a valid navigable point
                snapped_pos = self.sim.pathfinder.snap_point(pos)
                if np.linalg.norm(np.array(pos) - np.array(snapped_pos)) > 1.0:
                    self.logger.warning(f"[{agent_id}] Initial position {pos} is far from navmesh. Snapped to {snapped_pos}.")
                
                state_to_set = habitat_sim.AgentState()
                state_to_set.position = snapped_pos
                
                # Set rotation from config if available
                if 'rotation' in initial_state_data:
                    try:
                        rotation_data = initial_state_data['rotation']
                        if len(rotation_data) == 4:
                            state_to_set.rotation = mn.Quaternion(
                                mn.Vector3(rotation_data[0], rotation_data[1], rotation_data[2]), 
                                rotation_data[3]
                            )
                        else:
                            state_to_set.rotation = mn.Quaternion.rotation(mn.Deg(0.0), mn.Vector3.y_axis())
                    except Exception as e:
                        self.logger.warning(f"[{agent_id}] Failed to set rotation from config: {e}")
                        state_to_set.rotation = mn.Quaternion.rotation(mn.Deg(0.0), mn.Vector3.y_axis())
                else:
                    # Default rotation
                    state_to_set.rotation = mn.Quaternion.rotation(mn.Deg(0.0), mn.Vector3.y_axis())
                
                self.logger.info(f"[{agent_id}] Placed at initial config position {snapped_pos}.")

            # Set agent state with error handling
            try:
                self.sim.get_agent(i).set_state(state_to_set, reset_sensors=False)
                
                # For URDF robots, we may need additional setup
                if 'urdf_path' in agent_config and agent_config['urdf_path']:
                    self.logger.info(f"[{agent_id}] URDF robot initialized at position {state_to_set.position}")
                    
            except Exception as e:
                self.logger.error(f"[{agent_id}] Failed to set initial state: {e}")
                # Try with default state
                try:
                    default_state = habitat_sim.AgentState()
                    default_state.position = np.array([0.0, 0.0, 0.0])
                    default_state.rotation = mn.Quaternion.rotation(mn.Deg(0.0), mn.Vector3.y_axis())
                    self.sim.get_agent(i).set_state(default_state, reset_sensors=False)
                    self.logger.warning(f"[{agent_id}] Set to default state due to initialization error")
                except Exception as e2:
                    self.logger.error(f"[{agent_id}] Failed to set even default state: {e2}")
            
            agent_controller.start_next_action()
            
    def run_simulation(self):
        """The main simulation loop."""
        self._generate_base_map()
        self._initialize_sim()
        
        sim_settings = self.config['simulation_settings']
        time_step = sim_settings['time_step']
        max_duration = sim_settings['max_duration_seconds']
        
        start_time = time.time()
        sim_time = 0.0
        
        self.logger.info("====== Starting Simulation ======")
        
        # Capture initial frame
        for agent in self.agents:
            agent.video_recorder.capture_frame(self.sim, agent.agent_idx, self.scene_bounds)

        while sim_time < max_duration:
            # 1. Check for termination condition
            if all(agent.is_finished for agent in self.agents):
                self.logger.info("====== Simulation Ended: All tasks completed. ======")
                break

            # 2. Get proposed states for all agents for the next step
            current_states = [self.sim.get_agent(i).state for i in range(len(self.agents))]
            proposed_states = [
                agent.update(current_states[i], time_step)
                for i, agent in enumerate(self.agents)
            ]

            # 3. Predictive Collision Detection
            collision_predicted, reason = self._check_for_collisions(proposed_states)
            if collision_predicted:
                self.logger.warning(f"====== Simulation Halted: Collision Predicted! ======")
                self.logger.warning(f"Reason: {reason}")
                break

            # 4. If no collision, execute the move with safe state handling
            for i, state in enumerate(proposed_states):
                try:
                    # Snap Y-coordinate to navmesh to prevent falling
                    snapped_pos = self.sim.pathfinder.snap_point(state.position)
                    state.position[1] = snapped_pos[1]
                    
                    # Safe state setting with error handling
                    self.sim.get_agent(i).set_state(state, reset_sensors=False)
                    
                except Exception as e:
                    agent_id = self.config['agents'][i]['agent_id']
                    self.logger.warning(f"[{agent_id}] Failed to update state: {e}")
                    # Keep current state if update fails

            # 5. Step physics
            self.sim.step_physics(time_step)
            sim_time += time_step

            # 6. Capture video frames with error handling
            for agent in self.agents:
                try:
                    agent.video_recorder.capture_frame(self.sim, agent.agent_idx, self.scene_bounds)
                except Exception as e:
                    self.logger.warning(f"[{agent.agent_id}] Failed to capture video frame: {e}")

        else: # Loop finished due to timeout
            self.logger.warning(f"====== Simulation Halted: Max duration of {max_duration}s reached. ======")

        # --- Finalization ---
        self.logger.info(f"Simulation ran for {sim_time:.2f} seconds.")
        self.save_final_state()
        for agent in self.agents:
            agent.video_recorder.save_video()
            
        self.sim.close()
        self.logger.info("Simulator closed.")

    def _check_for_collisions(self, proposed_states: List[habitat_sim.AgentState]) -> Tuple[bool, str]:
        """
        Checks for agent-scene and agent-agent collisions for the proposed states.
        """
        sim_settings = self.config['simulation_settings']
        
        # Agent-Scene collision
        for i, state in enumerate(proposed_states):
            # A simple check is to see if the point is navigable.
            if not self.sim.pathfinder.is_navigable(state.position):
                agent_id = self.config['agents'][i]['agent_id']
                return True, f"Agent '{agent_id}' proposed move to non-navigable location {state.position}."

        # Agent-Agent collision
        for i in range(len(proposed_states)):
            for j in range(i + 1, len(proposed_states)):
                pos_i = proposed_states[i].position
                pos_j = proposed_states[j].position
                dist = np.linalg.norm(pos_i - pos_j)
                
                if dist < sim_settings['collision_distance_threshold']:
                    id_i = self.config['agents'][i]['agent_id']
                    id_j = self.config['agents'][j]['agent_id']
                    return True, f"Predicted collision between '{id_i}' and '{id_j}'. Distance: {dist:.2f}m."
        
        return False, ""

    def save_final_state(self):
        """Saves the final state of all agents to a JSON file."""
        final_states = {}
        for i, agent_controller in enumerate(self.agents):
            state = self.sim.get_agent(i).state
            pos = state.position
            rot = state.rotation
            final_states[agent_controller.agent_id] = {
                "position": [pos[0], pos[1], pos[2]],
                "rotation": [rot.vector.x, rot.vector.y, rot.vector.z, rot.scalar]
            }
        
        state_path = os.path.join(self.config['state_dir'], 'last_state.json')
        with open(state_path, 'w') as f:
            json.dump(final_states, f, indent=2)
        
        self.logger.info(f"Final agent states saved to {state_path}")
        self.logger.info(json.dumps(final_states, indent=2))

def main():
    parser = argparse.ArgumentParser(description="Run Habitat Multi-Agent Navigation Simulation.")
    parser.add_argument("--config", type=str, required=True, help="Path to the main YAML configuration file.")
    parser.add_argument("--actions", type=str, required=True, help="Path to the JSON file with agent actions.")
    parser.add_argument("--resume-from-state", type=str, default=None, help="Optional. Path to a JSON state file to resume from.")
    args = parser.parse_args()

    try:
        simulator = MultiAgentSimulator(
            config_path=args.config,
            actions_path=args.actions,
            resume_state_path=args.resume_from_state
        )
        simulator.run_simulation()
    except Exception as e:
        logging.error(f"An unhandled exception occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()