#!/usr/bin/env python3
"""
Discriminator System for ObjectNav Model Comparison

This script implements a comprehensive system to:
1. Identify controversial episodes where two models disagree
2. Use a discriminator LLM to re-evaluate these episodes 
3. Generate final statistics combining original and discriminated results

Usage:
    python discriminator_system.py [--config_path /path/to/config.json]
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys

# Add current directory to path to import current_topdown
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from current_topdown import render_topdown_view

# Import LLM client
import requests


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


@dataclass
class EpisodeResult:
    """Data class to store episode evaluation results"""
    scene_id: str
    episode_id: str
    sr: bool
    spl: float
    success: bool
    geodesic_distance_to_target: float
    path_length: float
    object_category: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sr': self.sr,
            'spl': self.spl,
            'success': self.success,
            'geodesic_distance_to_target': self.geodesic_distance_to_target,
            'path_length': self.path_length,
            'object_category': self.object_category
        }


@dataclass
class EpisodeData:
    """Data class to store episode preprocessing data"""
    scene_path: str
    agent_position: List[float]
    target_coordinate: List[float]
    target_name: str
    wall_mask_path: str
    navigation_nodes: List[Dict[str, Any]]
    
    
@dataclass
class ControversialEpisode:
    """Data class for controversial episodes"""
    scene_id: str
    episode_id: str
    model1_result: EpisodeResult
    model2_result: EpisodeResult
    episode_data1: EpisodeData
    episode_data2: EpisodeData
    
    @property
    def is_controversial(self) -> bool:
        """Check if models disagree on success"""
        return self.model1_result.success != self.model2_result.success


class LLMClient:
    """Client for LLM API interactions"""
    
    def __init__(self, config: Dict[str, Any]):
        self.api_key = config['api_key']
        self.base_url = config['base_url']
        self.model = config['model']
        self.max_tokens = config['max_tokens']
        
    def query(self, prompt: str, image_paths: Optional[List[str]] = None) -> str:
        """Query the LLM with optional multiple images"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        messages = []
        
        if image_paths and all(os.path.exists(path) for path in image_paths if path):
            # For multimodal models, encode images as base64
            import base64
            
            content = [{"type": "text", "text": prompt}]
            
            for i, image_path in enumerate(image_paths):
                if image_path and os.path.exists(image_path):
                    with open(image_path, 'rb') as f:
                        image_b64 = base64.b64encode(f.read()).decode()
                    
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        }
                    })
            
            messages.append({
                "role": "user",
                "content": content
            })
        else:
            messages.append({"role": "user", "content": prompt})
        
        payload = {
            'model': self.model,
            'messages': messages,
            'max_tokens': self.max_tokens,
            'temperature': 0
        }
        
        try:
            response = requests.post(f"{self.base_url}/chat/completions", 
                                   headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            logging.error(f"LLM query failed: {e}")
            return f"ERROR: {str(e)}"


class DiscriminatorSystem:
    """Main discriminator system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model1_dir = Path(config['model_paths']['model1_output'])
        self.model2_dir = Path(config['model_paths']['model2_output'])
        self.output_dir = Path(config['output_config']['discriminator_output'])
        self.output_dir.mkdir(exist_ok=True)
        
        self.llm_client = LLMClient(config['llm_config'])
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'discriminator.log'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
    def load_batch_results(self, model_dir: Path) -> Dict[str, EpisodeResult]:
        """Load batch results from individual episode directories"""
        results = {}
        
        # Walk through all scene directories
        for scene_dir in model_dir.iterdir():
            if scene_dir.is_dir() and scene_dir.name != "batch_output.json":
                scene_id = scene_dir.name
                
                # Walk through all episode directories in this scene
                for episode_dir in scene_dir.iterdir():
                    if episode_dir.is_dir() and episode_dir.name.isdigit():
                        episode_id = episode_dir.name
                        episode_key = f"{scene_id}/{episode_id}"
                        
                        # Check if output.json exists
                        output_file = episode_dir / "output.json"
                        if output_file.exists():
                            try:
                                with open(output_file, 'r') as f:
                                    data = json.load(f)
                                
                                # Extract evaluation results
                                eval_results = data.get('evaluation_results')
                                if eval_results is None:
                                    continue
                                
                                # Get target object name from action.json (more reliable)
                                target_object_name = 'unknown'
                                preprocess_dir = episode_dir / "preprocess"
                                action_file = preprocess_dir / "action.json"
                                
                                if action_file.exists():
                                    try:
                                        with open(action_file, 'r') as f:
                                            action_data = json.load(f)
                                        target_object_name = action_data.get('target_info', {}).get('name', 'unknown')
                                    except (json.JSONDecodeError, KeyError) as e:
                                        # Fallback to output.json
                                        target_object_name = data.get('object_category', 'unknown')
                                else:
                                    # Fallback to output.json
                                    target_object_name = data.get('object_category', 'unknown')
                                
                                results[episode_key] = EpisodeResult(
                                    scene_id=scene_id,
                                    episode_id=episode_id,
                                    sr=bool(eval_results.get('sr', False)),
                                    spl=float(eval_results.get('spl', 0.0)),
                                    success=bool(eval_results.get('success', False)),
                                    geodesic_distance_to_target=float(eval_results.get('geodesic_distance_to_target', 0.0)),
                                    path_length=float(eval_results.get('path_length', 0.0)),
                                    object_category=target_object_name
                                )
                            except (json.JSONDecodeError, KeyError) as e:
                                self.logger.warning(f"Could not load results for {episode_key}: {e}")
                                continue
        
        return results
    
    def load_episode_data(self, model_dir: Path, scene_id: str, episode_id: str) -> Optional[EpisodeData]:
        """Load episode preprocessing data"""
        episode_dir = model_dir / scene_id / episode_id / "preprocess"
        
        if not episode_dir.exists():
            self.logger.warning(f"Episode directory not found: {episode_dir}")
            return None
            
        # Load action.json
        action_file = episode_dir / "action.json"
        metadata_file = episode_dir / "metadata.json"
        navigation_file = episode_dir / "navigation_nodes.json"
        
        if not all(f.exists() for f in [action_file, metadata_file]):
            self.logger.warning(f"Required files missing for {scene_id}/{episode_id}")
            return None
            
        try:
            with open(action_file, 'r') as f:
                action_data = json.load(f)
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            navigation_nodes = []
            if navigation_file.exists():
                with open(navigation_file, 'r') as f:
                    navigation_nodes = json.load(f)
                    
            return EpisodeData(
                scene_path=metadata['scene_info']['scene_path'],
                agent_position=action_data['agent_state']['position'],
                target_coordinate=action_data['target_info']['coordinate'],
                target_name=action_data['target_info']['name'],
                wall_mask_path=action_data.get('wall_mask', ''),
                navigation_nodes=navigation_nodes
            )
            
        except Exception as e:
            self.logger.error(f"Error loading episode data for {scene_id}/{episode_id}: {e}")
            return None
    
    def identify_controversial_episodes(self) -> List[ControversialEpisode]:
        """Identify episodes where models disagree"""
        self.logger.info("Loading model results...")
        
        model1_results = self.load_batch_results(self.model1_dir)
        model2_results = self.load_batch_results(self.model2_dir)
        
        # Find common episodes
        common_episodes = set(model1_results.keys()) & set(model2_results.keys())
        self.logger.info(f"Found {len(common_episodes)} common episodes")
        
        controversial = []
        
        for episode_key in common_episodes:
            result1 = model1_results[episode_key]
            result2 = model2_results[episode_key]
            
            # Check if models disagree on success
            if result1.success != result2.success:
                scene_id, episode_id = episode_key.split('/')
                
                # Load episode data for both models
                data1 = self.load_episode_data(self.model1_dir, scene_id, episode_id)
                data2 = self.load_episode_data(self.model2_dir, scene_id, episode_id)
                
                if data1 and data2:
                    controversial.append(ControversialEpisode(
                        scene_id=scene_id,
                        episode_id=episode_id,
                        model1_result=result1,
                        model2_result=result2,
                        episode_data1=data1,
                        episode_data2=data2
                    ))
        
        self.logger.info(f"Identified {len(controversial)} controversial episodes")
        return controversial
    
    def discriminate_episode(self, episode: ControversialEpisode) -> Dict[str, Any]:
        """Use LLM to discriminate a controversial episode"""
        self.logger.info(f"Discriminating episode {episode.scene_id}/{episode.episode_id}")
        
        # Create separate topdown images for both models
        model1_image_path, model2_image_path = self.create_model_topdown_images(episode)
        if not model1_image_path or not model2_image_path:
            return {
                "error": "Failed to create model topdown images",
                "decision": None,
                "reasoning": "Could not generate topdown visualizations for both models"
            }
        
        # Create discrimination prompt
        prompt = self.create_discrimination_prompt(episode)
        
        # Query LLM with both images
        image_paths = [model1_image_path, model2_image_path]
        response = self.llm_client.query(prompt, image_paths)
        
        # Parse response to extract decision
        decision = None
        reasoning = response
        
        if "Decision: Model 1" in response or "Decision: 1" in response:
            decision = "Model 1"
        elif "Decision: Model 2" in response or "Decision: 2" in response:
            decision = "Model 2"
        
        return {
            "scene_id": episode.scene_id,
            "episode_id": episode.episode_id,
            "model1_result": episode.model1_result.to_dict(),
            "model2_result": episode.model2_result.to_dict(),
            "model1_image": model1_image_path,
            "model2_image": model2_image_path,
            "llm_response": response,
            "decision": decision,
            "reasoning": reasoning,
            "target_object": episode.episode_data1.target_name
        }
    
    def create_model_topdown_images(self, episode: ControversialEpisode) -> Tuple[Optional[str], Optional[str]]:
        """Create separate topdown images for each model with their respective target nodes and crop around 5m radius"""
        try:
            # Use the scene path from episode data
            scene_path = episode.episode_data1.scene_path
            agent_position = episode.episode_data1.agent_position
            
            # Render topdown view at the agent's floor
            image, coords, metadata = render_topdown_view(
                scene_path, 
                target_floor=agent_position,  # Use agent position to determine floor
                draw_coordinates=self.config.get('rendering_config', {}).get('draw_coordinates', False),
                resolution=self.config.get('rendering_config', {}).get('resolution', [1024, 1024])
            )
            
            if image is None:
                self.logger.error(f"Failed to render topdown for {episode.scene_id}/{episode.episode_id}")
                return None, None
            
            # Save the rendered image with proper directory structure
            scene_output_dir = self.output_dir / episode.scene_id
            episode_output_dir = scene_output_dir / episode.episode_id
            episode_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load metadata to convert world coordinates to pixel coordinates
            metadata_file = self.model1_dir / episode.scene_id / episode.episode_id / "preprocess" / "metadata.json"
            if not metadata_file.exists():
                self.logger.warning(f"Metadata file not found: {metadata_file}")
                return None, None
                
            with open(metadata_file, 'r') as f:
                metadata_json = json.load(f)
            
            # Check if topdown metadata exists
            topdown_meta = metadata_json.get('topdown_metadata')
            if not topdown_meta:
                self.logger.warning(f"No topdown metadata found in {metadata_file}")
                return None, None
            
            def world_to_pixel(world_x: float, world_z: float) -> Tuple[int, int]:
                """Convert world coordinates to pixel coordinates using topdown metadata"""
                origin = topdown_meta['origin_in_pixels']
                spacing = topdown_meta['spacing_in_meters_per_pixel']
                
                pixel_x = int(world_x / spacing + origin[0])
                pixel_y = int(world_z / spacing + origin[1])
                
                return pixel_x, pixel_y
            
            # Get target coordinates from both models
            target1 = episode.episode_data1.target_coordinate  # From Model 1's action.json
            target2 = episode.episode_data2.target_coordinate  # From Model 2's action.json
            
            # Handle different image formats from habitat-sim
            if len(image.shape) == 3:
                if image.shape[2] == 4:  # RGBA format
                    # Convert RGBA to RGB
                    rgb_image = image[:, :, :3]
                elif image.shape[2] == 3:  # RGB format
                    rgb_image = image
                else:
                    rgb_image = image
            else:
                rgb_image = image
            
            height, width = rgb_image.shape[:2]
            
            # Calculate 2.5m radius in pixels for cropping
            spacing = topdown_meta['spacing_in_meters_per_pixel']
            radius_pixels = int(2.5 / spacing)  # 2.5 meters in pixels
            
            # Calculate circle radius based on spacing - approximately 0.1 meters diameter (0.05m radius) in world coordinates
            # This ensures consistent real-world size regardless of resolution
            circle_radius_meters = 0.05  # 0.05 meters radius (0.1 meters diameter)
            circle_radius = max(1, int(circle_radius_meters / spacing))  # Convert to pixels, minimum 1 pixel
            
            # Create Model 1 image
            model1_image_path = None
            if target1 and len(target1) >= 2:
                tx1, tz1 = target1[0], target1[1]
                px1, py1 = world_to_pixel(tx1, tz1)
                
                self.logger.info(f"Model 1 target: world({tx1:.2f}, {tz1:.2f}) -> pixel({px1}, {py1})")
                
                if 0 <= px1 < width and 0 <= py1 < height:
                    # Create PIL image for drawing
                    pil_image1 = Image.fromarray(rgb_image)
                    draw1 = ImageDraw.Draw(pil_image1)
                    
                    # Draw circle with dynamic radius in blue
                    draw1.ellipse([px1-circle_radius, py1-circle_radius, px1+circle_radius, py1+circle_radius], 
                                fill=(0, 100, 255), outline=(255, 255, 255), width=2)

                    # Convert back to numpy
                    annotated_image1 = np.array(pil_image1)
                    
                    crop_left = max(0, px1 - radius_pixels)
                    crop_right = min(width, px1 + radius_pixels)
                    crop_top = max(0, py1 - radius_pixels)
                    crop_bottom = min(height, py1 + radius_pixels)
                    
                    # Crop the image
                    cropped_image1 = annotated_image1[crop_top:crop_bottom, crop_left:crop_right]
                    
                    # Save Model 1 image
                    model1_image_path = str(episode_output_dir / "model1_topdown.png")
                    bgr_image1 = cv2.cvtColor(cropped_image1, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(model1_image_path, bgr_image1)
                    
                    self.logger.info(f"Model 1 image saved: {model1_image_path}, crop size: {cropped_image1.shape}")
            
            # Create Model 2 image  
            model2_image_path = None
            if target2 and len(target2) >= 2:
                tx2, tz2 = target2[0], target2[1]
                px2, py2 = world_to_pixel(tx2, tz2)
                
                self.logger.info(f"Model 2 target: world({tx2:.2f}, {tz2:.2f}) -> pixel({px2}, {py2})")
                
                if 0 <= px2 < width and 0 <= py2 < height:
                    # Create PIL image for drawing
                    pil_image2 = Image.fromarray(rgb_image)
                    draw2 = ImageDraw.Draw(pil_image2)
                    
                    # Draw circle with dynamic radius in red
                    draw2.ellipse([px2-circle_radius, py2-circle_radius, px2+circle_radius, py2+circle_radius], 
                                fill=(255, 50, 50), outline=(255, 255, 255), width=2)
                    
                    # Convert back to numpy
                    annotated_image2 = np.array(pil_image2)
                    
                    # Calculate crop boundaries with 5m radius, handling edge cases
                    crop_left = max(0, px2 - radius_pixels)
                    crop_right = min(width, px2 + radius_pixels)
                    crop_top = max(0, py2 - radius_pixels)
                    crop_bottom = min(height, py2 + radius_pixels)
                    
                    # Crop the image
                    cropped_image2 = annotated_image2[crop_top:crop_bottom, crop_left:crop_right]
                    
                    # Save Model 2 image
                    model2_image_path = str(episode_output_dir / "model2_topdown.png")
                    bgr_image2 = cv2.cvtColor(cropped_image2, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(model2_image_path, bgr_image2)
                    
                    self.logger.info(f"Model 2 image saved: {model2_image_path}, crop size: {cropped_image2.shape}")
            
            return model1_image_path, model2_image_path
            
        except Exception as e:
            self.logger.error(f"Error creating model topdown images for {episode.scene_id}/{episode.episode_id}: {e}")
            return None, None
    
    def create_discrimination_prompt(self, episode: ControversialEpisode) -> str:
        """Create prompt for LLM discrimination with separate model images"""
        return f"""You are an expert navigation system evaluator. I need you to analyze a controversial episode where two different AI models disagreed on the success of an object navigation task.

**Task Details:**
- Target Object: {episode.episode_data1.target_name}

**Images Provided:**
You will see two separate images:
1. **First Image (Model 1)**: Shows a area around Model 1's target selection with a **BLUE** circle marking the chosen target location.
2. **Second Image (Model 2)**: Shows a area around Model 2's target selection with a **RED** circle marking the chosen target location.

Each image is cropped to show only the 5-meter radius around the respective model's target selection to focus your analysis on the immediate area.

**Your Task:**
Please analyze these navigation scenarios and determine which model made the better decision. Consider:

1. **Target Identification**: Which model identified a more plausible target location for "{episode.episode_data1.target_name}"? Look at the environment around each colored circle, find the corresponding node that is closer to the {episode.episode_data1.target_name}.

2. **Accessibility**: If two nodes are both at a plausible position of the {episode.episode_data1.target_name}, which target location appears more accessible and reachable in a real navigation scenario? (i.e. more close to the navigable/walkable area and more close to the open space. (e.g. Two chairs. The one that is closer to the open area is more suitable than the one that is closer to the wall. ))

**Color Coding:**
- **Blue Circle**: Model 1's target selection
- **Red Circle**: Model 2's target selection

**Please respond with:**
1. Your analysis of both models' target selections based on the environmental context
2. Which model you believe made the better decision (Model 1 or Model 2)
3. Key reasoning points for your decision

**Output Format:**
Decision: [Model 1 or Model 2]
"""

    def discriminate_episode(self, episode: ControversialEpisode) -> Dict[str, Any]:
        """Use LLM to discriminate a controversial episode"""
        self.logger.info(f"Discriminating episode {episode.scene_id}/{episode.episode_id}")
        
        # Generate separate topdown views for each model
        image_paths = self.create_model_topdown_images(episode)
        if not image_paths:
            return {
                "error": "Failed to render topdown views",
                "decision": None,
                "reasoning": "Could not generate topdown visualizations"
            }
        
        # Create discrimination prompt
        prompt = self.create_discrimination_prompt(episode)
        
        # Query LLM with both images
        response = self.llm_client.query(prompt, image_paths)
        
        # Parse response to extract decision
        decision = None
        reasoning = response
        
        if "Decision: Model 1" in response or "Decision: 1" in response:
            decision = "Model 1"
        elif "Decision: Model 2" in response or "Decision: 2" in response:
            decision = "Model 2"
        
        return {
            "scene_id": episode.scene_id,
            "episode_id": episode.episode_id,
            "model1_result": episode.model1_result.to_dict(),
            "model2_result": episode.model2_result.to_dict(),
            "model1_image": image_paths[0] if image_paths else None,
            "model2_image": image_paths[1] if image_paths else None,
            "llm_response": response,
            "decision": decision,
            "reasoning": reasoning,
            "target_object": episode.episode_data1.target_name
        }
    
    def load_controversial_episodes_from_file(self, file_path: str) -> List[ControversialEpisode]:
        """Load controversial episodes from an existing JSON file"""
        self.logger.info(f"Loading controversial episodes from {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            controversial_episodes = []
            
            # Handle different file formats
            if isinstance(data, list):
                # Direct list of episodes
                episodes_list = data
            elif isinstance(data, dict) and 'controversial_episodes' in data:
                # Nested format
                episodes_list = data['controversial_episodes']
            else:
                self.logger.error(f"Unexpected file format in {file_path}")
                return []
            
            for episode_data in episodes_list:
                try:
                    # Load episode data for both models
                    scene_id = episode_data['scene_id']
                    episode_id = episode_data['episode_id']
                    
                    episode_data1 = self.load_episode_data(self.model1_dir, scene_id, episode_id)
                    episode_data2 = self.load_episode_data(self.model2_dir, scene_id, episode_id)
                    
                    if episode_data1 and episode_data2:
                        # Create EpisodeResult objects from the JSON data
                        model1_result = EpisodeResult(
                            scene_id=scene_id,
                            episode_id=episode_id,
                            sr=episode_data.get('model1_sr', False),
                            spl=episode_data.get('model1_spl', 0.0),
                            success=episode_data.get('model1_success', False),
                            geodesic_distance_to_target=episode_data.get('model1_geodesic_distance', 0.0),
                            path_length=episode_data.get('model1_path_length', 0.0),
                            object_category=episode_data.get('object_category', 'unknown')
                        )
                        
                        model2_result = EpisodeResult(
                            scene_id=scene_id,
                            episode_id=episode_id,
                            sr=episode_data.get('model2_sr', False),
                            spl=episode_data.get('model2_spl', 0.0),
                            success=episode_data.get('model2_success', False),
                            geodesic_distance_to_target=episode_data.get('model2_geodesic_distance', 0.0),
                            path_length=episode_data.get('model2_path_length', 0.0),
                            object_category=episode_data.get('object_category', 'unknown')
                        )
                        
                        episode = ControversialEpisode(
                            scene_id=scene_id,
                            episode_id=episode_id,
                            model1_result=model1_result,
                            model2_result=model2_result,
                            episode_data1=episode_data1,
                            episode_data2=episode_data2
                        )
                        controversial_episodes.append(episode)
                    else:
                        self.logger.warning(f"Could not load episode data for {scene_id}/{episode_id}")
                        
                except Exception as e:
                    self.logger.error(f"Error loading episode {episode_data}: {e}")
                    continue
            
            self.logger.info(f"Successfully loaded {len(controversial_episodes)} controversial episodes")
            return controversial_episodes
            
        except Exception as e:
            self.logger.error(f"Error loading controversial episodes from {file_path}: {e}")
            return []
    
    def run_discrimination(self, controversial_episodes_file: Optional[str] = None) -> Dict[str, Any]:
        """Run discrimination on all controversial episodes"""
        self.logger.info("Starting discrimination process...")
        
        # Load controversial episodes from file or identify them
        if controversial_episodes_file:
            controversial_episodes = self.load_controversial_episodes_from_file(controversial_episodes_file)
        else:
            controversial_episodes = self.identify_controversial_episodes()
        
        if not controversial_episodes:
            self.logger.info("No controversial episodes found!")
            return {"controversial_episodes": [], "discriminated_results": []}
        
        self.logger.info(f"Processing {len(controversial_episodes)} controversial episodes...")
        
        discriminated_results = []
        
        for i, episode in enumerate(controversial_episodes, 1):
            self.logger.info(f"Processing episode {i}/{len(controversial_episodes)}: {episode.scene_id}/{episode.episode_id}")
            
            try:
                result = self.discriminate_episode(episode)
                discriminated_results.append(result)
                
                # Save individual episode result to its own directory
                scene_output_dir = self.output_dir / episode.scene_id
                episode_output_dir = scene_output_dir / episode.episode_id
                episode_output_dir.mkdir(parents=True, exist_ok=True)
                
                with open(episode_output_dir / "discrimination_result.json", 'w') as f:
                    json.dump(result, f, indent=2)
                
                # Save intermediate results
                temp_results = {
                    "controversial_episodes": len(controversial_episodes),
                    "processed": i,
                    "discriminated_results": discriminated_results
                }
                
                with open(self.output_dir / "discrimination_results_temp.json", 'w') as f:
                    json.dump(temp_results, f, indent=2)
                    
            except Exception as e:
                self.logger.error(f"Error processing episode {episode.scene_id}/{episode.episode_id}: {e}")
                discriminated_results.append({
                    "scene_id": episode.scene_id,
                    "episode_id": episode.episode_id,
                    "error": str(e),
                    "decision": None
                })
        
        # Save final results
        final_results = {
            "controversial_episodes": len(controversial_episodes),
            "discriminated_results": discriminated_results,
            "summary": self._generate_discrimination_summary(discriminated_results)
        }
        
        with open(self.output_dir / "discrimination_results.json", 'w') as f:
            json.dump(final_results, f, indent=2)
        
        self.logger.info("Discrimination process completed!")
        return final_results
    
    def _generate_discrimination_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics from discrimination results"""
        total = len(results)
        model1_wins = sum(1 for r in results if r.get('decision') == 'Model 1')
        model2_wins = sum(1 for r in results if r.get('decision') == 'Model 2')
        no_decision = total - model1_wins - model2_wins
        
        return {
            "total_episodes": total,
            "model1_wins": model1_wins,
            "model2_wins": model2_wins,
            "no_decision": no_decision,
            "model1_win_rate": model1_wins / total if total > 0 else 0,
            "model2_win_rate": model2_wins / total if total > 0 else 0
        }


def main():
    parser = argparse.ArgumentParser(description="Discriminator System for ObjectNav Model Comparison")
    parser.add_argument("--config_path", default="discriminator_config.json", 
                       help="Path to configuration file")
    parser.add_argument("--controversial_episodes", 
                       help="Path to existing controversial episodes JSON file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config_path)
    
    # Initialize and run discriminator system
    discriminator = DiscriminatorSystem(config)
    
    results = discriminator.run_discrimination(args.controversial_episodes)
    
    print(f"\n=== Discrimination Results ===")
    print(f"Total controversial episodes: {results['summary']['total_episodes']}")
    print(f"Model 1 wins: {results['summary']['model1_wins']} ({results['summary']['model1_win_rate']:.1%})")
    print(f"Model 2 wins: {results['summary']['model2_wins']} ({results['summary']['model2_win_rate']:.1%})")
    print(f"No decision: {results['summary']['no_decision']}")
    print(f"\nResults saved to: {config['output_config']['discriminator_output']}/discrimination_results.json")


if __name__ == "__main__":
    main()
