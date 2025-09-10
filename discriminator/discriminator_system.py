#!/usr/bin/env python3
"""
Discriminator System for ObjectNav Model Comparison

This script implements a comprehensive system to:
1. Identify controversial episodes where two models disagree
2. Use a discriminator LLM to re-evaluate these episodes 
3. Generate final statistics combining original and discriminated results

Usage:
    python discriminator_system.py --model1_dir /path/to/model1/output --model2_dir /path/to/model2/output
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
try:
    from discriminator.current_topdown import render_topdown_view
except ImportError:
    # Fallback to direct import if discriminator module not found
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'discriminator'))
    from current_topdown import render_topdown_view

# Import LLM client
import requests


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
        
    def query(self, prompt: str, image_path: Optional[str] = None) -> str:
        """Query the LLM with optional image"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        messages = []
        
        if image_path and os.path.exists(image_path):
            # For multimodal models, encode image as base64
            import base64
            with open(image_path, 'rb') as f:
                image_b64 = base64.b64encode(f.read()).decode()
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        }
                    }
                ]
            })
        else:
            messages.append({"role": "user", "content": prompt})
        
        payload = {
            'model': self.model,
            'messages': messages,
            'max_tokens': self.max_tokens,
            'temperature': 0.1
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
    
    def __init__(self, model1_dir: str, model2_dir: str, llm_config: Dict[str, Any], output_dir: str = "discriminator_output"):
        self.model1_dir = Path(model1_dir)
        self.model2_dir = Path(model2_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.llm_client = LLMClient(llm_config)
        
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
        """Load batch results from model directory"""
        batch_file = model_dir / "batch_output.json"
        if not batch_file.exists():
            raise FileNotFoundError(f"Batch output file not found: {batch_file}")
            
        with open(batch_file, 'r') as f:
            data = json.load(f)
        
        results = {}
        for episode_key, episode_data in data.get('episode_details', {}).items():
            scene_id, episode_id = episode_key.split('/')
            results[episode_key] = EpisodeResult(
                scene_id=scene_id,
                episode_id=episode_id,
                sr=bool(episode_data.get('sr', False)),
                spl=float(episode_data.get('spl', 0.0)),
                success=bool(episode_data.get('success', False)),
                geodesic_distance_to_target=float(episode_data.get('geodesic_distance_to_target', 0.0)),
                path_length=float(episode_data.get('path_length', 0.0)),
                object_category=str(episode_data.get('object_category', 'unknown'))
            )
        
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
    
    def render_episode_topdown(self, episode: ControversialEpisode) -> Optional[str]:
        """Render topdown view for an episode"""
        try:
            # Use the scene path from episode data
            scene_path = episode.episode_data1.scene_path
            agent_position = episode.episode_data1.agent_position
            
            # Render topdown view at the agent's floor
            image, coords, metadata = render_topdown_view(
                scene_path, 
                target_floor=agent_position,  # Use agent position to determine floor
                draw_coordinates=True,
                resolution=[1024, 1024]  # Smaller resolution for faster processing
            )
            
            if image is None:
                self.logger.error(f"Failed to render topdown for {episode.scene_id}/{episode.episode_id}")
                return None
            
            # Save the rendered image
            output_path = self.output_dir / f"{episode.scene_id}_{episode.episode_id}_topdown.png"
            
            # Convert to BGR for OpenCV if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                # If it's RGB, convert to BGR for OpenCV
                cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(str(output_path), image)
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error rendering topdown for {episode.scene_id}/{episode.episode_id}: {e}")
            return None
    
    def annotate_topdown_with_nodes(self, image_path: str, episode: ControversialEpisode) -> str:
        """Annotate topdown image with navigation nodes from both models"""
        try:
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            height, width = image.shape[:2]
            
            # Convert to PIL for easier text drawing
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
                small_font = ImageFont.truetype("DejaVuSans.ttf", 12)
            except:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Load metadata to convert world coordinates to pixel coordinates
            metadata_file = self.model1_dir / episode.scene_id / episode.episode_id / "preprocess" / "metadata.json"
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            def world_to_pixel(world_x: float, world_z: float) -> Tuple[int, int]:
                """Convert world coordinates to pixel coordinates"""
                origin = metadata['topdown_metadata']['origin_in_pixels']
                spacing = metadata['topdown_metadata']['spacing_in_meters_per_pixel']
                
                pixel_x = int(world_x / spacing + origin[0])
                pixel_y = int(world_z / spacing + origin[1])
                
                return pixel_x, pixel_y
            
            # Annotate Model 1 nodes (blue)
            if episode.episode_data1.navigation_nodes:
                for i, node in enumerate(episode.episode_data1.navigation_nodes):
                    if 'world_position' in node:
                        x, z = node['world_position'][0], node['world_position'][2]
                        px, py = world_to_pixel(x, z)
                        
                        if 0 <= px < width and 0 <= py < height:
                            # Draw circle for node
                            draw.ellipse([px-8, py-8, px+8, py+8], fill=(0, 100, 255), outline=(255, 255, 255), width=2)
                            draw.text((px+12, py-6), f"M1-{i}", fill=(0, 100, 255), font=small_font)
            
            # Annotate Model 2 nodes (red)
            if episode.episode_data2.navigation_nodes:
                for i, node in enumerate(episode.episode_data2.navigation_nodes):
                    if 'world_position' in node:
                        x, z = node['world_position'][0], node['world_position'][2]
                        px, py = world_to_pixel(x, z)
                        
                        if 0 <= px < width and 0 <= py < height:
                            # Draw circle for node
                            draw.ellipse([px-8, py-8, px+8, py+8], fill=(255, 50, 50), outline=(255, 255, 255), width=2)
                            draw.text((px+12, py+6), f"M2-{i}", fill=(255, 50, 50), font=small_font)
            
            # Annotate target coordinates
            target1 = episode.episode_data1.target_coordinate
            target2 = episode.episode_data2.target_coordinate
            
            # Convert target coordinates (assuming they're already in pixel space)
            if target1:
                tx1, ty1 = int(target1[0]), int(target1[1])
                if 0 <= tx1 < width and 0 <= ty1 < height:
                    draw.ellipse([tx1-12, ty1-12, tx1+12, ty1+12], fill=(0, 255, 0), outline=(255, 255, 255), width=3)
                    draw.text((tx1+15, ty1-15), "Target M1", fill=(0, 255, 0), font=font)
            
            if target2:
                tx2, ty2 = int(target2[0]), int(target2[1])
                if 0 <= tx2 < width and 0 <= ty2 < height:
                    draw.ellipse([tx2-12, ty2-12, tx2+12, ty2+12], fill=(255, 165, 0), outline=(255, 255, 255), width=3)
                    draw.text((tx2+15, ty2+15), "Target M2", fill=(255, 165, 0), font=font)
            
            # Add legend
            legend_y = 20
            draw.text((20, legend_y), "Legend:", fill=(255, 255, 255), font=font)
            draw.ellipse([20, legend_y+25, 35, legend_y+40], fill=(0, 100, 255), outline=(255, 255, 255))
            draw.text((45, legend_y+25), "Model 1 Nodes", fill=(255, 255, 255), font=small_font)
            draw.ellipse([20, legend_y+50, 35, legend_y+65], fill=(255, 50, 50), outline=(255, 255, 255))
            draw.text((45, legend_y+50), "Model 2 Nodes", fill=(255, 255, 255), font=small_font)
            draw.ellipse([20, legend_y+75, 35, legend_y+90], fill=(0, 255, 0), outline=(255, 255, 255))
            draw.text((45, legend_y+75), "Target M1", fill=(255, 255, 255), font=small_font)
            draw.ellipse([20, legend_y+100, 35, legend_y+115], fill=(255, 165, 0), outline=(255, 255, 255))
            draw.text((45, legend_y+100), "Target M2", fill=(255, 255, 255), font=small_font)
            
            # Convert back to OpenCV format and save
            annotated_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            annotated_path = image_path.replace('.png', '_annotated.png')
            cv2.imwrite(annotated_path, annotated_image)
            
            return annotated_path
            
        except Exception as e:
            self.logger.error(f"Error annotating image: {e}")
            return image_path  # Return original if annotation fails
    
    def create_discrimination_prompt(self, episode: ControversialEpisode) -> str:
        """Create prompt for LLM discrimination"""
        return f"""You are an expert navigation system evaluator. I need you to analyze a controversial episode where two different AI models disagreed on the success of an object navigation task.

**Task Details:**
- Scene: {episode.scene_id}
- Episode: {episode.episode_id}
- Target Object: {episode.episode_data1.target_name}

**Model Results:**
- Model 1: {'SUCCESS' if episode.model1_result.success else 'FAILURE'} (SR: {episode.model1_result.sr}, SPL: {episode.model1_result.spl:.3f})
- Model 2: {'SUCCESS' if episode.model2_result.success else 'FAILURE'} (SR: {episode.model2_result.sr}, SPL: {episode.model2_result.spl:.3f})

**Navigation Nodes:**
Model 1 had {len(episode.episode_data1.navigation_nodes)} navigation nodes.
Model 2 had {len(episode.episode_data2.navigation_nodes)} navigation nodes.

The attached image shows a top-down view of the scene with:
- Blue circles: Model 1 navigation nodes
- Red circles: Model 2 navigation nodes  
- Green circle: Model 1 target location
- Orange circle: Model 2 target location

**Your Task:**
Please analyze this navigation scenario and determine which model made the better decision. Consider:

1. **Target Identification**: Which model identified a more plausible target location for "{episode.episode_data1.target_name}"?

2. **Navigation Strategy**: Which model's navigation nodes show a more logical path planning approach?

3. **Spatial Reasoning**: Based on the room layout and obstacles, which model's approach seems more feasible?

4. **Goal Achievement**: Considering the task is to navigate to a "{episode.episode_data1.target_name}", which model's target selection and path planning is more likely to succeed?

**Please respond with:**
1. Your analysis of both models' approaches
2. Which model you believe made the better decision (Model 1 or Model 2)
3. Your confidence level (High/Medium/Low)
4. Key reasoning points for your decision

**Response Format:**
Analysis: [Your detailed analysis]
Decision: [Model 1 or Model 2]
Confidence: [High/Medium/Low]
Reasoning: [Key points that led to your decision]
"""

    def discriminate_episode(self, episode: ControversialEpisode) -> Dict[str, Any]:
        """Use LLM to discriminate a controversial episode"""
        self.logger.info(f"Discriminating episode {episode.scene_id}/{episode.episode_id}")
        
        # Render topdown view
        topdown_path = self.render_episode_topdown(episode)
        if not topdown_path:
            return {
                "error": "Failed to render topdown view",
                "decision": None,
                "reasoning": "Could not generate topdown visualization"
            }
        
        # Annotate with navigation nodes
        annotated_path = self.annotate_topdown_with_nodes(topdown_path, episode)
        
        # Create discrimination prompt
        prompt = self.create_discrimination_prompt(episode)
        
        # Query LLM
        response = self.llm_client.query(prompt, annotated_path)
        
        # Parse response to extract decision
        decision = None
        confidence = "Unknown"
        reasoning = response
        
        if "Decision: Model 1" in response or "Decision: 1" in response:
            decision = "Model 1"
        elif "Decision: Model 2" in response or "Decision: 2" in response:
            decision = "Model 2"
        
        if "Confidence: High" in response:
            confidence = "High"
        elif "Confidence: Medium" in response:
            confidence = "Medium"
        elif "Confidence: Low" in response:
            confidence = "Low"
        
        return {
            "scene_id": episode.scene_id,
            "episode_id": episode.episode_id,
            "model1_result": episode.model1_result.to_dict(),
            "model2_result": episode.model2_result.to_dict(),
            "topdown_image": annotated_path,
            "llm_response": response,
            "decision": decision,
            "confidence": confidence,
            "reasoning": reasoning,
            "target_object": episode.episode_data1.target_name
        }
    
    def run_discrimination(self) -> Dict[str, Any]:
        """Run discrimination on all controversial episodes"""
        self.logger.info("Starting discrimination process...")
        
        # Identify controversial episodes
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
        
        high_confidence = sum(1 for r in results if r.get('confidence') == 'High')
        medium_confidence = sum(1 for r in results if r.get('confidence') == 'Medium')
        low_confidence = sum(1 for r in results if r.get('confidence') == 'Low')
        
        return {
            "total_episodes": total,
            "model1_wins": model1_wins,
            "model2_wins": model2_wins,
            "no_decision": no_decision,
            "model1_win_rate": model1_wins / total if total > 0 else 0,
            "model2_win_rate": model2_wins / total if total > 0 else 0,
            "high_confidence_decisions": high_confidence,
            "medium_confidence_decisions": medium_confidence,
            "low_confidence_decisions": low_confidence
        }


def main():
    parser = argparse.ArgumentParser(description="Discriminator System for ObjectNav Model Comparison")
    parser.add_argument("--model1_dir", required=True, help="Path to Model 1 output directory")
    parser.add_argument("--model2_dir", required=True, help="Path to Model 2 output directory")
    parser.add_argument("--output_dir", default="discriminator_output", help="Output directory for results")
    parser.add_argument("--config", help="Path to configuration file (optional)")
    
    args = parser.parse_args()
    
    # Default LLM configuration
    llm_config = {
        "api_key": "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",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "google/gemini-2.5-pro",
        "max_tokens": 35000
    }
    
    # Load custom config if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            llm_config.update(custom_config.get('llm_config', {}))
    
    # Initialize and run discriminator system
    discriminator = DiscriminatorSystem(
        model1_dir=args.model1_dir,
        model2_dir=args.model2_dir,
        llm_config=llm_config,
        output_dir=args.output_dir
    )
    
    results = discriminator.run_discrimination()
    
    print(f"\n=== Discrimination Results ===")
    print(f"Total controversial episodes: {results['summary']['total_episodes']}")
    print(f"Model 1 wins: {results['summary']['model1_wins']} ({results['summary']['model1_win_rate']:.1%})")
    print(f"Model 2 wins: {results['summary']['model2_wins']} ({results['summary']['model2_win_rate']:.1%})")
    print(f"No decision: {results['summary']['no_decision']}")
    print(f"\nResults saved to: {args.output_dir}/discrimination_results.json")


if __name__ == "__main__":
    main()
