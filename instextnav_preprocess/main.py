#!/usr/bin/env python3
"""
Main workflow orchestrator for Text Navigation pipeline.
Coordinates all modules to complete the full workflow using text descriptions instead of images.
"""

import os
import sys
import json
from typing import Dict, Any

# Import all workflow modules
from step_0_render_goal_image import render_goal_image
from step_1_generate_topdown import generate_topdown_view
from step_2_generate_wall_mask import generate_wall_mask
from step_3_room_segmentation import perform_room_segmentation
from step_4_llm_room_selection import select_room_with_llm
from step_5_graph_generation import generate_navigation_graph
from step_6_node_selection import select_navigation_node
from step_7_path_planning import path_planning_step

# Import text description loader
from text_description_loader import TextDescriptionLoader

class TextNavigationOrchestrator:
    """Main workflow orchestrator class for Text Navigation."""
    
    def __init__(self, config_path: str):
        """Initialize the orchestrator with configuration."""
        self.config_path = config_path
        self.config = self._load_config()
        self.output_data = {
            "workflow_status": "initialized",
            "steps_completed": [],
            "generated_files": {},
            "llm_responses": {},
            "final_results": {},
            "errors": []
        }
        self._setup_output_directory()
        self._load_episode_data()
        self._load_text_descriptions()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration from {self.config_path}: {e}")
    
    def _setup_output_directory(self):
        """Create output directory if it doesn't exist."""
        output_dir = self.config['output']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
    
    def _load_episode_data(self):
        """Load episode data from episodes file."""
        episodes_file = self.config['scene_config']['episodes_file']
        episode_id = self.config['scene_config']['episode_id']
        
        try:
            with open(episodes_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.goals_data = data['goals']
            self.episodes_data = data['episodes']
            
            if episode_id >= len(self.episodes_data):
                raise ValueError(f"Episode ID {episode_id} is out of range. Available episodes: 0-{len(self.episodes_data)-1}")
            
            self.current_episode = self.episodes_data[episode_id]
            
            # Get goal information
            goal_object_id = self.current_episode['goal_object_id']
            goal_image_id = self.current_episode['goal_image_id']
            
            # Find the corresponding goal
            goal_key = None
            for key, goal in self.goals_data.items():
                if str(goal['object_id']) == str(goal_object_id):
                    goal_key = key
                    break
            
            if goal_key is None:
                raise ValueError(f"Goal object ID {goal_object_id} not found in goals data")
            
            self.goal_data = self.goals_data[goal_key]
            
            if goal_image_id >= len(self.goal_data['image_goals']):
                raise ValueError(f"Goal image ID {goal_image_id} is out of range for goal {goal_object_id}")
            
            self.goal_image_data = self.goal_data['image_goals'][goal_image_id]
            
            # Extract agent starting position from episode
            self.agent_start_position = self.current_episode['start_position']
            self.agent_start_rotation = self.current_episode['start_rotation']
            
            print(f"✓ Episode data loaded:")
            print(f"  - Episode ID: {episode_id}")
            print(f"  - Goal object: {self.goal_data['object_name']} (ID: {goal_object_id})")
            print(f"  - Goal image: {goal_image_id}")
            print(f"  - Agent start position: {self.agent_start_position}")
            print(f"  - Goal object category: {self.goal_data['object_category']}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load episode data: {e}")
    
    def _load_text_descriptions(self):
        """Load text descriptions for the target object."""
        try:
            # Get val_text.json path from config or use default
            val_text_path = self.config.get('val_text_path', 'val_text.json')
            if not os.path.isabs(val_text_path):
                val_text_path = os.path.join(os.path.dirname(self.config_path), val_text_path)
            
            # Initialize text description loader
            self.text_loader = TextDescriptionLoader(val_text_path)
            
            # Extract scene ID from scene path
            scene_path = self.config['scene_config']['scene_path']
            self.scene_id = self.text_loader.extract_scene_id_from_path(scene_path)
            
            # Get text description for the target object
            goal_object_id = self.current_episode['goal_object_id']
            object_category = self.goal_data['object_category']
            
            self.text_description = self.text_loader.create_combined_description(
                self.scene_id, str(goal_object_id), object_category
            )
            
            if self.text_description is None:
                print(f"⚠️ No text description found for object {goal_object_id} in scene {self.scene_id}")
                # Create a fallback description
                self.text_description = f"Target Object: {object_category} (ID: {goal_object_id})\n\nNo detailed description available."
            
            print(f"✓ Text description loaded for {self.scene_id}_{goal_object_id}")
            
        except Exception as e:
            print(f"⚠️ Failed to load text descriptions: {e}")
            # Create a minimal fallback
            goal_object_id = self.current_episode['goal_object_id']
            object_category = self.goal_data['object_category']
            self.text_description = f"Target Object: {object_category} (ID: {goal_object_id})"
            self.scene_id = "unknown"
    
    def _save_output(self):
        """Save the output data to JSON file."""
        output_json_path = os.path.join(self.output_dir, "output.json")
        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(self.output_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Output data saved to: {output_json_path}")
        except Exception as e:
            print(f"✗ Failed to save output data: {e}")
    
    def _update_step_status(self, step_name: str, success: bool, data: Dict[str, Any] = None):
        """Update the status of a workflow step."""
        if success:
            self.output_data["steps_completed"].append(step_name)
            if data:
                if "generated_files" in data:
                    self.output_data["generated_files"].update(data["generated_files"])
                if "llm_responses" in data:
                    self.output_data["llm_responses"].update(data["llm_responses"])
                if "results" in data:
                    if step_name not in self.output_data["final_results"]:
                        self.output_data["final_results"][step_name] = {}
                    self.output_data["final_results"][step_name].update(data["results"])
        else:
            error_msg = data.get("error", "Unknown error") if data else "Unknown error"
            self.output_data["errors"].append({"step": step_name, "error": error_msg})
    
    def run_step_0(self) -> bool:
        """Step 0: Skip image rendering for TextNav - use text descriptions instead."""
        print("\\n" + "="*60)
        print("STEP 0: Skipping image rendering for TextNav")
        print("="*60)
        
        # Create a placeholder "goal image" file with text description
        goal_text_path = os.path.join(self.output_dir, "goal_description.txt")
        try:
            with open(goal_text_path, 'w', encoding='utf-8') as f:
                f.write(self.text_description)
            
            result = {
                "generated_files": {
                    "goal_description": goal_text_path
                },
                "results": {
                    "text_description": self.text_description,
                    "scene_id": self.scene_id,
                    "object_id": self.current_episode['goal_object_id']
                }
            }
            
            print(f"✓ Text description saved to: {goal_text_path}")
            self._update_step_status("step_0_text_description", True, result)
            return True
            
        except Exception as e:
            print(f"✗ Failed to save text description: {e}")
            self._update_step_status("step_0_text_description", False, {"error": str(e)})
            return False

    
    def run_step_1(self) -> bool:
        """Step 1: Generate topdown view and metadata."""
        print("\\n" + "="*60)
        print("STEP 1: Generating topdown view and metadata")
        print("="*60)
        
        # Use the goal image position as the target coordinate for topdown generation
        # This ensures the topdown view is centered around the target area
        target_coordinate = self.goal_image_data['position']  # 3D world coordinate [x, y, z]
        
        print(f"🎯 Using goal image position as target coordinate: {target_coordinate}")
        
        try:
            result = generate_topdown_view(
                scene_path=self.config['scene_config']['scene_path'],
                target_floor=target_coordinate,  # Pass the full 3D coordinate
                custom_ortho_scale=self.config['scene_config'].get('custom_ortho_scale'),
                target_coverage=self.config['scene_config'].get('target_coverage', 0.9),
                draw_coordinates=self.config['scene_config'].get('draw_coordinates', False),
                resolution=self.config.get('resolution', [2048, 2048]),
                output_dir=self.output_dir
            )
            self._update_step_status("step_1_topdown_generation", True, result)
            return True
        except Exception as e:
            print(f"✗ Step 1 failed: {e}")
            self._update_step_status("step_1_topdown_generation", False, {"error": str(e)})
            return False
    
    def run_step_2(self) -> bool:
        """Step 2: Generate wall mask."""
        print("\\n" + "="*60)
        print("STEP 2: Generating wall mask")
        print("="*60)
        
        try:
            topdown_path = os.path.join(self.output_dir, "topdown_view.png")
            result = generate_wall_mask(topdown_path, self.output_dir)
            self._update_step_status("step_2_wall_mask_generation", True, result)
            return True
        except Exception as e:
            print(f"✗ Step 2 failed: {e}")
            self._update_step_status("step_2_wall_mask_generation", False, {"error": str(e)})
            return False
    
    def run_step_3(self) -> bool:
        """Step 3: Room segmentation and annotation."""
        print("\\n" + "="*60)
        print("STEP 3: Room segmentation and annotation")
        print("="*60)
        
        try:
            topdown_path = os.path.join(self.output_dir, "topdown_view.png")
            wall_mask_path = os.path.join(self.output_dir, "wall_mask.png")
            metadata_path = os.path.join(self.output_dir, "metadata.json")
            
            result = perform_room_segmentation(
                topdown_path, wall_mask_path, metadata_path,
                self.config, self.output_dir
            )
            self._update_step_status("step_3_room_segmentation", True, result)
            return True
        except Exception as e:
            print(f"✗ Step 3 failed: {e}")
            self._update_step_status("step_3_room_segmentation", False, {"error": str(e)})
            return False
    
    def run_step_4(self) -> bool:
        """Step 4: LLM room selection with text description."""
        print("\\n" + "="*60)
        print("STEP 4: LLM room selection (TextNav)")
        print("="*60)
        
        try:
            topdown_path = os.path.join(self.output_dir, "topdown_view.png")
            room_annotation_path = os.path.join(self.output_dir, "room_annotation.png")
            goal_description_path = os.path.join(self.output_dir, "goal_description.txt")
            
            # Update config to include goal object category and text description
            config_with_goal = self.config.copy()
            config_with_goal['scene_config'] = config_with_goal['scene_config'].copy()
            config_with_goal['scene_config']['goal_object'] = self.goal_data['object_category']
            config_with_goal['scene_config']['text_description'] = self.text_description
            config_with_goal['scene_config']['use_text_nav'] = True
            
            result = select_room_with_llm(
                topdown_path, room_annotation_path, goal_description_path,
                config_with_goal, self.output_dir, self.output_data
            )
            self._update_step_status("step_4_room_selection", True, result)
            return True
        except Exception as e:
            print(f"✗ Step 4 failed: {e}")
            self._update_step_status("step_4_room_selection", False, {"error": str(e)})
            return False
    
    def run_step_5(self) -> bool:
        """Step 5: Navigation graph generation."""
        print("\\n" + "="*60)
        print("STEP 5: Navigation graph generation")
        print("="*60)
        
        
        topdown_path = os.path.join(self.output_dir, "topdown_view.png")
        wall_mask_path = os.path.join(self.output_dir, "wall_mask.png")
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        
        result = generate_navigation_graph(
            topdown_path, wall_mask_path, metadata_path,
            self.config, self.output_dir
        )
        self._update_step_status("step_5_graph_generation", True, result)
        return True

    
    def run_step_6(self) -> bool:
        """Step 6: Navigation node selection with retry logic."""
        print("\\n" + "="*60)
        print("STEP 6: Navigation node selection")
        print("="*60)
        
        try:
            # Load room selection results
            room_log_path = os.path.join(self.output_dir, "llm_room_selection_log.json")
            with open(room_log_path, 'r', encoding='utf-8') as f:
                room_data = json.load(f)
            
            selected_room = room_data['final_selected_room']
            
            # Get room bounding box from orchestrator's internal data
            step3_results = self.output_data.get('final_results', {}).get('step_3_room_segmentation', {})
            room_bboxes = step3_results.get('room_bounding_boxes', {})
            
            if not room_bboxes:
                raise ValueError("No room bounding boxes found in Step 3 results")
            
            room_bbox = room_bboxes.get(str(selected_room))
            
            if room_bbox is None:
                raise ValueError(f"Room {selected_room} bounding box not found in available rooms: {list(room_bboxes.keys())}")
            
            print(f"✓ Found room {selected_room} bounding box: {room_bbox}")
            
            graph_path = os.path.join(self.output_dir, "graph_with_topdown_view.png")
            topdown_path = os.path.join(self.output_dir, "topdown_view.png")
            goal_description_path = os.path.join(self.output_dir, "goal_description.txt")
            
            # Update config to include goal object category and text description
            config_with_goal = self.config.copy()
            config_with_goal['scene_config'] = config_with_goal['scene_config'].copy()
            config_with_goal['scene_config']['goal_object'] = self.goal_data['object_category']
            config_with_goal['scene_config']['text_description'] = self.text_description
            config_with_goal['scene_config']['use_text_nav'] = True
            
            result = select_navigation_node(
                graph_path, topdown_path, goal_description_path, room_bbox,
                selected_room, config_with_goal, self.output_dir
            )
            self._update_step_status("step_6_node_selection", True, result)
            return True
        except Exception as e:
            print(f"✗ Step 6 failed: {e}")
            self._update_step_status("step_6_node_selection", False, {"error": str(e)})
            return False

    
    def run_step_7(self) -> bool:
        """Step 7: Generate action.json."""
        print("\\n" + "="*60)
        print("STEP 7: Generating action.json")
        print("="*60)
        
        try:
            # Update config with agent state
            config_with_agent = self.config.copy()
            config_with_agent['scene_config'] = config_with_agent['scene_config'].copy()
            config_with_agent['scene_config']['agent_position'] = self.agent_start_position
            config_with_agent['scene_config']['agent_rotation'] = self.agent_start_rotation
            config_with_agent['scene_config']['goal_object'] = self.goal_data['object_category']
            
            result = path_planning_step(config_with_agent, self.output_dir)
            self._update_step_status("step_7_action_generation", True, result)
            return True
        except Exception as e:
            print(f"✗ Step 7 failed: {e}")
            self._update_step_status("step_7_action_generation", False, {"error": str(e)})
            return False
    
    def run_workflow(self) -> bool:
        """Execute the complete Image-Instance Navigation workflow."""
        print("\\n" + "🚀" + "="*58 + "🚀")
        print("🎯 STARTING IMAGE-INSTANCE NAVIGATION WORKFLOW 🎯")
        print("🚀" + "="*58 + "🚀")
        
        steps = [
            self.run_step_0,  # Render goal image
            self.run_step_1,  # Generate topdown
            self.run_step_2,  # Generate wall mask
            self.run_step_3,  # Room segmentation
            self.run_step_4,  # LLM room selection
            self.run_step_5,  # Graph generation
            self.run_step_6,  # Node selection
            self.run_step_7,  # Generate action.json
        ]
        
        for i, step_func in enumerate(steps):
            step_num = i
            print(f"\\n🔄 Executing Step {step_num}...")
            
            if not step_func():
                print(f"\\n❌ Workflow failed at Step {step_num}")
                self.output_data["workflow_status"] = f"failed_at_step_{step_num}"
                self._save_output()
                return False
            
            print(f"✅ Step {step_num} completed successfully")
        
        self.output_data["workflow_status"] = "completed_successfully"
        self._save_output()
        
        print("\\n" + "🎉" + "="*58 + "🎉")
        print("🎯 IMAGE-INSTANCE NAVIGATION WORKFLOW COMPLETED! 🎯")
        print("🎉" + "="*58 + "🎉")
        
        return True

def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_path>")
        print("Example: python main.py input_config.json")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    orchestrator = TextNavigationOrchestrator(config_path)
    success = orchestrator.run_workflow()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
