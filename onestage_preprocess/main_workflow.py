#!/usr/bin/env python3
"""
Main workflow orchestrator for the navigation target selection pipeline.
Coordinates all modules to complete the full workflow.
"""

import os
import json
import sys
from typing import Dict, Any, Tuple, Optional
import traceback

# Import all workflow modules
from step_0_generate_topdown import generate_topdown_view
from step_1_generate_wall_mask import generate_wall_mask
from step_2_room_segmentation import perform_room_segmentation
from step_3_llm_room_selection import select_room_with_llm
from step_4_graph_generation import generate_navigation_graph
from step_5_node_selection import select_navigation_node
from step_6_path_planning import path_planning_step

class WorkflowOrchestrator:
    """Main workflow orchestrator class."""
    
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
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration from {self.config_path}: {e}")
    
    def _setup_output_directory(self):
        """Create output directory if it doesn't exist."""
        output_dir = self.config['output']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
    
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
                if "llm_response" in data:
                    self.output_data["llm_responses"][step_name] = data["llm_response"]
                if "results" in data:
                    self.output_data["final_results"].update(data["results"])
        else:
            if data and "error" in data:
                self.output_data["errors"].append(f"{step_name}: {data['error']}")
    
    def run_step_0(self) -> bool:
        """Step 0: Generate topdown view and metadata."""
        print("\\n" + "="*60)
        print("STEP 0: Generating topdown view and metadata")
        print("="*60)
        
        result = generate_topdown_view(self.config, self.output_dir)
        self._update_step_status("step_0_topdown_generation", True, result)
        return True

    
    def run_step_1(self) -> bool:
        """Step 1: Generate wall mask."""
        print("\\n" + "="*60)
        print("STEP 1: Generating wall mask")
        print("="*60)
        
        try:
            topdown_path = os.path.join(self.output_dir, "topdown_view.png")
            result = generate_wall_mask(topdown_path, self.output_dir)
            self._update_step_status("step_1_wall_mask", True, result)
            return True
        except Exception as e:
            error_msg = f"Step 1 failed: {str(e)}"
            print(f"✗ {error_msg}")
            self._update_step_status("step_1_wall_mask", False, {"error": error_msg})
            return False
    
    def run_step_2(self) -> bool:
        """Step 2: Room segmentation and annotation."""
        print("\\n" + "="*60)
        print("STEP 2: Room segmentation and annotation")
        print("="*60)
        
        try:
            topdown_path = os.path.join(self.output_dir, "topdown_view.png")
            wall_mask_path = os.path.join(self.output_dir, "wall_mask.png")
            metadata_path = os.path.join(self.output_dir, "metadata.json")
            
            result = perform_room_segmentation(
                topdown_path, wall_mask_path, metadata_path, 
                self.config, self.output_dir
            )
            
            # Save step 2 intermediate results for step 3 to access
            step2_result_path = os.path.join(self.output_dir, "step2_results.json")
            try:
                with open(step2_result_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"✓ Step 2 intermediate results saved to: {step2_result_path}")
            except Exception as e:
                print(f"⚠️ Warning: Could not save step 2 intermediate results: {e}")
            
            self._update_step_status("step_2_room_segmentation", True, result)
            return True
        except Exception as e:
            error_msg = f"Step 2 failed: {str(e)}"
            print(f"✗ {error_msg}")
            self._update_step_status("step_2_room_segmentation", False, {"error": error_msg})
            return False
    
    def run_step_3(self) -> bool:
        """Step 3: LLM room selection with retry logic."""
        print("\\n" + "="*60)
        print("STEP 3: LLM room selection")
        print("="*60)
        
        try:
            topdown_path = os.path.join(self.output_dir, "topdown_view.png")
            room_annotation_path = os.path.join(self.output_dir, "room_annotation.png")
            
            result = select_room_with_llm(
                topdown_path, room_annotation_path, 
                self.config, self.output_dir
            )
            
            # 检查是否成功选择了有效房间
            selected_room = result["llm_response"]["selected_room"]
            attempts_made = result["llm_response"].get("attempts_made", 1)
            
            if selected_room is not None:
                print(f"✓ Successfully selected room {selected_room} after {attempts_made} attempts")
                self._update_step_status("step_3_room_selection", True, result)
                return True
            else:
                error_msg = "Failed to select a valid room even with retries"
                print(f"✗ {error_msg}")
                self._update_step_status("step_3_room_selection", False, {"error": error_msg})
                return False
                
        except Exception as e:
            error_msg = f"Step 3 failed: {str(e)}"
            print(f"✗ {error_msg}")
            self._update_step_status("step_3_room_selection", False, {"error": error_msg})
            return False
    
    def run_step_4(self) -> bool:
        """Step 4: Navigation graph generation."""
        print("\\n" + "="*60)
        print("STEP 4: Navigation graph generation")
        print("="*60)
        
        try:
            topdown_path = os.path.join(self.output_dir, "topdown_view.png")
            wall_mask_path = os.path.join(self.output_dir, "wall_mask.png")
            metadata_path = os.path.join(self.output_dir, "metadata.json")
            
            result = generate_navigation_graph(
                topdown_path, wall_mask_path, metadata_path,
                self.config, self.output_dir
            )
            self._update_step_status("step_4_graph_generation", True, result)
            return True
        except Exception as e:
            error_msg = f"Step 4 failed: {str(e)}"
            print(f"✗ {error_msg}")
            self._update_step_status("step_4_graph_generation", False, {"error": error_msg})
            return False
    
    def run_step_5(self) -> bool:
        """Step 5: Navigation node selection with retry logic."""
        print("\\n" + "="*60)
        print("STEP 5: Navigation node selection")
        print("="*60)
        
        try:
            # Get room bounding box from step 2 results
            room_bboxes = self.output_data["final_results"].get("room_bounding_boxes", {})
            selected_room = self.output_data["llm_responses"].get("step_3_room_selection", {}).get("selected_room")
            
            if not selected_room or str(selected_room) not in room_bboxes:
                raise ValueError(f"Room {selected_room} bounding box not found")
            
            room_bbox = room_bboxes[str(selected_room)]
            graph_path = os.path.join(self.output_dir, "graph_with_topdown_view.png")
            topdown_path = os.path.join(self.output_dir, "topdown_view.png")
            
            result = select_navigation_node(
                graph_path, topdown_path, room_bbox, selected_room,
                self.config, self.output_dir
            )
            
            # 检查是否成功选择了有效节点
            selected_node_id = result["llm_response"]["selected_node_id"]
            attempts_made = result["llm_response"].get("attempts_made", 1)
            
            if selected_node_id is not None:
                print(f"✓ Successfully selected node {selected_node_id} after {attempts_made} attempts")
                self._update_step_status("step_5_node_selection", True, result)
                return True
            else:
                error_msg = "Failed to select a valid node even with retries"
                print(f"✗ {error_msg}")
                self._update_step_status("step_5_node_selection", False, {"error": error_msg})
                return False
                
        except Exception as e:
            error_msg = f"Step 5 failed: {str(e)}"
            print(f"✗ {error_msg}")
            self._update_step_status("step_5_node_selection", False, {"error": error_msg})
            return False
    
    def run_step_6(self) -> bool:
        """Step 6: Path planning from target_coordinate to selected node."""
        print("\\n" + "="*60)
        print("STEP 6: Generating action.json")
        print("="*60)
        
        try:
            result = path_planning_step(self.config, self.output_dir)
            self._update_step_status("step_6_path_planning", True, result)
            return True
        except Exception as e:
            error_msg = f"Step 6 failed: {str(e)}"
            print(f"✗ {error_msg}")
            self._update_step_status("step_6_path_planning", False, {"error": error_msg})
            return False
    
    def run_workflow(self) -> bool:
        print("\\n" + "🚀" + "="*58 + "🚀")
        print("🎯 STARTING NAVIGATION TARGET SELECTION WORKFLOW 🎯")
        print("🚀" + "="*58 + "🚀")
        
        steps = [
            self.run_step_0,
            self.run_step_1, 
            self.run_step_2,
            self.run_step_3,
            self.run_step_4,
            self.run_step_5,
            self.run_step_6
        ]
        
        for i, step_func in enumerate(steps):
            try:
                if not step_func():
                    self.output_data["workflow_status"] = f"failed_at_step_{i}"
                    self._save_output()
                    return False
            except Exception as e:
                error_msg = f"Unexpected error in step {i}: {str(e)}"
                print(f"✗ {error_msg}")
                traceback.print_exc()
                self.output_data["errors"].append(error_msg)
                self.output_data["workflow_status"] = f"failed_at_step_{i}"
                self._save_output()
                return False
        
        self.output_data["workflow_status"] = "completed_successfully"
        self._save_output()
        
        print("\\n" + "🎉" + "="*58 + "🎉")
        print("🎯 WORKFLOW COMPLETED SUCCESSFULLY! 🎯")
        print("🎉" + "="*58 + "🎉")
        
        return True

def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python main_workflow.py <config_path>")
        print("Example: python main_workflow.py input_config.json")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    orchestrator = WorkflowOrchestrator(config_path)
    success = orchestrator.run_workflow()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
