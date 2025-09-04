#!/usr/bin/env python3
"""
Main workflow orchestrator for the navigation target selection pipeline.
Coordinates all modules to complete the full workflow with multi-point selection support.
"""

import os
import json
import sys
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import traceback

# Import all workflow modules
from step_0_generate_topdown import generate_topdown_view
from step_1_generate_wall_mask import generate_wall_mask
from step_2_room_segmentation import perform_room_segmentation
from step_3_llm_room_selection import select_room_with_llm
from step_4_graph_generation import generate_navigation_graph
from step_5_node_selection import select_navigation_node
from step_6_path_planning import path_planning_step

# Import multi-point utilities
from multi_point_utils import (
    validate_k_points_config, 
    save_marked_images,
    create_multi_point_summary,
    mark_selected_node
)

class WorkflowOrchestrator:
    """Main workflow orchestrator class."""
    
    def __init__(self, config_path: str):
        """Initialize the orchestrator with configuration."""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Validate k_points configuration
        self.k_points = validate_k_points_config(self.config)
        
        self.output_data = {
            "workflow_status": "initialized",
            "steps_completed": [],
            "generated_files": {},
            "llm_responses": {},
            "final_results": {},
            "errors": [],
            "multi_point_results": {
                "k_points": self.k_points,
                "iterations": []
            }
        }
        self._setup_output_directory()
        
        # Initialize storage for original images (will be loaded after Step 0, 2)
        self.original_topdown = None
        self.original_room_annotation = None
    
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
        
        # Load and store the original topdown image for multi-point marking
        topdown_path = os.path.join(self.output_dir, "topdown_view.png")
        if os.path.exists(topdown_path):
            self.original_topdown = cv2.imread(topdown_path)
            print(f"✓ Original topdown image loaded for multi-point marking")
        else:
            print(f"⚠️  Warning: Could not load original topdown image from {topdown_path}")
        
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
            
            # Load and store the original room annotation image for multi-point marking
            room_annotation_path = os.path.join(self.output_dir, "room_annotation.png")
            if os.path.exists(room_annotation_path):
                self.original_room_annotation = cv2.imread(room_annotation_path)
                print(f"✓ Original room annotation image loaded for multi-point marking")
            else:
                print(f"⚠️  Warning: Could not load original room annotation image from {room_annotation_path}")
            
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
    
    def run_step_3_iteration(self, iteration: int, selected_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run Step 3 (room selection) for a specific iteration with marked images."""
        print(f"\\n🏠 STEP 3 - ITERATION {iteration + 1}: Room selection with marked context")
        
        try:
            # Save marked images for this iteration
            marked_topdown_path, marked_room_annotation_path = save_marked_images(
                self.original_topdown, self.original_room_annotation,
                selected_results, iteration, self.config, self.output_dir
            )
            
            # Call room selection with marked images
            result = select_room_with_llm(
                marked_topdown_path, marked_room_annotation_path,
                self.config, self.output_dir, iteration, selected_results
            )
            
            # Check if room selection was successful
            selected_room = result["llm_response"]["selected_room"]
            attempts_made = result["llm_response"].get("attempts_made", 1)
            
            if selected_room is not None:
                print(f"✓ Successfully selected room {selected_room} after {attempts_made} attempts (iteration {iteration + 1})")
                return result
            else:
                raise RuntimeError("Failed to select a valid room even with retries")
                
        except Exception as e:
            error_msg = f"Step 3 iteration {iteration + 1} failed: {str(e)}"
            print(f"✗ {error_msg}")
            raise RuntimeError(error_msg)
    
    def run_step_5_iteration(self, iteration: int, room_result: Dict[str, Any], 
                           selected_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run Step 5 (node selection) for a specific iteration with marked context."""
        print(f"\\n🎯 STEP 5 - ITERATION {iteration + 1}: Node selection with marked context")
        
        try:
            # Get room bounding box from step 2 results
            room_bboxes = self.output_data["final_results"].get("room_bounding_boxes", {})
            selected_room = room_result["llm_response"]["selected_room"]
            
            if str(selected_room) not in room_bboxes:
                raise ValueError(f"Room {selected_room} bounding box not found")
            
            room_bbox = room_bboxes[str(selected_room)]
            graph_path = os.path.join(self.output_dir, "graph_with_topdown_view.png")
            
            # Use the marked topdown from this iteration
            marked_topdown_path = os.path.join(self.output_dir, f"topdown_marked_iter_{iteration}.png")
            
            # Call node selection with marked topdown and selected results
            result = select_navigation_node(
                graph_path, marked_topdown_path, room_bbox, selected_room,
                self.config, self.output_dir, iteration, selected_results
            )
            
            # Check if node selection was successful
            selected_node_id = result["llm_response"]["selected_node_id"]
            attempts_made = result["llm_response"].get("attempts_made", 1)
            
            if selected_node_id is not None:
                print(f"✓ Successfully selected node {selected_node_id} after {attempts_made} attempts (iteration {iteration + 1})")
                return result
            else:
                raise RuntimeError("Failed to select a valid node even with retries")
                
        except Exception as e:
            error_msg = f"Step 5 iteration {iteration + 1} failed: {str(e)}"
            print(f"✗ {error_msg}")
            raise RuntimeError(error_msg)
    
    def run_step_6_iteration(self, iteration: int, node_result: Dict[str, Any]) -> Dict[str, Any]:
        """Run Step 6 (action generation) for a specific iteration."""
        print(f"\\n📋 STEP 6 - ITERATION {iteration + 1}: Generating action file")
        
        try:
            # Generate action file with iteration suffix
            result = path_planning_step(self.config, self.output_dir, iteration)
            print(f"✓ Successfully generated action file for iteration {iteration + 1}")
            return result
        except Exception as e:
            error_msg = f"Step 6 iteration {iteration + 1} failed: {str(e)}"
            print(f"✗ {error_msg}")
            raise RuntimeError(error_msg)
    
    def run_multi_point_iterations(self) -> List[Dict[str, Any]]:
        """Run the iterative multi-point selection process."""
        print(f"\\n🔄 STARTING MULTI-POINT SELECTION ({self.k_points} points)")
        print("="*60)
        
        selected_results = []
        
        for iteration in range(self.k_points):
            print(f"\\n{'='*20} ITERATION {iteration + 1}/{self.k_points} {'='*20}")
            
            try:
                # Step 3: Room selection with marked context
                room_result = self.run_step_3_iteration(iteration, selected_results)
                
                # Step 5: Node selection with marked context  
                node_result = self.run_step_5_iteration(iteration, room_result, selected_results)
                
                # Step 6: Generate action file
                action_result = self.run_step_6_iteration(iteration, node_result)
                
                # Record results for this iteration
                # Add standardized selected_node field for mark_selected_node compatibility
                standardized_node_result = node_result.copy()
                if 'llm_response' in node_result and 'selected_node_data' in node_result['llm_response']:
                    selected_node_data = node_result['llm_response']['selected_node_data']
                    standardized_node_result['selected_node'] = {
                        'pixel_coordinates': selected_node_data.get('pixel_coordinates'),
                        'node_id': selected_node_data.get('node_id')
                    }
                    print(f"✓ Added standardized selected_node field for iteration {iteration + 1}")
                
                iteration_result = {
                    'iteration': iteration + 1,
                    'room_result': room_result,
                    'node_result': standardized_node_result,
                    'action_result': action_result
                }
                selected_results.append(iteration_result)
                
                # Update multi-point results in output data
                self.output_data["multi_point_results"]["iterations"].append({
                    "iteration": iteration + 1,
                    "room_selected": room_result["llm_response"]["selected_room"],
                    "node_selected": node_result["llm_response"]["selected_node_id"],
                    "action_file": os.path.basename(action_result["generated_files"]["action_json"])
                })
                
                print(f"✅ ITERATION {iteration + 1} COMPLETED SUCCESSFULLY")
                
            except Exception as e:
                error_msg = f"Multi-point iteration {iteration + 1} failed: {str(e)}"
                print(f"❌ {error_msg}")
                self.output_data["errors"].append(error_msg)
                # Continue with next iteration instead of failing completely
                continue
        
        return selected_results
    
    def run_workflow(self) -> bool:
        print("\\n" + "🚀" + "="*58 + "🚀")
        print("🎯 STARTING MULTI-POINT NAVIGATION TARGET SELECTION WORKFLOW 🎯")
        print(f"🎯 K_POINTS = {self.k_points} 🎯")
        print("🚀" + "="*58 + "🚀")
        
        try:
            # Phase 1: Run base steps once (Steps 0, 1, 2, 4)
            print("\\n📋 PHASE 1: BASE SETUP (Steps 0, 1, 2, 4)")
            print("="*60)
            
            base_steps = [
                ("Step 0", self.run_step_0),
                ("Step 1", self.run_step_1), 
                ("Step 2", self.run_step_2),
                ("Step 4", self.run_step_4)
            ]
            
            for step_name, step_func in base_steps:
                try:
                    print(f"\\n▶️  Running {step_name}...")
                    if not step_func():
                        self.output_data["workflow_status"] = f"failed_at_{step_name.lower().replace(' ', '_')}"
                        self._save_output()
                        return False
                    print(f"✅ {step_name} completed successfully")
                except Exception as e:
                    error_msg = f"Unexpected error in {step_name}: {str(e)}"
                    print(f"❌ {error_msg}")
                    traceback.print_exc()
                    self.output_data["errors"].append(error_msg)
                    self.output_data["workflow_status"] = f"failed_at_{step_name.lower().replace(' ', '_')}"
                    self._save_output()
                    return False
            
            # Validate that we have the required images for multi-point marking
            if self.original_topdown is None or self.original_room_annotation is None:
                error_msg = "Failed to load original images required for multi-point marking"
                print(f"❌ {error_msg}")
                self.output_data["errors"].append(error_msg)
                self.output_data["workflow_status"] = "failed_missing_images"
                self._save_output()
                return False
            
            print("✅ PHASE 1 COMPLETED: Base setup successful")
            
            # Phase 2: Run multi-point iterations (Steps 3, 5, 6 repeated k times)
            print("\\n📋 PHASE 2: MULTI-POINT SELECTION")
            print("="*60)
            
            selected_results = self.run_multi_point_iterations()
            
            if not selected_results:
                error_msg = "No successful iterations completed"
                print(f"❌ {error_msg}")
                self.output_data["errors"].append(error_msg)
                self.output_data["workflow_status"] = "failed_no_iterations"
                self._save_output()
                return False
            
            # Phase 3: Generate summary
            print("\\n📋 PHASE 3: GENERATING SUMMARY")
            print("="*60)
            
            summary_path = create_multi_point_summary(selected_results, self.config, self.output_dir)
            self.output_data["generated_files"]["multi_point_summary"] = summary_path
            
            # Update final status
            successful_iterations = len(selected_results)
            self.output_data["workflow_status"] = "completed_successfully"
            self.output_data["multi_point_results"]["successful_iterations"] = successful_iterations
            self.output_data["multi_point_results"]["total_requested"] = self.k_points
            
            self._save_output()
            
            print("\\n" + "🎉" + "="*58 + "🎉")
            print("🎯 MULTI-POINT WORKFLOW COMPLETED SUCCESSFULLY! 🎯")
            print(f"🎯 GENERATED {successful_iterations}/{self.k_points} NAVIGATION TARGETS 🎯")
            print("🎉" + "="*58 + "🎉")
            
            return True
            
        except Exception as e:
            error_msg = f"Unexpected workflow error: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            self.output_data["errors"].append(error_msg)
            self.output_data["workflow_status"] = "failed_unexpected_error"
            self._save_output()
            return False

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
