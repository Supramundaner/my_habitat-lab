#!/usr/bin/env python3
"""Ordered preprocessing pipeline for two-stage ObjectNav target selection."""

import json
import os
import sys
from typing import Any, Dict, Optional, Sequence
import traceback

# Import all pipeline stages.
if __package__:
    from .generate_topdown import generate_topdown_view
    from .generate_wall_mask import generate_wall_mask
    from .segment_rooms import segment_rooms
    from .select_room import select_room
    from .generate_navigation_graph import generate_navigation_graph
    from .select_navigation_node import select_navigation_node
    from .build_navigation_request import build_navigation_request
    from .config import load_preprocessing_config, safe_config_dict
else:
    from generate_topdown import generate_topdown_view
    from generate_wall_mask import generate_wall_mask
    from segment_rooms import segment_rooms
    from select_room import select_room
    from generate_navigation_graph import generate_navigation_graph
    from select_navigation_node import select_navigation_node
    from build_navigation_request import build_navigation_request
    from config import load_preprocessing_config, safe_config_dict

class PreprocessingPipeline:
    """Run the seven deterministic/LLM preprocessing stages in order."""
    
    def __init__(self, config_path: str):
        """Initialize the pipeline from a JSON configuration file."""
        self.config_path = config_path
        self.config = self._load_config()
        self.output_data = {
            "workflow_status": "initialized",
            "failed_stage": None,
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
            return load_preprocessing_config(self.config_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration from {self.config_path}: {e}")
    
    def _setup_output_directory(self):
        """Create output directory if it doesn't exist."""
        output_dir = self.config['output']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
    
    def _save_output(self):
        """Atomically save a strict, sanitized pipeline result."""
        output_json_path = os.path.join(self.output_dir, "output.json")
        temporary_path = f"{output_json_path}.tmp"
        try:
            with open(temporary_path, 'w', encoding='utf-8') as f:
                json.dump(
                    safe_config_dict(self.output_data),
                    f,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary_path, output_json_path)
            print(f"✓ Output data saved to: {output_json_path}")
        except Exception:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
            raise
    
    def _record_stage(
        self,
        stage_name: str,
        success: bool,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record one completed or failed preprocessing stage."""
        if success:
            self.output_data["steps_completed"].append(stage_name)
            if data:
                if "generated_files" in data:
                    self.output_data["generated_files"].update(data["generated_files"])
                if "llm_response" in data:
                    self.output_data["llm_responses"][stage_name] = data["llm_response"]
                if "results" in data:
                    self.output_data["final_results"].update(data["results"])
        else:
            if data and "error" in data:
                self.output_data["errors"].append(
                    f"{stage_name}: {data['error']}"
                )
    
    def _render_topdown(self) -> bool:
        """Step 0: Generate topdown view and metadata."""
        print("\\n" + "="*60)
        print("STEP 0: Generating topdown view and metadata")
        print("="*60)
        
        result = generate_topdown_view(self.config, self.output_dir)
        self._record_stage("topdown", True, result)
        return True

    
    def _generate_wall_mask(self) -> bool:
        """Step 1: Generate wall mask."""
        print("\\n" + "="*60)
        print("STEP 1: Generating wall mask")
        print("="*60)
        
        try:
            topdown_path = os.path.join(self.output_dir, "topdown_view.png")
            result = generate_wall_mask(topdown_path, self.output_dir)
            self._record_stage("wall_mask", True, result)
            return True
        except Exception as e:
            error_msg = f"Step 1 failed: {str(e)}"
            print(f"✗ {error_msg}")
            self._record_stage("wall_mask", False, {"error": error_msg})
            return False
    
    def _segment_rooms(self) -> bool:
        """Step 2: Room segmentation and annotation."""
        print("\\n" + "="*60)
        print("STEP 2: Room segmentation and annotation")
        print("="*60)
        
        try:
            topdown_path = os.path.join(self.output_dir, "topdown_view.png")
            wall_mask_path = os.path.join(self.output_dir, "wall_mask.png")
            metadata_path = os.path.join(self.output_dir, "metadata.json")
            
            result = segment_rooms(
                topdown_path, wall_mask_path, metadata_path, 
                self.config, self.output_dir
            )
            
            # Save step 2 intermediate results for step 3 to access
            step2_result_path = os.path.join(self.output_dir, "step2_results.json")
            with open(step2_result_path, 'w', encoding='utf-8') as f:
                json.dump(
                    result,
                    f,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                f.write("\n")
            print(f"✓ Step 2 intermediate results saved to: {step2_result_path}")
            
            self._record_stage("rooms", True, result)
            return True
        except Exception as e:
            error_msg = f"Step 2 failed: {str(e)}"
            print(f"✗ {error_msg}")
            self._record_stage("rooms", False, {"error": error_msg})
            return False
    
    def _select_room(self) -> bool:
        """Step 3: LLM room selection with retry logic."""
        print("\\n" + "="*60)
        print("STEP 3: LLM room selection")
        print("="*60)
        
        try:
            topdown_path = os.path.join(self.output_dir, "topdown_view.png")
            room_annotation_path = os.path.join(self.output_dir, "room_annotation.png")
            
            result = select_room(
                topdown_path, room_annotation_path, 
                self.config, self.output_dir
            )
            
            # 检查是否成功选择了有效房间
            selected_room = result["llm_response"]["selected_room"]
            attempts_made = result["llm_response"].get("attempts_made", 1)
            
            if selected_room is not None:
                print(f"✓ Successfully selected room {selected_room} after {attempts_made} attempts")
                self._record_stage("room_selection", True, result)
                return True
            else:
                error_msg = "Failed to select a valid room even with retries"
                print(f"✗ {error_msg}")
                self._record_stage("room_selection", False, {"error": error_msg})
                return False
                
        except Exception as e:
            error_msg = f"Step 3 failed: {str(e)}"
            print(f"✗ {error_msg}")
            self._record_stage("room_selection", False, {"error": error_msg})
            return False
    
    def _generate_navigation_graph(self) -> bool:
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
            self._record_stage("navigation_graph", True, result)
            return True
        except Exception as e:
            error_msg = f"Step 4 failed: {str(e)}"
            print(f"✗ {error_msg}")
            self._record_stage("navigation_graph", False, {"error": error_msg})
            return False
    
    def _select_navigation_node(self) -> bool:
        """Step 5: Navigation node selection with retry logic."""
        print("\\n" + "="*60)
        print("STEP 5: Navigation node selection")
        print("="*60)
        
        try:
            # Get room bounding box from step 2 results
            room_bboxes = self.output_data["final_results"].get("room_bounding_boxes", {})
            selected_room = self.output_data["llm_responses"].get(
                "room_selection", {}
            ).get("selected_room")
            
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
                self._record_stage("node_selection", True, result)
                return True
            else:
                error_msg = "Failed to select a valid node even with retries"
                print(f"✗ {error_msg}")
                self._record_stage("node_selection", False, {"error": error_msg})
                return False
                
        except Exception as e:
            error_msg = f"Step 5 failed: {str(e)}"
            print(f"✗ {error_msg}")
            self._record_stage("node_selection", False, {"error": error_msg})
            return False
    
    def _build_navigation_request(self) -> bool:
        """Step 6: Build the portable request consumed by navigation."""
        print("\\n" + "="*60)
        print("STEP 6: Generating action.json")
        print("="*60)
        
        try:
            result = build_navigation_request(self.config, self.output_dir)
            self._record_stage("navigation_request", True, result)
            return True
        except Exception as e:
            error_msg = f"Step 6 failed: {str(e)}"
            print(f"✗ {error_msg}")
            self._record_stage("navigation_request", False, {"error": error_msg})
            return False
    
    def run(self) -> bool:
        print("\\n" + "🚀" + "="*58 + "🚀")
        print("🎯 STARTING NAVIGATION TARGET SELECTION PIPELINE 🎯")
        print("🚀" + "="*58 + "🚀")
        
        stages = [
            ("topdown", self._render_topdown),
            ("wall_mask", self._generate_wall_mask),
            ("rooms", self._segment_rooms),
            ("room_selection", self._select_room),
            ("navigation_graph", self._generate_navigation_graph),
            ("node_selection", self._select_navigation_node),
            ("navigation_request", self._build_navigation_request),
        ]
        
        for stage_name, stage in stages:
            try:
                if not stage():
                    self.output_data["workflow_status"] = "failed"
                    self.output_data["failed_stage"] = stage_name
                    self._save_output()
                    return False
            except Exception as e:
                error_msg = f"Unexpected error in {stage_name}: {str(e)}"
                print(f"✗ {error_msg}")
                traceback.print_exc()
                self.output_data["errors"].append(error_msg)
                self.output_data["workflow_status"] = "failed"
                self.output_data["failed_stage"] = stage_name
                self._save_output()
                return False
        
        self.output_data["workflow_status"] = "completed_successfully"
        self._save_output()
        
        print("\\n" + "🎉" + "="*58 + "🎉")
        print("🎯 PIPELINE COMPLETED SUCCESSFULLY! 🎯")
        print("🎉" + "="*58 + "🎉")
        
        return True

def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run preprocessing and return a process-compatible status code."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        print("Usage: python3 -m reason_navi.preprocessing <config_path>")
        print(
            "Example: python3 -m reason_navi.preprocessing "
            "reason_navi/preprocessing/config.example.json"
        )
        return 1
    
    config_path = arguments[0]
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        return 1
    
    pipeline = PreprocessingPipeline(config_path)
    success = pipeline.run()
    
    return 0 if success else 1

if __name__ == "__main__":
    raise SystemExit(main())
