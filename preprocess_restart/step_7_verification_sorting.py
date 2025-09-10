#!/usr/bin/env python3
"""
Step 7: Candidate verification and sorting using LLM.
Verifies selected navigation candidates and ranks them by probability of finding the target.
"""

import os
import json
import cv2
import numpy as np
import re
from typing import Dict, Any, List, Tuple, Optional

try:
    from byteplussdkarkruntime import Ark
    import base64
except ImportError:
    print("Warning: volcenginesdkarkruntime not found. Please install them with: pip install volcenginesdkarkruntime")
    Ark = None


def encode_image(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def extract_candidate_info(multi_point_summary_path: str) -> List[Dict[str, Any]]:
    """
    Extract candidate information from multi_point_summary.json
    
    Returns:
        List of candidate info with node_id, iteration, etc.
    """
    try:
        with open(multi_point_summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        candidates = []
        for iteration_data in summary.get("iterations", []):
            candidate = {
                "iteration": iteration_data["iteration"],
                "node_id": iteration_data["node_selection"]["selected_node_id"],
                "room_id": iteration_data["room_selection"]["selected_room"]
            }
            candidates.append(candidate)
        
        print(f"✓ Extracted {len(candidates)} candidates from summary")
        return candidates
        
    except Exception as e:
        print(f"✗ Error extracting candidate info: {e}")
        return []


def extract_candidate_info_from_results(selected_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract candidate information from selected_results directly.
    
    Args:
        selected_results: List of selection results from iterations
        
    Returns:
        List of candidate info with node_id, iteration, etc.
    """
    try:
        candidates = []
        for result in selected_results:
            iteration = result.get("iteration", 1)
            node_result = result.get("node_result", {})
            room_result = result.get("room_result", {})
            
            # Extract node ID from node selection
            node_selection = node_result.get("llm_response", {})
            node_id = node_selection.get("selected_node_id")
            
            # Extract room ID from room selection  
            room_selection = room_result.get("llm_response", {})
            room_id = room_selection.get("selected_room")
            
            if node_id is not None:
                candidate = {
                    "iteration": iteration,
                    "node_id": node_id,
                    "room_id": room_id
                }
                candidates.append(candidate)
                
        print(f"✓ Extracted {len(candidates)} candidates from selected_results")
        return candidates
        
    except Exception as e:
        print(f"✗ Error extracting candidate info from results: {e}")
        return []


def create_verification_image(original_topdown_path: str, candidates: List[Dict[str, Any]], 
                            navigation_nodes_path: str, output_dir: str) -> str:
    """
    Create an image with all candidate points clearly marked with indices.
    
    Args:
        original_topdown_path: Path to original topdown image
        candidates: List of candidate information
        navigation_nodes_path: Path to navigation nodes JSON file
        output_dir: Output directory
        
    Returns:
        Path to the verification image
    """
    # Load original topdown image
    image = cv2.imread(original_topdown_path)
    if image is None:
        raise ValueError(f"Could not load topdown image: {original_topdown_path}")
    
    # Load navigation nodes to get pixel coordinates
    try:
        with open(navigation_nodes_path, 'r', encoding='utf-8') as f:
            nodes_data = json.load(f)
        nodes = nodes_data.get("nodes", [])
        print(f"✓ Loaded {len(nodes)} navigation nodes")
    except Exception as e:
        raise ValueError(f"Could not load navigation nodes: {e}")
    
    # Create node lookup by ID
    node_lookup = {node["node_id"]: node for node in nodes}
    print(f"✓ Created node lookup with {len(node_lookup)} entries")
    
    # Define colors for different points (cycling through if more candidates than colors)
    colors = [
        (0, 0, 255),    # Red
        (255, 0, 0),    # Blue  
        (0, 255, 0),    # Green
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0)   # Orange
    ]
    
    print(f"🎨 Marking {len(candidates)} candidate points on verification image")
    
    # Mark each candidate point
    for i, candidate in enumerate(candidates):
        node_id = candidate["node_id"]
        
        if node_id not in node_lookup:
            print(f"⚠️  Warning: Node {node_id} not found in navigation nodes")
            continue
        
        node = node_lookup[node_id]
        pixel_coords = node.get("pixel_coordinates")
        
        if not pixel_coords or len(pixel_coords) < 2:
            print(f"⚠️  Warning: No valid pixel coordinates for node {node_id}")
            continue
        
        x, y = int(pixel_coords[0]), int(pixel_coords[1])
        color = colors[i % len(colors)]
        
        # Draw large circle marker
        cv2.circle(image, (x, y), 15, color, -1)  # -1 means filled
        # Center dot (for precise positioning)
        cv2.circle(image, (x, y), 3, (0, 0, 0), -1)
        
        # Draw node ID number above the circle with high visibility
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_thickness = 4
        text = str(node_id)
        
        # Calculate text size and position
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = x - text_size[0] // 2
        text_y = y - 35
        
        # Draw text with white outline + black shadow for maximum visibility
        cv2.putText(image, text, (text_x+1, text_y+1), font, font_scale, (0, 0, 0), font_thickness + 1)  # Shadow
        cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)     # Outline
        cv2.putText(image, text, (text_x, text_y), font, font_scale, color, 1)                            # Colored text
        
        print(f"✓ Marked candidate {i+1}: Node {node_id} at ({x}, {y}) with color {color}")
    
    # Save verification image
    verification_image_path = os.path.join(output_dir, "verification_candidates.png")
    cv2.imwrite(verification_image_path, image)
    print(f"✓ Verification image saved: {verification_image_path}")
    
    return verification_image_path


def load_verification_prompt(prompt_path: str, goal_object: str, candidate_indices: List[int]) -> str:
    """
    Load and format the verification prompt template.
    """
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        
        # Format the prompt with actual values
        candidate_indices_str = ", ".join(map(str, candidate_indices))
        
        formatted_prompt = prompt_template.format(
            goal_object=goal_object,
            candidate_indices=candidate_indices_str
        )
        
        return formatted_prompt
        
    except Exception as e:
        raise ValueError(f"Could not load verification prompt: {e}")


def call_llm_for_verification(client, verification_image_path: str, prompt_text: str, model: str) -> Dict:
    """Call LLM API for candidate verification."""
    verification_image_base64 = encode_image(verification_image_path)
    
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/png;base64,{verification_image_base64}"
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                ],
            },
        ],
    )
    return response


def parse_verification_response(response_text: str, valid_indices: List[int]) -> Tuple[List[int], Dict[str, Any]]:
    """
    Parse LLM verification response with multiple fallback strategies.
    
    Args:
        response_text: Raw LLM response
        valid_indices: List of valid candidate indices
        
    Returns:
        Tuple of (verified_node_ids, parsing_details)
    """
    parsing_details = {
        "raw_response": response_text,
        "parsing_method": None,
        "warnings": []
    }
    
    # Method 1: Try direct JSON parsing
    try:
        response_json = json.loads(response_text.strip())
        final_ranking = response_json.get("final_ranking", [])
        
        # Validate that all indices are in valid_indices
        validated_ranking = [idx for idx in final_ranking if idx in valid_indices]
        
        if len(validated_ranking) != len(final_ranking):
            invalid_indices = [idx for idx in final_ranking if idx not in valid_indices]
            parsing_details["warnings"].append(f"Removed invalid indices: {invalid_indices}")
        
        parsing_details["parsing_method"] = "direct_json"
        parsing_details["analysis"] = response_json.get("analysis", {})
        
        print(f"✓ Successfully parsed JSON response: {validated_ranking}")
        return validated_ranking, parsing_details
        
    except json.JSONDecodeError as e:
        parsing_details["warnings"].append(f"JSON parsing failed: {str(e)}")
    
    # Method 2: Extract JSON fragment from response
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            response_json = json.loads(json_match.group())
            final_ranking = response_json.get("final_ranking", [])
            
            validated_ranking = [idx for idx in final_ranking if idx in valid_indices]
            parsing_details["parsing_method"] = "extracted_json"
            parsing_details["analysis"] = response_json.get("analysis", {})
            
            print(f"✓ Extracted and parsed JSON fragment: {validated_ranking}")
            return validated_ranking, parsing_details
            
    except Exception as e:
        parsing_details["warnings"].append(f"JSON extraction failed: {str(e)}")
    
    # Method 3: Regular expression to find list pattern
    try:
        list_match = re.search(r'\[([0-9,\s]+)\]', response_text)
        if list_match:
            numbers_str = list_match.group(1)
            numbers = [int(x.strip()) for x in numbers_str.split(',') if x.strip().isdigit()]
            validated_ranking = [n for n in numbers if n in valid_indices]
            
            parsing_details["parsing_method"] = "regex_list"
            print(f"✓ Extracted list via regex: {validated_ranking}")
            return validated_ranking, parsing_details
            
    except Exception as e:
        parsing_details["warnings"].append(f"Regex parsing failed: {str(e)}")
    
    # Method 4: Fallback - return first candidate
    parsing_details["parsing_method"] = "fallback_first"
    parsing_details["warnings"].append("All parsing methods failed, using fallback strategy")
    
    fallback_result = [valid_indices[0]] if valid_indices else []
    print(f"⚠️  Using fallback strategy: {fallback_result}")
    
    return fallback_result, parsing_details


def verify_and_sort_candidates(config: Dict[str, Any], output_dir: str, 
                               selected_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Main function to verify and sort navigation candidates using LLM.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory path
        
    Returns:
        Dictionary containing verification results
    """
    print("🔍 Starting candidate verification and sorting")
    
    try:
        # Extract paths and configuration
        original_topdown_path = os.path.join(output_dir, "topdown_view.png")
        navigation_nodes_path = os.path.join(output_dir, "navigation_nodes.json")
        verification_prompt_path = config["prompts"]["verify_candidates_prompt"]
        goal_object = config["scene_config"]["goal_object"]
        
        # Extract candidate information from selected_results
        candidates = extract_candidate_info_from_results(selected_results)
        if not candidates:
            raise ValueError("No candidates found in selected_results")
        
        candidate_indices = [c["node_id"] for c in candidates]
        print(f"📝 Candidate node IDs: {candidate_indices}")
        
        # Create verification image
        verification_image_path = create_verification_image(
            original_topdown_path, candidates, navigation_nodes_path, output_dir
        )
        
        # Load and format prompt
        prompt_text = load_verification_prompt(
            verification_prompt_path, goal_object, candidate_indices
        )
        
        # Initialize LLM client
        if Ark is None:
            raise ImportError("Ark client not available")
        
        client = Ark(
            api_key=config["llm_config"]["api_key"],
            base_url=config["llm_config"]["base_url"]
        )
        
        # Call LLM for verification
        print("🤖 Calling LLM for candidate verification...")
        print(f"📝 Goal object: {goal_object}")
        print(f"📝 Candidate indices: {candidate_indices}")
        print(f"📝 Verification image: {verification_image_path}")
        
        llm_response = call_llm_for_verification(
            client, verification_image_path, prompt_text, config["llm_config"]["model"]
        )
        
        response_text = llm_response.choices[0].message.content
        print(f"📥 LLM Response received ({len(response_text)} characters)")
        print(f"📄 LLM Response content:")
        print("=" * 60)
        print(response_text)
        print("=" * 60)
        
        # Parse response
        print("🔧 Parsing LLM response...")
        verified_node_ids, parsing_details = parse_verification_response(
            response_text, candidate_indices
        )
        
        print(f"✅ Parsing completed:")
        print(f"   - Method: {parsing_details.get('parsing_method', 'unknown')}")
        print(f"   - Verified IDs: {verified_node_ids}")
        print(f"   - Warnings: {parsing_details.get('warnings', [])}")
        
        # Save verification log
        verification_log = {
            "candidates": candidates,
            "candidate_indices": candidate_indices,
            "goal_object": goal_object,
            "llm_response": response_text,
            "parsing_details": parsing_details,
            "verified_node_ids": verified_node_ids,
            "verification_image": verification_image_path
        }
        
        verification_log_path = os.path.join(output_dir, "verification_log.json")
        with open(verification_log_path, 'w', encoding='utf-8') as f:
            json.dump(verification_log, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Verification log saved: {verification_log_path}")
        
        # Prepare result
        result = {
            "verified_node_ids": verified_node_ids,
            "verification_details": parsing_details,
            "generated_files": {
                "verification_image": verification_image_path,
                "verification_log": verification_log_path
            },
            "llm_response": {
                "raw_response": response_text,
                "verified_count": len(verified_node_ids),
                "parsing_method": parsing_details["parsing_method"]
            }
        }
        
        print(f"🎯 Verification completed: {len(verified_node_ids)} valid candidates")
        print(f"📊 Verified node IDs: {verified_node_ids}")
        
        return result
        
    except Exception as e:
        error_msg = f"Verification failed: {str(e)}"
        print(f"❌ {error_msg}")
        
        # Return fallback result
        return {
            "verified_node_ids": [],
            "verification_details": {"error": error_msg},
            "generated_files": {},
            "llm_response": {"error": error_msg}
        }
