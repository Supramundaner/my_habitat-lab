#!/usr/bin/env python3
"""
Action generation utilities for converting verified candidates to final action files.
"""

import os
import json
import shutil
from typing import Dict, Any, List


def copy_candidate_to_action(candidate_path: str, action_path: str) -> str:
    """
    Copy a candidate.json file to an action.json file.
    
    Args:
        candidate_path: Path to source candidate file
        action_path: Path to destination action file
        
    Returns:
        Path to the created action file
    """
    if not os.path.exists(candidate_path):
        raise FileNotFoundError(f"Candidate file not found: {candidate_path}")
    
    # Simply copy the file
    shutil.copy2(candidate_path, action_path)
    print(f"✓ Copied candidate to action: {candidate_path} -> {action_path}")
    
    return action_path


def find_candidate_by_node_id(node_id: int, candidate_results: List[Dict[str, Any]], 
                             output_dir: str) -> str:
    """
    Find the candidate file path that corresponds to a specific node ID.
    
    Args:
        node_id: Target node ID to find
        candidate_results: List of candidate generation results
        output_dir: Output directory path
        
    Returns:
        Path to the candidate file containing the specified node ID
    """
    for result in candidate_results:
        if result.get("results", {}).get("target_node_id") == node_id:
            return result["generated_files"]["candidate_json"]
    
    # Fallback: search by filename pattern and check content
    for i in range(len(candidate_results)):
        if i == 0:
            candidate_file = os.path.join(output_dir, "candidate.json")
        else:
            candidate_file = os.path.join(output_dir, f"candidate_iter_{i + 1}.json")
        
        if os.path.exists(candidate_file):
            try:
                with open(candidate_file, 'r', encoding='utf-8') as f:
                    candidate_data = json.load(f)
                
                # Check if this candidate contains the target node_id
                # We need to extract it from the corresponding node_selection_log
                iteration = candidate_data.get("iteration", 1)
                if iteration == 1:
                    log_file = os.path.join(output_dir, "node_selection_log.json")
                else:
                    log_file = os.path.join(output_dir, f"node_selection_log_iter_{iteration}.json")
                
                if os.path.exists(log_file):
                    with open(log_file, 'r', encoding='utf-8') as f:
                        log_data = json.load(f)
                    
                    if log_data.get("selected_node", {}).get("node_id") == node_id:
                        return candidate_file
                        
            except Exception as e:
                print(f"⚠️  Warning: Could not check candidate file {candidate_file}: {e}")
                continue
    
    raise ValueError(f"No candidate file found for node ID {node_id}")


def generate_final_actions(verified_node_ids: List[int], candidate_results: List[Dict[str, Any]],
                          output_dir: str) -> List[str]:
    """
    Generate final action files based on verification results.
    
    Args:
        verified_node_ids: List of verified node IDs in priority order
        candidate_results: List of candidate generation results
        output_dir: Output directory path
        
    Returns:
        List of generated action file paths
    """
    action_files = []
    
    if not verified_node_ids:
        # Empty list case: use first candidate as fallback
        print("⚠️  No verified candidates, using fallback strategy (first candidate)")
        
        fallback_candidate = os.path.join(output_dir, "candidate.json")
        if not os.path.exists(fallback_candidate):
            raise FileNotFoundError(f"Fallback candidate file not found: {fallback_candidate}")
        
        fallback_action = os.path.join(output_dir, "action_iter_1.json")
        action_path = copy_candidate_to_action(fallback_candidate, fallback_action)
        action_files.append(action_path)
        
        print(f"✓ Generated fallback action: {action_path}")
        
    else:
        # Generate action files for each verified node in priority order
        print(f"🎯 Generating {len(verified_node_ids)} action files for verified candidates")
        
        for i, node_id in enumerate(verified_node_ids):
            try:
                # Find candidate file for this node
                candidate_file = find_candidate_by_node_id(node_id, candidate_results, output_dir)
                
                # Generate action file
                action_file = os.path.join(output_dir, f"action_iter_{i + 1}.json")
                action_path = copy_candidate_to_action(candidate_file, action_file)
                action_files.append(action_path)
                
                print(f"✓ Generated action {i + 1}: Node {node_id} -> {action_path}")
                
            except Exception as e:
                print(f"❌ Failed to generate action for node {node_id}: {e}")
                continue
    
    return action_files


def create_action_summary(verified_node_ids: List[int], action_files: List[str], 
                         output_dir: str) -> str:
    """
    Create a summary of generated action files.
    
    Args:
        verified_node_ids: List of verified node IDs
        action_files: List of generated action file paths
        output_dir: Output directory path
        
    Returns:
        Path to action summary file
    """
    summary_data = {
        "total_verified_candidates": len(verified_node_ids),
        "total_action_files": len(action_files),
        "verified_node_ids": verified_node_ids,
        "action_files": [os.path.basename(path) for path in action_files],
        "fallback_used": len(verified_node_ids) == 0,
        "action_file_details": []
    }
    
    # Add details for each action file
    for i, (action_path, node_id) in enumerate(zip(action_files, 
                                                   verified_node_ids if verified_node_ids else [None])):
        detail = {
            "priority": i + 1,
            "action_file": os.path.basename(action_path),
            "node_id": node_id,
            "is_fallback": len(verified_node_ids) == 0
        }
        summary_data["action_file_details"].append(detail)
    
    # Save summary
    summary_path = os.path.join(output_dir, "action_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Action summary saved: {summary_path}")
    return summary_path
