#!/usr/bin/env python3
"""
Debug script to check key consistency across the workflow.
"""

# Test key existence in navigation nodes
def test_navigation_nodes():
    import json
    import os
    
    try:
        nodes_path = "output/navigation_nodes.json"
        if os.path.exists(nodes_path):
            with open(nodes_path, 'r') as f:
                data = json.load(f)
            
            nodes = data.get("nodes", [])
            if nodes:
                first_node = nodes[0]
                print("✓ Navigation nodes structure:")
                print(f"  Sample node keys: {list(first_node.keys())}")
                print(f"  Has 'node_id': {'node_id' in first_node}")
                print(f"  Has 'id': {'id' in first_node}")
                return True
            else:
                print("✗ No nodes in navigation_nodes.json")
                return False
        else:
            print("✗ navigation_nodes.json not found")
            return False
    except Exception as e:
        print(f"✗ Error reading navigation_nodes.json: {e}")
        return False

# Test key existence in multi_point_summary  
def test_multi_point_summary():
    import json
    import os
    
    try:
        summary_path = "output/multi_point_summary.json"
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                data = json.load(f)
            
            iterations = data.get("iterations", [])
            if iterations:
                first_iter = iterations[0]
                print("✓ Multi-point summary structure:")
                print(f"  Iteration keys: {list(first_iter.keys())}")
                
                node_selection = first_iter.get("node_selection", {})
                print(f"  Node selection keys: {list(node_selection.keys())}")
                print(f"  Has 'selected_node_id': {'selected_node_id' in node_selection}")
                return True
            else:
                print("✗ No iterations in multi_point_summary.json")
                return False
        else:
            print("✗ multi_point_summary.json not found")
            return False
    except Exception as e:
        print(f"✗ Error reading multi_point_summary.json: {e}")
        return False

# Test node selection log structure
def test_node_selection_log():
    import json
    import os
    
    try:
        log_path = "output/node_selection_log.json"
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                data = json.load(f)
            
            print("✓ Node selection log structure:")
            print(f"  Log keys: {list(data.keys())}")
            
            if "selected_node" in data:
                selected_node = data["selected_node"]
                print(f"  Selected node keys: {list(selected_node.keys())}")
                print(f"  Has 'node_id': {'node_id' in selected_node}")
                print(f"  Has 'id': {'id' in selected_node}")
            
            if "llm_response" in data:
                llm_response = data["llm_response"]
                print(f"  LLM response keys: {list(llm_response.keys())}")
                print(f"  Has 'selected_node_id': {'selected_node_id' in llm_response}")
                print(f"  Has 'selected_node_data': {'selected_node_data' in llm_response}")
                
                if "selected_node_data" in llm_response:
                    node_data = llm_response["selected_node_data"]
                    if node_data:
                        print(f"  Selected node data keys: {list(node_data.keys())}")
                        print(f"  Node data has 'node_id': {'node_id' in node_data}")
                        print(f"  Node data has 'id': {'id' in node_data}")
            
            return True
        else:
            print("✗ node_selection_log.json not found")
            return False
    except Exception as e:
        print(f"✗ Error reading node_selection_log.json: {e}")
        return False

if __name__ == "__main__":
    print("=== KEY CONSISTENCY DEBUG ===")
    print()
    
    print("1. Testing navigation_nodes.json:")
    test_navigation_nodes()
    print()
    
    print("2. Testing multi_point_summary.json:")
    test_multi_point_summary()
    print()
    
    print("3. Testing node_selection_log.json:")
    test_node_selection_log()
    print()
    
    print("=== DEBUG COMPLETE ===")
