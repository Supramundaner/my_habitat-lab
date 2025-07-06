#!/usr/bin/env python3
"""
Agent State Viewer
智能体状态查看器 - 用于查看和管理保存的智能体状态
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any

def load_states(state_file: str) -> Dict[str, Any]:
    """加载智能体状态文件"""
    try:
        with open(state_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading state file: {e}")
        return {}

def display_states(states: Dict[str, Any]):
    """显示智能体状态"""
    if not states:
        print("No states found or state file is empty.")
        return
    
    print("=" * 60)
    print("AGENT STATES")
    print("=" * 60)
    
    for agent_id, state in states.items():
        print(f"\n🤖 Agent: {agent_id}")
        print("-" * 30)
        
        position = state.get("position", [0, 0, 0])
        rotation = state.get("rotation", [0, 0, 0, 1])
        velocity = state.get("velocity", 0.0)
        angular_velocity = state.get("angular_velocity", 0.0)
        
        print(f"Position:        [{position[0]:7.3f}, {position[1]:7.3f}, {position[2]:7.3f}]")
        print(f"Rotation (quat): [{rotation[0]:7.3f}, {rotation[1]:7.3f}, {rotation[2]:7.3f}, {rotation[3]:7.3f}]")
        print(f"Velocity:        {velocity:7.3f} m/s")
        print(f"Angular Vel:     {angular_velocity:7.3f} rad/s")
        
        # 计算朝向角度（从四元数）
        import math
        qx, qy, qz, qw = rotation
        yaw = math.atan2(2.0 * (qw * qy + qx * qz), 1.0 - 2.0 * (qy * qy + qz * qz))
        yaw_deg = math.degrees(yaw)
        print(f"Heading:         {yaw_deg:7.1f}°")

def modify_state(states: Dict[str, Any], agent_id: str, 
                position: list = None, rotation: list = None) -> Dict[str, Any]:
    """修改智能体状态"""
    if agent_id not in states:
        print(f"Agent {agent_id} not found in states.")
        return states
    
    if position:
        states[agent_id]["position"] = position
        print(f"Updated {agent_id} position to {position}")
    
    if rotation:
        states[agent_id]["rotation"] = rotation
        print(f"Updated {agent_id} rotation to {rotation}")
    
    return states

def save_states(states: Dict[str, Any], output_file: str):
    """保存修改后的状态"""
    try:
        with open(output_file, 'w') as f:
            json.dump(states, f, indent=2)
        print(f"States saved to: {output_file}")
    except Exception as e:
        print(f"Error saving states: {e}")

def create_template_states(num_agents: int = 3) -> Dict[str, Any]:
    """创建模板状态文件"""
    states = {}
    
    positions = [
        [0.0, 0.0, 0.0],    # agent_0 at origin
        [2.0, 0.0, 0.0],    # agent_1 at +X
        [0.0, 0.0, 2.0],    # agent_2 at +Z
    ]
    
    for i in range(num_agents):
        agent_id = f"agent_{i}"
        pos = positions[i] if i < len(positions) else [float(i), 0.0, 0.0]
        
        states[agent_id] = {
            "position": pos,
            "rotation": [0.0, 0.0, 0.0, 1.0],  # No rotation
            "velocity": 0.0,
            "angular_velocity": 0.0
        }
    
    return states

def interactive_editor(state_file: str):
    """交互式状态编辑器"""
    states = load_states(state_file)
    
    if not states:
        print("No states loaded. Creating template...")
        states = create_template_states()
    
    while True:
        print("\n" + "=" * 60)
        print("INTERACTIVE STATE EDITOR")
        print("=" * 60)
        
        display_states(states)
        
        print("\nCommands:")
        print("  show                     - Display current states")
        print("  edit <agent_id>          - Edit agent state")
        print("  move <agent_id> x y z    - Set agent position")
        print("  rotate <agent_id> x y z w - Set agent rotation (quaternion)")
        print("  add <agent_id>           - Add new agent")
        print("  remove <agent_id>        - Remove agent")
        print("  save [filename]          - Save states")
        print("  quit                     - Exit editor")
        
        try:
            command = input("\nEnter command: ").strip().split()
            
            if not command:
                continue
                
            cmd = command[0].lower()
            
            if cmd == "quit":
                break
            elif cmd == "show":
                continue  # Will redisplay in next loop
            elif cmd == "save":
                output_file = command[1] if len(command) > 1 else state_file
                save_states(states, output_file)
            elif cmd == "edit":
                if len(command) < 2:
                    print("Usage: edit <agent_id>")
                    continue
                agent_id = command[1]
                edit_agent_interactive(states, agent_id)
            elif cmd == "move":
                if len(command) < 5:
                    print("Usage: move <agent_id> x y z")
                    continue
                agent_id = command[1]
                position = [float(command[2]), float(command[3]), float(command[4])]
                states = modify_state(states, agent_id, position=position)
            elif cmd == "rotate":
                if len(command) < 6:
                    print("Usage: rotate <agent_id> x y z w")
                    continue
                agent_id = command[1]
                rotation = [float(command[2]), float(command[3]), float(command[4]), float(command[5])]
                states = modify_state(states, agent_id, rotation=rotation)
            elif cmd == "add":
                if len(command) < 2:
                    print("Usage: add <agent_id>")
                    continue
                agent_id = command[1]
                states[agent_id] = {
                    "position": [0.0, 0.0, 0.0],
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                    "velocity": 0.0,
                    "angular_velocity": 0.0
                }
                print(f"Added agent {agent_id}")
            elif cmd == "remove":
                if len(command) < 2:
                    print("Usage: remove <agent_id>")
                    continue
                agent_id = command[1]
                if agent_id in states:
                    del states[agent_id]
                    print(f"Removed agent {agent_id}")
                else:
                    print(f"Agent {agent_id} not found")
            else:
                print(f"Unknown command: {cmd}")
                
        except (ValueError, IndexError) as e:
            print(f"Invalid command format: {e}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

def edit_agent_interactive(states: Dict[str, Any], agent_id: str):
    """交互式编辑单个智能体"""
    if agent_id not in states:
        print(f"Agent {agent_id} not found.")
        return
    
    state = states[agent_id]
    print(f"\nEditing agent: {agent_id}")
    print(f"Current position: {state['position']}")
    print(f"Current rotation: {state['rotation']}")
    
    try:
        # 编辑位置
        pos_input = input("New position [x y z] (or press Enter to keep current): ").strip()
        if pos_input:
            pos_parts = pos_input.split()
            if len(pos_parts) == 3:
                state["position"] = [float(pos_parts[0]), float(pos_parts[1]), float(pos_parts[2])]
                print("Position updated")
        
        # 编辑旋转
        rot_input = input("New rotation [x y z w] (or press Enter to keep current): ").strip()
        if rot_input:
            rot_parts = rot_input.split()
            if len(rot_parts) == 4:
                state["rotation"] = [float(rot_parts[0]), float(rot_parts[1]), 
                                   float(rot_parts[2]), float(rot_parts[3])]
                print("Rotation updated")
        
        # 编辑速度
        vel_input = input("New velocity (or press Enter to keep current): ").strip()
        if vel_input:
            state["velocity"] = float(vel_input)
            print("Velocity updated")
        
    except ValueError as e:
        print(f"Invalid input: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Agent State Viewer and Editor")
    parser.add_argument("state_file", nargs="?", default="./outputs/agent_states.json",
                       help="Path to agent states JSON file")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Launch interactive editor")
    parser.add_argument("--create-template", "-t", action="store_true",
                       help="Create template state file")
    parser.add_argument("--agents", "-n", type=int, default=3,
                       help="Number of agents for template (default: 3)")
    parser.add_argument("--output", "-o", 
                       help="Output file for modified states")
    
    args = parser.parse_args()
    
    if args.create_template:
        print(f"Creating template state file with {args.agents} agents...")
        states = create_template_states(args.agents)
        output_file = args.output or args.state_file
        save_states(states, output_file)
        print(f"Template created: {output_file}")
        return 0
    
    if args.interactive:
        interactive_editor(args.state_file)
        return 0
    
    # 默认：显示状态
    states = load_states(args.state_file)
    display_states(states)
    
    return 0

if __name__ == "__main__":
    exit(main())
