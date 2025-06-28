#!/usr/bin/env python3
"""
测试代理初始化逻辑
"""

# 模拟测试HabitatVideoGenerator的初始化逻辑

class MockVideoGenerator:
    def __init__(self):
        self.agent_initialized = False
        print("MockVideoGenerator initialized")
        print("Agent will be positioned at the first command location")
    
    def process_command_sequence(self, commands):
        """模拟处理指令序列"""
        print(f"\nProcessing command sequence with {len(commands)} commands")
        
        # 如果代理还未初始化位置，使用第一个指令来设置初始位置
        if not self.agent_initialized and len(commands) > 0:
            first_command = commands[0]
            print(f"First command: {first_command}")
            
            # 检查第一个指令是否是移动指令（包含坐标）
            if not isinstance(first_command[0], str):
                # 第一个指令是移动指令 [x, z]
                target_x = float(first_command[0])
                target_z = float(first_command[1])
                
                print(f"Initializing agent at first command position ({target_x:.2f}, {target_z:.2f})")
                self.agent_initialized = True
                
                # 跳过第一个指令（因为代理已经在目标位置）
                commands = commands[1:]
                print(f"Skipping first command, remaining commands: {len(commands)}")
            else:
                # 第一个指令是旋转指令，使用场景中心作为初始位置
                print("First command is rotation, initializing at scene center")
                self.agent_initialized = True
        
        # 处理剩余指令
        for i, command in enumerate(commands):
            print(f"  Executing command {i+1}/{len(commands)}: {command}")
        
        return True

# 测试不同的指令序列
def test_scenarios():
    print("=== Test Scenario 1: First command is movement ===")
    generator1 = MockVideoGenerator()
    commands1 = [[5.0, 3.0], ["left", 45], [8.0, 7.0]]
    generator1.process_command_sequence(commands1)
    
    print("\n=== Test Scenario 2: First command is rotation ===")
    generator2 = MockVideoGenerator()
    commands2 = [["right", 90], [2.0, 4.0], ["left", 30]]
    generator2.process_command_sequence(commands2)
    
    print("\n=== Test Scenario 3: Empty command list ===")
    generator3 = MockVideoGenerator()
    commands3 = []
    generator3.process_command_sequence(commands3)

if __name__ == "__main__":
    test_scenarios()
