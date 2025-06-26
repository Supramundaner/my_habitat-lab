# 测试文件说明

## 主要测试文件

### 完整验证
- `final_verification.py` - 最终完整功能验证，无需GUI
- `test_all_fixes.py` - 测试所有5个问题的修复
- `fix_all_issues.py` - 检查GPU和智能体配置  

### 功能专项测试
- `test_view_commands.py` - 视角命令测试
- `test_coordinate_system.py` - 坐标系统测试
- `test_coordinate_input.py` - 坐标输入测试
- `test_fpv_simple.py` - 第一人称视角测试
- `test_simulator_only.py` - 模拟器基础功能测试

### 问题修复测试
- `test_quaternion_fix.py` - 四元数修复测试
- `test_fpv_animation_fix.py` - FPV动画修复测试
- `test_final_fixes.py` - 最终修复测试
- `test_fixed_alignment.py` - 坐标对齐修复测试

### 调试脚本
- `debug_quaternion*.py` - 四元数相关调试
- `debug_fpv_issue.py` - FPV显示问题调试
- `debug_simulator_init.py` - 模拟器初始化调试
- `debug_user_input.py` - 用户输入调试

### 诊断脚本
- `diagnose_coordinate_alignment.py` - 坐标对齐诊断
- `verify_coordinate_alignment.py` - 坐标对齐验证

## 运行方法

### 快速验证所有功能
```bash
python3 final_verification.py
```

### 测试特定功能
```bash
python3 test_all_fixes.py        # 所有修复
python3 test_view_commands.py    # 视角命令
python3 test_coordinate_system.py # 坐标系统
```

### 调试特定问题
```bash
python3 debug_quaternion.py      # 四元数问题
python3 debug_fpv_issue.py       # FPV显示问题
```

## 测试结果

所有测试都已通过，验证了：
- ✅ GPU加速渲染 (838+ FPS)
- ✅ 视角转换命令功能
- ✅ 导航朝向修正 (A→B)
- ✅ 坐标标签可见性
- ✅ 1.5米人眼高度
