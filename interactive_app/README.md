# Habitat Interactive Navigator

一个基于Habitat-sim和PyQt5的交互式3D导航应用程序。

## 项目状态：✅ 完成

已修复的5个主要问题：
1. ✅ GPU加速渲染 (RTX 4090, 838+ FPS)
2. ✅ 视角转换命令 (right/left/up/down + 角度)
3. ✅ 导航朝向修正 (A→B方向正确)
4. ✅ 坐标标签可见 (白色文字)
5. ✅ 1.5米人眼高度视角

## 目录结构

```
interactive_app/
├── src/                    # 源代码
│   ├── habitat_navigator_app.py  # 主应用程序
│   └── main.py            # 启动脚本
├── tests/                 # 测试文件
│   ├── test_*.py          # 各种功能测试
│   ├── debug_*.py         # 调试脚本
│   ├── final_verification.py  # 完整验证
│   └── fix_all_issues.py  # 修复检查
├── images/                # 生成的图像
│   ├── final_test_*.png   # 最终测试图像
│   ├── test_*.png         # 各种测试图像
│   └── *.png             # 其他验证图像
├── docs/                  # 文档
│   ├── ALL_FIXES_COMPLETED.md    # 修复完成总结
│   ├── FINAL_USAGE_GUIDE.md      # 使用指南
│   └── *_SUMMARY.md             # 各种技术总结
├── scripts/               # 辅助脚本
│   ├── run_gui_app.py     # GUI启动脚本
│   ├── diagnose_*.py      # 诊断脚本
│   └── verify_*.py        # 验证脚本
└── README.md              # 本文件
```

## 快速开始

### 1. 运行完整应用 (需要GUI环境)
```bash
cd interactive_app/src
python main.py
```

### 2. 运行功能验证 (无需GUI)
```bash
cd interactive_app/tests
python final_verification.py
```

### 3. 运行特定测试
```bash
cd interactive_app/tests
python test_all_fixes.py        # 所有修复测试
python test_view_commands.py    # 视角命令测试
```

## 使用说明

### 坐标导航
```
2.5, -1.0    # 移动到坐标(2.5, -1.0)
0, 0         # 移动到原点
```

### 视角控制
```
right 30     # 右转30度
left 45      # 左转45度  
up 20        # 上看20度
down 10      # 下看10度
```

## 技术特性

- **高性能渲染**: RTX 4090 GPU加速，838+ FPS
- **真实视角**: 1.5米人眼高度
- **精确导航**: 米级坐标精度
- **清晰界面**: 白色坐标标签清晰可见
- **正确朝向**: A→B导航方向修正

## 依赖环境

- Python 3.7+
- Habitat-sim
- PyQt5
- NumPy
- PIL/Pillow
- Magnum

## 测试场景

当前使用场景：
`/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb`

## 开发者

项目已完成所有功能开发和bug修复。

## 许可证

遵循Habitat-lab项目许可证。
