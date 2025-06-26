#!/usr/bin/env python3
"""
用于在有GUI环境中运行的简化启动脚本
"""

import os
import sys

def check_display():
    """检查是否有可用的显示器"""
    display = os.environ.get('DISPLAY')
    if not display:
        print("警告: 没有检测到DISPLAY环境变量")
        print("请确保在有图形界面的环境中运行此应用程序")
        return False
    return True

def main():
    """主函数"""
    print("Habitat导航应用程序启动器")
    print("=" * 40)
    
    # 检查显示器
    if not check_display():
        print("\n解决方案:")
        print("1. 在有桌面环境的Linux系统中运行")
        print("2. 使用X11转发: ssh -X username@hostname")
        print("3. 使用VNC或其他远程桌面解决方案")
        return False
    
    # 检查依赖
    try:
        from PyQt5.QtWidgets import QApplication
        print("✓ PyQt5 可用")
    except ImportError:
        print("✗ PyQt5 不可用，请安装: pip install PyQt5")
        return False
    
    try:
        import habitat_sim
        print("✓ Habitat-sim 可用")
    except ImportError:
        print("✗ Habitat-sim 不可用")
        return False
    
    # 尝试启动应用程序
    try:
        print("\n启动Habitat导航应用程序...")
        from habitat_navigator_app import main as app_main
        app_main()
        return True
    except Exception as e:
        print(f"✗ 启动失败: {e}")
        return False

if __name__ == '__main__':
    success = main()
    if success:
        print("应用程序正常退出")
    else:
        print("启动失败，请检查环境配置")
        sys.exit(1)
