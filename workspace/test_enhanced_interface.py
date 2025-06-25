#!/usr/bin/env python3
"""
Quick test script for the enhanced terminal interface

This script performs basic functionality tests to ensure the enhanced
terminal interface is working correctly.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
workspace_path = Path(__file__).parent
src_path = workspace_path / "src"
sys.path.append(str(src_path))


def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from enhanced_terminal_interface import EnhancedTerminalInterface, Colors, ProgressBar
        print("✅ Enhanced terminal interface imports successful")
        return True
    except ImportError as e:
        print(f"❌ Enhanced terminal interface import failed: {e}")
        return False


def test_colors():
    """Test color output"""
    print("🎨 Testing color output...")
    
    try:
        from enhanced_terminal_interface import Colors
        
        print(f"{Colors.GREEN}✅ Green text{Colors.ENDC}")
        print(f"{Colors.BLUE}🔵 Blue text{Colors.ENDC}")
        print(f"{Colors.WARNING}⚠️  Warning text{Colors.ENDC}")
        print(f"{Colors.FAIL}❌ Error text{Colors.ENDC}")
        print(f"{Colors.CYAN}ℹ️  Info text{Colors.ENDC}")
        
        return True
    except Exception as e:
        print(f"❌ Color test failed: {e}")
        return False


def test_progress_bar():
    """Test progress bar functionality"""
    print("📊 Testing progress bar...")
    
    try:
        from enhanced_terminal_interface import ProgressBar
        import time
        
        progress = ProgressBar(10, 30)
        for i in range(11):
            progress.update(i)
            time.sleep(0.05)
        progress.finish()
        
        print("✅ Progress bar test successful")
        return True
    except Exception as e:
        print(f"❌ Progress bar test failed: {e}")
        return False


def test_interface_creation():
    """Test interface creation without full initialization"""
    print("🖥️  Testing interface creation...")
    
    try:
        from enhanced_terminal_interface import EnhancedTerminalInterface
        
        # Create interface but don't initialize navigation system
        interface = EnhancedTerminalInterface()
        
        # Test some basic methods
        interface.print_success("Test success message")
        interface.print_error("Test error message")
        interface.print_warning("Test warning message")
        interface.print_info("Test info message")
        
        print("✅ Interface creation test successful")
        return True
    except Exception as e:
        print(f"❌ Interface creation test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Enhanced Terminal Interface - Quick Test Suite")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Color Output", test_colors),
        ("Progress Bar", test_progress_bar),
        ("Interface Creation", test_interface_creation)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {passed + failed}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print(f"\n🎉 All tests passed! Enhanced terminal interface is ready to use.")
        print(f"🚀 Run: python launch.py")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the issues above.")
        print("💡 Try installing missing dependencies: pip install -r requirements.txt")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
