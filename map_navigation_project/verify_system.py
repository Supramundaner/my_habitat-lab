#!/usr/bin/env python3
"""
System verification utility for Habitat Map Navigation Project.

This script checks the system configuration and dependencies to ensure
everything is properly set up before running the navigation system.
"""

import os
import sys
import importlib
from pathlib import Path


class SystemVerifier:
    """System verification utility class."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.errors = []
        self.warnings = []
    
    def check_python_version(self):
        """Check Python version compatibility."""
        print("Checking Python version...")
        
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 7):
            self.errors.append(f"Python {version.major}.{version.minor} detected. Python 3.7+ required.")
        else:
            print(f"✓ Python {version.major}.{version.minor}.{version.micro} OK")
    
    def check_required_packages(self):
        """Check if required packages are available."""
        print("\nChecking Python packages...")
        
        required_packages = [
            ('numpy', 'numpy'),
            ('cv2', 'opencv-python'),
            ('matplotlib', 'matplotlib'),
            ('PIL', 'Pillow'),
            ('yaml', 'PyYAML'),
        ]
        
        optional_packages = [
            ('habitat', 'habitat-lab'),
            ('habitat_sim', 'habitat-sim'),
            ('quaternion', 'quaternion'),
        ]
        
        # Check required packages
        for module_name, package_name in required_packages:
            try:
                importlib.import_module(module_name)
                print(f"✓ {package_name} OK")
            except ImportError:
                self.errors.append(f"Required package '{package_name}' not found. Install with: pip install {package_name}")
        
        # Check optional packages
        for module_name, package_name in optional_packages:
            try:
                importlib.import_module(module_name)
                print(f"✓ {package_name} OK")
            except ImportError:
                if package_name in ['habitat-lab', 'habitat-sim']:
                    self.errors.append(f"Critical package '{package_name}' not found. Please install habitat-sim and habitat-lab following official instructions.")
                else:
                    self.warnings.append(f"Optional package '{package_name}' not found. Install with: pip install {package_name}")
    
    def check_project_structure(self):
        """Check if project files are present."""
        print("\nChecking project structure...")
        
        required_files = [
            'main_controller.py',
            'habitat_env.py',
            'map_visualizer.py',
            'configs/navigation_config.yaml',
            'requirements.txt',
            'README.md'
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                print(f"✓ {file_path} exists")
            else:
                self.errors.append(f"Required file '{file_path}' not found")
        
        # Check if output directory can be created
        output_dir = self.project_root / 'output_images'
        try:
            output_dir.mkdir(exist_ok=True)
            print(f"✓ Output directory accessible: {output_dir}")
        except Exception as e:
            self.warnings.append(f"Cannot create output directory: {e}")
    
    def check_habitat_data(self):
        """Check for Habitat data directories."""
        print("\nChecking Habitat data setup...")
        
        # Common data directory locations
        possible_data_dirs = [
            self.project_root.parent / 'data',
            Path.home() / 'habitat-lab' / 'data',
            Path('/habitat-lab/data'),
            Path('./data')
        ]
        
        data_dir_found = False
        for data_dir in possible_data_dirs:
            if data_dir.exists():
                print(f"✓ Found data directory: {data_dir}")
                data_dir_found = True
                
                # Check for scene datasets
                scene_dir = data_dir / 'scene_datasets'
                if scene_dir.exists():
                    print(f"✓ Scene datasets directory found: {scene_dir}")
                else:
                    self.warnings.append(f"Scene datasets not found in {scene_dir}")
                
                # Check for MP3D
                mp3d_dir = scene_dir / 'mp3d' if scene_dir.exists() else None
                if mp3d_dir and mp3d_dir.exists():
                    print(f"✓ MP3D dataset found: {mp3d_dir}")
                else:
                    self.warnings.append("MP3D dataset not found. Download with habitat data tools.")
                
                break
        
        if not data_dir_found:
            self.warnings.append("No Habitat data directory found. Make sure to download datasets.")
    
    def check_config_files(self):
        """Check configuration files."""
        print("\nChecking configuration files...")
        
        config_file = self.project_root / 'configs' / 'navigation_config.yaml'
        if config_file.exists():
            print(f"✓ Configuration file found: {config_file}")
            
            try:
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                print("✓ Configuration file is valid YAML")
            except Exception as e:
                self.errors.append(f"Configuration file has syntax errors: {e}")
        else:
            self.errors.append("Main configuration file not found")
    
    def run_verification(self):
        """Run all verification checks."""
        print("="*60)
        print("HABITAT MAP NAVIGATION - SYSTEM VERIFICATION")
        print("="*60)
        
        self.check_python_version()
        self.check_required_packages()
        self.check_project_structure()
        self.check_habitat_data()
        self.check_config_files()
        
        # Print summary
        print("\n" + "="*60)
        print("VERIFICATION SUMMARY")
        print("="*60)
        
        if not self.errors and not self.warnings:
            print("✓ All checks passed! Your system is ready to run the navigation project.")
            return True
        
        if self.errors:
            print(f"❌ Found {len(self.errors)} error(s):")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        
        if self.warnings:
            print(f"⚠️  Found {len(self.warnings)} warning(s):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        
        if self.errors:
            print("\n❌ Please fix the errors above before running the navigation system.")
            return False
        else:
            print("\n⚠️  Warnings detected but system should still work. Consider addressing them for optimal performance.")
            return True


def main():
    """Main verification function."""
    verifier = SystemVerifier()
    success = verifier.run_verification()
    
    if success:
        print("\n🚀 Ready to launch! Run: python launch.py")
        return 0
    else:
        print("\n🔧 Please fix the issues above and run verification again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
