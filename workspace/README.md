# Habitat Lab Coordinate Navigation System

A sophisticated coordinate-based navigation system for Habitat Lab using Matterport3D and other datasets. This system allows precise 2D map coordinate navigation with automatic 3D world coordinate conversion, providing both first-person and third-person views along with topdown map visualization.

## 🌟 Key Features

- **Enhanced Terminal Interface**: Advanced command-line interface with colors, progress bars, and auto-completion
- **2D Coordinate Input System**: Navigate using intuitive 2D map coordinates
- **Automatic Height Adaptation**: Automatically converts 2D coordinates to 3D with proper height
- **Multi-View Visualization**: First-person, third-person, and topdown map views
- **Real-time Position Tracking**: Visual position markers on topdown maps
- **Interactive Exploration**: Guided exploration mode and interactive demos
- **Session Management**: Command history, statistics, and session tracking
- **Multiple Dataset Support**: Works with Matterport3D, Habitat Test Scenes, HM3D, and Replica CAD
- **Advanced Commands**: Random navigation, center positioning, and exploration tools
- **Image Export**: Automatically saves all views to organized image folders

## 🏗️ Project Structure

```
workspace/
├── src/
│   ├── coordinate_navigation.py         # Core navigation system
│   ├── interactive_navigation.py        # Basic command interface
│   └── enhanced_terminal_interface.py   # Enhanced terminal interface
├── images/                              # Generated images (auto-created)
├── launch.py                           # Multi-interface launcher script
├── test_system.py                      # System testing suite
├── download_datasets.py                # Dataset download utility
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install core dependencies manually
pip install habitat-sim habitat-lab opencv-python matplotlib numpy
```

### 2. Download Datasets

```bash
# Run the interactive dataset downloader
python download_datasets.py

# Or download specific datasets manually
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path ../data
```

### 3. Run the Navigation System

```bash
# 🚀 Enhanced terminal interface (recommended)
python launch.py

# Or use specific interfaces:
python launch.py --basic           # Basic interface
python launch.py --test            # Run tests
python launch.py --download        # Download datasets

# Direct access to interfaces:
python src/enhanced_terminal_interface.py  # Enhanced interface
python src/interactive_navigation.py       # Basic interface
```

## 🎮 Usage Guide

### Interactive Commands

Once the system is running, you can use these commands:

#### 📍 Navigation Commands  
- `move <x> <y>` - Move to map coordinates (e.g., `move 512 256`)
- `goto <x> <y>` - Same as move
- `center` - Move to center of the map
- `random` - Move to a random navigable position
- `position` - Show current position in both coordinate systems
- `bounds` - Display map coordinate bounds

#### 👁️ View Commands
- `rotate <yaw> [pitch]` - Rotate view by degrees (e.g., `rotate 90 15`)
- `turn <angle>` - Turn by specified angle
- `view` or `show` - Display current views
- `save [prefix]` - Save current images with optional prefix

#### 🗺️ Scene Commands
- `scenes` - List all available scenes
- `switch <scene_id>` - Switch to different scene
- `reload` - Reload current scene

#### 🎮 Interactive Commands
- `demo` - Run interactive demonstration
- `explore` - Start guided exploration mode
- `history` - Show command history
- `stats` - Display session statistics

#### 💾 Utility Commands
- `info` - Show detailed system information
- `clear` - Clear terminal screen
- `help` - Display command help
- `quit` or `exit` - Exit the program

### Coordinate System

The system uses a dual coordinate system:

- **Map Coordinates**: 2D pixel coordinates (0-1024 by default)
  - Origin (0,0) is top-left of the map
  - X-axis goes left to right
  - Y-axis goes top to bottom

- **World Coordinates**: 3D meters in Habitat world space
  - Automatically converted from map coordinates
  - Height (Y) is automatically determined for navigation

### Example Session

```bash
# Enhanced Terminal Interface Session
🤖 Command > move 512 256
🚶 Moving to map coordinates (512.0, 256.0)...
[████████████████████████████████] 100.0%
✅ Movement successful!
📍 New position: Map(512.0, 256.0) World(2.45, -1.23)
[Displays updated views with colored output]

🤖 Command > rotate 45 10  
🔄 Rotating view - Yaw: 45°, Pitch: 10°
↻ Medium rotation applied
[Shows rotated view]

🤖 Command > demo
🎬 Starting interactive demonstration...
⚠️  This demo will move the agent and rotate the view.
Press Enter to continue with step 1: Moving to center of map

🤖 Command > explore
🗺️  Starting guided exploration mode...
Suggested exploration points:
  1. Center of the map - Map coords: (512, 512)
  2. Northwest area - Map coords: (312, 312)
  3. Northeast area - Map coords: (712, 312)
  4. Southeast area - Map coords: (712, 712)
  5. Southwest area - Map coords: (312, 712)

🤖 Command > save exploration_
💾 Saving current images...
[█████████████████████████] 100.0%
✅ Images saved to: workspace/images/
  📷 first_person: exploration_first_person_van-gogh-room_20250625_143022.png (2.1 MB)
  📷 third_person: exploration_third_person_van-gogh-room_20250625_143022.png (1.8 MB)
  📷 topdown_map: exploration_topdown_map_van-gogh-room_20250625_143022.png (0.5 MB)

🤖 Command > stats
📊 SESSION STATISTICS
========================================
Session Duration: 00:05:42
Commands Executed: 8
Movements: 3
View Rotations: 2
Images Saved: 9
Scene Switches: 1
Current Scene: habitat-test-scenes/van-gogh-room
Current Position: (512.0, 256.0)
```

## 🎯 Advanced Features

### Automatic Image Management

All generated images are automatically saved to the `images/` directory with timestamps and scene information:

- `first_person_<scene>_<timestamp>.png` - First-person view
- `third_person_<scene>_<timestamp>.png` - Third-person view  
- `topdown_map_<scene>_<timestamp>.png` - Topdown map with position markers

### Multi-Dataset Support

The system automatically detects and supports multiple datasets:

1. **Habitat Test Scenes** - Small scenes perfect for development
2. **Matterport3D** - Large-scale real-world indoor environments
3. **HM3D** - High-quality photorealistic scenes
4. **Replica CAD** - Synthetic indoor environments

### Position Visualization

The topdown map includes:
- 🔴 Red circle: Current agent position
- 🔵 Blue arrow: Current facing direction
- 📊 Coordinate system overlay with bounds and current position

## 🛠️ Development

### Core Classes

#### `CoordinateNavigationSystem`
The main navigation system class providing:
- Scene loading and management
- Coordinate system conversion
- Agent movement and rotation
- Multi-view rendering
- Image export functionality

#### `InteractiveNavigationInterface`
Command-line interface providing:
- Command parsing and validation
- User-friendly interaction
- Error handling and help system
- Session management

### Extending the System

You can extend the system by:

1. **Adding new datasets**: Modify `_get_available_scenes()` method
2. **Custom sensors**: Add sensor specifications in `_create_config()`
3. **New commands**: Extend `process_command()` in the interface
4. **Visualization options**: Customize the display and export functions

## 🔧 Configuration

### Map Settings
- `map_resolution`: Resolution of topdown map (default: 1024)
- `map_scale`: Meters per pixel (default: 0.1)
- `camera_height`: Agent camera height in meters (default: 1.5)
- `fov`: Field of view in degrees (default: 90)

### Paths
- `workspace_path`: Current workspace directory
- `data_path`: Habitat data directory (default: ../data)
- `images_path`: Output images directory (default: workspace/images)

## 📋 Requirements

- Python 3.7+
- habitat-sim >= 0.2.4
- habitat-lab >= 0.2.4
- OpenCV >= 4.5.0
- matplotlib >= 3.3.0
- numpy >= 1.19.0

## 🐛 Troubleshooting

### Common Issues

1. **"No datasets found"**
   - Run `python download_datasets.py` to download datasets
   - Check that data directory exists and contains scene_datasets/

2. **"Position not navigable"**
   - Try coordinates closer to the center of the map
   - Use `bounds` command to see valid coordinate ranges
   - Check topdown map for navigable (gray) areas

3. **"Scene file not found"**
   - Verify dataset is properly downloaded
   - Check scene ID format (e.g., "habitat-test-scenes/apartment_1")

4. **Import errors**
   - Install all requirements: `pip install -r requirements.txt`
   - Ensure habitat-sim and habitat-lab are properly installed

### Performance Tips

- Use smaller map resolutions for faster processing
- Close and reopen scenes periodically for memory management
- Save images selectively to reduce disk usage

## 📚 References

- [Habitat Lab Documentation](https://habitat-sim.readthedocs.io/)
- [Matterport3D Dataset](https://niessner.github.io/Matterport/)
- [HM3D Dataset](https://aihabitat.org/datasets/hm3d/)
- [Habitat Paper](https://arxiv.org/abs/1904.01201)

## 📄 License

This project follows the same license as Habitat Lab. Please refer to the original Habitat Lab repository for licensing information.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review Habitat Lab documentation
3. Open an issue with detailed error information and system specs
