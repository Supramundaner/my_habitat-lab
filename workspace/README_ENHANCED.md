# Enhanced Terminal Interface for Habitat Lab Navigation

An advanced, feature-rich terminal interface for the Habitat Lab Coordinate Navigation System with modern UI elements, interactive features, comprehensive command support, and enhanced coordinate system visualization.

## 🌟 New Features in v2.0

### 🗺️ Enhanced Coordinate System
- **Grid Overlay**: Visual coordinate grid on all maps with major/minor grid lines
- **Coordinate Mapping**: Clear mapping between map coordinates [0-1024] and world coordinates
- **User Position Selection**: Interactive map display for choosing initial starting position
- **Enhanced Visualization**: Improved coordinate system overlay with comprehensive information

### 🏠 HM3D Dataset Priority
- **HM3D First**: Prioritized support for HM3D dataset from https://aihabitat.org/datasets/hm3d/
- **Dataset Auto-Detection**: Automatic detection and prioritization of available datasets
- **Improved Download**: Enhanced dataset download script with HM3D as primary option

### 🎯 Interactive Initialization
- **Position Selection Map**: Display coordinate grid map before initialization
- **User Choice**: Let users select their preferred starting coordinates
- **Validation**: Check position navigability with fallback to nearest navigable point

## 🌟 Features

### 🎨 Visual Enhancements
- **Colored Output**: Different colors for success, error, warning, and info messages
- **Progress Bars**: Visual progress indicators for long operations
- **Emojis**: Intuitive emoji indicators for different types of messages
- **Formatted Output**: Well-structured information display with borders and sections

### 🚀 Advanced Functionality
- **Command Auto-completion**: Tab completion for all available commands
- **Command History**: Persistent command history across sessions
- **Session Statistics**: Track your usage patterns and navigation statistics
- **Interactive Demos**: Guided tours and exploration modes
- **Error Recovery**: Robust error handling with helpful suggestions

### 🎮 Interactive Features
- **Demo Mode**: Automated demonstration of system capabilities
- **Exploration Mode**: Guided exploration with suggested locations
- **Random Navigation**: Discover new areas with random positioning
- **Quick Actions**: Shortcuts like `center` and `random` commands

## 🚀 Quick Start

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Test the enhanced interface
python test_enhanced_interface.py

# Launch the enhanced interface
python launch.py
```

### First Run
```bash
# Start with the launcher (recommended)
python launch.py

# Or directly run the enhanced interface
python src/enhanced_terminal_interface.py
```

## 🎯 Command Guide

### Coordinate System Understanding
The enhanced system uses a comprehensive coordinate mapping:
- **Map Coordinates**: [0-1024] x [0-1024] pixel-based system with visual grid
- **World Coordinates**: Real-world 3D coordinates in meters
- **Grid Overlay**: Visual grid with major lines every 100 units, minor every 50
- **Interactive Selection**: Choose starting position from coordinate-mapped visualization

### Basic Navigation
```bash
# Move to specific coordinates
move 512 256

# Move to center of map
center

# Move to random location
random

# Check current position
position
```

### View Control
```bash
# Rotate view with yaw and pitch
rotate 90 15

# Simple turn
turn 45

# Show current view with enhanced coordinate overlay
view
```

### Interactive Features
```bash
# Start interactive demo
demo

# Begin guided exploration
explore

# Show command history
history

# Display session statistics
stats
```

### Scene Management
```bash
# List available scenes (HM3D prioritized)
scenes

# Switch to HM3D scene
switch hm3d/00800-TEEsavR23oF

# Reload current scene (will show position selection map)
reload
```

### Utilities
```bash
# Save current images
save my_exploration_

# Show system information
info

# Clear screen
clear

# Show help
help
```

## 🎨 Visual Examples

### Colored Output
The interface uses different colors to categorize information:
- 🟢 **Green**: Success messages and confirmations
- 🔴 **Red**: Error messages and failures
- 🟡 **Yellow**: Warnings and important notices
- 🔵 **Blue**: Information and system messages
- 🟣 **Purple**: Headers and banners

### Progress Indicators
Long operations show visual progress:
```
🚶 Moving to map coordinates (512.0, 256.0)...
[████████████████████████████████] 100.0%
✅ Movement successful!
```

### Session Statistics
Track your exploration progress:
```
📊 SESSION STATISTICS
========================================
Session Duration: 00:15:42
Commands Executed: 23
Movements: 8
View Rotations: 5
Images Saved: 12
Scene Switches: 2
Current Scene: habitat-test-scenes/van-gogh-room
Current Position: (512.0, 256.0)
```

## 🎮 Interactive Modes

### Demo Mode
Automated demonstration showing system capabilities:
```bash
🤖 Command > demo
🎬 Starting interactive demonstration...
⚠️  This demo will move the agent and rotate the view.
Press Enter to continue with step 1: Moving to center of map
```

### Exploration Mode
Guided exploration with suggested locations:
```bash
🤖 Command > explore
🗺️  Starting guided exploration mode...
Suggested exploration points:
  1. Center of the map - Map coords: (512, 512)
  2. Northwest area - Map coords: (312, 312)
  3. Northeast area - Map coords: (712, 312)
  4. Southeast area - Map coords: (712, 712)
  5. Southwest area - Map coords: (312, 712)
Choose point (1-5) or 'q' to quit:
```

## 🔧 Advanced Features

### Command History
- Persistent across sessions
- Searchable with arrow keys
- Stored in `.command_history` file

### Auto-completion
- Tab completion for all commands
- Intelligent command suggestions
- Context-aware completion

### Session Management
- Automatic session tracking
- Statistics collection
- Error recovery and logging

### Enhanced Error Handling
- Detailed error messages
- Helpful suggestions
- Graceful degradation

## 🛠️ Customization

### Colors
Modify the `Colors` class in `enhanced_terminal_interface.py`:
```python
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    # ... customize as needed
```

### Progress Bar
Adjust progress bar appearance:
```python
progress = ProgressBar(total=10, width=40)
```

### Session Statistics
Add custom statistics in the `session_stats` dictionary:
```python
self.session_stats = {
    'commands_executed': 0,
    'movements': 0,
    'custom_metric': 0,
    # ... add your metrics
}
```

## 🐛 Troubleshooting

### Common Issues

**HM3D dataset not found**
- Run: `python download_datasets.py` and select option 1
- Or manually: `python -m habitat_sim.utils.datasets_download --uids hm3d_minival --data-path ./data`
- Check that the data directory contains `scene_datasets/hm3d/`

**Position selection map not displaying**
- Ensure matplotlib is installed: `pip install matplotlib`
- Check that you have a display environment (for remote servers, use X11 forwarding)
- The map is also saved as an image file in the images/ directory

**Coordinate grid not showing**
- Ensure terminal supports Unicode characters for grid symbols
- Update terminal to support ANSI colors
- Grid overlay is also saved in image files

**Colors not showing**
- Ensure terminal supports ANSI colors
- On Windows, use Windows Terminal or enable color support

**Tab completion not working**
- Install `readline` module: `pip install readline`
- Some terminals may not support tab completion

**Progress bars appearing garbled**
- Update terminal to support Unicode characters
- Use `--basic` flag for basic interface

**Command history not persisting**
- Check write permissions in workspace directory
- Ensure `.command_history` file is not read-only

### Performance Tips
- Use `clear` command to refresh screen
- Use `stats` to monitor system resources
- Close and reopen for memory management

## 📋 Requirements

- Python 3.7+
- Terminal with ANSI color support
- Unicode support for emojis and progress bars
- readline module for command completion

## 🔗 Related Files

- `enhanced_terminal_interface.py` - Main enhanced interface
- `interactive_navigation.py` - Basic interface fallback
- `coordinate_navigation.py` - Core navigation system
- `launch.py` - Multi-interface launcher
- `test_enhanced_interface.py` - Interface testing

## 🤝 Contributing

To add new features to the enhanced interface:

1. Add new commands to the `commands` list
2. Implement command handling in `process_command()`
3. Add help text in `show_enhanced_help()`
4. Update session statistics if needed
5. Add color coding for new message types

## 📞 Support

If you encounter issues with the enhanced interface:
1. Run `python test_enhanced_interface.py`
2. Check terminal compatibility
3. Use `python launch.py --basic` as fallback
4. Report issues with terminal type and Python version
