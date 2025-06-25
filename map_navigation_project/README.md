# Habitat Map Navigation Project

An interactive 2D map-based navigation system for Habitat-sim environments using Matterport3D (MP3D) datasets.

## Overview

This project provides a complete navigation system that allows users to:
- Load MP3D scenes in Habitat-sim
- View colorized top-down maps with coordinate grids
- Command an agent to navigate using 2D map coordinates
- Control agent orientation and camera angles
- Generate comprehensive visualization outputs

## Features

### 🗺️ Map System
- **Colorized Top-Down Maps**: Uses MP3D dataset's native colorized top-down views
- **Coordinate Grid Overlay**: Black grid lines with world coordinate labels for precise navigation
- **Agent Visualization**: Red marker with directional arrow showing agent position and orientation
- **Coordinate System Alignment**: Perfect alignment between map coordinates and world coordinates

### 🤖 Agent Control
- **Map-Based Navigation**: `move <x> <y>` - Navigate to specific map coordinates
- **Rotation Control**: `turn left/right <degrees>` - Rotate agent horizontally
- **Camera Control**: `look up/down <degrees>` - Adjust camera pitch angle
- **Automatic Pathfinding**: Intelligent navigation to valid positions

### 📸 Visualization Output
- **First-Person View**: Agent's RGB camera perspective
- **Third-Person View**: Simulated external view (placeholder implementation)
- **Top-Down Map**: Annotated map with agent position and coordinate grid
- **Composite View**: Combined visualization showing all perspectives

## Project Structure

```
map_navigation_project/
├── main_controller.py      # Main control loop and user interface
├── habitat_env.py          # Habitat environment management
├── map_visualizer.py       # Map visualization and rendering
├── configs/
│   └── navigation_config.yaml  # Habitat configuration
├── output_images/          # Generated visualization images
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

### Prerequisites
- Python 3.7+
- Habitat-sim installed and configured
- Habitat-lab installed
- MP3D dataset downloaded and configured

### Setup Steps

1. **Clone or copy this project** to your habitat-lab root directory:
   ```bash
   # From habitat-lab root directory
   cd habitat-lab/
   # Copy the map_navigation_project folder here
   ```

2. **Install dependencies**:
   ```bash
   cd map_navigation_project/
   pip install -r requirements.txt
   ```

3. **Verify Habitat installation**:
   ```bash
   # Make sure habitat-lab is in your Python path
   python -c "import habitat; print('Habitat imported successfully')"
   ```

4. **Configure MP3D dataset path**:
   - Edit `configs/navigation_config.yaml`
   - Update the `scenes_dir` and `data_path` to match your MP3D dataset location
   - Optionally change the default scene in the `scenes` list

## Usage

### Quick Start

1. **Run the system**:
   ```bash
   python main_controller.py
   ```

2. **Wait for initialization**:
   - The system will load the scene and generate initial images
   - Initial images are saved as `init_*.png` in the `output_images/` folder

3. **Start navigating**:
   ```
   > move 5.2 -3.8    # Move to map coordinates (5.2, -3.8)
   > turn right 45    # Turn right 45 degrees
   > look up 20       # Look up 20 degrees
   > help             # Show all commands
   > quit             # Exit the system
   ```

### Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `move <x> <y>` | Navigate to map coordinates | `move 10.5 -2.3` |
| `turn <direction> <degrees>` | Rotate agent | `turn left 90` |
| `look <direction> <degrees>` | Adjust camera pitch | `look down 15` |
| `help` | Show command help | `help` |
| `quit` or `exit` | Exit the program | `quit` |

### Understanding the Coordinate System

- **Map Coordinates**: The coordinate grid displayed on the map shows world coordinates
- **World Alignment**: Map coordinates directly correspond to Habitat world coordinates
- **Navigation**: Use the coordinate labels on the map to determine where to move
- **Agent Marker**: Red circle with arrow shows current position and facing direction

### Output Images

After each command, the system generates:

1. **`stepXX_fpv.png`** - First-person RGB view
2. **`stepXX_tpv.png`** - Third-person view (simulated)
3. **`stepXX_map.png`** - Annotated top-down map
4. **`stepXX_composite.png`** - Combined view with all perspectives

## Configuration

### Scene Selection

Edit `configs/navigation_config.yaml` to change the scene:

```yaml
habitat:
  dataset:
    scenes: ["17DRP5sb8fy"]  # Change to your desired scene ID
```

### Map Resolution

Adjust map resolution in the config:

```yaml
habitat:
  task:
    measurements:
      top_down_map:
        map_resolution: 1024  # Higher = more detailed
```

### Agent Parameters

Modify agent settings:

```yaml
habitat:
  simulator:
    agents:
      main_agent:
        height: 1.5      # Agent height
        radius: 0.1      # Agent radius
```

## Advanced Usage

### Custom Scene Loading

```python
from main_controller import NavigationController

# Initialize with specific scene
controller = NavigationController(
    config_path="configs/navigation_config.yaml",
    output_dir="custom_output",
    scene_id="your_scene_id"
)

controller.initialize_system()
controller.run_interactive_loop()
```

### Programmatic Control

```python
# Execute commands programmatically
controller.initialize_system()

# Move to specific coordinates
controller._execute_move_command(10.0, -5.0)

# Turn and look
controller._execute_turn_command("right", 45)
controller._execute_look_command("up", 20)
```

## Troubleshooting

### Common Issues

1. **"Import habitat could not be resolved"**
   - Ensure habitat-lab is installed: `pip install habitat-sim`
   - Check Python path includes habitat-lab directory

2. **"Failed to load map data"**
   - Verify MP3D dataset is downloaded and configured
   - Check scene ID exists in your dataset
   - Ensure proper permissions for data directory

3. **"Target position is not navigable"**
   - Choose coordinates within the white (navigable) areas on the map
   - Avoid black (obstacle) or gray (non-navigable) areas

4. **Empty or black images**
   - Check OpenGL/graphics drivers
   - Verify GPU compatibility with Habitat-sim
   - Try running with `--headless` flag if needed

### Debug Mode

Add debug prints by modifying the logging level:

```python
import logging
logging.getLogger("habitat").setLevel(logging.DEBUG)
```

## Technical Details

### Coordinate System

- **Habitat World**: X (right), Y (up), Z (forward)
- **Map Display**: X (horizontal), Y (vertical, inverted)
- **Conversion**: Automatic conversion between coordinate systems

### Map Generation

1. Load scene's native top-down map from MP3D dataset
2. Apply coordinate grid overlay with world coordinate labels
3. Mark agent position with position and orientation indicators
4. Generate high-resolution output images

### Navigation Algorithm

1. Parse user input coordinates (map space)
2. Convert to world coordinates using scene bounds
3. Validate target position is navigable
4. Set agent position and orientation directly
5. Update visualizations

## Contributing

This project is designed to be extensible. Key areas for enhancement:

- **Real Third-Person View**: Implement actual third-person camera rendering
- **Path Visualization**: Show planned path on the map
- **Multi-Agent Support**: Support for multiple agents
- **Interactive Map**: Click-to-navigate interface
- **Scene Comparison**: Side-by-side scene comparisons

## Requirements

See `requirements.txt` for the complete list of dependencies.

## License

This project follows the same license as Habitat-lab.

## Acknowledgments

- Meta AI Research for Habitat-sim and Habitat-lab
- Matterport for the MP3D dataset
- The open-source community for supporting libraries
