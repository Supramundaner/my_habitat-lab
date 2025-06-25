# Quick Start Guide - Habitat Map Navigation

## 🚀 Get Started in 5 Minutes

### Step 1: Verify Your System
```bash
python verify_system.py
```
This checks if all dependencies are installed correctly.

### Step 2: Run Basic Tests
```bash
python test_basic.py
```
This tests core functionality without requiring Habitat data.

### Step 3: Launch the System
```bash
python launch.py
```
This starts the interactive navigation system.

## 📋 First Session Commands

Once the system starts, try these commands:

```
> help                    # Show all available commands
> move 0 0               # Move to map center
> turn right 45          # Turn right 45 degrees
> look up 20             # Look up 20 degrees
> move 5.0 -3.0          # Move to specific coordinates
> quit                   # Exit the system
```

## 📁 Generated Files

After each command, check the `output_images/` folder for:
- `init_*.png` - Initial system state
- `step_01_*.png` - After first command
- `step_02_*.png` - After second command
- etc.

Each step generates:
- `*_fpv.png` - First-person view
- `*_tpv.png` - Third-person view  
- `*_map.png` - Annotated map
- `*_composite.png` - Combined view

## 🗺️ Understanding the Map

- **White areas**: Navigable space
- **Gray areas**: Non-navigable/obstacles
- **Black grid lines**: Coordinate system
- **Numbers on grid**: World coordinates
- **Red marker with arrow**: Agent position and direction

## ⚠️ Troubleshooting

### "Failed to initialize system"
1. Check that Habitat-sim and Habitat-lab are installed
2. Verify MP3D dataset is downloaded
3. Run `python verify_system.py` for detailed diagnosis

### "Target position is not navigable"
- Only move to white (navigable) areas on the map
- Avoid gray (obstacle) areas
- Use coordinates shown on the grid labels

### Import errors
- Make sure you're running from the project directory
- Check that all requirements are installed: `pip install -r requirements.txt`

## 🎯 Example Session

```
> python launch.py

# System initializes and shows initial position
# Initial images saved as init_*.png

> move 10.5 -5.2         # Move to coordinates (10.5, -5.2)
# Agent moves, new images saved as step_01_*.png

> turn left 90           # Turn left 90 degrees  
# Agent rotates, new images saved as step_02_*.png

> look down 15           # Tilt camera down 15 degrees
# Camera adjusts, new images saved as step_03_*.png

> quit                   # Exit system
```

## 🔧 Configuration

### Change Scene
Edit `configs/navigation_config.yaml`:
```yaml
habitat:
  dataset:
    scenes: ["your_scene_id"]
```

### Custom Output Directory
```bash
python launch.py --output my_custom_images
```

### Use Different Config
```bash
python launch.py --config configs/example_config.yaml
```

## 📞 Need Help?

1. **Read the full README.md** for detailed documentation
2. **Run verification**: `python verify_system.py`
3. **Check basic functionality**: `python test_basic.py`
4. **View example config**: `configs/example_config.yaml`

## 🎉 Success Indicators

You'll know everything is working when:
- ✅ Verification script passes all checks
- ✅ System initializes without errors
- ✅ Images are generated in `output_images/`
- ✅ Map shows your scene with coordinate grid
- ✅ Agent marker appears on the map
- ✅ Commands execute successfully

Happy navigating! 🧭
