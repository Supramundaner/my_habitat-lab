# Image-Instance Navigation Preprocessing Pipeline

This pipeline preprocesses scenes and episodes for Image-Instance Navigation tasks, where an agent must find a specific object instance shown in a target image.

## Overview

The pipeline consists of 8 steps (0-7) that process a scene and episode data to generate navigation targets:

1. **Step 0**: Render goal image from episode data
2. **Step 1**: Generate topdown view and metadata  
3. **Step 2**: Generate wall mask from topdown view
4. **Step 3**: Perform room segmentation and annotation
5. **Step 4**: Use LLM to select target room based on goal image
6. **Step 5**: Generate navigation graph using Poisson Disk Sampling
7. **Step 6**: Use LLM to select optimal navigation node within target room
8. **Step 7**: Generate final action.json file

## File Structure

```
insnav_preprocess/
├── main.py                      # Main workflow orchestrator
├── input_config.json           # Configuration template
├── current_topdown.py          # Topdown view generation (copied from preprocess/)
├── step_0_render_goal_image.py # Render target image from episode data
├── step_1_generate_topdown.py  # Generate scene topdown view
├── step_2_generate_wall_mask.py# Generate wall mask
├── step_3_room_segmentation.py # Segment rooms
├── step_4_llm_room_selection.py# LLM room selection
├── step_5_graph_generation.py  # Generate navigation graph
├── step_6_node_selection.py    # LLM node selection
├── step_7_path_planning.py     # Generate action.json
└── prompts/
    ├── choose_room_prompt.txt   # Prompt template for room selection
    └── choose_node_prompt.txt   # Prompt template for node selection
```

## Configuration

Edit `input_config.json` with your specific paths and parameters:

```json
{
    "scene_config": {
        "scene_path": "/path/to/your/scene.glb",
        "episodes_file": "/path/to/your/episodes.json", 
        "episode_id": 0,
        "custom_ortho_scale": null,
        "target_coverage": 0.9,
        "draw_coordinates": false
    },
    "output": {
        "output_dir": "output_insnav"
    },
    "resolution": [2048, 2048],
    "room_segmentation": {
        "closing_width_m": 0.05,
        "min_room_area_m2": 2.0,
        "seed_distance_m": null,
        "boundary_length_threshold_m": 0.3
    },
    "graph_generation": {
        "node_spacing_m": 1.0,
        "wall_padding_m": 0.5,
        "min_component_area": 100
    },
    "llm_config": {
        "api_key": "your_doubao_api_key",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model": "doubao-seed-1-6-250615",
        "max_tokens": 1000,
        "max_retries": 3
    },
    "prompts": {
        "choose_room_prompt": "prompts/choose_room_prompt.txt",
        "choose_node_prompt": "prompts/choose_node_prompt.txt"
    }
}
```

## Episodes JSON Format

Your episodes JSON file should follow this structure (example from `example.json`):

```json
{
    "goals": {
        "scene_object_id": {
            "position": [x, y, z],
            "object_id": 123,
            "object_name": "chair_123",
            "object_category": "chair", 
            "object_surface_area": 1.5,
            "view_points": [...],
            "image_goals": [
                {
                    "position": [x, y, z],
                    "rotation": [x, y, z, w],
                    "frame_coverage": 0.1,
                    "object_coverage": 0.7,
                    "hfov": 63,
                    "image_dimensions": [512, 512]
                }
            ]
        }
    },
    "episodes": [
        {
            "scene_dataset_config": "path/to/config.json",
            "start_position": [x, y, z],
            "start_rotation": [x, y, z, w],
            "goal_object_id": "123",
            "goal_image_id": 0,
            "object_category": "chair"
        }
    ]
}
```

## Usage

1. **Install Dependencies**:
   ```bash
   pip install habitat-sim
   pip install volcenginesdkarkruntime  # For Doubao LLM API
   pip install opencv-python numpy scikit-learn scipy
   ```

2. **Copy current_topdown.py**:
   The `current_topdown.py` file should already be present in the insnav_preprocess folder (copied from preprocess/).

3. **Configure**:
   - Update `input_config.json` with your scene path, episodes file, and API key
   - Adjust episode_id to select which episode to process
   - Configure LLM settings (API key, model, etc.)

4. **Run Pipeline**:
   ```bash
   cd insnav_preprocess/
   python main.py input_config.json
   ```

## Output

The pipeline generates:

- `output_insnav/` directory containing:
  - `goal_image.png` - Rendered target image
  - `topdown_view.png` - Scene topdown view
  - `wall_mask.png` - Binary wall mask
  - `room_segmentation.png` - Colored room segmentation
  - `room_annotation.png` - Room annotation with numbers
  - `graph_with_topdown.png` - Navigation graph visualization
  - `action.json` - Final output with agent state and target info
  - Various metadata and log files

## Key Differences from Object-Goal Navigation

1. **Target Specification**: Uses specific object instances with target images instead of object categories
2. **Episode-based**: Processes individual episodes with specific start positions and target instances  
3. **Enhanced LLM Prompts**: Uses goal images to guide room and node selection
4. **Image Rendering**: Step 0 renders the target image that the agent needs to find

## Troubleshooting

- **LLM Not Available**: The pipeline includes fallback mechanisms if LLM API is unavailable
- **No Nodes in Room**: Adjust `node_spacing_m` or `wall_padding_m` if rooms have no navigation nodes
- **Room Segmentation Issues**: Adjust `closing_width_m` and `min_room_area_m2` parameters
- **Import Errors**: Ensure all dependencies are installed and paths are correct

## Example Command

```bash
# Process episode 0 from your episodes file
python main.py input_config.json
```

The pipeline will automatically:
1. Load episode 0 from your episodes.json file
2. Extract the target object and image goal information  
3. Render the target image
4. Process the scene and generate navigation targets
5. Output action.json for the navigation system
