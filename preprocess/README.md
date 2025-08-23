# Navigation Target Selection Workflow

A modular workflow for selecting navigation targets in 3D environments using Habitat-Sim, computer vision, and Large Language Models (LLMs).

## Overview

This workflow takes a 3D scene and uses multiple AI techniques to automatically select optimal navigation targets:

1. **Topdown View Generation**: Creates orthographic top-down views of 3D scenes
2. **Wall Mask Generation**: Identifies walkable vs non-walkable areas
3. **Room Segmentation**: Automatically segments and annotates rooms
4. **LLM Room Selection**: Uses AI to select target rooms
5. **Graph Generation**: Creates navigation graphs using Poisson Disk Sampling
6. **Node Selection**: Uses AI to select optimal navigation nodes

## Files Structure

```
preprocess/
├── input_config.json              # Configuration file
├── main_workflow.py               # Main orchestrator
├── current_topdown.py             # Topdown view generation (existing)
├── step_0_generate_topdown.py     # Step 0: Generate topdown view
├── step_1_generate_wall_mask.py   # Step 1: Generate wall mask
├── step_2_room_segmentation.py    # Step 2: Room segmentation
├── step_3_llm_room_selection.py   # Step 3: LLM room selection  
├── step_4_graph_generation.py     # Step 4: Navigation graph generation
├── step_5_node_selection.py       # Step 5: Navigation node selection
├── prompts/
│   ├── choose_a_room.txt          # LLM prompt for room selection
│   └── choose_a_node.txt          # LLM prompt for node selection
└── output/                        # Generated outputs
    ├── output.json                # Complete workflow log
    ├── topdown_view.png           # Generated topdown view
    ├── wall_mask.png              # Wall/walkable area mask
    ├── room_annotation.png        # Annotated rooms with numbers
    ├── graph_with_topdown_view.png # Navigation graph overlay
    ├── room_with_graph.png        # Cropped room with nodes
    └── node_with_topdown.png      # Final result with selected node
```

## Configuration

Edit `input_config.json` to configure the workflow:

```json
{
    "scene_config": {
        "scene_path": "/path/to/your/scene.glb",
        "target_floor": 0,
        "target_coordinate": null,
        "custom_ortho_scale": null,
        "target_coverage": 0.9,
        "draw_coordinates": true
    },
    "room_segmentation": {
        "morph_closing_width_meters": 0.01,
        "seed_min_distance_from_wall_meters": 0.9,
        "min_room_area_pixels": 1000
    },
    "graph_generation": {
        "pds_radius": 1.0,
        "max_attempts": 10000,
        "node_radius_pixels": 8
    },
    "llm_config": {
        "api_key": "your_gemini_api_key",
        "model": "gemini-2.5-flash",
        "max_tokens": 1000
    },
    "prompts": {
        "choose_room_prompt": "preprocess/prompts/choose_a_room.txt",
        "choose_node_prompt": "preprocess/prompts/choose_a_node.txt"
    },
    "output": {
        "output_dir": "preprocess/output"
    }
}
```

## Usage

### Running the Complete Workflow

```bash
cd /home/yaoaa/habitat-lab
python preprocess/main_workflow.py preprocess/input_config.json
```

### Running Individual Steps

Each step can be run independently for testing:

```bash
# Step 0: Generate topdown view
python step_0_generate_topdown.py input_config.json

# Step 1: Generate wall mask
python step_1_generate_wall_mask.py output/topdown_view.png output/

# Step 2: Room segmentation
python step_2_room_segmentation.py output/topdown_view.png output/wall_mask.png output/metadata.json input_config.json

# Step 3: LLM room selection
python step_3_llm_room_selection.py output/topdown_view.png output/room_annotation.png input_config.json

# Step 4: Navigation graph generation
python step_4_graph_generation.py output/topdown_view.png output/wall_mask.png output/metadata.json input_config.json

# Step 5: Node selection
python step_5_node_selection.py output/graph_with_topdown_view.png output/topdown_view.png '{"x_min":100,"y_min":100,"x_max":300,"y_max":300}' 1 input_config.json
```

## Dependencies

- OpenCV (`cv2`)
- NumPy
- scikit-learn (for DBSCAN clustering)
- PIL (Pillow)
- Habitat-Sim
- requests (for REST API calls)

Install dependencies:
```bash
pip install opencv-python numpy scikit-learn pillow requests
```

## API Configuration

This workflow uses the Google Gemini REST API. The configuration uses the curl-compatible format:

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent" \
  -H 'Content-Type: application/json' \
  -H 'X-goog-api-key: YOUR_API_KEY' \
  -X POST \
  -d '{
    "contents": [
      {
        "parts": [
          {
            "text": "Your prompt here"
          }
        ]
      }
    ]
  }'
```

Configure in `input_config.json`:
```json
{
    "llm_config": {
        "api_key": "your_gemini_api_key",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/models",
        "model": "gemini-2.0-flash",
        "max_tokens": 1000
    }
}
```

## Workflow Steps Details

### Step 0: Topdown View Generation
- Uses Habitat-Sim to create orthographic top-down views
- Automatically detects floors and renders specified floor
- Outputs: `topdown_view.png`, `metadata.json`

### Step 1: Wall Mask Generation  
- Processes topdown view to identify walls vs walkable areas
- Creates binary mask (white=walkable, black=walls)
- Outputs: `wall_mask.png`, `wall_mask_visualization.png`

### Step 2: Room Segmentation
- Uses watershed algorithm with distance transform
- Segments rooms based on walkable areas
- Creates annotated visualization with room numbers
- Outputs: `room_annotation.png`, `room_segmentation.png`, room bounding boxes

### Step 3: LLM Room Selection
- Uses Google Gemini to analyze room layouts
- Selects target room based on scene understanding
- Outputs: LLM response log, selected room number

### Step 4: Navigation Graph Generation
- Uses Poisson Disk Sampling to create evenly distributed nodes
- Ensures nodes are placed only in walkable areas
- Converts pixel coordinates to world coordinates
- Outputs: `graph_with_topdown_view.png`, `navigation_nodes.json`

### Step 5: Navigation Node Selection
- Crops graph to selected room
- Uses LLM to choose optimal navigation node
- Highlights selected node on full topdown view
- Outputs: `room_with_graph.png`, `node_with_topdown.png`, node selection log

## Output

The workflow generates `output.json` containing:

```json
{
    "workflow_status": "completed_successfully",
    "steps_completed": ["step_0_topdown_generation", "step_1_wall_mask", ...],
    "generated_files": {
        "topdown_view": "path/to/topdown_view.png",
        "wall_mask": "path/to/wall_mask.png",
        ...
    },
    "llm_responses": {
        "step_3_room_selection": {"selected_room": 2, "raw_response": "2"},
        "step_5_node_selection": {"selected_node_id": 15, "raw_response": "15"}
    },
    "final_results": {
        "selected_room_number": 2,
        "selected_node_id": 15,
        "selected_node_world_coordinates": [1.23, -0.5, 2.45],
        "room_bounding_boxes": {...}
    },
    "errors": []
}
```

## Troubleshooting

### Common Issues

1. **Scene file not found**: Ensure the scene path in `input_config.json` is correct
2. **LLM API errors**: Verify your Gemini API key is valid
3. **No rooms detected**: Adjust room segmentation parameters
4. **No navigation nodes**: Reduce PDS radius or increase max attempts

### Debug Mode

Set environment variable for debugging:
```bash
export DEBUG_WORKFLOW=1
python main_workflow.py input_config.json
```

### Manual Testing

Each module can be tested independently. Use the manual selection mode for testing without LLM:

```bash
# Manual room selection (select room 1)
python step_3_llm_room_selection.py topdown.png room_annotation.png config.json 1
```

## API Requirements

This workflow uses the Google Gemini REST API for LLM functionality:

- Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- Set in `input_config.json` under `llm_config.api_key`
- Model used: `gemini-2.0-flash` (configurable)
- Base URL: `https://generativelanguage.googleapis.com/v1beta/models`

### Testing the API

Before running the full workflow, test the API integration:

```bash
cd /home/yaoaa/habitat-lab/preprocess
python test_api_integration.py
```

This will test both text-only and image+text API calls to ensure everything is working correctly.

## License

This workflow is part of the Habitat-Lab project. Please refer to the main project license.
