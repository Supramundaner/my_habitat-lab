# One-Stage Coordinate-Based Navigation Target Selection

This project implements a direct coordinate-based navigation target selection method where the LLM selects optimal navigation coordinates directly from the topdown view.

## Overview

### Coordinate-Based Method
1. Generate topdown view and metadata
2. LLM directly selects optimal navigation coordinate from normalized topdown image
3. Generate path planning with selected coordinate

## Key Features

- **Direct Coordinate Selection**: LLM chooses (x, y) coordinates directly from a normalized 1000x1000 image
- **No Node Generation**: Eliminates the need for PDS sampling and node generation
- **Simplified Pipeline**: Only 3 steps instead of complex multi-step processes
- **Flexible Targeting**: Can select any coordinate in navigable space, not limited to pre-generated points

## Usage

```bash
python main_workflow_onestage.py input_config_onestage.json
```

## Configuration

The configuration file (`input_config_onestage.json`) includes:

```json
{
  "workflow_type": "one_stage_coordinate",
  "coordinate_selection": {
    "image_normalize_size": 1000,
    "coordinate_validation": true
  },
  "prompts": {
    "choose_coordinate_prompt": "prompts/choose_coordinate.txt"
  }
}
```

### Key Configuration Options

- `image_normalize_size`: Target size for image normalization (default: 1000)
- `coordinate_validation`: Whether to validate coordinates are in navigable areas
- `choose_coordinate_prompt`: Path to the coordinate selection prompt

## Coordinate System

- **Normalized Image**: 1000x1000 pixels
- **Origin**: Top-left corner (0, 0)
- **Range**: x ∈ [0, 1000], y ∈ [0, 1000]
- **Output**: Integer coordinates (x, y)

## Workflow Steps

### Step 0: Generate Topdown View
- Creates topdown view of the environment
- Generates metadata for coordinate conversion

### Step 1: Coordinate Selection
- Normalizes topdown image to 1000x1000
- LLM analyzes the image and selects optimal coordinate
- Validates coordinate is in navigable space
- Converts coordinate back to original image space

### Step 2: Path Planning
- Converts pixel coordinates to world coordinates
- Generates action.json for navigation

## Output Files

### Generated Files
- `normalized_topdown.png` - 1000x1000 normalized topdown view
- `selected_coordinate_visualization.png` - Original image with selected coordinate
- `selected_coordinate_normalized_visualization.png` - Normalized image with coordinate
- `coordinate_selection_log.json` - Complete selection log
- `action.json` - Final navigation action file

### Preprocessing Files
- `preprocess/llm_coordinate_selection_content.json` - LLM response details
- `preprocess/llm_reasoning.txt` - Human-readable reasoning

## Advantages Over Node-Based Methods

1. **Higher Flexibility**: Can select any coordinate, not limited to predefined nodes
2. **Simpler Pipeline**: No need for wall mask generation or graph construction
3. **Better Precision**: Direct coordinate selection for optimal positioning
4. **Faster Processing**: Fewer computation steps
5. **Intuitive Interface**: Clear coordinate system that's easy to understand

## Prompt Design

The LLM prompt (`prompts/choose_coordinate.txt`) instructs the model to:
- Analyze the goal object and typical placement
- Identify suitable rooms/areas
- Select optimal navigation coordinates
- Avoid obstacles and walls
- Return coordinates in the specified format

Example prompt format:
```
Your final answer should be: coordinate (x, y)
Example: coordinate (342, 567)
```

## Manual Testing

For testing without LLM:

```bash
python step_3_onestage_coordinate_selection.py topdown.png config.json 500,600
```

This will manually select coordinate (500, 600) instead of using LLM.

## Coordinate Conversion

The system handles three coordinate spaces:
1. **Normalized**: 1000x1000 for LLM interaction
2. **Original**: Native topdown image resolution
3. **World**: 3D environment coordinates for navigation

Conversions are handled automatically with proper scaling and offset calculations.

## Error Handling

- **Invalid Coordinates**: Automatic fallback to image center
- **LLM Failures**: Retry mechanism with fallback selection
- **Validation Errors**: Optional coordinate validation for navigability
- **Conversion Errors**: Robust coordinate space conversion with fallbacks

## Best Practices

1. Use `coordinate_validation: true` for better reliability
2. Adjust `image_normalize_size` based on environment complexity
3. Customize prompts for specific object types or environments
4. Test coordinate conversions with known reference points
