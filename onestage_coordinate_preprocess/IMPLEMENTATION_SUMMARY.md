# One-Stage Coordinate-Based Implementation Summary

## Changes Made

### 1. Configuration File Updated
- **File**: `input_config_onestage.json`
- **Changes**:
  - Changed `workflow_type` from "one_stage" to "one_stage_coordinate"
  - Removed `wall_mask` and `graph_generation` sections (no longer needed)
  - Added `coordinate_selection` section with normalization and validation settings
  - Updated prompt path to use `choose_coordinate_prompt`

### 2. New Prompt File Created
- **File**: `prompts/choose_coordinate.txt`
- **Purpose**: Instructs LLM to select coordinates directly from normalized 1000x1000 image
- **Format**: Expects output in format "coordinate (x, y)"

### 3. New Coordinate Selection Module
- **File**: `step_3_onestage_coordinate_selection.py`
- **Features**:
  - Direct coordinate selection from topdown image
  - Image normalization to 1000x1000 pixels
  - Coordinate validation for navigable areas
  - Robust coordinate space conversion
  - LLM retry mechanism with fallback
  - Manual selection option for testing

### 4. Updated Main Workflow
- **File**: `main_workflow_onestage.py`
- **Changes**:
  - Removed steps for wall mask generation and graph generation
  - Updated to use coordinate selection instead of node selection
  - Simplified workflow to only 3 steps:
    1. Generate topdown view
    2. Select coordinate
    3. Generate action.json

### 5. Updated Path Planning
- **File**: `step_5_path_planning.py`
- **Changes**:
  - Modified to read coordinate selection results instead of node selection
  - Added pixel-to-world coordinate conversion
  - Updated action.json format to include coordinate information

### 6. New Documentation
- **File**: `README_coordinate.md`
- **Content**: Complete documentation for the coordinate-based approach

## Key Advantages of the New Implementation

1. **Simplified Pipeline**: No need for wall mask or graph generation
2. **Direct Selection**: LLM selects coordinates directly, not from pre-generated nodes
3. **Flexible Positioning**: Can select any coordinate in navigable space
4. **Standardized Interface**: Consistent 1000x1000 coordinate system
5. **Better Error Handling**: Robust fallback mechanisms

## Usage Instructions

### Basic Usage
```bash
cd /home/yaoaa/habitat-lab/onestage_coordinate_preprocess
python main_workflow_onestage.py input_config_onestage.json
```

### Manual Testing
```bash
python step_3_onestage_coordinate_selection.py topdown.png input_config_onestage.json 500,600
```

### Configuration Options
- `image_normalize_size`: Target normalization size (default: 1000)
- `coordinate_validation`: Enable/disable coordinate validation
- `max_retries`: Number of LLM retry attempts
- `goal_object`: Target object to find

## Output Files Structure

```
output/
├── topdown_view.png                           # Original topdown view
├── normalized_topdown.png                     # 1000x1000 normalized image
├── selected_coordinate_visualization.png      # Original with selected coordinate
├── selected_coordinate_normalized_visualization.png  # Normalized with coordinate
├── coordinate_selection_log.json              # Complete selection log
├── action.json                                # Final navigation action
├── metadata.json                              # Topdown generation metadata
└── preprocess/
    ├── llm_coordinate_selection_content.json  # LLM response details
    └── llm_reasoning.txt                       # Human-readable reasoning
```

## Coordinate System

- **Input**: 1000x1000 normalized image
- **Origin**: Top-left (0, 0)
- **Range**: x, y ∈ [0, 1000]
- **Output**: Integer coordinates
- **Conversion**: Automatic conversion between normalized, original, and world coordinates

## Error Handling

- Invalid coordinates → fallback to image center
- LLM failures → retry mechanism with fallback
- Validation errors → optional coordinate validation
- Conversion errors → robust coordinate space conversion

The implementation is now complete and ready for testing. The system provides a much simpler and more flexible approach to navigation target selection compared to the previous node-based method.
