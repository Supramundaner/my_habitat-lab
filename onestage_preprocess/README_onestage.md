# One-Stage vs Two-Stage Navigation Target Selection

This project has been updated to support both two-stage and one-stage navigation target selection methods.

## Overview

### Two-Stage Method (Original)
1. Generate topdown view and metadata
2. Generate wall mask
3. Perform room segmentation
4. LLM selects target room
5. Generate navigation graph
6. LLM selects navigation node within the target room
7. Generate path planning

### One-Stage Method (New)
1. Generate topdown view and metadata
2. Generate wall mask
3. Generate navigation graph (skip room segmentation)
4. LLM directly selects navigation node from the entire environment
5. Generate path planning

## Usage

### Running Two-Stage Method
```bash
python main_workflow.py input_config.json
```

### Running One-Stage Method
```bash
# Option 1: Use the unified main_workflow.py with one-stage config
python main_workflow.py input_config_onestage.json

# Option 2: Use the dedicated one-stage workflow
python main_workflow_onestage.py input_config_onestage.json
```

## Configuration Files

### Two-Stage Configuration (`input_config.json`)
- Includes `"workflow_type": "two_stage"`
- Has room segmentation parameters
- Uses both room selection and node selection prompts

### One-Stage Configuration (`input_config_onestage.json`)
- Includes `"workflow_type": "one_stage"`
- No room segmentation parameters needed
- Uses only the one-stage node selection prompt

## Key Differences

| Aspect | Two-Stage | One-Stage |
|--------|-----------|-----------|
| Steps | 7 steps | 5 steps |
| Room Segmentation | Required | Skipped |
| LLM Calls | 2 (room + node) | 1 (node only) |
| Prompt Files | `choose_a_room.txt`, `choose_a_node.txt` | `choose_onestage_node.txt` |
| Processing Time | Longer | Faster |
| Precision | Higher (room-focused) | Lower (whole environment) |

## Files Structure

### Core Files
- `main_workflow.py` - Unified workflow supporting both methods
- `main_workflow_onestage.py` - Dedicated one-stage workflow
- `step_3_onestage_node_selection.py` - One-stage node selection implementation

### Configuration Files
- `input_config.json` - Two-stage configuration
- `input_config_onestage.json` - One-stage configuration

### Prompt Files
- `prompts/choose_a_room.txt` - Room selection prompt (two-stage)
- `prompts/choose_a_node.txt` - Node selection prompt (two-stage)
- `prompts/choose_onestage_node.txt` - Direct node selection prompt (one-stage)

## Output Files

Both methods generate similar output structures, but one-stage method:
- Skips room-related outputs (`room_annotation.png`, room selection logs)
- Generates `selected_node_with_topdown.png` instead of `node_with_topdown.png`
- Creates `onestage_node_selection_log.json` with method-specific information

## When to Use Which Method

### Use Two-Stage When:
- High precision is required
- The environment has clear room boundaries
- You want more interpretable decision-making
- Processing time is not critical

### Use One-Stage When:
- Speed is important
- The environment has open layouts
- Room boundaries are unclear
- You want simpler processing pipeline

## Example Commands

```bash
# Two-stage with TV target
python main_workflow.py input_config.json

# One-stage with TV target
python main_workflow.py input_config_onestage.json

# Modify goal object in config file before running
# Change "goal_object": "tv" to "goal_object": "sofa", etc.
```
