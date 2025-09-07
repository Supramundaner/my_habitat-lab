# One-Stage Navigation Evaluation

This evaluation system uses the **one-stage** method for navigation target selection, which directly selects navigation nodes without room segmentation.

## Overview

The one-stage evaluation pipeline consists of:

1. **Episode Data Loading**: Load episode information and goal objects
2. **One-Stage Preprocessing**: Generate topdown view, wall mask, navigation graph, and directly select optimal navigation node using LLM
3. **Video Generation**: Execute navigation actions and generate video
4. **Success Evaluation**: Evaluate navigation success using geodesic distance metrics

## Key Differences from Two-Stage Method

### One-Stage Method (This Implementation)
- Skips room segmentation step
- Uses single LLM call to select navigation node directly from the entire environment
- Faster processing with fewer steps
- Uses `choose_onestage_node.txt` prompt

### Two-Stage Method (Original)
- Includes room segmentation step
- Uses two LLM calls: first to select room, then to select node within that room
- More precise but slower processing
- Uses both room and node selection prompts

## Usage

### Single Episode Evaluation
```bash
python run_eval.py /home/yaoaa/habitat-lab/habitat_video_project/onestage_eval/eval_config_episode_427.json
```

### Batch Evaluation
```bash
python batch_eval.py /home/yaoaa/habitat-lab/habitat_video_project/onestage_eval/challenge_batch_episodes_12_19.json
```

## Configuration

### Main Configuration Structure
```json
{
    "episode": {
        "episode_json_path": "path/to/episodes.json",
        "episode_id": "episode_id"
    },
    "preprocess": {
        "workflow_type": "one_stage",
        "scene_config": {...},
        "wall_mask": {...},
        "graph_generation": {...},
        "llm_config": {...}
    },
    "video_generation": {...},
    "evaluation": {...}
}
```

### Key One-Stage Specific Settings
- `preprocess.workflow_type`: Set to `"one_stage"`
- No `room_segmentation` section needed
- Uses `wall_mask` instead of room segmentation parameters
- LLM prompt points to `choose_onestage_node.txt`

## File Structure

```
onestage_eval/
├── run_eval.py              # Main evaluation script
├── batch_eval.py            # Batch evaluation script
├── test_config.json         # Test configuration
├── eval_config_episode_427.json  # Example episode configuration
├── batch_eval.json          # Batch evaluation configuration
├── prompts/
│   └── choose_onestage_node.txt  # One-stage node selection prompt
└── README.md               # This file
```

## Dependencies

This evaluation system depends on:
- `onestage_preprocess/` - One-stage preprocessing pipeline
- `habitat_video_project/video/` - Video generation system
- Habitat-Sim and related packages for simulation

## Output

Each evaluation produces:
- `preprocess/` - Preprocessing outputs (topdown view, graph, selected node)
- `video/` - Navigation video and execution report
- `output.json` - Final evaluation results with SR/SPL metrics

## Prompt Engineering

The one-stage method uses a single, comprehensive prompt (`choose_onestage_node.txt`) that:
1. Analyzes the goal object and its typical locations
2. Scans the entire environment for rooms/areas
3. Directly selects the optimal navigation node
4. Considers room context and object placement patterns

This approach is faster but may be less precise than the two-stage method for complex environments with unclear room boundaries.

## Example Commands

```bash
# Run single evaluation
python run_eval.py eval_config_episode.json

# Run with verbose output
python run_eval.py test_config.json --verbose

# Run batch evaluation
python batch_eval.py batch_eval.json
```