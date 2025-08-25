# Automated Episode Evaluation System

This directory contains the automated evaluation system for single episode navigation tasks in the Habitat environment.

## Overview

The evaluation system integrates the complete pipeline:
1. **Preprocessing**: Generate navigation actions based on episode data
2. **Video Generation**: Execute actions and generate video output
3. **Evaluation**: Assess navigation success based on distance to target viewpoints

## Files

- `run_eval.py`: Main evaluation script
- `eval_config_template.json`: Template configuration file
- `eval_config_episode_427.json`: Example configuration for episode 427
- `output/`: Directory where evaluation results are stored

## Usage

### Basic Usage

```bash
cd /home/yaoaa/habitat-lab
python run_eval.py eval_config_episode_427.json
python /home/yaoaa/habitat-lab/habitat_video_project/eval/batch_eval.py /home/yaoaa/habitat-lab/habitat_video_project/eval/batch_eval_seen.json
python /home/yaoaa/habitat-lab/habitat_video_project/eval/batch_eval.py /home/yaoaa/habitat-lab/habitat_video_project/eval/batch_episodes_example.json
```

### Verbose Output

```bash
python run_eval.py eval_config_episode_427.json --verbose
```

## Configuration File Format

The configuration file contains several sections:

### Episode Configuration
```json
{
    "episode": {
        "episode_json_path": "/path/to/episode.json",
        "episode_id": "427"
    }
}
```

### Scene Configuration
```json
{
    "scene": {
        "scene_file": "/path/to/scene.glb",
        "robot_urdf": "/path/to/robot.urdf"
    }
}
```

### Preprocessing Configuration
Contains settings for:
- Scene configuration (target coverage, coordinates)
- Room segmentation parameters
- Graph generation parameters
- LLM configuration for room/node selection

### Video Generation Configuration
Contains settings for:
- Video parameters (fps, resolution)
- Agent parameters (speed, sensor height)
- GPU settings
- Navigation parameters
- Object detection settings

### Evaluation Configuration
```json
{
    "evaluation": {
        "success_distance_threshold": 0.25,
        "visible_radius": 3.0
    }
}
```

## Output Structure

For each evaluation run, the system creates a directory under `output/<episode_scene_id>/`:

```
output/y9hTuugGdiq/
├── output.json          # Final evaluation results
├── output.mp4          # Generated navigation video
├── preprocess_config.json  # Generated preprocessing config
├── video_config.json   # Generated video config
├── action.json         # Generated action sequence
├── preprocess/         # Preprocessing outputs
│   ├── topdown_view.png
│   ├── room_annotation.png
│   └── action.json
└── video/              # Video generation outputs
    ├── navigation_video.mp4
    └── execution_report.json
```

## Evaluation Metrics

The system calculates several metrics:

- **Success Rate (SR)**: Whether the agent reached within the success threshold distance
- **SPL (Success weighted by Path Length)**: Success rate weighted by path efficiency
- **Distance to Target**: Minimum distance to any target viewpoint
- **Object Category**: The target object category for this episode

## Success Criteria

Navigation is considered successful if the final agent position is within the `success_distance_threshold` (default: 0.25 meters) of any target viewpoint.

## Example Output

```json
{
    "evaluation_results": {
        "success": true,
        "min_distance_to_target": 0.18,
        "success_threshold": 0.25,
        "spl": 0.85,
        "final_position": [-0.467, 0.041, 4.078],
        "object_category": "tv"
    }
}
```

## Dependencies

- Python 3.7+
- Habitat-Sim
- Habitat-Lab
- All preprocessing dependencies (see `/preprocess/requirements.txt`)
- All video generation dependencies

## Troubleshooting

### Common Issues

1. **Episode not found**: Check that the episode ID exists in the JSON file
2. **Preprocessing fails**: Check LLM API configuration and prompts
3. **Video generation fails**: Check GPU settings and scene file paths
4. **Timeout errors**: Increase timeout values in the script for complex scenes

### Debug Mode

Run with `--verbose` flag to see detailed error messages and stack traces.

### Log Files

Check the following for detailed logs:
- Preprocessing outputs in `output/<scene_id>/preprocess/`
- Video generation reports in `output/<scene_id>/video/execution_report.json`

## Customization

### Adding New Episodes

1. Copy `eval_config_template.json`
2. Update episode information (JSON path, episode ID)
3. Update scene file path
4. Adjust other parameters as needed

### Modifying Success Criteria

Edit the `evaluation.success_distance_threshold` in your config file.

### Changing Video Quality

Modify the `video_generation.video` section in your config file.

## Integration with Batch Evaluation

This single-episode evaluator can be integrated into larger batch evaluation systems by:

1. Creating multiple config files for different episodes
2. Running evaluations in parallel or sequence
3. Aggregating results across episodes
4. Computing dataset-wide metrics (average SR, SPL, etc.)
