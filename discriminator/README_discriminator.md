# Discriminator System for ObjectNav Model Comparison

This system provides a comprehensive framework for comparing two ObjectNav models by identifying controversial episodes where they disagree and using an LLM discriminator to make final judgments.

## Overview

The system consists of several components:

1. **Controversy Extraction** (`extract_controversial.py`) - Identifies episodes where models disagree
2. **Discriminator System** (`discriminator_system.py`) - Uses LLM to evaluate controversial episodes  
3. **Final Evaluation** (`final_evaluation.py`) - Combines original and discriminated results
4. **Pipeline Runner** (`run_discriminator_pipeline.py`) - Orchestrates the entire process

## Prerequisites

- Python 3.8+
- OpenCV (`pip install opencv-python`)
- PIL/Pillow (`pip install Pillow`)
- Pandas (`pip install pandas`)
- Numpy
- Requests (`pip install requests`)
- Habitat-Sim and related dependencies

## Quick Start

### Option 1: Run Complete Pipeline

```bash
python run_discriminator_pipeline.py \
    --model1_dir /home/yaoaa/habitat-lab/habitat_video_project/eval/output_challenge \
    --model2_dir /home/yaoaa/habitat-lab/habitat_video_project/eval_model/output \
    --config discriminator_config.json
```

### Option 2: Run Steps Individually

1. **Extract controversial episodes:**
```bash
python extract_controversial.py \
    --model1_dir /home/yaoaa/habitat-lab/habitat_video_project/eval/output_challenge \
    --model2_dir /home/yaoaa/habitat-lab/habitat_video_project/eval_model/output \
    --output_dir controversy_analysis
```

2. **Run discrimination:**
```bash
python discriminator_system.py \
    --model1_dir /home/yaoaa/habitat-lab/habitat_video_project/eval/output_challenge \
    --model2_dir /home/yaoaa/habitat-lab/habitat_video_project/eval_model/output \
    --config discriminator_config.json \
    --output_dir discrimination_results
```

3. **Generate final evaluation:**
```bash
python final_evaluation.py \
    --model1_dir /home/yaoaa/habitat-lab/habitat_video_project/eval/output_challenge \
    --model2_dir /home/yaoaa/habitat-lab/habitat_video_project/eval_model/output \
    --discrimination_results discrimination_results/discrimination_results.json \
    --output final_evaluation_results.json
```

## Configuration

Edit `discriminator_config.json` to customize:

- **LLM Settings**: API key, model, base URL
- **Rendering**: Resolution, coordinate display
- **Processing**: Batch sizes, confidence analysis

## Expected Input Format

The system expects model output directories with this structure:

```
model_output/
├── batch_output.json              # Overall results
├── scene_id1/
│   ├── episode_id1/
│   │   ├── preprocess/
│   │   │   ├── action.json         # Agent state and target info
│   │   │   ├── metadata.json       # Scene metadata
│   │   │   └── navigation_nodes.json  # Navigation nodes (optional)
│   │   └── output.json            # Episode details
│   └── episode_id2/...
└── scene_id2/...
```

### Required File Formats

**batch_output.json:**
```json
{
  "episode_details": {
    "scene_id/episode_id": {
      "sr": true/false,
      "spl": 0.123,
      "success": true/false,
      "geodesic_distance_to_target": 1.23,
      "path_length": 5.67,
      "object_category": "bed"
    }
  }
}
```

**action.json:**
```json
{
  "agent_state": {
    "position": [x, y, z],
    "rotation": [x, y, z, w]
  },
  "target_info": {
    "coordinate": [pixel_x, pixel_y],
    "name": "bed"
  }
}
```

**metadata.json:**
```json
{
  "scene_info": {
    "scene_path": "/path/to/scene.glb"
  },
  "topdown_metadata": {
    "origin_in_pixels": [x, y],
    "spacing_in_meters_per_pixel": 0.123
  }
}
```

## Output

### Controversy Analysis
- `controversial_episodes.json` - List of episodes where models disagree
- `controversial_episodes.csv` - CSV format for easy analysis
- `controversy_statistics.json` - Summary statistics

### Discrimination Results  
- `discrimination_results.json` - LLM decisions for each controversial episode
- `scene_episode_topdown_annotated.png` - Annotated topdown views
- `discriminator.log` - Processing logs

### Final Evaluation
- `final_evaluation_results.json` - Combined results with overall metrics

## Key Features

1. **Automatic Floor Detection** - Uses agent position to determine correct floor for topdown rendering
2. **Visual Annotation** - Annotates topdown views with navigation nodes from both models
3. **LLM Discrimination** - Uses multimodal LLM to evaluate controversial episodes
4. **Comprehensive Statistics** - Provides detailed analysis of model performance and controversies
5. **Robust Error Handling** - Continues processing even if individual episodes fail

## Example Output

```
=== Discrimination Results ===
Total controversial episodes: 45
Model 1 wins: 28 (62.2%)
Model 2 wins: 17 (37.8%)
No decision: 0

=== Final Evaluation Summary ===
Total Episodes: 532
Final Success Rate (SR): 0.573
Final SPL: 0.322
Controversial Episodes: 45
Controversy Rate: 0.085
```

## Troubleshooting

1. **Scene file not found**: Ensure scene paths in metadata.json are correct
2. **LLM API errors**: Check API key and rate limits
3. **Import errors**: Ensure current_topdown.py is in discriminator/ subdirectory
4. **Memory issues**: Reduce rendering resolution in config

## Customization

The system is modular and can be extended:

- Add new discrimination criteria
- Implement different LLM providers  
- Customize visualization annotations
- Add new evaluation metrics

## Dependencies

Make sure these files are available:
- `discriminator/current_topdown.py` - Topdown rendering functionality
- Scene files (.glb) at paths specified in metadata.json
- Valid LLM API access

## Support

For issues or questions, check the logs in the output directory or review the error messages in the terminal output.
