<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Habitat Lab Coordinate Navigation Project

This project implements a coordinate-based navigation system using Habitat Lab and Matterport3D dataset.

## Key Components:
- **Coordinate System**: 2D map coordinates with automatic 3D height adaptation
- **Visualization**: First-person, third-person, and topdown map views
- **Navigation**: Coordinate-based movement with view angle adjustment
- **Output**: All images saved to workspace/images/ directory

## Code Guidelines:
- Use numpy arrays for coordinate transformations
- Implement proper quaternion handling for rotations
- Ensure thread-safe image saving operations
- Follow Habitat Lab best practices for simulator management
- Use matplotlib for visualization with consistent styling

## Dependencies:
- habitat-sim
- habitat-lab
- numpy
- matplotlib
- opencv-python
- quaternion (for rotation handling)
