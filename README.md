# Light Matching Project

An AI-powered system for matching lighting conditions between real scenes and computer-generated (CG) objects to achieve seamless integration.

## Project Overview

This project aims to solve the common problem in visual effects and augmented reality where CG objects appear unnatural in real scenes due to lighting mismatches. Our solution uses deep learning techniques to analyze lighting conditions in real images/sequences and automatically adjust CG object lighting to match the scene.

## Features

- **Light Estimation**: Analyze lighting direction, intensity, and color temperature from real images
- **HDR Environment Map Generation**: Create environment maps for realistic lighting
- **Neural Relighting**: Use AI models to relight CG objects based on scene analysis
- **Real-time Processing**: Support for both single images and video sequences
- **Multiple CG Formats**: Support for various 3D object formats

## Project Structure

```
light-matching-project/
├── src/                    # Source code
│   ├── light_estimation/   # Light analysis modules
│   ├── neural_relighting/  # AI relighting models
│   ├── preprocessing/      # Image/video preprocessing
│   └── postprocessing/     # Output enhancement
├── models/                 # AI models
│   ├── pretrained/        # Pre-trained model weights
│   └── checkpoints/       # Training checkpoints
├── data/                  # Data directory
│   ├── input/            # Input images/videos
│   ├── output/           # Processed results
│   └── training/         # Training datasets
├── utils/                # Utility functions
├── tests/               # Unit tests
├── examples/            # Example usage scripts
└── docs/               # Documentation
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd light-matching-project

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from src.light_matcher import LightMatcher

# Initialize the light matcher
matcher = LightMatcher()

# Process an image with CG object
result = matcher.match_lighting(
    scene_image="data/input/scene.jpg",
    cg_object="data/input/object.obj",
    output_path="data/output/result.jpg"
)
```

## Methodology

1. **Scene Analysis**: Extract lighting information from the background scene
2. **Light Estimation**: Estimate light direction, intensity, and color properties
3. **Environment Mapping**: Generate HDR environment maps
4. **Neural Relighting**: Apply AI-based relighting to CG objects
5. **Composition**: Blend the relit CG object with the original scene

## Technologies Used

- **Deep Learning**: PyTorch/TensorFlow for neural networks
- **Computer Vision**: OpenCV for image processing
- **3D Graphics**: Open3D/Blender for 3D object handling
- **HDR Processing**: Custom HDR environment map generation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Research papers on inverse rendering and light estimation
- Open source computer vision and graphics libraries
- Deep learning frameworks and pretrained models
