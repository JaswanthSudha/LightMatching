# Light Matching Project

An AI-powered system for matching lighting conditions between real scenes and computer-generated (CG) objects to achieve seamless integration using **pretrained computer vision models**.

## Project Overview

This project solves the common problem in visual effects and augmented reality where CG objects appear unnatural in real scenes due to lighting mismatches. Our solution leverages **existing pretrained models** (ResNet, VGG, DPT) combined with advanced computer graphics techniques to analyze lighting conditions and automatically adjust CG object lighting to match the scene.

### 🚀 **No Training Required!**
This system uses pretrained models from leading AI research, so you can get professional-quality results immediately without training custom neural networks.

## ✨ Enhanced Features

### **🧠 Pretrained AI Models**
- **ResNet18**: Deep feature extraction for advanced scene analysis
- **Intel DPT-Large**: Depth estimation for 3D-aware lighting
- **VGG19**: Style-based refinement for realistic appearance
- **Hybrid Processing**: Automatic fallback to traditional methods when needed

### **💡 Advanced Light Analysis**
- **Multi-Modal Light Estimation**: Combined CNN, depth, and traditional features
- **Accurate Color Temperature**: Professional color science (2000K-10000K range)
- **Directional Lighting**: Surface normal estimation with gradient analysis
- **Shadow Detection**: Intelligent shadow mapping and strength calculation
- **Ambient Light Separation**: Distinguishes ambient from directional lighting

### **🎨 Enhanced Relighting**
- **Multi-Stage Pipeline**: Color temp → Directional → Ambient → Shadows → Style
- **LAB Color Processing**: Perceptually accurate color adjustments
- **CLAHE Enhancement**: Adaptive contrast for better detail preservation
- **Real-time Processing**: Optimized for both single images and video sequences
- **Multiple CG Formats**: Support for PNG, JPG, and 3D model files

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

## 💻 Installation

### Quick Setup (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd light-matching-project

# Install core dependencies
pip install torch torchvision transformers
pip install opencv-python pillow numpy scikit-learn pyyaml

# Install optional enhancements
pip install timm  # For better segmentation
pip install accelerate datasets  # For faster model loading
```

### Full Installation
```bash
# Install all dependencies
pip install -r requirements.txt
```

### ⚡ First Run
The system will automatically download pretrained models (~2GB) on first use:
- ResNet18 (~45MB)
- Intel DPT-Large (~1.4GB) 
- VGG19 (~550MB)

## 🚀 Quick Start

### Method 1: Simple Script
```python
# Edit paths in quick_test.py and run:
python quick_test.py
```

### Method 2: Command Line
```bash
# Test with your own images
python examples/test_real_images.py --scene path/to/scene.jpg --cg path/to/object.png

# Analyze lighting only
python examples/test_real_images.py --scene scene.jpg --cg object.png --analyze-only
```

### Method 3: Python API
```python
from src.light_matcher import LightMatcher

# Initialize with pretrained models (default)
matcher = LightMatcher()

# Analyze scene lighting
analysis = matcher.get_lighting_analysis("scene.jpg")
print(f"Light direction: {analysis['lighting_direction']}")
print(f"Color temperature: {analysis['color_temperature']}K")

# Apply light matching
result = matcher.match_lighting(
    scene_image="data/input/scene.jpg",
    cg_object="data/input/object.png", 
    output_path="data/output/result.jpg"
)
```

## 🔬 Advanced Methodology

### **Stage 1: Enhanced Scene Analysis**
- **Multi-Modal Feature Extraction**: ResNet18 deep features + traditional CV
- **Depth-Aware Analysis**: Intel DPT-Large for 3D scene understanding
- **Shadow & Highlight Detection**: Intelligent region segmentation
- **Color Science**: Professional color temperature estimation

### **Stage 2: AI-Powered Light Estimation** 
- **CNN Feature Analysis**: Deep semantic understanding of lighting
- **Depth-Based Refinement**: 3D-aware light direction calculation
- **Hybrid Prediction**: Combines AI and physics-based methods
- **Robust Fallbacks**: Graceful degradation when models unavailable

### **Stage 3: Multi-Stage Relighting**
- **Color Temperature Adjustment**: LAB color space processing
- **Directional Lighting**: Surface normal estimation with gradients
- **Ambient Integration**: Separate ambient and direct lighting
- **Shadow Enhancement**: Physics-based shadow generation
- **Style Refinement**: VGG19-powered appearance matching
- **Detail Enhancement**: CLAHE adaptive contrast improvement

### **Stage 4: Intelligent Composition**
- **Alpha-Aware Blending**: Smart object boundary detection
- **Edge Smoothing**: Gaussian blur refinement
- **Color Matching**: Histogram-based color correction

## 🛠️ Technologies & Models

### **Pretrained AI Models**
- **ResNet18** (ImageNet): Deep feature extraction
- **Intel DPT-Large**: Monocular depth estimation
- **VGG19** (ImageNet): Style and texture analysis
- **Transformers Pipeline**: Automated model loading

### **Core Technologies**
- **PyTorch**: Deep learning framework with CUDA support
- **OpenCV**: Computer vision and image processing
- **scikit-learn**: Machine learning utilities
- **NumPy/SciPy**: Numerical computing
- **Pillow**: Advanced image handling

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
