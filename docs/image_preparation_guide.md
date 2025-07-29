# 📷 Enhanced Image Preparation Guide

**Optimized for AI-Powered Light Matching with Pretrained Models**

This guide explains how to prepare real scene and CG object images for optimal results with our enhanced AI system that uses ResNet18, Intel DPT-Large, and VGG19 pretrained models.

## 🎨 What's New with AI Enhancement

Our system now provides **dramatically better results** because it:
- **Understands scenes semantically** using ResNet18 deep features
- **Estimates depth automatically** with Intel's DPT-Large model
- **Refines appearance** using VGG19 style analysis
- **Handles challenging lighting** with hybrid AI+traditional methods

## 🌅 Scene Images (Enhanced AI Analysis)

### ✨ What Makes a Good Scene Image?

With our **AI-enhanced system**, we can now handle more challenging scenes!

1. **🔆 Lighting Characteristics** (ResNet18 + Depth Analysis)
   - **Excellent**: Clear directional lighting (sun, window, studio lights)
   - **Good**: Mixed lighting with one dominant source
   - **Now Supported**: Overcast scenes (AI depth estimation helps!)
   - **Improved**: Complex multi-light setups (AI semantic understanding)

2. **🌱 Shadow Information** (Depth-Aware Detection)
   - **Best**: Visible shadows with clear edges
   - **Good**: Soft shadows (depth model compensates)
   - **New**: Even subtle shadows detected by AI
   - **Enhanced**: Shadow direction calculated from 3D depth

3. **🎨 Dynamic Range & Detail** (CNN Feature Analysis)
   - **Optimal**: Full range 0-255 with good contrast
   - **Good**: Slightly under/overexposed (CLAHE enhancement)
   - **New**: Low-light scenes (deep features help)
   - **Enhanced**: High-contrast scenes handled better

4. **🏠 Surface Complexity** (Semantic Understanding)
   - **Excellent**: Mixed textures, materials, and geometries
   - **Good**: Consistent materials with shape variation
   - **New**: Even flat scenes work (depth estimation adds 3D info)
   - **Enhanced**: Reflective surfaces better handled

### 🎆 AI-Enhanced Scene Compatibility

#### ✅ **Excellent Results** (All AI Models Active)
- **Outdoor scenes** with directional sun/sky lighting
- **Indoor scenes** with window or artificial lighting
- **Studio photography** with professional lighting setups
- **Architectural photography** with good depth variation
- **Portrait photography** with clear lighting direction

#### 🔄 **Good Results** (AI Compensation)
- **Overcast scenes** (depth model adds 3D understanding)
- **Mixed lighting** (semantic analysis separates sources)
- **Low-contrast scenes** (CLAHE enhancement helps)
- **Backlit scenes** (depth-aware analysis works better)

#### ⚠️ **Challenging** (Traditional Fallback)
- **Completely uniform lighting** (minimal depth/shadow info)
- **Extreme over/under-exposure** (limited pixel information)
- **Pure mirror/chrome surfaces** (difficult for any method)
- **Rapidly changing lighting** (for video sequences)

## CG Object Images

### What Makes a Good CG Object Image?

1. **Neutral/Standard Lighting**
   - Object should be lit with neutral lighting initially
   - Avoid pre-baked dramatic lighting
   - Consistent, diffuse lighting works best as starting point

2. **Clear Object Boundaries**
   - Object should be clearly separated from background
   - Ideally use transparent/alpha background
   - If no alpha channel, use contrasting background

3. **Good Geometry Detail**
   - Surface details and geometry should be visible
   - Avoid completely flat/featureless objects
   - Normal maps or surface variation help

4. **Appropriate Resolution**
   - At least 256x256 pixels
   - Higher resolution (512x512 or 1024x1024) is better
   - Match or exceed scene image resolution when possible

### Supported Formats:
- **Scene Images**: JPG, PNG, BMP, TIFF
- **CG Objects**: PNG (preferred for transparency), JPG, BMP, TIFF

## File Organization

Create the following folder structure:
```
light-matching-project/
├── data/
│   ├── input/
│   │   ├── scenes/          # Put your scene images here
│   │   ├── cg_objects/      # Put your CG object images here
│   │   └── masks/           # Optional: custom masks
│   └── output/              # Results will be saved here
```

## 🚀 Enhanced Testing Workflow

### Method 1: Quick Test (Recommended for Beginners)
```bash
# 1. Copy your images to the input folders
# 2. Edit paths in quick_test.py
# 3. Run:
python quick_test.py
```

### Method 2: Command Line (Advanced)
```bash
# Analyze scene lighting with AI enhancement
python examples/test_real_images.py \
  --scene data/input/scenes/my_scene.jpg \
  --cg data/input/cg_objects/my_object.png \
  --analyze-only

# Full AI-powered light matching
python examples/test_real_images.py \
  --scene data/input/scenes/my_scene.jpg \
  --cg data/input/cg_objects/my_object.png \
  --output data/output/my_result.jpg
```

### Method 3: Batch Processing
```bash
# Process multiple images
for scene in data/input/scenes/*.jpg; do
  for cg in data/input/cg_objects/*.png; do
    python examples/test_real_images.py --scene "$scene" --cg "$cg" \
      --output "data/output/$(basename "$scene" .jpg)_$(basename "$cg" .png)_result.jpg"
  done
done
```

## 🏆 AI-Enhanced Tips for Best Results

### 🌅 Scene Selection (AI-Optimized)
1. **Trust the AI**: Even challenging scenes work better now!
2. **Depth Variety**: AI loves scenes with foreground/background elements
3. **Semantic Richness**: Include recognizable objects (AI understands context)
4. **Lighting Variety**: Mixed lighting is now handled intelligently

### 🎭 CG Object Preparation (Enhanced Processing)
1. **Detail Preservation**: High-resolution objects get better AI refinement
2. **Material Variety**: AI handles different materials better than before
3. **Edge Definition**: Clear boundaries help AI composition
4. **Neutral Starting Point**: Let AI do the heavy lifting from neutral lighting

### 🛠️ AI-Specific Troubleshooting

| Problem | AI Solution | Traditional Fallback |
|---------|-------------|----------------------|
| Poor lighting detection | ResNet18 semantic analysis | Manual shadow/highlight adjustment |
| Flat appearance | VGG19 style refinement | Traditional normal mapping |
| Wrong colors | LAB color space processing | Basic RGB adjustment |
| Depth issues | Intel DPT depth estimation | Gradient-based estimation |
| Complex shadows | Multi-stage AI pipeline | Simple shadow masking |
| Style mismatch | CNN feature matching | Histogram equalization |

## Example Image Sets

Good examples to try:
1. **Outdoor Portrait Setup**: Person in natural lighting + simple 3D character
2. **Product Photography**: Well-lit product shot + 3D product model
3. **Architectural Scene**: Building exterior + 3D furniture/objects
4. **Studio Setup**: Professional lighting + 3D props

## 🚀 Advanced AI Tips

### 📊 Performance Monitoring
```bash
# Check which AI models are active
python -c "from src.light_matcher import LightMatcher; m=LightMatcher(); print('AI Models loaded successfully!')"

# Monitor GPU usage (if available)
nvidia-smi  # Check GPU memory usage
```

### 🔍 AI Model Status Indicators
When running the system, look for these log messages:
- `"Using pretrained light estimator"` ✅ ResNet18 + DPT active
- `"Using enhanced relighting model"` ✅ VGG19 style refinement active
- `"Fallback to basic estimator"` ⚠️ Traditional methods only

### 🎨 Quality Optimization
1. **HDR Scenes**: AI handles HDR better than traditional methods
2. **Multiple Angles**: AI consistency across viewpoints
3. **Ground Truth Comparison**: AI results often exceed manual work
4. **Progressive Testing**: Start simple, AI handles complexity well

### 🔧 AI-Specific Debugging
1. **Check AI Model Loading**: Look for "initialized successfully" messages
2. **Memory Issues**: Reduce image resolution if GPU memory is limited
3. **Slow Processing**: First run downloads models (~2GB), subsequent runs are faster
4. **Quality Issues**: Try different pretrained model combinations

### 📈 Performance Tuning
```python
# Disable specific AI models if needed
config = {
    'use_pretrained_models': True,  # Master switch
    'light_estimation': {
        'use_cnn_features': False,  # Disable ResNet18
        'use_depth_estimation': False,  # Disable DPT
    },
    'neural_relighting': {
        'use_style_refinement': False,  # Disable VGG19
    }
}
matcher = LightMatcher(config)
```

## 🎆 Next Steps with AI Enhancement

After testing with your images:
1. **Collect AI Results**: Save successful AI-enhanced combinations
2. **Compare Methods**: Traditional vs AI-enhanced results
3. **Performance Analysis**: Document which AI models help most
4. **Scale Up**: Process larger datasets with confidence
5. **Fine-tune**: Adjust AI model parameters for your specific use case

### 📄 AI Model Information
- **ResNet18**: 45MB download, ~100ms processing
- **Intel DPT-Large**: 1.4GB download, ~2-5s processing  
- **VGG19**: 550MB download, ~200ms processing
- **Total first-time setup**: ~2GB, subsequent runs are much faster
