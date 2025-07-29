# Image Preparation Guide for Light Matching

This guide explains how to prepare real scene and CG object images for optimal light matching results.

## Scene Images

### What Makes a Good Scene Image?

1. **Clear Lighting Direction**
   - The lighting should have a dominant direction (not completely diffuse)
   - Avoid scenes with multiple competing light sources
   - Natural outdoor scenes or well-lit indoor scenes work best

2. **Visible Shadows**
   - Shadows help the system determine light direction
   - Hard shadows work better than very soft shadows
   - Avoid completely shadowless scenes

3. **Good Dynamic Range**
   - Not too dark or overexposed
   - Should have both bright and dark areas
   - Histogram should span most of the 0-255 range

4. **Surface Variety**
   - Mix of matte and slightly reflective surfaces
   - Different textures help with analysis
   - Avoid scenes with only highly reflective surfaces

### Recommended Scene Types:
- ✅ Outdoor scenes with clear sky/sun direction
- ✅ Indoor scenes with window lighting
- ✅ Product photography setups
- ✅ Architectural photography
- ❌ Completely overcast/flat lighting
- ❌ Multiple conflicting light sources
- ❌ Very dark or underexposed images
- ❌ Highly reflective/mirror-like surfaces only

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

## Testing Workflow

### Step 1: Analyze Scene Lighting Only
```bash
python examples/test_real_images.py --scene data/input/scenes/my_scene.jpg --cg data/input/cg_objects/my_object.png --analyze-only
```

### Step 2: Full Light Matching
```bash
python examples/test_real_images.py --scene data/input/scenes/my_scene.jpg --cg data/input/cg_objects/my_object.png --output data/output/my_result.jpg
```

## Tips for Best Results

### Scene Selection Tips:
1. **Start Simple**: Begin with outdoor scenes in clear weather
2. **Single Light Source**: Avoid complex multi-light setups initially
3. **Good Contrast**: Ensure the scene has both bright and dark areas
4. **Reference Objects**: Include objects with known surface properties

### CG Object Tips:
1. **Start with Simple Shapes**: Spheres, cubes, simple models work well
2. **Matte Materials**: Start with non-reflective materials
3. **Good UVs**: Ensure proper texture mapping
4. **Clean Geometry**: Avoid artifacts or topology issues

### Common Issues and Solutions:

| Problem | Solution |
|---------|----------|
| Poor lighting detection | Try scenes with stronger shadows |
| Object doesn't blend well | Ensure CG object has proper alpha channel |
| Colors look wrong | Check scene's color temperature |
| Shadows don't match | Use scenes with clearer directional lighting |
| Object looks flat | Ensure CG object has surface detail/normals |

## Example Image Sets

Good examples to try:
1. **Outdoor Portrait Setup**: Person in natural lighting + simple 3D character
2. **Product Photography**: Well-lit product shot + 3D product model
3. **Architectural Scene**: Building exterior + 3D furniture/objects
4. **Studio Setup**: Professional lighting + 3D props

## Advanced Tips

### For Better Results:
1. **HDR Images**: Use HDR scene images if available
2. **Multiple Views**: Test same scene from different angles
3. **Ground Truth**: Compare with manually lit versions
4. **Iterative Testing**: Start simple, gradually increase complexity

### Debugging Poor Results:
1. Check the lighting analysis output first
2. Verify image dimensions and formats
3. Try different CG objects with the same scene
4. Experiment with different scenes with the same object

## Next Steps

After testing with your images:
1. **Collect Results**: Save successful combinations
2. **Document Findings**: Note what works and what doesn't
3. **Improve Training Data**: Use good results for model training
4. **Iterate**: Gradually test more complex scenarios
