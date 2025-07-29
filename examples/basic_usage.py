"""
Basic usage example for the Light Matching Project.

This script demonstrates how to use the light matching system
to analyze lighting in a scene and apply it to a CG object.
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.light_matcher import LightMatcher
from utils.logger import setup_logger


def create_sample_images():
    """Create sample images for demonstration if none exist."""
    data_dir = project_root / "data" / "input"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple scene image
    scene_image = np.zeros((512, 512, 3), dtype=np.uint8)
    # Add a gradient to simulate lighting
    for i in range(512):
        for j in range(512):
            intensity = int(128 + 64 * np.sin(i * 0.01) * np.cos(j * 0.01))
            scene_image[i, j] = [intensity * 0.8, intensity * 0.9, intensity]  # Blueish tint
    
    # Add some "objects" with shadows
    cv2.rectangle(scene_image, (100, 100), (200, 300), (80, 80, 80), -1)  # Dark rectangle
    cv2.circle(scene_image, (350, 250), 50, (200, 200, 180), -1)  # Bright circle
    
    cv2.imwrite(str(data_dir / "sample_scene.jpg"), scene_image)
    
    # Create a simple CG object image
    cg_object = np.ones((256, 256, 3), dtype=np.uint8) * 128  # Gray object
    # Add some basic shading
    center = (128, 128)
    for i in range(256):
        for j in range(256):
            distance = np.sqrt((i - center[1])**2 + (j - center[0])**2)
            if distance < 100:
                intensity = int(180 - distance * 0.5)
                cg_object[i, j] = [intensity, intensity, intensity]
    
    cv2.imwrite(str(data_dir / "sample_cg_object.jpg"), cg_object)
    
    return str(data_dir / "sample_scene.jpg"), str(data_dir / "sample_cg_object.jpg")


def main():
    """Main demonstration function."""
    # Set up logging
    logger = setup_logger('light_matching_demo', level='INFO')
    logger.info("Starting Light Matching Demo")
    
    try:
        # Create sample images if they don't exist
        scene_path, cg_path = create_sample_images()
        logger.info(f"Using scene image: {scene_path}")
        logger.info(f"Using CG object: {cg_path}")
        
        # Initialize the light matcher
        logger.info("Initializing Light Matcher...")
        matcher = LightMatcher()
        
        # Analyze lighting in the scene
        logger.info("Analyzing scene lighting...")
        lighting_analysis = matcher.get_lighting_analysis(scene_path)
        
        print("\n=== Lighting Analysis Results ===")
        print(f"Light Direction: {lighting_analysis['lighting_direction']}")
        print(f"Intensity: {lighting_analysis['intensity']:.3f}")
        print(f"Color Temperature: {lighting_analysis['color_temperature']:.0f}K")
        print(f"Ambient Color: {lighting_analysis['ambient_color']}")
        print(f"Shadow Strength: {lighting_analysis['shadow_strength']:.3f}")
        print(f"Has Environment Map: {lighting_analysis['has_environment_map']}")
        
        # Apply light matching
        logger.info("Applying light matching...")
        output_path = str(project_root / "data" / "output" / "matched_result.jpg")
        
        result = matcher.match_lighting(
            scene_image=scene_path,
            cg_object=cg_path,
            output_path=output_path
        )
        
        if result['success']:
            print(f"\n=== Light Matching Completed Successfully ===")
            print(f"Result saved to: {output_path}")
            print("You can view the result image to see the lighting-matched CG object.")
        else:
            print(f"\n=== Light Matching Failed ===")
            print(f"Error: {result['message']}")
        
        logger.info("Demo completed")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        print(f"Error running demo: {e}")
        print("This is expected since we haven't installed all dependencies yet.")
        print("Please install the required packages using: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
