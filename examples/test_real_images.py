"""
Test script for real scene and CG images.

This script allows you to test the light matching system with your own
real scene images and CG object images.
"""

import sys
import os
import argparse
from pathlib import Path
import cv2
import numpy as np

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.light_matcher import LightMatcher
from utils.logger import setup_logger


def validate_image(image_path: str) -> bool:
    """Validate that an image file exists and can be loaded."""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return False
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Cannot load image: {image_path}")
            return False
        
        print(f"✓ Image loaded successfully: {image_path}")
        print(f"  - Dimensions: {image.shape[1]}x{image.shape[0]} pixels")
        print(f"  - Channels: {image.shape[2]}")
        return True
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return False


def analyze_scene_lighting(matcher: LightMatcher, scene_path: str):
    """Analyze and display lighting information from a scene."""
    print(f"\n=== Analyzing Scene Lighting ===")
    
    try:
        analysis = matcher.get_lighting_analysis(scene_path)
        
        print(f"Light Direction: [{analysis['lighting_direction'][0]:.3f}, "
              f"{analysis['lighting_direction'][1]:.3f}, "
              f"{analysis['lighting_direction'][2]:.3f}]")
        print(f"Light Intensity: {analysis['intensity']:.3f}")
        print(f"Color Temperature: {analysis['color_temperature']:.0f}K")
        print(f"Ambient Color (RGB): [{analysis['ambient_color'][0]:.3f}, "
              f"{analysis['ambient_color'][1]:.3f}, "
              f"{analysis['ambient_color'][2]:.3f}]")
        print(f"Shadow Strength: {analysis['shadow_strength']:.3f}")
        print(f"Has Environment Map: {analysis['has_environment_map']}")
        
        # Interpret the results
        print(f"\n=== Lighting Interpretation ===")
        
        # Light direction interpretation
        x, y, z = analysis['lighting_direction']
        if z > 0.7:
            print("• Light appears to be coming from above (overhead lighting)")
        elif z < -0.7:
            print("• Light appears to be coming from below (unusual)")
        else:
            print("• Light appears to be coming from the side")
        
        if abs(x) > abs(y):
            side = "right" if x > 0 else "left"
            print(f"• Light is predominantly from the {side} side")
        elif abs(y) > abs(x):
            side = "bottom" if y > 0 else "top"
            print(f"• Light is predominantly from the {side}")
        
        # Intensity interpretation
        if analysis['intensity'] > 0.8:
            print("• Scene is brightly lit")
        elif analysis['intensity'] > 0.5:
            print("• Scene has moderate lighting")
        else:
            print("• Scene is dimly lit")
        
        # Color temperature interpretation
        temp = analysis['color_temperature']
        if temp < 3000:
            print("• Warm lighting (tungsten/candlelight)")
        elif temp < 4000:
            print("• Warm white lighting")
        elif temp < 5000:
            print("• Neutral white lighting")
        elif temp < 6500:
            print("• Cool white lighting")
        else:
            print("• Daylight/cool lighting")
        
        # Shadow interpretation
        if analysis['shadow_strength'] > 0.7:
            print("• Strong shadows detected (harsh lighting)")
        elif analysis['shadow_strength'] > 0.3:
            print("• Moderate shadows detected")
        else:
            print("• Soft shadows or diffuse lighting")
        
        return analysis
    
    except Exception as e:
        print(f"Error analyzing scene lighting: {e}")
        return None


def process_images(scene_path: str, cg_path: str, output_path: str, config_path: str = None):
    """Process real scene and CG images."""
    
    # Validate input images
    print("=== Validating Input Images ===")
    if not validate_image(scene_path):
        return False
    if not validate_image(cg_path):
        return False
    
    # Set up logging
    logger = setup_logger('real_image_test', level='INFO')
    
    try:
        # Initialize the light matcher
        print(f"\n=== Initializing Light Matcher ===")
        if config_path and os.path.exists(config_path):
            matcher = LightMatcher(config_path)
            print(f"Using configuration file: {config_path}")
        else:
            matcher = LightMatcher()
            print("Using default configuration")
        
        # Analyze scene lighting
        lighting_analysis = analyze_scene_lighting(matcher, scene_path)
        if lighting_analysis is None:
            return False
        
        # Apply light matching
        print(f"\n=== Applying Light Matching ===")
        print("This may take a few moments...")
        
        result = matcher.match_lighting(
            scene_image=scene_path,
            cg_object=cg_path,
            output_path=output_path
        )
        
        if result['success']:
            print(f"\n✓ Light matching completed successfully!")
            print(f"✓ Result saved to: {output_path}")
            
            # Display some statistics about the result
            result_image = cv2.imread(output_path)
            if result_image is not None:
                print(f"✓ Output image dimensions: {result_image.shape[1]}x{result_image.shape[0]}")
            
            print(f"\nYou can now view the result image to see how well the CG object")
            print(f"has been integrated with the scene lighting.")
            
            return True
        else:
            print(f"\n✗ Light matching failed!")
            print(f"Error: {result['message']}")
            return False
    
    except Exception as e:
        print(f"\n✗ Processing failed with error: {e}")
        logger.error(f"Processing failed: {e}")
        return False


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Test light matching with real scene and CG images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python test_real_images.py --scene my_scene.jpg --cg my_object.png
  
  # Specify output path
  python test_real_images.py --scene scene.jpg --cg object.png --output result.jpg
  
  # Use custom configuration
  python test_real_images.py --scene scene.jpg --cg object.png --config custom_config.yaml
        """
    )
    
    parser.add_argument('--scene', '-s', required=True,
                        help='Path to the scene image')
    parser.add_argument('--cg', '-c', required=True,
                        help='Path to the CG object image')
    parser.add_argument('--output', '-o', 
                        default='data/output/real_test_result.jpg',
                        help='Output path for the result image')
    parser.add_argument('--config', 
                        help='Path to configuration file (optional)')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only analyze scene lighting, don\'t perform matching')
    
    args = parser.parse_args()
    
    # Convert paths to absolute paths
    scene_path = os.path.abspath(args.scene)
    cg_path = os.path.abspath(args.cg)
    output_path = os.path.abspath(args.output)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Real Image Light Matching Test ===")
    print(f"Scene image: {scene_path}")
    print(f"CG object image: {cg_path}")
    print(f"Output path: {output_path}")
    
    if args.analyze_only:
        # Only analyze scene lighting
        logger = setup_logger('lighting_analysis', level='INFO')
        matcher = LightMatcher(args.config if args.config else None)
        
        if validate_image(scene_path):
            analyze_scene_lighting(matcher, scene_path)
    else:
        # Full processing
        success = process_images(scene_path, cg_path, output_path, args.config)
        
        if success:
            print(f"\n🎉 Processing completed successfully!")
            print(f"Check the result at: {output_path}")
        else:
            print(f"\n❌ Processing failed. Check the logs for details.")
            sys.exit(1)


if __name__ == "__main__":
    main()
