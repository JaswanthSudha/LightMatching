"""
Quick Test Script for Light Matching

This script provides an easy way to test the light matching system
with your own images without command line arguments.

Just modify the paths below and run: python quick_test.py
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from examples.test_real_images import process_images, analyze_scene_lighting
from src.light_matcher import LightMatcher
from utils.logger import setup_logger


def main():
    print("=== Quick Light Matching Test ===\n")
    
    # ======================================
    # MODIFY THESE PATHS FOR YOUR IMAGES
    # ======================================
    
    # Path to your scene image (the real photo)
    scene_image_path = "data/input/scenes/your_scene.jpg"
    
    # Path to your CG object image 
    cg_object_path = "data/input/cg_objects/your_object.png"
    
    # Output path for the result
    output_path = "data/output/quick_test_result.jpg"
    
    # ======================================
    # END OF CONFIGURATION
    # ======================================
    
    # Check if images exist
    if not os.path.exists(scene_image_path):
        print(f"❌ Scene image not found: {scene_image_path}")
        print("Please:")
        print("1. Place your scene image in: data/input/scenes/")
        print("2. Update the 'scene_image_path' in this script")
        print("3. Re-run this script")
        return
    
    if not os.path.exists(cg_object_path):
        print(f"❌ CG object image not found: {cg_object_path}")
        print("Please:")
        print("1. Place your CG object image in: data/input/cg_objects/")
        print("2. Update the 'cg_object_path' in this script")
        print("3. Re-run this script")
        return
    
    print(f"✓ Scene image: {scene_image_path}")
    print(f"✓ CG object: {cg_object_path}")
    print(f"✓ Output will be saved to: {output_path}")
    
    # Ask user what they want to do
    print("\\nWhat would you like to do?")
    print("1. Analyze scene lighting only (quick)")
    print("2. Full light matching process")
    print("3. Both (analyze first, then match)")
    
    choice = input("\\nEnter your choice (1, 2, or 3): ").strip()
    
    if choice in ['1', '3']:
        # Analyze scene lighting
        print("\\n" + "="*50)
        print("ANALYZING SCENE LIGHTING")
        print("="*50)
        
        try:
            matcher = LightMatcher()
            analysis = analyze_scene_lighting(matcher, scene_image_path)
            
            if analysis:
                print("\\n✅ Lighting analysis completed successfully!")
                
                if choice == '1':
                    print("\\nDone! You can now examine the lighting characteristics above.")
                    return
            else:
                print("\\n❌ Lighting analysis failed!")
                return
                
        except Exception as e:
            print(f"\\n❌ Error during analysis: {e}")
            return
    
    if choice in ['2', '3']:
        # Full light matching
        print("\\n" + "="*50)
        print("PERFORMING FULL LIGHT MATCHING")
        print("="*50)
        
        success = process_images(scene_image_path, cg_object_path, output_path)
        
        if success:
            print("\\n🎉 SUCCESS!")
            print(f"\\nYour result has been saved to: {output_path}")
            print("\\nNext steps:")
            print("1. Open the result image to see how it looks")
            print("2. Try different scene/object combinations")
            print("3. Experiment with the configuration settings")
        else:
            print("\\n❌ Light matching failed!")
            print("\\nTroubleshooting tips:")
            print("1. Check that your images are valid and readable")
            print("2. Try with simpler scenes first")
            print("3. Ensure your CG object has good contrast with background")
    
    elif choice not in ['1', '2', '3']:
        print("\\nInvalid choice. Please run the script again and choose 1, 2, or 3.")


if __name__ == "__main__":
    main()
