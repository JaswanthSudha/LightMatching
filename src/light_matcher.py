"""
Light Matcher - Main class for AI-based lighting matching between real scenes and CG objects
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Union, Optional, Dict, Any, Tuple
import logging

from .data_structures import LightingParameters
from .light_estimation.scene_analyzer import SceneAnalyzer
from .light_estimation.light_estimator import LightEstimator
from .light_estimation.pretrained_light_estimator import PretrainedLightEstimator
from .neural_relighting.relighting_model import RelightingModel
from .neural_relighting.pretrained_relighting import HybridRelightingModel
from .preprocessing.image_processor import ImageProcessor
from .postprocessing.compositor import Compositor
from utils.config import Config
from utils.logger import setup_logger


class LightMatcher:
    """
    Main class for AI-powered light matching between real scenes and CG objects.
    
    This class orchestrates the entire pipeline:
    1. Scene analysis and light estimation
    2. Neural relighting of CG objects
    3. Composition and final output generation
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the LightMatcher with configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.logger = setup_logger(__name__)
        self.config = Config(config_path) if config_path else Config()
        
        # Initialize components
        self.scene_analyzer = SceneAnalyzer(self.config.scene_analysis)
        
        # Use pretrained models if available, fallback to basic estimator
        use_pretrained = self.config.get('use_pretrained_models', True)
        if use_pretrained:
            try:
                self.light_estimator = PretrainedLightEstimator(self.config.light_estimation)
                self.logger.info("Using pretrained light estimator")
            except Exception as e:
                self.logger.warning(f"Failed to initialize pretrained estimator: {e}")
                self.light_estimator = LightEstimator(self.config.light_estimation)
                self.logger.info("Fallback to basic light estimator")
        else:
            self.light_estimator = LightEstimator(self.config.light_estimation)
        
        # Use enhanced relighting if available
        if use_pretrained:
            try:
                self.relighting_model = HybridRelightingModel(self.config.neural_relighting)
                self.logger.info("Using enhanced relighting model")
            except Exception as e:
                self.logger.warning(f"Failed to initialize enhanced relighting: {e}")
                self.relighting_model = RelightingModel(self.config.neural_relighting)
                self.logger.info("Fallback to basic relighting model")
        else:
            self.relighting_model = RelightingModel(self.config.neural_relighting)
        self.image_processor = ImageProcessor(self.config.preprocessing)
        self.compositor = Compositor(self.config.postprocessing)
        
        self.logger.info("LightMatcher initialized successfully")
    
    def match_lighting(
        self,
        scene_image: Union[str, np.ndarray],
        cg_object: Union[str, np.ndarray],
        output_path: Optional[str] = None,
        mask: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main method to match lighting between scene and CG object.
        
        Args:
            scene_image: Path to scene image or numpy array
            cg_object: Path to CG object file or rendered image array
            output_path: Path to save the final result
            mask: Optional mask for CG object placement
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing results and metadata
        """
        try:
            self.logger.info("Starting light matching process")
            
            # Step 1: Load and preprocess inputs
            scene_img = self._load_image(scene_image)
            cg_img = self._load_cg_object(cg_object)
            
            self.logger.info("Images loaded successfully")
            
            # Step 2: Analyze scene for lighting conditions
            lighting_params = self._analyze_scene_lighting(scene_img)
            self.logger.info("Scene lighting analysis completed")
            
            # Step 3: Apply neural relighting to CG object
            relit_cg = self._relight_cg_object(cg_img, lighting_params)
            self.logger.info("CG object relighting completed")
            
            # Step 4: Composite the relit CG object into the scene
            final_result = self._composite_result(scene_img, relit_cg, mask)
            self.logger.info("Composition completed")
            
            # Step 5: Post-process and save result
            if output_path:
                self._save_result(final_result, output_path)
                self.logger.info(f"Result saved to {output_path}")
            
            return {
                'result': final_result,
                'lighting_params': lighting_params,
                'success': True,
                'message': 'Light matching completed successfully'
            }
            
        except Exception as e:
            self.logger.error(f"Error in light matching: {str(e)}")
            return {
                'result': None,
                'lighting_params': None,
                'success': False,
                'message': f'Error: {str(e)}'
            }
    
    def _load_image(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """Load image from path or return if already numpy array."""
        if isinstance(image, str):
            return cv2.imread(image, cv2.IMREAD_COLOR)
        return image
    
    def _load_cg_object(self, cg_object: Union[str, np.ndarray]) -> np.ndarray:
        """Load CG object - could be 3D model file or rendered image."""
        if isinstance(cg_object, str):
            # If it's a 3D model file, render it first
            if cg_object.endswith(('.obj', '.ply', '.fbx', '.blend')):
                return self._render_3d_object(cg_object)
            else:
                # Assume it's an image file
                return cv2.imread(cg_object, cv2.IMREAD_COLOR)
        return cg_object
    
    def _render_3d_object(self, model_path: str) -> np.ndarray:
        """Render 3D object to image for processing."""
        # This would use a 3D rendering engine like Open3D or Blender
        # For now, return a placeholder
        self.logger.warning("3D object rendering not fully implemented yet")
        return np.zeros((512, 512, 3), dtype=np.uint8)
    
    def _analyze_scene_lighting(self, scene_img: np.ndarray) -> LightingParameters:
        """Analyze the scene to extract lighting parameters."""
        # Use scene analyzer to extract features
        scene_features = self.scene_analyzer.analyze(scene_img)
        
        # Use light estimator to predict lighting parameters
        lighting_estimate = self.light_estimator.estimate(scene_img, scene_features)
        
        return LightingParameters(
            direction=lighting_estimate['direction'],
            intensity=lighting_estimate['intensity'],
            color_temp=lighting_estimate['color_temp'],
            ambient=lighting_estimate['ambient'],
            shadows=lighting_estimate['shadows'],
            environment_map=lighting_estimate.get('env_map')
        )
    
    def _relight_cg_object(
        self, 
        cg_img: np.ndarray, 
        lighting_params: LightingParameters
    ) -> np.ndarray:
        """Apply neural relighting to match the scene lighting."""
        return self.relighting_model.relight(cg_img, lighting_params)
    
    def _composite_result(
        self, 
        scene_img: np.ndarray, 
        relit_cg: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Composite the relit CG object into the scene."""
        return self.compositor.composite(scene_img, relit_cg, mask)
    
    def _save_result(self, result: np.ndarray, output_path: str) -> None:
        """Save the final result to disk."""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, result)
    
    def process_video_sequence(
        self,
        video_path: str,
        cg_object: Union[str, np.ndarray],
        output_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a video sequence for temporal consistency.
        
        Args:
            video_path: Path to input video
            cg_object: CG object to insert
            output_path: Path for output video
            **kwargs: Additional parameters
            
        Returns:
            Processing results and metadata
        """
        self.logger.info("Video sequence processing not fully implemented yet")
        return {
            'success': False,
            'message': 'Video processing will be implemented in future versions'
        }
    
    def get_lighting_analysis(self, scene_image: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Get detailed lighting analysis of a scene without performing relighting.
        
        Args:
            scene_image: Input scene image
            
        Returns:
            Detailed lighting analysis results
        """
        scene_img = self._load_image(scene_image)
        lighting_params = self._analyze_scene_lighting(scene_img)
        
        return {
            'lighting_direction': lighting_params.direction.tolist(),
            'intensity': float(lighting_params.intensity),
            'color_temperature': float(lighting_params.color_temp),
            'ambient_color': lighting_params.ambient.tolist(),
            'shadow_strength': float(lighting_params.shadows),
            'has_environment_map': lighting_params.environment_map is not None
        }


if __name__ == "__main__":
    # Example usage
    matcher = LightMatcher()
    
    # Analyze lighting in a scene
    analysis = matcher.get_lighting_analysis("data/input/scene.jpg")
    print("Lighting analysis:", analysis)
    
    # Match lighting between scene and CG object
    result = matcher.match_lighting(
        scene_image="data/input/scene.jpg",
        cg_object="data/input/cg_object.jpg",
        output_path="data/output/result.jpg"
    )
    
    print("Light matching result:", result['message'])
