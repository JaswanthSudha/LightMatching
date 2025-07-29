"""
Compositor for blending CG objects into scene images
"""

import numpy as np
import cv2
from typing import Optional
import logging


class Compositor:
    """
    Compositor for blending computer-generated objects into real scenes.
    
    Supports alpha blending, edge smoothing, and color matching.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the compositor.
        
        Args:
            config: Configuration dictionary with post-processing parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Compositing parameters
        self.method = config.get('composition_method', 'alpha_blend')
        self.edge_smoothing = config.get('edge_smoothing', True)
        self.color_matching = config.get('color_matching', True)
    
    def composite(
        self,
        scene_image: np.ndarray,
        cg_object: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Blend the CG object into the scene image.
        
        Args:
            scene_image: Background scene image
            cg_object: Foreground CG object with alpha channel
            mask: Optional mask for object placement
            
        Returns:
            Composited result
        """
        # Ensure CG object matches scene dimensions
        if cg_object.shape[:2] != scene_image.shape[:2]:
            cg_object = cv2.resize(cg_object, (scene_image.shape[1], scene_image.shape[0]))
        
        result = scene_image.copy()
        
        if mask is None:
            # Check if CG object has alpha channel
            if cg_object.shape[2] == 4:
                alpha = cg_object[:, :, 3] / 255.0  # Use alpha channel
            else:
                # Create a simple mask based on non-black pixels
                gray_cg = cv2.cvtColor(cg_object, cv2.COLOR_BGR2GRAY)
                alpha = (gray_cg > 10).astype(np.float32)  # Simple threshold
        else:
            alpha = mask.astype(np.float32) / 255.0
        
        # Blend CG object into the scene
        for c in range(3):  # For each color channel
            result[:, :, c] = (alpha * cg_object[:, :, c] + (1 - alpha) * result[:, :, c])
        
        # Apply edge smoothing if enabled
        if self.edge_smoothing:
            result = self._smooth_edges(result, alpha)
        
        # Apply color matching if enabled
        if self.color_matching:
            result = self._match_colors(result, scene_image)
        
        return result
    
    def _smooth_edges(self, image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Apply edge smoothing using Gaussian blur."""
        blurred_alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        for c in range(3):
            image[:, :, c] = blurred_alpha * image[:, :, c] + (1 - blurred_alpha) * image[:, :, c]
        return image
    
    def _match_colors(self, composite: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Apply color matching to make the composite look more natural."""
        composite_lab = cv2.cvtColor(composite, cv2.COLOR_BGR2LAB)
        original_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
        
        # Match histograms of the LAB channels
        for i in range(3):
            composite_lab[:, :, i] = cv2.equalizeHist(composite_lab[:, :, i])
            composite_lab[:, :, i] = cv2.addWeighted(composite_lab[:, :, i], 0.5, original_lab[:, :, i], 0.5, 0)
        
        return cv2.cvtColor(composite_lab, cv2.COLOR_LAB2BGR)
