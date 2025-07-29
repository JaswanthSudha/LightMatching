"""
Image preprocessing utilities for the light matching pipeline.
"""

import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional
import logging


class ImageProcessor:
    """
    Handles preprocessing of input images for the light matching pipeline.
    
    Includes resizing, normalization, format conversion, and augmentation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the image processor.
        
        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Processing parameters
        self.resize_method = config.get('resize_method', 'bilinear')
        self.normalize = config.get('normalize', True)
        self.augmentation = config.get('augmentation', False)
    
    def process(self, image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Apply preprocessing to an input image.
        
        Args:
            image: Input image in BGR format
            target_size: Target size (width, height). If None, keeps original size
            
        Returns:
            Processed image
        """
        processed = image.copy()
        
        # Resize if target size specified
        if target_size:
            processed = self._resize_image(processed, target_size)
        
        # Normalize if enabled
        if self.normalize:
            processed = self._normalize_image(processed)
        
        # Apply augmentation if enabled (typically for training)
        if self.augmentation:
            processed = self._apply_augmentation(processed)
        
        return processed
    
    def _resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image to target size."""
        width, height = target_size
        
        if self.resize_method == 'bilinear':
            return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        elif self.resize_method == 'bicubic':
            return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
        elif self.resize_method == 'nearest':
            return cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
        else:
            self.logger.warning(f"Unknown resize method: {self.resize_method}. Using bilinear.")
            return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        return image.astype(np.float32) / 255.0
    
    def _apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply random augmentations for training."""
        # Simple augmentations - can be expanded
        if np.random.random() > 0.5:
            # Random brightness adjustment
            brightness_factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)
        
        if np.random.random() > 0.5:
            # Random horizontal flip
            image = cv2.flip(image, 1)
        
        return image
