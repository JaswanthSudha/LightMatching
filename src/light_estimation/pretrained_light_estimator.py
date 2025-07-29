"""
Enhanced Light Estimator using pretrained computer vision models.

This module leverages existing pretrained models to improve lighting estimation
without requiring custom training.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import logging
from pathlib import Path

try:
    import torchvision.transforms as transforms
    import torchvision.models as models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class PretrainedLightEstimator:
    """
    Light estimator using pretrained computer vision models.
    
    Uses models like ResNet, VGG, and depth estimation models to extract
    better lighting features without custom training.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with pretrained models."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize pretrained models
        self._init_feature_extractor()
        self._init_depth_estimator()
        self._init_segmentation_model()
        
    def _init_feature_extractor(self):
        """Initialize pretrained CNN for feature extraction."""
        if not TORCHVISION_AVAILABLE:
            self.logger.warning("Torchvision not available, using basic features")
            self.feature_extractor = None
            return
        
        try:
            # Use pretrained ResNet18 for feature extraction
            self.feature_extractor = models.resnet18(pretrained=True)
            # Remove the final classification layer
            self.feature_extractor = torch.nn.Sequential(
                *list(self.feature_extractor.children())[:-1]
            )
            self.feature_extractor.eval()
            self.feature_extractor.to(self.device)
            
            # Image preprocessing for ResNet
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            self.logger.info("ResNet18 feature extractor initialized")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize feature extractor: {e}")
            self.feature_extractor = None
    
    def _init_depth_estimator(self):
        """Initialize depth estimation model."""
        if not TRANSFORMERS_AVAILABLE:
            self.logger.warning("Transformers not available, skipping depth estimation")
            self.depth_estimator = None
            return
        
        try:
            # Use Intel's MiDaS model for depth estimation
            self.depth_estimator = pipeline(
                "depth-estimation",
                model="Intel/dpt-large",
                device=0 if torch.cuda.is_available() else -1
            )
            self.logger.info("Depth estimation model initialized")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize depth estimator: {e}")
            self.depth_estimator = None
    
    def _init_segmentation_model(self):
        """Initialize semantic segmentation model."""
        if not TRANSFORMERS_AVAILABLE:
            self.logger.warning("Transformers not available, skipping segmentation")
            self.segmentation_model = None
            return
        
        try:
            # Use DeepLab for semantic segmentation
            self.segmentation_model = pipeline(
                "image-segmentation",
                model="facebook/detr-resnet-50-panoptic",
                device=0 if torch.cuda.is_available() else -1
            )
            self.logger.info("Segmentation model initialized")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize segmentation model: {e}")
            self.segmentation_model = None
    
    def estimate(self, image: np.ndarray, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced lighting estimation using pretrained models.
        
        Args:
            image: Input BGR image
            features: Basic scene features from SceneAnalyzer
            
        Returns:
            Enhanced lighting parameters
        """
        # Get enhanced features using pretrained models
        enhanced_features = self._extract_enhanced_features(image)
        
        # Combine with basic features
        combined_features = {**features, **enhanced_features}
        
        # Estimate lighting parameters
        lighting_params = self._predict_lighting(image, combined_features)
        
        return lighting_params
    
    def _extract_enhanced_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract features using pretrained models."""
        features = {}
        
        # CNN features
        if self.feature_extractor is not None:
            features.update(self._extract_cnn_features(image))
        
        # Depth features
        if self.depth_estimator is not None:
            features.update(self._extract_depth_features(image))
        
        # Segmentation features
        if self.segmentation_model is not None:
            features.update(self._extract_segmentation_features(image))
        
        return features
    
    def _extract_cnn_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract features using pretrained CNN."""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply transforms and get features
            input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.feature_extractor(input_tensor)
                features = features.squeeze().cpu().numpy()
            
            # Analyze the feature vector
            feature_stats = {
                'cnn_feature_mean': float(np.mean(features)),
                'cnn_feature_std': float(np.std(features)),
                'cnn_feature_max': float(np.max(features)),
                'cnn_feature_energy': float(np.sum(features**2)),
                'cnn_feature_sparsity': float(np.sum(features > 0) / len(features))
            }
            
            return feature_stats
            
        except Exception as e:
            self.logger.warning(f"CNN feature extraction failed: {e}")
            return {}
    
    def _extract_depth_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract depth-based lighting features."""
        try:
            # Convert to PIL format for the model
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get depth map
            depth_result = self.depth_estimator(rgb_image)
            depth_map = np.array(depth_result['depth'])
            
            # Analyze depth for lighting cues
            depth_features = {
                'depth_variance': float(np.var(depth_map)),
                'depth_gradient_x': float(np.mean(np.abs(np.gradient(depth_map, axis=1)))),
                'depth_gradient_y': float(np.mean(np.abs(np.gradient(depth_map, axis=0)))),
                'depth_range': float(np.max(depth_map) - np.min(depth_map)),
                'depth_mean': float(np.mean(depth_map))
            }
            
            return depth_features
            
        except Exception as e:
            self.logger.warning(f"Depth feature extraction failed: {e}")
            return {}
    
    def _extract_segmentation_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract segmentation-based features."""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get segmentation
            segments = self.segmentation_model(rgb_image)
            
            # Analyze segments for lighting information
            num_segments = len(segments) if segments else 0
            
            seg_features = {
                'num_segments': num_segments,
                'scene_complexity': min(num_segments / 10.0, 1.0)  # Normalize
            }
            
            return seg_features
            
        except Exception as e:
            self.logger.warning(f"Segmentation feature extraction failed: {e}")
            return {}
    
    def _predict_lighting(self, image: np.ndarray, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict lighting parameters from enhanced features."""
        
        # Enhanced light direction estimation
        direction = self._estimate_light_direction_enhanced(image, features)
        
        # Enhanced intensity estimation
        intensity = self._estimate_intensity_enhanced(image, features)
        
        # Enhanced color temperature estimation
        color_temp = self._estimate_color_temperature_enhanced(image, features)
        
        # Enhanced ambient estimation
        ambient = self._estimate_ambient_enhanced(image, features)
        
        # Enhanced shadow estimation
        shadows = self._estimate_shadows_enhanced(image, features)
        
        return {
            'direction': direction,
            'intensity': intensity,
            'color_temp': color_temp,
            'ambient': ambient,
            'shadows': shadows
        }
    
    def _estimate_light_direction_enhanced(self, image: np.ndarray, features: Dict[str, Any]) -> np.ndarray:
        """Enhanced light direction estimation."""
        # Start with basic gradient-based estimation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Use gradients to estimate direction
        mean_grad_x = np.mean(grad_x)
        mean_grad_y = np.mean(grad_y)
        
        # Enhance with depth information if available
        if 'depth_gradient_x' in features and 'depth_gradient_y' in features:
            depth_weight = 0.3
            mean_grad_x = (1 - depth_weight) * mean_grad_x + depth_weight * features['depth_gradient_x']
            mean_grad_y = (1 - depth_weight) * mean_grad_y + depth_weight * features['depth_gradient_y']
        
        # Convert to 3D direction (assume light from above)
        direction = np.array([mean_grad_x, mean_grad_y, abs(mean_grad_x) + abs(mean_grad_y) + 0.5])
        
        # Normalize
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        else:
            direction = np.array([0, 0, 1])  # Default to overhead
        
        return direction
    
    def _estimate_intensity_enhanced(self, image: np.ndarray, features: Dict[str, Any]) -> float:
        """Enhanced intensity estimation."""
        # Basic intensity from image brightness
        mean_intensity = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        base_intensity = mean_intensity / 255.0
        
        # Enhance with CNN features if available
        if 'cnn_feature_energy' in features:
            cnn_factor = min(features['cnn_feature_energy'] / 1000.0, 1.0)
            base_intensity = 0.7 * base_intensity + 0.3 * cnn_factor
        
        # Consider depth variance (more variance = more directional lighting)
        if 'depth_variance' in features:
            depth_factor = min(features['depth_variance'] / 100.0, 1.0)
            base_intensity = base_intensity * (0.8 + 0.4 * depth_factor)
        
        return min(max(base_intensity, 0.0), 1.0)
    
    def _estimate_color_temperature_enhanced(self, image: np.ndarray, features: Dict[str, Any]) -> float:
        """Enhanced color temperature estimation."""
        # Basic color temperature from color balance
        mean_color = np.mean(image.reshape(-1, 3), axis=0)
        b, g, r = mean_color
        
        # Enhanced estimation using color ratios
        if b > 0:
            rg_ratio = r / b
            # Map to temperature range
            if rg_ratio > 1.2:  # Warm
                temp = 3000 + (rg_ratio - 1.2) * 1000
                temp = min(temp, 4000)
            elif rg_ratio < 0.8:  # Cool
                temp = 6500 + (0.8 - rg_ratio) * 2000
                temp = min(temp, 10000)
            else:  # Neutral
                temp = 5500
        else:
            temp = 6500  # Default daylight
        
        return float(temp)
    
    def _estimate_ambient_enhanced(self, image: np.ndarray, features: Dict[str, Any]) -> np.ndarray:
        """Enhanced ambient lighting estimation."""
        # Use shadow regions for ambient estimation
        shadows = features.get('shadows', {})
        shadow_mask = shadows.get('mask', None)
        
        if shadow_mask is not None and np.any(shadow_mask):
            # Use shadow regions to estimate ambient
            shadow_pixels = image[shadow_mask == 1]
            if len(shadow_pixels) > 0:
                ambient_bgr = np.mean(shadow_pixels, axis=0)
                # Convert BGR to RGB and normalize
                ambient_rgb = np.array([ambient_bgr[2], ambient_bgr[1], ambient_bgr[0]]) / 255.0
                return ambient_rgb * 0.5  # Darken for ambient
        
        # Fallback to overall image tone
        mean_color = np.mean(image.reshape(-1, 3), axis=0)
        ambient_rgb = np.array([mean_color[2], mean_color[1], mean_color[0]]) / 255.0
        return ambient_rgb * 0.3
    
    def _estimate_shadows_enhanced(self, image: np.ndarray, features: Dict[str, Any]) -> float:
        """Enhanced shadow strength estimation."""
        # Basic shadow estimation from intensity variance
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        intensity_var = np.var(gray) / (255.0 ** 2)
        
        # Enhance with depth variance if available
        if 'depth_variance' in features:
            depth_factor = min(features['depth_variance'] / 100.0, 1.0)
            shadow_strength = 0.6 * intensity_var + 0.4 * depth_factor
        else:
            shadow_strength = intensity_var
        
        # Consider scene complexity
        if 'scene_complexity' in features:
            complexity_factor = features['scene_complexity']
            shadow_strength = shadow_strength * (0.5 + 0.5 * complexity_factor)
        
        return min(max(shadow_strength, 0.0), 1.0)
