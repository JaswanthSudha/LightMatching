"""
Light Estimator - AI-based lighting parameter prediction from scene features
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import logging
import cv2
from pathlib import Path


class LightEstimationNetwork(nn.Module):
    """
    Neural network for predicting lighting parameters from image features.
    
    Architecture: Feature encoder -> Lighting parameter decoder
    Outputs: Light direction, intensity, color temperature, ambient lighting
    """
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        
        # Feature encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Light direction prediction head (3D unit vector)
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Tanh()  # Constrain to [-1, 1] range
        )
        
        # Light intensity prediction head
        self.intensity_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Constrain to [0, 1] range
        )
        
        # Color temperature prediction head
        self.color_temp_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Will be scaled to temperature range
        )
        
        # Ambient color prediction head (RGB)
        self.ambient_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()  # RGB values in [0, 1]
        )
        
        # Shadow strength prediction head
        self.shadow_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the network."""
        features = self.encoder(x)
        
        # Predict lighting parameters
        direction = self.direction_head(features)
        intensity = self.intensity_head(features)
        color_temp = self.color_temp_head(features)
        ambient = self.ambient_head(features)
        shadows = self.shadow_head(features)
        
        # Normalize direction vector
        direction = F.normalize(direction, p=2, dim=-1)
        
        return {
            'direction': direction,
            'intensity': intensity,
            'color_temp': color_temp,
            'ambient': ambient,
            'shadows': shadows
        }


class LightEstimator:
    """
    AI-based lighting parameter estimator using deep learning models.
    
    Takes scene features from SceneAnalyzer and predicts lighting parameters
    including direction, intensity, color temperature, and ambient lighting.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the light estimator.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = config.get('model_path', 'models/pretrained/light_estimator.pth')
        self.input_dim = config.get('input_dim', 512)
        self.use_pretrained = config.get('use_pretrained', True)
        
        # Initialize model
        self.model = LightEstimationNetwork(self.input_dim)
        self.model.to(self.device)
        
        # Load pretrained weights if available
        if self.use_pretrained and Path(self.model_path).exists():
            self._load_model()
        else:
            self.logger.warning("No pretrained model found. Using randomly initialized weights.")
        
        self.model.eval()
    
    def estimate(self, image: np.ndarray, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate lighting parameters from image and extracted features.
        
        Args:
            image: Input BGR image
            features: Dictionary of extracted scene features
            
        Returns:
            Dictionary containing predicted lighting parameters
        """
        # Convert features to model input
        feature_vector = self._features_to_vector(features)
        
        # Predict lighting parameters using the neural network
        with torch.no_grad():
            input_tensor = torch.FloatTensor(feature_vector).unsqueeze(0).to(self.device)
            predictions = self.model(input_tensor)
        
        # Convert predictions to numpy and process
        results = self._process_predictions(predictions)
        
        # Add environment map if requested
        if self.config.get('generate_env_map', False):
            results['env_map'] = self._generate_environment_map(image, results)
        
        return results
    
    def _features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert scene features dictionary to a fixed-size feature vector."""
        vector_parts = []
        
        # Shadow features (10 dimensions)
        shadow_features = features.get('shadows', {})
        vector_parts.extend([
            shadow_features.get('num_regions', 0) / 10.0,  # Normalize
            shadow_features.get('total_area', 0) / 100000.0,
            shadow_features.get('avg_intensity', 0),
            shadow_features.get('coverage', 0),
            len(shadow_features.get('directions', [])) / 5.0,
            *([np.mean(shadow_features.get('directions', [0])), 
               np.std(shadow_features.get('directions', [0]))] if shadow_features.get('directions') else [0, 0])
        ])
        
        # Pad to 10 dimensions
        while len(vector_parts) < 10:
            vector_parts.append(0.0)
        vector_parts = vector_parts[:10]
        
        # Highlight features (8 dimensions)
        highlight_features = features.get('highlights', {})
        vector_parts.extend([
            highlight_features.get('num_regions', 0) / 20.0,
            highlight_features.get('avg_intensity', 0),
            highlight_features.get('coverage', 0),
            len(highlight_features.get('positions', [])) / 10.0
        ])
        
        # Add highlight position statistics if available
        positions = highlight_features.get('positions', [])
        if positions:
            pos_array = np.array(positions)
            vector_parts.extend([
                np.mean(pos_array[:, 0]) / features['image_shape'][1],  # Normalized x
                np.mean(pos_array[:, 1]) / features['image_shape'][0],  # Normalized y
                np.std(pos_array[:, 0]) / features['image_shape'][1],
                np.std(pos_array[:, 1]) / features['image_shape'][0]
            ])
        else:
            vector_parts.extend([0.5, 0.5, 0.0, 0.0])  # Center with no variance
        
        # Color analysis features (20 dimensions)
        color_features = features.get('color_analysis', {})
        dominant_colors = color_features.get('dominant_colors', [[128, 128, 128]] * 8)
        color_percentages = color_features.get('color_percentages', [0.125] * 8)
        
        # Flatten dominant colors and percentages (limited to first 6 colors for space)
        for i in range(min(6, len(dominant_colors))):
            vector_parts.extend([c / 255.0 for c in dominant_colors[i]])  # RGB normalized
        while len([x for x in vector_parts[-18:]]) < 18:  # Pad color values
            vector_parts.extend([0.5, 0.5, 0.5])
        
        # Add color percentages (first 6)
        vector_parts.extend(color_percentages[:6])
        while len([x for x in vector_parts[-6:]]) < 6:
            vector_parts.append(0.0)
        
        # Mean color and color temperature
        mean_color = color_features.get('mean_color', [128, 128, 128])
        vector_parts.extend([c / 255.0 for c in mean_color])
        vector_parts.append(color_features.get('color_temp_estimate', 6500) / 10000.0)
        
        # Gradient features (4 dimensions)
        gradient_features = features.get('gradients', {})
        vector_parts.extend([
            gradient_features.get('peak_direction', 0) / (2 * np.pi) + 0.5,  # Normalize to [0,1]
            gradient_features.get('avg_magnitude', 0) / 255.0
        ])
        
        # Surface normal features (3 dimensions)
        normal_features = features.get('surface_normals', {})
        mean_normal = normal_features.get('mean_normal', [0, 0, 1])
        vector_parts.extend(mean_normal)
        
        # Texture features (4 dimensions)
        texture_features = features.get('texture_features', {})
        vector_parts.extend([
            texture_features.get('texture_energy', 0) / 1000000.0,  # Normalize
            texture_features.get('mean_intensity', 0) / 255.0,
            texture_features.get('intensity_variance', 0) / 10000.0
        ])
        
        # Lighting cues (6 dimensions)
        lighting_cues = features.get('lighting_cues', {})
        vector_parts.extend([
            lighting_cues.get('mean_intensity', 0) / 255.0,
            lighting_cues.get('intensity_std', 0) / 255.0,
            lighting_cues.get('ambient_ratio', 0.5),
            lighting_cues.get('dynamic_range', 0) / 255.0,
            len(lighting_cues.get('intensity_peaks', [])) / 5.0
        ])
        
        # Ensure we have exactly the expected input dimension
        vector = np.array(vector_parts, dtype=np.float32)
        
        if len(vector) > self.input_dim:
            vector = vector[:self.input_dim]
        elif len(vector) < self.input_dim:
            padding = np.zeros(self.input_dim - len(vector), dtype=np.float32)
            vector = np.concatenate([vector, padding])
        
        return vector
    
    def _process_predictions(self, predictions: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Process raw model predictions into final lighting parameters."""
        # Convert tensors to numpy
        direction = predictions['direction'].cpu().numpy().flatten()
        intensity = predictions['intensity'].cpu().numpy().item()
        color_temp_norm = predictions['color_temp'].cpu().numpy().item()
        ambient = predictions['ambient'].cpu().numpy().flatten()
        shadows = predictions['shadows'].cpu().numpy().item()
        
        # Scale color temperature from [0,1] to [2000, 10000] Kelvin
        color_temp = 2000 + color_temp_norm * 8000
        
        return {
            'direction': direction,
            'intensity': intensity,
            'color_temp': color_temp,
            'ambient': ambient,
            'shadows': shadows
        }
    
    def _generate_environment_map(self, image: np.ndarray, lighting_params: Dict[str, Any]) -> Optional[np.ndarray]:
        """Generate a simple HDR environment map based on lighting parameters."""
        # This is a simplified environment map generation
        # In practice, you might use more sophisticated methods
        
        env_size = self.config.get('env_map_size', (64, 32))  # Width x Height
        env_map = np.ones((env_size[1], env_size[0], 3), dtype=np.float32)
        
        # Set ambient lighting
        ambient_color = lighting_params['ambient']
        env_map *= ambient_color.reshape(1, 1, 3)
        
        # Add directional light
        direction = lighting_params['direction']
        intensity = lighting_params['intensity']
        
        # Convert 3D direction to spherical coordinates for environment map
        # Simplified mapping: assume direction is in world coordinates
        phi = np.arctan2(direction[1], direction[0])  # Azimuth
        theta = np.arccos(np.clip(direction[2], -1, 1))  # Elevation
        
        # Map to environment map coordinates
        u = int((phi / (2 * np.pi) + 0.5) * env_size[0]) % env_size[0]
        v = int((theta / np.pi) * env_size[1])
        v = max(0, min(env_size[1] - 1, v))
        
        # Add bright spot for main light source
        light_radius = 3
        for dy in range(-light_radius, light_radius + 1):
            for dx in range(-light_radius, light_radius + 1):
                env_u = (u + dx) % env_size[0]
                env_v = max(0, min(env_size[1] - 1, v + dy))
                
                distance = np.sqrt(dx*dx + dy*dy)
                if distance <= light_radius:
                    brightness = intensity * np.exp(-distance / light_radius)
                    env_map[env_v, env_u] += brightness
        
        # Ensure HDR values (can be > 1.0)
        env_map = np.clip(env_map, 0, 10.0)
        
        return env_map
    
    def _load_model(self) -> None:
        """Load pretrained model weights."""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Loaded pretrained model from {self.model_path}")
        except Exception as e:
            self.logger.warning(f"Failed to load pretrained model: {e}")
    
    def save_model(self, path: str) -> None:
        """Save current model weights."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        self.logger.info(f"Model saved to {path}")
    
    def train_mode(self) -> None:
        """Set model to training mode."""
        self.model.train()
    
    def eval_mode(self) -> None:
        """Set model to evaluation mode."""
        self.model.eval()


class TraditionalLightEstimator:
    """
    Traditional (non-AI) light estimation methods as fallback.
    
    Uses classical computer vision techniques for light estimation
    when neural network models are not available.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def estimate(self, image: np.ndarray, features: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate lighting using traditional computer vision methods."""
        
        # Estimate light direction from shadows and gradients
        direction = self._estimate_light_direction(features)
        
        # Estimate intensity from image brightness
        intensity = self._estimate_intensity(image, features)
        
        # Estimate color temperature from color analysis
        color_temp = features.get('color_analysis', {}).get('color_temp_estimate', 6500)
        
        # Estimate ambient lighting
        ambient = self._estimate_ambient(image, features)
        
        # Estimate shadow strength
        shadows = features.get('shadows', {}).get('coverage', 0.3)
        
        return {
            'direction': direction,
            'intensity': intensity,
            'color_temp': color_temp,
            'ambient': ambient,
            'shadows': shadows
        }
    
    def _estimate_light_direction(self, features: Dict[str, Any]) -> np.ndarray:
        """Estimate light direction from shadows and gradients."""
        # Use shadow directions if available
        shadow_directions = features.get('shadows', {}).get('directions', [])
        if shadow_directions:
            # Average shadow direction (opposite to light direction)
            avg_shadow_dir = np.mean(shadow_directions)
            # Convert to 3D direction (assume light from above)
            light_dir = np.array([
                -np.cos(avg_shadow_dir),
                -np.sin(avg_shadow_dir),
                0.5  # Assume light comes from above
            ])
        else:
            # Use gradient peak direction as fallback
            gradient_dir = features.get('gradients', {}).get('peak_direction', 0)
            light_dir = np.array([
                np.cos(gradient_dir + np.pi/2),  # Perpendicular to gradient
                np.sin(gradient_dir + np.pi/2),
                0.7  # Default elevation
            ])
        
        # Normalize
        return light_dir / np.linalg.norm(light_dir)
    
    def _estimate_intensity(self, image: np.ndarray, features: Dict[str, Any]) -> float:
        """Estimate light intensity from image brightness."""
        mean_intensity = features.get('lighting_cues', {}).get('mean_intensity', 128)
        # Normalize to [0, 1] range
        return min(1.0, mean_intensity / 128.0)
    
    def _estimate_ambient(self, image: np.ndarray, features: Dict[str, Any]) -> np.ndarray:
        """Estimate ambient lighting color."""
        mean_color = features.get('color_analysis', {}).get('mean_color', [128, 128, 128])
        # Convert BGR to RGB and normalize
        ambient_rgb = np.array([mean_color[2], mean_color[1], mean_color[0]]) / 255.0
        # Darken for ambient (typically darker than direct lighting)
        return ambient_rgb * 0.3
