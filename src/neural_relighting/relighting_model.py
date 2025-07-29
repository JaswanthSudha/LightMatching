"""
Neural Relighting Model - AI-based relighting of CG objects to match scene lighting
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path

from ..data_structures import LightingParameters


class RelightingUNet(nn.Module):
    """
    U-Net architecture for neural relighting.
    
    Takes an input image and lighting parameters, outputs a relit version.
    The network learns to apply lighting changes while preserving object structure.
    """
    
    def __init__(self, input_channels: int = 4, output_channels: int = 3, lighting_dim: int = 16):
        super().__init__()
        
        self.lighting_dim = lighting_dim
        
        # Lighting parameter encoder
        self.lighting_encoder = nn.Sequential(
            nn.Linear(10, 32),  # Input: direction(3) + intensity(1) + color_temp(1) + ambient(3) + shadows(1) + reserved(1)
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, lighting_dim),
            nn.ReLU()
        )
        
        # Encoder (downsampling path)
        self.enc1 = self._conv_block(input_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck with lighting conditioning
        self.bottleneck = self._conv_block(512 + lighting_dim, 1024)
        
        # Decoder (upsampling path)
        self.dec4 = self._upconv_block(1024, 512)
        self.dec3 = self._upconv_block(1024, 256)  # 1024 = 512 + 512 (skip connection)
        self.dec2 = self._upconv_block(512, 128)   # 512 = 256 + 256
        self.dec1 = self._upconv_block(256, 64)    # 256 = 128 + 128
        
        # Final output layer
        self.final = nn.Conv2d(128, output_channels, kernel_size=1)  # 128 = 64 + 64
        
    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Convolutional block with batch norm and ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _upconv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Upconvolutional block for decoder."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, image: torch.Tensor, lighting_params: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the relighting network.
        
        Args:
            image: Input image tensor [B, C, H, W]
            lighting_params: Lighting parameters tensor [B, 10]
            
        Returns:
            Relit image tensor [B, 3, H, W]
        """
        # Encode lighting parameters
        lighting_features = self.lighting_encoder(lighting_params)  # [B, lighting_dim]
        
        # Encoder path
        enc1_out = self.enc1(image)
        enc1_pool = F.max_pool2d(enc1_out, 2)
        
        enc2_out = self.enc2(enc1_pool)
        enc2_pool = F.max_pool2d(enc2_out, 2)
        
        enc3_out = self.enc3(enc2_pool)
        enc3_pool = F.max_pool2d(enc3_out, 2)
        
        enc4_out = self.enc4(enc3_pool)
        enc4_pool = F.max_pool2d(enc4_out, 2)
        
        # Add lighting conditioning to bottleneck
        B, C, H, W = enc4_pool.shape
        lighting_map = lighting_features.view(B, self.lighting_dim, 1, 1).expand(B, self.lighting_dim, H, W)
        bottleneck_input = torch.cat([enc4_pool, lighting_map], dim=1)
        
        bottleneck_out = self.bottleneck(bottleneck_input)
        
        # Decoder path with skip connections
        dec4_out = self.dec4(bottleneck_out)
        dec4_concat = torch.cat([dec4_out, enc4_out], dim=1)
        
        dec3_out = self.dec3(dec4_concat)
        dec3_concat = torch.cat([dec3_out, enc3_out], dim=1)
        
        dec2_out = self.dec2(dec3_concat)
        dec2_concat = torch.cat([dec2_out, enc2_out], dim=1)
        
        dec1_out = self.dec1(dec2_concat)
        dec1_concat = torch.cat([dec1_out, enc1_out], dim=1)
        
        # Final output
        output = self.final(dec1_concat)
        output = torch.sigmoid(output)  # Ensure output is in [0, 1] range
        
        return output


class RelightingModel:
    """
    Neural relighting model that applies lighting changes to CG objects.
    
    Uses a U-Net architecture to learn how to relight objects based on
    target lighting parameters extracted from real scenes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the relighting model.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = config.get('model_path', 'models/pretrained/relighting_model.pth')
        self.input_size = config.get('input_size', (512, 512))
        self.use_pretrained = config.get('use_pretrained', True)
        
        # Initialize model
        self.model = RelightingUNet()
        self.model.to(self.device)
        
        # Load pretrained weights if available
        if self.use_pretrained and Path(self.model_path).exists():
            self._load_model()
        else:
            self.logger.warning("No pretrained relighting model found. Using randomly initialized weights.")
        
        self.model.eval()
    
    def relight(self, cg_image: np.ndarray, lighting_params: LightingParameters) -> np.ndarray:
        """
        Apply neural relighting to a CG object image.
        
        Args:
            cg_image: Input CG object image (BGR format)
            lighting_params: Target lighting parameters
            
        Returns:
            Relit CG object image (BGR format)
        """
        # Preprocess input
        input_tensor, original_size = self._preprocess_image(cg_image)
        lighting_tensor = self._lighting_to_tensor(lighting_params)
        
        # Apply relighting
        with torch.no_grad():
            relit_tensor = self.model(input_tensor, lighting_tensor)
        
        # Postprocess output
        relit_image = self._postprocess_image(relit_tensor, original_size)
        
        # Blend with original if needed (for subtle effects)
        blend_factor = self.config.get('blend_factor', 0.8)
        if blend_factor < 1.0:
            relit_image = cv2.addWeighted(
                relit_image, blend_factor,
                cg_image, 1.0 - blend_factor,
                0
            )
        
        return relit_image
    
    def _preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Preprocess image for neural network input."""
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        
        # Resize to model input size
        resized = cv2.resize(image, self.input_size)
        
        # Convert BGR to RGB and normalize to [0, 1]
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Create alpha channel (assume full opacity for now)
        alpha = np.ones((self.input_size[1], self.input_size[0], 1), dtype=np.float32)
        
        # Combine RGBA
        rgba_image = np.concatenate([rgb_image, alpha], axis=2)
        
        # Convert to tensor and add batch dimension
        tensor = torch.FloatTensor(rgba_image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        return tensor, original_size
    
    def _lighting_to_tensor(self, lighting_params: LightingParameters) -> torch.Tensor:
        """Convert lighting parameters to tensor format."""
        # Pack lighting parameters into a single vector
        params_vector = np.concatenate([
            lighting_params.direction,                    # 3 elements
            [lighting_params.intensity],                  # 1 element
            [lighting_params.color_temp / 10000.0],      # 1 element, normalized
            lighting_params.ambient,                      # 3 elements
            [lighting_params.shadows],                    # 1 element
            [0.0]                                        # 1 reserved element
        ])
        
        # Convert to tensor and add batch dimension
        tensor = torch.FloatTensor(params_vector).unsqueeze(0).to(self.device)
        
        return tensor
    
    def _postprocess_image(self, tensor: torch.Tensor, original_size: Tuple[int, int]) -> np.ndarray:
        """Postprocess neural network output back to image format."""
        # Remove batch dimension and convert to numpy
        output = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Convert from [0, 1] to [0, 255] and to uint8
        output = (output * 255.0).astype(np.uint8)
        
        # Convert RGB back to BGR
        bgr_output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        # Resize back to original size
        resized_output = cv2.resize(bgr_output, original_size)
        
        return resized_output
    
    def _load_model(self) -> None:
        """Load pretrained model weights."""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Loaded pretrained relighting model from {self.model_path}")
        except Exception as e:
            self.logger.warning(f"Failed to load pretrained relighting model: {e}")
    
    def save_model(self, path: str) -> None:
        """Save current model weights."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        self.logger.info(f"Relighting model saved to {path}")


class TraditionalRelighting:
    """
    Traditional (non-AI) relighting methods as fallback.
    
    Uses classical image processing techniques to simulate
    lighting changes when neural models are not available.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def relight(self, cg_image: np.ndarray, lighting_params: LightingParameters) -> np.ndarray:
        """Apply traditional relighting techniques."""
        result = cg_image.copy().astype(np.float32)
        
        # Apply intensity adjustment
        intensity_factor = lighting_params.intensity
        result = result * intensity_factor
        
        # Apply color temperature adjustment
        result = self._adjust_color_temperature(result, lighting_params.color_temp)
        
        # Apply ambient lighting
        result = self._apply_ambient_lighting(result, lighting_params.ambient)
        
        # Apply directional lighting simulation
        result = self._simulate_directional_lighting(result, lighting_params.direction, lighting_params.intensity)
        
        # Apply shadow simulation
        if lighting_params.shadows > 0:
            result = self._simulate_shadows(result, lighting_params.shadows)
        
        # Ensure values are in valid range
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def _adjust_color_temperature(self, image: np.ndarray, color_temp: float) -> np.ndarray:
        """Adjust image color temperature."""
        # Simple color temperature adjustment
        if color_temp < 5500:  # Warm light
            factor = (5500 - color_temp) / 3500  # Normalize to [0, 1]
            # Increase red, decrease blue
            image[:, :, 2] = image[:, :, 2] * (1 + factor * 0.3)  # More red
            image[:, :, 0] = image[:, :, 0] * (1 - factor * 0.2)  # Less blue
        elif color_temp > 6500:  # Cool light
            factor = (color_temp - 6500) / 3500  # Normalize to [0, 1]
            # Decrease red, increase blue
            image[:, :, 2] = image[:, :, 2] * (1 - factor * 0.2)  # Less red
            image[:, :, 0] = image[:, :, 0] * (1 + factor * 0.3)  # More blue
        
        return image
    
    def _apply_ambient_lighting(self, image: np.ndarray, ambient_color: np.ndarray) -> np.ndarray:
        """Apply ambient lighting color."""
        # Convert ambient color from RGB to BGR
        ambient_bgr = np.array([ambient_color[2], ambient_color[1], ambient_color[0]])
        
        # Apply ambient lighting as additive component
        ambient_contribution = image * 0.3  # 30% of original for ambient
        ambient_contribution = ambient_contribution * ambient_bgr.reshape(1, 1, 3)
        
        # Blend with original
        result = image * 0.7 + ambient_contribution
        
        return result
    
    def _simulate_directional_lighting(self, image: np.ndarray, direction: np.ndarray, intensity: float) -> np.ndarray:
        """Simulate directional lighting effects."""
        # Convert to grayscale for lighting calculation
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Create a simple lighting gradient based on direction
        h, w = gray.shape
        y_grad, x_grad = np.gradient(gray)
        
        # Simulate lighting interaction with surface gradients
        # This is a simplified approximation
        light_x, light_y = direction[0], direction[1]
        
        # Create directional lighting effect
        lighting_effect = (x_grad * light_x + y_grad * light_y) * intensity * 0.5
        lighting_effect = np.clip(lighting_effect, -50, 50)
        
        # Apply to all channels
        for c in range(3):
            image[:, :, c] += lighting_effect
        
        return image
    
    def _simulate_shadows(self, image: np.ndarray, shadow_strength: float) -> np.ndarray:
        """Simulate shadow effects."""
        # Create a simple shadow mask based on image intensity
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Areas with lower intensity get more shadow
        shadow_mask = 1.0 - (gray / 255.0)
        shadow_mask = np.power(shadow_mask, 0.5)  # Soften the effect
        
        # Apply shadow
        shadow_factor = 1.0 - (shadow_mask * shadow_strength * 0.3)
        
        # Apply to all channels
        for c in range(3):
            image[:, :, c] *= shadow_factor
        
        return image
