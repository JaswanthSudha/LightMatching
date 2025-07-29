"""
Scene Analyzer - Extracts visual features from scene images for light estimation
"""

import numpy as np
import cv2
from typing import Dict, Any, Tuple, List
import logging
from sklearn.cluster import KMeans
from scipy import ndimage


class SceneAnalyzer:
    """
    Analyzes scene images to extract features relevant for lighting estimation.
    
    Features extracted:
    - Shadow regions and directions
    - Highlight/specular regions
    - Color distribution and dominant colors  
    - Texture gradients
    - Surface normals estimation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the scene analyzer.
        
        Args:
            config: Configuration dictionary with analysis parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Analysis parameters
        self.shadow_threshold = config.get('shadow_threshold', 0.3)
        self.highlight_threshold = config.get('highlight_threshold', 0.8)
        self.num_color_clusters = config.get('num_color_clusters', 8)
        self.gradient_kernel_size = config.get('gradient_kernel_size', 3)
    
    def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Perform comprehensive scene analysis.
        
        Args:
            image: Input BGR image
            
        Returns:
            Dictionary containing extracted features
        """
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        features = {
            'image_shape': image.shape,
            'shadows': self._analyze_shadows(gray, hsv),
            'highlights': self._analyze_highlights(gray, hsv),
            'color_analysis': self._analyze_colors(image, lab),
            'gradients': self._analyze_gradients(gray),
            'surface_normals': self._estimate_surface_normals(gray),
            'texture_features': self._analyze_texture(gray),
            'lighting_cues': self._extract_lighting_cues(image, hsv)
        }
        
        return features
    
    def _analyze_shadows(self, gray: np.ndarray, hsv: np.ndarray) -> Dict[str, Any]:
        """Detect and analyze shadow regions."""
        # Create shadow mask based on value channel and intensity
        v_channel = hsv[:, :, 2].astype(np.float32) / 255.0
        shadow_mask = (v_channel < self.shadow_threshold).astype(np.uint8)
        
        # Clean up shadow mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        
        # Find shadow contours
        contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze shadow properties
        shadow_areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 100]
        shadow_directions = self._estimate_shadow_directions(contours)
        
        return {
            'mask': shadow_mask,
            'num_regions': len(shadow_areas),
            'total_area': sum(shadow_areas),
            'avg_intensity': np.mean(v_channel[shadow_mask == 1]) if np.any(shadow_mask) else 0,
            'directions': shadow_directions,
            'coverage': np.sum(shadow_mask) / shadow_mask.size
        }
    
    def _analyze_highlights(self, gray: np.ndarray, hsv: np.ndarray) -> Dict[str, Any]:
        """Detect and analyze highlight/specular regions."""
        v_channel = hsv[:, :, 2].astype(np.float32) / 255.0
        highlight_mask = (v_channel > self.highlight_threshold).astype(np.uint8)
        
        # Clean up highlight mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        highlight_mask = cv2.morphologyEx(highlight_mask, cv2.MORPH_OPEN, kernel)
        
        # Find highlight contours
        contours, _ = cv2.findContours(highlight_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze highlight properties  
        highlight_areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 10]
        highlight_positions = [self._get_contour_center(c) for c in contours if cv2.contourArea(c) > 10]
        
        return {
            'mask': highlight_mask,
            'num_regions': len(highlight_areas),
            'positions': highlight_positions,
            'avg_intensity': np.mean(v_channel[highlight_mask == 1]) if np.any(highlight_mask) else 0,
            'coverage': np.sum(highlight_mask) / highlight_mask.size
        }
    
    def _analyze_colors(self, image: np.ndarray, lab: np.ndarray) -> Dict[str, Any]:
        """Analyze color distribution and dominant colors."""
        # Reshape for clustering
        pixels = image.reshape(-1, 3)
        
        # K-means clustering to find dominant colors
        kmeans = KMeans(n_clusters=self.num_color_clusters, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        dominant_colors = kmeans.cluster_centers_.astype(np.uint8)
        labels = kmeans.labels_
        
        # Calculate color statistics
        color_percentages = np.bincount(labels) / len(labels)
        
        # Analyze color temperature indicators
        mean_color = np.mean(pixels, axis=0)
        color_temp_estimate = self._estimate_color_temperature(mean_color)
        
        return {
            'dominant_colors': dominant_colors.tolist(),
            'color_percentages': color_percentages.tolist(),
            'mean_color': mean_color.tolist(),
            'color_temp_estimate': color_temp_estimate,
            'color_variance': np.var(pixels, axis=0).tolist()
        }
    
    def _analyze_gradients(self, gray: np.ndarray) -> Dict[str, Any]:
        """Analyze image gradients for lighting direction cues."""
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.gradient_kernel_size)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.gradient_kernel_size)
        
        # Compute gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Analyze dominant gradient directions
        valid_grads = magnitude > np.percentile(magnitude, 75)
        if np.any(valid_grads):
            dominant_directions = direction[valid_grads]
            direction_hist, bins = np.histogram(dominant_directions, bins=36, range=(-np.pi, np.pi))
            peak_direction = bins[np.argmax(direction_hist)]
        else:
            peak_direction = 0
        
        return {
            'magnitude': magnitude,
            'direction': direction,
            'peak_direction': float(peak_direction),
            'avg_magnitude': float(np.mean(magnitude))
        }
    
    def _estimate_surface_normals(self, gray: np.ndarray) -> Dict[str, Any]:
        """Estimate surface normals from image gradients."""
        # Smooth the image to reduce noise
        smooth = cv2.GaussianBlur(gray, (5, 5), 1.0)
        
        # Compute gradients
        grad_x = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)
        
        # Estimate surface normals (simplified shape-from-shading)
        # Assume orthographic projection and Lambertian surfaces
        norm_factor = np.sqrt(grad_x**2 + grad_y**2 + 1)
        
        normal_x = -grad_x / norm_factor
        normal_y = -grad_y / norm_factor  
        normal_z = 1.0 / norm_factor
        
        # Stack to create normal map
        normals = np.stack([normal_x, normal_y, normal_z], axis=2)
        
        return {
            'normals': normals,
            'mean_normal': np.mean(normals, axis=(0, 1)).tolist()
        }
    
    def _analyze_texture(self, gray: np.ndarray) -> Dict[str, Any]:
        """Analyze texture features that may indicate surface properties."""
        # Local Binary Pattern for texture analysis
        def lbp(image, radius=1, n_points=8):
            """Simple Local Binary Pattern implementation."""
            h, w = image.shape
            lbp_image = np.zeros_like(image)
            
            for i in range(radius, h - radius):
                for j in range(radius, w - radius):
                    center = image[i, j]
                    code = 0
                    for k in range(n_points):
                        angle = 2 * np.pi * k / n_points
                        x = int(i + radius * np.cos(angle))
                        y = int(j + radius * np.sin(angle))
                        if image[x, y] >= center:
                            code |= (1 << k)
                    lbp_image[i, j] = code
            return lbp_image
        
        # Compute texture features
        lbp_image = lbp(gray)
        texture_hist, _ = np.histogram(lbp_image.ravel(), bins=256, range=(0, 256))
        
        # Compute texture energy and homogeneity
        texture_energy = np.sum(texture_hist**2)
        
        return {
            'lbp_histogram': texture_hist.tolist(),
            'texture_energy': float(texture_energy),
            'mean_intensity': float(np.mean(gray)),
            'intensity_variance': float(np.var(gray))
        }
    
    def _extract_lighting_cues(self, image: np.ndarray, hsv: np.ndarray) -> Dict[str, Any]:
        """Extract various lighting cues from the image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Analyze intensity distribution
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Find peaks in histogram (may indicate multiple light sources)
        hist_smooth = ndimage.gaussian_filter1d(hist.flatten(), sigma=2)
        peaks = []
        for i in range(1, len(hist_smooth) - 1):
            if hist_smooth[i] > hist_smooth[i-1] and hist_smooth[i] > hist_smooth[i+1]:
                if hist_smooth[i] > 0.01 * np.max(hist_smooth):
                    peaks.append(i)
        
        # Estimate ambient vs direct lighting ratio
        low_intensity = np.sum(hist[:64])
        high_intensity = np.sum(hist[192:])
        ambient_ratio = low_intensity / (low_intensity + high_intensity) if (low_intensity + high_intensity) > 0 else 0.5
        
        return {
            'intensity_histogram': hist.flatten().tolist(),
            'intensity_peaks': peaks,
            'mean_intensity': float(np.mean(gray)),
            'intensity_std': float(np.std(gray)),
            'ambient_ratio': float(ambient_ratio),
            'dynamic_range': float(np.max(gray) - np.min(gray))
        }
    
    def _estimate_shadow_directions(self, contours: List) -> List[float]:
        """Estimate shadow directions from contour shapes."""
        directions = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                # Fit ellipse to get major axis direction
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    angle = ellipse[2]  # Angle in degrees
                    directions.append(np.radians(angle))
        return directions
    
    def _get_contour_center(self, contour: np.ndarray) -> Tuple[float, float]:
        """Get the center point of a contour."""
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            return (float(cx), float(cy))
        return (0.0, 0.0)
    
    def _estimate_color_temperature(self, rgb_color: np.ndarray) -> float:
        """Estimate color temperature from RGB values (simplified)."""
        # Convert BGR to RGB
        r, g, b = rgb_color[2], rgb_color[1], rgb_color[0]
        
        # Simple color temperature estimation
        # Warmer colors (higher red) -> lower temperature
        # Cooler colors (higher blue) -> higher temperature
        if b > 0:
            ratio = r / b
            # Map ratio to temperature range (2000K - 10000K)
            temp = 6500 - (ratio - 1) * 2000
            return max(2000, min(10000, temp))
        return 6500  # Default daylight temperature
