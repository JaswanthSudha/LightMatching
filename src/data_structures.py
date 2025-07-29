"""
Data structures used throughout the light matching project.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class LightingParameters:
    """Data class to store lighting parameters"""
    direction: np.ndarray  # Light direction vector (3D)
    intensity: float       # Light intensity (0-1)
    color_temp: float     # Color temperature in Kelvin
    ambient: np.ndarray   # Ambient color (RGB)
    shadows: float        # Shadow strength (0-1)
    environment_map: Optional[np.ndarray] = None  # HDR environment map
