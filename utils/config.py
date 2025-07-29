"""
Configuration management for the light matching project
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging


class Config:
    """
    Configuration manager that handles loading and accessing project settings.
    
    Supports both YAML and JSON configuration files with fallback to default values.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default settings.
        """
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        self._config = self._get_default_config()
        
        # Load configuration from file if provided
        if config_path and Path(config_path).exists():
            self._load_config(config_path)
        elif config_path:
            self.logger.warning(f"Configuration file not found: {config_path}. Using defaults.")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            'scene_analysis': {
                'shadow_threshold': 0.3,
                'highlight_threshold': 0.8,
                'num_color_clusters': 8,
                'gradient_kernel_size': 3
            },
            'light_estimation': {
                'model_path': 'models/pretrained/light_estimator.pth',
                'input_dim': 512,
                'use_pretrained': True,
                'generate_env_map': False,
                'env_map_size': (64, 32)
            },
            'neural_relighting': {
                'model_path': 'models/pretrained/relighting_model.pth',
                'input_size': (512, 512),
                'use_pretrained': True,
                'blend_factor': 0.8
            },
            'preprocessing': {
                'resize_method': 'bilinear',
                'normalize': True,
                'augmentation': False
            },
            'postprocessing': {
                'composition_method': 'alpha_blend',
                'edge_smoothing': True,
                'color_matching': True,
                'output_format': 'BGR'
            },
            'training': {
                'batch_size': 16,
                'learning_rate': 0.001,
                'num_epochs': 100,
                'validation_split': 0.2,
                'checkpoint_interval': 10,
                'early_stopping_patience': 15
            },
            'data': {
                'input_dir': 'data/input',
                'output_dir': 'data/output',
                'training_dir': 'data/training',
                'supported_formats': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'logs/light_matching.log'
            },
            'performance': {
                'use_gpu': True,
                'num_workers': 4,
                'cache_features': True,
                'parallel_processing': True
            }
        }
    
    def _load_config(self, config_path: str) -> None:
        """Load configuration from file."""
        try:
            config_file = Path(config_path)
            
            if config_file.suffix.lower() == '.yaml' or config_file.suffix.lower() == '.yml':
                with open(config_file, 'r') as f:
                    user_config = yaml.safe_load(f)
            elif config_file.suffix.lower() == '.json':
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_file.suffix}")
            
            # Merge user config with defaults
            self._merge_config(user_config)
            self.logger.info(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_path}: {e}")
            self.logger.info("Using default configuration")
    
    def _merge_config(self, user_config: Dict[str, Any]) -> None:
        """Recursively merge user configuration with defaults."""
        def _deep_merge(default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
            """Deep merge two dictionaries."""
            result = default.copy()
            
            for key, value in user.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = _deep_merge(result[key], value)
                else:
                    result[key] = value
            
            return result
        
        self._config = _deep_merge(self._config, user_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'scene_analysis.shadow_threshold')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent dict
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def save(self, output_path: str) -> None:
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save configuration file
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if output_file.suffix.lower() == '.yaml' or output_file.suffix.lower() == '.yml':
                with open(output_file, 'w') as f:
                    yaml.dump(self._config, f, default_flow_style=False, indent=2)
            elif output_file.suffix.lower() == '.json':
                with open(output_file, 'w') as f:
                    json.dump(self._config, f, indent=2)
            else:
                raise ValueError(f"Unsupported output format: {output_file.suffix}")
            
            self.logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration to {output_path}: {e}")
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with multiple key-value pairs.
        
        Args:
            updates: Dictionary of configuration updates
        """
        self._merge_config(updates)
    
    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return self._config.copy()
    
    # Property accessors for major configuration sections
    @property
    def scene_analysis(self) -> Dict[str, Any]:
        """Scene analysis configuration."""
        return self._config['scene_analysis']
    
    @property
    def light_estimation(self) -> Dict[str, Any]:
        """Light estimation configuration."""
        return self._config['light_estimation']
    
    @property
    def neural_relighting(self) -> Dict[str, Any]:
        """Neural relighting configuration."""
        return self._config['neural_relighting']
    
    @property
    def preprocessing(self) -> Dict[str, Any]:
        """Preprocessing configuration."""
        return self._config['preprocessing']
    
    @property
    def postprocessing(self) -> Dict[str, Any]:
        """Postprocessing configuration."""
        return self._config['postprocessing']
    
    @property
    def training(self) -> Dict[str, Any]:
        """Training configuration."""
        return self._config['training']
    
    @property
    def data(self) -> Dict[str, Any]:
        """Data configuration."""
        return self._config['data']
    
    @property
    def logging_config(self) -> Dict[str, Any]:
        """Logging configuration."""
        return self._config['logging']
    
    @property
    def performance(self) -> Dict[str, Any]:
        """Performance configuration."""
        return self._config['performance']
    
    def create_sample_config(self, output_path: str) -> None:
        """
        Create a sample configuration file with all available options.
        
        Args:
            output_path: Path to save sample configuration
        """
        sample_config = {
            '# Light Matching Project Configuration': None,
            '# This file contains all configurable parameters for the light matching system': None,
            '': None,
            'scene_analysis': {
                '# Shadow detection threshold (0.0 - 1.0)': None,
                'shadow_threshold': 0.3,
                '# Highlight detection threshold (0.0 - 1.0)': None,
                'highlight_threshold': 0.8,
                '# Number of dominant colors to extract': None,
                'num_color_clusters': 8,
                '# Gradient kernel size for edge detection': None,
                'gradient_kernel_size': 3
            },
            'light_estimation': {
                '# Path to pretrained light estimation model': None,
                'model_path': 'models/pretrained/light_estimator.pth',
                '# Input feature vector dimension': None,
                'input_dim': 512,
                '# Whether to use pretrained weights': None,
                'use_pretrained': True,
                '# Generate HDR environment maps': None,
                'generate_env_map': False,
                '# Environment map size (width, height)': None,
                'env_map_size': [64, 32]
            },
            'neural_relighting': {
                '# Path to pretrained relighting model': None,
                'model_path': 'models/pretrained/relighting_model.pth',
                '# Neural network input size (width, height)': None,
                'input_size': [512, 512],
                '# Whether to use pretrained weights': None,
                'use_pretrained': True,
                '# Blending factor with original image (0.0 - 1.0)': None,
                'blend_factor': 0.8
            },
            'data': {
                '# Input data directory': None,
                'input_dir': 'data/input',
                '# Output results directory': None,
                'output_dir': 'data/output',
                '# Training data directory': None,
                'training_dir': 'data/training',
                '# Supported image formats': None,
                'supported_formats': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            },
            'performance': {
                '# Use GPU acceleration if available': None,
                'use_gpu': True,
                '# Number of worker processes': None,
                'num_workers': 4,
                '# Cache extracted features': None,
                'cache_features': True,
                '# Enable parallel processing': None,
                'parallel_processing': True
            }
        }
        
        # Remove comment keys for actual saving
        cleaned_config = {}
        for key, value in sample_config.items():
            if not key.startswith('#') and key != '':
                if isinstance(value, dict):
                    cleaned_config[key] = {}
                    for sub_key, sub_value in value.items():
                        if not sub_key.startswith('#'):
                            cleaned_config[key][sub_key] = sub_value
                else:
                    cleaned_config[key] = value
        
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                if output_file.suffix.lower() in ['.yaml', '.yml']:
                    f.write("# Light Matching Project Configuration\n")
                    f.write("# Modify these values to customize the behavior\n\n")
                    yaml.dump(cleaned_config, f, default_flow_style=False, indent=2)
                else:
                    json.dump(cleaned_config, f, indent=2)
            
            self.logger.info(f"Sample configuration created at {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create sample configuration: {e}")


# Convenience function to create a sample config file
def create_sample_config(output_path: str = "config.yaml") -> None:
    """Create a sample configuration file."""
    config = Config()
    config.create_sample_config(output_path)
