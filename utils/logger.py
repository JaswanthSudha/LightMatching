"""
Logging utilities for the light matching project
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
import sys
from datetime import datetime


def setup_logger(
    name: str = __name__,
    level: str = 'INFO',
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up a logger with both console and file output.
    
    Args:
        name: Logger name
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Path to log file. If None, no file logging
        format_string: Custom format string
        max_file_size: Maximum log file size in bytes before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Set level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use rotating file handler to prevent huge log files
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class ProgressLogger:
    """
    A logger that can track progress of long-running operations.
    
    Useful for training models, processing large datasets, etc.
    """
    
    def __init__(self, logger: logging.Logger, total_steps: int, update_interval: int = 10):
        """
        Initialize progress logger.
        
        Args:
            logger: Base logger to use for output
            total_steps: Total number of steps in the operation
            update_interval: How often to log progress (every N steps)
        """
        self.logger = logger
        self.total_steps = total_steps
        self.update_interval = update_interval
        self.current_step = 0
        self.start_time = datetime.now()
    
    def update(self, step: Optional[int] = None, message: str = "Processing") -> None:
        """
        Update progress and log if necessary.
        
        Args:
            step: Current step number. If None, increments by 1
            message: Custom message to include in progress log
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        # Log progress at specified intervals
        if self.current_step % self.update_interval == 0 or self.current_step == self.total_steps:
            progress_percent = (self.current_step / self.total_steps) * 100
            elapsed_time = datetime.now() - self.start_time
            
            # Estimate remaining time
            if self.current_step > 0:
                avg_time_per_step = elapsed_time.total_seconds() / self.current_step
                remaining_steps = self.total_steps - self.current_step
                remaining_time = remaining_steps * avg_time_per_step
                remaining_time_str = f"{remaining_time:.1f}s"
            else:
                remaining_time_str = "unknown"
            
            self.logger.info(
                f"{message}: {self.current_step}/{self.total_steps} "
                f"({progress_percent:.1f}%) - "
                f"Elapsed: {elapsed_time.total_seconds():.1f}s - "
                f"Remaining: {remaining_time_str}"
            )
    
    def complete(self, message: str = "Completed") -> None:
        """Log completion message."""
        total_time = datetime.now() - self.start_time
        self.logger.info(f"{message} in {total_time.total_seconds():.1f}s")


class MetricsLogger:
    """
    Logger for tracking and reporting metrics during training or evaluation.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize metrics logger.
        
        Args:
            logger: Base logger to use for output
        """
        self.logger = logger
        self.metrics_history: Dict[str, list] = {}
    
    def log_metrics(self, metrics: Dict[str, float], epoch: Optional[int] = None, prefix: str = "") -> None:
        """
        Log metrics for current step/epoch.
        
        Args:
            metrics: Dictionary of metric names and values
            epoch: Current epoch number (optional)
            prefix: Prefix to add to metric names
        """
        # Store metrics in history
        for name, value in metrics.items():
            full_name = f"{prefix}{name}" if prefix else name
            if full_name not in self.metrics_history:
                self.metrics_history[full_name] = []
            self.metrics_history[full_name].append(value)
        
        # Format metrics for logging
        metric_strings = []
        for name, value in metrics.items():
            if isinstance(value, float):
                metric_strings.append(f"{name}: {value:.4f}")
            else:
                metric_strings.append(f"{name}: {value}")
        
        # Create log message
        if epoch is not None:
            log_message = f"Epoch {epoch} - " + " | ".join(metric_strings)
        else:
            log_message = " | ".join(metric_strings)
        
        if prefix:
            log_message = f"[{prefix.strip()}] {log_message}"
        
        self.logger.info(log_message)
    
    def log_best_metrics(self, metric_name: str = "loss", minimize: bool = True) -> None:
        """
        Log the best value achieved for a specific metric.
        
        Args:
            metric_name: Name of the metric to find best value for
            minimize: Whether lower values are better (True) or higher values are better (False)
        """
        if metric_name in self.metrics_history:
            values = self.metrics_history[metric_name]
            if minimize:
                best_value = min(values)
                best_epoch = values.index(best_value)
            else:
                best_value = max(values)
                best_epoch = values.index(best_value)
            
            self.logger.info(f"Best {metric_name}: {best_value:.4f} (epoch {best_epoch + 1})")
        else:
            self.logger.warning(f"Metric '{metric_name}' not found in history")
    
    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all tracked metrics.
        
        Returns:
            Dictionary with metric names as keys and summary stats as values
        """
        summary = {}
        for name, values in self.metrics_history.items():
            if values:
                summary[name] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': sum(values) / len(values),
                    'last': values[-1],
                    'count': len(values)
                }
        return summary


def log_system_info(logger: logging.Logger) -> None:
    """
    Log system information for debugging and reproducibility.
    
    Args:
        logger: Logger to use for output
    """
    import platform
    import sys
    
    logger.info("=== System Information ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"Python Executable: {sys.executable}")
    
    # GPU information
    try:
        import torch
        logger.info(f"PyTorch Version: {torch.__version__}")
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                logger.info(f"GPU {i}: {gpu_name}")
    except ImportError:
        logger.info("PyTorch not available")
    
    # Memory information
    try:
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"Total Memory: {memory.total / (1024**3):.1f} GB")
        logger.info(f"Available Memory: {memory.available / (1024**3):.1f} GB")
    except ImportError:
        logger.info("psutil not available for memory information")
    
    logger.info("=== End System Information ===")


def configure_logging_from_config(config: Dict[str, Any]) -> logging.Logger:
    """
    Configure logging based on configuration dictionary.
    
    Args:
        config: Configuration dictionary with logging settings
        
    Returns:
        Configured logger instance
    """
    logging_config = config.get('logging', {})
    
    return setup_logger(
        name='light_matching',
        level=logging_config.get('level', 'INFO'),
        log_file=logging_config.get('file', 'logs/light_matching.log'),
        format_string=logging_config.get('format', None)
    )


# Convenience function to get a standard logger for the project
def get_project_logger(name: str = 'light_matching') -> logging.Logger:
    """Get a standard logger for the light matching project."""
    return setup_logger(
        name=name,
        level='INFO',
        log_file='logs/light_matching.log'
    )
