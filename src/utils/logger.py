"""
Logger module - Centralized logging configuration for the natural gas trading system.
"""

import os
import logging
import yaml
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional


def setup_logger(
    name: str,
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    console: bool = True,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up a logger with file and/or console output.
    
    Args:
        name: Name of the logger
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, no file logging.
        console: Whether to output logs to console
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Load config if no log_level specified
    if log_level is None:
        config_path = Path(__file__).parents[2] / 'config' / 'config.yaml'
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            log_level = config.get('logging', {}).get('level', 'INFO')
        except Exception:
            log_level = 'INFO'
    
    # Map string log level to logging constant
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    level = level_map.get(log_level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers if any
    if logger.handlers:
        logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add file handler if log_file specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Check if rotation is daily or by size
        config_path = Path(__file__).parents[2] / 'config' / 'config.yaml'
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            rotation = config.get('logging', {}).get('rotation', '1 day')
        except Exception:
            rotation = '1 day'
        
        if rotation.endswith('day'):
            # Time-based rotation
            handler = TimedRotatingFileHandler(
                log_file,
                when='midnight',
                interval=int(rotation.split()[0]),
                backupCount=backup_count
            )
        else:
            # Size-based rotation
            handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    # Add console handler if specified
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the default configuration from the config file.
    
    Args:
        name: Name of the logger (typically __name__)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    config_path = Path(__file__).parents[2] / 'config' / 'config.yaml'
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logging_config = config.get('logging', {})
        log_level = logging_config.get('level', 'INFO')
        log_to_file = logging_config.get('log_to_file', False)
        log_file = logging_config.get('log_file', 'logs/trading_system.log')
        
        # Convert relative path to absolute
        if not os.path.isabs(log_file):
            log_file = str(Path(__file__).parents[2] / log_file)
        
        return setup_logger(
            name=name,
            log_level=log_level,
            log_file=log_file if log_to_file else None
        )
    
    except Exception as e:
        # Fallback to console logging if config file can't be read
        print(f"Warning: Could not read config file for logging: {e}")
        return setup_logger(name=name)


if __name__ == "__main__":
    # Example usage
    logger = get_logger(__name__)
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message") 