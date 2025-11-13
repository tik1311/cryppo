"""
Logging Configuration for CRYPPO
Separate logs for server and training operations
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging():
    """
    Setup logging configuration with separate files for server and training

    Log Files:
    - logs/server.log: API server operations, requests, errors
    - logs/training.log: Model training progress, metrics, checkpoints
    - logs/combined.log: All logs combined (for debugging)
    """
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Common formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # ===== Server Logger =====
    server_logger = logging.getLogger('server')
    server_logger.setLevel(logging.INFO)
    server_logger.propagate = False

    # Server file handler with rotation (max 10MB, keep 5 backups)
    server_handler = RotatingFileHandler(
        log_dir / 'server.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    server_handler.setLevel(logging.INFO)
    server_handler.setFormatter(formatter)

    # Server console handler
    server_console = logging.StreamHandler()
    server_console.setLevel(logging.INFO)
    server_console.setFormatter(formatter)

    server_logger.addHandler(server_handler)
    server_logger.addHandler(server_console)

    # ===== Training Logger =====
    training_logger = logging.getLogger('training')
    training_logger.setLevel(logging.DEBUG)
    training_logger.propagate = False

    # Training file handler with rotation (max 20MB, keep 10 backups)
    training_handler = RotatingFileHandler(
        log_dir / 'training.log',
        maxBytes=20*1024*1024,  # 20MB
        backupCount=10,
        encoding='utf-8'
    )
    training_handler.setLevel(logging.DEBUG)
    training_handler.setFormatter(formatter)

    # Training console handler (only WARNING and above)
    training_console = logging.StreamHandler()
    training_console.setLevel(logging.WARNING)
    training_console.setFormatter(formatter)

    training_logger.addHandler(training_handler)
    training_logger.addHandler(training_console)

    # ===== Combined Logger (for debugging) =====
    combined_handler = RotatingFileHandler(
        log_dir / 'combined.log',
        maxBytes=50*1024*1024,  # 50MB
        backupCount=3,
        encoding='utf-8'
    )
    combined_handler.setLevel(logging.DEBUG)
    combined_handler.setFormatter(formatter)

    # Add combined handler to root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(combined_handler)

    return server_logger, training_logger


def get_server_logger():
    """Get server logger instance"""
    return logging.getLogger('server')


def get_training_logger():
    """Get training logger instance"""
    return logging.getLogger('training')


# Initialize loggers on import
SERVER_LOGGER, TRAINING_LOGGER = setup_logging()
