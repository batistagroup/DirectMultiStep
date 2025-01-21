import logging
import logging.config
import os
from pathlib import Path

import tomli


def setup_logging() -> None:
    """Setup logging configuration from pyproject.toml with environment variable override"""
    pyproject_path = Path(__file__).parents[3] / "pyproject.toml"

    with open(pyproject_path, "rb") as f:
        config = tomli.load(f)

    # Get log level from environment variable, default to INFO if not set
    log_level = os.getenv("DIRECTMULTISTEP_LOG_LEVEL", "INFO").upper()

    # Validate the log level
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if log_level not in valid_levels:
        print(f"Invalid log level {log_level}, defaulting to INFO")
        log_level = "INFO"

    # Override the log level from config
    logging_config = config["tool"]["logging"]
    logging_config["loggers"]["directmultistep"]["level"] = log_level

    logging.config.dictConfig(logging_config)


logger = logging.getLogger("directmultistep")
