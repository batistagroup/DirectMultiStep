import logging
import logging.config
import os
from typing import Any

# --- Hardcoded Configuration ---
LOGGING_CONFIG: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        }
    },
    "loggers": {
        "directmultistep": {
            "handlers": ["console"],
            "propagate": False,
            "level": "INFO",  # Default level
        }
    },
}
# --- End Hardcoded Configuration ---


def setup_logging() -> None:
    """Setup logging configuration from hardcoded dict with environment variable override"""

    # Get log level from environment variable, default to INFO if not set
    log_level = os.getenv("DIRECTMULTISTEP_LOG_LEVEL", "INFO").upper()

    # Validate the log level
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if log_level not in valid_levels:
        print(f"Invalid log level {log_level}, defaulting to INFO")
        log_level = "INFO"  # Make sure to reset if invalid

    # Override the log level in the copied config
    LOGGING_CONFIG["loggers"]["directmultistep"]["level"] = log_level

    logging.config.dictConfig(LOGGING_CONFIG)


logger = logging.getLogger("directmultistep")
