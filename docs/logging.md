# Logging Best Practices

This guide explains how to effectively use Python's logging module in our codebase, whether you're writing modules, running scripts from CLI, or working in Jupyter notebooks.

## Environment Variables

The application's log level can be controlled using the `DIRECTMULTISTEP_LOG_LEVEL` environment variable:

```bash
# Set log level for the current session
export DIRECTMULTISTEP_LOG_LEVEL=DEBUG
python your_script.py

# Or set it for a single command
DIRECTMULTISTEP_LOG_LEVEL=DEBUG python your_script.py
```

Valid log levels are:

- `DEBUG`: Most verbose, detailed debugging information
- `INFO`: General operational information (default)
- `WARNING`: Unexpected situations that aren't errors
- `ERROR`: Serious problems that need attention
- `CRITICAL`: Critical issues that may cause program failure

## Module Development

When writing a module, follow these guidelines:

```python
from directmultistep.utils.logging_config import logger

def my_function():
    # Use appropriate log levels
    logger.debug("Detailed information for debugging")
    logger.info("General information about progress")
    logger.warning("Something unexpected but not error")
    logger.error("A more serious problem")
    logger.critical("Program may not be able to continue")
```

Key points:

- Don't configure the logger in your modules
- Always use `from directmultistep.utils.logging_config import logger`
- Choose appropriate log levels
- Don't use print statements for debugging
- Don't add parameters like `verbose` to your functions

## Jupyter Notebook Usage

For Jupyter notebooks, put this in your first cell:

```python
from directmultistep.utils.logging_config import logger

logger.setLevel(logging.DEBUG)  # To see debug messages
logger.setLevel(logging.INFO)   # Back to info only

```

## Log Levels Guide

Choose the appropriate level based on the message importance:

- **DEBUG**: Detailed information for diagnosing problems

  ```python
  logger.debug(f"Processing data frame with shape {df.shape}")
  ```

- **INFO**: Confirmation that things are working as expected

  ```python
  logger.info("Model training started")
  ```

- **WARNING**: Indication that something unexpected happened

  ```python
  logger.warning("Using fallback parameter value")
  ```

- **ERROR**: More serious problem that prevented function from working

  ```python
  logger.error("Failed to load model weights")
  ```

- **CRITICAL**: Program may not be able to continue

  ```python
  logger.critical("Out of memory - cannot continue processing")
  ```

## Common Pitfalls

1. **Configuring Loggers in Modules**: Only configure logging in your entry points (main scripts, notebooks)

2. **Using Print Statements**: Avoid print statements for debugging; use logger.debug instead

3. **Hard-coding Log Levels**: Don't set log levels in your modules; let the application control them

4. **Creating Multiple Handlers**: Clear existing handlers in notebooks to avoid duplicate logs

5. **Using f-strings for Debug Messages**: For expensive operations, check level first:

```python
# Bad (string formatting happens regardless of level)
logger.debug(f"Expensive operation result: {expensive_operation()}")

# Good (string formatting only happens if needed)
if logger.isEnabledFor(logging.DEBUG):
    logger.debug(f"Expensive operation result: {expensive_operation()}")
```
