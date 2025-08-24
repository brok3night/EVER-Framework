"""
Standardized Error Handling for EVER
"""
from typing import Dict, Any, Callable, TypeVar, Optional
import logging
import functools
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ever')

# Custom exceptions
class EverError(Exception):
    """Base class for EVER exceptions"""
    pass

class EnergySignatureError(EverError):
    """Error related to energy signatures"""
    pass

class ReasoningError(EverError):
    """Error in reasoning operations"""
    pass

class ComprehensionError(EverError):
    """Error in comprehension operations"""
    pass

# Type variable for generic functions
T = TypeVar('T')

def safe_operation(default_return: Optional[T] = None, 
                  log_error: bool = True) -> Callable:
    """
    Decorator for safe operation with standardized error handling
    
    Args:
        default_return: Value to return if operation fails
        log_error: Whether to log the error
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except EverError as e:
                # Handle known EVER errors
                if log_error:
                    logger.error(f"EVER error in {func.__name__}: {e}")
                return default_return
            except Exception as e:
                # Handle unexpected errors
                if log_error:
                    logger.error(f"Unexpected error in {func.__name__}: {e}")
                    logger.debug(traceback.format_exc())
                return default_return
        return wrapper
    return decorator

def validate_energy_signature(energy: Dict) -> bool:
    """
    Validate energy signature format
    
    Args:
        energy: Energy signature to validate
        
    Returns:
        True if valid
        
    Raises:
        EnergySignatureError: If validation fails
    """
    # Check for required keys
    required_keys = ['magnitude', 'frequency', 'entropy', 'vector']
    
    for key in required_keys:
        if key not in energy:
            raise EnergySignatureError(f"Missing required key: {key}")
        
        if not isinstance(energy[key], dict) or 'value' not in energy[key]:
            raise EnergySignatureError(f"Invalid format for {key}")
    
    # Check vector
    vector = energy['vector'].get('value')
    if not isinstance(vector, list) or len(vector) < 3:
        raise EnergySignatureError("Vector must be a list with at least 3 elements")
    
    # Check numerical values
    for key in ['magnitude', 'frequency', 'entropy']:
        value = energy[key].get('value')
        if not isinstance(value, (int, float)) or value < 0 or value > 1:
            raise EnergySignatureError(f"{key} must be a number between 0 and 1")
    
    return True