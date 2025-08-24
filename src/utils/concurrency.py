"""
Concurrency Utilities for EVER
"""
import threading
from typing import Dict, Any, Callable, TypeVar
import functools
import time

# Type variable for generic functions
T = TypeVar('T')

class ThreadSafeCache:
    """Thread-safe caching mechanism"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
        self.lock = threading.RLock()
        self.access_times = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        with self.lock:
            if key not in self.cache:
                return default
            
            # Check if expired
            access_time = self.access_times.get(key, 0)
            if time.time() - access_time > self.ttl:
                # Expired
                del self.cache[key]
                del self.access_times[key]
                return default
            
            # Update access time
            self.access_times[key] = time.time()
            
            return self.cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache"""
        with self.lock:
            # Check if cache is full
            if len(self.cache) >= self.max_size and key not in self.cache:
                # Evict oldest entry
                oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            
            # Set value
            self.cache[key] = value
            self.access_times[key] = time.time()
    
    def clear(self) -> None:
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()

def synchronized(lock=None):
    """
    Decorator for thread-safe method execution
    
    Args:
        lock: Lock to use (creates a new one if None)
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        # Create lock if not provided
        func_lock = lock or threading.RLock()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with func_lock:
                return func(*args, **kwargs)
        
        return wrapper
    
    return decorator

class AtomicCounter:
    """Thread-safe counter"""
    
    def __init__(self, initial_value: int = 0):
        self.value = initial_value
        self.lock = threading.RLock()
    
    def increment(self, amount: int = 1) -> int:
        """Increment counter and return new value"""
        with self.lock:
            self.value += amount
            return self.value
    
    def decrement(self, amount: int = 1) -> int:
        """Decrement counter and return new value"""
        with self.lock:
            self.value -= amount
            return self.value
    
    def get(self) -> int:
        """Get current value"""
        with self.lock:
            return self.value
    
    def set(self, value: int) -> None:
        """Set value"""
        with self.lock:
            self.value = value