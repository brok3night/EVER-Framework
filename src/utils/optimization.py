"""
Optimization Utilities for EVER
"""
from typing import Dict, List, Any, Callable, Tuple
import numpy as np
import heapq
import time

def optimize_vector_operations(vectors: List[List[float]], 
                             operation: str = 'similarity') -> np.ndarray:
    """
    Optimize vector operations using numpy
    
    Args:
        vectors: List of vectors
        operation: Operation to perform ('similarity', 'mean', etc.)
        
    Returns:
        Result of operation
    """
    # Convert to numpy array
    arr = np.array(vectors)
    
    if operation == 'similarity':
        # Compute all pairwise similarities
        # Normalize vectors
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        normalized = arr / np.maximum(norms, 1e-10)  # Avoid division by zero
        
        # Compute cosine similarity
        similarity_matrix = np.dot(normalized, normalized.T)
        
        return similarity_matrix
    
    elif operation == 'mean':
        # Compute mean vector
        return np.mean(arr, axis=0)
    
    elif operation == 'weighted_mean':
        # This would require weights as additional input
        # For now, return simple mean
        return np.mean(arr, axis=0)
    
    return None

class PriorityQueue:
    """Efficient priority queue implementation"""
    
    def __init__(self):
        self.queue = []
        self.entry_finder = {}
        self.counter = 0
        self.REMOVED = '<removed>'
    
    def add_task(self, task: Any, priority: float = 0) -> None:
        """Add new task or update priority of existing task"""
        if task in self.entry_finder:
            self.remove_task(task)
        
        entry = [priority, self.counter, task]
        self.counter += 1
        self.entry_finder[task] = entry
        heapq.heappush(self.queue, entry)
    
    def remove_task(self, task: Any) -> None:
        """Mark an existing task as removed"""
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED
    
    def pop_task(self) -> Any:
        """Remove and return the lowest priority task"""
        while self.queue:
            priority, count, task = heapq.heappop(self.queue)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task
        
        # Queue is empty
        raise KeyError('pop from an empty priority queue')
    
    def peek(self) -> Tuple[float, Any]:
        """View the lowest priority task without removing it"""
        while self.queue:
            priority, count, task = self.queue[0]
            if task is self.REMOVED:
                heapq.heappop(self.queue)
            else:
                return (priority, task)
        
        # Queue is empty
        raise KeyError('peek from an empty priority queue')
    
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return len(self.entry_finder) == 0

def memoize(max_size: int = 100):
    """
    Memoization decorator with size limit
    
    Args:
        max_size: Maximum number of results to cache
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        timestamps = {}
        
        def wrapper(*args, **kwargs):
            # Create key from arguments
            key = str(args) + str(sorted(kwargs.items()))
            
            if key in cache:
                # Update timestamp
                timestamps[key] = time.time()
                return cache[key]
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Check if cache is full
            if len(cache) >= max_size:
                # Remove oldest entry
                oldest_key = min(timestamps, key=timestamps.get)
                del cache[oldest_key]
                del timestamps[oldest_key]
            
            # Add to cache
            cache[key] = result
            timestamps[key] = time.time()
            
            return result
        
        return wrapper
    
    return decorator

def batch_process(items: List, process_func: Callable, batch_size: int = 100) -> List:
    """
    Process items in batches for better efficiency
    
    Args:
        items: Items to process
        process_func: Function to apply to each batch
        batch_size: Size of each batch
        
    Returns:
        List of processed results
    """
    results = []
    
    # Process in batches
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        batch_results = process_func(batch)
        results.extend(batch_results)
    
    return results