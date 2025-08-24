"""
Memory Management Utilities for EVER
"""
from typing import Dict, List, Any, Callable
import time
import threading

class MemoryManager:
    """Manages memory collections with size limits and pruning strategies"""
    
    def __init__(self, max_items: int = 1000, 
                prune_threshold: float = 0.9, 
                prune_strategy: str = 'lru'):
        self.max_items = max_items
        self.prune_threshold = prune_threshold
        self.prune_strategy = prune_strategy
        
        # Mapping of collection IDs to their metadata
        self.collections = {}
        
        # Lock for thread safety
        self.lock = threading.RLock()
    
    def register_collection(self, collection_id: str, collection: List,
                          item_score_func: Callable = None,
                          max_items: int = None) -> None:
        """
        Register a collection for memory management
        
        Args:
            collection_id: Unique identifier for this collection
            collection: The list or dict to manage
            item_score_func: Function to score items for pruning (higher = keep)
            max_items: Maximum items for this collection (overrides default)
        """
        with self.lock:
            self.collections[collection_id] = {
                'collection': collection,
                'score_func': item_score_func,
                'max_items': max_items or self.max_items,
                'last_access': time.time(),
                'access_count': 0
            }
    
    def add_item(self, collection_id: str, item: Any) -> bool:
        """
        Add an item to a managed collection
        
        Args:
            collection_id: Collection identifier
            item: Item to add
            
        Returns:
            True if added successfully
        """
        with self.lock:
            # Check if collection exists
            if collection_id not in self.collections:
                return False
            
            collection_info = self.collections[collection_id]
            collection = collection_info['collection']
            
            # Update access stats
            collection_info['last_access'] = time.time()
            collection_info['access_count'] += 1
            
            # Add item
            collection.append(item)
            
            # Check if pruning is needed
            if len(collection) >= collection_info['max_items'] * self.prune_threshold:
                self._prune_collection(collection_id)
            
            return True
    
    def get_collection(self, collection_id: str) -> List:
        """Get a managed collection"""
        with self.lock:
            if collection_id not in self.collections:
                return None
            
            # Update access stats
            self.collections[collection_id]['last_access'] = time.time()
            self.collections[collection_id]['access_count'] += 1
            
            return self.collections[collection_id]['collection']
    
    def _prune_collection(self, collection_id: str) -> None:
        """Prune a collection to its maximum size"""
        collection_info = self.collections[collection_id]
        collection = collection_info['collection']
        max_items = collection_info['max_items']
        
        # If already under limit, do nothing
        if len(collection) <= max_items:
            return
        
        # Determine items to keep
        if self.prune_strategy == 'lru' and isinstance(collection, list):
            # For LRU, keep most recently added items
            items_to_remove = len(collection) - max_items
            del collection[:items_to_remove]
            
        elif self.prune_strategy == 'score' and collection_info['score_func']:
            # Score-based pruning
            score_func = collection_info['score_func']
            
            # Score all items
            scored_items = [(item, score_func(item)) for item in collection]
            
            # Sort by score (highest first)
            scored_items.sort(key=lambda x: x[1], reverse=True)
            
            # Keep only the highest-scoring items
            collection.clear()
            collection.extend([item for item, _ in scored_items[:max_items]])
            
        else:
            # Default to simple truncation
            del collection[max_items:]