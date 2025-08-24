"""
Serialization Utilities for EVER
"""
from typing import Dict, Any, List
import json
import pickle
import gzip
import os
import time
import logging

logger = logging.getLogger('ever')

class Serializer:
    """Handles serialization and deserialization of EVER components"""
    
    def __init__(self, storage_dir: str = "storage"):
        self.storage_dir = storage_dir
        
        # Ensure storage directory exists
        os.makedirs(storage_dir, exist_ok=True)
    
    def save_component(self, component_id: str, data: Any, 
                      format: str = 'json') -> bool:
        """
        Save component data to storage
        
        Args:
            component_id: Identifier for this component
            data: Data to save
            format: Format to use ('json' or 'pickle')
            
        Returns:
            True if saved successfully
        """
        try:
            # Create timestamp
            timestamp = int(time.time())
            
            # Create filename
            filename = f"{component_id}_{timestamp}"
            
            if format == 'json':
                filepath = os.path.join(self.storage_dir, f"{filename}.json")
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
            
            elif format == 'pickle':
                filepath = os.path.join(self.storage_dir, f"{filename}.pkl.gz")
                with gzip.open(filepath, 'wb') as f:
                    pickle.dump(data, f)
            
            else:
                logger.error(f"Unsupported format: {format}")
                return False
            
            # Create latest link
            latest_path = os.path.join(self.storage_dir, f"{component_id}_latest")
            
            if os.path.exists(latest_path):
                os.remove(latest_path)
            
            os.symlink(os.path.basename(filepath), latest_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving component {component_id}: {e}")
            return False
    
    def load_component(self, component_id: str, timestamp: int = None,
                      format: str = 'json') -> Any:
        """
        Load component data from storage
        
        Args:
            component_id: Identifier for this component
            timestamp: Specific timestamp to load (None = latest)
            format: Format to use ('json' or 'pickle')
            
        Returns:
            Loaded data or None if not found
        """
        try:
            if timestamp is None:
                # Load latest
                latest_path = os.path.join(self.storage_dir, f"{component_id}_latest")
                
                if not os.path.exists(latest_path):
                    logger.error(f"No latest file found for {component_id}")
                    return None
                
                filepath = os.path.join(self.storage_dir, os.readlink(latest_path))
            else:
                # Load specific timestamp
                if format == 'json':
                    filepath = os.path.join(self.storage_dir, f"{component_id}_{timestamp}.json")
                else:
                    filepath = os.path.join(self.storage_dir, f"{component_id}_{timestamp}.pkl.gz")
            
            if not os.path.exists(filepath):
                logger.error(f"File not found: {filepath}")
                return None
            
            if format == 'json':
                with open(filepath, 'r') as f:
                    return json.load(f)
            
            elif format == 'pickle':
                with gzip.open(filepath, 'rb') as f:
                    return pickle.load(f)
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading component {component_id}: {e}")
            return None
    
    def list_available_versions(self, component_id: str) -> List[int]:
        """List available timestamps for a component"""
        try:
            versions = []
            
            # Check all files in storage directory
            for filename in os.listdir(self.storage_dir):
                # Check if filename matches pattern
                if filename.startswith(f"{component_id}_") and not filename.endswith("_latest"):
                    # Extract timestamp
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        timestamp_part = parts[1].split('.')[0]
                        if timestamp_part.isdigit():
                            versions.append(int(timestamp_part))
            
            # Sort by timestamp (newest first)
            versions.sort(reverse=True)
            
            return versions
            
        except Exception as e:
            logger.error(f"Error listing versions for {component_id}: {e}")
            return []