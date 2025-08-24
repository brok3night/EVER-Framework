"""
Configuration Management for EVER
"""
from typing import Dict, Any, List
import json
import os
import logging

logger = logging.getLogger('ever')

class Configuration:
    """Manages EVER configuration"""
    
    def __init__(self, config_path: str = None):
        # Default configuration
        self.config = {
            'system': {
                'storage_dir': 'storage',
                'log_level': 'INFO',
                'thread_safe': True
            },
            'memory': {
                'working_memory_limit': 100,
                'short_term_memory_limit': 1000,
                'long_term_storage_enabled': True
            },
            'energy': {
                'default_vector_dimensions': 3,
                'similarity_threshold': 0.7,
                'energy_decay_rate': 0.05
            },
            'reasoning': {
                'max_reasoning_steps': 10,
                'max_navigation_paths': 5,
                'philosophical_patterns_limit': 50,
                'discovery_threshold': 0.7
            },
            'performance': {
                'cache_enabled': True,
                'cache_size': 1000,
                'batch_processing': True,
                'batch_size': 100
            }
        }
        
        # Load configuration if provided
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> bool:
        """
        Load configuration from file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            True if loaded successfully
        """
        try:
            if not os.path.exists(config_path):
                logger.warning(f"Configuration file not found: {config_path}")
                return False
            
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            
            # Update configuration
            self._update_config_recursive(self.config, loaded_config)
            
            logger.info(f"Configuration loaded from {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return False
    
    def save_config(self, config_path: str) -> bool:
        """
        Save configuration to file
        
        Args:
            config_path: Path to save configuration
            
        Returns:
            True if saved successfully
        """
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Configuration saved to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            path: Dot-separated path to configuration value
            default: Default value if not found
            
        Returns:
            Configuration value or default
        """
        parts = path.split('.')
        
        # Navigate through config
        current = self.config
        for part in parts:
            if part in current:
                current = current[part]
            else:
                return default
        
        return current
    
    def set(self, path: str, value: Any) -> bool:
        """
        Set configuration value
        
        Args:
            path: Dot-separated path to configuration value
            value: Value to set
            
        Returns:
            True if set successfully
        """
        parts = path.split('.')
        
        # Navigate through config
        current = self.config
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            
            current = current[part]
        
        # Set value
        current[parts[-1]] = value
        
        return True
    
    def _update_config_recursive(self, target: Dict, source: Dict) -> None:
        """Update configuration recursively"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recursively update nested dictionaries
                self._update_config_recursive(target[key], value)
            else:
                # Update value
                target[key] = value