"""
EVR Connector - Native connector for EVER to access .evr datasets
"""
import os
import json
import zipfile
import tempfile
from typing import Dict, List, Any, Tuple
import logging

class EVRConnector:
    """Connects EVER to .evr format datasets"""
    
    def __init__(self, consciousness_core, auto_convert=True):
        """
        Initialize the EVR connector
        
        Args:
            consciousness_core: The EVER consciousness core
            auto_convert: Whether to automatically convert non-EVR datasets
        """
        self.consciousness = consciousness_core
        self.auto_convert = auto_convert
        
        # Registry of connected EVR datasets
        self.connected_datasets = {}
        
        # Cache for frequently accessed entities
        self.entity_cache = {}
        self.cache_size = 1000
        
        # Set up logging
        self.logger = logging.getLogger('EVRConnector')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def connect(self, path: str, dataset_name: str = None) -> bool:
        """
        Connect to an EVR dataset
        
        Args:
            path: Path to .evr file or other dataset
            dataset_name: Optional custom name for the dataset
        """
        # Check if file is EVR format
        if path.lower().endswith('.evr'):
            return self._connect_evr(path, dataset_name)
        elif self.auto_convert:
            # Auto-convert non-EVR format
            return self._convert_and_connect(path, dataset_name)
        else:
            self.logger.error(f"Not an EVR file: {path}")
            return False
    
    def _connect_evr(self, evr_path: str, dataset_name: str = None) -> bool:
        """Connect to an existing EVR file"""
        try:
            # Create temporary directory for EVR contents
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract metadata and energy map
                with zipfile.ZipFile(evr_path, 'r') as zipf:
                    # Extract metadata
                    zipf.extract('evr_metadata.json', temp_dir)
                    zipf.extract('energy_map.json', temp_dir)
                    
                    # Read metadata and energy map
                    with open(os.path.join(temp_dir, 'evr_metadata.json'), 'r') as f:
                        metadata = json.load(f)
                    
                    with open(os.path.join(temp_dir, 'energy_map.json'), 'r') as f:
                        energy_map = json.load(f)
                    
                    # Use provided name or name from metadata
                    if not dataset_name:
                        dataset_name = metadata.get('dataset_name', os.path.basename(evr_path).split('.')[0])
                    
                    # Create connection
                    connection = {
                        'name': dataset_name,
                        'path': evr_path,
                        'metadata': metadata,
                        'energy_map': energy_map,
                        'status': 'connected',
                        'zipfile': evr_path,
                        'resonance': 0.1  # Initial resonance with consciousness
                    }
                    
                    # Register connection
                    self.connected_datasets[dataset_name] = connection
                    
                    # Load indexes if available
                    indexes = {}
                    for filename in zipf.namelist():
                        if filename.startswith('indexes/'):
                            zipf.extract(filename, temp_dir)
                            
                            index_name = os.path.splitext(os.path.basename(filename))[0].split('_')[0]
                            with open(os.path.join(temp_dir, filename), 'r') as f:
                                indexes[index_name] = json.load(f)
                    
                    connection['indexes'] = indexes
            
            self.logger.info(f"Connected to EVR dataset: {dataset_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error connecting to EVR dataset: {e}")
            return False
    
    def _convert_and_connect(self, path: str, dataset_name: str = None) -> bool:
        """Convert a non-EVR dataset and connect to it"""
        try:
            # Import the converter here to avoid circular imports
            from src.formats.evr_converter import EVRConverter
            
            # Create temporary directory for conversion
            with tempfile.TemporaryDirectory() as temp_dir:
                # Initialize converter
                converter = EVRConverter(output_dir=temp_dir)
                
                # Convert dataset
                self.logger.info(f"Auto-converting {path} to EVR format")
                evr_path = converter.convert(path, dataset_name=dataset_name)
                
                # Connect to the converted dataset
                return self._connect_evr(evr_path, dataset_name)
                
        except Exception as e:
            self.logger.error(f"Error converting dataset: {e}")
            return False
    
    def query_by_energy(self, dataset_name: str, energy_signature: Dict, 
                        similarity_threshold: float = 0.7, limit: int = 100) -> List[Dict]:
        """
        Query dataset using energy signature matching
        
        Args:
            dataset_name: Name of the dataset to query
            energy_signature: Energy signature to match
            similarity_threshold: Minimum similarity threshold
            limit: Maximum number of results to return
        """
        if dataset_name not in self.connected_datasets:
            self.logger.error(f"Dataset {dataset_name} not connected")
            return []
        
        connection = self.connected_datasets[dataset_name]
        
        # Use energy-based indexing to pre-filter candidates
        candidate_ids = self._find_candidates(connection, energy_signature)
        
        # Calculate similarity for each candidate
        results = []
        
        for entity_id in candidate_ids:
            # Get entity signature from energy map
            if entity_id in connection['energy_map']:
                entity_signature = connection['energy_map'][entity_id]
                
                # Calculate similarity
                similarity = self._calculate_energy_similarity(energy_signature, entity_signature)
                
                # Add to results if above threshold
                if similarity >= similarity_threshold:
                    # Get entity data
                    entity_data = self._get_entity(connection, entity_id)
                    
                    if entity_data:
                        results.append({
                            'entity_id': entity_id,
                            'similarity': similarity,
                            'entity_data': entity_data
                        })
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Apply limit
        if limit > 0:
            results = results[:limit]
        
        return results
    
    def _get_entity(self, connection: Dict, entity_id: str) -> Dict:
        """Get entity data from cache or EVR file"""
        cache_key = f"{connection['name']}:{entity_id}"
        
        # Check cache first
        if cache_key in self.entity_cache:
            return self.entity_cache[cache_key]
        
        # Load from EVR file
        try:
            with zipfile.ZipFile(connection['zipfile'], 'r') as zipf:
                entity_path = f"entities/entity_{entity_id}.json"
                
                with zipf.open(entity_path) as f:
                    entity_data = json.load(f)
                
                # Add to cache
                if len(self.entity_cache) >= self.cache_size:
                    # Remove random item if cache full
                    self.entity_cache.pop(next(iter(self.entity_cache)))
                
                self.entity_cache[cache_key] = entity_data
                
                return entity_data
                
        except Exception as e:
            self.logger.error(f"Error loading entity {entity_id}: {e}")
        
        return None
    
    def _find_candidates(self, connection: Dict, energy_signature: Dict) -> List[str]:
        """Find candidate entities using energy indexes"""
        # Implementation would use indexes from the EVR file to find candidates
        # For simplicity, returning all entity IDs here
        return list(connection['energy_map'].keys())
    
    def _calculate_energy_similarity(self, sig1: Dict, sig2: Dict) -> float:
        """Calculate similarity between energy signatures"""
        # Simplified implementation
        return 0.8  # Return a constant for this example
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """Get information about a connected dataset"""
        if dataset_name not in self.connected_datasets:
            self.logger.error(f"Dataset {dataset_name} not connected")
            return {}
        
        connection = self.connected_datasets[dataset_name]
        
        return {
            'name': connection['name'],
            'path': connection['path'],
            'format_version': connection['metadata'].get('format_version'),
            'original_format': connection['metadata'].get('original_format'),
            'entry_count': connection['metadata'].get('entry_count'),
            'creation_timestamp': connection['metadata'].get('creation_timestamp'),
            'status': connection['status']
        }
    
    def list_datasets(self) -> List[Dict]:
        """List all connected datasets"""
        return [self.get_dataset_info(name) for name in self.connected_datasets]