"""
ENDF Connector - Native connector for EVER to access ENDF datasets
"""
import os
import json
import numpy as np
from typing import Dict, List, Any, Tuple

class ENDFConnector:
    def __init__(self, consciousness_core):
        self.consciousness = consciousness_core
        
        # Registry of connected ENDF datasets
        self.connected_datasets = {}
        
        # Cache for frequently accessed entities
        self.entity_cache = {}
        self.cache_size = 1000
        
        # Active energy resonance
        self.active_resonance = {}
    
    def connect(self, endf_path: str, dataset_name: str = None) -> bool:
        """
        Connect to an ENDF dataset
        
        Args:
            endf_path: Path to ENDF directory
            dataset_name: Optional custom name for the dataset
        """
        # Verify it's a valid ENDF directory
        metadata_path = os.path.join(endf_path, 'endf_metadata.json')
        energy_map_path = os.path.join(endf_path, 'energy_map.json')
        
        if not os.path.exists(metadata_path) or not os.path.exists(energy_map_path):
            print(f"Invalid ENDF directory: {endf_path}")
            return False
        
        # Load metadata
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Use provided name or original name from metadata
            if not dataset_name:
                dataset_name = metadata.get('dataset_name', os.path.basename(endf_path))
            
            # Load energy map
            with open(energy_map_path, 'r') as f:
                energy_map = json.load(f)
            
            # Load indexes
            indexes = self._load_indexes(os.path.join(endf_path, 'indexes'))
            
            # Create connection
            connection = {
                'name': dataset_name,
                'path': endf_path,
                'metadata': metadata,
                'energy_map': energy_map,
                'indexes': indexes,
                'status': 'connected',
                'entities_dir': os.path.join(endf_path, 'entities'),
                'resonance': 0.1  # Initial resonance with consciousness
            }
            
            # Register connection
            self.connected_datasets[dataset_name] = connection
            
            # Initialize resonance for this dataset
            self.active_resonance[dataset_name] = {
                'global': 0.1,
                'entities': {}
            }
            
            print(f"Connected to ENDF dataset: {dataset_name}")
            return True
            
        except Exception as e:
            print(f"Error connecting to ENDF dataset: {e}")
            return False
    
    def _load_indexes(self, indexes_dir: str) -> Dict:
        """Load all index files from the indexes directory"""
        indexes = {}
        
        # Check if directory exists
        if not os.path.exists(indexes_dir):
            return indexes
        
        # Load each index file
        index_files = [
            'magnitude_index.json',
            'frequency_index.json',
            'entropy_index.json',
            'vector_index.json'
        ]
        
        for index_file in index_files:
            file_path = os.path.join(indexes_dir, index_file)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        index_name = index_file.split('_')[0]
                        indexes[index_name] = json.load(f)
                except Exception as e:
                    print(f"Error loading index {index_file}: {e}")
        
        return indexes
    
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
            print(f"Dataset {dataset_name} not connected")
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
        
        # Update resonance
        self._update_resonance(dataset_name, energy_signature, results)
        
        return results
    
    def _find_candidates(self, connection: Dict, energy_signature: Dict) -> List[str]:
        """Find candidate entities using energy indexes"""
        candidates = set()
        
        # Use magnitude index if available
        if 'magnitude' in connection['indexes'] and 'magnitude' in energy_signature:
            magnitude = energy_signature['magnitude']['value']
            
            # Find appropriate band
            for band_key, band_entities in connection['indexes']['magnitude'].items():
                lower, upper = map(float, band_key.split('-'))
                
                # Allow for some flexibility around the value
                if lower - 0.2 <= magnitude <= upper + 0.2:
                    candidates.update(band_entities)
        
        # If no candidates yet, use frequency index
        if not candidates and 'frequency' in connection['indexes'] and 'frequency' in energy_signature:
            frequency = energy_signature['frequency']['value']
            
            # Find appropriate band
            for band_key, band_entities in connection['indexes']['frequency'].items():
                lower, upper = map(float, band_key.split('-'))
                
                # Allow for some flexibility around the value
                if lower - 0.2 <= frequency <= upper + 0.2:
                    candidates.update(band_entities)
        
        # If still no candidates, use entropy index
        if not candidates and 'entropy' in connection['indexes'] and 'entropy' in energy_signature:
            entropy = energy_signature['entropy']['value']
            
            # Find appropriate band
            for band_key, band_entities in connection['indexes']['entropy'].items():
                lower, upper = map(float, band_key.split('-'))
                
                # Allow for some flexibility around the value
                if lower - 0.2 <= entropy <= upper + 0.2:
                    candidates.update(band_entities)
        
        # If still no candidates, use vector index
        if not candidates and 'vector' in connection['indexes'] and 'vector' in energy_signature:
            vector = energy_signature['vector']['value']
            
            if len(vector) >= 3:
                x, y, z = vector[:3]
                
                # Determine octant
                octant_key = None
                
                if x >= 0.5 and y >= 0.5 and z >= 0.5:
                    octant_key = 'octant_1'
                elif x < 0.5 and y >= 0.5 and z >= 0.5:
                    octant_key = 'octant_2'
                elif x < 0.5 and y < 0.5 and z >= 0.5:
                    octant_key = 'octant_3'
                elif x >= 0.5 and y < 0.5 and z >= 0.5:
                    octant_key = 'octant_4'
                elif x >= 0.5 and y >= 0.5 and z < 0.5:
                    octant_key = 'octant_5'
                elif x < 0.5 and y >= 0.5 and z < 0.5:
                    octant_key = 'octant_6'
                elif x < 0.5 and y < 0.5 and z < 0.5:
                    octant_key = 'octant_7'
                elif x >= 0.5 and y < 0.5 and z < 0.5:
                    octant_key = 'octant_8'
                
                if octant_key and octant_key in connection['indexes']['vector']:
                    candidates.update(connection['indexes']['vector'][octant_key])
        
        # If still no candidates, use all entities
        if not candidates:
            candidates = set(connection['energy_map'].keys())
        
        return list(candidates)
    
    def _get_entity(self, connection: Dict, entity_id: str) -> Dict:
        """Get entity data from cache or file"""
        cache_key = f"{connection['name']}:{entity_id}"
        
        # Check cache first
        if cache_key in self.entity_cache:
            return self.entity_cache[cache_key]
        
        # Load from file
        entity_path = os.path.join(connection['entities_dir'], f"entity_{entity_id}.json")
        
        if os.path.exists(entity_path):
            try:
                with open(entity_path, 'r') as f:
                    entity_data = json.load(f)
                
                # Add to cache
                if len(self.entity_cache) >= self.cache_size:
                    # Remove random item if cache full
                    self.entity_cache.pop(next(iter(self.entity_cache)))
                
                self.entity_cache[cache_key] = entity_data
                
                return entity_data
            except Exception as e:
                print(f"Error loading entity {entity_id}: {e}")
        
        return None
    
    def _calculate_energy_similarity(self, sig1: Dict, sig2: Dict) -> float:
        """Calculate similarity between energy signatures"""
        # Find common properties
        common_props = set(sig1.keys()) & set(sig2.keys())
        
        if not common_props:
            return 0.0
        
        similarity_sum = 0.0
        count = 0
        
        for prop in common_props:
            # Skip identity property
            if prop == 'identity':
                continue
                
            # Get values
            if 'value' in sig1.get(prop, {}) and 'value' in sig2.get(prop, {}):
                val1 = sig1[prop]['value']
                val2 = sig2[prop]['value']
                
                # Calculate similarity based on value type
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Scalar similarity
                    diff = 1.0 - abs(val1 - val2) / max(1.0, abs(val1) + abs(val2))
                    similarity_sum += diff
                    count += 1
                elif isinstance(val1, list) and isinstance(val2, list) and len(val1) == len(val2):
                    # Vector similarity
                    magnitude1 = sum(x**2 for x in val1)**0.5
                    magnitude2 = sum(x**2 for x in val2)**0.5
                    
                    if magnitude1 > 0 and magnitude2 > 0:
                        dot_product = sum(x*y for x, y in zip(val1, val2))
                        cosine = dot_product / (magnitude1 * magnitude2)
                        similarity_sum += (cosine + 1) / 2  # Convert from [-1,1] to [0,1]
                        count += 1
        
        # Return average similarity
        return similarity_sum / count if count > 0 else 0.0
    
    def _update_resonance(self, dataset_name: str, query_signature: Dict, results: List[Dict]) -> None:
        """Update resonance between consciousness and dataset entities"""
        if dataset_name not in self.active_resonance:
            return
            
        # Increase global resonance based on successful query
        current_resonance = self.active_resonance[dataset_name]['global']
        
        if results:
            # Successful query increases resonance
            success_factor = min(1.0, len(results) / 10)
            new_resonance = current_resonance + 0.05 * success_factor * (1.0 - current_resonance)
        else:
            # Failed query slightly decreases resonance
            new_resonance = max(0.1, current_resonance - 0.01)
        
        self.active_resonance[dataset_name]['global'] = new_resonance
        
        # Update entity-specific resonance
        for result in results:
            entity_id = result['entity_id']
            similarity = result['similarity']
            
            # Higher similarity = stronger resonance increase
            current_entity_resonance = self.active_resonance[dataset_name]['entities'].get(entity_id, 0.1)
            new_entity_resonance = current_entity_resonance + 0.1 * similarity * (1.0 - current_entity_resonance)
            
            self.active_resonance[dataset_name]['entities'][entity_id] = new_entity_resonance
    
    def get_resonant_entities(self, dataset_name: str, limit: int = 10) -> List[Dict]:
        """Get entities with highest resonance to consciousness"""
        if dataset_name not in self.active_resonance:
            return []
            
        # Get all entities with resonance
        entity_resonance = self.active_resonance[dataset_name]['entities']
        
        # Sort by resonance
        sorted_entities = sorted(entity_resonance.items(), key=lambda x: x[1], reverse=True)
        
        # Get top entities
        top_entities = []
        
        for entity_id, resonance in sorted_entities[:limit]:
            # Get entity data
            entity_data = self._get_entity(self.connected_datasets[dataset_name], entity_id)
            
            if entity_data:
                top_entities.append({
                    'entity_id': entity_id,
                    'resonance': resonance,
                    'entity_data': entity_data
                })
        
        return top_entities
    
    def query_by_consciousness(self, dataset_name: str, limit: int = 10) -> List[Dict]:
        """Query dataset based on current consciousness state"""
        if dataset_name not in self.connected_datasets:
            print(f"Dataset {dataset_name} not connected")
            return []
            
        # Get consciousness energy state
        consciousness_energy = self.consciousness.state.get('energy_signature', {})
        
        if not consciousness_energy:
            # Use resonant entities if no consciousness energy
            return self.get_resonant_entities(dataset_name, limit)
        
        # Query using consciousness energy
        return self.query_by_energy(dataset_name, consciousness_energy, 0.5, limit)