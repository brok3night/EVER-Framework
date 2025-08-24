"""
EVER-Native Data Format (ENDF) Converter
Transforms various data sources into EVER's native understanding format
"""
import os
import json
import numpy as np
import hashlib
from typing import Dict, List, Any, Union, Tuple
import multiprocessing as mp

class ENDFConverter:
    def __init__(self, output_dir: str, parallel: bool = True):
        self.output_dir = output_dir
        self.parallel = parallel
        self.processors = max(1, mp.cpu_count() - 1) if parallel else 1
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # ENDF format version
        self.format_version = "1.0.0"
        
        # Track conversion metadata
        self.conversion_stats = {}
    
    def convert_dataset(self, dataset_path: str, format_type: str, 
                        dataset_name: str = None, schema: Dict = None) -> str:
        """
        Convert a dataset to ENDF format
        
        Returns:
            Path to the converted ENDF directory
        """
        # Set dataset name if not provided
        if not dataset_name:
            dataset_name = os.path.basename(dataset_path).split('.')[0]
        
        # Create output directory for this dataset
        endf_dir = os.path.join(self.output_dir, f"{dataset_name}_endf")
        os.makedirs(endf_dir, exist_ok=True)
        
        print(f"Converting {dataset_path} to ENDF format in {endf_dir}")
        
        # Load data based on format type
        if format_type == 'json':
            data = self._load_json(dataset_path)
        elif format_type == 'csv':
            data = self._load_csv(dataset_path)
        elif format_type == 'xml':
            data = self._load_xml(dataset_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        # Analyze dataset structure
        structure = self._analyze_structure(data, format_type)
        
        # Create metadata file
        metadata = {
            'format_version': self.format_version,
            'dataset_name': dataset_name,
            'original_format': format_type,
            'original_path': dataset_path,
            'structure': structure,
            'schema': schema or {},
            'entry_count': self._count_entries(data),
            'creation_timestamp': np.datetime64('now').astype(str),
            'energy_spectrum': {}  # Will be filled during conversion
        }
        
        # Begin conversion process
        if isinstance(data, list):
            # Process list of items
            energy_map, entity_files = self._process_entity_list(data, endf_dir)
        elif isinstance(data, dict):
            # Process dictionary structure
            energy_map, entity_files = self._process_entity_dict(data, endf_dir)
        else:
            raise ValueError(f"Unsupported data structure: {type(data)}")
        
        # Update metadata with energy spectrum
        metadata['energy_spectrum'] = self._calculate_energy_spectrum(energy_map)
        metadata['entity_files'] = entity_files
        
        # Write metadata file
        with open(os.path.join(endf_dir, 'endf_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Write energy map file
        with open(os.path.join(endf_dir, 'energy_map.json'), 'w') as f:
            json.dump(energy_map, f, indent=2)
        
        # Create energy index files
        self._create_energy_indexes(energy_map, endf_dir)
        
        print(f"Conversion complete. ENDF dataset available at: {endf_dir}")
        return endf_dir
    
    def _load_json(self, path: str) -> Any:
        """Load data from JSON file"""
        with open(path, 'r') as f:
            return json.load(f)
    
    def _load_csv(self, path: str) -> List[Dict]:
        """Load data from CSV file"""
        import csv
        
        data = []
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        
        return data
    
    def _load_xml(self, path: str) -> Dict:
        """Load data from XML file"""
        import xmltodict
        
        with open(path, 'r') as f:
            return xmltodict.parse(f.read())
    
    def _analyze_structure(self, data: Any, format_type: str) -> Dict:
        """Analyze the structure of the dataset"""
        structure = {
            'type': type(data).__name__,
            'format': format_type
        }
        
        if isinstance(data, list) and data:
            # Sample first few items
            sample_items = data[:min(5, len(data))]
            
            if all(isinstance(item, dict) for item in sample_items):
                # List of dictionaries - analyze common keys
                all_keys = set()
                for item in sample_items:
                    all_keys.update(item.keys())
                
                # Find keys present in all items
                common_keys = set(sample_items[0].keys())
                for item in sample_items[1:]:
                    common_keys &= set(item.keys())
                
                structure['common_keys'] = list(common_keys)
                structure['all_keys'] = list(all_keys)
                structure['sample_types'] = {
                    key: type(sample_items[0][key]).__name__ for key in sample_items[0] if key in common_keys
                }
        
        elif isinstance(data, dict):
            # Dictionary structure
            structure['keys'] = list(data.keys())
            
            # Sample some values
            sample_values = {}
            for key in list(data.keys())[:5]:
                value = data[key]
                sample_values[key] = {
                    'type': type(value).__name__,
                    'is_list': isinstance(value, list),
                    'is_dict': isinstance(value, dict)
                }
            
            structure['sample_values'] = sample_values
        
        return structure
    
    def _count_entries(self, data: Any) -> int:
        """Count entries in the dataset"""
        if isinstance(data, list):
            return len(data)
        elif isinstance(data, dict):
            # Count recursive entries
            count = 1  # The dict itself
            for value in data.values():
                if isinstance(value, (list, dict)):
                    count += self._count_entries(value)
            return count
        else:
            return 1
    
    def _process_entity_list(self, entities: List[Dict], output_dir: str) -> Tuple[Dict, List[str]]:
        """Process a list of entities into ENDF format"""
        # Create entities directory
        entities_dir = os.path.join(output_dir, 'entities')
        os.makedirs(entities_dir, exist_ok=True)
        
        # Energy map to track all energy signatures
        energy_map = {}
        entity_files = []
        
        # Process in parallel if enabled
        if self.parallel and len(entities) > 1000:
            # Split into chunks for parallel processing
            chunk_size = max(100, len(entities) // self.processors)
            chunks = [entities[i:i+chunk_size] for i in range(0, len(entities), chunk_size)]
            
            # Create pool and process chunks
            with mp.Pool(processes=self.processors) as pool:
                results = pool.map(self._process_entity_chunk, 
                                   [(chunk, entities_dir, i) for i, chunk in enumerate(chunks)])
            
            # Merge results
            for chunk_map, chunk_files in results:
                energy_map.update(chunk_map)
                entity_files.extend(chunk_files)
        else:
            # Process sequentially
            for i, entity in enumerate(entities):
                # Generate entity ID if not present
                entity_id = entity.get('id', str(i))
                
                # Convert to ENDF format
                endf_entity = self._convert_to_endf(entity)
                
                # Save entity file
                entity_file = os.path.join(entities_dir, f"entity_{entity_id}.json")
                with open(entity_file, 'w') as f:
                    json.dump(endf_entity, f)
                
                # Add to energy map
                energy_map[entity_id] = endf_entity['energy_signature']
                entity_files.append(entity_file)
                
                # Log progress
                if (i+1) % 1000 == 0:
                    print(f"Processed {i+1}/{len(entities)} entities")
        
        return energy_map, entity_files
    
    def _process_entity_chunk(self, args: Tuple) -> Tuple[Dict, List[str]]:
        """Process a chunk of entities (for parallel processing)"""
        entities, entities_dir, chunk_id = args
        chunk_map = {}
        chunk_files = []
        
        for i, entity in enumerate(entities):
            # Generate entity ID if not present
            entity_id = entity.get('id', f"{chunk_id}_{i}")
            
            # Convert to ENDF format
            endf_entity = self._convert_to_endf(entity)
            
            # Save entity file
            entity_file = os.path.join(entities_dir, f"entity_{entity_id}.json")
            with open(entity_file, 'w') as f:
                json.dump(endf_entity, f)
            
            # Add to energy map
            chunk_map[entity_id] = endf_entity['energy_signature']
            chunk_files.append(entity_file)
        
        return chunk_map, chunk_files
    
    def _process_entity_dict(self, data: Dict, output_dir: str) -> Tuple[Dict, List[str]]:
        """Process a dictionary structure into ENDF format"""
        # Create entities directory
        entities_dir = os.path.join(output_dir, 'entities')
        os.makedirs(entities_dir, exist_ok=True)
        
        # Energy map to track all energy signatures
        energy_map = {}
        entity_files = []
        
        # Process each top-level key as an entity
        for key, value in data.items():
            # Create entity from key-value pair
            entity = {
                'id': key,
                'value': value
            }
            
            # Convert to ENDF format
            endf_entity = self._convert_to_endf(entity)
            
            # Save entity file
            entity_file = os.path.join(entities_dir, f"entity_{key}.json")
            with open(entity_file, 'w') as f:
                json.dump(endf_entity, f)
            
            # Add to energy map
            energy_map[key] = endf_entity['energy_signature']
            entity_files.append(entity_file)
        
        return energy_map, entity_files
    
    def _convert_to_endf(self, entity: Dict) -> Dict:
        """Convert an entity to ENDF format with energy signatures"""
        # Create ENDF entity structure
        endf_entity = {
            'original_data': entity,
            'energy_signature': self._generate_energy_signature(entity),
            'structural_properties': self._extract_structural_properties(entity),
            'connections': self._identify_connections(entity),
            'endf_version': self.format_version
        }
        
        return endf_entity
    
    def _generate_energy_signature(self, entity: Dict) -> Dict:
        """Generate energy signature for an entity"""
        # This would be a sophisticated algorithm based on EVER principles
        # For now, a simplified implementation
        
        # Get properties to analyze
        text_content = self._extract_text_content(entity)
        numeric_values = self._extract_numeric_values(entity)
        structure_complexity = self._calculate_structure_complexity(entity)
        
        # Generate base energy signature
        signature = {
            'magnitude': {
                'value': min(1.0, 0.3 + 0.1 * structure_complexity + 0.05 * len(numeric_values))
            },
            'frequency': {
                'value': min(1.0, 0.2 + 0.1 * len(text_content) / 100)
            },
            'duration': {
                'value': 0.5  # Neutral default
            },
            'vector': {
                'value': [
                    0.5,  # x component - neutral default
                    0.5,  # y component - neutral default
                    0.5   # z component - neutral default
                ]
            },
            'entropy': {
                'value': min(1.0, 0.3 + 0.1 * structure_complexity)
            },
            'boundary': {
                'value': [0.0, 1.0]  # Default full range
            },
            'identity': {
                'value': str(entity.get('id', self._generate_entity_hash(entity)))
            }
        }
        
        # Adjust vector based on content
        if text_content:
            # Simple sentiment-like analysis to affect y-component
            positive_words = ['good', 'great', 'excellent', 'positive', 'success']
            negative_words = ['bad', 'poor', 'negative', 'failure', 'wrong']
            
            text_lower = text_content.lower()
            positive_count = sum(text_lower.count(word) for word in positive_words)
            negative_count = sum(text_lower.count(word) for word in negative_words)
            
            if positive_count > negative_count:
                signature['vector']['value'][1] += min(0.3, 0.05 * positive_count)
            elif negative_count > positive_count:
                signature['vector']['value'][1] -= min(0.3, 0.05 * negative_count)
        
        # Adjust magnitude based on numeric values
        if numeric_values:
            avg_value = sum(numeric_values) / len(numeric_values)
            normalized_avg = min(1.0, abs(avg_value) / 100)  # Simple normalization
            signature['magnitude']['value'] = (signature['magnitude']['value'] + normalized_avg) / 2
        
        return signature
    
    def _extract_structural_properties(self, entity: Dict) -> Dict:
        """Extract structural properties from an entity"""
        properties = {
            'complexity': self._calculate_structure_complexity(entity),
            'connection_count': len(self._identify_connections(entity)),
            'attribute_count': self._count_attributes(entity)
        }
        
        # Check for temporal aspects
        if any(key in str(k).lower() for k in entity.keys() 
               for key in ['date', 'time', 'year', 'month', 'day']):
            properties['temporal'] = True
        
        # Check for spatial aspects
        if any(key in str(k).lower() for k in entity.keys()
               for key in ['location', 'place', 'coordinate', 'latitude', 'longitude']):
            properties['spatial'] = True
        
        # Check for categorical aspects
        if any(key in str(k).lower() for k in entity.keys()
               for key in ['category', 'type', 'class', 'group']):
            properties['categorical'] = True
        
        return properties
    
    def _identify_connections(self, entity: Dict) -> Dict:
        """Identify potential connections to other entities"""
        connections = {}
        
        # Look for ID references
        for key, value in self._iter_items(entity):
            if 'id' in str(key).lower() and key != 'id':
                connections[key] = value
            
            # Look for list of IDs
            if isinstance(value, list) and all(isinstance(item, (str, int)) for item in value):
                connections[key] = value
        
        return connections
    
    def _extract_text_content(self, entity: Dict) -> str:
        """Extract all text content from an entity"""
        text_content = []
        
        for key, value in self._iter_items(entity):
            if isinstance(value, str) and len(value) > 2:  # Skip very short strings
                text_content.append(value)
        
        return " ".join(text_content)
    
    def _extract_numeric_values(self, entity: Dict) -> List[float]:
        """Extract all numeric values from an entity"""
        values = []
        
        for key, value in self._iter_items(entity):
            if isinstance(value, (int, float)):
                values.append(float(value))
        
        return values
    
    def _calculate_structure_complexity(self, entity: Dict) -> float:
        """Calculate structural complexity of an entity"""
        if not isinstance(entity, dict):
            return 0.0
        
        # Count nested levels
        max_depth = 0
        
        def get_depth(obj, current_depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, current_depth)
            
            if isinstance(obj, dict):
                for value in obj.values():
                    get_depth(value, current_depth + 1)
            elif isinstance(obj, list) and obj:
                for item in obj:
                    get_depth(item, current_depth + 1)
        
        get_depth(entity)
        
        # Count total keys
        total_keys = sum(1 for _ in self._iter_items(entity))
        
        # Normalize complexity score between 0 and 1
        complexity = 0.1 * max_depth + 0.01 * total_keys
        return min(1.0, complexity)
    
    def _count_attributes(self, entity: Dict) -> int:
        """Count total attributes in an entity"""
        return sum(1 for _ in self._iter_items(entity))
    
    def _iter_items(self, obj, parent_key=''):
        """Recursively iterate through all items in a nested structure"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_key = f"{parent_key}.{key}" if parent_key else key
                yield current_key, value
                
                if isinstance(value, (dict, list)):
                    yield from self._iter_items(value, current_key)
                    
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                current_key = f"{parent_key}[{i}]"
                
                if isinstance(item, (dict, list)):
                    yield from self._iter_items(item, current_key)
                else:
                    yield current_key, item
    
    def _generate_entity_hash(self, entity: Dict) -> str:
        """Generate a hash identifier for an entity"""
        entity_str = json.dumps(entity, sort_keys=True)
        return hashlib.md5(entity_str.encode()).hexdigest()
    
    def _calculate_energy_spectrum(self, energy_map: Dict) -> Dict:
        """Calculate the energy spectrum across all entities"""
        spectrum = {
            'magnitude': {'min': 1.0, 'max': 0.0, 'avg': 0.0},
            'frequency': {'min': 1.0, 'max': 0.0, 'avg': 0.0},
            'entropy': {'min': 1.0, 'max': 0.0, 'avg': 0.0}
        }
        
        # Skip if no entities
        if not energy_map:
            return spectrum
        
        # Collect values
        magnitudes = []
        frequencies = []
        entropies = []
        
        for entity_id, signature in energy_map.items():
            if 'magnitude' in signature and 'value' in signature['magnitude']:
                magnitude = signature['magnitude']['value']
                magnitudes.append(magnitude)
                spectrum['magnitude']['min'] = min(spectrum['magnitude']['min'], magnitude)
                spectrum['magnitude']['max'] = max(spectrum['magnitude']['max'], magnitude)
            
            if 'frequency' in signature and 'value' in signature['frequency']:
                frequency = signature['frequency']['value']
                frequencies.append(frequency)
                spectrum['frequency']['min'] = min(spectrum['frequency']['min'], frequency)
                spectrum['frequency']['max'] = max(spectrum['frequency']['max'], frequency)
            
            if 'entropy' in signature and 'value' in signature['entropy']:
                entropy = signature['entropy']['value']
                entropies.append(entropy)
                spectrum['entropy']['min'] = min(spectrum['entropy']['min'], entropy)
                spectrum['entropy']['max'] = max(spectrum['entropy']['max'], entropy)
        
        # Calculate averages
        if magnitudes:
            spectrum['magnitude']['avg'] = sum(magnitudes) / len(magnitudes)
        
        if frequencies:
            spectrum['frequency']['avg'] = sum(frequencies) / len(frequencies)
        
        if entropies:
            spectrum['entropy']['avg'] = sum(entropies) / len(entropies)
        
        return spectrum
    
    def _create_energy_indexes(self, energy_map: Dict, output_dir: str) -> None:
        """Create energy-based index files for efficient access"""
        # Create indexes directory
        indexes_dir = os.path.join(output_dir, 'indexes')
        os.makedirs(indexes_dir, exist_ok=True)
        
        # Create magnitude index (10 bands)
        magnitude_index = self._create_property_bands(energy_map, 'magnitude', 10)
        with open(os.path.join(indexes_dir, 'magnitude_index.json'), 'w') as f:
            json.dump(magnitude_index, f, indent=2)
        
        # Create frequency index (10 bands)
        frequency_index = self._create_property_bands(energy_map, 'frequency', 10)
        with open(os.path.join(indexes_dir, 'frequency_index.json'), 'w') as f:
            json.dump(frequency_index, f, indent=2)
        
        # Create entropy index (10 bands)
        entropy_index = self._create_property_bands(energy_map, 'entropy', 10)
        with open(os.path.join(indexes_dir, 'entropy_index.json'), 'w') as f:
            json.dump(entropy_index, f, indent=2)
        
        # Create vector direction index (8 octants)
        vector_index = self._create_vector_octants(energy_map)
        with open(os.path.join(indexes_dir, 'vector_index.json'), 'w') as f:
            json.dump(vector_index, f, indent=2)
    
    def _create_property_bands(self, energy_map: Dict, property_name: str, band_count: int) -> Dict:
        """Create banded index for a property"""
        index = {}
        
        # Create empty bands
        for i in range(band_count):
            lower = i / band_count
            upper = (i + 1) / band_count
            band_key = f"{lower:.1f}-{upper:.1f}"
            index[band_key] = []
        
        # Assign entities to bands
        for entity_id, signature in energy_map.items():
            if property_name in signature and 'value' in signature[property_name]:
                value = signature[property_name]['value']
                
                # Find appropriate band
                band_idx = min(band_count - 1, int(value * band_count))
                lower = band_idx / band_count
                upper = (band_idx + 1) / band_count
                band_key = f"{lower:.1f}-{upper:.1f}"
                
                index[band_key].append(entity_id)
        
        return index
    
    def _create_vector_octants(self, energy_map: Dict) -> Dict:
        """Create octant-based index for vectors"""
        octants = {
            'octant_1': [],  # (+,+,+)
            'octant_2': [],  # (-,+,+)
            'octant_3': [],  # (-,-,+)
            'octant_4': [],  # (+,-,+)
            'octant_5': [],  # (+,+,-)
            'octant_6': [],  # (-,+,-)
            'octant_7': [],  # (-,-,-)
            'octant_8': []   # (+,-,-)
        }
        
        for entity_id, signature in energy_map.items():
            if 'vector' in signature and 'value' in signature['vector']:
                vector = signature['vector']['value']
                
                if len(vector) >= 3:
                    x, y, z = vector[:3]
                    
                    # Determine octant
                    if x >= 0.5 and y >= 0.5 and z >= 0.5:
                        octants['octant_1'].append(entity_id)
                    elif x < 0.5 and y >= 0.5 and z >= 0.5:
                        octants['octant_2'].append(entity_id)
                    elif x < 0.5 and y < 0.5 and z >= 0.5:
                        octants['octant_3'].append(entity_id)
                    elif x >= 0.5 and y < 0.5 and z >= 0.5:
                        octants['octant_4'].append(entity_id)
                    elif x >= 0.5 and y >= 0.5 and z < 0.5:
                        octants['octant_5'].append(entity_id)
                    elif x < 0.5 and y >= 0.5 and z < 0.5:
                        octants['octant_6'].append(entity_id)
                    elif x < 0.5 and y < 0.5 and z < 0.5:
                        octants['octant_7'].append(entity_id)
                    elif x >= 0.5 and y < 0.5 and z < 0.5:
                        octants['octant_8'].append(entity_id)
        
        return octants