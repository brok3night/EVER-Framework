"""
Dataset Connector - Allows EVER to dynamically link to and reason with external datasets
"""
import os
import json
import numpy as np
from typing import Dict, List, Any, Callable
import importlib.util

class DatasetConnector:
    def __init__(self, consciousness_core):
        self.consciousness = consciousness_core
        
        # Registry of connected datasets
        self.connected_datasets = {}
        
        # Dataset connectors for different formats
        self.format_handlers = {
            'json': self._handle_json_dataset,
            'csv': self._handle_csv_dataset,
            'sql': self._handle_sql_dataset,
            'api': self._handle_api_dataset,
            'graph': self._handle_graph_dataset
        }
        
        # Energy mapping functions
        self.energy_mappers = {}
    
    def connect_dataset(self, name: str, location: str, format_type: str, 
                        schema: Dict = None, access_params: Dict = None) -> bool:
        """
        Connect EVER to an external dataset
        
        Args:
            name: Identifier for the dataset
            location: Path or URL to the dataset
            format_type: Data format (json, csv, sql, api, graph)
            schema: Optional schema information
            access_params: Optional parameters for authentication, etc.
        """
        if format_type not in self.format_handlers:
            print(f"Unsupported format: {format_type}")
            return False
        
        # Initialize connection
        connection = {
            'name': name,
            'location': location,
            'format': format_type,
            'schema': schema or {},
            'access_params': access_params or {},
            'status': 'disconnected',
            'energy_mappings': {}
        }
        
        # Attempt to connect
        try:
            # Call appropriate handler
            handler = self.format_handlers[format_type]
            connection = handler(connection)
            
            if connection['status'] == 'connected':
                # Register the connected dataset
                self.connected_datasets[name] = connection
                print(f"Successfully connected to dataset: {name}")
                return True
            else:
                print(f"Failed to connect to dataset: {name}")
                return False
                
        except Exception as e:
            print(f"Error connecting to dataset {name}: {e}")
            return False
    
    def _handle_json_dataset(self, connection: Dict) -> Dict:
        """Handle connection to JSON dataset"""
        try:
            location = connection['location']
            
            # Check if file exists for local files
            if os.path.exists(location):
                # Load the JSON file to verify it's valid
                with open(location, 'r') as f:
                    data = json.load(f)
                
                # Analyze structure
                connection['structure'] = self._analyze_json_structure(data)
                connection['entry_count'] = self._count_entries(data)
                connection['status'] = 'connected'
                connection['data_accessor'] = lambda query: self._query_json(location, query)
                
            # For URLs, just verify the location format for now
            elif location.startswith(('http://', 'https://')):
                connection['status'] = 'connected'
                connection['data_accessor'] = lambda query: self._query_json_api(location, query, 
                                                                              connection['access_params'])
            else:
                connection['status'] = 'error'
                connection['error'] = f"Invalid JSON location: {location}"
            
            return connection
        except Exception as e:
            connection['status'] = 'error'
            connection['error'] = str(e)
            return connection
    
    def _handle_csv_dataset(self, connection: Dict) -> Dict:
        """Handle connection to CSV dataset"""
        # Implementation would depend on whether we're using pandas or another CSV handler
        connection['status'] = 'connected'
        connection['data_accessor'] = lambda query: self._query_csv(connection['location'], query)
        return connection
    
    def _handle_sql_dataset(self, connection: Dict) -> Dict:
        """Handle connection to SQL database"""
        # Would implement connection to SQL databases
        connection['status'] = 'connected'
        connection['data_accessor'] = lambda query: self._query_sql(connection, query)
        return connection
    
    def _handle_api_dataset(self, connection: Dict) -> Dict:
        """Handle connection to API dataset"""
        # Would implement connection to REST APIs
        connection['status'] = 'connected' 
        connection['data_accessor'] = lambda query: self._query_api(connection, query)
        return connection
    
    def _handle_graph_dataset(self, connection: Dict) -> Dict:
        """Handle connection to graph database"""
        # Would implement connection to graph databases like Neo4j
        connection['status'] = 'connected'
        connection['data_accessor'] = lambda query: self._query_graph(connection, query)
        return connection
    
    def _analyze_json_structure(self, data: Any) -> Dict:
        """Analyze the structure of JSON data"""
        if isinstance(data, dict):
            return {
                'type': 'object',
                'keys': list(data.keys()),
                'sample_values': {k: type(v).__name__ for k, v in list(data.items())[:5]}
            }
        elif isinstance(data, list):
            if data:
                if isinstance(data[0], dict):
                    # Get common keys
                    common_keys = set(data[0].keys())
                    for item in data[1:5]:
                        if isinstance(item, dict):
                            common_keys &= set(item.keys())
                    
                    return {
                        'type': 'array_of_objects',
                        'length': len(data),
                        'common_keys': list(common_keys)
                    }
                else:
                    return {
                        'type': 'array',
                        'length': len(data),
                        'element_type': type(data[0]).__name__
                    }
            else:
                return {
                    'type': 'array',
                    'length': 0
                }
        else:
            return {
                'type': type(data).__name__
            }
    
    def _count_entries(self, data: Any) -> int:
        """Count the number of entries in a dataset"""
        if isinstance(data, list):
            return len(data)
        elif isinstance(data, dict):
            return 1
        else:
            return 0
    
    def _query_json(self, location: str, query: Dict) -> List[Dict]:
        """Query a local JSON file"""
        try:
            with open(location, 'r') as f:
                data = json.load(f)
            
            return self._filter_json(data, query)
        except Exception as e:
            print(f"Error querying JSON: {e}")
            return []
    
    def _query_json_api(self, url: str, query: Dict, access_params: Dict) -> List[Dict]:
        """Query a JSON API"""
        # Would implement API querying
        return []
    
    def _filter_json(self, data: Any, query: Dict) -> List[Dict]:
        """Filter JSON data based on query"""
        results = []
        
        # Handle list of objects
        if isinstance(data, list):
            for item in data:
                if self._matches_query(item, query):
                    results.append(item)
        # Handle single object
        elif isinstance(data, dict):
            if self._matches_query(data, query):
                results.append(data)
        
        return results
    
    def _matches_query(self, item: Dict, query: Dict) -> bool:
        """Check if an item matches a query"""
        if not isinstance(item, dict) or not query:
            return True
        
        for key, value in query.items():
            # Handle nested keys with dot notation
            if '.' in key:
                parts = key.split('.')
                current = item
                for part in parts[:-1]:
                    if part not in current:
                        return False
                    current = current[part]
                
                if parts[-1] not in current or current[parts[-1]] != value:
                    return False
            # Handle direct key match
            elif key not in item or item[key] != value:
                return False
        
        return True
    
    def _query_csv(self, location: str, query: Dict) -> List[Dict]:
        """Query a CSV file"""
        # Would implement CSV querying
        return []
    
    def _query_sql(self, connection: Dict, query: Dict) -> List[Dict]:
        """Query a SQL database"""
        # Would implement SQL querying
        return []
    
    def _query_api(self, connection: Dict, query: Dict) -> List[Dict]:
        """Query an API"""
        # Would implement API querying
        return []
    
    def _query_graph(self, connection: Dict, query: Dict) -> List[Dict]:
        """Query a graph database"""
        # Would implement graph database querying
        return []
    
    def define_energy_mapping(self, dataset_name: str, mapping_function: Callable) -> bool:
        """
        Define how dataset elements map to energy signatures
        
        Args:
            dataset_name: Name of the dataset
            mapping_function: Function that converts dataset items to energy signatures
        """
        if dataset_name not in self.connected_datasets:
            print(f"Dataset {dataset_name} not connected")
            return False
        
        self.energy_mappers[dataset_name] = mapping_function
        return True
    
    def query_with_energy(self, dataset_name: str, energy_signature: Dict, 
                          similarity_threshold: float = 0.7) -> List[Dict]:
        """
        Query dataset using energy signature matching
        
        Args:
            dataset_name: Name of the dataset to query
            energy_signature: Energy signature to match
            similarity_threshold: Minimum similarity threshold
        """
        if dataset_name not in self.connected_datasets:
            print(f"Dataset {dataset_name} not connected")
            return []
        
        if dataset_name not in self.energy_mappers:
            print(f"No energy mapping defined for dataset {dataset_name}")
            return []
        
        # Get dataset accessor
        connection = self.connected_datasets[dataset_name]
        accessor = connection.get('data_accessor')
        
        if not accessor:
            print(f"No data accessor available for dataset {dataset_name}")
            return []
        
        # Get all data (could be optimized for large datasets)
        all_data = accessor({})
        
        # Apply energy mapping and filter by similarity
        results = []
        mapper = self.energy_mappers[dataset_name]
        
        for item in all_data:
            # Map item to energy signature
            item_energy = mapper(item)
            
            # Calculate similarity
            similarity = self._calculate_energy_similarity(energy_signature, item_energy)
            
            # Add to results if above threshold
            if similarity >= similarity_threshold:
                results.append({
                    'item': item,
                    'similarity': similarity,
                    'energy_signature': item_energy
                })
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results
    
    def _calculate_energy_similarity(self, sig1: Dict, sig2: Dict) -> float:
        """Calculate similarity between energy signatures"""
        # Find common keys
        common_keys = set(sig1.keys()) & set(sig2.keys())
        
        if not common_keys:
            return 0.0
        
        similarity_sum = 0.0
        
        for key in common_keys:
            # Skip non-numeric values or nested structures
            if isinstance(sig1[key], dict) and 'value' in sig1[key] and \
               isinstance(sig2[key], dict) and 'value' in sig2[key]:
                
                val1 = sig1[key]['value']
                val2 = sig2[key]['value']
                
                # Calculate similarity based on value type
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Scalar similarity
                    diff = 1.0 - abs(val1 - val2) / max(1.0, abs(val1) + abs(val2))
                    similarity_sum += diff
                elif isinstance(val1, list) and isinstance(val2, list) and len(val1) == len(val2):
                    # Vector similarity
                    magnitude1 = sum(x**2 for x in val1)**0.5
                    magnitude2 = sum(x**2 for x in val2)**0.5
                    
                    if magnitude1 > 0 and magnitude2 > 0:
                        dot_product = sum(x*y for x, y in zip(val1, val2))
                        cosine = dot_product / (magnitude1 * magnitude2)
                        similarity_sum += (cosine + 1) / 2  # Convert from [-1,1] to [0,1]
        
        # Average similarity across all common keys
        return similarity_sum / len(common_keys) if common_keys else 0.0

    def reason_across_datasets(self, question: str) -> Dict:
        """
        Apply EVER reasoning to answer questions across multiple datasets
        
        Args:
            question: Natural language question to answer
        """
        # Process question through consciousness to get energy signature
        question_energy = self.consciousness.energy.process_text(question, self.consciousness.state)
        
        # Identify relevant datasets based on question energy
        dataset_relevance = {}
        
        for dataset_name, connection in self.connected_datasets.items():
            # Calculate relevance based on dataset structure and question energy
            relevance = self._calculate_dataset_relevance(connection, question_energy)
            dataset_relevance[dataset_name] = relevance
        
        # Query relevant datasets
        results = {}
        for dataset_name, relevance in sorted(dataset_relevance.items(), key=lambda x: x[1], reverse=True):
            if relevance > 0.3:  # Only query datasets with sufficient relevance
                # Query dataset with energy signature
                dataset_results = self.query_with_energy(dataset_name, question_energy, 0.5)
                
                if dataset_results:
                    results[dataset_name] = dataset_results
        
        # Process results through consciousness to synthesize answer
        synthesis = self._synthesize_answer(question, results, question_energy)
        
        return {
            'question': question,
            'question_energy': question_energy,
            'dataset_relevance': dataset_relevance,
            'results': results,
            'synthesis': synthesis
        }
    
    def _calculate_dataset_relevance(self, connection: Dict, question_energy: Dict) -> float:
        """Calculate relevance of a dataset to a question"""
        # This would be implemented based on dataset structure and question energy
        # For now, a simplified implementation
        return 0.7  # Default relevance
    
    def _synthesize_answer(self, question: str, results: Dict, question_energy: Dict) -> Dict:
        """Synthesize an answer from multiple dataset results"""
        # Process through consciousness
        synthesis_result = self.consciousness.process({
            'type': 'dataset_synthesis',
            'question': question,
            'results': results,
            'question_energy': question_energy
        })
        
        # Generate answer text based on consciousness processing
        combined_items = []
        for dataset_name, dataset_results in results.items():
            for result in dataset_results:
                combined_items.append({
                    'dataset': dataset_name,
                    'item': result['item'],
                    'similarity': result['similarity']
                })
        
        # Sort by similarity
        combined_items.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Generate answer
        answer = "Based on the available data, "
        if combined_items:
            # This would be replaced with actual reasoning
            answer += f"I found {len(combined_items)} relevant items across {len(results)} datasets."
        else:
            answer += "I couldn't find relevant information in the connected datasets."
        
        return {
            'answer_text': answer,
            'consciousness_state': synthesis_result.get('consciousness_state', {}),
            'supporting_items': combined_items[:5]  # Top 5 supporting items
        }