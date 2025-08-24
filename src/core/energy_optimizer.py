"""
Energy Optimization Engine - Optimizes energy signature processing
"""
import numpy as np
from typing import Dict, List, Set, Callable
import time

class EnergyOptimizer:
    """Optimizes energy signature processing for efficiency"""
    
    def __init__(self):
        # Caching system
        self.transformation_cache = {}
        self.similarity_cache = {}
        
        # Operation statistics
        self.operation_stats = {
            'similarity_calls': 0,
            'transformation_calls': 0,
            'cache_hits': 0,
            'execution_times': []
        }
        
        # Approximate nearest neighbor index for signatures
        self.signature_index = {}
        self.vector_dimensions = 3  # Default vector dimensions
        
        # Processing optimizations
        self.optimizations = {
            'vectorized_processing': True,
            'similarity_threshold': 0.7,
            'cache_enabled': True,
            'quantization_enabled': False,
            'quantization_levels': 10
        }
    
    def optimize_signature(self, signature: Dict) -> Dict:
        """
        Optimize an energy signature for efficient processing
        
        Args:
            signature: Energy signature to optimize
        """
        # Start timing
        start_time = time.time()
        
        # Make a copy to avoid modifying original
        optimized = signature.copy()
        
        # Apply optimizations
        if self.optimizations['quantization_enabled']:
            optimized = self._quantize_signature(optimized)
        
        # Record execution time
        self.operation_stats['execution_times'].append(time.time() - start_time)
        
        return optimized
    
    def batch_calculate_similarity(self, query_signature: Dict, 
                                  candidate_signatures: List[Dict]) -> List[float]:
        """
        Calculate similarity between query and multiple candidates efficiently
        
        Args:
            query_signature: Query energy signature
            candidate_signatures: List of candidate signatures to compare
        """
        # Start timing
        start_time = time.time()
        
        # Update stats
        self.operation_stats['similarity_calls'] += len(candidate_signatures)
        
        # Check cache if enabled
        if self.optimizations['cache_enabled']:
            similarities = []
            cache_hits = 0
            
            for candidate in candidate_signatures:
                cache_key = self._create_similarity_cache_key(query_signature, candidate)
                
                if cache_key in self.similarity_cache:
                    similarities.append(self.similarity_cache[cache_key])
                    cache_hits += 1
                else:
                    # Calculate and cache
                    similarity = self._calculate_single_similarity(query_signature, candidate)
                    self.similarity_cache[cache_key] = similarity
                    similarities.append(similarity)
            
            # Update stats
            self.operation_stats['cache_hits'] += cache_hits
            
            # Record execution time
            self.operation_stats['execution_times'].append(time.time() - start_time)
            
            return similarities
        
        # For vectorized processing
        if self.optimizations['vectorized_processing'] and all('vector' in sig for sig in candidate_signatures):
            # Extract vectors
            query_vector = np.array(query_signature.get('vector', {}).get('value', [0.5, 0.5, 0.5]))
            candidate_vectors = [
                np.array(sig.get('vector', {}).get('value', [0.5, 0.5, 0.5]))
                for sig in candidate_signatures
            ]
            
            # Vectorized cosine similarity
            similarities = self._batch_cosine_similarity(query_vector, candidate_vectors)
            
            # Record execution time
            self.operation_stats['execution_times'].append(time.time() - start_time)
            
            return similarities
        
        # Fallback to individual calculations
        similarities = [self._calculate_single_similarity(query_signature, candidate) 
                       for candidate in candidate_signatures]
        
        # Record execution time
        self.operation_stats['execution_times'].append(time.time() - start_time)
        
        return similarities
    
    def find_similar_signatures(self, query_signature: Dict, 
                               all_signatures: Dict[str, Dict],
                               top_k: int = 10) -> List[str]:
        """
        Find most similar signatures efficiently
        
        Args:
            query_signature: Query energy signature
            all_signatures: Dictionary mapping IDs to signatures
            top_k: Number of top matches to return
        """
        # Start timing
        start_time = time.time()
        
        # Ensure vector index is built
        if not self.signature_index:
            self._build_vector_index(all_signatures)
        
        # Extract query vector
        query_vector = np.array(query_signature.get('vector', {}).get('value', [0.5, 0.5, 0.5]))
        
        # Find candidate signatures using vector index
        candidates = self._find_candidates_from_index(query_vector)
        
        # If we have too few candidates, use all signatures
        if len(candidates) < top_k * 2:
            candidates = list(all_signatures.keys())
        
        # Get candidate signatures
        candidate_signatures = {key: all_signatures[key] for key in candidates if key in all_signatures}
        
        # Calculate similarities
        similarities = self.batch_calculate_similarity(
            query_signature, list(candidate_signatures.values()))
        
        # Create (id, similarity) pairs
        id_sim_pairs = list(zip(candidate_signatures.keys(), similarities))
        
        # Sort by similarity
        id_sim_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k IDs
        top_ids = [id for id, sim in id_sim_pairs[:top_k]]
        
        # Record execution time
        self.operation_stats['execution_times'].append(time.time() - start_time)
        
        return top_ids
    
    def batch_transform_signatures(self, signatures: List[Dict], 
                                  transform_function: Callable[[Dict], Dict]) -> List[Dict]:
        """
        Apply a transformation to multiple signatures efficiently
        
        Args:
            signatures: List of signatures to transform
            transform_function: Function to apply to each signature
        """
        # Start timing
        start_time = time.time()
        
        # Update stats
        self.operation_stats['transformation_calls'] += len(signatures)
        
        # Apply transformation to each signature
        transformed = [transform_function(sig) for sig in signatures]
        
        # Record execution time
        self.operation_stats['execution_times'].append(time.time() - start_time)
        
        return transformed
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        stats = self.operation_stats.copy()
        
        # Calculate average execution time
        if self.operation_stats['execution_times']:
            stats['avg_execution_time'] = np.mean(self.operation_stats['execution_times'])
        else:
            stats['avg_execution_time'] = 0
        
        # Calculate cache hit rate
        total_calls = stats['similarity_calls'] + stats['transformation_calls']
        if total_calls > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / total_calls
        else:
            stats['cache_hit_rate'] = 0
        
        return stats
    
    def _quantize_signature(self, signature: Dict) -> Dict:
        """Quantize signature values for efficiency"""
        quantized = signature.copy()
        levels = self.optimizations['quantization_levels']
        
        for key, value in signature.items():
            if isinstance(value, dict) and 'value' in value:
                inner_value = value['value']
                
                if isinstance(inner_value, (int, float)):
                    # Quantize scalar to discrete levels
                    quantized_value = round(inner_value * levels) / levels
                    quantized[key] = {'value': quantized_value}
                elif isinstance(inner_value, list):
                    # Quantize each vector component
                    quantized_vector = [round(v * levels) / levels for v in inner_value]
                    quantized[key] = {'value': quantized_vector}
        
        return quantized
    
    def _calculate_single_similarity(self, sig1: Dict, sig2: Dict) -> float:
        """Calculate similarity between two signatures"""
        similarity_sum = 0.0
        weight_sum = 0.0
        
        # Property weights
        weights = {
            'magnitude': 1.0,
            'frequency': 0.8,
            'entropy': 0.7,
            'vector': 1.5  # Vector gets higher weight
        }
        
        # Calculate similarity for each property
        for prop, weight in weights.items():
            if prop in sig1 and prop in sig2:
                val1 = sig1[prop].get('value')
                val2 = sig2[prop].get('value')
                
                if val1 is not None and val2 is not None:
                    # Calculate property similarity
                    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                        # Scalar similarity
                        prop_sim = 1.0 - min(1.0, abs(val1 - val2))
                        similarity_sum += prop_sim * weight
                        weight_sum += weight
                    elif isinstance(val1, list) and isinstance(val2, list):
                        # Vector similarity
                        if len(val1) == len(val2):
                            # Convert to numpy arrays
                            v1 = np.array(val1)
                            v2 = np.array(val2)
                            
                            # Calculate cosine similarity
                            norm1 = np.linalg.norm(v1)
                            norm2 = np.linalg.norm(v2)
                            
                            if norm1 > 0 and norm2 > 0:
                                dot = np.dot(v1, v2)
                                cosine = dot / (norm1 * norm2)
                                # Convert from [-1,1] to [0,1]
                                prop_sim = (cosine + 1) / 2
                                similarity_sum += prop_sim * weight
                                weight_sum += weight
        
        # Return weighted average
        if weight_sum > 0:
            return similarity_sum / weight_sum
        else:
            return 0.0
    
    def _batch_cosine_similarity(self, query_vector: np.ndarray, 
                                candidate_vectors: List[np.ndarray]) -> List[float]:
        """Calculate cosine similarity in batch"""
        # Ensure arrays have compatible dimensions
        max_dim = max([len(query_vector)] + [len(v) for v in candidate_vectors])
        
        # Pad query vector if needed
        if len(query_vector) < max_dim:
            query_vector = np.pad(query_vector, (0, max_dim - len(query_vector)), 
                                 mode='constant', constant_values=0.5)
        
        # Pad candidate vectors if needed
        padded_candidates = []
        for vec in candidate_vectors:
            if len(vec) < max_dim:
                padded_vec = np.pad(vec, (0, max_dim - len(vec)),
                                   mode='constant', constant_values=0.5)
                padded_candidates.append(padded_vec)
            else:
                padded_candidates.append(vec)
        
        # Stack candidates into a matrix
        candidate_matrix = np.vstack(padded_candidates)
        
        # Calculate query norm
        query_norm = np.linalg.norm(query_vector)
        
        # Calculate candidate norms
        candidate_norms = np.linalg.norm(candidate_matrix, axis=1)
        
        # Calculate dot products
        dot_products = np.dot(candidate_matrix, query_vector)
        
        # Calculate cosine similarities
        with np.errstate(divide='ignore', invalid='ignore'):
            cosine_similarities = dot_products / (candidate_norms * query_norm)
            cosine_similarities = np.nan_to_num(cosine_similarities, nan=0.0)
        
        # Convert from [-1,1] to [0,1]
        return ((cosine_similarities + 1) / 2).tolist()
    
    def _create_similarity_cache_key(self, sig1: Dict, sig2: Dict) -> str:
        """Create cache key for similarity calculation"""
        # Create simplified signature representations
        repr1 = self._signature_representation(sig1)
        repr2 = self._signature_representation(sig2)
        
        # Sort to ensure same key regardless of order
        if repr1 > repr2:
            repr1, repr2 = repr2, repr1
        
        return f"{repr1}:{repr2}"
    
    def _signature_representation(self, signature: Dict) -> str:
        """Create string representation of signature for caching"""
        # Extract key values for cache key
        repr_parts = []
        
        for prop in ['magnitude', 'frequency', 'entropy']:
            if prop in signature and 'value' in signature[prop]:
                value = signature[prop]['value']
                if isinstance(value, (int, float)):
                    # Round to reduce cache variants
                    repr_parts.append(f"{prop}:{round(value, 2)}")
        
        if 'vector' in signature and 'value' in signature['vector']:
            vector = signature['vector']['value']
            if isinstance(vector, list):
                # Use first 3 dimensions only
                vec_str = ','.join(str(round(v, 2)) for v in vector[:3])
                repr_parts.append(f"vec:[{vec_str}]")
        
        return "|".join(repr_parts)
    
    def _build_vector_index(self, signatures: Dict[str, Dict]) -> None:
        """Build approximate nearest neighbor index for vectors"""
        self.signature_index = {}
        
        # Determine vector dimensions
        dims = []
        for sig in signatures.values():
            if 'vector' in sig and 'value' in sig['vector']:
                vector = sig['vector']['value']
                if isinstance(vector, list):
                    dims.append(len(vector))
        
        if dims:
            self.vector_dimensions = max(dims)
        
        # Create grid cells for vectors
        grid_size = 4  # Number of divisions per dimension
        
        for sig_id, signature in signatures.items():
            if 'vector' in signature and 'value' in signature['vector']:
                vector = signature['vector']['value']
                
                if isinstance(vector, list):
                    # Ensure vector has required dimensions
                    if len(vector) < self.vector_dimensions:
                        vector = vector + [0.5] * (self.vector_dimensions - len(vector))
                    else:
                        vector = vector[:self.vector_dimensions]
                    
                    # Calculate grid cell
                    grid_cell = tuple(min(grid_size-1, int(v * grid_size)) for v in vector)
                    
                    # Add to index
                    if grid_cell not in self.signature_index:
                        self.signature_index[grid_cell] = []
                    
                    self.signature_index[grid_cell].append(sig_id)
    
    def _find_candidates_from_index(self, query_vector: np.ndarray) -> List[str]:
        """Find candidate signatures from vector index"""
        if not self.signature_index:
            return []
        
        # Ensure vector has required dimensions
        if len(query_vector) < self.vector_dimensions:
            query_vector = np.concatenate([query_vector, 
                                         np.ones(self.vector_dimensions - len(query_vector)) * 0.5])
        else:
            query_vector = query_vector[:self.vector_dimensions]
        
        grid_size = 4  # Same as in build_vector_index
        
        # Calculate query grid cell
        query_cell = tuple(min(grid_size-1, int(v * grid_size)) for v in query_vector)
        
        # Get candidates from this cell and neighbors
        candidates = []
        
        # Add candidates from query cell
        if query_cell in self.signature_index:
            candidates.extend(self.signature_index[query_cell])
        
        # Add candidates from neighboring cells
        for i in range(len(query_cell)):
            for offset in [-1, 1]:
                neighbor_cell = list(query_cell)
                neighbor_cell[i] = max(0, min(grid_size-1, neighbor_cell[i] + offset))
                neighbor_cell = tuple(neighbor_cell)
                
                if neighbor_cell in self.signature_index:
                    candidates.extend(self.signature_index[neighbor_cell])
        
        return list(set(candidates))  # Remove duplicates