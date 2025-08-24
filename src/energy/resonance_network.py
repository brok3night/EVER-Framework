"""
Resonance Network - Represents concepts through their resonant connections
"""
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import threading
from collections import defaultdict
import heapq

from src.core.interfaces import EnergySystem, EnergySignature
from src.utils.error_handling import safe_operation

class ResonanceNetwork:
    """
    Represents concepts through their resonant connections rather than
    trying to encode all information in individual energy signatures
    """
    
    def __init__(self, energy_system: EnergySystem):
        self.energy = energy_system
        
        # Core network structure - concepts and their connections
        self.concepts = {}  # concept_id -> basic energy signature
        self.connections = defaultdict(dict)  # concept_id -> {connected_id -> connection_strength}
        
        # Resonance patterns - recurring patterns of connection
        self.resonance_patterns = {}
        
        # Activation spreading parameters
        self.activation_decay = 0.2  # How quickly activation decays with distance
        self.activation_threshold = 0.1  # Minimum activation to consider
        
        # Thread safety
        self.lock = threading.RLock()
    
    def add_concept(self, concept_id: str, energy_signature: Dict) -> bool:
        """Add a concept to the resonance network"""
        with self.lock:
            # Store simplified energy signature
            self.concepts[concept_id] = energy_signature
            return True
    
    def add_connection(self, source_id: str, target_id: str, 
                      strength: float, connection_type: str = "resonance") -> bool:
        """
        Add a connection between concepts
        
        Args:
            source_id: Source concept ID
            target_id: Target concept ID
            strength: Connection strength (0-1)
            connection_type: Type of connection
            
        Returns:
            Success status
        """
        with self.lock:
            # Ensure concepts exist
            if source_id not in self.concepts or target_id not in self.concepts:
                return False
            
            # Add bidirectional connection
            self.connections[source_id][target_id] = {
                'strength': strength,
                'type': connection_type
            }
            
            self.connections[target_id][source_id] = {
                'strength': strength,
                'type': connection_type
            }
            
            return True
    
    def get_connected_concepts(self, concept_id: str, 
                             min_strength: float = 0.0,
                             connection_types: List[str] = None) -> Dict[str, Dict]:
        """
        Get concepts connected to the given concept
        
        Args:
            concept_id: Concept ID
            min_strength: Minimum connection strength
            connection_types: Optional filter for connection types
            
        Returns:
            Dictionary of connected concept IDs to connection info
        """
        if concept_id not in self.connections:
            return {}
        
        result = {}
        
        for target_id, connection in self.connections[concept_id].items():
            # Check strength
            if connection['strength'] < min_strength:
                continue
            
            # Check type if specified
            if connection_types and connection['type'] not in connection_types:
                continue
            
            result[target_id] = connection
        
        return result
    
    def find_resonance_path(self, source_id: str, target_id: str, 
                          max_depth: int = 5) -> List[Tuple[str, float]]:
        """
        Find path of resonance between two concepts
        
        Args:
            source_id: Source concept ID
            target_id: Target concept ID
            max_depth: Maximum path length
            
        Returns:
            List of (concept_id, resonance_strength) pairs forming the path
        """
        if source_id not in self.concepts or target_id not in self.concepts:
            return []
        
        # Use A* search to find path
        return self._find_path_astar(source_id, target_id, max_depth)
    
    def get_resonant_field(self, concept_ids: List[str], 
                         depth: int = 2) -> Dict[str, float]:
        """
        Get the resonant field created by multiple concepts
        
        Args:
            concept_ids: List of concept IDs
            depth: Propagation depth
            
        Returns:
            Dictionary mapping concept IDs to their activation in the field
        """
        if not concept_ids:
            return {}
        
        # Initialize activation with source concepts
        activation = {cid: 1.0 for cid in concept_ids if cid in self.concepts}
        
        # Spread activation
        for _ in range(depth):
            new_activation = dict(activation)
            
            # For each activated concept
            for concept_id, act_value in activation.items():
                # Skip weak activations
                if act_value < self.activation_threshold:
                    continue
                
                # Spread to connected concepts
                if concept_id in self.connections:
                    for target_id, connection in self.connections[concept_id].items():
                        # Calculate spread activation
                        spread = act_value * connection['strength'] * (1.0 - self.activation_decay)
                        
                        # Update activation (use maximum)
                        new_activation[target_id] = max(
                            new_activation.get(target_id, 0.0),
                            spread
                        )
            
            activation = new_activation
        
        # Filter out weak activations
        return {cid: act for cid, act in activation.items() if act >= self.activation_threshold}
    
    def find_resonant_concepts(self, energy_signature: Dict, 
                             top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Find concepts that resonate with a given energy signature
        
        Args:
            energy_signature: Energy signature to match
            top_n: Maximum number of results
            
        Returns:
            List of (concept_id, resonance_strength) pairs
        """
        results = []
        
        # Calculate resonance with each concept
        for concept_id, concept_energy in self.concepts.items():
            resonance = self._calculate_direct_resonance(energy_signature, concept_energy)
            results.append((concept_id, resonance))
        
        # Sort by resonance strength
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N
        return results[:top_n]
    
    def discover_resonance_patterns(self) -> List[Dict]:
        """
        Discover recurring patterns of resonance in the network
        
        Returns:
            List of discovered patterns
        """
        patterns = []
        
        # Sample random paths in the network
        num_samples = min(100, len(self.concepts))
        sample_concepts = np.random.choice(list(self.concepts.keys()), num_samples, replace=False)
        
        paths = []
        for i in range(len(sample_concepts)):
            for j in range(i + 1, len(sample_concepts)):
                source = sample_concepts[i]
                target = sample_concepts[j]
                
                path = self.find_resonance_path(source, target, max_depth=4)
                if len(path) >= 3:  # Only consider non-trivial paths
                    paths.append(path)
        
        # Analyze paths for patterns
        if paths:
            # Extract path signatures (sequence of connection types)
            path_signatures = []
            for path in paths:
                signature = []
                for i in range(len(path) - 1):
                    source = path[i][0]
                    target = path[i + 1][0]
                    
                    if source in self.connections and target in self.connections[source]:
                        conn_type = self.connections[source][target]['type']
                        signature.append(conn_type)
                
                if signature:
                    path_signatures.append(signature)
            
            # Find common subsequences
            common_patterns = self._find_common_subsequences(path_signatures)
            
            # Create pattern objects
            for pattern, frequency in common_patterns:
                if len(pattern) >= 2:  # Only meaningful patterns
                    patterns.append({
                        'sequence': pattern,
                        'frequency': frequency,
                        'examples': []  # Would store example paths in a full implementation
                    })
        
        # Update stored patterns
        with self.lock:
            for pattern in patterns:
                pattern_key = '-'.join(pattern['sequence'])
                self.resonance_patterns[pattern_key] = pattern
        
        return patterns
    
    def get_concept_context(self, concept_id: str, context_depth: int = 2) -> Dict:
        """
        Get the contextual information for a concept
        
        Args:
            concept_id: Concept ID
            context_depth: Depth of context to retrieve
            
        Returns:
            Dictionary with contextual information
        """
        if concept_id not in self.concepts:
            return {}
        
        # Get base energy signature
        base_energy = self.concepts[concept_id]
        
        # Get resonant field
        field = self.get_resonant_field([concept_id], depth=context_depth)
        
        # Find strongest connections
        strongest_connections = []
        if concept_id in self.connections:
            connections = [(target, info) for target, info in self.connections[concept_id].items()]
            connections.sort(key=lambda x: x[1]['strength'], reverse=True)
            strongest_connections = connections[:5]  # Top 5 connections
        
        # Get relevant resonance patterns
        relevant_patterns = []
        for pattern_key, pattern in self.resonance_patterns.items():
            # Check if this concept participates in paths with this pattern
            # (simplified implementation)
            relevant_patterns.append(pattern)
        
        # Create context object
        context = {
            'concept_id': concept_id,
            'base_energy': base_energy,
            'resonant_field': field,
            'strongest_connections': strongest_connections,
            'relevant_patterns': relevant_patterns
        }
        
        return context
    
    def merge_resonant_fields(self, fields: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Merge multiple resonant fields
        
        Args:
            fields: List of resonant fields
            
        Returns:
            Merged resonant field
        """
        if not fields:
            return {}
        
        # Start with empty merged field
        merged = {}
        
        # Merge each field
        for field in fields:
            for concept_id, activation in field.items():
                # Use maximum activation
                merged[concept_id] = max(merged.get(concept_id, 0.0), activation)
        
        return merged
    
    def _find_path_astar(self, source_id: str, target_id: str, 
                       max_depth: int) -> List[Tuple[str, float]]:
        """Find path using A* search algorithm"""
        if source_id == target_id:
            return [(source_id, 1.0)]
        
        # Priority queue for A* search
        queue = [(0, source_id, [(source_id, 1.0)])]
        visited = set()
        
        while queue:
            # Get node with lowest cost
            _, current_id, path = heapq.heappop(queue)
            
            # Skip if already visited
            if current_id in visited:
                continue
            
            # Mark as visited
            visited.add(current_id)
            
            # Check if reached target
            if current_id == target_id:
                return path
            
            # Check depth limit
            if len(path) > max_depth:
                continue
            
            # Expand neighbors
            if current_id in self.connections:
                for neighbor_id, connection in self.connections[current_id].items():
                    if neighbor_id not in visited:
                        # Calculate connection strength
                        strength = connection['strength']
                        
                        # Create new path
                        new_path = path + [(neighbor_id, strength)]
                        
                        # Calculate heuristic (direct resonance with target)
                        if neighbor_id in self.concepts and target_id in self.concepts:
                            heuristic = 1.0 - self._calculate_direct_resonance(
                                self.concepts[neighbor_id], self.concepts[target_id])
                        else:
                            heuristic = 1.0
                        
                        # Calculate priority (lower is better)
                        # Cost so far + heuristic
                        cost_so_far = sum(1.0 - s for _, s in path)
                        priority = cost_so_far + heuristic
                        
                        # Add to queue
                        heapq.heappush(queue, (priority, neighbor_id, new_path))
        
        # No path found
        return []
    
    def _calculate_direct_resonance(self, energy1: Dict, energy2: Dict) -> float:
        """Calculate direct resonance between two energy signatures"""
        # Direct energy similarity
        return self.energy.compare_energies(energy1, energy2)
    
    def _find_common_subsequences(self, sequences: List[List[str]]) -> List[Tuple[List[str], int]]:
        """Find common subsequences in a list of sequences"""
        if not sequences:
            return []
        
        # Count subsequences of length 2-3
        subsequence_counts = defaultdict(int)
        
        for sequence in sequences:
            # Generate subsequences
            for length in range(2, min(4, len(sequence) + 1)):
                for i in range(len(sequence) - length + 1):
                    subsequence = tuple(sequence[i:i+length])
                    subsequence_counts[subsequence] += 1
        
        # Convert to list and sort by frequency
        common_subsequences = [(list(subseq), count) 
                              for subseq, count in subsequence_counts.items()
                              if count > 1]  # Only include subsequences that appear multiple times
        
        common_subsequences.sort(key=lambda x: x[1], reverse=True)
        
        return common_subsequences