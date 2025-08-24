"""
Hierarchical Navigation - Efficient navigation of large concept networks
"""
from typing import Dict, List, Set, Tuple, Any
import numpy as np
from collections import defaultdict

class HierarchicalNavigation:
    """Enables efficient navigation of large concept networks through hierarchical clustering"""
    
    def __init__(self, resonance_network):
        self.network = resonance_network
        
        # Concept clusters at different abstraction levels
        self.concept_clusters = {}
        
        # Cluster representatives (prototype concepts for each cluster)
        self.cluster_representatives = {}
        
        # Connections between clusters
        self.cluster_connections = defaultdict(dict)
        
        # Navigation heuristics
        self.navigation_heuristics = {
            'semantic_distance': self._heuristic_semantic_distance,
            'path_frequency': self._heuristic_path_frequency,
            'concept_importance': self._heuristic_concept_importance
        }
    
    def build_hierarchical_structure(self, levels: int = 3) -> None:
        """
        Build hierarchical clustering of the concept network
        
        Args:
            levels: Number of hierarchical levels
        """
        # Get all concepts
        all_concepts = list(self.network.concepts.keys())
        
        if not all_concepts:
            return
        
        # Build each level
        for level in range(levels):
            # Calculate number of clusters for this level
            # More clusters at lower levels, fewer at higher levels
            num_clusters = max(1, len(all_concepts) // (10 * (level + 1)))
            
            # Build clusters
            clusters = self._cluster_concepts(all_concepts, num_clusters)
            
            # Store clusters
            self.concept_clusters[level] = clusters
            
            # Find representatives for each cluster
            representatives = {}
            for cluster_id, concepts in clusters.items():
                rep = self._find_cluster_representative(concepts)
                representatives[cluster_id] = rep
            
            # Store representatives
            self.cluster_representatives[level] = representatives
            
            # Build connections between clusters
            self._build_cluster_connections(level)
    
    def find_efficient_path(self, source_id: str, target_id: str,
                          max_depth: int = 10) -> List[Tuple[str, float]]:
        """
        Find efficient path between concepts using hierarchical navigation
        
        Args:
            source_id: Source concept ID
            target_id: Target concept ID
            max_depth: Maximum path depth
            
        Returns:
            Path as list of (concept_id, connection_strength) pairs
        """
        # If concepts are directly connected or very close, use direct search
        direct_path = self._check_direct_path(source_id, target_id, 2)
        if direct_path:
            return direct_path
        
        # Otherwise, use hierarchical navigation
        return self._hierarchical_search(source_id, target_id, max_depth)
    
    def get_efficient_field(self, concept_ids: List[str], 
                          depth: int = 2) -> Dict[str, float]:
        """
        Get resonant field efficiently using hierarchical structure
        
        Args:
            concept_ids: Seed concept IDs
            depth: Activation spread depth
            
        Returns:
            Resonant field as dict of concept_id -> activation
        """
        # For small depth, use standard field calculation
        if depth <= 1:
            return self.network.get_resonant_field(concept_ids, depth)
        
        # For larger depth, use hierarchical approach
        field = {}
        
        # First, find clusters containing seed concepts
        active_clusters = self._find_containing_clusters(concept_ids)
        
        # Spread activation within and between clusters
        for level, cluster_ids in active_clusters.items():
            for cluster_id in cluster_ids:
                # Get concepts in this cluster
                cluster_concepts = self.concept_clusters[level].get(cluster_id, [])
                
                # Calculate activation for these concepts
                cluster_field = self._calculate_cluster_field(
                    concept_ids, cluster_concepts, depth
                )
                
                # Add to overall field
                for concept_id, activation in cluster_field.items():
                    field[concept_id] = max(field.get(concept_id, 0.0), activation)
                
                # Add connected clusters (with lower activation)
                if cluster_id in self.cluster_connections[level]:
                    for connected_id, strength in self.cluster_connections[level][cluster_id].items():
                        connected_concepts = self.concept_clusters[level].get(connected_id, [])
                        
                        # Calculate activation for connected cluster (reduced by connection strength)
                        connected_field = self._calculate_cluster_field(
                            concept_ids, connected_concepts, depth - 1
                        )
                        
                        # Add to overall field with reduced activation
                        for concept_id, activation in connected_field.items():
                            reduced_activation = activation * strength
                            field[concept_id] = max(field.get(concept_id, 0.0), reduced_activation)
        
        return field
    
    def _cluster_concepts(self, concepts: List[str], 
                        num_clusters: int) -> Dict[int, List[str]]:
        """Cluster concepts based on their energy signatures"""
        if not concepts:
            return {}
        
        # Simple clustering implementation
        # In a full system, this would use sophisticated clustering algorithms
        
        # Initialize clusters with random seeds
        np.random.shuffle(concepts)
        seeds = concepts[:num_clusters]
        
        clusters = {i: [seed] for i, seed in enumerate(seeds)}
        remaining = concepts[num_clusters:]
        
        # Assign each remaining concept to nearest cluster
        for concept_id in remaining:
            best_cluster = 0
            best_similarity = -1
            
            for cluster_id, cluster_concepts in clusters.items():
                # Calculate average similarity to concepts in this cluster
                similarities = []
                for cluster_concept in cluster_concepts:
                    if concept_id in self.network.concepts and cluster_concept in self.network.concepts:
                        similarity = self.network._calculate_direct_resonance(
                            self.network.concepts[concept_id],
                            self.network.concepts[cluster_concept]
                        )
                        similarities.append(similarity)
                
                if similarities:
                    avg_similarity = sum(similarities) / len(similarities)
                    if avg_similarity > best_similarity:
                        best_similarity = avg_similarity
                        best_cluster = cluster_id
            
            # Add to best cluster
            clusters[best_cluster].append(concept_id)
        
        return clusters
    
    def _find_cluster_representative(self, concepts: List[str]) -> str:
        """Find the concept that best represents a cluster"""
        if not concepts:
            return None
        
        if len(concepts) == 1:
            return concepts[0]
        
        # Calculate centrality for each concept
        centrality = {}
        
        for concept_id in concepts:
            total_similarity = 0.0
            for other_id in concepts:
                if concept_id != other_id:
                    if concept_id in self.network.concepts and other_id in self.network.concepts:
                        similarity = self.network._calculate_direct_resonance(
                            self.network.concepts[concept_id],
                            self.network.concepts[other_id]
                        )
                        total_similarity += similarity
            
            centrality[concept_id] = total_similarity
        
        # Return most central concept
        if centrality:
            return max(centrality.items(), key=lambda x: x[1])[0]
        
        # Fallback to first concept
        return concepts[0]
    
    def _build_cluster_connections(self, level: int) -> None:
        """Build connections between clusters at a given level"""
        if level not in self.concept_clusters or level not in self.cluster_representatives:
            return
        
        clusters = self.concept_clusters[level]
        representatives = self.cluster_representatives[level]
        
        # Build connections between clusters
        for cluster1_id, rep1 in representatives.items():
            for cluster2_id, rep2 in representatives.items():
                if cluster1_id != cluster2_id:
                    # Calculate connection strength between representatives
                    if rep1 in self.network.concepts and rep2 in self.network.concepts:
                        strength = self.network._calculate_direct_resonance(
                            self.network.concepts[rep1],
                            self.network.concepts[rep2]
                        )
                        
                        # Only store strong connections
                        if strength > 0.3:
                            self.cluster_connections[level][cluster1_id][cluster2_id] = strength
    
    def _find_containing_clusters(self, concept_ids: List[str]) -> Dict[int, List[int]]:
        """Find clusters containing the given concepts at each level"""
        result = defaultdict(list)
        
        for level, clusters in self.concept_clusters.items():
            for cluster_id, concepts in clusters.items():
                # Check if any seed concepts are in this cluster
                if any(concept_id in concepts for concept_id in concept_ids):
                    result[level].append(cluster_id)
        
        return result
    
    def _calculate_cluster_field(self, seed_concepts: List[str],
                               cluster_concepts: List[str],
                               depth: int) -> Dict[str, float]:
        """Calculate resonant field within a cluster"""
        # Find seed concepts in this cluster
        cluster_seeds = [c for c in seed_concepts if c in cluster_concepts]
        
        # If no seeds, calculate from representatives
        if not cluster_seeds:
            return {}
        
        # Calculate field within cluster
        field = {}
        for concept_id in cluster_concepts:
            # Initialize with zero activation
            field[concept_id] = 0.0
            
            # Calculate activation from each seed
            for seed_id in cluster_seeds:
                # Direct connection
                if seed_id == concept_id:
                    field[concept_id] = 1.0
                    continue
                
                # Find path within cluster
                path = self._find_local_path(seed_id, concept_id, cluster_concepts, depth)
                
                if path:
                    # Calculate activation based on path
                    path_strength = 1.0
                    for _, strength in path[1:]:  # Skip first node (seed)
                        path_strength *= strength
                    
                    # Apply decay based on path length
                    activation = path_strength * (0.8 ** (len(path) - 1))
                    
                    # Update field (take maximum activation)
                    field[concept_id] = max(field[concept_id], activation)
        
        return field
    
    def _find_local_path(self, source_id: str, target_id: str,
                       allowed_concepts: List[str],
                       max_depth: int) -> List[Tuple[str, float]]:
        """Find path between concepts within a limited set"""
        if source_id == target_id:
            return [(source_id, 1.0)]
        
        # Simple breadth-first search within allowed concepts
        visited = {source_id}
        queue = [[(source_id, 1.0)]]
        
        while queue:
            path = queue.pop(0)
            current_id = path[-1][0]
            
            # Check if reached target
            if current_id == target_id:
                return path
            
            # Check depth limit
            if len(path) > max_depth:
                continue
            
            # Expand neighbors
            if current_id in self.network.connections:
                for neighbor_id, connection in self.network.connections[current_id].items():
                    if neighbor_id not in visited and neighbor_id in allowed_concepts:
                        visited.add(neighbor_id)
                        new_path = path + [(neighbor_id, connection['strength'])]
                        queue.append(new_path)
        
        # No path found
        return []
    
    def _check_direct_path(self, source_id: str, target_id: str,
                         max_depth: int) -> List[Tuple[str, float]]:
        """Check if a short direct path exists between concepts"""
        # Simple breadth-first search for short paths
        visited = {source_id}
        queue = [[(source_id, 1.0)]]
        
        while queue:
            path = queue.pop(0)
            current_id = path[-1][0]
            
            # Check if reached target
            if current_id == target_id:
                return path
            
            # Check depth limit
            if len(path) > max_depth:
                continue
            
            # Expand neighbors
            if current_id in self.network.connections:
                for neighbor_id, connection in self.network.connections[current_id].items():
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        new_path = path + [(neighbor_id, connection['strength'])]
                        queue.append(new_path)
        
        # No short path found
        return None
    
    def _hierarchical_search(self, source_id: str, target_id: str,
                           max_depth: int) -> List[Tuple[str, float]]:
        """Find path using hierarchical search"""
        # Find containing clusters for source and target
        source_clusters = {}
        target_clusters = {}
        
        for level, clusters in self.concept_clusters.items():
            for cluster_id, concepts in clusters.items():
                if source_id in concepts:
                    source_clusters[level] = cluster_id
                if target_id in concepts:
                    target_clusters[level] = cluster_id
        
        # Try to find path at each level, starting from highest (most abstract)
        for level in sorted(self.concept_clusters.keys(), reverse=True):
            if level in source_clusters and level in target_clusters:
                source_cluster = source_clusters[level]
                target_cluster = target_clusters[level]
                
                # If in same cluster, search within cluster
                if source_cluster == target_cluster:
                    cluster_concepts = self.concept_clusters[level][source_cluster]
                    path = self._find_local_path(source_id, target_id, cluster_concepts, max_depth)
                    if path:
                        return path
                else:
                    # Find path between clusters
                    cluster_path = self._find_cluster_path(source_cluster, target_cluster, level)
                    
                    if cluster_path:
                        # Convert cluster path to concept path
                        return self._convert_cluster_path_to_concept_path(
                            cluster_path, source_id, target_id, level
                        )
        
        # Fallback to direct search with heuristic
        return self._heuristic_search(source_id, target_id, max_depth)
    
    def _find_cluster_path(self, source_cluster: int, target_cluster: int,
                         level: int) -> List[Tuple[int, float]]:
        """Find path between clusters"""
        if source_cluster == target_cluster:
            return [(source_cluster, 1.0)]
        
        # Simple breadth-first search
        visited = {source_cluster}
        queue = [[(source_cluster, 1.0)]]
        
        while queue:
            path = queue.pop(0)
            current_cluster = path[-1][0]
            
            # Check if reached target
            if current_cluster == target_cluster:
                return path
            
            # Check depth limit
            if len(path) > 5:  # Limit cluster path length
                continue
            
            # Expand neighbors
            if current_cluster in self.cluster_connections[level]:
                for neighbor_cluster, strength in self.cluster_connections[level][current_cluster].items():
                    if neighbor_cluster not in visited:
                        visited.add(neighbor_cluster)
                        new_path = path + [(neighbor_cluster, strength)]
                        queue.append(new_path)
        
        # No path found
        return None
    
    def _convert_cluster_path_to_concept_path(self, cluster_path: List[Tuple[int, float]],
                                           source_id: str, target_id: str,
                                           level: int) -> List[Tuple[str, float]]:
        """Convert a path between clusters to a path between concepts"""
        if not cluster_path or len(cluster_path) < 2:
            return []
        
        # Initialize with source
        concept_path = [(source_id, 1.0)]
        
        # Add representatives for intermediate clusters
        for i in range(1, len(cluster_path) - 1):
            cluster_id, strength = cluster_path[i]
            rep_id = self.cluster_representatives[level].get(cluster_id)
            
            if rep_id:
                concept_path.append((rep_id, strength))
        
        # Add target
        concept_path.append((target_id, cluster_path[-1][1]))
        
        # Refine path by finding connections between consecutive concepts
        refined_path = [concept_path[0]]
        
        for i in range(1, len(concept_path)):
            prev_id = refined_path[-1][0]
            current_id = concept_path[i][0]
            
            # Find path between consecutive concepts
            connecting_path = self._find_local_path(
                prev_id, current_id, 
                list(self.network.concepts.keys()),  # All concepts
                3  # Small depth for local connection
            )
            
            if connecting_path and len(connecting_path) > 1:
                # Add intermediate concepts from connecting path
                refined_path.extend(connecting_path[1:])
            else:
                # Just add current concept
                refined_path.append(concept_path[i])
        
        return refined_path
    
    def _heuristic_search(self, source_id: str, target_id: str,
                        max_depth: int) -> List[Tuple[str, float]]:
        """Find path using heuristic search"""
        # A* search with heuristic
        if source_id == target_id:
            return [(source_id, 1.0)]
        
        # Choose heuristic
        heuristic = self.navigation_heuristics['semantic_distance']
        
        # Priority queue for A* search (priority, path)
        from heapq import heappush, heappop
        queue = [(0, [(source_id, 1.0)])]
        visited = {source_id}
        
        while queue:
            _, path = heappop(queue)
            current_id = path[-1][0]
            
            # Check if reached target
            if current_id == target_id:
                return path
            
            # Check depth limit
            if len(path) > max_depth:
                continue
            
            # Expand neighbors
            if current_id in self.network.connections:
                for neighbor_id, connection in self.network.connections[current_id].items():
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        new_path = path + [(neighbor_id, connection['strength'])]
                        
                        # Calculate priority using heuristic
                        h_value = heuristic(neighbor_id, target_id)
                        g_value = len(new_path)  # Path length
                        priority = g_value + h_value
                        
                        heappush(queue, (priority, new_path))
        
        # No path found
        return []
    
    # Heuristic functions
    
    def _heuristic_semantic_distance(self, concept_id: str, target_id: str) -> float:
        """Estimate distance based on semantic properties"""
        if concept_id not in self.network.concepts or target_id not in self.network.concepts:
            return 10.0  # Large value for unknown concepts
        
        # Calculate direct resonance and convert to distance
        resonance = self.network._calculate_direct_resonance(
            self.network.concepts[concept_id],
            self.network.concepts[target_id]
        )
        
        # Convert resonance to distance (1 - resonance)
        return 1.0 - resonance
    
    def _heuristic_path_frequency(self, concept_id: str, target_id: str) -> float:
        """Estimate distance based on path frequency"""
        # This would use learned path frequencies in a full implementation
        return 5.0  # Default value
    
    def _heuristic_concept_importance(self, concept_id: str, target_id: str) -> float:
        """Estimate distance based on concept importance"""
        # This would use concept centrality in a full implementation
        return 5.0  # Default value