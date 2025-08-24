"""
Hierarchical Resonance - Self-organizing hierarchical traversal with topic resonance
"""
from typing import Dict, List, Set, Tuple, Any
import numpy as np
from collections import defaultdict

class HierarchicalResonance:
    """
    Enables efficient conceptual traversal through self-organizing hierarchies
    with resonant topic acceleration
    """
    
    def __init__(self, resonance_network):
        self.network = resonance_network
        
        # Hierarchical structure
        self.hierarchy = {
            'levels': {},  # Level -> clusters at that level
            'membership': {},  # Concept -> clusters it belongs to at each level
            'parents': {},  # Concept -> parent concepts at each level
            'children': {}  # Concept -> child concepts at each level
        }
        
        # Resonance propagation parameters
        self.resonance_params = {
            'upward_decay': 0.8,   # How much resonance decays moving upward
            'downward_decay': 0.7,  # How much resonance decays moving downward
            'lateral_decay': 0.5,   # How much resonance decays moving laterally
            'activation_threshold': 0.2  # Minimum resonance to activate a concept
        }
        
        # Traversal statistics
        self.traversal_stats = {
            'hierarchical_jumps': 0,
            'lateral_moves': 0,
            'pruned_branches': 0,
            'total_traversals': 0
        }
    
    def build_hierarchy(self, levels: int = 3, method: str = 'energy_clustering') -> None:
        """
        Build hierarchical structure from the concept network
        
        Args:
            levels: Number of hierarchical levels
            method: Method for building hierarchy
        """
        if method == 'energy_clustering':
            self._build_hierarchy_energy_clustering(levels)
        elif method == 'connectivity_clustering':
            self._build_hierarchy_connectivity_clustering(levels)
        else:
            self._build_hierarchy_energy_clustering(levels)  # Default
    
    def find_path(self, source_id: str, target_id: str, max_depth: int = 10) -> List[Tuple[str, float]]:
        """
        Find path between concepts using hierarchical traversal
        
        Args:
            source_id: Source concept
            target_id: Target concept
            max_depth: Maximum path depth
            
        Returns:
            Path as list of (concept_id, resonance) pairs
        """
        if source_id == target_id:
            return [(source_id, 1.0)]
        
        # Update statistics
        self.traversal_stats['total_traversals'] += 1
        
        # Check direct connection first
        direct_path = self._check_direct_connection(source_id, target_id)
        if direct_path:
            self.traversal_stats['lateral_moves'] += 1
            return direct_path
        
        # Find the hierarchical levels of both concepts
        source_levels = self._get_concept_levels(source_id)
        target_levels = self._get_concept_levels(target_id)
        
        # Try to find common ancestor
        common_level, common_ancestor = self._find_common_ancestor(
            source_id, target_id, source_levels, target_levels
        )
        
        if common_ancestor:
            # Found common ancestor, use hierarchical path
            self.traversal_stats['hierarchical_jumps'] += 1
            
            # Build path: source -> common ancestor -> target
            path = self._build_hierarchical_path(
                source_id, target_id, common_ancestor, common_level
            )
            
            return path
        
        # No clear hierarchical path, fall back to optimized search
        self.traversal_stats['lateral_moves'] += 1
        return self._optimized_search(source_id, target_id, max_depth)
    
    def get_resonant_field(self, seed_concepts: List[str], 
                          depth: int = 2) -> Dict[str, float]:
        """
        Get resonant field using hierarchical acceleration
        
        Args:
            seed_concepts: Seed concepts
            depth: Resonance depth
            
        Returns:
            Dictionary of concept_id -> resonance
        """
        # Start with empty field
        field = {}
        
        # Add seeds with full resonance
        for concept_id in seed_concepts:
            field[concept_id] = 1.0
        
        # Propagate resonance hierarchically
        field = self._propagate_hierarchical_resonance(seed_concepts, field, depth)
        
        # Filter by activation threshold
        threshold = self.resonance_params['activation_threshold']
        field = {cid: res for cid, res in field.items() if res >= threshold}
        
        return field
    
    def update_hierarchy(self, changed_concepts: List[str] = None) -> None:
        """
        Update hierarchy based on changed concepts
        
        Args:
            changed_concepts: Concepts that have changed
        """
        if not changed_concepts:
            # Complete rebuild if no specific concepts provided
            levels = len(self.hierarchy['levels'])
            self.build_hierarchy(levels=levels)
            return
        
        # Update only affected parts of hierarchy
        for concept_id in changed_concepts:
            self._update_concept_hierarchy(concept_id)
    
    def _build_hierarchy_energy_clustering(self, levels: int) -> None:
        """Build hierarchy using energy signature clustering"""
        # Get all concepts
        concepts = list(self.network.concepts.keys())
        
        if not concepts:
            return
        
        # Clear current hierarchy
        self.hierarchy = {
            'levels': {},
            'membership': {},
            'parents': {},
            'children': {}
        }
        
        # Build each level
        for level in range(levels):
            # Number of clusters decreases as we go up the hierarchy
            num_clusters = max(1, len(concepts) // (5 * (level + 1)))
            
            # Cluster concepts based on energy signatures
            clusters = self._cluster_by_energy(concepts, num_clusters)
            
            # Store level clusters
            self.hierarchy['levels'][level] = clusters
            
            # Update membership
            for cluster_id, cluster_concepts in clusters.items():
                # Find cluster representative (parent concept)
                parent = self._find_cluster_representative(cluster_concepts, level)
                
                # Update hierarchy relations
                for concept_id in cluster_concepts:
                    # Initialize if needed
                    if concept_id not in self.hierarchy['membership']:
                        self.hierarchy['membership'][concept_id] = {}
                        self.hierarchy['parents'][concept_id] = {}
                    
                    # Store membership and parent
                    self.hierarchy['membership'][concept_id][level] = cluster_id
                    
                    if concept_id != parent:
                        self.hierarchy['parents'][concept_id][level] = parent
                    
                    # Update children relation
                    if parent not in self.hierarchy['children']:
                        self.hierarchy['children'][parent] = {}
                    
                    if level not in self.hierarchy['children'][parent]:
                        self.hierarchy['children'][parent][level] = set()
                    
                    if concept_id != parent:
                        self.hierarchy['children'][parent][level].add(concept_id)
    
    def _build_hierarchy_connectivity_clustering(self, levels: int) -> None:
        """Build hierarchy using connectivity clustering"""
        # Get all concepts
        concepts = list(self.network.concepts.keys())
        
        if not concepts:
            return
        
        # Clear current hierarchy
        self.hierarchy = {
            'levels': {},
            'membership': {},
            'parents': {},
            'children': {}
        }
        
        # Build each level
        for level in range(levels):
            # Number of clusters decreases as we go up the hierarchy
            num_clusters = max(1, len(concepts) // (5 * (level + 1)))
            
            # Cluster concepts based on connectivity
            clusters = self._cluster_by_connectivity(concepts, num_clusters)
            
            # Store level clusters
            self.hierarchy['levels'][level] = clusters
            
            # Update membership
            for cluster_id, cluster_concepts in clusters.items():
                # Find cluster representative (parent concept)
                parent = self._find_cluster_representative(cluster_concepts, level)
                
                # Update hierarchy relations
                for concept_id in cluster_concepts:
                    # Initialize if needed
                    if concept_id not in self.hierarchy['membership']:
                        self.hierarchy['membership'][concept_id] = {}
                        self.hierarchy['parents'][concept_id] = {}
                    
                    # Store membership and parent
                    self.hierarchy['membership'][concept_id][level] = cluster_id
                    
                    if concept_id != parent:
                        self.hierarchy['parents'][concept_id][level] = parent
                    
                    # Update children relation
                    if parent not in self.hierarchy['children']:
                        self.hierarchy['children'][parent] = {}
                    
                    if level not in self.hierarchy['children'][parent]:
                        self.hierarchy['children'][parent][level] = set()
                    
                    if concept_id != parent:
                        self.hierarchy['children'][parent][level].add(concept_id)
    
    def _cluster_by_energy(self, concepts: List[str], num_clusters: int) -> Dict[int, List[str]]:
        """Cluster concepts based on energy signatures"""
        if not concepts:
            return {}
        
        # Extract energy signatures
        energy_vectors = []
        valid_concepts = []
        
        for concept_id in concepts:
            if concept_id in self.network.concepts:
                energy = self.network.concepts[concept_id]
                
                if 'vector' in energy and 'value' in energy['vector']:
                    vector = energy['vector']['value']
                    
                    if isinstance(vector, list):
                        energy_vectors.append(vector)
                        valid_concepts.append(concept_id)
        
        if not valid_concepts:
            return {0: concepts}  # Fall back to single cluster
        
        # Convert to numpy array for clustering
        X = np.array(energy_vectors)
        
        # Simple k-means clustering
        clusters = self._simple_kmeans(X, num_clusters)
        
        # Map cluster IDs to concepts
        result = defaultdict(list)
        
        for i, cluster_id in enumerate(clusters):
            result[cluster_id].append(valid_concepts[i])
        
        # Add any concepts that didn't have valid vectors to first cluster
        missing = set(concepts) - set(valid_concepts)
        if missing and result:
            first_cluster = list(result.keys())[0]
            result[first_cluster].extend(list(missing))
        
        return dict(result)
    
    def _cluster_by_connectivity(self, concepts: List[str], num_clusters: int) -> Dict[int, List[str]]:
        """Cluster concepts based on connectivity patterns"""
        if not concepts:
            return {}
        
        # Create connectivity matrix
        n = len(concepts)
        connectivity = np.zeros((n, n))
        
        # Build connectivity matrix
        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts):
                if i == j:
                    connectivity[i, j] = 1.0  # Self-connection
                elif (concept1 in self.network.connections and 
                      concept2 in self.network.connections[concept1]):
                    # Direct connection
                    connection = self.network.connections[concept1][concept2]
                    connectivity[i, j] = connection.get('strength', 0.5)
                elif (concept2 in self.network.connections and 
                      concept1 in self.network.connections[concept2]):
                    # Reverse connection
                    connection = self.network.connections[concept2][concept1]
                    connectivity[i, j] = connection.get('strength', 0.5)
        
        # Spectral clustering on connectivity matrix
        clusters = self._simple_spectral_clustering(connectivity, num_clusters)
        
        # Map cluster IDs to concepts
        result = defaultdict(list)
        
        for i, cluster_id in enumerate(clusters):
            result[cluster_id].append(concepts[i])
        
        return dict(result)
    
    def _simple_kmeans(self, X: np.ndarray, k: int) -> np.ndarray:
        """Simple k-means clustering implementation"""
        # This is a simplified implementation
        # A real system would use scikit-learn or a more optimized implementation
        
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly
        indices = np.random.choice(n_samples, k, replace=False)
        centroids = X[indices]
        
        # Iterate until convergence
        for _ in range(10):  # Limited iterations for simplicity
            # Assign samples to nearest centroid
            distances = np.zeros((n_samples, k))
            
            for i in range(k):
                # Euclidean distance
                distances[:, i] = np.sqrt(np.sum((X - centroids[i])**2, axis=1))
            
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros((k, n_features))
            
            for i in range(k):
                cluster_samples = X[labels == i]
                if len(cluster_samples) > 0:
                    new_centroids[i] = np.mean(cluster_samples, axis=0)
                else:
                    # If no samples, keep old centroid
                    new_centroids[i] = centroids[i]
            
            # Check convergence
            if np.allclose(centroids, new_centroids):
                break
            
            centroids = new_centroids
        
        return labels
    
    def _simple_spectral_clustering(self, affinity: np.ndarray, k: int) -> np.ndarray:
        """Simple spectral clustering implementation"""
        # This is a simplified implementation
        # A real system would use scikit-learn or a more optimized implementation
        
        # Create Laplacian matrix
        D = np.diag(np.sum(affinity, axis=1))
        L = D - affinity
        
        # Compute eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(L)
        
        # Sort eigenvectors by eigenvalues
        idx = np.argsort(eigenvalues)
        eigenvectors = eigenvectors[:, idx]
        
        # Use first k eigenvectors
        features = eigenvectors[:, :k]
        
        # Apply k-means to the features
        labels = self._simple_kmeans(features, k)
        
        return labels
    
    def _find_cluster_representative(self, cluster: List[str], level: int) -> str:
        """Find the most representative concept in a cluster"""
        if not cluster:
            return None
        
        if len(cluster) == 1:
            return cluster[0]
        
        # Calculate centrality for each concept
        centrality = {}
        
        for concept_id in cluster:
            if concept_id not in self.network.concepts:
                continue
            
            # Calculate energy centrality
            energy_centrality = 0.0
            concept_energy = self.network.concepts[concept_id]
            
            for other_id in cluster:
                if other_id == concept_id or other_id not in self.network.concepts:
                    continue
                
                other_energy = self.network.concepts[other_id]
                
                # Calculate energy similarity
                similarity = self._calculate_energy_similarity(concept_energy, other_energy)
                energy_centrality += similarity
            
            # Calculate connectivity centrality
            connectivity_centrality = 0.0
            
            if concept_id in self.network.connections:
                for other_id in cluster:
                    if other_id == concept_id:
                        continue
                    
                    if other_id in self.network.connections[concept_id]:
                        connection = self.network.connections[concept_id][other_id]
                        connectivity_centrality += connection.get('strength', 0.5)
            
            # Combine centrality measures
            centrality[concept_id] = 0.7 * energy_centrality + 0.3 * connectivity_centrality
        
        # Return concept with highest centrality
        if centrality:
            return max(centrality, key=centrality.get)
        
        # Fall back to first concept
        return cluster[0]
    
    def _calculate_energy_similarity(self, energy1: Dict, energy2: Dict) -> float:
        """Calculate similarity between energy signatures"""
        similarity = 0.5  # Default moderate similarity
        
        # Vector similarity
        if ('vector' in energy1 and 'value' in energy1['vector'] and
            'vector' in energy2 and 'value' in energy2['vector']):
            
            vector1 = energy1['vector']['value']
            vector2 = energy2['vector']['value']
            
            if isinstance(vector1, list) and isinstance(vector2, list):
                # Calculate cosine similarity
                min_dims = min(len(vector1), len(vector2))
                
                if min_dims > 0:
                    dot_product = sum(vector1[i] * vector2[i] for i in range(min_dims))
                    norm1 = sum(v**2 for v in vector1[:min_dims]) ** 0.5
                    norm2 = sum(v**2 for v in vector2[:min_dims]) ** 0.5
                    
                    if norm1 > 0 and norm2 > 0:
                        vector_sim = dot_product / (norm1 * norm2)
                        similarity = 0.7 * vector_sim + 0.3 * similarity
        
        # Other property similarities
        for prop in ['frequency', 'entropy', 'magnitude']:
            if (prop in energy1 and 'value' in energy1[prop] and
                prop in energy2 and 'value' in energy2[prop]):
                
                value1 = energy1[prop]['value']
                value2 = energy2[prop]['value']
                
                if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                    # Calculate similarity as 1 - normalized difference
                    prop_sim = 1.0 - abs(value1 - value2)
                    similarity = 0.9 * similarity + 0.1 * prop_sim
        
        return similarity
    
    def _check_direct_connection(self, source_id: str, target_id: str) -> List[Tuple[str, float]]:
        """Check if there's a direct connection between concepts"""
        if (source_id in self.network.connections and 
            target_id in self.network.connections[source_id]):
            
            connection = self.network.connections[source_id][target_id]
            strength = connection.get('strength', 0.5)
            
            return [(source_id, 1.0), (target_id, strength)]
        
        if (target_id in self.network.connections and 
            source_id in self.network.connections[target_id]):
            
            connection = self.network.connections[target_id][source_id]
            strength = connection.get('strength', 0.5)
            
            return [(source_id, 1.0), (target_id, strength)]
        
        return None
    
    def _get_concept_levels(self, concept_id: str) -> Dict[int, int]:
        """Get the hierarchical levels and clusters for a concept"""
        if concept_id in self.hierarchy['membership']:
            return self.hierarchy['membership'][concept_id]
        
        return {}
    
    def _find_common_ancestor(self, source_id: str, target_id: str,
                           source_levels: Dict[int, int],
                           target_levels: Dict[int, int]) -> Tuple[int, str]:
        """Find common ancestor in hierarchy"""
        # Check each level, starting from highest
        max_level = max(list(self.hierarchy['levels'].keys()) + [-1])
        
        for level in range(max_level, -1, -1):
            # Check if both concepts exist at this level
            if level in source_levels and level in target_levels:
                source_cluster = source_levels[level]
                target_cluster = target_levels[level]
                
                # If in same cluster, find parent
                if source_cluster == target_cluster:
                    # Get parent (cluster representative)
                    if source_id in self.hierarchy['parents'] and level in self.hierarchy['parents'][source_id]:
                        return level, self.hierarchy['parents'][source_id][level]
                    
                    if target_id in self.hierarchy['parents'] and level in self.hierarchy['parents'][target_id]:
                        return level, self.hierarchy['parents'][target_id][level]
        
        # No common ancestor found
        return -1, None
    
    def _build_hierarchical_path(self, source_id: str, target_id: str, 
                               ancestor_id: str, ancestor_level: int) -> List[Tuple[str, float]]:
        """Build path through hierarchical structure"""
        # Path: source -> ancestor -> target
        path = [(source_id, 1.0)]
        
        # Add path from source to ancestor
        if source_id != ancestor_id:
            source_to_ancestor = self._path_to_ancestor(
                source_id, ancestor_id, ancestor_level
            )
            
            if source_to_ancestor:
                # Skip the first element (source) as it's already in the path
                path.extend(source_to_ancestor[1:])
            else:
                # Fallback if hierarchical path fails
                path.append((ancestor_id, self.resonance_params['upward_decay']))
        
        # Add path from ancestor to target
        if ancestor_id != target_id:
            ancestor_to_target = self._path_from_ancestor(
                ancestor_id, target_id, ancestor_level
            )
            
            if ancestor_to_target:
                # Skip the first element (ancestor) as it's already in the path
                path.extend(ancestor_to_target[1:])
            else:
                # Fallback if hierarchical path fails
                path.append((target_id, self.resonance_params['downward_decay']))
        
        return path
    
    def _path_to_ancestor(self, concept_id: str, ancestor_id: str, 
                        level: int) -> List[Tuple[str, float]]:
        """Find path from concept to ancestor"""
        path = [(concept_id, 1.0)]
        current = concept_id
        
        # Check if ancestor is a direct parent
        if (concept_id in self.hierarchy['parents'] and 
            level in self.hierarchy['parents'][concept_id] and
            self.hierarchy['parents'][concept_id][level] == ancestor_id):
            
            path.append((ancestor_id, self.resonance_params['upward_decay']))
            return path
        
        # Otherwise, traverse up the hierarchy
        visited = {concept_id}
        
        while current != ancestor_id:
            if (current in self.hierarchy['parents'] and 
                level in self.hierarchy['parents'][current]):
                
                parent = self.hierarchy['parents'][current][level]
                
                if parent in visited:
                    break  # Avoid loops
                
                visited.add(parent)
                path.append((parent, path[-1][1] * self.resonance_params['upward_decay']))
                
                current = parent
                
                if current == ancestor_id:
                    break
            else:
                break  # No path found
        
        if current == ancestor_id:
            return path
        
        # No path found
        return None
    
    def _path_from_ancestor(self, ancestor_id: str, concept_id: str, 
                          level: int) -> List[Tuple[str, float]]:
        """Find path from ancestor to concept"""
        path = [(ancestor_id, 1.0)]
        
        # Check if concept is a direct child
        if (ancestor_id in self.hierarchy['children'] and 
            level in self.hierarchy['children'][ancestor_id] and
            concept_id in self.hierarchy['children'][ancestor_id][level]):
            
            path.append((concept_id, self.resonance_params['downward_decay']))
            return path
        
        # Otherwise, need to find path through the hierarchy
        # This is more complex as there could be multiple paths down
        
        # Breadth-first search
        queue = [(ancestor_id, 1.0)]
        visited = {ancestor_id}
        parent_map = {}  # To reconstruct path
        
        while queue:
            current, resonance = queue.pop(0)
            
            if current == concept_id:
                break  # Found the target
            
            # Check children
            if (current in self.hierarchy['children'] and 
                level in self.hierarchy['children'][current]):
                
                for child in self.hierarchy['children'][current][level]:
                    if child not in visited:
                        visited.add(child)
                        new_resonance = resonance * self.resonance_params['downward_decay']
                        queue.append((child, new_resonance))
                        parent_map[child] = (current, new_resonance)
        
        # Reconstruct path
        if concept_id in visited:
            # Start from concept and work backward
            current = concept_id
            reverse_path = []
            
            while current != ancestor_id:
                parent, resonance = parent_map[current]
                reverse_path.append((current, resonance))
                current = parent
            
            # Reverse and add to path
            for item in reversed(reverse_path):
                path.append(item)
            
            return path
        
        # No path found
        return None
    
    def _optimized_search(self, source_id: str, target_id: str, 
                        max_depth: int) -> List[Tuple[str, float]]:
        """Optimized search when hierarchical path fails"""
        # This is a simplified A* search using hierarchical information as heuristic
        
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
                        
                        # Calculate resonance
                        strength = connection.get('strength', 0.5)
                        new_resonance = path[-1][1] * strength
                        
                        # New path
                        new_path = path + [(neighbor_id, new_resonance)]
                        
                        # Calculate priority using hierarchical heuristic
                        h_value = self._hierarchical_heuristic(neighbor_id, target_id)
                        g_value = len(new_path)  # Path length
                        priority = g_value - h_value  # Lower priority = better
                        
                        heappush(queue, (priority, new_path))
        
        # No path found
        return [(source_id, 1.0)]  # Just return source
    
    def _hierarchical_heuristic(self, concept_id: str, target_id: str) -> float:
        """Heuristic function using hierarchical information"""
        # Start with default heuristic
        heuristic = 0.0
        
        # Get levels for both concepts
        concept_levels = self._get_concept_levels(concept_id)
        target_levels = self._get_concept_levels(target_id)
        
        # Check if concepts share any clusters
        shared_level = -1
        
        for level in range(len(self.hierarchy['levels'])):
            if (level in concept_levels and level in target_levels and
                concept_levels[level] == target_levels[level]):
                
                shared_level = level
                break
        
        if shared_level >= 0:
            # Concepts share a cluster, improve heuristic
            heuristic = 2.0  # Higher value = more likely to explore this path
        
        return heuristic
    
    def _propagate_hierarchical_resonance(self, seed_concepts: List[str],
                                        field: Dict[str, float],
                                        depth: int) -> Dict[str, float]:
        """Propagate resonance through hierarchical structure"""
        # This is a multi-stage process:
        # 1. Propagate upward from seeds to ancestors
        # 2. Propagate laterally within clusters
        # 3. Propagate downward from ancestors to other concepts
        
        # Upward propagation
        ancestors = self._propagate_upward(seed_concepts, field)
        
        # Lateral propagation within clusters
        field = self._propagate_laterally(ancestors, field)
        
        # Downward propagation
        if depth > 1:
            field = self._propagate_downward(ancestors, field, depth - 1)
        
        return field
    
    def _propagate_upward(self, concepts: List[str],
                         field: Dict[str, float]) -> Dict[int, Dict[str, float]]:
        """Propagate resonance upward to ancestors"""
        # Track ancestors at each level
        ancestors = defaultdict(dict)
        
        # For each concept
        for concept_id in concepts:
            if concept_id not in self.hierarchy['membership']:
                continue
            
            # Get resonance
            resonance = field.get(concept_id, 0.0)
            
            if resonance <= 0.0:
                continue
            
            # Propagate to parents at each level
            for level, cluster_id in self.hierarchy['membership'][concept_id].items():
                # Get parent
                if (concept_id in self.hierarchy['parents'] and 
                    level in self.hierarchy['parents'][concept_id]):
                    
                    parent = self.hierarchy['parents'][concept_id][level]
                    
                    # Calculate parent resonance
                    parent_resonance = resonance * self.resonance_params['upward_decay']
                    
                    # Update parent in field
                    if parent in field:
                        field[parent] = max(field[parent], parent_resonance)
                    else:
                        field[parent] = parent_resonance
                    
                    # Track ancestor
                    ancestors[level][parent] = field[parent]
        
        return ancestors
    
    def _propagate_laterally(self, ancestors: Dict[int, Dict[str, float]],
                           field: Dict[str, float]) -> Dict[str, float]:
        """Propagate resonance laterally within clusters"""
        # For each level
        for level, level_ancestors in ancestors.items():
            # Get clusters at this level
            if level not in self.hierarchy['levels']:
                continue
            
            clusters = self.hierarchy['levels'][level]
            
            # For each ancestor
            for ancestor_id, resonance in level_ancestors.items():
                # Find cluster
                cluster_id = None
                
                if (ancestor_id in self.hierarchy['membership'] and 
                    level in self.hierarchy['membership'][ancestor_id]):
                    cluster_id = self.hierarchy['membership'][ancestor_id][level]
                else:
                    # Search for ancestor in clusters
                    for cid, concepts in clusters.items():
                        if ancestor_id in concepts:
                            cluster_id = cid
                            break
                
                if cluster_id is None or cluster_id not in clusters:
                    continue
                
                # Propagate to cluster members
                cluster_concepts = clusters[cluster_id]
                
                for concept_id in cluster_concepts:
                    if concept_id == ancestor_id:
                        continue  # Skip ancestor itself
                    
                    # Calculate lateral resonance
                    lateral_resonance = resonance * self.resonance_params['lateral_decay']
                    
                    # Update concept in field
                    if concept_id in field:
                        field[concept_id] = max(field[concept_id], lateral_resonance)
                    else:
                        field[concept_id] = lateral_resonance
        
        return field
    
    def _propagate_downward(self, ancestors: Dict[int, Dict[str, float]],
                          field: Dict[str, float],
                          depth: int) -> Dict[str, float]:
        """Propagate resonance downward from ancestors"""
        # For each level
        for level, level_ancestors in ancestors.items():
            # For each ancestor
            for ancestor_id, resonance in level_ancestors.items():
                # Check if ancestor has children
                if (ancestor_id in self.hierarchy['children'] and 
                    level in self.hierarchy['children'][ancestor_id]):
                    
                    # Propagate to children
                    children = self.hierarchy['children'][ancestor_id][level]
                    
                    for child_id in children:
                        # Calculate child resonance
                        child_resonance = resonance * self.resonance_params['downward_decay']
                        
                        # Update child in field
                        if child_id in field:
                            field[child_id] = max(field[child_id], child_resonance)
                        else:
                            field[child_id] = child_resonance
                        
                        # Recursively propagate to grandchildren if depth allows
                        if depth > 1:
                            field = self._recursive_propagate_downward(
                                child_id, child_resonance, field, depth - 1
                            )
        
        return field
    
    def _recursive_propagate_downward(self, concept_id: str, resonance: float,
                                    field: Dict[str, float],
                                    depth: int) -> Dict[str, float]:
        """Recursively propagate resonance downward"""
        if depth <= 0 or resonance < self.resonance_params['activation_threshold']:
            return field
        
        # Check connections
        if concept_id in self.network.connections:
            for neighbor_id, connection in self.network.connections[concept_id].items():
                # Calculate neighbor resonance
                strength = connection.get('strength', 0.5)
                neighbor_resonance = resonance * strength * self.resonance_params['lateral_decay']
                
                # Update neighbor in field
                if neighbor_id in field:
                    field[neighbor_id] = max(field[neighbor_id], neighbor_resonance)
                else:
                    field[neighbor_id] = neighbor_resonance
                
                # Recursively propagate if depth allows
                if depth > 1:
                    field = self._recursive_propagate_downward(
                        neighbor_id, neighbor_resonance, field, depth - 1
                    )
        
        return field
    
    def _update_concept_hierarchy(self, concept_id: str) -> None:
        """Update hierarchy for a single concept"""
        # This would be a more optimized implementation in a full system
        # For simplicity, we just rebuild the hierarchy
        levels = len(self.hierarchy['levels'])
        self.build_hierarchy(levels=levels)