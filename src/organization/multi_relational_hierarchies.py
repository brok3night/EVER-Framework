"""
Multi-Relational Hierarchies - Dynamic discovery and navigation of concept hierarchies
"""
from typing import Dict, List, Set, Tuple, Any
import numpy as np
from collections import defaultdict

class MultiRelationalHierarchies:
    """
    Enables EVER to naturally discover, organize and navigate multiple
    simultaneous hierarchical relationships between concepts
    """
    
    def __init__(self, resonance_network):
        self.network = resonance_network
        
        # Multiple hierarchy types
        self.hierarchies = {
            'categorical': {  # is-a, type-of
                'structure': {},  # parents, children
                'levels': {},  # abstraction levels
                'detection_threshold': 0.7  # confidence threshold for detection
            },
            'compositional': {  # part-of, contains
                'structure': {},
                'levels': {},
                'detection_threshold': 0.75
            },
            'sequential': {  # precedes, follows
                'structure': {},
                'levels': {},
                'detection_threshold': 0.7
            },
            'causal': {  # causes, enables, prevents
                'structure': {},
                'levels': {},
                'detection_threshold': 0.8
            },
            'spatial': {  # near, contains, within
                'structure': {},
                'levels': {},
                'detection_threshold': 0.7
            }
        }
        
        # Relation types that suggest hierarchical connections
        self.hierarchical_relation_types = {
            'categorical': [
                'is_a', 'type_of', 'instance_of', 'subclass_of', 
                'example_of', 'category_of', 'subcategory_of'
            ],
            'compositional': [
                'part_of', 'contains', 'component_of', 'composed_of',
                'element_of', 'made_of', 'consists_of'
            ],
            'sequential': [
                'precedes', 'follows', 'before', 'after',
                'earlier_than', 'later_than', 'leads_to'
            ],
            'causal': [
                'causes', 'enables', 'prevents', 'results_in',
                'produces', 'inhibits', 'necessary_for'
            ],
            'spatial': [
                'within', 'contains', 'near', 'above', 'below',
                'surrounds', 'inside', 'outside'
            ]
        }
        
        # Energy patterns suggesting hierarchical relationships
        self.hierarchy_energy_patterns = {
            'categorical': {
                'parent_vector': [0.5, 0.8, 0.5],  # Higher abstraction (Y-axis)
                'child_vector': [0.5, 0.3, 0.5],   # Lower abstraction (Y-axis)
                'vector_tolerance': 0.25,
                'parent_entropy': 0.3,  # Parents have lower entropy (fewer details)
                'child_entropy': 0.6,   # Children have higher entropy (more details)
                'entropy_tolerance': 0.2
            },
            'compositional': {
                'parent_vector': [0.5, 0.6, 0.6],  # Moderately abstract, objective
                'child_vector': [0.5, 0.4, 0.5],   # More concrete
                'vector_tolerance': 0.3,
                'parent_magnitude': 0.7,  # Wholes have higher magnitude
                'child_magnitude': 0.5,   # Parts have lower magnitude
                'magnitude_tolerance': 0.2
            }
            # Other hierarchy types would have their own patterns
        }
        
        # Concept membership in hierarchies
        self.concept_hierarchies = {}  # concept_id -> {hierarchy_type -> [parent_concepts]}
        
        # Abstraction levels within each hierarchy
        self.abstraction_levels = {}  # hierarchy_type -> {concept_id -> level}
    
    def discover_hierarchies(self) -> Dict:
        """
        Discover hierarchical relationships across the concept network
        
        Returns:
            Dictionary of discovered hierarchical relationships
        """
        # Clear existing hierarchies
        self._initialize_hierarchies()
        
        # Get all concepts
        concepts = list(self.network.concepts.keys())
        
        # First pass: detect explicit hierarchical relationships from connections
        self._detect_explicit_hierarchies(concepts)
        
        # Second pass: infer hierarchical relationships from energy patterns
        self._infer_energy_hierarchies(concepts)
        
        # Third pass: infer transitive hierarchical relationships
        self._infer_transitive_hierarchies()
        
        # Calculate abstraction levels within each hierarchy
        self._calculate_abstraction_levels()
        
        # Return summary of discovered hierarchies
        return self._summarize_hierarchies()
    
    def get_concept_hierarchies(self, concept_id: str) -> Dict:
        """
        Get all hierarchies that a concept participates in
        
        Args:
            concept_id: Concept to analyze
            
        Returns:
            Dictionary of hierarchy information for this concept
        """
        if concept_id not in self.concept_hierarchies:
            return {}
        
        result = {}
        
        # For each hierarchy type
        for hierarchy_type, memberships in self.concept_hierarchies[concept_id].items():
            # Get parents in this hierarchy
            parents = memberships.get('parents', [])
            
            # Get children in this hierarchy
            children = []
            for child_id, child_memberships in self.concept_hierarchies.items():
                if (hierarchy_type in child_memberships and 
                    'parents' in child_memberships[hierarchy_type] and
                    concept_id in child_memberships[hierarchy_type]['parents']):
                    children.append(child_id)
            
            # Get abstraction level
            level = self.abstraction_levels.get(hierarchy_type, {}).get(concept_id, 0)
            
            # Add to result
            result[hierarchy_type] = {
                'parents': parents,
                'children': children,
                'level': level
            }
        
        return result
    
    def find_hierarchical_path(self, source_id: str, target_id: str,
                             hierarchy_types: List[str] = None) -> Dict:
        """
        Find hierarchical path between concepts
        
        Args:
            source_id: Source concept
            target_id: Target concept
            hierarchy_types: Optional list of hierarchy types to consider
            
        Returns:
            Dictionary with path information for each applicable hierarchy
        """
        # Use all hierarchy types if none specified
        if hierarchy_types is None:
            hierarchy_types = list(self.hierarchies.keys())
        
        result = {}
        
        # Check each hierarchy type
        for hierarchy_type in hierarchy_types:
            # Skip if hierarchy doesn't exist
            if hierarchy_type not in self.hierarchies:
                continue
            
            # Find path in this hierarchy
            path = self._find_path_in_hierarchy(source_id, target_id, hierarchy_type)
            
            if path:
                result[hierarchy_type] = path
        
        return result
    
    def get_hierarchy_level(self, concept_id: str, level: int,
                          hierarchy_type: str = 'categorical') -> List[str]:
        """
        Get concepts at a specific level in a hierarchy
        
        Args:
            concept_id: Concept to start from
            level: Target abstraction level (relative to concept)
            hierarchy_type: Hierarchy type to navigate
            
        Returns:
            List of concepts at the specified level
        """
        # Check if concept exists in this hierarchy
        if (concept_id not in self.concept_hierarchies or
            hierarchy_type not in self.concept_hierarchies[concept_id]):
            return []
        
        # Get concept's current level
        current_level = self.abstraction_levels.get(hierarchy_type, {}).get(concept_id, 0)
        
        # Target level
        target_level = current_level + level
        
        # Get concepts at target level
        result = []
        
        for cid, lvl in self.abstraction_levels.get(hierarchy_type, {}).items():
            if lvl == target_level:
                result.append(cid)
        
        return result
    
    def navigate_hierarchy(self, concept_id: str, direction: str,
                         hierarchy_type: str = 'categorical',
                         steps: int = 1) -> List[str]:
        """
        Navigate up or down a hierarchy from a concept
        
        Args:
            concept_id: Starting concept
            direction: 'up' or 'down'
            hierarchy_type: Hierarchy type to navigate
            steps: Number of steps to take
            
        Returns:
            List of concepts at the destination
        """
        # Check if concept exists in this hierarchy
        if (concept_id not in self.concept_hierarchies or
            hierarchy_type not in self.concept_hierarchies[concept_id]):
            return []
        
        # Navigate up the hierarchy
        if direction == 'up':
            result = self._navigate_up(concept_id, hierarchy_type, steps)
        
        # Navigate down the hierarchy
        elif direction == 'down':
            result = self._navigate_down(concept_id, hierarchy_type, steps)
        
        # Invalid direction
        else:
            result = []
        
        return result
    
    def explain_hierarchical_relationship(self, concept1_id: str, 
                                        concept2_id: str) -> Dict:
        """
        Explain the hierarchical relationship between two concepts
        
        Args:
            concept1_id: First concept
            concept2_id: Second concept
            
        Returns:
            Dictionary explaining relationships in each applicable hierarchy
        """
        result = {}
        
        # Check each hierarchy type
        for hierarchy_type in self.hierarchies.keys():
            explanation = self._explain_relationship_in_hierarchy(
                concept1_id, concept2_id, hierarchy_type
            )
            
            if explanation:
                result[hierarchy_type] = explanation
        
        return result
    
    def _initialize_hierarchies(self) -> None:
        """Initialize hierarchy structures"""
        # Reset hierarchies
        for hierarchy_type in self.hierarchies:
            self.hierarchies[hierarchy_type]['structure'] = {
                'parents': defaultdict(set),
                'children': defaultdict(set)
            }
            self.hierarchies[hierarchy_type]['levels'] = {}
        
        # Reset concept hierarchies
        self.concept_hierarchies = {}
        
        # Reset abstraction levels
        self.abstraction_levels = {}
    
    def _detect_explicit_hierarchies(self, concepts: List[str]) -> None:
        """Detect explicit hierarchical relationships from connections"""
        # For each concept
        for concept_id in concepts:
            # Skip if concept doesn't exist
            if concept_id not in self.network.connections:
                continue
            
            # Check connections for hierarchical relationships
            for target_id, connection in self.network.connections[concept_id].items():
                connection_type = connection.get('type', '')
                
                # Check which hierarchy type this connection suggests
                for hierarchy_type, relation_types in self.hierarchical_relation_types.items():
                    if connection_type in relation_types:
                        # Found hierarchical relationship
                        self._add_hierarchical_relationship(
                            concept_id, target_id, hierarchy_type, connection_type,
                            connection.get('strength', 0.5)
                        )
    
    def _infer_energy_hierarchies(self, concepts: List[str]) -> None:
        """Infer hierarchical relationships from energy patterns"""
        # For each concept pair
        for i, parent_id in enumerate(concepts):
            # Skip if concept doesn't exist
            if parent_id not in self.network.concepts:
                continue
            
            parent_energy = self.network.concepts[parent_id]
            
            for j, child_id in enumerate(concepts):
                if i == j:
                    continue  # Skip self
                
                # Skip if concept doesn't exist
                if child_id not in self.network.concepts:
                    continue
                
                child_energy = self.network.concepts[child_id]
                
                # Check each hierarchy type
                for hierarchy_type, pattern in self.hierarchy_energy_patterns.items():
                    # Check if energy patterns match hierarchical relationship
                    confidence = self._check_energy_hierarchy_match(
                        parent_energy, child_energy, pattern
                    )
                    
                    # Add if confidence exceeds threshold
                    threshold = self.hierarchies[hierarchy_type]['detection_threshold']
                    if confidence >= threshold:
                        self._add_hierarchical_relationship(
                            child_id, parent_id, hierarchy_type, 'energy_inferred',
                            confidence
                        )
    
    def _infer_transitive_hierarchies(self) -> None:
        """Infer transitive hierarchical relationships"""
        # For each hierarchy type
        for hierarchy_type, hierarchy in self.hierarchies.items():
            structure = hierarchy['structure']
            
            # Keep track of new relationships
            new_relationships = []
            
            # For each concept
            for concept_id in structure['parents']:
                # Get parents
                parents = structure['parents'][concept_id]
                
                # For each parent
                for parent_id in parents:
                    # Get grandparents
                    if parent_id in structure['parents']:
                        grandparents = structure['parents'][parent_id]
                        
                        # Add transitive relationships
                        for grandparent_id in grandparents:
                            # Avoid cycles
                            if grandparent_id != concept_id and grandparent_id not in parents:
                                new_relationships.append((concept_id, grandparent_id))
            
            # Add new relationships
            for child_id, parent_id in new_relationships:
                self._add_hierarchical_relationship(
                    child_id, parent_id, hierarchy_type, 'transitively_inferred',
                    0.6  # Lower confidence for inferred relationships
                )
    
    def _calculate_abstraction_levels(self) -> None:
        """Calculate abstraction levels within each hierarchy"""
        # For each hierarchy type
        for hierarchy_type, hierarchy in self.hierarchies.items():
            structure = hierarchy['structure']
            
            # Initialize levels
            levels = {}
            
            # Find root concepts (no parents)
            roots = []
            
            for concept_id in self.network.concepts:
                if (concept_id in structure['children'] and 
                    (concept_id not in structure['parents'] or not structure['parents'][concept_id])):
                    roots.append(concept_id)
            
            # If no roots found, find concepts with fewest parents
            if not roots:
                min_parents = float('inf')
                
                for concept_id in structure['parents']:
                    num_parents = len(structure['parents'][concept_id])
                    
                    if num_parents < min_parents:
                        min_parents = num_parents
                        roots = [concept_id]
                    elif num_parents == min_parents:
                        roots.append(concept_id)
            
            # Assign level 0 to roots
            for root in roots:
                levels[root] = 0
            
            # BFS to assign levels
            queue = [(root, 0) for root in roots]
            visited = set(roots)
            
            while queue:
                concept_id, level = queue.pop(0)
                
                # Get children
                if concept_id in structure['children']:
                    children = structure['children'][concept_id]
                    
                    for child_id in children:
                        # If child already visited, take minimum level
                        if child_id in visited:
                            levels[child_id] = min(levels.get(child_id, float('inf')), level + 1)
                        else:
                            levels[child_id] = level + 1
                            visited.add(child_id)
                            queue.append((child_id, level + 1))
            
            # Store levels
            self.abstraction_levels[hierarchy_type] = levels
            self.hierarchies[hierarchy_type]['levels'] = levels
    
    def _summarize_hierarchies(self) -> Dict:
        """Summarize discovered hierarchies"""
        summary = {}
        
        # For each hierarchy type
        for hierarchy_type, hierarchy in self.hierarchies.items():
            structure = hierarchy['structure']
            
            # Count relationships
            num_relationships = sum(len(parents) for parents in structure['parents'].values())
            
            # Count concepts
            num_concepts = len(set(structure['parents'].keys()) | set(structure['children'].keys()))
            
            # Calculate max level
            max_level = 0
            if hierarchy_type in self.abstraction_levels:
                max_level = max(self.abstraction_levels[hierarchy_type].values(), default=0)
            
            # Add to summary
            summary[hierarchy_type] = {
                'num_concepts': num_concepts,
                'num_relationships': num_relationships,
                'max_level': max_level
            }
        
        return summary
    
    def _add_hierarchical_relationship(self, child_id: str, parent_id: str,
                                    hierarchy_type: str, relation_type: str,
                                    confidence: float) -> None:
        """Add a hierarchical relationship"""
        # Update hierarchy structure
        self.hierarchies[hierarchy_type]['structure']['parents'][child_id].add(parent_id)
        self.hierarchies[hierarchy_type]['structure']['children'][parent_id].add(child_id)
        
        # Update concept hierarchies
        if child_id not in self.concept_hierarchies:
            self.concept_hierarchies[child_id] = {}
        
        if hierarchy_type not in self.concept_hierarchies[child_id]:
            self.concept_hierarchies[child_id][hierarchy_type] = {
                'parents': [],
                'relation_types': {}
            }
        
        # Add parent if not already present
        if parent_id not in self.concept_hierarchies[child_id][hierarchy_type]['parents']:
            self.concept_hierarchies[child_id][hierarchy_type]['parents'].append(parent_id)
            self.concept_hierarchies[child_id][hierarchy_type]['relation_types'][parent_id] = relation_type
        
        # Update parent's concept hierarchies
        if parent_id not in self.concept_hierarchies:
            self.concept_hierarchies[parent_id] = {}
        
        if hierarchy_type not in self.concept_hierarchies[parent_id]:
            self.concept_hierarchies[parent_id][hierarchy_type] = {
                'parents': [],
                'relation_types': {}
            }
    
    def _check_energy_hierarchy_match(self, parent_energy: Dict, 
                                   child_energy: Dict,
                                   pattern: Dict) -> float:
        """Check if energy signatures match a hierarchical pattern"""
        # Start with neutral confidence
        confidence = 0.5
        
        # Check vector pattern
        if ('vector' in parent_energy and 'value' in parent_energy['vector'] and
            'vector' in child_energy and 'value' in child_energy['vector']):
            
            parent_vector = parent_energy['vector']['value']
            child_vector = child_energy['vector']['value']
            
            # Check if vectors match the pattern
            parent_pattern = pattern.get('parent_vector', [0.5, 0.8, 0.5])
            child_pattern = pattern.get('child_vector', [0.5, 0.3, 0.5])
            tolerance = pattern.get('vector_tolerance', 0.25)
            
            # Calculate vector similarities
            parent_sim = self._vector_similarity(parent_vector, parent_pattern)
            child_sim = self._vector_similarity(child_vector, child_pattern)
            
            # Update confidence based on vector match
            if parent_sim > tolerance and child_sim > tolerance:
                confidence += 0.2
            elif parent_sim > tolerance/2 and child_sim > tolerance/2:
                confidence += 0.1
        
        # Check entropy pattern
        if ('entropy' in parent_energy and 'value' in parent_energy['entropy'] and
            'entropy' in child_energy and 'value' in child_energy['entropy']):
            
            parent_entropy = parent_energy['entropy']['value']
            child_entropy = child_energy['entropy']['value']
            
            parent_pattern = pattern.get('parent_entropy')
            child_pattern = pattern.get('child_entropy')
            tolerance = pattern.get('entropy_tolerance', 0.2)
            
            # Check if entropies match the pattern
            if parent_pattern is not None and child_pattern is not None:
                parent_match = abs(parent_entropy - parent_pattern) <= tolerance
                child_match = abs(child_entropy - child_pattern) <= tolerance
                
                # Update confidence based on entropy match
                if parent_match and child_match:
                    confidence += 0.15
                elif parent_match or child_match:
                    confidence += 0.05
        
        # Check magnitude pattern
        if ('magnitude' in parent_energy and 'value' in parent_energy['magnitude'] and
            'magnitude' in child_energy and 'value' in child_energy['magnitude']):
            
            parent_magnitude = parent_energy['magnitude']['value']
            child_magnitude = child_energy['magnitude']['value']
            
            parent_pattern = pattern.get('parent_magnitude')
            child_pattern = pattern.get('child_magnitude')
            tolerance = pattern.get('magnitude_tolerance', 0.2)
            
            # Check if magnitudes match the pattern
            if parent_pattern is not None and child_pattern is not None:
                parent_match = abs(parent_magnitude - parent_pattern) <= tolerance
                child_match = abs(child_magnitude - child_pattern) <= tolerance
                
                # Update confidence based on magnitude match
                if parent_match and child_match:
                    confidence += 0.15
                elif parent_match or child_match:
                    confidence += 0.05
        
        return confidence
    
    def _vector_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """Calculate similarity between two vectors"""
        # Calculate Euclidean distance
        min_dims = min(len(vector1), len(vector2))
        
        if min_dims == 0:
            return 0.0
        
        squared_diff = sum((vector1[i] - vector2[i])**2 for i in range(min_dims))
        distance = squared_diff ** 0.5
        
        # Convert distance to similarity (0-1)
        # Max distance in unit hypercube = sqrt(dimensions)
        max_distance = min_dims ** 0.5
        similarity = 1.0 - (distance / max_distance)
        
        return max(0.0, similarity)
    
    def _find_path_in_hierarchy(self, source_id: str, target_id: str,
                              hierarchy_type: str) -> List[Dict]:
        """Find path between concepts in a specific hierarchy"""
        # Check if concepts exist in this hierarchy
        if (source_id not in self.concept_hierarchies or
            hierarchy_type not in self.concept_hierarchies[source_id] or
            target_id not in self.concept_hierarchies or
            hierarchy_type not in self.concept_hierarchies[target_id]):
            return None
        
        # Get hierarchy structure
        structure = self.hierarchies[hierarchy_type]['structure']
        
        # Check direct relationship
        if source_id in structure['parents'] and target_id in structure['parents'][source_id]:
            return [
                {'concept': source_id, 'relation': 'child'},
                {'concept': target_id, 'relation': 'parent'}
            ]
        
        if target_id in structure['parents'] and source_id in structure['parents'][target_id]:
            return [
                {'concept': source_id, 'relation': 'parent'},
                {'concept': target_id, 'relation': 'child'}
            ]
        
        # Try to find common ancestor
        source_ancestors = self._get_all_ancestors(source_id, hierarchy_type)
        target_ancestors = self._get_all_ancestors(target_id, hierarchy_type)
        
        common_ancestors = source_ancestors.intersection(target_ancestors)
        
        if common_ancestors:
            # Find closest common ancestor
            closest_ancestor = None
            min_distance = float('inf')
            
            for ancestor in common_ancestors:
                source_distance = self._get_path_distance(source_id, ancestor, hierarchy_type)
                target_distance = self._get_path_distance(target_id, ancestor, hierarchy_type)
                total_distance = source_distance + target_distance
                
                if total_distance < min_distance:
                    min_distance = total_distance
                    closest_ancestor = ancestor
            
            # Build path
            path = []
            
            # Add path from source to ancestor
            source_to_ancestor = self._get_path_to_ancestor(source_id, closest_ancestor, hierarchy_type)
            path.extend(source_to_ancestor)
            
            # Add path from ancestor to target
            ancestor_to_target = self._get_path_to_ancestor(target_id, closest_ancestor, hierarchy_type)
            
            # Reverse and adjust relations
            reversed_path = []
            for item in reversed(ancestor_to_target[:-1]):  # Skip ancestor which is already in path
                if item['relation'] == 'child':
                    reversed_path.append({'concept': item['concept'], 'relation': 'parent'})
                else:
                    reversed_path.append({'concept': item['concept'], 'relation': 'child'})
            
            path.extend(reversed_path)
            
            return path
        
        # No path found
        return None
    
    def _get_all_ancestors(self, concept_id: str, hierarchy_type: str) -> Set[str]:
        """Get all ancestors of a concept in a hierarchy"""
        ancestors = set()
        
        # Check if concept exists in this hierarchy
        if (concept_id not in self.concept_hierarchies or
            hierarchy_type not in self.concept_hierarchies[concept_id]):
            return ancestors
        
        # Get direct parents
        parents = self.concept_hierarchies[concept_id][hierarchy_type]['parents']
        
        # Add parents
        ancestors.update(parents)
        
        # Recursively add ancestors of parents
        for parent_id in parents:
            parent_ancestors = self._get_all_ancestors(parent_id, hierarchy_type)
            ancestors.update(parent_ancestors)
        
        return ancestors
    
    def _get_path_distance(self, source_id: str, target_id: str,
                         hierarchy_type: str) -> int:
        """Get distance between concepts in a hierarchy"""
        # BFS to find distance
        queue = [(source_id, 0)]
        visited = {source_id}
        
        while queue:
            concept_id, distance = queue.pop(0)
            
            if concept_id == target_id:
                return distance
            
            # Get parents
            if (concept_id in self.concept_hierarchies and
                hierarchy_type in self.concept_hierarchies[concept_id]):
                
                parents = self.concept_hierarchies[concept_id][hierarchy_type]['parents']
                
                for parent_id in parents:
                    if parent_id not in visited:
                        visited.add(parent_id)
                        queue.append((parent_id, distance + 1))
        
        # No path found
        return float('inf')
    
    def _get_path_to_ancestor(self, concept_id: str, ancestor_id: str,
                            hierarchy_type: str) -> List[Dict]:
        """Get path from concept to ancestor"""
        # BFS to find path
        queue = [(concept_id, [{'concept': concept_id, 'relation': 'child'}])]
        visited = {concept_id}
        
        while queue:
            current_id, path = queue.pop(0)
            
            if current_id == ancestor_id:
                return path
            
            # Get parents
            if (current_id in self.concept_hierarchies and
                hierarchy_type in self.concept_hierarchies[current_id]):
                
                parents = self.concept_hierarchies[current_id][hierarchy_type]['parents']
                
                for parent_id in parents:
                    if parent_id not in visited:
                        visited.add(parent_id)
                        new_path = path + [{'concept': parent_id, 'relation': 'parent'}]
                        queue.append((parent_id, new_path))
        
        # No path found
        return [{'concept': concept_id, 'relation': 'unknown'}]
    
    def _navigate_up(self, concept_id: str, hierarchy_type: str, steps: int) -> List[str]:
        """Navigate up a hierarchy from a concept"""
        result = []
        
        # Check if concept exists in this hierarchy
        if (concept_id not in self.concept_hierarchies or
            hierarchy_type not in self.concept_hierarchies[concept_id]):
            return result
        
        # BFS to navigate up
        queue = [(concept_id, 0)]
        visited = {concept_id}
        
        while queue:
            current_id, distance = queue.pop(0)
            
            if distance == steps:
                result.append(current_id)
                continue
            
            if distance > steps:
                break
            
            # Get parents
            if (current_id in self.concept_hierarchies and
                hierarchy_type in self.concept_hierarchies[current_id]):
                
                parents = self.concept_hierarchies[current_id][hierarchy_type]['parents']
                
                for parent_id in parents:
                    if parent_id not in visited:
                        visited.add(parent_id)
                        queue.append((parent_id, distance + 1))
        
        return result
    
    def _navigate_down(self, concept_id: str, hierarchy_type: str, steps: int) -> List[str]:
        """Navigate down a hierarchy from a concept"""
        result = []
        
        # Check if concept exists in this hierarchy
        if (concept_id not in self.concept_hierarchies or
            hierarchy_type not in self.concept_hierarchies[concept_id]):
            return result
        
        # Get hierarchy structure
        structure = self.hierarchies[hierarchy_type]['structure']
        
        # BFS to navigate down
        queue = [(concept_id, 0)]
        visited = {concept_id}
        
        while queue:
            current_id, distance = queue.pop(0)
            
            if distance == steps:
                result.append(current_id)
                continue
            
            if distance > steps:
                break
            
            # Get children
            if current_id in structure['children']:
                children = structure['children'][current_id]
                
                for child_id in children:
                    if child_id not in visited:
                        visited.add(child_id)
                        queue.append((child_id, distance + 1))
        
        return result
    
    def _explain_relationship_in_hierarchy(self, concept1_id: str, 
                                        concept2_id: str,
                                        hierarchy_type: str) -> Dict:
        """Explain relationship between concepts in a hierarchy"""
        # Check if concepts exist in this hierarchy
        if (concept1_id not in self.concept_hierarchies or
            hierarchy_type not in self.concept_hierarchies[concept1_id] or
            concept2_id not in self.concept_hierarchies or
            hierarchy_type not in self.concept_hierarchies[concept2_id]):
            return None
        
        # Get hierarchy structure
        structure = self.hierarchies[hierarchy_type]['structure']
        
        # Check direct relationships
        if concept1_id in structure['parents'] and concept2_id in structure['parents'][concept1_id]:
            relation_type = self.concept_hierarchies[concept1_id][hierarchy_type]['relation_types'].get(
                concept2_id, 'hierarchical'
            )
            
            return {
                'relationship': 'child_to_parent',
                'explanation': f"'{concept1_id}' is a child of '{concept2_id}' in the {hierarchy_type} hierarchy",
                'relation_type': relation_type
            }
        
        if concept2_id in structure['parents'] and concept1_id in structure['parents'][concept2_id]:
            relation_type = self.concept_hierarchies[concept2_id][hierarchy_type]['relation_types'].get(
                concept1_id, 'hierarchical'
            )
            
            return {
                'relationship': 'parent_to_child',
                'explanation': f"'{concept1_id}' is a parent of '{concept2_id}' in the {hierarchy_type} hierarchy",
                'relation_type': relation_type
            }
        
        # Check for common ancestor
        concept1_ancestors = self._get_all_ancestors(concept1_id, hierarchy_type)
        concept2_ancestors = self._get_all_ancestors(concept2_id, hierarchy_type)
        
        common_ancestors = concept1_ancestors.intersection(concept2_ancestors)
        
        if common_ancestors:
            # Find closest common ancestor
            closest_ancestor = None
            min_distance = float('inf')
            
            for ancestor in common_ancestors:
                distance1 = self._get_path_distance(concept1_id, ancestor, hierarchy_type)
                distance2 = self._get_path_distance(concept2_id, ancestor, hierarchy_type)
                total_distance = distance1 + distance2
                
                if total_distance < min_distance:
                    min_distance = total_distance
                    closest_ancestor = ancestor
            
            return {
                'relationship': 'siblings',
                'explanation': f"'{concept1_id}' and '{concept2_id}' share '{closest_ancestor}' as a common ancestor in the {hierarchy_type} hierarchy",
                'common_ancestor': closest_ancestor,
                'distance1': self._get_path_distance(concept1_id, closest_ancestor, hierarchy_type),
                'distance2': self._get_path_distance(concept2_id, closest_ancestor, hierarchy_type)
            }
        
        # Check for ancestor-descendant relationship
        concept1_descendants = self._get_all_descendants(concept1_id, hierarchy_type)
        
        if concept2_id in concept1_descendants:
            return {
                'relationship': 'ancestor_to_descendant',
                'explanation': f"'{concept1_id}' is an ancestor of '{concept2_id}' in the {hierarchy_type} hierarchy",
                'distance': self._get_path_distance(concept2_id, concept1_id, hierarchy_type)
            }
        
        concept2_descendants = self._get_all_descendants(concept2_id, hierarchy_type)
        
        if concept1_id in concept2_descendants:
            return {
                'relationship': 'descendant_to_ancestor',
                'explanation': f"'{concept1_id}' is a descendant of '{concept2_id}' in the {hierarchy_type} hierarchy",
                'distance': self._get_path_distance(concept1_id, concept2_id, hierarchy_type)
            }
        
        # No direct relationship
        return {
            'relationship': 'unrelated',
            'explanation': f"'{concept1_id}' and '{concept2_id}' are not directly related in the {hierarchy_type} hierarchy"
        }
    
    def _get_all_descendants(self, concept_id: str, hierarchy_type: str) -> Set[str]:
        """Get all descendants of a concept in a hierarchy"""
        descendants = set()
        
        # Get hierarchy structure
        structure = self.hierarchies[hierarchy_type]['structure']
        
        # Check if concept has children
        if concept_id not in structure['children']:
            return descendants
        
        # Get direct children
        children = structure['children'][concept_id]
        
        # Add children
        descendants.update(children)
        
        # Recursively add descendants of children
        for child_id in children:
            child_descendants = self._get_all_descendants(child_id, hierarchy_type)
            descendants.update(child_descendants)
        
        return descendants