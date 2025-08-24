"""
Experiential Grounding - Grounding concepts in experiential reality
"""
from typing import Dict, List, Any, Tuple

class ExperientialGrounding:
    """
    Grounds philosophical concepts in experiential reality through
    empirical mappings and reality checks
    """
    
    def __init__(self, resonance_network, config=None):
        self.network = resonance_network
        
        # Experiential domains (sources of grounding)
        self.experiential_domains = {
            'sensory': {
                'description': 'Direct sensory experiences',
                'grounding_strength': 0.9  # Very strong grounding
            },
            'action': {
                'description': 'Practical actions and their consequences',
                'grounding_strength': 0.8  # Strong grounding
            },
            'social': {
                'description': 'Social interactions and their outcomes',
                'grounding_strength': 0.7  # Moderate-strong grounding
            },
            'emotional': {
                'description': 'Emotional experiences and responses',
                'grounding_strength': 0.6  # Moderate grounding
            },
            'linguistic': {
                'description': 'Linguistic usage patterns',
                'grounding_strength': 0.5  # Moderate grounding
            }
        }
        
        # Concepts with empirical grounding
        self.grounded_concepts = {}
        
        # Grounding connections (concept -> empirical manifestation)
        self.grounding_connections = {}
    
    def register_grounding(self, concept_id: str, 
                         domain: str,
                         manifestations: List[Dict],
                         grounding_strength: float = None) -> bool:
        """
        Register empirical grounding for a concept
        
        Args:
            concept_id: Concept to ground
            domain: Experiential domain
            manifestations: List of empirical manifestations
            grounding_strength: Optional override for domain strength
            
        Returns:
            Success status
        """
        if domain not in self.experiential_domains:
            return False
        
        # Use domain strength if not specified
        if grounding_strength is None:
            grounding_strength = self.experiential_domains[domain]['grounding_strength']
        
        # Ensure concept exists
        if concept_id not in self.network.concepts:
            return False
        
        # Register grounding
        if concept_id not in self.grounded_concepts:
            self.grounded_concepts[concept_id] = {}
        
        self.grounded_concepts[concept_id][domain] = {
            'manifestations': manifestations,
            'strength': grounding_strength
        }
        
        # Create grounding connections
        if concept_id not in self.grounding_connections:
            self.grounding_connections[concept_id] = []
        
        for manifestation in manifestations:
            connection = {
                'domain': domain,
                'manifestation': manifestation,
                'strength': grounding_strength
            }
            self.grounding_connections[concept_id].append(connection)
        
        return True
    
    def get_grounding(self, concept_id: str) -> Dict:
        """
        Get empirical grounding for a concept
        
        Args:
            concept_id: Concept ID
            
        Returns:
            Grounding information
        """
        if concept_id not in self.grounded_concepts:
            return {}
        
        return self.grounded_concepts[concept_id]
    
    def calculate_grounding_strength(self, concept_id: str) -> float:
        """
        Calculate overall empirical grounding strength for a concept
        
        Args:
            concept_id: Concept ID
            
        Returns:
            Grounding strength (0-1)
        """
        if concept_id not in self.grounded_concepts:
            return 0.0
        
        # Calculate weighted average of domain strengths
        total_strength = 0.0
        total_weight = 0.0
        
        for domain, info in self.grounded_concepts[concept_id].items():
            strength = info['strength']
            
            # Weight by number of manifestations
            weight = len(info['manifestations'])
            total_strength += strength * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return total_strength / total_weight
    
    def find_empirical_path(self, abstract_concept_id: str) -> List[Dict]:
        """
        Find path from abstract concept to empirical grounding
        
        Args:
            abstract_concept_id: Abstract concept ID
            
        Returns:
            Path of concepts leading to empirical grounding
        """
        # Check if concept already has direct grounding
        if abstract_concept_id in self.grounded_concepts:
            return [{'concept_id': abstract_concept_id, 
                    'grounding': self.grounded_concepts[abstract_concept_id]}]
        
        # Search for path to grounded concept
        visited = {abstract_concept_id}
        queue = [[(abstract_concept_id, 1.0)]]
        
        while queue:
            path = queue.pop(0)
            current_id = path[-1][0]
            
            # Check if current concept is grounded
            if current_id in self.grounded_concepts:
                # Convert path to result format
                result = []
                for concept_id, strength in path:
                    if concept_id in self.grounded_concepts:
                        result.append({
                            'concept_id': concept_id,
                            'grounding': self.grounded_concepts[concept_id]
                        })
                    else:
                        result.append({
                            'concept_id': concept_id,
                            'grounding': None
                        })
                
                return result
            
            # Check depth limit
            if len(path) > 5:  # Limit path length
                continue
            
            # Expand neighbors
            if current_id in self.network.connections:
                for neighbor_id, connection in self.network.connections[current_id].items():
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        
                        # Prioritize 'example_of' and 'part_of' connections
                        connection_strength = connection['strength']
                        if connection['type'] in ['example_of', 'part_of']:
                            connection_strength *= 1.5  # Boost these connection types
                        
                        new_path = path + [(neighbor_id, connection_strength)]
                        queue.append(new_path)
        
        # No path found
        return []
    
    def ground_insight(self, insight: Dict) -> Dict:
        """
        Ground a philosophical insight in empirical reality
        
        Args:
            insight: Philosophical insight
            
        Returns:
            Grounded insight
        """
        # Clone insight
        grounded_insight = dict(insight)
        
        # Add grounding information
        grounding_info = self._generate_grounding_info(insight)
        
        if grounding_info:
            grounded_insight['grounding'] = grounding_info
        
        return grounded_insight
    
    def generate_reality_check(self, insight: Dict) -> Dict:
        """
        Generate empirical reality check for a philosophical insight
        
        Args:
            insight: Philosophical insight
            
        Returns:
            Reality check information
        """
        # Extract concepts from insight
        concepts = []
        
        if 'source_concepts' in insight:
            concepts.extend(insight['source_concepts'])
        
        if 'target_concepts' in insight:
            concepts.extend(insight['target_concepts'])
        
        # Find grounding for these concepts
        grounding_paths = []
        for concept_id in concepts:
            path = self.find_empirical_path(concept_id)
            if path:
                grounding_paths.append({
                    'concept_id': concept_id,
                    'path': path
                })
        
        # Generate reality check
        reality_check = {
            'grounding_paths': grounding_paths,
            'empirical_testability': self._calculate_testability(insight, grounding_paths),
            'empirical_manifestations': self._generate_manifestations(grounding_paths)
        }
        
        return reality_check
    
    def _generate_grounding_info(self, insight: Dict) -> Dict:
        """Generate grounding information for an insight"""
        # Extract concepts from insight
        concepts = []
        
        if 'source_concepts' in insight:
            concepts.extend(insight['source_concepts'])
        
        if 'target_concepts' in insight:
            concepts.extend(insight['target_concepts'])
        
        # Find directly grounded concepts
        grounded_concepts = []
        for concept_id in concepts:
            if concept_id in self.grounded_concepts:
                grounded_concepts.append({
                    'concept_id': concept_id,
                    'grounding': self.grounded_concepts[concept_id]
                })
        
        # If no directly grounded concepts, find paths to grounded concepts
        if not grounded_concepts and concepts:
            for concept_id in concepts:
                path = self.find_empirical_path(concept_id)
                if path:
                    grounded_concepts.append({
                        'concept_id': concept_id,
                        'grounding_path': path
                    })
        
        # Generate grounding info
        if grounded_concepts:
            return {
                'grounded_concepts': grounded_concepts,
                'grounding_strength': self._calculate_grounding_strength(grounded_concepts)
            }
        
        return {}
    
    def _calculate_grounding_strength(self, grounded_concepts: List[Dict]) -> float:
        """Calculate overall grounding strength"""
        if not grounded_concepts:
            return 0.0
        
        total_strength = 0.0
        
        for item in grounded_concepts:
            if 'grounding' in item:
                # Direct grounding
                concept_strength = 0.0
                for domain, info in item['grounding'].items():
                    concept_strength = max(concept_strength, info['strength'])
                
                total_strength += concept_strength
            elif 'grounding_path' in item:
                # Path to grounding
                path = item['grounding_path']
                if path:
                    # Use strength of final grounded concept, reduced by path length
                    final_concept = path[-1]
                    if 'grounding' in final_concept:
                        # Find maximum domain strength
                        max_strength = 0.0
                        for domain, info in final_concept['grounding'].items():
                            max_strength = max(max_strength, info['strength'])
                        
                        # Reduce by path length
                        path_strength = max_strength * (0.8 ** (len(path) - 1))
                        total_strength += path_strength
        
        # Normalize
        return min(1.0, total_strength / len(grounded_concepts))
    
    def _calculate_testability(self, insight: Dict, 
                             grounding_paths: List[Dict]) -> float:
        """Calculate empirical testability of an insight"""
        if not grounding_paths:
            return 0.0
        
        # Calculate based on grounding domains
        sensory_grounding = 0.0
        action_grounding = 0.0
        
        for path_info in grounding_paths:
            path = path_info['path']
            
            for concept in path:
                if 'grounding' in concept:
                    # Check for sensory and action domains
                    if 'sensory' in concept['grounding']:
                        sensory_strength = concept['grounding']['sensory']['strength']
                        sensory_grounding = max(sensory_grounding, sensory_strength)
                    
                    if 'action' in concept['grounding']:
                        action_strength = concept['grounding']['action']['strength']
                        action_grounding = max(action_grounding, action_strength)
        
        # Combine sensory and action grounding
        testability = 0.7 * sensory_grounding + 0.3 * action_grounding
        
        # Adjust based on insight type
        if 'type' in insight:
            insight_type = insight['type']
            
            # Some types are more testable than others
            if insight_type in ['causal_relationships', 'practical_consequences']:
                testability *= 1.2  # More testable
            elif insight_type in ['abstraction_path', 'existential_analysis']:
                testability *= 0.8  # Less testable
        
        return min(1.0, testability)
    
    def _generate_manifestations(self, grounding_paths: List[Dict]) -> List[Dict]:
        """Generate empirical manifestations from grounding paths"""
        manifestations = []
        
        for path_info in grounding_paths:
            path = path_info['path']
            
            for concept in path:
                if 'grounding' in concept:
                    # Extract manifestations from grounding
                    for domain, info in concept['grounding'].items():
                        for manifestation in info['manifestations']:
                            # Add source concept and domain
                            enriched = dict(manifestation)
                            enriched['source_concept'] = concept['concept_id']
                            enriched['domain'] = domain
                            
                            manifestations.append(enriched)
        
        return manifestations