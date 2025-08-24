"""
Network Reasoning - Philosophical reasoning through resonance networks
"""
from typing import Dict, List, Set, Tuple, Any
import numpy as np

class NetworkReasoning:
    """Performs philosophical reasoning through resonance networks"""
    
    def __init__(self, resonance_network, dynamic_primitives):
        self.network = resonance_network
        self.primitives = dynamic_primitives
        
        # Known reasoning strategies (paths through the network)
        self.reasoning_strategies = {
            'similarity': self._reason_by_similarity,
            'contrast': self._reason_by_contrast,
            'analogy': self._reason_by_analogy,
            'composition': self._reason_by_composition,
            'abstraction': self._reason_by_abstraction,
            'concretization': self._reason_by_concretization,
            'causation': self._reason_by_causation,
            'dialectic': self._reason_by_dialectic
        }
    
    def navigate_concept_space(self, starting_concepts: List[str],
                             target_energy: Dict = None,
                             philosophical_framework: str = None,
                             max_steps: int = 5) -> Dict:
        """
        Navigate through concept space using resonance
        
        Args:
            starting_concepts: Starting concept IDs
            target_energy: Optional target energy signature
            philosophical_framework: Optional framework to guide navigation
            max_steps: Maximum navigation steps
            
        Returns:
            Navigation results
        """
        if not starting_concepts:
            return {'path': [], 'insights': []}
        
        # Track the navigation path
        path = []
        for concept_id in starting_concepts:
            if concept_id in self.network.concepts:
                path.append({
                    'concept_id': concept_id,
                    'energy': self.network.concepts[concept_id]
                })
        
        # Initialize current state
        current_concepts = starting_concepts
        current_field = self.network.get_resonant_field(current_concepts)
        
        # Navigation loop
        insights = []
        
        for step in range(max_steps):
            # Select next step based on current state
            next_step = self._select_next_step(
                current_concepts, current_field, target_energy, philosophical_framework
            )
            
            if not next_step:
                break
            
            # Apply step
            new_concepts = next_step['concepts']
            reasoning_type = next_step['reasoning_type']
            
            # Generate insight from this step
            insight = self._generate_insight(
                current_concepts, new_concepts, reasoning_type
            )
            
            if insight:
                insights.append(insight)
            
            # Update path
            for concept_id in new_concepts:
                if concept_id in self.network.concepts and concept_id not in current_concepts:
                    path.append({
                        'concept_id': concept_id,
                        'energy': self.network.concepts[concept_id],
                        'reasoning_type': reasoning_type
                    })
            
            # Update current state
            current_concepts = new_concepts
            current_field = self.network.get_resonant_field(current_concepts)
            
            # Check if we've reached the target
            if target_energy and self._check_target_reached(current_field, target_energy):
                break
        
        # Create navigation results
        results = {
            'path': path,
            'insights': insights,
            'final_concepts': current_concepts,
            'final_field': current_field
        }
        
        return results
    
    def apply_philosophical_framework(self, concept_ids: List[str],
                                   framework: str) -> Dict:
        """
        Apply a philosophical framework to a set of concepts
        
        Args:
            concept_ids: Concepts to analyze
            framework: Philosophical framework to apply
            
        Returns:
            Analysis results
        """
        if not concept_ids:
            return {}
        
        # Get resonant field
        field = self.network.get_resonant_field(concept_ids)
        
        # Get concept energies
        energies = []
        for concept_id in concept_ids:
            if concept_id in self.network.concepts:
                energies.append(self.network.concepts[concept_id])
        
        if not energies:
            return {}
        
        # Apply framework primitives to each energy
        transformed_energies = []
        for energy in energies:
            # Apply framework through dynamic primitives
            transformed = self.primitives.apply_framework(framework, energy)
            transformed_energies.append(transformed)
        
        # Generate insights based on framework
        insights = self._generate_framework_insights(
            concept_ids, framework, transformed_energies
        )
        
        # Find new resonant concepts after transformation
        new_resonant_concepts = set()
        for energy in transformed_energies:
            resonant = self.network.find_resonant_concepts(energy, top_n=3)
            for concept_id, strength in resonant:
                if concept_id not in concept_ids and strength > 0.7:
                    new_resonant_concepts.add(concept_id)
        
        # Create analysis results
        results = {
            'original_concepts': concept_ids,
            'framework': framework,
            'transformed_energies': transformed_energies,
            'insights': insights,
            'new_resonant_concepts': list(new_resonant_concepts)
        }
        
        return results
    
    def reason_across_concepts(self, concept_ids: List[str],
                             reasoning_strategy: str) -> Dict:
        """
        Apply a specific reasoning strategy across concepts
        
        Args:
            concept_ids: Concepts to reason across
            reasoning_strategy: Strategy to use
            
        Returns:
            Reasoning results
        """
        if not concept_ids or reasoning_strategy not in self.reasoning_strategies:
            return {}
        
        # Get reasoning function
        reason_func = self.reasoning_strategies[reasoning_strategy]
        
        # Apply reasoning
        result = reason_func(concept_ids)
        
        return result
    
    def discover_insights(self, concept_ids: List[str]) -> List[Dict]:
        """
        Discover insights about a set of concepts
        
        Args:
            concept_ids: Concepts to analyze
            
        Returns:
            List of insights
        """
        insights = []
        
        # Try each reasoning strategy
        for strategy_name in self.reasoning_strategies:
            result = self.reason_across_concepts(concept_ids, strategy_name)
            
            if result and 'insights' in result:
                for insight in result['insights']:
                    insight['strategy'] = strategy_name
                    insights.append(insight)
        
        return insights
    
    def _select_next_step(self, current_concepts: List[str],
                        current_field: Dict[str, float],
                        target_energy: Dict = None,
                        framework: str = None) -> Dict:
        """Select next navigation step"""
        # Get all concepts in the field
        field_concepts = list(current_field.keys())
        
        # Filter out current concepts
        new_concepts = [c for c in field_concepts if c not in current_concepts]
        
        if not new_concepts:
            return None
        
        # Score potential next steps
        candidates = []
        
        for concept_id in new_concepts:
            # Skip if concept doesn't exist
            if concept_id not in self.network.concepts:
                continue
            
            # Calculate base score from field activation
            score = current_field[concept_id]
            
            # Adjust score based on target if provided
            if target_energy and concept_id in self.network.concepts:
                resonance = self.network._calculate_direct_resonance(
                    self.network.concepts[concept_id], target_energy
                )
                score += resonance * 0.5
            
            # Adjust score based on framework if provided
            if framework:
                # This would use the primitives system to calculate framework alignment
                # Simplified implementation
                framework_bonus = 0.2
                score += framework_bonus
            
            # Determine reasoning type
            reasoning_type = self._determine_reasoning_type(
                current_concepts, concept_id
            )
            
            candidates.append({
                'concept_id': concept_id,
                'score': score,
                'reasoning_type': reasoning_type
            })
        
        if not candidates:
            return None
        
        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Select top candidate
        selected = candidates[0]
        
        # Create step
        step = {
            'concepts': current_concepts + [selected['concept_id']],
            'reasoning_type': selected['reasoning_type']
        }
        
        return step
    
    def _determine_reasoning_type(self, current_concepts: List[str],
                                candidate: str) -> str:
        """Determine the type of reasoning that leads to a candidate"""
        # Default reasoning type
        reasoning_type = 'similarity'
        
        # Check connections from current concepts to candidate
        for concept_id in current_concepts:
            if concept_id in self.network.connections:
                if candidate in self.network.connections[concept_id]:
                    conn_type = self.network.connections[concept_id][candidate]['type']
                    
                    # Map connection type to reasoning type
                    if conn_type == 'contrast':
                        reasoning_type = 'contrast'
                    elif conn_type == 'analogy':
                        reasoning_type = 'analogy'
                    elif conn_type == 'part_of':
                        reasoning_type = 'composition'
                    elif conn_type == 'abstraction':
                        reasoning_type = 'abstraction'
                    elif conn_type == 'example_of':
                        reasoning_type = 'concretization'
                    elif conn_type == 'causes':
                        reasoning_type = 'causation'
                    elif conn_type == 'opposes':
                        reasoning_type = 'dialectic'
                    
                    # Once we find a meaningful connection, return it
                    if reasoning_type != 'similarity':
                        return reasoning_type
        
        return reasoning_type
    
    def _check_target_reached(self, current_field: Dict[str, float],
                            target_energy: Dict) -> bool:
        """Check if navigation has reached the target"""
        # Find concepts with highest activation
        if not current_field:
            return False
        
        top_concepts = sorted(
            current_field.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]  # Top 3 concepts
        
        # Check resonance with target
        for concept_id, activation in top_concepts:
            if concept_id in self.network.concepts:
                resonance = self.network._calculate_direct_resonance(
                    self.network.concepts[concept_id], target_energy
                )
                
                if resonance > 0.8:  # High resonance threshold
                    return True
        
        return False
    
    def _generate_insight(self, source_concepts: List[str],
                        target_concepts: List[str],
                        reasoning_type: str) -> Dict:
        """Generate insight from a reasoning step"""
        # Create basic insight template
        insight = {
            'type': reasoning_type,
            'source_concepts': source_concepts,
            'target_concepts': [c for c in target_concepts if c not in source_concepts],
            'description': ""
        }
        
        # Generate description based on reasoning type
        if reasoning_type == 'similarity':
            insight['description'] = "These concepts share resonant patterns."
        elif reasoning_type == 'contrast':
            insight['description'] = "These concepts represent contrasting perspectives."
        elif reasoning_type == 'analogy':
            insight['description'] = "These concepts demonstrate analogous structures."
        elif reasoning_type == 'composition':
            insight['description'] = "These concepts form a compositional relationship."
        elif reasoning_type == 'abstraction':
            insight['description'] = "Moving to a higher level of abstraction."
        elif reasoning_type == 'concretization':
            insight['description'] = "Moving to more concrete examples."
        elif reasoning_type == 'causation':
            insight['description'] = "These concepts have causal relationships."
        elif reasoning_type == 'dialectic':
            insight['description'] = "These concepts form a dialectical tension."
        
        return insight
    
    def _generate_framework_insights(self, concept_ids: List[str],
                                   framework: str,
                                   transformed_energies: List[Dict]) -> List[Dict]:
        """Generate insights based on philosophical framework"""
        insights = []
        
        # Basic framework insight
        insights.append({
            'type': 'framework_application',
            'framework': framework,
            'description': f"Applied {framework} framework to these concepts."
        })
        
        # Framework-specific insights
        if framework == 'dialectical':
            insights.append({
                'type': 'dialectical_tension',
                'description': "Identified dialectical tensions between these concepts."
            })
        elif framework == 'existentialist':
            insights.append({
                'type': 'existential_meaning',
                'description': "Examined existential implications of these concepts."
            })
        elif framework == 'pragmatic':
            insights.append({
                'type': 'practical_consequences',
                'description': "Analyzed practical consequences of these concepts."
            })
        
        return insights
    
    # Reasoning strategy implementations
    
    def _reason_by_similarity(self, concept_ids: List[str]) -> Dict:
        """Reason across concepts by finding similarities"""
        if len(concept_ids) < 2:
            return {}
        
        # Find common resonant concepts
        fields = []
        for concept_id in concept_ids:
            field = self.network.get_resonant_field([concept_id], depth=1)
            fields.append(field)
        
        # Find concepts that appear in multiple fields
        common_concepts = {}
        for field in fields:
            for concept_id, activation in field.items():
                if concept_id not in concept_ids:  # Exclude original concepts
                    common_concepts[concept_id] = common_concepts.get(concept_id, 0) + 1
        
        # Find concepts that appear in all fields
        shared_concepts = [
            concept_id for concept_id, count in common_concepts.items()
            if count == len(concept_ids)
        ]
        
        # Generate insights
        insights = []
        if shared_concepts:
            insights.append({
                'type': 'shared_resonance',
                'shared_concepts': shared_concepts,
                'description': "These concepts share resonant patterns through their connections."
            })
        
        return {
            'shared_concepts': shared_concepts,
            'insights': insights
        }
    
    def _reason_by_contrast(self, concept_ids: List[str]) -> Dict:
        """Reason across concepts by finding contrasts"""
        if len(concept_ids) < 2:
            return {}
        
        # Find contrasting connections
        contrasts = []
        
        for i, concept1 in enumerate(concept_ids):
            for concept2 in concept_ids[i+1:]:
                # Check if concepts are connected by contrast
                if (concept1 in self.network.connections and 
                    concept2 in self.network.connections[concept1]):
                    connection = self.network.connections[concept1][concept2]
                    if connection['type'] == 'contrast' or connection['type'] == 'opposes':
                        contrasts.append((concept1, concept2, connection))
        
        # Generate insights
        insights = []
        if contrasts:
            insights.append({
                'type': 'conceptual_contrast',
                'contrasts': [(c1, c2) for c1, c2, _ in contrasts],
                'description': "These concepts demonstrate important contrasts."
            })
        
        return {
            'contrasts': contrasts,
            'insights': insights
        }
    
    def _reason_by_analogy(self, concept_ids: List[str]) -> Dict:
        """Reason across concepts by finding analogies"""
        # Analogical reasoning is more complex - would be implemented in full system
        insights = []
        
        # Placeholder implementation
        insights.append({
            'type': 'analogical_structure',
            'description': "These concepts may have analogical relationships."
        })
        
        return {
            'insights': insights
        }
    
    def _reason_by_composition(self, concept_ids: List[str]) -> Dict:
        """Reason across concepts by analyzing composition relationships"""
        # Composition reasoning - would be implemented in full system
        insights = []
        
        # Placeholder implementation
        insights.append({
            'type': 'compositional_structure',
            'description': "These concepts may form compositional relationships."
        })
        
        return {
            'insights': insights
        }
    
    def _reason_by_abstraction(self, concept_ids: List[str]) -> Dict:
        """Reason by moving to higher levels of abstraction"""
        # Abstraction reasoning - would be implemented in full system
        insights = []
        
        # Placeholder implementation
        insights.append({
            'type': 'abstraction_path',
            'description': "These concepts can be understood through higher abstractions."
        })
        
        return {
            'insights': insights
        }
    
    def _reason_by_concretization(self, concept_ids: List[str]) -> Dict:
        """Reason by moving to more concrete examples"""
        # Concretization reasoning - would be implemented in full system
        insights = []
        
        # Placeholder implementation
        insights.append({
            'type': 'concretization_examples',
            'description': "These concepts can be illustrated through concrete examples."
        })
        
        return {
            'insights': insights
        }
    
    def _reason_by_causation(self, concept_ids: List[str]) -> Dict:
        """Reason by analyzing causal relationships"""
        # Causation reasoning - would be implemented in full system
        insights = []
        
        # Placeholder implementation
        insights.append({
            'type': 'causal_relationships',
            'description': "These concepts may have causal relationships."
        })
        
        return {
            'insights': insights
        }
    
    def _reason_by_dialectic(self, concept_ids: List[str]) -> Dict:
        """Reason by analyzing dialectical tensions"""
        # Dialectical reasoning - would be implemented in full system
        insights = []
        
        # Placeholder implementation
        insights.append({
            'type': 'dialectical_tensions',
            'description': "These concepts may form dialectical tensions."
        })
        
        return {
            'insights': insights
        }