"""
Fundamental Philosophy System - Uses fundamental operations instead of named philosophical methods
"""
from typing import Dict, List, Tuple, Set
import numpy as np
from src.reasoning.philosophical_meta import PhilosophicalMetaLayer

class FundamentalPhilosophy:
    """Implements philosophical reasoning through fundamental operations"""
    
    def __init__(self, energy_system):
        self.energy = energy_system
        self.meta_layer = PhilosophicalMetaLayer()
        
        # Operation patterns that approximately reconstruct traditional methods
        # Instead of separate implementations, we use operation sequences
        self.reasoning_patterns = {
            # Traditional reasoning as operation sequences
            'dialectical': ['negation', 'synthesis'],
            'deductive': ['abstraction', 'extension', 'concretization'],
            'inductive': ['concretization', 'abstraction'],
            'abductive': ['contextualization', 'extension'],
            'analogical': ['abstraction', 'extension', 'concretization'],
            'phenomenological': ['reduction', 'contextualization'],
            'hermeneutic': ['contextualization', 'recursion'],
            'critical': ['negation', 'contextualization'],
            'pragmatic': ['concretization', 'extension'],
            'existential': ['transcendence', 'recursion']
        }
        
        # Track philosophical growth
        self.concept_connections = {}  # Much lighter weight now
        self.operation_efficacy = {op: 0.5 for op in self.meta_layer.fundamental_operations}
        
        # Concept lenses (track how concepts are being viewed)
        self.concept_lenses = {}
    
    def apply_reasoning(self, reasoning_type: str, concepts: List[Dict], 
                        lens_type: str = None) -> Dict:
        """
        Apply philosophical reasoning through fundamental operations
        
        Args:
            reasoning_type: Type of reasoning to apply
            concepts: List of concepts to reason with
            lens_type: Philosophical lens to apply
            
        Returns:
            Resulting concept with transformed energy signature
        """
        # Get operation sequence for this reasoning type
        operation_sequence = self.reasoning_patterns.get(reasoning_type, [])
        
        if not operation_sequence:
            # Custom sequence or invalid type - use default
            operation_sequence = ['abstraction', 'extension']
        
        # Extract energy signatures
        energy_signatures = [c.get('energy_signature', {}) for c in concepts]
        
        # Combine base signatures (simple averaging for demonstration)
        base_signature = self._combine_signatures(energy_signatures)
        
        # Apply operation sequence through meta-layer
        transformed_signature = self.meta_layer.apply_meta_layer(
            base_signature,
            lens_type=lens_type,
            operation_sequence=operation_sequence
        )
        
        # Create result concept
        result = {
            'name': f"Result of {reasoning_type} reasoning",
            'energy_signature': transformed_signature,
            'source_concepts': [c.get('name', 'unnamed') for c in concepts],
            'reasoning_type': reasoning_type,
            'lens': lens_type,
            'operations_applied': operation_sequence
        }
        
        # Update concept connections
        self._update_concept_connections(concepts, result, operation_sequence)
        
        # Track lens application
        for concept in concepts:
            concept_name = concept.get('name')
            if concept_name:
                if concept_name not in self.concept_lenses:
                    self.concept_lenses[concept_name] = set()
                if lens_type:
                    self.concept_lenses[concept_name].add(lens_type)
        
        return result
    
    def custom_operation_sequence(self, concepts: List[Dict], 
                                 operations: List[str],
                                 lens_type: str = None) -> Dict:
        """
        Apply a custom sequence of fundamental operations
        
        Args:
            concepts: List of concepts to transform
            operations: Sequence of operations to apply
            lens_type: Philosophical lens to apply
        """
        # Validate operations
        valid_operations = [op for op in operations 
                           if op in self.meta_layer.fundamental_operations]
        
        if not valid_operations:
            return {'error': 'No valid operations specified'}
        
        # Extract energy signatures
        energy_signatures = [c.get('energy_signature', {}) for c in concepts]
        
        # Combine base signatures
        base_signature = self._combine_signatures(energy_signatures)
        
        # Apply operation sequence through meta-layer
        transformed_signature = self.meta_layer.apply_meta_layer(
            base_signature,
            lens_type=lens_type,
            operation_sequence=valid_operations
        )
        
        # Create result concept
        result = {
            'name': f"Result of custom operation sequence",
            'energy_signature': transformed_signature,
            'source_concepts': [c.get('name', 'unnamed') for c in concepts],
            'lens': lens_type,
            'operations_applied': valid_operations
        }
        
        # Update concept connections and operation efficacy
        self._update_concept_connections(concepts, result, valid_operations)
        
        return result
    
    def _combine_signatures(self, signatures: List[Dict]) -> Dict:
        """Combine multiple energy signatures into one base signature"""
        if not signatures:
            return {}
            
        # Start with first signature
        combined = self.meta_layer._deep_copy_signature(signatures[0])
        
        if len(signatures) == 1:
            return combined
            
        # Process each property
        for prop in combined:
            if prop == 'meta':
                continue  # Skip meta information
                
            if 'value' in combined[prop]:
                value = combined[prop]['value']
                
                # Collect values from all signatures
                values = []
                for sig in signatures:
                    if prop in sig and 'value' in sig[prop]:
                        values.append(sig[prop]['value'])
                
                if not values:
                    continue
                    
                # Process based on value type
                if isinstance(value, (int, float)):
                    # Average scalar values
                    avg_value = sum(v for v in values if isinstance(v, (int, float))) / len(values)
                    combined[prop]['value'] = avg_value
                    
                elif isinstance(value, list):
                    # For vectors, ensure all have same length
                    vector_values = [v for v in values if isinstance(v, list) and len(v) == len(value)]
                    
                    if vector_values:
                        # Average each component
                        avg_vector = []
                        for i in range(len(value)):
                            component_values = [v[i] for v in vector_values]
                            avg_vector.append(sum(component_values) / len(component_values))
                        
                        combined[prop]['value'] = avg_vector
        
        return combined
    
    def _update_concept_connections(self, concepts: List[Dict], result: Dict, 
                                   operations: List[str]) -> None:
        """Update concept connections based on operations"""
        # Extract concept names
        concept_names = [c.get('name') for c in concepts if c.get('name')]
        result_name = result.get('name')
        
        if not concept_names or not result_name:
            return
        
        # Update connections (much lighter weight than before)
        for name in concept_names:
            if name not in self.concept_connections:
                self.concept_connections[name] = set()
            
            # Connect concepts to result
            self.concept_connections[name].add(result_name)
        
        # Update operation efficacy based on result quality
        # (In a real system, this would use feedback)
        for operation in operations:
            if operation in self.operation_efficacy:
                # Small improvement from practice (can be refined with feedback)
                current = self.operation_efficacy[operation]
                self.operation_efficacy[operation] = min(0.95, current + 0.01 * (1 - current))
    
    def get_concept_lenses(self, concept_name: str) -> Set[str]:
        """Get philosophical lenses applied to a concept"""
        return self.concept_lenses.get(concept_name, set())
    
    def discover_perspectives(self, concept: Dict, max_lenses: int = 3) -> List[Dict]:
        """
        Discover new philosophical perspectives on a concept
        by applying different lenses
        """
        concept_name = concept.get('name')
        energy_signature = concept.get('energy_signature', {})
        
        if not concept_name or not energy_signature:
            return []
        
        # Get currently applied lenses
        current_lenses = self.get_concept_lenses(concept_name)
        
        # Find unused lenses
        available_lenses = set(self.meta_layer.lens_types.keys()) - current_lenses
        
        if not available_lenses:
            return []
        
        # Select top lenses based on concept properties
        lens_scores = {}
        
        for lens in available_lenses:
            # Score each lens based on concept properties
            score = self._score_lens_compatibility(energy_signature, lens)
            lens_scores[lens] = score
        
        # Select top scoring lenses
        top_lenses = sorted(lens_scores.items(), key=lambda x: x[1], reverse=True)[:max_lenses]
        
        # Generate perspectives with these lenses
        perspectives = []
        
        for lens, score in top_lenses:
            # Select most effective operations for this lens
            operations = self._select_operations_for_lens(lens)
            
            # Apply operations through meta-layer
            transformed = self.meta_layer.apply_meta_layer(
                energy_signature,
                lens_type=lens,
                operation_sequence=operations
            )
            
            perspectives.append({
                'name': f"{lens} perspective on {concept_name}",
                'energy_signature': transformed,
                'lens': lens,
                'compatibility_score': score,
                'operations': operations,
                'description': self.meta_layer.lens_types[lens]['description']
            })
        
        return perspectives
    
    def _score_lens_compatibility(self, signature: Dict, lens: str) -> float:
        """Score compatibility between energy signature and philosophical lens"""
        # Simple compatibility scoring based on energy properties
        score = 0.5  # Base score
        
        # Different lenses align with different energy properties
        if 'vector' in signature and 'value' in signature['vector']:
            vector = signature['vector']['value']
            if isinstance(vector, list) and len(vector) > 1:
                # Y-component (abstraction) alignment
                if lens in ['metaphysical', 'epistemological', 'ontological']:
                    # These lenses work better with abstract concepts
                    score += (vector[1] - 0.5) * 0.4  # Higher y-component = better match
                elif lens in ['phenomenological', 'ethical', 'political']:
                    # These work better with middle-range abstraction
                    score += (1 - abs(vector[1] - 0.6)) * 0.3  # Closer to 0.6 = better match
        
        if 'magnitude' in signature and 'value' in signature['magnitude']:
            magnitude = signature['magnitude']['value']
            if isinstance(magnitude, (int, float)):
                # Magnitude (definiteness) alignment
                if lens in ['logical', 'epistemological']:
                    # These lenses work better with high magnitude
                    score += magnitude * 0.3
                elif lens in ['existential', 'aesthetic']:
                    # These work better with lower magnitude
                    score += (1 - magnitude) * 0.3
        
        return min(1.0, max(0.0, score))
    
    def _select_operations_for_lens(self, lens: str) -> List[str]:
        """Select most appropriate operations for a philosophical lens"""
        # Different lenses work well with different operations
        lens_operation_affinities = {
            'ontological': ['abstraction', 'negation', 'transcendence'],
            'epistemological': ['reduction', 'extension', 'recursion'],
            'ethical': ['contextualization', 'negation', 'synthesis'],
            'aesthetic': ['transcendence', 'synthesis', 'contextualization'],
            'logical': ['reduction', 'extension', 'negation'],
            'political': ['contextualization', 'synthesis', 'extension'],
            'metaphysical': ['transcendence', 'abstraction', 'recursion'],
            'phenomenological': ['reduction', 'contextualization', 'concretization'],
            'existential': ['transcendence', 'negation', 'recursion']
        }
        
        # Get operations with affinity for this lens
        affinity_operations = lens_operation_affinities.get(lens, 
            ['abstraction', 'extension', 'synthesis'])
        
        # Return operations sorted by current efficacy
        return sorted(affinity_operations, 
                     key=lambda op: self.operation_efficacy.get(op, 0.5),
                     reverse=True)