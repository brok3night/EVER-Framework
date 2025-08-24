"""
Philosophical Meta-Layer - Implements philosophical lens tracking and fundamental operations
"""
from typing import Dict, List, Tuple, Set
import numpy as np

class PhilosophicalMetaLayer:
    """Provides meta-level philosophical context to energy signatures"""
    
    def __init__(self):
        # Fundamental philosophical operations (recurring across traditions)
        self.fundamental_operations = {
            # Meta-operations that recur in all philosophical traditions
            'abstraction': self._operation_abstraction,        # Move toward generalization
            'concretization': self._operation_concretization,  # Move toward specificity
            'negation': self._operation_negation,              # Logical opposition
            'synthesis': self._operation_synthesis,            # Combining opposing views
            'recursion': self._operation_recursion,            # Self-referential application
            'extension': self._operation_extension,            # Linear continuation of idea
            'transcendence': self._operation_transcendence,    # Moving beyond limitations
            'reduction': self._operation_reduction,            # Breaking into components
            'contextualization': self._operation_context,      # Placing in broader framework
            'decontextualization': self._operation_decontext   # Isolating from context
        }
        
        # Philosophical lens types (how concepts are viewed)
        self.lens_types = {
            'ontological': {'description': 'Concerned with nature of being/existence'},
            'epistemological': {'description': 'Concerned with knowledge/truth'},
            'ethical': {'description': 'Concerned with right/wrong/values'},
            'aesthetic': {'description': 'Concerned with beauty/art/expression'},
            'logical': {'description': 'Concerned with valid reasoning'},
            'political': {'description': 'Concerned with power/governance'},
            'metaphysical': {'description': 'Concerned with fundamental reality'},
            'phenomenological': {'description': 'Concerned with lived experience'},
            'existential': {'description': 'Concerned with meaning/purpose'}
        }
    
    def apply_meta_layer(self, energy_signature: Dict, 
                         lens_type: str = None,
                         operation_sequence: List[str] = None) -> Dict:
        """
        Apply philosophical meta-layer to energy signature
        
        Args:
            energy_signature: Base energy signature
            lens_type: Type of philosophical lens being applied
            operation_sequence: Sequence of fundamental operations to apply
        
        Returns:
            Energy signature with meta-layer information
        """
        # Create copy to avoid modifying original
        modified = self._deep_copy_signature(energy_signature)
        
        # Add meta-layer section if not present
        if 'meta' not in modified:
            modified['meta'] = {}
        
        # Add/update lens information
        if lens_type and lens_type in self.lens_types:
            modified['meta']['lens'] = {
                'type': lens_type,
                'description': self.lens_types[lens_type]['description']
            }
        
        # Apply operation sequence if provided
        if operation_sequence:
            # Record operations
            modified['meta']['operations'] = operation_sequence
            
            # Apply each operation in sequence
            for operation in operation_sequence:
                if operation in self.fundamental_operations:
                    operation_func = self.fundamental_operations[operation]
                    modified = operation_func(modified)
        
        return modified
    
    def _deep_copy_signature(self, signature: Dict) -> Dict:
        """Create deep copy of energy signature"""
        copied = {}
        
        for key, value in signature.items():
            if isinstance(value, dict):
                copied[key] = self._deep_copy_signature(value)
            elif isinstance(value, list):
                copied[key] = value.copy()
            else:
                copied[key] = value
        
        return copied
    
    def _operation_abstraction(self, signature: Dict) -> Dict:
        """Apply abstraction operation to energy signature"""
        modified = signature.copy()
        
        # Abstraction increases y-component of vector (if present)
        if 'vector' in modified and 'value' in modified['vector']:
            vector = modified['vector']['value']
            if isinstance(vector, list) and len(vector) > 1:
                # Move y-component toward abstraction (higher)
                vector[1] = min(1.0, vector[1] + 0.2)
        
        # Abstraction decreases entropy (more structured)
        if 'entropy' in modified and 'value' in modified['entropy']:
            entropy = modified['entropy']['value']
            if isinstance(entropy, (int, float)):
                modified['entropy']['value'] = max(0.1, entropy - 0.15)
        
        return modified
    
    def _operation_concretization(self, signature: Dict) -> Dict:
        """Apply concretization operation to energy signature"""
        modified = signature.copy()
        
        # Concretization decreases y-component of vector (if present)
        if 'vector' in modified and 'value' in modified['vector']:
            vector = modified['vector']['value']
            if isinstance(vector, list) and len(vector) > 1:
                # Move y-component toward concreteness (lower)
                vector[1] = max(0.0, vector[1] - 0.2)
        
        # Concretization increases magnitude (more definite)
        if 'magnitude' in modified and 'value' in modified['magnitude']:
            magnitude = modified['magnitude']['value']
            if isinstance(magnitude, (int, float)):
                modified['magnitude']['value'] = min(1.0, magnitude + 0.15)
        
        return modified
    
    def _operation_negation(self, signature: Dict) -> Dict:
        """Apply negation operation to energy signature"""
        modified = signature.copy()
        
        # Negation inverts vector orientation (if present)
        if 'vector' in modified and 'value' in modified['vector']:
            vector = modified['vector']['value']
            if isinstance(vector, list):
                # Invert around 0.5 (center point)
                for i in range(len(vector)):
                    vector[i] = 1.0 - vector[i]
        
        # Negation preserves magnitude but may increase entropy
        if 'entropy' in modified and 'value' in modified['entropy']:
            entropy = modified['entropy']['value']
            if isinstance(entropy, (int, float)):
                modified['entropy']['value'] = min(1.0, entropy + 0.1)
        
        return modified
    
    def _operation_synthesis(self, signature: Dict) -> Dict:
        """
        Apply synthesis operation to energy signature
        Note: In practice, synthesis would combine multiple signatures
        This implementation modifies a single signature for demonstration
        """
        modified = signature.copy()
        
        # Synthesis increases boundary width (if present)
        if 'boundary' in modified and 'value' in modified['boundary']:
            boundary = modified['boundary']['value']
            if isinstance(boundary, list) and len(boundary) >= 2:
                center = (boundary[0] + boundary[1]) / 2
                width = boundary[1] - boundary[0]
                new_width = width * 1.3  # Expand by 30%
                modified['boundary']['value'] = [
                    max(0.0, center - new_width/2),
                    min(1.0, center + new_width/2)
                ]
        
        # Synthesis may introduce new vector components (emergent properties)
        if 'vector' in modified and 'value' in modified['vector']:
            vector = modified['vector']['value']
            if isinstance(vector, list):
                # Add emergent component or modify existing if enough dimensions
                if len(vector) >= 3:
                    # Create emergent z-component
                    vector[2] = (vector[0] + vector[1]) / 2 + 0.1
                elif len(vector) == 2:
                    # Add new emergent dimension
                    vector.append((vector[0] + vector[1]) / 2 + 0.1)
        
        return modified
    
    def _operation_recursion(self, signature: Dict) -> Dict:
        """Apply recursion operation to energy signature"""
        modified = signature.copy()
        
        # Recursion intensifies existing properties
        for prop in ['magnitude', 'frequency', 'entropy']:
            if prop in modified and 'value' in modified[prop]:
                value = modified[prop]['value']
                if isinstance(value, (int, float)):
                    # Intensify: values < 0.5 decrease, values > 0.5 increase
                    if value > 0.5:
                        modified[prop]['value'] = min(1.0, value + (value - 0.5) * 0.4)
                    else:
                        modified[prop]['value'] = max(0.0, value - (0.5 - value) * 0.4)
        
        # Add recursion marker
        if 'meta' not in modified:
            modified['meta'] = {}
        
        if 'recursion_level' not in modified['meta']:
            modified['meta']['recursion_level'] = 1
        else:
            modified['meta']['recursion_level'] += 1
        
        return modified
    
    def _operation_extension(self, signature: Dict) -> Dict:
        """Apply extension operation to energy signature"""
        modified = signature.copy()
        
        # Extension increases duration
        if 'duration' in modified and 'value' in modified['duration']:
            duration = modified['duration']['value']
            if isinstance(duration, (int, float)):
                modified['duration']['value'] = min(1.0, duration + 0.2)
        
        # Extension may extend vector in primary direction
        if 'vector' in modified and 'value' in modified['vector']:
            vector = modified['vector']['value']
            if isinstance(vector, list) and vector:
                # Find dominant dimension
                max_dim = max(range(len(vector)), key=lambda i: abs(vector[i] - 0.5))
                # Extend in that dimension
                if vector[max_dim] > 0.5:
                    vector[max_dim] = min(1.0, vector[max_dim] + 0.15)
                else:
                    vector[max_dim] = max(0.0, vector[max_dim] - 0.15)
        
        return modified
    
    def _operation_transcendence(self, signature: Dict) -> Dict:
        """Apply transcendence operation to energy signature"""
        modified = signature.copy()
        
        # Transcendence adds new dimension to vector if possible
        if 'vector' in modified and 'value' in modified['vector']:
            vector = modified['vector']['value']
            if isinstance(vector, list):
                # Add new dimension with high value (transcendent)
                if len(vector) < 7:  # Limit to 7 dimensions
                    vector.append(0.8)  # High value in new dimension
                    
                    # Record dimension expansion
                    if 'meta' not in modified:
                        modified['meta'] = {}
                    modified['meta']['dimension_expansion'] = True
        
        # Transcendence often reduces entropy (greater coherence)
        if 'entropy' in modified and 'value' in modified['entropy']:
            entropy = modified['entropy']['value']
            if isinstance(entropy, (int, float)):
                modified['entropy']['value'] = max(0.1, entropy - 0.25)
        
        return modified
    
    def _operation_reduction(self, signature: Dict) -> Dict:
        """Apply reduction operation to energy signature"""
        modified = signature.copy()
        
        # Reduction simplifies by removing dimensions
        if 'vector' in modified and 'value' in modified['vector']:
            vector = modified['vector']['value']
            if isinstance(vector, list) and len(vector) > 3:
                # Remove least significant dimensions
                # Keep only the 3 dimensions with values furthest from 0.5
                significance = [abs(v - 0.5) for v in vector]
                indices = sorted(range(len(significance)), key=lambda i: significance[i], reverse=True)
                
                # Keep only top 3 dimensions
                new_vector = [vector[i] for i in indices[:3]]
                modified['vector']['value'] = new_vector
                
                # Record dimension reduction
                if 'meta' not in modified:
                    modified['meta'] = {}
                modified['meta']['dimension_reduction'] = True
        
        # Reduction often increases magnitude (more definite)
        if 'magnitude' in modified and 'value' in modified['magnitude']:
            magnitude = modified['magnitude']['value']
            if isinstance(magnitude, (int, float)):
                modified['magnitude']['value'] = min(1.0, magnitude + 0.15)
        
        return modified
    
    def _operation_context(self, signature: Dict) -> Dict:
        """Apply contextualization operation to energy signature"""
        modified = signature.copy()
        
        # Contextualization expands boundaries
        if 'boundary' in modified and 'value' in modified['boundary']:
            boundary = modified['boundary']['value']
            if isinstance(boundary, list) and len(boundary) >= 2:
                # Expand boundaries
                modified['boundary']['value'] = [
                    max(0.0, boundary[0] - 0.15),
                    min(1.0, boundary[1] + 0.15)
                ]
        
        # Contextualization often increases entropy (more connections)
        if 'entropy' in modified and 'value' in modified['entropy']:
            entropy = modified['entropy']['value']
            if isinstance(entropy, (int, float)):
                modified['entropy']['value'] = min(1.0, entropy + 0.2)
        
        return modified
    
    def _operation_decontext(self, signature: Dict) -> Dict:
        """Apply decontextualization operation to energy signature"""
        modified = signature.copy()
        
        # Decontextualization narrows boundaries
        if 'boundary' in modified and 'value' in modified['boundary']:
            boundary = modified['boundary']['value']
            if isinstance(boundary, list) and len(boundary) >= 2:
                center = (boundary[0] + boundary[1]) / 2
                width = boundary[1] - boundary[0]
                new_width = max(0.1, width * 0.7)  # Shrink by 30%, minimum 0.1
                modified['boundary']['value'] = [
                    max(0.0, center - new_width/2),
                    min(1.0, center + new_width/2)
                ]
        
        # Decontextualization reduces entropy (fewer connections)
        if 'entropy' in modified and 'value' in modified['entropy']:
            entropy = modified['entropy']['value']
            if isinstance(entropy, (int, float)):
                modified['entropy']['value'] = max(0.1, entropy - 0.2)
        
        return modified