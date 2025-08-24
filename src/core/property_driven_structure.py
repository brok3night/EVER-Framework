"""
Property-Driven Structure - Properties influence structural behavior of definitions
"""
import numpy as np
from typing import Dict, Any, List, Set, Tuple

class PropertyDrivenStructure:
    """
    System that allows energy properties to influence the structure and behavior
    of definitions within the EVER framework.
    """
    
    def __init__(self):
        # Structural influence mappings
        self.property_influences = {
            'magnitude': self._magnitude_influence,
            'frequency': self._frequency_influence,
            'duration': self._duration_influence,
            'vector': self._vector_influence,
            'entropy': self._entropy_influence,
            'phase': self._phase_influence,
            'boundary': self._boundary_influence
        }
        
        # Track structural modifications
        self.structural_history = {}
    
    def apply_property_structure(self, word: str, definition: Dict, energy_system: Any) -> Dict:
        """
        Apply property-driven structural modifications to a definition
        """
        # Skip if no energy signature
        if 'energy_signature' not in definition:
            return definition
            
        # Create modified definition
        modified_def = definition.copy()
        energy_sig = definition['energy_signature']
        
        # Track structural changes
        if word not in self.structural_history:
            self.structural_history[word] = []
            
        # Apply each property influence
        for prop, influence_func in self.property_influences.items():
            if prop in energy_sig:
                prop_details = energy_sig[prop]
                modified_def = influence_func(word, modified_def, prop_details, energy_system)
        
        # Record structural change
        self.structural_history[word].append({
            'timestamp': np.datetime64('now'),
            'structure_type': self._classify_structure(modified_def)
        })
        
        return modified_def
    
    def _classify_structure(self, definition: Dict) -> str:
        """Classify the structural pattern of a definition"""
        # Count connections
        connection_count = len(definition.get('related_words', []))
        
        # Check energy signature properties
        energy_sig = definition.get('energy_signature', {})
        
        # Check for dominant properties
        dominant_props = []
        for prop, details in energy_sig.items():
            if isinstance(details, dict) and 'value' in details:
                if isinstance(details['value'], (int, float)) and details['value'] > 0.7:
                    dominant_props.append(prop)
        
        # Classify based on patterns
        if connection_count > 10:
            return "highly_connected"
        elif connection_count < 2:
            return "isolated"
        elif 'entropy' in dominant_props:
            return "entropic"
        elif 'magnitude' in dominant_props:
            return "dominant"
        elif 'frequency' in dominant_props:
            return "oscillating"
        else:
            return "balanced"
    
    def _magnitude_influence(self, word: str, definition: Dict, prop_details: Dict, energy_system: Any) -> Dict:
        """
        Magnitude influences the stability and dominance of a definition
        - High magnitude: more resistant to change, influences other definitions more
        - Low magnitude: more adaptable, influenced by other definitions
        """
        magnitude = prop_details.get('value', 0.5)
        modified = definition.copy()
        
        # Influence related words based on magnitude
        if 'related_words' in modified:
            related = modified['related_words']
            
            if magnitude > 0.7:
                # High magnitude - this concept influences others more
                for related_word in related:
                    if related_word in energy_system.definitions:
                        # Increase this word's influence on related words
                        if 'influenced_by' not in energy_system.definitions[related_word]:
                            energy_system.definitions[related_word]['influenced_by'] = {}
                        
                        energy_system.definitions[related_word]['influenced_by'][word] = magnitude
            
            elif magnitude < 0.3:
                # Low magnitude - more easily influenced by stronger concepts
                if 'influenced_by' not in modified:
                    modified['influenced_by'] = {}
                
                # Find influential related words
                for related_word in related:
                    if related_word in energy_system.definitions:
                        related_magnitude = energy_system.definitions[related_word].get('energy_signature', {}).get(
                            'magnitude', {}).get('value', 0.5)
                        
                        if related_magnitude > 0.6:
                            modified['influenced_by'][related_word] = related_magnitude
        
        # Influence concept stability
        if magnitude > 0.8:
            # Very stable concept
            if 'structural_properties' not in modified:
                modified['structural_properties'] = {}
            modified['structural_properties']['stability'] = magnitude
            modified['structural_properties']['change_resistance'] = magnitude
        
        return modified
    
    def _frequency_influence(self, word: str, definition: Dict, prop_details: Dict, energy_system: Any) -> Dict:
        """
        Frequency influences how often a definition updates and its oscillation patterns
        - High frequency: updates more often, forms oscillating connections
        - Low frequency: updates rarely, forms stable connections
        """
        frequency = prop_details.get('value', 0.5)
        modified = definition.copy()
        
        # Set update frequency in structural properties
        if 'structural_properties' not in modified:
            modified['structural_properties'] = {}
            
        modified['structural_properties']['update_frequency'] = frequency
        
        # High frequency concepts form oscillating connection patterns
        if frequency > 0.7 and 'related_words' in modified:
            # Create oscillation groups - concepts that interact in waves
            oscillation_group = []
            
            for related_word in modified.get('related_words', []):
                if related_word in energy_system.definitions:
                    related_freq = energy_system.definitions[related_word].get('energy_signature', {}).get(
                        'frequency', {}).get('value', 0.5)
                    
                    # Words with complementary frequencies form oscillation groups
                    if abs(frequency - related_freq) < 0.2 or abs(frequency - related_freq) > 0.8:
                        oscillation_group.append(related_word)
            
            if oscillation_group:
                modified['structural_properties']['oscillation_group'] = oscillation_group
        
        # Low frequency concepts maintain long-term stable connections
        if frequency < 0.3:
            modified['structural_properties']['connection_stability'] = 1.0 - frequency
        
        return modified
    
    def _duration_influence(self, word: str, definition: Dict, prop_details: Dict, energy_system: Any) -> Dict:
        """
        Duration influences the persistence of definitions and their temporal connections
        - High duration: forms long-term memory connections
        - Low duration: forms transient connections
        """
        duration = prop_details.get('value', 0.5)
        modified = definition.copy()
        
        # Set memory persistence in structural properties
        if 'structural_properties' not in modified:
            modified['structural_properties'] = {}
            
        modified['structural_properties']['memory_persistence'] = duration
        
        # High duration concepts are stored in long-term structure
        if duration > 0.7:
            modified['structural_properties']['storage_type'] = 'long_term'
            
            # Form temporal connections with other persistent concepts
            temporal_connections = []
            
            for related_word in modified.get('related_words', []):
                if related_word in energy_system.definitions:
                    related_duration = energy_system.definitions[related_word].get('energy_signature', {}).get(
                        'duration', {}).get('value', 0.5)
                    
                    if related_duration > 0.6:
                        temporal_connections.append(related_word)
            
            if temporal_connections:
                modified['structural_properties']['temporal_connections'] = temporal_connections
        
        # Low duration concepts use more dynamic processing
        elif duration < 0.3:
            modified['structural_properties']['storage_type'] = 'transient'
            modified['structural_properties']['refresh_priority'] = 1.0 - duration
        
        return modified
    
    def _vector_influence(self, word: str, definition: Dict, prop_details: Dict, energy_system: Any) -> Dict:
        """
        Vector influences the directional relationships between concepts
        - Vector components determine dimensional relationships
        - Vector magnitude affects influence strength
        """
        vector = prop_details.get('value', [0.0, 0.0, 0.0])
        modified = definition.copy()
        
        if not isinstance(vector, list) or len(vector) < 3:
            return modified
            
        # Set directional properties
        if 'structural_properties' not in modified:
            modified['structural_properties'] = {}
            
        # Convert vector to directional influence
        x, y, z = vector[:3]
        
        # Calculate vector magnitude
        magnitude = (x**2 + y**2 + z**2)**0.5
        
        # Organize connections by dimension
        if 'related_words' in modified and magnitude > 0.5:
            dimensional_relations = {
                'x_dimension': [],
                'y_dimension': [],
                'z_dimension': []
            }
            
            # Analyze each related word's vector
            for related_word in modified.get('related_words', []):
                if related_word in energy_system.definitions:
                    related_vector = energy_system.definitions[related_word].get('energy_signature', {}).get(
                        'vector', {}).get('value', [0.0, 0.0, 0.0])
                    
                    if isinstance(related_vector, list) and len(related_vector) >= 3:
                        rx, ry, rz = related_vector[:3]
                        
                        # Check strongest dimensional alignment
                        x_alignment = abs(x * rx)
                        y_alignment = abs(y * ry)
                        z_alignment = abs(z * rz)
                        
                        max_alignment = max(x_alignment, y_alignment, z_alignment)
                        
                        if max_alignment > 0.5:
                            if max_alignment == x_alignment:
                                dimensional_relations['x_dimension'].append(related_word)
                            elif max_alignment == y_alignment:
                                dimensional_relations['y_dimension'].append(related_word)
                            else:
                                dimensional_relations['z_dimension'].append(related_word)
            
            # Store dimensional relations if any found
            for dim, words in dimensional_relations.items():
                if words:
                    modified['structural_properties'][dim] = words
            
            # Set overall directional influence
            modified['structural_properties']['directional_strength'] = magnitude
        
        return modified
    
    def _entropy_influence(self, word: str, definition: Dict, prop_details: Dict, energy_system: Any) -> Dict:
        """
        Entropy influences the organizational structure and connection diversity
        - High entropy: forms diverse, less predictable connections
        - Low entropy: forms ordered, predictable connections
        """
        entropy = prop_details.get('value', 0.5)
        modified = definition.copy()
        
        # Set organizational properties
        if 'structural_properties' not in modified:
            modified['structural_properties'] = {}
            
        modified['structural_properties']['organization_level'] = 1.0 - entropy
        
        # High entropy concepts form diverse connections
        if entropy > 0.7:
            modified['structural_properties']['connection_diversity'] = entropy
            
            # Enable creative connections to seemingly unrelated concepts
            if 'related_words' in modified:
                # Find potential creative connections
                all_words = set(energy_system.definitions.keys())
                current_relations = set(modified.get('related_words', []))
                
                # Potential new connections (limited to avoid explosion)
                potential_new = all_words - current_relations - {word}
                
                # The higher the entropy, the more unexpected connections
                max_creative = int(entropy * 10)
                if potential_new and max_creative > 0:
                    creative_connections = list(potential_new)
                    np.random.shuffle(creative_connections)
                    creative_connections = creative_connections[:max_creative]
                    
                    modified['structural_properties']['creative_connections'] = creative_connections
        
        # Low entropy concepts form highly organized structures
        elif entropy < 0.3:
            modified['structural_properties']['organizational_pattern'] = 'hierarchical'
            
            # Form hierarchical connections
            if 'related_words' in modified:
                hierarchical = {
                    'parent_concepts': [],
                    'child_concepts': []
                }
                
                for related_word in modified.get('related_words', []):
                    if related_word in energy_system.definitions:
                        related_magnitude = energy_system.definitions[related_word].get('energy_signature', {}).get(
                            'magnitude', {}).get('value', 0.5)
                        
                        own_magnitude = definition.get('energy_signature', {}).get(
                            'magnitude', {}).get('value', 0.5)
                        
                        if related_magnitude > own_magnitude:
                            hierarchical['parent_concepts'].append(related_word)
                        else:
                            hierarchical['child_concepts'].append(related_word)
                
                modified['structural_properties']['hierarchical_relations'] = hierarchical
        
        return modified
    
    def _phase_influence(self, word: str, definition: Dict, prop_details: Dict, energy_system: Any) -> Dict:
        """
        Phase influences the cyclical behavior and interaction timing
        - Phase determines when concepts interact most strongly
        - Phase relationships create harmonic or dissonant connections
        """
        phase = prop_details.get('value', 0.0)
        modified = definition.copy()
        
        # Set phase properties
        if 'structural_properties' not in modified:
            modified['structural_properties'] = {}
            
        modified['structural_properties']['interaction_phase'] = phase
        
        # Identify harmonic and dissonant relationships
        if 'related_words' in modified:
            harmonic = []
            dissonant = []
            
            for related_word in modified.get('related_words', []):
                if related_word in energy_system.definitions:
                    related_phase = energy_system.definitions[related_word].get('energy_signature', {}).get(
                        'phase', {}).get('value', 0.0)
                    
                    # Check phase relationship
                    phase_diff = abs((phase - related_phase) % (2 * np.pi))
                    
                    # In-phase or harmonic relationships (difference near 0, π/2, π, 3π/2)
                    harmonic_points = [0, np.pi/2, np.pi, 3*np.pi/2]
                    
                    is_harmonic = any(abs(phase_diff - p) < 0.2 for p in harmonic_points)
                    
                    if is_harmonic:
                        harmonic.append(related_word)
                    else:
                        dissonant.append(related_word)
            
            if harmonic:
                modified['structural_properties']['harmonic_relations'] = harmonic
            
            if dissonant:
                modified['structural_properties']['dissonant_relations'] = dissonant
        
        return modified
    
    def _boundary_influence(self, word: str, definition: Dict, prop_details: Dict, energy_system: Any) -> Dict:
        """
        Boundary influences the scope and limitations of concept relationships
        - Boundaries determine what connections are possible
        - Narrow boundaries create specialized concepts
        - Wide boundaries create general concepts
        """
        boundary = prop_details.get('value', [0.0, 1.0])
        modified = definition.copy()
        
        if not isinstance(boundary, list) or len(boundary) < 2:
            return modified
            
        # Extract boundary range
        lower, upper = boundary[:2]
        boundary_width = upper - lower
        
        # Set boundary properties
        if 'structural_properties' not in modified:
            modified['structural_properties'] = {}
            
        modified['structural_properties']['concept_scope'] = boundary_width
        
        # Narrow boundaries create specialized concepts
        if boundary_width < 0.3:
            modified['structural_properties']['specialization'] = 1.0 - boundary_width
            
            # Limit connections to those within boundary
            if 'related_words' in modified:
                specialized_relations = []
                
                for related_word in modified.get('related_words', []):
                    if related_word in energy_system.definitions:
                        related_boundary = energy_system.definitions[related_word].get('energy_signature', {}).get(
                            'boundary', {}).get('value', [0.0, 1.0])
                        
                        if isinstance(related_boundary, list) and len(related_boundary) >= 2:
                            r_lower, r_upper = related_boundary[:2]
                            
                            # Check for boundary overlap
                            if (lower <= r_lower <= upper) or (lower <= r_upper <= upper) or \
                               (r_lower <= lower and r_upper >= upper):
                                specialized_relations.append(related_word)
                
                modified['structural_properties']['boundary_limited_relations'] = specialized_relations
        
        # Wide boundaries create general concepts
        elif boundary_width > 0.7:
            modified['structural_properties']['generalization'] = boundary_width
            
            # Enable connections to diverse concepts
            if 'related_words' in modified:
                # Calculate boundary center
                center = (lower + upper) / 2
                
                # Group related concepts by their position relative to this center
                position_groups = {
                    'lower_concepts': [],
                    'center_concepts': [],
                    'upper_concepts': []
                }
                
                for related_word in modified.get('related_words', []):
                    if related_word in energy_system.definitions:
                        related_boundary = energy_system.definitions[related_word].get('energy_signature', {}).get(
                            'boundary', {}).get('value', [0.0, 1.0])
                        
                        if isinstance(related_boundary, list) and len(related_boundary) >= 2:
                            r_lower, r_upper = related_boundary[:2]
                            r_center = (r_lower + r_upper) / 2
                            
                            # Position relative to center
                            if r_center < center - 0.2:
                                position_groups['lower_concepts'].append(related_word)
                            elif r_center > center + 0.2:
                                position_groups['upper_concepts'].append(related_word)
                            else:
                                position_groups['center_concepts'].append(related_word)
                
                modified['structural_properties']['boundary_position_groups'] = position_groups
        
        return modified