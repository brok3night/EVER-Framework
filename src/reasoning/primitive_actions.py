"""
Primitive Reasoning Actions - Basic operations that can be combined to represent philosophical reasoning
"""
from typing import Dict, List, Set, Tuple
import numpy as np

class PrimitiveActions:
    """Primitive reasoning actions that serve as building blocks for philosophical reasoning"""
    
    def __init__(self, energy_system):
        self.energy = energy_system
        
        # Core primitive operations
        self.primitives = {
            'shift_up': self._shift_energy_up,         # Move energy signature upward (abstraction)
            'shift_down': self._shift_energy_down,     # Move energy signature downward (concretization)
            'shift_left': self._shift_energy_left,     # Move energy signature leftward
            'shift_right': self._shift_energy_right,   # Move energy signature rightward
            'invert': self._invert_energy,             # Invert energy signature (negation)
            'merge': self._merge_energies,             # Merge energy signatures (synthesis)
            'oscillate': self._oscillate_energy,       # Create oscillation in energy signature
            'expand': self._expand_boundaries,         # Expand energy boundaries (open perspective)
            'contract': self._contract_boundaries,     # Contract energy boundaries (focus)
            'bifurcate': self._bifurcate_energy,       # Split energy into dual paths (dialectical)
            'resonate': self._amplify_resonance,       # Amplify resonance with other energies
            'dampen': self._dampen_signature,          # Reduce intensity of energy signature
            'loop': self._create_feedback_loop         # Create feedback loop (recursion)
        }
        
        # Philosophical approaches as combinations of primitives
        self.philosophical_patterns = {
            'dialectical': ['bifurcate', 'invert', 'merge'],
            'deductive': ['shift_up', 'expand', 'shift_down'],
            'inductive': ['shift_down', 'expand', 'shift_up'],
            'abductive': ['resonate', 'shift_right', 'expand'],
            'analogical': ['shift_up', 'shift_right', 'shift_down'],
            'phenomenological': ['contract', 'shift_down', 'expand'],
            'existential': ['shift_up', 'invert', 'loop'],
            'pragmatic': ['shift_down', 'resonate', 'shift_right'],
            'critical': ['invert', 'expand', 'resonate'],
            'hermeneutic': ['loop', 'expand', 'shift_right']
        }
        
        # Action efficacy tracking (improves with use)
        self.primitive_efficacy = {primitive: 0.5 for primitive in self.primitives}
        
        # Resonance thresholds for triggering primitives
        self.resonance_thresholds = {
            'shift_up': 0.6,      # Requires moderate resonance with abstract concepts
            'shift_down': 0.6,    # Requires moderate resonance with concrete concepts
            'invert': 0.7,        # Requires stronger resonance for negation
            'merge': 0.65,        # Requires moderate-high resonance for synthesis
            'loop': 0.75,         # Requires strong resonance for recursive operations
            'bifurcate': 0.6,     # Requires moderate resonance for dialectical thinking
            'resonate': 0.5       # Requires less resonance to amplify existing resonance
        }
        
        # Discovered action sequences (learned through experience)
        self.discovered_sequences = []
    
    def suggest_actions(self, energy_signature: Dict, 
                       context_energies: List[Dict] = None,
                       philosophical_bias: str = None) -> List[str]:
        """
        Suggest primitive actions based on energy signature and context
        
        Args:
            energy_signature: Current energy signature
            context_energies: Other energy signatures in context
            philosophical_bias: Optional bias toward a philosophical approach
        """
        suggested_actions = []
        
        # Check resonance with fundamental operations
        resonances = self._calculate_primitive_resonances(energy_signature)
        
        # Add actions that meet resonance threshold
        for primitive, resonance in resonances.items():
            threshold = self.resonance_thresholds.get(primitive, 0.6)
            
            if resonance >= threshold:
                suggested_actions.append(primitive)
        
        # If we have a philosophical bias, prioritize its pattern
        if philosophical_bias and philosophical_bias in self.philosophical_patterns:
            pattern_actions = self.philosophical_patterns[philosophical_bias]
            
            # Add any missing pattern actions
            for action in pattern_actions:
                if action not in suggested_actions:
                    suggested_actions.append(action)
            
            # Sort to prioritize pattern actions
            suggested_actions.sort(key=lambda x: pattern_actions.index(x) if x in pattern_actions else 99)
        
        # Check for discovered sequences that might apply
        relevant_sequences = self._find_relevant_sequences(energy_signature)
        if relevant_sequences:
            # Add first action from most relevant sequence
            best_sequence = relevant_sequences[0]['sequence']
            if best_sequence and best_sequence[0] not in suggested_actions:
                suggested_actions.insert(0, best_sequence[0])
        
        # Ensure we have at least some actions
        if not suggested_actions:
            # Default to basic operations
            suggested_actions = ['shift_up', 'shift_down', 'expand']
        
        return suggested_actions
    
    def apply_action(self, action: str, energy_signature: Dict, 
                    secondary_signature: Dict = None) -> Dict:
        """
        Apply a primitive action to an energy signature
        
        Args:
            action: Name of primitive action to apply
            energy_signature: Energy signature to modify
            secondary_signature: Secondary signature for actions that need two inputs
        """
        if action not in self.primitives:
            return energy_signature
        
        # Get action function
        action_func = self.primitives[action]
        
        # Apply the action
        if action in ['merge', 'resonate'] and secondary_signature:
            # Actions that require two signatures
            result = action_func(energy_signature, secondary_signature)
        else:
            # Actions that work on a single signature
            result = action_func(energy_signature)
        
        # Update efficacy based on result quality
        # This is simplified - in a real system we'd use feedback
        self.primitive_efficacy[action] = min(0.95, self.primitive_efficacy[action] + 0.01)
        
        return result
    
    def apply_sequence(self, actions: List[str], energy_signature: Dict,
                      context_energies: List[Dict] = None) -> Dict:
        """
        Apply a sequence of primitive actions
        
        Args:
            actions: List of primitive actions to apply
            energy_signature: Starting energy signature
            context_energies: Other energy signatures in context
        """
        result = dict(energy_signature)  # Start with copy of original
        
        # Track sequence quality
        sequence_quality = 1.0
        
        # Apply each action in sequence
        for i, action in enumerate(actions):
            if action in self.primitives:
                # For actions that need secondary input
                if action in ['merge', 'resonate'] and context_energies:
                    # Use the most relevant context energy
                    best_context = self._find_best_context(result, context_energies)
                    if best_context:
                        result = self.apply_action(action, result, best_context)
                    else:
                        sequence_quality *= 0.8  # Penalize for missing context
                else:
                    # Standard single-input action
                    result = self.apply_action(action, result)
        
        # Record this sequence if it's novel
        self._record_action_sequence(actions, energy_signature, result, sequence_quality)
        
        return result
    
    def get_philosophical_actions(self, approach: str) -> List[str]:
        """Get the primitive actions for a philosophical approach"""
        return self.philosophical_patterns.get(approach, [])
    
    def _calculate_primitive_resonances(self, energy_signature: Dict) -> Dict:
        """Calculate resonance between energy signature and primitive operations"""
        resonances = {}
        
        # Different primitives resonate with different energy characteristics
        
        # Abstraction (shift_up) resonates with high-frequency, low-entropy signatures
        if 'frequency' in energy_signature and 'entropy' in energy_signature:
            freq = energy_signature['frequency'].get('value', 0.5)
            entropy = energy_signature['entropy'].get('value', 0.5)
            resonances['shift_up'] = (freq * 0.7 + (1 - entropy) * 0.3)
        
        # Concretization (shift_down) resonates with high-magnitude, high-entropy signatures
        if 'magnitude' in energy_signature and 'entropy' in energy_signature:
            mag = energy_signature['magnitude'].get('value', 0.5)
            entropy = energy_signature['entropy'].get('value', 0.5)
            resonances['shift_down'] = (mag * 0.7 + entropy * 0.3)
        
        # Negation (invert) resonates with extreme vector components
        if 'vector' in energy_signature and 'value' in energy_signature['vector']:
            vector = energy_signature['vector']['value']
            if isinstance(vector, list) and vector:
                # Calculate how far components are from center (0.5)
                extremity = sum(abs(v - 0.5) for v in vector) / len(vector)
                resonances['invert'] = extremity
        
        # Synthesis (merge) resonates with balanced signatures
        if all(p in energy_signature for p in ['magnitude', 'frequency', 'entropy']):
            mag = energy_signature['magnitude'].get('value', 0.5)
            freq = energy_signature['frequency'].get('value', 0.5)
            entropy = energy_signature['entropy'].get('value', 0.5)
            
            # Balance measure - how close properties are to each other
            balance = 1.0 - max(abs(mag - freq), abs(mag - entropy), abs(freq - entropy))
            resonances['merge'] = balance
        
        # Recursion (loop) resonates with high-frequency, high-magnitude signatures
        if 'frequency' in energy_signature and 'magnitude' in energy_signature:
            freq = energy_signature['frequency'].get('value', 0.5)
            mag = energy_signature['magnitude'].get('value', 0.5)
            resonances['loop'] = (freq * 0.6 + mag * 0.4)
        
        # Dialectical (bifurcate) resonates with signatures having opposing components
        if 'vector' in energy_signature and 'value' in energy_signature['vector']:
            vector = energy_signature['vector']['value']
            if isinstance(vector, list) and len(vector) >= 2:
                # Look for opposing components (one high, one low)
                opposition = 0
                for i in range(len(vector) - 1):
                    opposition = max(opposition, abs(vector[i] - vector[i+1]))
                resonances['bifurcate'] = opposition
        
        # Resonance amplification resonates with all signatures
        resonances['resonate'] = 0.7  # Generally available
        
        # For other primitives, use moderate default resonance
        for primitive in self.primitives:
            if primitive not in resonances:
                resonances[primitive] = 0.5
        
        return resonances
    
    def _find_relevant_sequences(self, energy_signature: Dict) -> List[Dict]:
        """Find discovered sequences relevant to this energy signature"""
        if not self.discovered_sequences:
            return []
        
        relevant = []
        
        for sequence_data in self.discovered_sequences:
            # Calculate similarity between this signature and sequence's initial signature
            if 'initial_signature' in sequence_data:
                similarity = self._calculate_signature_similarity(
                    energy_signature, sequence_data['initial_signature'])
                
                if similarity > 0.7:  # Reasonable similarity threshold
                    relevant.append({
                        'sequence': sequence_data['actions'],
                        'similarity': similarity,
                        'quality': sequence_data.get('quality', 0.5)
                    })
        
        # Sort by combination of similarity and quality
        if relevant:
            for item in relevant:
                item['relevance'] = item['similarity'] * 0.7 + item['quality'] * 0.3
            
            relevant.sort(key=lambda x: x['relevance'], reverse=True)
        
        return relevant
    
    def _find_best_context(self, energy_signature: Dict, context_energies: List[Dict]) -> Dict:
        """Find the most relevant context energy signature"""
        if not context_energies:
            return None
        
        best_similarity = -1
        best_context = None
        
        for context in context_energies:
            similarity = self._calculate_signature_similarity(energy_signature, context)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_context = context
        
        return best_context
    
    def _record_action_sequence(self, actions: List[str], initial_signature: Dict, 
                              result_signature: Dict, quality: float) -> None:
        """Record a new action sequence if it's novel"""
        # Skip very short sequences
        if len(actions) < 2:
            return
        
        # Check if this sequence is novel
        is_novel = True
        
        for existing in self.discovered_sequences:
            if existing['actions'] == actions:
                # Update quality if this instance was better
                if quality > existing.get('quality', 0):
                    existing['quality'] = quality
                
                is_novel = False
                break
        
        # Record if novel
        if is_novel:
            self.discovered_sequences.append({
                'actions': actions,
                'initial_signature': initial_signature,
                'result_signature': result_signature,
                'quality': quality,
                'usage_count': 1
            })
            
            # Limit total sequences
            if len(self.discovered_sequences) > 100:
                # Remove lowest quality sequence
                self.discovered_sequences.sort(key=lambda x: x.get('quality', 0))
                self.discovered_sequences.pop(0)
    
    def _calculate_signature_similarity(self, sig1: Dict, sig2: Dict) -> float:
        """Calculate similarity between energy signatures"""
        similarity_sum = 0.0
        weight_sum = 0.0
        
        # Compare common properties
        for prop in ['magnitude', 'frequency', 'entropy', 'vector']:
            if prop in sig1 and prop in sig2:
                if 'value' in sig1[prop] and 'value' in sig2[prop]:
                    val1 = sig1[prop]['value']
                    val2 = sig2[prop]['value']
                    
                    # Weight for this property
                    weight = 1.0 if prop != 'vector' else 1.5
                    
                    # Calculate property similarity
                    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                        # Scalar similarity
                        prop_sim = 1.0 - min(1.0, abs(val1 - val2))
                        similarity_sum += prop_sim * weight
                        weight_sum += weight
                    elif isinstance(val1, list) and isinstance(val2, list):
                        # Vector similarity - use shorter length
                        min_len = min(len(val1), len(val2))
                        if min_len > 0:
                            # Compare available components
                            vector_sim = 1.0 - sum(abs(val1[i] - val2[i]) for i in range(min_len)) / min_len
                            similarity_sum += vector_sim * weight
                            weight_sum += weight
        
        # Return weighted average
        return similarity_sum / weight_sum if weight_sum > 0 else 0.0
    
    # Primitive implementation methods
    
    def _shift_energy_up(self, signature: Dict) -> Dict:
        """Shift energy signature upward (abstraction)"""
        result = dict(signature)
        
        # Increase y-component of vector if present
        if 'vector' in result and 'value' in result['vector']:
            vector = result['vector']['value']
            if isinstance(vector, list) and len(vector) > 1:
                vector[1] = min(1.0, vector[1] + 0.2)
        
        # Decrease entropy (more ordered)
        if 'entropy' in result and 'value' in result['entropy']:
            entropy = result['entropy']['value']
            if isinstance(entropy, (int, float)):
                result['entropy']['value'] = max(0.1, entropy - 0.15)
        
        # Increase frequency (faster oscillation)
        if 'frequency' in result and 'value' in result['frequency']:
            freq = result['frequency']['value']
            if isinstance(freq, (int, float)):
                result['frequency']['value'] = min(1.0, freq + 0.15)
        
        return result
    
    def _shift_energy_down(self, signature: Dict) -> Dict:
        """Shift energy signature downward (concretization)"""
        result = dict(signature)
        
        # Decrease y-component of vector if present
        if 'vector' in result and 'value' in result['vector']:
            vector = result['vector']['value']
            if isinstance(vector, list) and len(vector) > 1:
                vector[1] = max(0.0, vector[1] - 0.2)
        
        # Increase entropy (more chaotic)
        if 'entropy' in result and 'value' in result['entropy']:
            entropy = result['entropy']['value']
            if isinstance(entropy, (int, float)):
                result['entropy']['value'] = min(1.0, entropy + 0.15)
        
        # Increase magnitude (more defined)
        if 'magnitude' in result and 'value' in result['magnitude']:
            mag = result['magnitude']['value']
            if isinstance(mag, (int, float)):
                result['magnitude']['value'] = min(1.0, mag + 0.15)
        
        return result
    
    def _shift_energy_left(self, signature: Dict) -> Dict:
        """Shift energy signature leftward"""
        result = dict(signature)
        
        # Decrease x-component of vector if present
        if 'vector' in result and 'value' in result['vector']:
            vector = result['vector']['value']
            if isinstance(vector, list) and len(vector) > 0:
                vector[0] = max(0.0, vector[0] - 0.2)
        
        return result
    
    def _shift_energy_right(self, signature: Dict) -> Dict:
        """Shift energy signature rightward"""
        result = dict(signature)
        
        # Increase x-component of vector if present
        if 'vector' in result and 'value' in result['vector']:
            vector = result['vector']['value']
            if isinstance(vector, list) and len(vector) > 0:
                vector[0] = min(1.0, vector[0] + 0.2)
        
        return result
    
    def _invert_energy(self, signature: Dict) -> Dict:
        """Invert energy signature (negation)"""
        result = dict(signature)
        
        # Invert vector components around 0.5
        if 'vector' in result and 'value' in result['vector']:
            vector = result['vector']['value']
            if isinstance(vector, list):
                result['vector']['value'] = [1.0 - v for v in vector]
        
        # Invert magnitude and frequency (1-x)
        for prop in ['magnitude', 'frequency']:
            if prop in result and 'value' in result[prop]:
                value = result[prop]['value']
                if isinstance(value, (int, float)):
                    result[prop]['value'] = 1.0 - value
        
        return result
    
    def _merge_energies(self, signature1: Dict, signature2: Dict) -> Dict:
        """Merge two energy signatures (synthesis)"""
        result = {}
        
        # Process each energy property
        for prop in set(signature1.keys()) | set(signature2.keys()):
            if prop in signature1 and prop in signature2:
                # Both signatures have this property
                if 'value' in signature1[prop] and 'value' in signature2[prop]:
                    val1 = signature1[prop]['value']
                    val2 = signature2[prop]['value']
                    
                    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                        # Average scalar values with slight emergent boost
                        avg = (val1 + val2) / 2
                        emergent = 0.1 * abs(val1 - val2)  # Emergence from difference
                        merged = avg + emergent if avg < 0.5 else avg - emergent
                        result[prop] = {'value': max(0.0, min(1.0, merged))}
                    elif isinstance(val1, list) and isinstance(val2, list):
                        # For vectors, ensure same length
                        if len(val1) == len(val2):
                            # Merge vectors with slight emergent component
                            merged_vector = []
                            for i in range(len(val1)):
                                avg = (val1[i] + val2[i]) / 2
                                emergent = 0.1 * abs(val1[i] - val2[i])
                                merged = avg + emergent if avg < 0.5 else avg - emergent
                                merged_vector.append(max(0.0, min(1.0, merged)))
                            result[prop] = {'value': merged_vector}
                        else:
                            # Different lengths, use longer one
                            longer = val1 if len(val1) > len(val2) else val2
                            result[prop] = {'value': longer}
            elif prop in signature1:
                # Only in signature1
                result[prop] = signature1[prop].copy()
            elif prop in signature2:
                # Only in signature2
                result[prop] = signature2[prop].copy()
        
        return result
    
    def _oscillate_energy(self, signature: Dict) -> Dict:
        """Create oscillation in energy signature"""
        result = dict(signature)
        
        # Increase frequency (faster oscillation)
        if 'frequency' in result and 'value' in result['frequency']:
            freq = result['frequency']['value']
            if isinstance(freq, (int, float)):
                result['frequency']['value'] = min(1.0, freq + 0.25)
        
        # Add oscillation to vector components
        if 'vector' in result and 'value' in result['vector']:
            vector = result['vector']['value']
            if isinstance(vector, list):
                # Introduce alternating pattern
                for i in range(len(vector)):
                    if i % 2 == 0:
                        vector[i] = min(1.0, vector[i] + 0.15)
                    else:
                        vector[i] = max(0.0, vector[i] - 0.15)
        
        return result
    
    def _expand_boundaries(self, signature: Dict) -> Dict:
        """Expand energy boundaries"""
        result = dict(signature)
        
        # Expand boundary if present
        if 'boundary' in result and 'value' in result['boundary']:
            boundary = result['boundary']['value']
            if isinstance(boundary, list) and len(boundary) >= 2:
                center = (boundary[0] + boundary[1]) / 2
                width = boundary[1] - boundary[0]
                new_width = width * 1.3  # Expand by 30%
                result['boundary']['value'] = [
                    max(0.0, center - new_width/2),
                    min(1.0, center + new_width/2)
                ]
        elif 'boundary' not in result:
            # Create boundary if not present
            result['boundary'] = {'value': [0.2, 0.8]}
        
        # Increase entropy (more possibilities)
        if 'entropy' in result and 'value' in result['entropy']:
            entropy = result['entropy']['value']
            if isinstance(entropy, (int, float)):
                result['entropy']['value'] = min(1.0, entropy + 0.15)
        
        return result
    
    def _contract_boundaries(self, signature: Dict) -> Dict:
        """Contract energy boundaries"""
        result = dict(signature)
        
        # Contract boundary if present
        if 'boundary' in result and 'value' in result['boundary']:
            boundary = result['boundary']['value']
            if isinstance(boundary, list) and len(boundary) >= 2:
                center = (boundary[0] + boundary[1]) / 2
                width = boundary[1] - boundary[0]
                new_width = width * 0.7  # Contract by 30%
                result['boundary']['value'] = [
                    max(0.0, center - new_width/2),
                    min(1.0, center + new_width/2)
                ]
        elif 'boundary' not in result:
            # Create narrow boundary if not present
            result['boundary'] = {'value': [0.4, 0.6]}
        
        # Decrease entropy (more focused)
        if 'entropy' in result and 'value' in result['entropy']:
            entropy = result['entropy']['value']
            if isinstance(entropy, (int, float)):
                result['entropy']['value'] = max(0.1, entropy - 0.15)
        
        return result
    
    def _bifurcate_energy(self, signature: Dict) -> Dict:
        """Split energy into dual paths (dialectical)"""
        result = dict(signature)
        
        # Create a more polarized vector
        if 'vector' in result and 'value' in result['vector']:
            vector = result['vector']['value']
            if isinstance(vector, list) and vector:
                # Polarize by pushing components away from center
                polarized = []
                for v in vector:
                    if v > 0.5:
                        polarized.append(min(1.0, v + 0.2))
                    else:
                        polarized.append(max(0.0, v - 0.2))
                result['vector']['value'] = polarized
        
        # Increase entropy (representing multiple paths)
        if 'entropy' in result and 'value' in result['entropy']:
            entropy = result['entropy']['value']
            if isinstance(entropy, (int, float)):
                result['entropy']['value'] = min(1.0, entropy + 0.2)
        
        # Add bifurcation marker in meta
        if 'meta' not in result:
            result['meta'] = {}
        result['meta']['bifurcated'] = True
        
        return result
    
    def _amplify_resonance(self, signature1: Dict, signature2: Dict) -> Dict:
        """Amplify resonance between energy signatures"""
        result = dict(signature1)
        
        # Calculate resonant properties
        resonant_props = {}
        
        for prop in set(signature1.keys()) & set(signature2.keys()):
            if 'value' in signature1.get(prop, {}) and 'value' in signature2.get(prop, {}):
                val1 = signature1[prop]['value']
                val2 = signature2[prop]['value']
                
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # For scalar properties, check similarity
                    similarity = 1.0 - abs(val1 - val2)
                    
                    if similarity > 0.7:  # High similarity
                        # Amplify in the direction of the second signature
                        shift = (val2 - val1) * 0.3
                        resonant_props[prop] = val1 + shift
                
                elif isinstance(val1, list) and isinstance(val2, list):
                    # For vector properties, find resonant components
                    if len(val1) == len(val2):
                        resonant_vector = []
                        for i in range(len(val1)):
                            similarity = 1.0 - abs(val1[i] - val2[i])
                            
                            if similarity > 0.7:  # High similarity
                                # Amplify in the direction of the second signature
                                shift = (val2[i] - val1[i]) * 0.3
                                resonant_vector.append(val1[i] + shift)
                            else:
                                resonant_vector.append(val1[i])
                        
                        resonant_props[prop] = resonant_vector
        
        # Apply resonant properties
        for prop, value in resonant_props.items():
            if prop in result:
                result[prop]['value'] = value
        
        # Add resonance marker in meta
        if 'meta' not in result:
            result['meta'] = {}
        result['meta']['resonant'] = True
        
        return result
    
    def _dampen_signature(self, signature: Dict) -> Dict:
        """Reduce intensity of energy signature"""
        result = dict(signature)
        
        # Reduce magnitude
        if 'magnitude' in result and 'value' in result['magnitude']:
            mag = result['magnitude']['value']
            if isinstance(mag, (int, float)):
                result['magnitude']['value'] = max(0.1, mag - 0.2)
        
        # Move vector components toward center (0.5)
        if 'vector' in result and 'value' in result['vector']:
            vector = result['vector']['value']
            if isinstance(vector, list):
                dampened = []
                for v in vector:
                    # Move 30% closer to 0.5
                    dampened.append(v * 0.7 + 0.5 * 0.3)
                result['vector']['value'] = dampened
        
        # Reduce frequency
        if 'frequency' in result and 'value' in result['frequency']:
            freq = result['frequency']['value']
            if isinstance(freq, (int, float)):
                result['frequency']['value'] = max(0.1, freq - 0.2)
        
        return result
    
    def _create_feedback_loop(self, signature: Dict) -> Dict:
        """Create feedback loop (recursion)"""
        result = dict(signature)
        
        # Increase frequency (faster oscillation)
        if 'frequency' in result and 'value' in result['frequency']:
            freq = result['frequency']['value']
            if isinstance(freq, (int, float)):
                result['frequency']['value'] = min(1.0, freq + 0.3)
        
        # Amplify existing patterns in vector
        if 'vector' in result and 'value' in result['vector']:
            vector = result['vector']['value']
            if isinstance(vector, list) and vector:
                # Find pattern direction (increasing or decreasing)
                patterns = []
                for i in range(len(vector) - 1):
                    patterns.append(1 if vector[i+1] > vector[i] else -1)
                
                # Amplify pattern
                if patterns:
                    new_vector = [vector[0]]  # Start with first element
                    for i, pattern in enumerate(patterns):
                        # Amplify the pattern
                        if pattern > 0:
                            # Increasing pattern - make it increase more
                            new_val = new_vector[-1] + (vector[i+1] - vector[i]) * 1.3
                        else:
                            # Decreasing pattern - make it decrease more
                            new_val = new_vector[-1] + (vector[i+1] - vector[i]) * 1.3
                        
                        # Keep within bounds
                        new_vector.append(max(0.0, min(1.0, new_val)))
                    
                    result['vector']['value'] = new_vector
        
        # Add recursion marker in meta
        if 'meta' not in result:
            result['meta'] = {}
        
        if 'recursion_level' not in result['meta']:
            result['meta']['recursion_level'] = 1
        else:
            result['meta']['recursion_level'] += 1
        
        return result