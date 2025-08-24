"""
Dynamic Primitives System - Flexible primitive-based philosophical reasoning
"""
from typing import Dict, List, Any, Callable, Tuple, Set
import numpy as np
import uuid
import threading
from functools import reduce

from src.core.interfaces import EnergySystem, EnergySignature
from src.utils.error_handling import safe_operation
from src.utils.concurrency import synchronized

class DynamicPrimitives:
    """
    Dynamic primitive-based reasoning system that allows for discovery and 
    composition of new philosophical operations
    """
    
    def __init__(self, energy_system: EnergySystem, config=None):
        self.energy = energy_system
        
        # Core primitive operations - the fundamental "alphabet" of operations
        self.core_primitives = {
            # Basic vector transformations
            'shift_up': self._shift_energy_up,
            'shift_down': self._shift_energy_down,
            'shift_left': self._shift_energy_left,
            'shift_right': self._shift_energy_right,
            'rotate': self._rotate_energy,
            'invert': self._invert_energy,
            
            # Pattern operations
            'merge': self._merge_energies,
            'bifurcate': self._bifurcate_energy,
            'oscillate': self._oscillate_energy,
            'amplify': self._amplify_component,
            'dampen': self._dampen_component,
            
            # Boundary operations
            'expand': self._expand_boundaries,
            'contract': self._contract_boundaries,
            'dissolve': self._dissolve_boundaries,
            'crystallize': self._crystallize_boundaries,
            
            # Relationship operations
            'connect': self._connect_energies,
            'disconnect': self._disconnect_energies,
            'resonate': self._amplify_resonance,
            'ground': self._ground_energy,
            
            # Recursive operations
            'loop': self._create_feedback_loop,
            'nest': self._nest_energy,
            'reflect': self._reflect_energy
        }
        
        # Composite primitives - built from core primitives
        self.composite_primitives = {}
        
        # Philosophical patterns - sequences of primitives
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
        
        # Dynamic philosophical frameworks - collections of patterns
        self.philosophical_frameworks = {
            'western_analytic': {
                'patterns': ['deductive', 'analytical', 'critical'],
                'primitives': ['shift_up', 'contract', 'connect', 'reflect'],
                'bias': {'vector': [0.3, 0.7, 0.5]}
            },
            'eastern_holistic': {
                'patterns': ['dialectical', 'synthetic', 'harmonic'],
                'primitives': ['merge', 'oscillate', 'expand', 'ground'],
                'bias': {'vector': [0.7, 0.4, 0.6]}
            },
            'existentialist': {
                'patterns': ['existential', 'phenomenological', 'subjective'],
                'primitives': ['invert', 'shift_up', 'reflect', 'bifurcate'],
                'bias': {'vector': [0.5, 0.8, 0.3]}
            }
            # Additional frameworks can be added dynamically
        }
        
        # Memory of successful compositions
        self.successful_compositions = []
        
        # Primitive efficacy tracking
        self.primitive_efficacy = {primitive: 0.5 for primitive in self.core_primitives}
        self.pattern_efficacy = {pattern: 0.5 for pattern in self.philosophical_patterns}
        self.framework_efficacy = {framework: 0.5 for framework in self.philosophical_frameworks}
        
        # Thread safety
        self.lock = threading.RLock()
    
    @synchronized()
    def create_composite_primitive(self, name: str, component_primitives: List[str], 
                                 description: str = "") -> bool:
        """
        Create a new composite primitive from existing primitives
        
        Args:
            name: Name for the new primitive
            component_primitives: List of primitives to compose
            description: Description of what this primitive does
            
        Returns:
            Success status
        """
        # Validate all component primitives exist
        for primitive in component_primitives:
            if not self.primitive_exists(primitive):
                return False
        
        # Create new composite primitive
        self.composite_primitives[name] = {
            'components': component_primitives,
            'description': description,
            'created': True,
            'efficacy': 0.5
        }
        
        # Initialize efficacy
        self.primitive_efficacy[name] = 0.5
        
        return True
    
    @synchronized()
    def create_philosophical_pattern(self, name: str, primitives: List[str],
                                  description: str = "") -> bool:
        """
        Create a new philosophical pattern from primitives
        
        Args:
            name: Pattern name
            primitives: List of primitives in this pattern
            description: Description of this pattern
            
        Returns:
            Success status
        """
        # Validate all primitives exist
        for primitive in primitives:
            if not self.primitive_exists(primitive):
                return False
        
        # Create pattern
        self.philosophical_patterns[name] = primitives
        
        # Initialize efficacy
        self.pattern_efficacy[name] = 0.5
        
        return True
    
    @synchronized()
    def create_philosophical_framework(self, name: str, patterns: List[str],
                                    primitives: List[str] = None,
                                    bias: Dict = None,
                                    description: str = "") -> bool:
        """
        Create a new philosophical framework
        
        Args:
            name: Framework name
            patterns: List of patterns in this framework
            primitives: Optional list of directly associated primitives
            bias: Optional energy biases for this framework
            description: Description of this framework
            
        Returns:
            Success status
        """
        # Validate patterns exist
        for pattern in patterns:
            if pattern not in self.philosophical_patterns:
                return False
        
        # Validate primitives exist
        if primitives:
            for primitive in primitives:
                if not self.primitive_exists(primitive):
                    return False
        
        # Create framework
        self.philosophical_frameworks[name] = {
            'patterns': patterns,
            'primitives': primitives or [],
            'bias': bias or {},
            'description': description
        }
        
        # Initialize efficacy
        self.framework_efficacy[name] = 0.5
        
        return True
    
    def primitive_exists(self, primitive_name: str) -> bool:
        """Check if a primitive exists (core or composite)"""
        return (primitive_name in self.core_primitives or 
                primitive_name in self.composite_primitives)
    
    def get_all_primitives(self) -> List[str]:
        """Get list of all available primitives"""
        return list(self.core_primitives.keys()) + list(self.composite_primitives.keys())
    
    def get_available_frameworks(self) -> List[str]:
        """Get list of all available philosophical frameworks"""
        return list(self.philosophical_frameworks.keys())
    
    def suggest_primitives(self, energy_signature: Dict, context: Dict = None,
                         framework: str = None) -> List[str]:
        """
        Suggest primitives that would be appropriate for this energy signature
        
        Args:
            energy_signature: Energy signature to analyze
            context: Optional context information
            framework: Optional philosophical framework to use
            
        Returns:
            List of suggested primitives
        """
        suggested = []
        
        # If framework specified, use its primitives as candidates
        if framework and framework in self.philosophical_frameworks:
            framework_info = self.philosophical_frameworks[framework]
            candidates = framework_info['primitives'].copy()
            
            # Add primitives from patterns in this framework
            for pattern in framework_info['patterns']:
                if pattern in self.philosophical_patterns:
                    candidates.extend(self.philosophical_patterns[pattern])
            
            # Remove duplicates while preserving order
            candidates = list(dict.fromkeys(candidates))
        else:
            # Use all primitives as candidates
            candidates = self.get_all_primitives()
        
        # Calculate resonance for each primitive
        resonances = {}
        for primitive in candidates:
            resonance = self._calculate_primitive_resonance(primitive, energy_signature, context)
            resonances[primitive] = resonance
        
        # Sort by resonance
        sorted_primitives = sorted(resonances.items(), key=lambda x: x[1], reverse=True)
        
        # Return top 5 primitives
        return [p for p, r in sorted_primitives[:5]]
    
    def apply_primitive(self, primitive_name: str, energy_signature: Dict,
                      secondary_signature: Dict = None,
                      context: Dict = None) -> Dict:
        """
        Apply a primitive to an energy signature
        
        Args:
            primitive_name: Name of primitive to apply
            energy_signature: Energy signature to modify
            secondary_signature: Optional secondary signature for binary operations
            context: Optional context information
            
        Returns:
            Modified energy signature
        """
        # Check if primitive exists
        if primitive_name in self.core_primitives:
            # Core primitive
            primitive_func = self.core_primitives[primitive_name]
            
            # Apply based on number of arguments needed
            if primitive_name in ['merge', 'connect', 'disconnect', 'resonate']:
                if secondary_signature is None:
                    return energy_signature
                return primitive_func(energy_signature, secondary_signature)
            else:
                return primitive_func(energy_signature, context)
                
        elif primitive_name in self.composite_primitives:
            # Composite primitive
            composite = self.composite_primitives[primitive_name]
            result = dict(energy_signature)
            
            # Apply each component primitive in sequence
            for component in composite['components']:
                result = self.apply_primitive(component, result, secondary_signature, context)
            
            return result
        
        # Primitive not found, return original
        return energy_signature
    
    def apply_pattern(self, pattern_name: str, energy_signature: Dict,
                    context_signatures: List[Dict] = None,
                    context: Dict = None) -> Dict:
        """
        Apply a philosophical pattern to an energy signature
        
        Args:
            pattern_name: Name of pattern to apply
            energy_signature: Energy signature to modify
            context_signatures: Optional list of context energy signatures
            context: Optional context information
            
        Returns:
            Modified energy signature
        """
        if pattern_name not in self.philosophical_patterns:
            return energy_signature
        
        # Get primitives in this pattern
        primitives = self.philosophical_patterns[pattern_name]
        result = dict(energy_signature)
        
        # Apply each primitive in sequence
        for i, primitive in enumerate(primitives):
            # For binary operations, use context if available
            if primitive in ['merge', 'connect', 'disconnect', 'resonate']:
                if context_signatures and i < len(context_signatures):
                    secondary = context_signatures[i]
                else:
                    secondary = None
                
                result = self.apply_primitive(primitive, result, secondary, context)
            else:
                result = self.apply_primitive(primitive, result, None, context)
        
        # Update pattern efficacy
        self._update_efficacy('pattern', pattern_name, 0.01)
        
        return result
    
    def apply_framework(self, framework_name: str, energy_signature: Dict,
                      context_signatures: List[Dict] = None,
                      context: Dict = None) -> Dict:
        """
        Apply a philosophical framework to an energy signature
        
        Args:
            framework_name: Name of framework to apply
            energy_signature: Energy signature to modify
            context_signatures: Optional list of context energy signatures
            context: Optional context information
            
        Returns:
            Modified energy signature
        """
        if framework_name not in self.philosophical_frameworks:
            return energy_signature
        
        # Get framework info
        framework = self.philosophical_frameworks[framework_name]
        result = dict(energy_signature)
        
        # Apply energy bias if specified
        if 'bias' in framework and framework['bias']:
            result = self._apply_energy_bias(result, framework['bias'])
        
        # Apply direct primitives first
        if 'primitives' in framework:
            for primitive in framework['primitives']:
                result = self.apply_primitive(primitive, result, None, context)
        
        # Apply patterns
        if 'patterns' in framework:
            for pattern in framework['patterns']:
                result = self.apply_pattern(pattern, result, context_signatures, context)
        
        # Update framework efficacy
        self._update_efficacy('framework', framework_name, 0.01)
        
        return result
    
    def discover_new_primitives(self, energy_signatures: List[Dict], 
                              success_criterion: Callable = None) -> List[str]:
        """
        Discover new composite primitives through experimentation
        
        Args:
            energy_signatures: Signatures to experiment with
            success_criterion: Function to evaluate success (defaults to energy change)
            
        Returns:
            List of newly discovered primitive names
        """
        if not energy_signatures or len(energy_signatures) < 2:
            return []
        
        # Default success criterion - significant energy change
        if success_criterion is None:
            def success_criterion(original, result):
                return self._calculate_energy_change(original, result) > 0.3
        
        discovered = []
        
        # Get existing primitives to experiment with
        existing_primitives = self.get_all_primitives()
        
        # Try random combinations of primitives
        max_combinations = min(20, len(existing_primitives) * (len(existing_primitives) - 1) // 2)
        
        for _ in range(max_combinations):
            # Select 2-3 random primitives
            num_primitives = np.random.choice([2, 3])
            primitive_combination = np.random.choice(existing_primitives, num_primitives, replace=False)
            
            # Test on multiple signatures
            success_count = 0
            
            for signature in energy_signatures:
                result = dict(signature)
                
                # Apply primitives in sequence
                for primitive in primitive_combination:
                    result = self.apply_primitive(primitive, result)
                
                # Check success
                if success_criterion(signature, result):
                    success_count += 1
            
            # If successful on majority of signatures, create new primitive
            if success_count > len(energy_signatures) // 2:
                # Generate name based on components
                name_components = [p[:3] for p in primitive_combination]
                name = f"comp_{'_'.join(name_components)}_{uuid.uuid4().hex[:4]}"
                
                # Create composite primitive
                self.create_composite_primitive(
                    name, 
                    list(primitive_combination),
                    f"Discovered composite of {', '.join(primitive_combination)}"
                )
                
                discovered.append(name)
        
        return discovered
    
    def discover_new_patterns(self, energy_signatures: List[Dict],
                            framework: str = None) -> List[str]:
        """
        Discover new philosophical patterns
        
        Args:
            energy_signatures: Signatures to experiment with
            framework: Optional framework to guide discovery
            
        Returns:
            List of newly discovered pattern names
        """
        if not energy_signatures:
            return []
        
        discovered = []
        
        # Get primitives to experiment with
        if framework and framework in self.philosophical_frameworks:
            # Use primitives from this framework
            framework_info = self.philosophical_frameworks[framework]
            candidates = set(framework_info['primitives'])
            
            # Add primitives from patterns in this framework
            for pattern in framework_info.get('patterns', []):
                if pattern in self.philosophical_patterns:
                    candidates.update(self.philosophical_patterns[pattern])
        else:
            # Use all primitives
            candidates = set(self.get_all_primitives())
        
        # Convert to list
        primitive_candidates = list(candidates)
        
        # Try sequences of 3-5 primitives
        max_attempts = 15
        
        for _ in range(max_attempts):
            # Generate random sequence length
            seq_length = np.random.randint(3, 6)
            
            # Generate random sequence
            sequence = list(np.random.choice(primitive_candidates, seq_length, replace=True))
            
            # Test sequence on signatures
            coherence_scores = []
            
            for signature in energy_signatures:
                result = dict(signature)
                
                # Apply sequence
                for primitive in sequence:
                    result = self.apply_primitive(primitive, result)
                
                # Measure coherence of result
                coherence = self._measure_energy_coherence(result)
                coherence_scores.append(coherence)
            
            # Calculate average coherence
            avg_coherence = sum(coherence_scores) / len(coherence_scores)
            
            # If coherent enough, create new pattern
            if avg_coherence > 0.7:
                name = f"pattern_{uuid.uuid4().hex[:8]}"
                
                # Create pattern
                self.create_philosophical_pattern(
                    name,
                    sequence,
                    f"Discovered pattern with coherence {avg_coherence:.2f}"
                )
                
                discovered.append(name)
        
        return discovered
    
    def explore_philosophical_space(self, starting_signature: Dict,
                                  framework: str = None,
                                  exploration_steps: int = 5) -> List[Dict]:
        """
        Explore philosophical space from a starting point
        
        Args:
            starting_signature: Energy signature to start from
            framework: Optional framework to guide exploration
            exploration_steps: Number of steps to take
            
        Returns:
            List of discovered energy signatures
        """
        if not starting_signature:
            return []
        
        results = [dict(starting_signature)]
        current = dict(starting_signature)
        
        for _ in range(exploration_steps):
            # Suggest primitives
            suggested = self.suggest_primitives(current, framework=framework)
            
            if not suggested:
                break
            
            # Apply first suggested primitive
            current = self.apply_primitive(suggested[0], current)
            results.append(dict(current))
        
        return results
    
    def _calculate_primitive_resonance(self, primitive: str, energy: Dict,
                                     context: Dict = None) -> float:
        """Calculate resonance between a primitive and energy signature"""
        # Default resonance based on efficacy
        base_resonance = self.primitive_efficacy.get(primitive, 0.5)
        
        # Specific resonance calculations for different primitives
        if primitive == 'shift_up':
            # Resonates with high-frequency, low-entropy signatures
            if 'frequency' in energy and 'entropy' in energy:
                freq = energy['frequency'].get('value', 0.5)
                entropy = energy['entropy'].get('value', 0.5)
                return 0.7 * base_resonance + 0.3 * (freq * (1 - entropy))
        
        elif primitive == 'shift_down':
            # Resonates with low-frequency, high-entropy signatures
            if 'frequency' in energy and 'entropy' in energy:
                freq = energy['frequency'].get('value', 0.5)
                entropy = energy['entropy'].get('value', 0.5)
                return 0.7 * base_resonance + 0.3 * ((1 - freq) * entropy)
        
        elif primitive == 'invert':
            # Resonates with signatures having extreme vector components
            if 'vector' in energy and 'value' in energy['vector']:
                vector = energy['vector']['value']
                if isinstance(vector, list) and vector:
                    extremity = sum(abs(v - 0.5) for v in vector) / len(vector)
                    return 0.7 * base_resonance + 0.3 * extremity
        
        elif primitive == 'merge':
            # Resonates with signatures that have balanced properties
            if all(p in energy for p in ['magnitude', 'frequency', 'entropy']):
                mag = energy['magnitude'].get('value', 0.5)
                freq = energy['frequency'].get('value', 0.5)
                entropy = energy['entropy'].get('value', 0.5)
                
                # Balance measure - how close properties are to each other
                balance = 1.0 - max(abs(mag - freq), abs(mag - entropy), abs(freq - entropy))
                return 0.7 * base_resonance + 0.3 * balance
        
        # For composite primitives, average component resonances
        elif primitive in self.composite_primitives:
            composite = self.composite_primitives[primitive]
            component_resonances = [
                self._calculate_primitive_resonance(comp, energy, context)
                for comp in composite['components']
            ]
            
            if component_resonances:
                avg_resonance = sum(component_resonances) / len(component_resonances)
                return 0.7 * base_resonance + 0.3 * avg_resonance
        
        # Default to efficacy-based resonance
        return base_resonance
    
    def _apply_energy_bias(self, energy: Dict, bias: Dict) -> Dict:
        """Apply energy bias to a signature"""
        result = dict(energy)
        
        # Apply each bias component
        for key, bias_value in bias.items():
            if key in result:
                if isinstance(bias_value, dict) and isinstance(result[key], dict):
                    # Merge dictionaries
                    for subkey, subvalue in bias_value.items():
                        if subkey in result[key]:
                            if isinstance(subvalue, list) and isinstance(result[key][subkey], list):
                                # For vectors, weighted average
                                result[key][subkey] = [
                                    0.7 * result[key][subkey][i] + 0.3 * subvalue[i]
                                    for i in range(min(len(result[key][subkey]), len(subvalue)))
                                ]
                            elif isinstance(subvalue, (int, float)) and isinstance(result[key][subkey], (int, float)):
                                # For scalars, weighted average
                                result[key][subkey] = 0.7 * result[key][subkey] + 0.3 * subvalue
                            else:
                                # Other types, replace
                                result[key][subkey] = subvalue
                        else:
                            # Add new subkey
                            result[key][subkey] = subvalue
                else:
                    # Replace value
                    result[key] = bias_value
            else:
                # Add new key
                result[key] = bias_value
        
        return result
    
    def _calculate_energy_change(self, original: Dict, result: Dict) -> float:
        """Calculate magnitude of change between energy signatures"""
        change_sum = 0.0
        components = 0
        
        # Compare common properties
        for prop in set(original.keys()) & set(result.keys()):
            if prop == 'meta':
                continue
            
            if 'value' in original.get(prop, {}) and 'value' in result.get(prop, {}):
                orig_val = original[prop]['value']
                res_val = result[prop]['value']
                
                if isinstance(orig_val, (int, float)) and isinstance(res_val, (int, float)):
                    # Scalar change
                    change_sum += abs(res_val - orig_val)
                    components += 1
                elif isinstance(orig_val, list) and isinstance(res_val, list):
                    # Vector change - use common components
                    min_len = min(len(orig_val), len(res_val))
                    if min_len > 0:
                        vector_change = sum(abs(res_val[i] - orig_val[i]) for i in range(min_len)) / min_len
                        change_sum += vector_change
                        components += 1
        
        # Return average change
        return change_sum / components if components > 0 else 0.0
    
    def _measure_energy_coherence(self, energy: Dict) -> float:
        """Measure internal coherence of an energy signature"""
        coherence = 0.5  # Default neutral coherence
        
        # Check vector alignment
        if 'vector' in energy and 'value' in energy['vector']:
            vector = energy['vector']['value']
            if isinstance(vector, list) and len(vector) >= 3:
                # Calculate variance of vector components
                variance = np.var(vector)
                
                # Lower variance indicates more coherence
                coherence += 0.2 * (1.0 - min(1.0, variance * 5))
        
        # Check property alignment
        properties = ['magnitude', 'frequency', 'entropy']
        property_values = []
        
        for prop in properties:
            if prop in energy and 'value' in energy[prop]:
                value = energy[prop]['value']
                if isinstance(value, (int, float)):
                    property_values.append(value)
        
        if len(property_values) >= 2:
            # Calculate variance of property values
            variance = np.var(property_values)
            
            # Lower variance indicates more coherence
            coherence += 0.2 * (1.0 - min(1.0, variance * 5))
        
        # Check boundary precision
        if 'boundary' in energy and 'value' in energy['boundary']:
            boundary = energy['boundary']['value']
            if isinstance(boundary, list) and len(boundary) >= 2:
                width = abs(boundary[1] - boundary[0])
                
                # Moderate width is most coherent (too narrow or too wide is less coherent)
                width_coherence = 1.0 - abs(width - 0.5) * 2
                coherence += 0.1 * width_coherence
        
        return min(1.0, max(0.0, coherence))
    
    @synchronized()
    def _update_efficacy(self, element_type: str, element_name: str, 
                       change: float) -> None:
        """Update efficacy for an element"""
        if element_type == 'primitive':
            if element_name in self.primitive_efficacy:
                current = self.primitive_efficacy[element_name]
                if change > 0:
                    # Increase with diminishing returns
                    self.primitive_efficacy[element_name] = min(0.95, current + change * (1.0 - current))
                else:
                    # Decrease
                    self.primitive_efficacy[element_name] = max(0.05, current + change)
                    
        elif element_type == 'pattern':
            if element_name in self.pattern_efficacy:
                current = self.pattern_efficacy[element_name]
                if change > 0:
                    # Increase with diminishing returns
                    self.pattern_efficacy[element_name] = min(0.95, current + change * (1.0 - current))
                else:
                    # Decrease
                    self.pattern_efficacy[element_name] = max(0.05, current + change)
                    
        elif element_type == 'framework':
            if element_name in self.framework_efficacy:
                current = self.framework_efficacy[element_name]
                if change > 0:
                    # Increase with diminishing returns
                    self.framework_efficacy[element_name] = min(0.95, current + change * (1.0 - current))
                else:
                    # Decrease
                    self.framework_efficacy[element_name] = max(0.05, current + change)
    
    # Core primitive implementations
    # (These would be similar to the original implementations but more flexible)
    
    def _shift_energy_up(self, energy: Dict, context: Dict = None) -> Dict:
        """Shift energy signature upward (abstraction)"""
        result = dict(energy)
        
        # Increase y-component of vector if present
        if 'vector' in result and 'value' in result['vector']:
            vector = result['vector']['value'].copy()
            if isinstance(vector, list) and len(vector) > 1:
                vector[1] = min(1.0, vector[1] + 0.2)
                result['vector']['value'] = vector
        
        # Decrease entropy (more ordered)
        if 'entropy' in result and 'value' in result['entropy']:
            entropy = result['entropy']['value']
            if isinstance(entropy, (int, float)):
                result['entropy']['value'] = max(0.1, entropy - 0.15)
        
        # Add dimensional expansion if context indicates high abstraction
        if context and context.get('abstraction_level', 0.5) > 0.7:
            if 'vector' in result and 'value' in result['vector']:
                vector = result['vector']['value']
                if isinstance(vector, list):
                    # Add new dimension for higher abstraction
                    if len(vector) < 7:  # Limit to 7 dimensions
                        result['vector']['value'] = vector + [0.8]  # High value in new dimension
        
        return result
    
    # Additional primitive implementations would follow...
    
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
                        # For vectors, handle different dimensions intelligently
                        len1 = len(val1)
                        len2 = len(val2)
                        
                        if len1 == len2:
                            # Same dimensions - standard merge
                            merged_vector = []
                            for i in range(len1):
                                avg = (val1[i] + val2[i]) / 2
                                emergent = 0.1 * abs(val1[i] - val2[i])
                                merged = avg + emergent if avg < 0.5 else avg - emergent
                                merged_vector.append(max(0.0, min(1.0, merged)))
                            result[prop] = {'value': merged_vector}
                        else:
                            # Different dimensions - preserve higher dimensionality
                            # For common dimensions, merge as above
                            common_len = min(len1, len2)
                            merged_vector = []
                            
                            # Merge common dimensions
                            for i in range(common_len):
                                avg = (val1[i] + val2[i]) / 2
                                emergent = 0.1 * abs(val1[i] - val2[i])
                                merged = avg + emergent if avg < 0.5 else avg - emergent
                                merged_vector.append(max(0.0, min(1.0, merged)))
                            
                            # Add remaining dimensions from longer vector
                            if len1 > len2:
                                merged_vector.extend(val1[common_len:])
                            else:
                                merged_vector.extend(val2[common_len:])
                            
                            result[prop] = {'value': merged_vector}
                else:
                    # Handle case where 'value' is missing
                    result[prop] = signature1[prop].copy()
            elif prop in signature1:
                # Only in signature1
                result[prop] = signature1[prop].copy()
            elif prop in signature2:
                # Only in signature2
                result[prop] = signature2[prop].copy()
        
        # Add emergence meta information
        if 'meta' not in result:
            result['meta'] = {}
        result['meta']['emerged_from_merge'] = True
        
        return result
    
    # Additional primitive implementations would follow...