"""
Philosophical Framework Discovery - Discovers and composes new philosophical frameworks
"""
from typing import Dict, List, Any, Tuple, Set
import numpy as np
import uuid
from collections import defaultdict

class FrameworkDiscovery:
    """Discovers and composes new philosophical frameworks"""
    
    def __init__(self, dynamic_primitives):
        self.primitives = dynamic_primitives
        
        # Framework components
        self.component_library = {
            # Epistemological components
            'rationalism': {
                'primitives': ['shift_up', 'reflect', 'connect'],
                'bias': {'vector': [0.3, 0.8, 0.4]}
            },
            'empiricism': {
                'primitives': ['shift_down', 'connect', 'ground'],
                'bias': {'vector': [0.6, 0.3, 0.5]}
            },
            'skepticism': {
                'primitives': ['invert', 'disconnect', 'reflect'],
                'bias': {'vector': [0.7, 0.6, 0.2]}
            },
            
            # Metaphysical components
            'idealism': {
                'primitives': ['shift_up', 'expand', 'nest'],
                'bias': {'vector': [0.4, 0.9, 0.5]}
            },
            'materialism': {
                'primitives': ['shift_down', 'crystallize', 'ground'],
                'bias': {'vector': [0.7, 0.2, 0.6]}
            },
            'dualism': {
                'primitives': ['bifurcate', 'reflect', 'connect'],
                'bias': {'vector': [0.5, 0.7, 0.5]}
            },
            
            # Ethical components
            'deontology': {
                'primitives': ['crystallize', 'reflect', 'connect'],
                'bias': {'vector': [0.3, 0.6, 0.7]}
            },
            'consequentialism': {
                'primitives': ['shift_right', 'expand', 'connect'],
                'bias': {'vector': [0.8, 0.5, 0.4]}
            },
            'virtue_ethics': {
                'primitives': ['oscillate', 'ground', 'amplify'],
                'bias': {'vector': [0.5, 0.6, 0.8]}
            },
            
            # Add more philosophical components as needed...
        }
        
        # Historical philosophical movements
        self.historical_movements = {
            'ancient_greek': ['rationalism', 'virtue_ethics', 'idealism'],
            'enlightenment': ['rationalism', 'empiricism', 'deontology'],
            'existentialist': ['skepticism', 'dualism', 'consequentialism'],
            'analytic': ['rationalism', 'materialism', 'consequentialism'],
            'continental': ['idealism', 'skepticism', 'virtue_ethics'],
            
            # Add more historical movements as needed...
        }
        
        # Discovered frameworks
        self.discovered_frameworks = []
    
    def discover_new_framework(self, energy_signatures: List[Dict],
                             coherence_threshold: float = 0.7) -> Dict:
        """
        Discover a new philosophical framework from energy signatures
        
        Args:
            energy_signatures: List of energy signatures to analyze
            coherence_threshold: Minimum coherence threshold
            
        Returns:
            Information about the discovered framework
        """
        if not energy_signatures:
            return None
        
        # Analyze energy signatures
        common_features = self._extract_common_features(energy_signatures)
        
        # Find resonant components
        resonant_components = self._find_resonant_components(common_features)
        
        if not resonant_components:
            return None
        
        # Combine top components into framework
        framework_components = resonant_components[:3]  # Use top 3 components
        
        # Collect primitives and patterns from components
        primitives = []
        patterns = []
        biases = []
        
        for component in framework_components:
            comp_info = self.component_library[component]
            primitives.extend(comp_info['primitives'])
            
            if 'bias' in comp_info:
                biases.append(comp_info['bias'])
        
        # Remove duplicates while preserving order
        primitives = list(dict.fromkeys(primitives))
        
        # Find patterns that use these primitives
        available_patterns = self.primitives.philosophical_patterns
        for pattern_name, pattern_primitives in available_patterns.items():
            if any(p in primitives for p in pattern_primitives):
                patterns.append(pattern_name)
        
        # Limit patterns
        patterns = patterns[:5]  # Use up to 5 patterns
        
        # Merge biases
        combined_bias = self._merge_biases(biases) if biases else {}
        
        # Generate framework name
        name = f"discovered_{uuid.uuid4().hex[:8]}"
        
        # Create framework description
        description = f"Framework combining {', '.join(framework_components)}"
        
        # Calculate expected coherence
        coherence = self._calculate_framework_coherence(
            primitives, patterns, combined_bias, energy_signatures
        )
        
        if coherence < coherence_threshold:
            return None
        
        # Create framework
        framework_info = {
            'name': name,
            'components': framework_components,
            'primitives': primitives,
            'patterns': patterns,
            'bias': combined_bias,
            'description': description,
            'coherence': coherence
        }
        
        # Register framework with primitive system
        success = self.primitives.create_philosophical_framework(
            name, patterns, primitives, combined_bias, description
        )
        
        if not success:
            return None
        
        # Record discovery
        self.discovered_frameworks.append(framework_info)
        
        return framework_info
    
    def combine_historical_movements(self, movements: List[str]) -> Dict:
        """
        Combine historical philosophical movements into a new framework
        
        Args:
            movements: List of historical movements to combine
            
        Returns:
            Information about the combined framework
        """
        if not movements:
            return None
        
        # Collect components from movements
        all_components = []
        for movement in movements:
            if movement in self.historical_movements:
                all_components.extend(self.historical_movements[movement])
        
        if not all_components:
            return None
        
        # Count component frequencies
        component_counts = defaultdict(int)
        for component in all_components:
            component_counts[component] += 1
        
        # Select most common components (up to 4)
        selected_components = sorted(
            component_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:4]
        
        framework_components = [comp for comp, count in selected_components]
        
        # Collect primitives and patterns from components
        primitives = []
        patterns = []
        biases = []
        
        for component in framework_components:
            if component in self.component_library:
                comp_info = self.component_library[component]
                primitives.extend(comp_info['primitives'])
                
                if 'bias' in comp_info:
                    biases.append(comp_info['bias'])
        
        # Remove duplicates while preserving order
        primitives = list(dict.fromkeys(primitives))
        
        # Find patterns that use these primitives
        available_patterns = self.primitives.philosophical_patterns
        for pattern_name, pattern_primitives in available_patterns.items():
            if any(p in primitives for p in pattern_primitives):
                patterns.append(pattern_name)
        
        # Limit patterns
        patterns = patterns[:5]  # Use up to 5 patterns
        
        # Merge biases
        combined_bias = self._merge_biases(biases) if biases else {}
        
        # Generate framework name
        name = f"historical_{uuid.uuid4().hex[:8]}"
        
        # Create framework description
        description = f"Framework combining historical movements: {', '.join(movements)}"
        
        # Create framework
        framework_info = {
            'name': name,
            'components': framework_components,
            'historical_movements': movements,
            'primitives': primitives,
            'patterns': patterns,
            'bias': combined_bias,
            'description': description
        }
        
        # Register framework with primitive system
        success = self.primitives.create_philosophical_framework(
            name, patterns, primitives, combined_bias, description
        )
        
        if not success:
            return None
        
        # Record discovery
        self.discovered_frameworks.append(framework_info)
        
        return framework_info
    
    def _extract_common_features(self, energy_signatures: List[Dict]) -> Dict:
        """Extract common features from energy signatures"""
        if not energy_signatures:
            return {}
        
        common_features = {}
        
        # Extract vector statistics
        vector_stats = self._extract_vector_statistics(energy_signatures)
        if vector_stats:
            common_features['vector_stats'] = vector_stats
        
        # Extract property statistics
        property_stats = {}
        for prop in ['magnitude', 'frequency', 'entropy']:
            values = []
            for sig in energy_signatures:
                if prop in sig and 'value' in sig[prop]:
                    value = sig[prop]['value']
                    if isinstance(value, (int, float)):
                        values.append(value)
            
            if values:
                property_stats[prop] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': min(values),
                    'max': max(values)
                }
        
        if property_stats:
            common_features['property_stats'] = property_stats
        
        return common_features
    
    def _extract_vector_statistics(self, energy_signatures: List[Dict]) -> Dict:
        """Extract statistics about vectors in energy signatures"""
        vectors = []
        
        for sig in energy_signatures:
            if 'vector' in sig and 'value' in sig['vector']:
                vector = sig['vector']['value']
                if isinstance(vector, list):
                    vectors.append(vector)
        
        if not vectors:
            return {}
        
        # Find minimum vector length
        min_length = min(len(v) for v in vectors)
        
        # Calculate statistics for each dimension
        dimension_stats = []
        for i in range(min_length):
            values = [v[i] for v in vectors]
            dimension_stats.append({
                'mean': np.mean(values),
                'std': np.std(values),
                'min': min(values),
                'max': max(values)
            })
        
        return {
            'dimensions': dimension_stats,
            'mean_length': np.mean([len(v) for v in vectors])
        }
    
    def _find_resonant_components(self, features: Dict) -> List[str]:
        """Find components that resonate with the extracted features"""
        if not features:
            return []
        
        component_scores = {}
        
        for comp_name, comp_info in self.component_library.items():
            score = self._calculate_component_resonance(comp_info, features)
            component_scores[comp_name] = score
        
        # Sort by score
        sorted_components = sorted(
            component_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return components that have positive resonance
        return [comp for comp, score in sorted_components if score > 0]
    
    def _calculate_component_resonance(self, component: Dict, features: Dict) -> float:
        """Calculate how strongly a component resonates with features"""
        resonance = 0.0
        
        # Check vector bias resonance
        if 'bias' in component and 'vector' in component['bias'] and 'vector_stats' in features:
            vector_bias = component['bias']['vector']
            dimension_stats = features['vector_stats'].get('dimensions', [])
            
            # Compare with available dimensions
            for i in range(min(len(vector_bias), len(dimension_stats))):
                bias_value = vector_bias[i]
                mean_value = dimension_stats[i]['mean']
                
                # Calculate similarity (1 - absolute difference)
                similarity = 1.0 - abs(bias_value - mean_value)
                
                # Weight earlier dimensions more
                weight = 1.0 / (i + 1)
                resonance += similarity * weight
        
        # Check primitive resonance
        if 'primitives' in component and 'property_stats' in features:
            property_stats = features['property_stats']
            
            for primitive in component['primitives']:
                # Each primitive may resonate with different properties
                if primitive == 'shift_up':
                    # Resonates with high frequency, low entropy
                    if 'frequency' in property_stats and 'entropy' in property_stats:
                        freq_mean = property_stats['frequency']['mean']
                        entropy_mean = property_stats['entropy']['mean']
                        
                        if freq_mean > 0.6 and entropy_mean < 0.4:
                            resonance += 0.3
                
                elif primitive == 'shift_down':
                    # Resonates with low frequency, high entropy
                    if 'frequency' in property_stats and 'entropy' in property_stats:
                        freq_mean = property_stats['frequency']['mean']
                        entropy_mean = property_stats['entropy']['mean']
                        
                        if freq_mean < 0.4 and entropy_mean > 0.6:
                            resonance += 0.3
                
                elif primitive == 'expand':
                    # Resonates with high magnitude
                    if 'magnitude' in property_stats:
                        mag_mean = property_stats['magnitude']['mean']
                        
                        if mag_mean > 0.7:
                            resonance += 0.2
                
                # Additional primitive resonances could be added...
        
        return resonance
    
    def _merge_biases(self, biases: List[Dict]) -> Dict:
        """Merge multiple energy biases"""
        if not biases:
            return {}
        
        if len(biases) == 1:
            return biases[0].copy()
        
        result = {}
        
        # Collect all keys
        all_keys = set()
        for bias in biases:
            all_keys.update(bias.keys())
        
        # Merge values for each key
        for key in all_keys:
            # Collect values for this key
            key_values = []
            for bias in biases:
                if key in bias:
                    key_values.append(bias[key])
            
            if not key_values:
                continue
            
            # Handle different value types
            if all(isinstance(v, dict) for v in key_values):
                # Recursively merge dictionaries
                result[key] = self._merge_biases(key_values)
            
            elif all(isinstance(v, list) for v in key_values):
                # For lists (like vectors), find maximum length
                max_len = max(len(v) for v in key_values)
                
                # Initialize result list
                merged_list = [0.0] * max_len
                
                # Average each component
                for i in range(max_len):
                    # Collect values for this index
                    index_values = [v[i] for v in key_values if i < len(v)]
                    
                    if index_values:
                        merged_list[i] = sum(index_values) / len(index_values)
                
                result[key] = merged_list
            
            elif all(isinstance(v, (int, float)) for v in key_values):
                # For scalars, take average
                result[key] = sum(key_values) / len(key_values)
            
            else:
                # Mixed types, use first value
                result[key] = key_values[0]
        
        return result
    
    def _calculate_framework_coherence(self, primitives: List[str],
                                     patterns: List[str],
                                     bias: Dict,
                                     energy_signatures: List[Dict]) -> float:
        """Calculate expected coherence of a framework"""
        if not energy_signatures:
            return 0.0
        
        # Apply framework to each signature
        coherence_scores = []
        
        for signature in energy_signatures:
            # Apply bias
            biased_signature = self._apply_bias(signature, bias)
            
            # Apply primitives
            result = dict(biased_signature)
            for primitive in primitives:
                if self.primitives.primitive_exists(primitive):
                    result = self.primitives.apply_primitive(primitive, result)
            
            # Measure coherence of result
            coherence = self.primitives._measure_energy_coherence(result)
            coherence_scores.append(coherence)
        
        # Return average coherence
        return sum(coherence_scores) / len(coherence_scores)
    
    def _apply_bias(self, signature: Dict, bias: Dict) -> Dict:
        """Apply bias to signature"""
        result = dict(signature)
        
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
                                # Other types, keep original
                                pass
        
        return result