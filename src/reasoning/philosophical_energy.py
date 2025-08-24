"""
Philosophical Energy Dynamics - Implements philosophical reasoning as energy transformations
"""
import numpy as np
from typing import Dict, List, Tuple

class PhilosophicalEnergy:
    """Implements philosophical reasoning patterns as energy transformations"""
    
    def __init__(self):
        # Register philosophical reasoning patterns
        self.reasoning_patterns = {
            'dialectical': self._dialectical_reasoning,
            'deductive': self._deductive_reasoning,
            'inductive': self._inductive_reasoning,
            'abductive': self._abductive_reasoning,
            'analogical': self._analogical_reasoning,
            'conceptual_blending': self._conceptual_blending
        }
        
        # Intellectual virtues as energy modulators
        self.intellectual_virtues = {
            'curiosity': {'frequency_boost': 0.2, 'entropy_tolerance': 0.3},
            'intellectual_humility': {'self_correction': 0.4, 'boundary_expansion': 0.2},
            'intellectual_courage': {'entropy_tolerance': 0.5, 'magnitude_boost': 0.2},
            'intellectual_autonomy': {'vector_independence': 0.3},
            'intellectual_perseverance': {'duration_extension': 0.4}
        }
    
    def apply_reasoning(self, reasoning_type: str, 
                        concepts: List[Dict], 
                        virtues: List[str] = None) -> Dict:
        """
        Apply philosophical reasoning to concepts
        
        Args:
            reasoning_type: Type of philosophical reasoning to apply
            concepts: List of concepts with energy signatures
            virtues: List of intellectual virtues to apply
            
        Returns:
            New concept with transformed energy signature
        """
        # Validate reasoning type
        if reasoning_type not in self.reasoning_patterns:
            raise ValueError(f"Unknown reasoning type: {reasoning_type}")
        
        # Apply reasoning pattern
        reasoning_function = self.reasoning_patterns[reasoning_type]
        result = reasoning_function(concepts)
        
        # Apply intellectual virtues if specified
        if virtues:
            for virtue in virtues:
                if virtue in self.intellectual_virtues:
                    result = self._apply_virtue(result, virtue)
        
        return result
    
    def _dialectical_reasoning(self, concepts: List[Dict]) -> Dict:
        """
        Implement Hegelian dialectical reasoning (thesis-antithesis-synthesis)
        
        This transforms two opposing concepts into a new synthesized concept
        """
        if len(concepts) < 2:
            raise ValueError("Dialectical reasoning requires at least two concepts")
        
        # Extract thesis and antithesis
        thesis = concepts[0]
        antithesis = concepts[1]
        
        # Create synthesis energy signature
        synthesis_energy = {}
        
        # For each energy property, create a dialectical synthesis
        for prop in set(thesis.get('energy_signature', {}).keys()) & set(antithesis.get('energy_signature', {}).keys()):
            if prop in thesis['energy_signature'] and prop in antithesis['energy_signature']:
                thesis_value = thesis['energy_signature'][prop].get('value')
                antithesis_value = antithesis['energy_signature'][prop].get('value')
                
                if thesis_value is not None and antithesis_value is not None:
                    # Handle different property types
                    if isinstance(thesis_value, (int, float)) and isinstance(antithesis_value, (int, float)):
                        # For scalar values, synthesis is not just the average
                        # It's a new emergent value that resolves the contradiction
                        contradiction_degree = abs(thesis_value - antithesis_value)
                        
                        # Higher contradiction leads to more novel synthesis (not just middle ground)
                        if contradiction_degree > 0.5:
                            # High contradiction: synthesis is transformative
                            synthesis_value = (thesis_value + antithesis_value) / 2 + 0.2 * contradiction_degree
                        else:
                            # Low contradiction: synthesis is more averaging
                            synthesis_value = (thesis_value + antithesis_value) / 2
                            
                        # Ensure within bounds
                        synthesis_value = min(1.0, max(0.0, synthesis_value))
                        
                        synthesis_energy[prop] = {'value': synthesis_value}
                        
                    elif isinstance(thesis_value, list) and isinstance(antithesis_value, list):
                        # For vector values, synthesis creates a new vector direction
                        if len(thesis_value) == len(antithesis_value):
                            # Normalize vectors
                            thesis_mag = np.sqrt(sum(x**2 for x in thesis_value))
                            antithesis_mag = np.sqrt(sum(x**2 for x in antithesis_value))
                            
                            if thesis_mag > 0 and antithesis_mag > 0:
                                thesis_norm = [x/thesis_mag for x in thesis_value]
                                antithesis_norm = [x/antithesis_mag for x in antithesis_value]
                                
                                # Calculate dot product to measure opposition
                                dot_product = sum(x*y for x,y in zip(thesis_norm, antithesis_norm))
                                
                                # Higher opposition (negative dot product) creates more emergence
                                emergence_factor = max(0, (1 - dot_product) / 2)
                                
                                # Create synthesis vector
                                synthesis_vector = []
                                for i in range(len(thesis_value)):
                                    # Base synthesis
                                    base = (thesis_value[i] + antithesis_value[i]) / 2
                                    
                                    # Add emergent component perpendicular to both
                                    if len(thesis_value) == 3:  # 3D vector
                                        # Cross product for 3D vectors creates perpendicular component
                                        cross = [
                                            thesis_norm[1]*antithesis_norm[2] - thesis_norm[2]*antithesis_norm[1],
                                            thesis_norm[2]*antithesis_norm[0] - thesis_norm[0]*antithesis_norm[2],
                                            thesis_norm[0]*antithesis_norm[1] - thesis_norm[1]*antithesis_norm[0]
                                        ]
                                        
                                        cross_mag = np.sqrt(sum(x**2 for x in cross))
                                        if cross_mag > 0:
                                            cross_norm = [x/cross_mag for x in cross]
                                            # Add emergent component
                                            base += 0.3 * emergence_factor * cross_norm[i]
                                    
                                    synthesis_vector.append(base)
                                
                                synthesis_energy[prop] = {'value': synthesis_vector}
        
        # Create synthesis concept
        synthesis = {
            'name': f"Synthesis of {thesis.get('name', 'thesis')} and {antithesis.get('name', 'antithesis')}",
            'energy_signature': synthesis_energy,
            'derivation': {
                'type': 'dialectical',
                'components': [
                    {'name': thesis.get('name', 'thesis'), 'role': 'thesis'},
                    {'name': antithesis.get('name', 'antithesis'), 'role': 'antithesis'}
                ]
            }
        }
        
        return synthesis
    
    def _deductive_reasoning(self, concepts: List[Dict]) -> Dict:
        """
        Implement deductive reasoning (from general to specific)
        
        This transforms a general concept and a rule into a specific conclusion
        """
        if len(concepts) < 2:
            raise ValueError("Deductive reasoning requires at least two concepts")
        
        # Extract premise and rule
        general_premise = concepts[0]
        rule = concepts[1]
        
        # Create conclusion energy signature
        conclusion_energy = {}
        
        # For deductive reasoning, the conclusion inherits the certainty of the premises
        # The magnitude of the conclusion is limited by the weakest premise
        premise_magnitude = general_premise.get('energy_signature', {}).get('magnitude', {}).get('value', 0.5)
        rule_magnitude = rule.get('energy_signature', {}).get('magnitude', {}).get('value', 0.5)
        
        # Deduction preserves truth, so conclusion magnitude is the minimum
        conclusion_magnitude = min(premise_magnitude, rule_magnitude)
        
        # The conclusion is more specific (higher frequency, lower entropy)
        conclusion_frequency = min(1.0, general_premise.get('energy_signature', {}).get('frequency', {}).get('value', 0.5) * 1.2)
        conclusion_entropy = max(0.0, general_premise.get('energy_signature', {}).get('entropy', {}).get('value', 0.5) * 0.8)
        
        # The vector shifts toward the rule's direction
        general_vector = general_premise.get('energy_signature', {}).get('vector', {}).get('value', [0.5, 0.5, 0.5])
        rule_vector = rule.get('energy_signature', {}).get('vector', {}).get('value', [0.5, 0.5, 0.5])
        
        if isinstance(general_vector, list) and isinstance(rule_vector, list) and len(general_vector) == len(rule_vector):
            # Conclusion vector is pulled toward the rule
            conclusion_vector = []
            for i in range(len(general_vector)):
                # Weighted average favoring rule direction
                conclusion_vector.append(0.3 * general_vector[i] + 0.7 * rule_vector[i])
        else:
            conclusion_vector = [0.5, 0.5, 0.5]
        
        # Assemble conclusion energy
        conclusion_energy = {
            'magnitude': {'value': conclusion_magnitude},
            'frequency': {'value': conclusion_frequency},
            'entropy': {'value': conclusion_entropy},
            'vector': {'value': conclusion_vector}
        }
        
        # Create conclusion concept
        conclusion = {
            'name': f"Deduction from {general_premise.get('name', 'premise')}",
            'energy_signature': conclusion_energy,
            'derivation': {
                'type': 'deductive',
                'components': [
                    {'name': general_premise.get('name', 'premise'), 'role': 'general_premise'},
                    {'name': rule.get('name', 'rule'), 'role': 'rule'}
                ]
            }
        }
        
        return conclusion
    
    def _inductive_reasoning(self, concepts: List[Dict]) -> Dict:
        """
        Implement inductive reasoning (from specific to general)
        
        This transforms specific observations into a general pattern
        """
        if len(concepts) < 2:
            raise ValueError("Inductive reasoning requires at least two concepts")
        
        # Create pattern energy signature
        pattern_energy = {}
        
        # Average the energy properties across all observations
        for prop in ['magnitude', 'frequency', 'entropy', 'vector']:
            values = []
            for concept in concepts:
                if prop in concept.get('energy_signature', {}):
                    value = concept['energy_signature'][prop].get('value')
                    if value is not None:
                        values.append(value)
            
            if values:
                if isinstance(values[0], (int, float)):
                    # For scalar values, take the average
                    avg_value = sum(values) / len(values)
                    
                    # Induction increases entropy (uncertainty) and decreases magnitude (confidence)
                    if prop == 'magnitude':
                        # Confidence decreases with more varied observations
                        variation = max(values) - min(values) if len(values) > 1 else 0
                        avg_value = avg_value * (1 - 0.2 * variation)
                    elif prop == 'entropy':
                        # Entropy increases with induction
                        avg_value = min(1.0, avg_value * 1.2)
                    
                    pattern_energy[prop] = {'value': avg_value}
                    
                elif isinstance(values[0], list) and all(isinstance(v, list) for v in values):
                    # For vector values, ensure all have same length
                    if all(len(v) == len(values[0]) for v in values):
                        # Average each component
                        avg_vector = []
                        for i in range(len(values[0])):
                            component_values = [v[i] for v in values]
                            avg_vector.append(sum(component_values) / len(component_values))
                        
                        pattern_energy[prop] = {'value': avg_vector}
        
        # Create pattern concept
        pattern = {
            'name': f"Inductive pattern from {len(concepts)} observations",
            'energy_signature': pattern_energy,
            'derivation': {
                'type': 'inductive',
                'components': [{'name': c.get('name', f'observation_{i}'), 'role': 'observation'} 
                              for i, c in enumerate(concepts)]
            }
        }
        
        return pattern
    
    def _abductive_reasoning(self, concepts: List[Dict]) -> Dict:
        """
        Implement abductive reasoning (inference to best explanation)
        
        This transforms an observation and possible explanations into the most likely explanation
        """
        if len(concepts) < 2:
            raise ValueError("Abductive reasoning requires at least two concepts")
        
        # Extract observation and explanations
        observation = concepts[0]
        explanations = concepts[1:]
        
        # Calculate best explanation based on energy resonance
        best_explanation = None
        best_resonance = -1
        
        for explanation in explanations:
            resonance = self._calculate_explanation_resonance(observation, explanation)
            if resonance > best_resonance:
                best_resonance = resonance
                best_explanation = explanation
        
        if not best_explanation:
            best_explanation = explanations[0]
        
        # Create explanation energy signature (based on best explanation but with abductive characteristics)
        explanation_energy = {}
        
        # Copy energy properties from best explanation
        for prop, details in best_explanation.get('energy_signature', {}).items():
            explanation_energy[prop] = details.copy()
        
        # Modify for abductive characteristics
        if 'magnitude' in explanation_energy:
            # Abduction has lower confidence than deduction
            explanation_energy['magnitude']['value'] = min(0.8, explanation_energy['magnitude']['value'])
        
        if 'entropy' in explanation_energy:
            # Abduction has higher entropy (potential for revision)
            explanation_energy['entropy']['value'] = min(1.0, explanation_energy['entropy']['value'] * 1.3)
        
        # Create explanation concept
        abduction = {
            'name': f"Abductive explanation for {observation.get('name', 'observation')}",
            'energy_signature': explanation_energy,
            'derivation': {
                'type': 'abductive',
                'components': [
                    {'name': observation.get('name', 'observation'), 'role': 'observation'},
                    {'name': best_explanation.get('name', 'explanation'), 'role': 'best_explanation', 
                     'resonance': best_resonance}
                ]
            }
        }
        
        return abduction
    
    def _analogical_reasoning(self, concepts: List[Dict]) -> Dict:
        """
        Implement analogical reasoning (mapping between domains)
        
        This transforms a source domain and target domain into mappings between them
        """
        if len(concepts) < 2:
            raise ValueError("Analogical reasoning requires at least two concepts")
        
        # Extract source and target domains
        source = concepts[0]
        target = concepts[1]
        
        # Create mapping energy signature
        mapping_energy = {}
        
        # For analogical reasoning, the energy signature embodies the relation between domains
        source_energy = source.get('energy_signature', {})
        target_energy = target.get('energy_signature', {})
        
        # Identify structural similarity
        structural_similarity = self._calculate_structural_similarity(source, target)
        
        # Magnitude represents confidence in the analogy
        mapping_energy['magnitude'] = {'value': 0.5 * structural_similarity}
        
        # Frequency represents how specific/general the mapping is
        source_freq = source_energy.get('frequency', {}).get('value', 0.5)
        target_freq = target_energy.get('frequency', {}).get('value', 0.5)
        mapping_energy['frequency'] = {'value': (source_freq + target_freq) / 2}
        
        # Vector represents the direction of mapping (source → target)
        source_vector = source_energy.get('vector', {}).get('value', [0.5, 0.5, 0.5])
        target_vector = target_energy.get('vector', {}).get('value', [0.5, 0.5, 0.5])
        
        if isinstance(source_vector, list) and isinstance(target_vector, list) and len(source_vector) == len(target_vector):
            # Create vector that represents the mapping direction
            mapping_vector = []
            for i in range(len(source_vector)):
                # Vector from source to target
                mapping_vector.append((target_vector[i] - source_vector[i]) * 0.5 + 0.5)
            
            mapping_energy['vector'] = {'value': mapping_vector}
        
        # Create mapping concept
        mapping = {
            'name': f"Analogy from {source.get('name', 'source')} to {target.get('name', 'target')}",
            'energy_signature': mapping_energy,
            'derivation': {
                'type': 'analogical',
                'components': [
                    {'name': source.get('name', 'source'), 'role': 'source_domain'},
                    {'name': target.get('name', 'target'), 'role': 'target_domain'},
                    {'similarity': structural_similarity}
                ]
            }
        }
        
        return mapping
    
    def _conceptual_blending(self, concepts: List[Dict]) -> Dict:
        """
        Implement conceptual blending (Fauconnier & Turner)
        
        This transforms input concepts into a blended space with emergent structure
        """
        if len(concepts) < 2:
            raise ValueError("Conceptual blending requires at least two concepts")
        
        # Extract input spaces
        input_spaces = concepts
        
        # Create blended space energy signature
        blend_energy = {}
        
        # For each energy property, create a blend with emergent structure
        all_props = set()
        for concept in input_spaces:
            all_props.update(concept.get('energy_signature', {}).keys())
        
        for prop in all_props:
            # Collect values from all input spaces
            values = []
            for concept in input_spaces:
                if prop in concept.get('energy_signature', {}):
                    value = concept['energy_signature'][prop].get('value')
                    if value is not None:
                        values.append(value)
            
            if not values:
                continue
                
            if isinstance(values[0], (int, float)):
                # For scalar values, create blend with emergence
                # Unlike simple averaging, blending can create values outside the input range
                
                # Calculate standard average
                avg_value = sum(values) / len(values)
                
                # Calculate variance as measure of conceptual distance
                variance = sum((v - avg_value)**2 for v in values) / len(values)
                
                # Higher variance can lead to more emergent properties
                emergence_factor = min(0.3, variance * 2)
                
                # Emergent value can be outside the input range
                if prop == 'entropy':
                    # Blending often reduces entropy through integration
                    blend_value = max(0.1, avg_value - emergence_factor)
                elif prop == 'magnitude':
                    # Blending can strengthen concepts through integration
                    blend_value = min(1.0, avg_value + emergence_factor)
                else:
                    # Other properties may show emergent patterns
                    # Either amplifying or diminishing based on harmony
                    harmony = self._calculate_conceptual_harmony(input_spaces)
                    if harmony > 0.7:
                        # Harmonious concepts amplify
                        blend_value = min(1.0, avg_value + emergence_factor)
                    else:
                        # Dissonant concepts may cancel out
                        blend_value = avg_value - emergence_factor
                
                blend_energy[prop] = {'value': blend_value}
                
            elif isinstance(values[0], list) and all(isinstance(v, list) for v in values):
                # For vector values, create emergent blend
                if all(len(v) == len(values[0]) for v in values):
                    # Calculate component averages
                    blend_vector = []
                    for i in range(len(values[0])):
                        component_values = [v[i] for v in values]
                        avg = sum(component_values) / len(component_values)
                        
                        # Add emergent vector component
                        min_val = min(component_values)
                        max_val = max(component_values)
                        
                        # Emergence can push beyond input ranges
                        if max_val - min_val > 0.3:
                            # Significant difference creates emergent property
                            # Direction depends on overall blend energy
                            if i == 0:  # x component - amplify differences
                                avg = avg + 0.2 * (max_val - min_val)
                            elif i == 1:  # y component - can go higher for abstraction
                                avg = avg + 0.1 * (max_val - min_val)
                            else:  # other components - moderate emergence
                                avg = avg + 0.05 * (max_val - min_val)
                        
                        blend_vector.append(avg)
                    
                    blend_energy[prop] = {'value': blend_vector}
        
        # Create blended concept
        blend = {
            'name': f"Blend of {', '.join(c.get('name', f'concept_{i}') for i, c in enumerate(input_spaces))}",
            'energy_signature': blend_energy,
            'derivation': {
                'type': 'conceptual_blend',
                'components': [{'name': c.get('name', f'input_{i}'), 'role': 'input_space'} 
                              for i, c in enumerate(input_spaces)]
            }
        }
        
        return blend
    
    def _apply_virtue(self, concept: Dict, virtue: str) -> Dict:
        """Apply an intellectual virtue to modulate a concept"""
        if virtue not in self.intellectual_virtues:
            return concept
        
        # Copy the concept to avoid modifying the original
        modulated = concept.copy()
        modulated['energy_signature'] = concept.get('energy_signature', {}).copy()
        
        # Apply virtue modulations
        virtue_effects = self.intellectual_virtues[virtue]
        
        for effect, value in virtue_effects.items():
            if effect == 'frequency_boost' and 'frequency' in modulated['energy_signature']:
                current = modulated['energy_signature']['frequency'].get('value', 0.5)
                modulated['energy_signature']['frequency']['value'] = min(1.0, current + value)
                
            elif effect == 'entropy_tolerance' and 'entropy' in modulated['energy_signature']:
                # Allows for higher entropy (uncertainty acceptance)
                current = modulated['energy_signature']['entropy'].get('value', 0.5)
                modulated['energy_signature']['entropy']['value'] = min(1.0, current + value)
                
            elif effect == 'self_correction' and 'magnitude' in modulated['energy_signature']:
                # Self-correction moderates excessive confidence
                current = modulated['energy_signature']['magnitude'].get('value', 0.5)
                if current > 0.7:  # Only moderate high confidence
                    modulated['energy_signature']['magnitude']['value'] = current * (1 - value)
                    
            elif effect == 'boundary_expansion' and 'boundary' in modulated['energy_signature']:
                # Expands conceptual boundaries
                current = modulated['energy_signature']['boundary'].get('value', [0.0, 1.0])
                if isinstance(current, list) and len(current) >= 2:
                    width = current[1] - current[0]
                    center = (current[0] + current[1]) / 2
                    new_width = width * (1 + value)
                    modulated['energy_signature']['boundary']['value'] = [
                        max(0.0, center - new_width/2),
                        min(1.0, center + new_width/2)
                    ]
                    
            elif effect == 'magnitude_boost' and 'magnitude' in modulated['energy_signature']:
                current = modulated['energy_signature']['magnitude'].get('value', 0.5)
                modulated['energy_signature']['magnitude']['value'] = min(1.0, current + value)
                
            elif effect == 'vector_independence' and 'vector' in modulated['energy_signature']:
                # Makes vector more distinct (farther from 0.5 center)
                current = modulated['energy_signature']['vector'].get('value', [0.5, 0.5, 0.5])
                if isinstance(current, list):
                    new_vector = []
                    for component in current:
                        # Push components away from center
                        direction = 1 if component > 0.5 else -1
                        new_vector.append(component + direction * value * abs(component - 0.5))
                    modulated['energy_signature']['vector']['value'] = new_vector
                    
            elif effect == 'duration_extension' and 'duration' in modulated['energy_signature']:
                current = modulated['energy_signature']['duration'].get('value', 0.5)
                modulated['energy_signature']['duration']['value'] = min(1.0, current + value)
        
        # Add virtue to derivation
        if 'derivation' not in modulated:
            modulated['derivation'] = {}
        
        if 'applied_virtues' not in modulated['derivation']:
            modulated['derivation']['applied_virtues'] = []
            
        modulated['derivation']['applied_virtues'].append(virtue)
        
        return modulated
    
    def _calculate_explanation_resonance(self, observation: Dict, explanation: Dict) -> float:
        """Calculate how well an explanation resonates with an observation"""
        # Simple resonance calculation
        obs_energy = observation.get('energy_signature', {})
        exp_energy = explanation.get('energy_signature', {})
        
        resonance = 0.0
        count = 0
        
        # Check common properties
        for prop in set(obs_energy.keys()) & set(exp_energy.keys()):
            obs_value = obs_energy[prop].get('value')
            exp_value = exp_energy[prop].get('value')
            
            if obs_value is not None and exp_value is not None:
                # Calculate property resonance
                if isinstance(obs_value, (int, float)) and isinstance(exp_value, (int, float)):
                    # For scalar values
                    prop_resonance = 1.0 - abs(obs_value - exp_value)
                    resonance += prop_resonance
                    count += 1
                elif isinstance(obs_value, list) and isinstance(exp_value, list):
                    # For vector values
                    if len(obs_value) == len(exp_value):
                        # Calculate cosine similarity
                        dot_product = sum(a*b for a, b in zip(obs_value, exp_value))
                        mag_a = sum(a**2 for a in obs_value) ** 0.5
                        mag_b = sum(b**2 for b in exp_value) ** 0.5
                        
                        if mag_a > 0 and mag_b > 0:
                            cosine = dot_product / (mag_a * mag_b)
                            prop_resonance = (cosine + 1) / 2  # Scale from [-1,1] to [0,1]
                            resonance += prop_resonance
                            count += 1
        
        # Return average resonance
        return resonance / count if count > 0 else 0.0
    
    def _calculate_structural_similarity(self, concept1: Dict, concept2: Dict) -> float:
        """Calculate structural similarity between concepts for analogical reasoning"""
        # Extract energy signatures
        energy1 = concept1.get('energy_signature', {})
        energy2 = concept2.get('energy_signature', {})
        
        # Calculate vector correlation (key for structural mapping)
        vec1 = energy1.get('vector', {}).get('value')
        vec2 = energy2.get('vector', {}).get('value')
        
        vector_similarity = 0.0
        
        if isinstance(vec1, list) and isinstance(vec2, list) and len(vec1) == len(vec2):
            # Calculate vector correlation coefficient
            mean1 = sum(vec1) / len(vec1)
            mean2 = sum(vec2) / len(vec2)
            
            numerator = sum((a - mean1) * (b - mean2) for a, b in zip(vec1, vec2))
            denom1 = sum((a - mean1)**2 for a in vec1) ** 0.5
            denom2 = sum((b - mean2)**2 for b in vec2) ** 0.5
            
            if denom1 > 0 and denom2 > 0:
                vector_similarity = numerator / (denom1 * denom2)
                # Convert from [-1,1] to [0,1]
                vector_similarity = (vector_similarity + 1) / 2
        
        # Calculate pattern similarity through energy ratios
        pattern_similarity = 0.0
        pattern_count = 0
        
        for prop in ['magnitude', 'frequency', 'entropy']:
            val1 = energy1.get(prop, {}).get('value')
            val2 = energy2.get(prop, {}).get('value')
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Similarity in pattern, not absolute values
                # Two concepts with the same relative energy distribution can be structurally similar
                pattern_count += 1
                
                # We look at relative position in energy space
                if pattern_count > 1:
                    prev_prop = ['magnitude', 'frequency', 'entropy'][pattern_count-2]
                    prev_val1 = energy1.get(prev_prop, {}).get('value')
                    prev_val2 = energy2.get(prev_prop, {}).get('value')
                    
                    if prev_val1 is not None and prev_val2 is not None:
                        # Compare the ratio/relationship between properties
                        if prev_val1 != 0 and prev_val2 != 0:
                            ratio1 = val1 / prev_val1
                            ratio2 = val2 / prev_val2
                            
                            ratio_similarity = 1.0 - min(1.0, abs(ratio1 - ratio2) / max(ratio1, ratio2))
                            pattern_similarity += ratio_similarity
        
        # Average pattern similarity
        if pattern_count > 1:  # Need at least 2 properties to establish pattern
            pattern_similarity /= (pattern_count - 1)
        else:
            pattern_similarity = 0.5  # Neutral if not enough data
        
        # Combine structural similarities (vector is more important for structure)
        return 0.7 * vector_similarity + 0.3 * pattern_similarity
    
    def _calculate_conceptual_harmony(self, concepts: List[Dict]) -> float:
        """Calculate conceptual harmony between multiple concepts"""
        if len(concepts) < 2:
            return 1.0  # Perfect harmony with itself
        
        # Calculate pairwise harmony
        harmony_sum = 0.0
        pair_count = 0
        
        for i in range(len(concepts)):
            for j in range(i+1, len(concepts)):
                energy_i = concepts[i].get('energy_signature', {})
                energy_j = concepts[j].get('energy_signature', {})
                
                # Calculate energy alignment
                alignment = 0.0
                prop_count = 0
                
                for prop in set(energy_i.keys()) & set(energy_j.keys()):
                    val_i = energy_i[prop].get('value')
                    val_j = energy_j[prop].get('value')
                    
                    if val_i is not None and val_j is not None:
                        if isinstance(val_i, (int, float)) and isinstance(val_j, (int, float)):
                            # For scalar values
                            prop_alignment = 1.0 - abs(val_i - val_j)
                            alignment += prop_alignment
                            prop_count += 1
                        elif isinstance(val_i, list) and isinstance(val_j, list):
                            # For vector values
                            if len(val_i) == len(val_j):
                                # Calculate cosine similarity
                                dot_product = sum(a*b for a, b in zip(val_i, val_j))
                                mag_i = sum(a**2 for a in val_i) ** 0.5
                                mag_j = sum(b**2 for b in val_j) ** 0.5
                                
                                if mag_i > 0 and mag_j > 0:
                                    cosine = dot_product / (mag_i * mag_j)
                                    prop_alignment = (cosine + 1) / 2  # Scale to [0,1]
                                    alignment += prop_alignment
                                    prop_count += 1
                
                # Calculate average alignment for this pair
                pair_harmony = alignment / prop_count if prop_count > 0 else 0.5
                harmony_sum += pair_harmony
                pair_count += 1
        
        # Return average harmony
        return harmony_sum / pair_count if pair_count > 0 else 0.0