"""
Energy Resonance Comprehension - Understanding through energy resonance
"""
from typing import Dict, List, Any, Tuple

class EnergyResonanceComprehension:
    """Comprehends meaning through energy resonance with primitives"""
    
    def __init__(self, dual_signature_processor, dynamic_primitives, network_reasoning):
        self.processor = dual_signature_processor
        self.primitives = dynamic_primitives
        self.reasoning = network_reasoning
        
        # Primitive resonance thresholds
        self.resonance_thresholds = {
            # For linguistic signatures
            'linguistic': {
                'shift_up': 0.6,      # Abstract linguistic constructs
                'shift_down': 0.6,    # Concrete linguistic constructs
                'connect': 0.5,       # Relational constructs
                'bifurcate': 0.6,     # Dichotomies
                'merge': 0.6          # Unifying concepts
            },
            # For semantic signatures
            'semantic': {
                'shift_up': 0.7,      # Abstract concepts
                'shift_down': 0.5,    # Concrete concepts
                'connect': 0.6,       # Relational concepts
                'bifurcate': 0.7,     # Contrasting concepts
                'merge': 0.6          # Synthetic concepts
            }
        }
    
    def comprehend(self, text: str) -> Dict:
        """
        Comprehend text through energy resonance
        
        Args:
            text: Input text
            
        Returns:
            Comprehension results
        """
        # Process input to get dual signatures
        processed = self.processor.process_input(text)
        
        # Determine resonant primitives for both signature types
        linguistic_primitives = self._find_resonant_primitives(
            processed['composite_linguistic'], 'linguistic'
        )
        
        semantic_primitives = self._find_resonant_primitives(
            processed['composite_semantic'], 'semantic'
        )
        
        # Find philosophical frameworks that resonate
        resonant_frameworks = self._find_resonant_frameworks(
            processed['composite_linguistic'],
            processed['composite_semantic']
        )
        
        # Apply philosophical reasoning based on primitives and frameworks
        philosophical_insights = self._apply_philosophical_reasoning(
            processed, linguistic_primitives, semantic_primitives, resonant_frameworks
        )
        
        # Create comprehension result
        comprehension = {
            'text': text,
            'dual_signatures': {
                'linguistic': processed['composite_linguistic'],
                'semantic': processed['composite_semantic']
            },
            'resonant_primitives': {
                'linguistic': linguistic_primitives,
                'semantic': semantic_primitives
            },
            'resonant_frameworks': resonant_frameworks,
            'philosophical_insights': philosophical_insights,
            'resonant_field': processed['resonant_field']
        }
        
        return comprehension
    
    def _find_resonant_primitives(self, energy_signature: Dict, 
                                signature_type: str) -> List[Tuple[str, float]]:
        """Find primitives that resonate with this energy signature"""
        if signature_type not in self.resonance_thresholds:
            return []
        
        thresholds = self.resonance_thresholds[signature_type]
        
        # Try all primitives
        all_primitives = self.primitives.get_all_primitives()
        resonant_primitives = []
        
        for primitive in all_primitives:
            # Calculate resonance
            resonance = self._calculate_primitive_resonance(
                primitive, energy_signature, signature_type
            )
            
            # Check threshold
            primitive_threshold = thresholds.get(primitive, 0.6)
            if resonance >= primitive_threshold:
                resonant_primitives.append((primitive, resonance))
        
        # Sort by resonance strength
        resonant_primitives.sort(key=lambda x: x[1], reverse=True)
        
        return resonant_primitives
    
    def _calculate_primitive_resonance(self, primitive: str, 
                                     energy_signature: Dict,
                                     signature_type: str) -> float:
        """Calculate resonance between a primitive and energy signature"""
        # Base resonance - would be more sophisticated in full implementation
        base_resonance = 0.5
        
        # Specific resonance calculations for different primitives
        if primitive == 'shift_up':
            # Resonates with high-frequency, low-entropy signatures
            if 'frequency' in energy_signature and 'entropy' in energy_signature:
                freq = energy_signature['frequency'].get('value', 0.5)
                entropy = energy_signature['entropy'].get('value', 0.5)
                return base_resonance + 0.3 * (freq * (1 - entropy))
        
        elif primitive == 'shift_down':
            # Resonates with low-frequency, high-entropy signatures
            if 'frequency' in energy_signature and 'entropy' in energy_signature:
                freq = energy_signature['frequency'].get('value', 0.5)
                entropy = energy_signature['entropy'].get('value', 0.5)
                return base_resonance + 0.3 * ((1 - freq) * entropy)
        
        elif primitive == 'connect':
            # Resonates with balanced signatures
            if 'vector' in energy_signature and 'value' in energy_signature['vector']:
                vector = energy_signature['vector']['value']
                if isinstance(vector, list) and len(vector) >= 3:
                    # Check for balance between components
                    balance = 1.0 - (max(vector) - min(vector))
                    return base_resonance + 0.3 * balance
        
        elif primitive == 'bifurcate':
            # Resonates with signatures having divergent components
            if 'vector' in energy_signature and 'value' in energy_signature['vector']:
                vector = energy_signature['vector']['value']
                if isinstance(vector, list) and len(vector) >= 3:
                    # Check for divergence between components
                    divergence = max(vector) - min(vector)
                    return base_resonance + 0.3 * divergence
        
        elif primitive == 'merge':
            # Resonates with signatures having coherent patterns
            if 'vector' in energy_signature and 'magnitude' in energy_signature:
                vector = energy_signature['vector'].get('value', [0.5])
                magnitude = energy_signature['magnitude'].get('value', 0.5)
                
                if isinstance(vector, list) and len(vector) >= 3:
                    # Calculate coherence (inverse of variance)
                    variance = sum((v - sum(vector)/len(vector))**2 for v in vector) / len(vector)
                    coherence = 1.0 - min(1.0, variance * 5)
                    return base_resonance + 0.2 * coherence + 0.1 * magnitude
        
        # Return base resonance if no specific calculation
        return base_resonance
    
    def _find_resonant_frameworks(self, linguistic_sig: Dict, 
                                semantic_sig: Dict) -> List[Tuple[str, float]]:
        """Find philosophical frameworks that resonate with these signatures"""
        # Get available frameworks
        frameworks = self.primitives.get_available_frameworks()
        resonant_frameworks = []
        
        for framework in frameworks:
            # Calculate framework resonance (combination of linguistic and semantic)
            linguistic_resonance = self._calculate_framework_resonance(
                framework, linguistic_sig, 'linguistic'
            )
            
            semantic_resonance = self._calculate_framework_resonance(
                framework, semantic_sig, 'semantic'
            )
            
            # Combined resonance (weighted average)
            combined_resonance = 0.4 * linguistic_resonance + 0.6 * semantic_resonance
            
            # Check threshold
            if combined_resonance >= 0.6:  # Framework resonance threshold
                resonant_frameworks.append((framework, combined_resonance))
        
        # Sort by resonance strength
        resonant_frameworks.sort(key=lambda x: x[1], reverse=True)
        
        return resonant_frameworks
    
    def _calculate_framework_resonance(self, framework: str, 
                                     energy_signature: Dict,
                                     signature_type: str) -> float:
        """Calculate resonance between a framework and energy signature"""
        # Get framework from primitives system
        if framework not in self.primitives.philosophical_frameworks:
            return 0.0
        
        framework_info = self.primitives.philosophical_frameworks[framework]
        
        # Calculate primitive resonances
        primitive_resonances = []
        for primitive in framework_info.get('primitives', []):
            resonance = self._calculate_primitive_resonance(
                primitive, energy_signature, signature_type
            )
            primitive_resonances.append(resonance)
        
        # Calculate bias resonance
        bias_resonance = 0.5  # Default
        if 'bias' in framework_info:
            bias = framework_info['bias']
            bias_resonance = self._calculate_bias_resonance(bias, energy_signature)
        
        # Calculate overall framework resonance
        if primitive_resonances:
            avg_primitive_resonance = sum(primitive_resonances) / len(primitive_resonances)
            return 0.7 * avg_primitive_resonance + 0.3 * bias_resonance
        else:
            return bias_resonance
    
    def _calculate_bias_resonance(self, bias: Dict, energy_signature: Dict) -> float:
        """Calculate resonance between a bias and energy signature"""
        resonance = 0.5  # Default
        
        # Check vector bias
        if 'vector' in bias and 'vector' in energy_signature:
            bias_vector = bias['vector']
            sig_vector = energy_signature['vector'].get('value', [])
            
            if isinstance(bias_vector, list) and isinstance(sig_vector, list):
                # Calculate vector similarity
                min_len = min(len(bias_vector), len(sig_vector))
                if min_len > 0:
                    similarity = 1.0 - sum(abs(bias_vector[i] - sig_vector[i]) 
                                          for i in range(min_len)) / min_len
                    resonance = similarity
        
        return resonance
    
    def _apply_philosophical_reasoning(self, processed: Dict,
                                     linguistic_primitives: List[Tuple[str, float]],
                                     semantic_primitives: List[Tuple[str, float]],
                                     resonant_frameworks: List[Tuple[str, float]]) -> List[Dict]:
        """Apply philosophical reasoning based on resonances"""
        insights = []
        
        # Get concept IDs from processed input
        concept_ids = []
        for element in processed.get('elements', []):
            for concept_id, _ in element.get('resonant_concepts', []):
                concept_ids.append(concept_id)
        
        # Apply network reasoning if concepts available
        if concept_ids:
            # Find insights through network reasoning
            network_insights = self.reasoning.discover_insights(concept_ids)
            insights.extend(network_insights)
        
        # Apply primitive-based insights
        primitive_insights = self._generate_primitive_insights(
            linguistic_primitives, semantic_primitives
        )
        insights.extend(primitive_insights)
        
        # Apply framework-based insights
        if resonant_frameworks:
            # Use top framework
            top_framework, _ = resonant_frameworks[0]
            
            # Apply framework to semantic signature
            framework_result = self.primitives.apply_framework(
                top_framework, processed['composite_semantic']
            )
            
            # Generate insights
            framework_insights = self._generate_framework_insights(
                top_framework, framework_result
            )
            insights.extend(framework_insights)
        
        return insights
    
    def _generate_primitive_insights(self, linguistic_primitives: List[Tuple[str, float]],
                                   semantic_primitives: List[Tuple[str, float]]) -> List[Dict]:
        """Generate insights based on resonant primitives"""
        insights = []
        
        # Combine primitive lists (take top 3 from each)
        top_linguistic = linguistic_primitives[:3] if linguistic_primitives else []
        top_semantic = semantic_primitives[:3] if semantic_primitives else []
        
        # Generate insights for linguistic primitives
        for primitive, resonance in top_linguistic:
            insight = {
                'type': 'linguistic_primitive',
                'primitive': primitive,
                'resonance': resonance,
                'description': f"The linguistic structure resonates with the {primitive} primitive."
            }
            insights.append(insight)
        
        # Generate insights for semantic primitives
        for primitive, resonance in top_semantic:
            insight = {
                'type': 'semantic_primitive',
                'primitive': primitive,
                'resonance': resonance,
                'description': f"The semantic content resonates with the {primitive} primitive."
            }
            insights.append(insight)
        
        # Generate insights for primitive combinations
        if top_linguistic and top_semantic:
            # Check for interesting combinations
            ling_primitive, _ = top_linguistic[0]
            sem_primitive, _ = top_semantic[0]
            
            if ling_primitive == sem_primitive:
                # Same primitive for both signatures - strong alignment
                insights.append({
                    'type': 'signature_alignment',
                    'primitive': ling_primitive,
                    'description': f"Both linguistic and semantic aspects strongly align with the {ling_primitive} primitive."
                })
            elif (ling_primitive == 'shift_up' and sem_primitive == 'shift_down') or \
                 (ling_primitive == 'shift_down' and sem_primitive == 'shift_up'):
                # Opposite primitives - interesting tension
                insights.append({
                    'type': 'signature_tension',
                    'linguistic_primitive': ling_primitive,
                    'semantic_primitive': sem_primitive,
                    'description': "There's a philosophical tension between the linguistic structure and semantic content."
                })
        
        return insights
    
    def _generate_framework_insights(self, framework: str, 
                                   result: Dict) -> List[Dict]:
        """Generate insights based on framework application"""
        insights = []
        
        # Basic framework insight
        insights.append({
            'type': 'framework_application',
            'framework': framework,
            'description': f"Applied {framework} philosophical framework to interpret the input."
        })
        
        # Framework-specific insights
        if framework == 'dialectical':
            insights.append({
                'type': 'dialectical_analysis',
                'description': "The input contains dialectical tensions that can be synthesized."
            })
        elif framework == 'existentialist':
            insights.append({
                'type': 'existential_analysis',
                'description': "The input relates to questions of meaning and existence."
            })
        elif framework == 'pragmatic':
            insights.append({
                'type': 'pragmatic_analysis',
                'description': "The input can be understood in terms of practical consequences."
            })
        
        return insights