"""
Energy-Conceptual Bridge - Connects energy-based reasoning to human conceptual understanding
"""
from typing import Dict, List, Any, Tuple
import numpy as np

class EnergyConceptualBridge:
    """
    Bridges between energy-based reasoning and human conceptual understanding,
    preserving energy insights while making them relatable to human experience
    """
    
    def __init__(self, resonance_network, dynamic_primitives, config=None):
        self.network = resonance_network
        self.primitives = dynamic_primitives
        
        # Conceptual mappings for energy patterns
        self.energy_concept_mappings = {
            # Vector component mappings (position in conceptual space)
            'vector_dimensions': {
                0: {  # X-axis
                    'name': 'temporal_orientation',
                    'high': 'future-oriented, progressive, emergent',
                    'mid': 'present-focused, immediate, contemporary',
                    'low': 'historically-grounded, foundational, original'
                },
                1: {  # Y-axis
                    'name': 'abstraction_level',
                    'high': 'abstract, theoretical, general',
                    'mid': 'balanced abstraction, moderately specific',
                    'low': 'concrete, specific, particular'
                },
                2: {  # Z-axis
                    'name': 'subjective_objective',
                    'high': 'objective, external, independent',
                    'mid': 'intersubjective, relational, contextual',
                    'low': 'subjective, internal, personal'
                }
            },
            
            # Frequency mappings (rate of change/variation)
            'frequency': {
                'name': 'dynamism',
                'high': 'rapidly changing, dynamic, volatile',
                'mid': 'moderately changing, adaptable, flexible',
                'low': 'stable, constant, enduring'
            },
            
            # Entropy mappings (order/disorder)
            'entropy': {
                'name': 'complexity',
                'high': 'complex, ambiguous, multifaceted',
                'mid': 'moderately complex, structured with nuance',
                'low': 'simple, clear, well-defined'
            },
            
            # Magnitude mappings (significance/presence)
            'magnitude': {
                'name': 'significance',
                'high': 'powerful, influential, dominant',
                'mid': 'moderately significant, relevant',
                'low': 'subtle, nuanced, background'
            }
        }
        
        # Primitive operation conceptual mappings
        self.primitive_concept_mappings = {
            'shift_up': {
                'concept': 'abstraction',
                'process': 'moving toward more general principles',
                'examples': [
                    'recognizing that "justice" is a form of "ethical principle"',
                    'seeing that both chess and debate are forms of "strategic interaction"',
                    'understanding how specific feelings contribute to broader emotional states'
                ]
            },
            'shift_down': {
                'concept': 'concretization',
                'process': 'exploring specific instances or examples',
                'examples': [
                    'examining how "equality" manifests in voting rights',
                    'identifying specific examples of "beauty" in nature',
                    'finding concrete applications of an abstract theory'
                ]
            },
            'shift_left': {
                'concept': 'historicization',
                'process': 'examining origins and foundations',
                'examples': [
                    'tracing a concept back to its historical roots',
                    'finding the foundational principles behind an idea',
                    'understanding the developmental origins of a phenomenon'
                ]
            },
            'shift_right': {
                'concept': 'projection',
                'process': 'exploring future implications',
                'examples': [
                    'considering the future consequences of an ethical stance',
                    'projecting how a concept might evolve',
                    'examining potential applications of a principle'
                ]
            },
            'invert': {
                'concept': 'negation',
                'process': 'considering the opposite perspective',
                'examples': [
                    'examining freedom through the lens of constraint',
                    'understanding order by analyzing chaos',
                    'seeing how absence can define presence'
                ]
            },
            'merge': {
                'concept': 'synthesis',
                'process': 'combining different elements into a unified whole',
                'examples': [
                    'integrating opposing viewpoints into a more comprehensive perspective',
                    'finding the common ground between competing theories',
                    'recognizing how seemingly different concepts share underlying unity'
                ]
            },
            'bifurcate': {
                'concept': 'differentiation',
                'process': 'recognizing essential distinctions',
                'examples': [
                    'distinguishing between knowledge and wisdom',
                    'separating correlation from causation',
                    'identifying the boundary between related concepts'
                ]
            },
            'oscillate': {
                'concept': 'dialectic',
                'process': 'moving between opposing positions',
                'examples': [
                    'examining the tension between individual freedom and social responsibility',
                    'recognizing the cyclical nature of philosophical trends',
                    'seeing how concepts often exist in dynamic opposition'
                ]
            },
            'expand': {
                'concept': 'extension',
                'process': 'broadening conceptual boundaries',
                'examples': [
                    'extending the concept of "mind" beyond the individual brain',
                    'broadening the application of ethical principles to new domains',
                    'recognizing connections between seemingly unrelated fields'
                ]
            },
            'contract': {
                'concept': 'refinement',
                'process': 'focusing on essential elements',
                'examples': [
                    'identifying the core principles of a complex theory',
                    'distilling a broad concept to its essential meaning',
                    'focusing on the most relevant aspects of a multifaceted issue'
                ]
            }
        }
        
        # Experiential metaphors that connect energy patterns to human experience
        self.experiential_metaphors = {
            'resonance': {
                'description': 'when concepts naturally connect or harmonize',
                'energy_pattern': 'similar frequency, aligned vectors',
                'experiential_examples': [
                    'the feeling when you hear an idea that immediately "clicks"',
                    'recognizing a pattern that feels deeply familiar',
                    'the natural connection between related concepts'
                ]
            },
            'friction': {
                'description': 'when concepts resist simple integration',
                'energy_pattern': 'conflicting frequencies, misaligned vectors',
                'experiential_examples': [
                    'the mental effort required to reconcile contradictory ideas',
                    'cognitive dissonance when beliefs conflict with evidence',
                    'the resistance felt when trying to connect disparate concepts'
                ]
            },
            'emergence': {
                'description': 'when new understanding arises from conceptual interaction',
                'energy_pattern': 'constructive interference patterns, energy amplification',
                'experiential_examples': [
                    'the "aha" moment when separate ideas suddenly connect',
                    'seeing a completely new pattern emerge from familiar elements',
                    'the sudden clarity that comes from rearranging concepts'
                ]
            },
            'grounding': {
                'description': 'connecting abstract concepts to concrete reality',
                'energy_pattern': 'reducing entropy, shifting vector orientation downward',
                'experiential_examples': [
                    'understanding an abstract principle through a tangible example',
                    'the satisfying feeling of connecting theory to practice',
                    'finding real-world applications for philosophical ideas'
                ]
            },
            'transcendence': {
                'description': 'moving beyond current conceptual limitations',
                'energy_pattern': 'increasing vector dimensionality, expanding boundaries',
                'experiential_examples': [
                    'the expansive feeling when breaking through limited thinking',
                    'seeing beyond apparent contradictions to deeper harmony',
                    'recognizing a higher-order pattern that resolves complexity'
                ]
            }
        }
        
        # Domain-specific interpretations for different philosophical areas
        self.domain_interpretations = {
            'ethics': {
                'vector_high_y': 'principle-based ethical reasoning',
                'vector_low_y': 'case-based ethical judgment',
                'frequency_high': 'ethical positions that adapt to circumstances',
                'frequency_low': 'stable ethical foundations',
                'entropy_high': 'complex moral ambiguity',
                'entropy_low': 'clear moral distinctions'
            },
            'epistemology': {
                'vector_high_y': 'a priori knowledge, rationalism',
                'vector_low_y': 'empirical knowledge, direct observation',
                'frequency_high': 'evolving understanding, provisional knowledge',
                'frequency_low': 'established truths, foundational knowledge',
                'entropy_high': 'uncertainty, ambiguity, partial knowledge',
                'entropy_low': 'certainty, clarity, definitive knowledge'
            },
            'metaphysics': {
                'vector_high_y': 'abstract principles of reality',
                'vector_low_y': 'concrete manifestations of being',
                'frequency_high': 'process philosophy, becoming',
                'frequency_low': 'substance philosophy, being',
                'entropy_high': 'indeterminism, chance, multiplicity',
                'entropy_low': 'determinism, necessity, unity'
            }
        }
    
    def translate_energy_signature(self, energy: Dict, domain: str = None) -> Dict:
        """
        Translate energy signature into human-understandable concepts
        while preserving energy information
        
        Args:
            energy: Energy signature
            domain: Optional philosophical domain for context
            
        Returns:
            Translation with both energy and conceptual elements
        """
        # Extract key energy properties
        properties = {}
        
        # Process vector components
        if 'vector' in energy and 'value' in energy['vector']:
            vector = energy['vector']['value']
            if isinstance(vector, list):
                properties['vector'] = self._translate_vector(vector, domain)
        
        # Process other properties
        for prop in ['frequency', 'entropy', 'magnitude']:
            if prop in energy and 'value' in energy[prop]:
                value = energy[prop]['value']
                if isinstance(value, (int, float)):
                    properties[prop] = self._translate_scalar_property(prop, value, domain)
        
        # Generate overall description
        conceptual_description = self._generate_conceptual_description(properties)
        
        # Create translation result
        translation = {
            'energy_signature': energy,  # Preserve original energy signature
            'conceptual_properties': properties,
            'description': conceptual_description,
            'domain_interpretation': self._generate_domain_interpretation(properties, domain) if domain else None
        }
        
        return translation
    
    def translate_energy_operation(self, operation: str,
                                 before_energy: Dict,
                                 after_energy: Dict) -> Dict:
        """
        Translate energy operation into conceptual understanding
        while preserving energy aspects
        
        Args:
            operation: Energy operation performed
            before_energy: Energy signature before operation
            after_energy: Energy signature after operation
            
        Returns:
            Translation with both energy and conceptual elements
        """
        # Get conceptual mapping for this operation
        operation_concept = self.primitive_concept_mappings.get(operation, {
            'concept': operation,
            'process': f"applying the {operation} operation",
            'examples': []
        })
        
        # Translate energy signatures
        before_translation = self.translate_energy_signature(before_energy)
        after_translation = self.translate_energy_signature(after_energy)
        
        # Identify key changes
        energy_changes = self._identify_energy_changes(before_energy, after_energy)
        conceptual_changes = self._translate_energy_changes(energy_changes)
        
        # Generate explanation of the operation
        explanation = self._generate_operation_explanation(
            operation, operation_concept, conceptual_changes
        )
        
        # Select relevant metaphor
        metaphor = self._select_relevant_metaphor(operation, energy_changes)
        
        # Select example based on operation and energy patterns
        example = self._select_example(operation_concept, energy_changes)
        
        # Create translation result
        translation = {
            'operation': operation,
            'energy': {
                'before': before_energy,
                'after': after_energy,
                'changes': energy_changes
            },
            'conceptual': {
                'operation_concept': operation_concept['concept'],
                'process': operation_concept['process'],
                'before': before_translation['description'],
                'after': after_translation['description'],
                'changes': conceptual_changes
            },
            'explanation': explanation,
            'metaphor': metaphor,
            'example': example
        }
        
        return translation
    
    def translate_philosophical_insight(self, insight: Dict,
                                      include_energy: bool = True) -> Dict:
        """
        Translate philosophical insight into human-understandable explanation
        with appropriate energy aspects
        
        Args:
            insight: Philosophical insight
            include_energy: Whether to include energy details
            
        Returns:
            Translation with appropriate levels of energy information
        """
        # Check if insight already has a description
        if 'description' in insight and insight['description']:
            human_description = insight['description']
        else:
            # Generate description based on insight type
            human_description = self._generate_insight_description(insight)
        
        # Extract energy patterns if present
        energy_patterns = self._extract_energy_patterns(insight)
        
        # Translate energy patterns to conceptual understanding
        conceptual_patterns = {}
        if energy_patterns:
            for pattern_name, pattern in energy_patterns.items():
                if isinstance(pattern, dict):
                    conceptual_patterns[pattern_name] = self.translate_energy_signature(pattern)
        
        # Find relevant metaphors for the insight
        metaphors = self._find_relevant_metaphors(insight, energy_patterns)
        
        # Generate examples
        examples = self._generate_insight_examples(insight, conceptual_patterns)
        
        # Create integrated explanation
        explanation = self._create_integrated_explanation(
            human_description, 
            conceptual_patterns,
            metaphors,
            examples,
            include_energy
        )
        
        # Create translation result
        translation = {
            'original_insight': insight,
            'human_description': human_description,
            'energy_patterns': energy_patterns if include_energy else None,
            'conceptual_patterns': conceptual_patterns,
            'metaphors': metaphors,
            'examples': examples,
            'explanation': explanation
        }
        
        return translation
    
    def explain_reasoning_path(self, path: List[Dict], 
                             include_energy_details: bool = True) -> Dict:
        """
        Explain a reasoning path with appropriate balance of 
        energy insights and human understanding
        
        Args:
            path: Reasoning path
            include_energy_details: Whether to include detailed energy information
            
        Returns:
            Explanation with both energy and conceptual elements
        """
        if not path:
            return {'explanation': "No reasoning path available."}
        
        # Translate each step
        steps = []
        
        for i, step in enumerate(path):
            # Extract step details
            concept_id = step.get('concept_id', '')
            energy_sig = step.get('energy', {})
            reasoning_type = step.get('reasoning_type', '')
            
            # Translate energy signature if available
            energy_translation = None
            if energy_sig:
                energy_translation = self.translate_energy_signature(energy_sig)
            
            # Generate step explanation
            if i == 0:
                # First step
                explanation = f"Beginning with the concept of '{concept_id}'"
                if energy_translation:
                    explanation += f", which represents {energy_translation['description']}"
            else:
                # Subsequent step
                explanation = f"Moving to '{concept_id}' through {reasoning_type}"
                if energy_translation:
                    explanation += f", revealing {energy_translation['description']}"
            
            # Find metaphor for this transition if not first step
            metaphor = None
            if i > 0:
                prev_energy = path[i-1].get('energy', {})
                if prev_energy and energy_sig:
                    changes = self._identify_energy_changes(prev_energy, energy_sig)
                    metaphor = self._select_relevant_metaphor(reasoning_type, changes)
            
            steps.append({
                'concept_id': concept_id,
                'reasoning_type': reasoning_type,
                'energy_translation': energy_translation,
                'explanation': explanation,
                'metaphor': metaphor
            })
        
        # Generate overall path explanation
        path_explanation = self._generate_path_explanation(steps, include_energy_details)
        
        # Create explanation result
        explanation = {
            'path_steps': steps,
            'explanation': path_explanation,
            'energy_details': path if include_energy_details else None
        }
        
        return explanation
    
    def explain_framework_application(self, framework: str,
                                    application_result: Dict,
                                    include_energy: bool = True) -> Dict:
        """
        Explain application of philosophical framework with
        balance of energy insights and human understanding
        
        Args:
            framework: Philosophical framework applied
            application_result: Result of framework application
            include_energy: Whether to include energy details
            
        Returns:
            Explanation with both energy and conceptual elements
        """
        # Translate before/after energy signatures if available
        before_translation = None
        after_translation = None
        
        if 'before_energy' in application_result:
            before_translation = self.translate_energy_signature(
                application_result['before_energy']
            )
        
        if 'after_energy' in application_result:
            after_translation = self.translate_energy_signature(
                application_result['after_energy']
            )
        
        # Translate insights
        insight_translations = []
        
        if 'insights' in application_result:
            for insight in application_result['insights']:
                translation = self.translate_philosophical_insight(
                    insight, include_energy=include_energy
                )
                insight_translations.append(translation)
        
        # Generate framework explanation
        framework_explanation = self._generate_framework_explanation(
            framework, 
            before_translation, 
            after_translation,
            insight_translations
        )
        
        # Create explanation result
        explanation = {
            'framework': framework,
            'before_translation': before_translation,
            'after_translation': after_translation,
            'insight_translations': insight_translations,
            'explanation': framework_explanation,
            'energy_details': application_result if include_energy else None
        }
        
        return explanation
    
    def _translate_vector(self, vector: List[float], domain: str = None) -> Dict:
        """Translate vector components to conceptual properties"""
        result = {}
        
        # Process each dimension
        for i, value in enumerate(vector):
            if i in self.energy_concept_mappings['vector_dimensions']:
                dim_map = self.energy_concept_mappings['vector_dimensions'][i]
                
                # Categorize value
                category = 'mid'
                if value > 0.7:
                    category = 'high'
                elif value < 0.3:
                    category = 'low'
                
                # Get dimension name and interpretation
                dim_name = dim_map['name']
                interpretation = dim_map[category]
                
                # Add domain-specific interpretation if available
                if domain and domain in self.domain_interpretations:
                    domain_key = f"vector_{category}_{dim_name[0]}"
                    if domain_key in self.domain_interpretations[domain]:
                        domain_interp = self.domain_interpretations[domain][domain_key]
                        interpretation = f"{interpretation} ({domain_interp})"
                
                result[dim_name] = {
                    'value': value,
                    'category': category,
                    'interpretation': interpretation
                }
        
        return result
    
    def _translate_scalar_property(self, property_name: str, 
                                 value: float, 
                                 domain: str = None) -> Dict:
        """Translate scalar energy property to conceptual understanding"""
        if property_name not in self.energy_concept_mappings:
            return {'value': value, 'interpretation': str(value)}
        
        prop_map = self.energy_concept_mappings[property_name]
        
        # Categorize value
        category = 'mid'
        if value > 0.7:
            category = 'high'
        elif value < 0.3:
            category = 'low'
        
        # Get interpretation
        interpretation = prop_map[category]
        
        # Add domain-specific interpretation if available
        if domain and domain in self.domain_interpretations:
            domain_key = f"{property_name}_{category}"
            if domain_key in self.domain_interpretations[domain]:
                domain_interp = self.domain_interpretations[domain][domain_key]
                interpretation = f"{interpretation} ({domain_interp})"
        
        return {
            'value': value,
            'category': category,
            'interpretation': interpretation
        }
    
    def _generate_conceptual_description(self, properties: Dict) -> str:
        """Generate natural language description from conceptual properties"""
        descriptions = []
        
        # Add vector dimension descriptions
        if 'vector' in properties:
            for dim_name, dim_info in properties['vector'].items():
                descriptions.append(f"{dim_name.replace('_', ' ')}: {dim_info['interpretation']}")
        
        # Add other property descriptions
        for prop_name in ['frequency', 'entropy', 'magnitude']:
            if prop_name in properties:
                prop_info = properties[prop_name]
                concept_name = self.energy_concept_mappings[prop_name]['name']
                descriptions.append(f"{concept_name}: {prop_info['interpretation']}")
        
        # Combine descriptions
        if descriptions:
            return " ".join(descriptions)
        
        return "This concept has balanced properties."
    
    def _generate_domain_interpretation(self, properties: Dict, domain: str) -> str:
        """Generate domain-specific interpretation"""
        if domain not in self.domain_interpretations:
            return None
        
        domain_interps = []
        domain_map = self.domain_interpretations[domain]
        
        # Check for vector properties
        if 'vector' in properties:
            for dim_name, dim_info in properties['vector'].items():
                domain_key = f"vector_{dim_info['category']}_{dim_name[0]}"
                if domain_key in domain_map:
                    domain_interps.append(domain_map[domain_key])
        
        # Check for other properties
        for prop_name in ['frequency', 'entropy', 'magnitude']:
            if prop_name in properties:
                prop_info = properties[prop_name]
                domain_key = f"{prop_name}_{prop_info['category']}"
                if domain_key in domain_map:
                    domain_interps.append(domain_map[domain_key])
        
        if domain_interps:
            return f"In {domain}, this represents {'; '.join(domain_interps)}."
        
        return None
    
    def _identify_energy_changes(self, before: Dict, after: Dict) -> Dict:
        """Identify key changes between energy signatures"""
        changes = {}
        
        # Check vector changes
        if ('vector' in before and 'value' in before['vector'] and
            'vector' in after and 'value' in after['vector']):
            before_vec = before['vector']['value']
            after_vec = after['vector']['value']
            
            if isinstance(before_vec, list) and isinstance(after_vec, list):
                # Compare available dimensions
                min_len = min(len(before_vec), len(after_vec))
                vec_changes = []
                
                for i in range(min_len):
                    change = after_vec[i] - before_vec[i]
                    if abs(change) > 0.1:  # Only significant changes
                        vec_changes.append({
                            'dimension': i,
                            'before': before_vec[i],
                            'after': after_vec[i],
                            'change': change
                        })
                
                if vec_changes:
                    changes['vector'] = vec_changes
        
        # Check other property changes
        for prop in ['frequency', 'entropy', 'magnitude']:
            if (prop in before and 'value' in before[prop] and
                prop in after and 'value' in after[prop]):
                before_val = before[prop]['value']
                after_val = after[prop]['value']
                
                if isinstance(before_val, (int, float)) and isinstance(after_val, (int, float)):
                    change = after_val - before_val
                    if abs(change) > 0.1:  # Only significant changes
                        changes[prop] = {
                            'before': before_val,
                            'after': after_val,
                            'change': change
                        }
        
        return changes
    
    def _translate_energy_changes(self, changes: Dict) -> Dict:
        """Translate energy changes to conceptual changes"""
        conceptual_changes = {}
        
        # Translate vector changes
        if 'vector' in changes:
            vector_changes = []
            
            for change_info in changes['vector']:
                dim = change_info['dimension']
                change_val = change_info['change']
                
                if dim in self.energy_concept_mappings['vector_dimensions']:
                    dim_map = self.energy_concept_mappings['vector_dimensions'][dim]
                    dim_name = dim_map['name']
                    
                    # Generate description based on direction
                    if change_val > 0:
                        # Increasing value
                        description = f"increasing {dim_name.replace('_', ' ')}"
                        if change_val > 0.3:
                            description += " significantly"
                    else:
                        # Decreasing value
                        description = f"decreasing {dim_name.replace('_', ' ')}"
                        if change_val < -0.3:
                            description += " significantly"
                    
                    vector_changes.append({
                        'dimension_name': dim_name,
                        'change': change_val,
                        'description': description
                    })
            
            if vector_changes:
                conceptual_changes['vector'] = vector_changes
        
        # Translate other property changes
        for prop in ['frequency', 'entropy', 'magnitude']:
            if prop in changes:
                change_info = changes[prop]
                change_val = change_info['change']
                
                if prop in self.energy_concept_mappings:
                    prop_map = self.energy_concept_mappings[prop]
                    concept_name = prop_map['name']
                    
                    # Generate description based on direction
                    if change_val > 0:
                        # Increasing value
                        description = f"increasing {concept_name}"
                        if change_val > 0.3:
                            description += " significantly"
                    else:
                        # Decreasing value
                        description = f"decreasing {concept_name}"
                        if change_val < -0.3:
                            description += " significantly"
                    
                    conceptual_changes[prop] = {
                        'concept_name': concept_name,
                        'change': change_val,
                        'description': description
                    }
        
        return conceptual_changes
    
    def _generate_operation_explanation(self, operation: str,
                                      operation_concept: Dict,
                                      conceptual_changes: Dict) -> str:
        """Generate explanation of operation with conceptual changes"""
        concept = operation_concept['concept']
        process = operation_concept['process']
        
        # Start with operation concept
        explanation = f"This is {concept}, {process}."
        
        # Add description of key changes
        changes = []
        
        if 'vector' in conceptual_changes:
            for change in conceptual_changes['vector']:
                changes.append(change['description'])
        
        for prop in ['frequency', 'entropy', 'magnitude']:
            if prop in conceptual_changes:
                changes.append(conceptual_changes[prop]['description'])
        
        if changes:
            explanation += f" This involves {', '.join(changes)}."
        
        return explanation
    
    def _select_relevant_metaphor(self, operation: str, changes: Dict) -> Dict:
        """Select relevant experiential metaphor based on operation and changes"""
        # Map operations to likely metaphors
        operation_metaphor_map = {
            'shift_up': 'transcendence',
            'shift_down': 'grounding',
            'merge': 'emergence',
            'bifurcate': 'friction',
            'oscillate': 'resonance',
            'connect': 'resonance',
            'disconnect': 'friction'
        }
        
        # Check if operation has direct metaphor mapping
        if operation in operation_metaphor_map:
            metaphor_name = operation_metaphor_map[operation]
            if metaphor_name in self.experiential_metaphors:
                return {
                    'name': metaphor_name,
                    **self.experiential_metaphors[metaphor_name]
                }
        
        # Otherwise select based on energy changes
        if 'vector' in changes:
            # Check for significant vertical movement
            for change in changes['vector']:
                if change['dimension'] == 1:  # Y-axis
                    if change['change'] > 0.3:
                        return {
                            'name': 'transcendence',
                            **self.experiential_metaphors['transcendence']
                        }
                    elif change['change'] < -0.3:
                        return {
                            'name': 'grounding',
                            **self.experiential_metaphors['grounding']
                        }
        
        # Check for changes in other properties
        if 'entropy' in changes:
            if changes['entropy']['change'] < -0.3:
                return {
                    'name': 'emergence',
                    **self.experiential_metaphors['emergence']
                }
            elif changes['entropy']['change'] > 0.3:
                return {
                    'name': 'friction',
                    **self.experiential_metaphors['friction']
                }
        
        # Default to resonance
        return {
            'name': 'resonance',
            **self.experiential_metaphors['resonance']
        }
    
    def _select_example(self, operation_concept: Dict, changes: Dict) -> str:
        """Select appropriate example based on operation and changes"""
        # If operation has examples, select one
        if 'examples' in operation_concept and operation_concept['examples']:
            # Simple selection for now - could be more sophisticated
            return np.random.choice(operation_concept['examples'])
        
        return None
    
    def _generate_insight_description(self, insight: Dict) -> str:
        """Generate description for an insight based on its type"""
        description = "This insight reveals a philosophical pattern."
        
        if 'type' in insight:
            insight_type = insight['type']
            
            if insight_type == 'similarity':
                description = "This insight reveals similarities between concepts."
            elif insight_type == 'contrast':
                description = "This insight highlights important contrasts."
            elif insight_type == 'analogy':
                description = "This insight demonstrates analogical relationships."
            elif insight_type == 'abstraction':
                description = "This insight moves toward higher abstraction."
            elif insight_type == 'concretization':
                description = "This insight connects abstract ideas to concrete examples."
            elif insight_type == 'causation':
                description = "This insight reveals causal relationships."
            elif insight_type == 'dialectic':
                description = "This insight explores dialectical tensions."
        
        # Add concept information if available
        concepts = []
        if 'source_concepts' in insight:
            concepts.extend(insight['source_concepts'])
        if 'target_concepts' in insight:
            concepts.extend(insight['target_concepts'])
        
        if concepts:
            concept_str = ", ".join(concepts)
            description += f" It involves the concepts: {concept_str}."
        
        return description
    
    def _extract_energy_patterns(self, insight: Dict) -> Dict:
        """Extract energy patterns from an insight"""
        patterns = {}
        
        # Different insights store energy differently
        if 'energy_signature' in insight:
            patterns['primary'] = insight['energy_signature']
        
        if 'source_energy' in insight:
            patterns['source'] = insight['source_energy']
        
        if 'target_energy' in insight:
            patterns['target'] = insight['target_energy']
        
        if 'result_energy' in insight:
            patterns['result'] = insight['result_energy']
        
        return patterns
    
    def _find_relevant_metaphors(self, insight: Dict, 
                               energy_patterns: Dict) -> List[Dict]:
        """Find metaphors relevant to this insight"""
        metaphors = []
        
        # Check insight type
        if 'type' in insight:
            insight_type = insight['type']
            
            # Map insight types to likely metaphors
            type_metaphor_map = {
                'similarity': 'resonance',
                'contrast': 'friction',
                'abstraction': 'transcendence',
                'concretization': 'grounding',
                'emergence': 'emergence',
                'dialectic': 'oscillate'
            }
            
            if insight_type in type_metaphor_map:
                metaphor_name = type_metaphor_map[insight_type]
                if metaphor_name in self.experiential_metaphors:
                    metaphors.append({
                        'name': metaphor_name,
                        **self.experiential_metaphors[metaphor_name]
                    })
        
        # Check energy patterns
        if 'source' in energy_patterns and 'target' in energy_patterns:
            changes = self._identify_energy_changes(
                energy_patterns['source'], energy_patterns['target']
            )
            
            metaphor = self._select_relevant_metaphor('unknown', changes)
            
            # Only add if different from already added
            if not metaphors or metaphors[0]['name'] != metaphor['name']:
                metaphors.append(metaphor)
        
        # If no metaphors found, add default
        if not metaphors:
            metaphors.append({
                'name': 'resonance',
                **self.experiential_metaphors['resonance']
            })
        
        return metaphors
    
    def _generate_insight_examples(self, insight: Dict, 
                                 conceptual_patterns: Dict) -> List[str]:
        """Generate examples for insight based on patterns"""
        examples = []
        
        # Check insight type
        if 'type' in insight:
            insight_type = insight['type']
            
            # Get concepts
            concepts = []
            if 'source_concepts' in insight:
                concepts.extend(insight['source_concepts'])
            if 'target_concepts' in insight:
                concepts.extend(insight['target_concepts'])
            
            # Generate examples based on insight type and concepts
            if insight_type == 'similarity' and len(concepts) >= 2:
                examples.append(f"Just as {concepts[0]} involves patterns of organization, "
                              f"{concepts[1]} also exhibits structured relationships.")
            
            elif insight_type == 'contrast' and len(concepts) >= 2:
                examples.append(f"While {concepts[0]} emphasizes unity and wholeness, "
                              f"{concepts[1]} focuses on differentiation and particularity.")
            
            elif insight_type == 'abstraction' and len(concepts) >= 2:
                examples.append(f"Moving from {concepts[0]} to {concepts[1]} involves "
                              f"recognizing the general principles beyond specific instances.")
            
            elif insight_type == 'concretization' and len(concepts) >= 2:
                examples.append(f"The abstract principle of {concepts[0]} takes concrete "
                              f"form in the specific example of {concepts[1]}.")
            
            elif insight_type == 'causation' and len(concepts) >= 2:
                examples.append(f"Changes in {concepts[0]} create ripple effects "
                              f"that influence {concepts[1]}.")
            
            elif insight_type == 'dialectic' and len(concepts) >= 2:
                examples.append(f"The tension between {concepts[0]} and {concepts[1]} "
                              f"creates a dynamic that drives philosophical development.")
        
        # Add domain-specific example if appropriate
        if 'domain' in insight:
            domain = insight['domain']
            
            if domain == 'ethics':
                examples.append("In ethical decision-making, this insight helps us recognize "
                              "how principles must be adapted to specific situations.")
            
            elif domain == 'epistemology':
                examples.append("This insight illuminates how knowledge develops through "
                              "the interaction of experience and theoretical frameworks.")
            
            elif domain == 'metaphysics':
                examples.append("This pattern reveals fundamental structures that "
                              "underlie diverse manifestations of reality.")
        
        return examples
    
    def _create_integrated_explanation(self, human_description: str,
                                     conceptual_patterns: Dict,
                                     metaphors: List[Dict],
                                     examples: List[str],
                                     include_energy: bool) -> str:
        """Create integrated explanation with both human and energy elements"""
        parts = [human_description]
        
        # Add conceptual pattern descriptions
        if conceptual_patterns:
            for pattern_name, pattern in conceptual_patterns.items():
                if 'description' in pattern:
                    if pattern_name == 'primary':
                        parts.append(f"This represents {pattern['description']}.")
                    else:
                        parts.append(f"The {pattern_name} element represents {pattern['description']}.")
        
        # Add metaphorical understanding
        if metaphors:
            metaphor = metaphors[0]
            parts.append(f"This is like {metaphor['description']} - {np.random.choice(metaphor['experiential_examples'])}")
        
        # Add example
        if examples:
            parts.append(f"For example: {examples[0]}")
        
        # Add energy explanation if requested
        if include_energy and conceptual_patterns:
            energy_parts = []
            
            for pattern_name, pattern in conceptual_patterns.items():
                if 'energy_signature' in pattern:
                    sig = pattern['energy_signature']
                    
                    # Add key energy properties
                    props = []
                    
                    if 'vector' in sig and 'value' in sig['vector']:
                        vector = sig['vector']['value']
                        if isinstance(vector, list) and len(vector) >= 3:
                            props.append(f"vector position [{', '.join(f'{v:.2f}' for v in vector[:3])}]")
                    
                    for prop in ['frequency', 'entropy', 'magnitude']:
                        if prop in sig and 'value' in sig[prop]:
                            value = sig[prop]['value']
                            if isinstance(value, (int, float)):
                                props.append(f"{prop} {value:.2f}")
                    
                    if props:
                        if pattern_name == 'primary':
                            energy_parts.append(f"Energy signature: {', '.join(props)}")
                        else:
                            energy_parts.append(f"{pattern_name.capitalize()} energy: {', '.join(props)}")
            
            if energy_parts:
                parts.append("In energy terms: " + " ".join(energy_parts))
        
        return " ".join(parts)
    
    def _generate_path_explanation(self, steps: List[Dict], 
                                 include_energy_details: bool) -> str:
        """Generate explanation of reasoning path"""
        if not steps:
            return "No reasoning path available."
        
        explanations = []
        
        # Add each step explanation
        for step in steps:
            explanations.append(step['explanation'])
            
            # Add metaphor if available
            if step.get('metaphor'):
                metaphor = step['metaphor']
                metaphor_example = np.random.choice(metaphor['experiential_examples'])
                explanations.append(f"This is like {metaphor['description']} - {metaphor_example}")
        
        # Add energy details if requested
        if include_energy_details:
            energy_patterns = []
            
            for step in steps:
                if 'energy_translation' in step and step['energy_translation']:
                    energy_sig = step['energy_translation'].get('energy_signature', {})
                    
                    # Add key energy properties
                    props = []
                    
                    if 'vector' in energy_sig and 'value' in energy_sig['vector']:
                        vector = energy_sig['vector']['value']
                        if isinstance(vector, list) and len(vector) >= 3:
                            props.append(f"vector position [{', '.join(f'{v:.2f}' for v in vector[:3])}]")
                    
                    for prop in ['frequency', 'entropy', 'magnitude']:
                        if prop in energy_sig and 'value' in energy_sig[prop]:
                            value = energy_sig[prop]['value']
                            if isinstance(value, (int, float)):
                                props.append(f"{prop} {value:.2f}")
                    
                    if props:
                        energy_patterns.append(f"At '{step['concept_id']}': {', '.join(props)}")
            
            if energy_patterns:
                explanations.append("Energy patterns: " + " ".join(energy_patterns))
        
        return " ".join(explanations)
    
    def _generate_framework_explanation(self, framework: str,
                                      before_translation: Dict,
                                      after_translation: Dict,
                                      insight_translations: List[Dict]) -> str:
        """Generate explanation of framework application"""
        parts = [f"Applied the {framework} philosophical framework."]
        
        # Add before/after comparison
        if before_translation and after_translation:
            parts.append(f"Starting with {before_translation['description']}, "
                        f"the framework reveals {after_translation['description']}.")
        
        # Add key insights
        if insight_translations:
            insights = []
            for i, insight in enumerate(insight_translations[:3]):  # Limit to top 3
                insights.append(insight['explanation'])
            
            parts.append("Key insights: " + " ".join(insights))
        
        return " ".join(parts)