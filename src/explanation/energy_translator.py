"""
Energy-to-Language Translator - Translates energy-based reasoning into natural language
"""
from typing import Dict, List, Any, Tuple

class EnergyTranslator:
    """Translates energy signatures and reasoning processes into natural language"""
    
    def __init__(self, config=None):
        # Translation templates for different energy patterns
        self.energy_templates = {
            # Vector orientation templates
            'vector_high_y': "This concept has high abstraction.",
            'vector_low_y': "This concept is very concrete.",
            'vector_high_x': "This concept has forward-looking momentum.",
            'vector_low_x': "This concept has historical or foundational quality.",
            'vector_balanced': "This concept has balanced properties.",
            
            # Frequency templates
            'high_frequency': "This concept involves rapid change or oscillation.",
            'low_frequency': "This concept has stability and permanence.",
            
            # Entropy templates
            'high_entropy': "This concept contains significant complexity or ambiguity.",
            'low_entropy': "This concept has clarity and precision.",
            
            # Magnitude templates
            'high_magnitude': "This concept has significant presence or impact.",
            'low_magnitude': "This concept has subtle or gentle influence."
        }
        
        # Translation templates for primitive operations
        self.primitive_templates = {
            'shift_up': "moving to a higher level of abstraction",
            'shift_down': "examining concrete manifestations",
            'shift_left': "exploring foundational aspects",
            'shift_right': "considering future implications",
            'invert': "considering the opposite perspective",
            'merge': "synthesizing different elements",
            'bifurcate': "recognizing a fundamental division",
            'oscillate': "observing cyclic patterns",
            'expand': "broadening the conceptual scope",
            'contract': "focusing on essential elements",
            'connect': "establishing relationships",
            'disconnect': "recognizing distinctions",
            'loop': "identifying recursive patterns",
            'ground': "connecting to empirical reality"
        }
        
        # Translation templates for philosophical frameworks
        self.framework_templates = {
            'dialectical': "Through dialectical analysis, examining thesis and antithesis to find synthesis",
            'existentialist': "Through existential inquiry, examining meaning and authentic existence",
            'pragmatic': "Through pragmatic assessment, focusing on practical consequences",
            'analytical': "Through analytical examination, clarifying concepts and logical structure",
            'phenomenological': "Through phenomenological reflection, exploring direct experience"
        }
        
        # Translation templates for reasoning types
        self.reasoning_templates = {
            'similarity': "recognizing patterns of similarity",
            'contrast': "identifying meaningful contrasts",
            'analogy': "drawing analogical connections",
            'composition': "analyzing compositional structure",
            'abstraction': "moving toward higher abstraction",
            'concretization': "exploring concrete instances",
            'causation': "examining causal relationships",
            'dialectic': "resolving dialectical tensions"
        }
    
    def translate_energy_signature(self, energy: Dict) -> str:
        """
        Translate energy signature into natural language description
        
        Args:
            energy: Energy signature to translate
            
        Returns:
            Natural language description
        """
        descriptions = []
        
        # Translate vector properties
        if 'vector' in energy and 'value' in energy['vector']:
            vector = energy['vector']['value']
            if isinstance(vector, list) and len(vector) >= 3:
                # Y component (vertical axis) - abstraction level
                if vector[1] > 0.7:
                    descriptions.append(self.energy_templates['vector_high_y'])
                elif vector[1] < 0.3:
                    descriptions.append(self.energy_templates['vector_low_y'])
                
                # X component (horizontal axis) - temporal dimension
                if vector[0] > 0.7:
                    descriptions.append(self.energy_templates['vector_high_x'])
                elif vector[0] < 0.3:
                    descriptions.append(self.energy_templates['vector_low_x'])
                
                # Check for balance
                if all(0.4 < v < 0.6 for v in vector):
                    descriptions.append(self.energy_templates['vector_balanced'])
        
        # Translate frequency
        if 'frequency' in energy and 'value' in energy['frequency']:
            freq = energy['frequency']['value']
            if isinstance(freq, (int, float)):
                if freq > 0.7:
                    descriptions.append(self.energy_templates['high_frequency'])
                elif freq < 0.3:
                    descriptions.append(self.energy_templates['low_frequency'])
        
        # Translate entropy
        if 'entropy' in energy and 'value' in energy['entropy']:
            entropy = energy['entropy']['value']
            if isinstance(entropy, (int, float)):
                if entropy > 0.7:
                    descriptions.append(self.energy_templates['high_entropy'])
                elif entropy < 0.3:
                    descriptions.append(self.energy_templates['low_entropy'])
        
        # Translate magnitude
        if 'magnitude' in energy and 'value' in energy['magnitude']:
            magnitude = energy['magnitude']['value']
            if isinstance(magnitude, (int, float)):
                if magnitude > 0.7:
                    descriptions.append(self.energy_templates['high_magnitude'])
                elif magnitude < 0.3:
                    descriptions.append(self.energy_templates['low_magnitude'])
        
        # Combine descriptions
        if descriptions:
            return " ".join(descriptions)
        
        return "This concept has balanced energy properties."
    
    def translate_primitive_operation(self, primitive: str, 
                                    source_energy: Dict = None,
                                    result_energy: Dict = None) -> str:
        """
        Translate a primitive operation into natural language
        
        Args:
            primitive: Primitive operation name
            source_energy: Original energy signature
            result_energy: Resulting energy signature
            
        Returns:
            Natural language description
        """
        if primitive in self.primitive_templates:
            base_description = self.primitive_templates[primitive]
            
            # If we have before/after energies, enhance the description
            if source_energy and result_energy:
                # Describe significant changes
                changes = self._describe_energy_changes(source_energy, result_energy)
                
                if changes:
                    return f"{base_description}, which results in {changes}"
            
            return base_description
        
        return f"applying the {primitive} operation"
    
    def translate_framework_application(self, framework: str,
                                      source_concepts: List[str] = None) -> str:
        """
        Translate framework application into natural language
        
        Args:
            framework: Philosophical framework
            source_concepts: Concepts being analyzed
            
        Returns:
            Natural language description
        """
        if framework in self.framework_templates:
            base_description = self.framework_templates[framework]
            
            if source_concepts:
                concept_list = ", ".join(source_concepts)
                return f"{base_description} of {concept_list}"
            
            return base_description
        
        return f"applying the {framework} framework"
    
    def translate_reasoning_process(self, reasoning_type: str,
                                  source_concepts: List[str] = None,
                                  target_concepts: List[str] = None) -> str:
        """
        Translate reasoning process into natural language
        
        Args:
            reasoning_type: Type of reasoning
            source_concepts: Starting concepts
            target_concepts: Resulting concepts
            
        Returns:
            Natural language description
        """
        if reasoning_type in self.reasoning_templates:
            base_description = self.reasoning_templates[reasoning_type]
            
            if source_concepts and target_concepts:
                source_list = ", ".join(source_concepts)
                target_list = ", ".join(target_concepts)
                return f"{base_description} between {source_list} and {target_list}"
            elif source_concepts:
                source_list = ", ".join(source_concepts)
                return f"{base_description} starting from {source_list}"
            
            return base_description
        
        return f"applying {reasoning_type} reasoning"
    
    def translate_insight(self, insight: Dict) -> str:
        """
        Translate philosophical insight into natural language
        
        Args:
            insight: Philosophical insight
            
        Returns:
            Natural language explanation
        """
        # Basic insight description
        description = insight.get('description', '')
        
        # If already in natural language, use that
        if description:
            explanation = description
        else:
            # Generate description based on insight type
            insight_type = insight.get('type', '')
            source_concepts = insight.get('source_concepts', [])
            target_concepts = insight.get('target_concepts', [])
            
            explanation = self.translate_reasoning_process(
                insight_type, source_concepts, target_concepts
            )
        
        # Enhance with philosophical framing if available
        if 'framework' in insight:
            framework = insight['framework']
            framework_description = self.translate_framework_application(
                framework, source_concepts
            )
            
            explanation = f"{framework_description}: {explanation}"
        
        # Add empirical grounding if available
        if 'grounding' in insight:
            grounding = insight['grounding']
            grounding_strength = grounding.get('grounding_strength', 0)
            
            if grounding_strength > 0.7:
                explanation += " This insight has strong empirical grounding."
            elif grounding_strength > 0.4:
                explanation += " This insight has moderate empirical support."
            elif grounding_strength > 0:
                explanation += " This insight has limited empirical connections."
        
        return explanation
    
    def explain_reasoning_path(self, path: List[Dict]) -> str:
        """
        Explain a reasoning path through concept space
        
        Args:
            path: Sequence of reasoning steps
            
        Returns:
            Natural language explanation
        """
        if not path:
            return "No reasoning path available."
        
        # Extract reasoning steps
        steps = []
        
        for i, step in enumerate(path):
            if i == 0:
                # First step
                concept_id = step.get('concept_id', '')
                steps.append(f"Starting with the concept of {concept_id}")
            