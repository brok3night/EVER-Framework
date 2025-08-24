"""
Philosophical Articulation - Integrates energy reasoning with human-understandable articulation
"""
from typing import Dict, List, Any

class PhilosophicalArticulation:
    """
    Integrates EVER's energy-based philosophical reasoning with
    human-understandable articulation
    """
    
    def __init__(self, network_reasoning, energy_conceptual_bridge, config=None):
        self.reasoning = network_reasoning
        self.bridge = energy_conceptual_bridge
        
        # Default configuration
        self.config = {
            'include_energy_details': True,
            'explanation_verbosity': 'medium',  # 'low', 'medium', 'high'
            'metaphor_use': 'moderate',  # 'minimal', 'moderate', 'extensive'
            'example_use': 'moderate'    # 'minimal', 'moderate', 'extensive'
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
    
    def articulate_concept_analysis(self, concept_id: str) -> Dict:
        """
        Analyze a concept and articulate the results
        
        Args:
            concept_id: Concept to analyze
            
        Returns:
            Analysis with human-understandable articulation
        """
        # Get concept from network
        if concept_id not in self.reasoning.network.concepts:
            return {
                'concept_id': concept_id,
                'error': 'Concept not found',
                'explanation': f"The concept '{concept_id}' is not recognized."
            }
        
        # Get energy signature
        energy_signature = self.reasoning.network.concepts[concept_id]
        
        # Get concept context
        context = self.reasoning.network.get_concept_context(concept_id)
        
        # Get connected concepts
        connected = self.reasoning.network.get_connected_concepts(concept_id)
        connected_concepts = list(connected.keys())
        
        # Find insights about this concept
        insights = self.reasoning.discover_insights([concept_id] + connected_concepts[:3])
        
        # Translate energy signature
        energy_translation = self.bridge.translate_energy_signature(energy_signature)
        
        # Translate insights
        insight_translations = []
        for insight in insights:
            translation = self.bridge.translate_philosophical_insight(
                insight, include_energy=self.config['include_energy_details']
            )
            insight_translations.append(translation)
        
        # Generate human-understandable explanation
        explanation = self._generate_concept_explanation(
            concept_id, energy_translation, context, insight_translations
        )
        
        # Create result
        result = {
            'concept_id': concept_id,
            'energy_signature': energy_signature,
            'energy_translation': energy_translation,
            'context': context,
            'connected_concepts': connected_concepts,
            'insights': insights,
            'insight_translations': insight_translations,
            'explanation': explanation
        }
        
        # Remove energy details if configured
        if not self.config['include_energy_details']:
            result.pop('energy_signature')
            result.pop('context')
        
        return result
    
    def articulate_concept_comparison(self, concept_id1: str, concept_id2: str) -> Dict:
        """
        Compare two concepts and articulate the results
        
        Args:
            concept_id1: First concept
            concept_id2: Second concept
            
        Returns:
            Comparison with human-understandable articulation
        """
        # Check if concepts exist
        if concept_id1 not in self.reasoning.network.concepts:
            return {
                'error': 'Concept not found',
                'explanation': f"The concept '{concept_id1}' is not recognized."
            }
        
        if concept_id2 not in self.reasoning.network.concepts:
            return {
                'error': 'Concept not found',
                'explanation': f"The concept '{concept_id2}' is not recognized."
            }
        
        # Get energy signatures
        energy1 = self.reasoning.network.concepts[concept_id1]
        energy2 = self.reasoning.network.concepts[concept_id2]
        
        # Find path between concepts
        path = self.reasoning.network.find_resonance_path(concept_id1, concept_id2)
        
        # Apply reasoning strategies
        similarity_result = self.reasoning.reason_across_concepts(
            [concept_id1, concept_id2], 'similarity'
        )
        
        contrast_result = self.reasoning.reason_across_concepts(
            [concept_id1, concept_id2], 'contrast'
        )
        
        # Translate energy signatures
        energy_translation1 = self.bridge.translate_energy_signature(energy1)
        energy_translation2 = self.bridge.translate_energy_signature(energy2)
        
        # Translate path
        path_explanation = self.bridge.explain_reasoning_path(
            path, include_energy_details=self.config['include_energy_details']
        )
        
        # Translate reasoning results
        similarity_insights = []
        if 'insights' in similarity_result:
            for insight in similarity_result['insights']:
                translation = self.bridge.translate_philosophical_insight(
                    insight, include_energy=self.config['include_energy_details']
                )
                similarity_insights.append(translation)
        
        contrast_insights = []
        if 'insights' in contrast_result:
            for insight in contrast_result['insights']:
                translation = self.bridge.translate_philosophical_insight(
                    insight, include_energy=self.config['include_energy_details']
                )
                contrast_insights.append(translation)
        
        # Generate human-understandable explanation
        explanation = self._generate_comparison_explanation(
            concept_id1, concept_id2,
            energy_translation1, energy_translation2,
            path_explanation, similarity_insights, contrast_insights
        )
        
        # Create result
        result = {
            'concepts': [concept_id1, concept_id2],
            'energy_signatures': [energy1, energy2],
            'energy_translations': [energy_translation1, energy_translation2],
            'path': path,
            'path_explanation': path_explanation,
            'similarity_insights': similarity_insights,
            'contrast_insights': contrast_insights,
            'explanation': explanation
        }
        
        # Remove energy details if configured
        if not self.config['include_energy_details']:
            result.pop('energy_signatures')
        
        return result
    
    def articulate_philosophical_question(self, question: str, 
                                        relevant_concepts: List[str]) -> Dict:
        """
        Analyze a philosophical question and articulate the results
        
        Args:
            question: Philosophical question
            relevant_concepts: Concepts relevant to the question
            
        Returns:
            Analysis with human-understandable articulation
        """
        # Check if concepts exist
        existing_concepts = []
        for concept_id in relevant_concepts:
            if concept_id in self.reasoning.network.concepts:
                existing_concepts.append(concept_id)
        
        if not existing_concepts:
            return {
                'question': question,
                'error': 'No valid concepts',
                'explanation': "None of the provided concepts are recognized."
            }
        
        # Get resonant field
        field = self.reasoning.network.get_resonant_field(existing_concepts)
        field_concepts = list(field.keys())
        
        # Navigate concept space
        navigation = self.reasoning.navigate_concept_space(existing_concepts, max_steps=5)
        
        # Apply different reasoning strategies
        reasoning_results = {}
        for strategy in ['dialectic', 'analogy', 'abstraction']:
            result = self.reasoning.reason_across_concepts(existing_concepts, strategy)
            reasoning_results[strategy] = result
        
        # Translate navigation
        path_explanation = self.bridge.explain_reasoning_path(
            navigation.get('path', []), 
            include_energy_details=self.config['include_energy_details']
        )
        
        # Translate reasoning results
        strategy_insights = {}
        for strategy, result in reasoning_results.items():
            insights = []
            if 'insights' in result:
                for insight in result['insights']:
                    translation = self.bridge.translate_philosophical_insight(
                        insight, include_energy=self.config['include_energy_details']
                    )
                    insights.append(translation)
            
            strategy_insights[strategy] = insights
        
        # Generate human-understandable explanation
        explanation = self._generate_question_explanation(
            question, existing_concepts, path_explanation, strategy_insights
        )
        
        # Create result
        result = {
            'question': question,
            'relevant_concepts': existing_concepts,
            'field_concepts': field_concepts,
            'navigation': navigation,
            'path_explanation': path_explanation,
            'strategy_insights': strategy_insights,
            'explanation': explanation
        }
        
        return result
    
    def articulate_framework_application(self, framework: str,
                                       concepts: List[str]) -> Dict:
        """
        Apply a philosophical framework and articulate the results
        
        Args:
            framework: Philosophical framework to apply
            concepts: Concepts to analyze
            
        Returns:
            Analysis with human-understandable articulation
        """
        # Check if framework exists
        available_frameworks = self.reasoning.primitives.get_available_frameworks()
        if framework not in available_frameworks:
            return {
                'error': 'Framework not found',
                'explanation': f"The philosophical framework '{framework}' is not recognized."
            }
        
        # Check if concepts exist
        existing_concepts = []
        for concept_id in concepts:
            if concept_id in self.reasoning.network.concepts:
                existing_concepts.append(concept_id)
        
        if not existing_concepts:
            return {
                'framework': framework,
                'error': 'No valid concepts',
                'explanation': "None of the provided concepts are recognized."
            }
        
        # Apply framework
        framework_result = self.reasoning.apply_philosophical_framework(
            existing_concepts, framework
        )
        
        # Translate framework application
        framework_explanation = self.bridge.explain_framework_application(
            framework, framework_result, include_energy=self.config['include_energy_details']
        )
        
        # Create result
        result = {
            'framework': framework,
            'concepts': existing_concepts,
            'framework_result': framework_result,
            'framework_explanation': framework_explanation,
            'explanation': framework_explanation['explanation']
        }
        
        # Remove energy details if configured
        if not self.config['include_energy_details']:
            result.pop('framework_result')
        
        return result
    
    def _generate_concept_explanation(self, concept_id: str,
                                    energy_translation: Dict,
                                    context: Dict,
                                    insight_translations: List[Dict]) -> str:
        """Generate explanation for concept analysis"""
        # Start with basic concept description
        explanation = [f"The concept of '{concept_id}' represents {energy_translation['description']}."]
        
        # Add strongest connections if available
        if 'strongest_connections' in context and context['strongest_connections']:
            connections = []
            for target, info in context['strongest_connections'][:3]:
                connections.append(target)
            
            if connections:
                explanation.append(f"It is most strongly connected to {', '.join(connections)}.")
        
        # Add key insights
        if insight_translations:
            insights = []
            verbosity = self.config['explanation_verbosity']
            
            # Determine how many insights to include
            insight_count = 1 if verbosity == 'low' else (2 if verbosity == 'medium' else 3)
            
            for i, insight in enumerate(insight_translations[:insight_count]):
                insights.append(insight['explanation'])
            
            explanation.append("Key insights: " + " ".join(insights))
        
        return " ".join(explanation)
    
    def _generate_comparison_explanation(self, concept_id1: str, concept_id2: str,
                                       energy_translation1: Dict,
                                       energy_translation2: Dict,
                                       path_explanation: Dict,
                                       similarity_insights: List[Dict],
                                       contrast_insights: List[Dict]) -> str:
        """Generate explanation for concept comparison"""
        # Start with basic comparison
        explanation = [
            f"Comparing '{concept_id1}' and '{concept_id2}':",
            f"'{concept_id1}' represents {energy_translation1['description']}",
            f"'{concept_id2}' represents {energy_translation2['description']}"
        ]
        
        # Add path explanation
        if 'explanation' in path_explanation:
            explanation.append("Connection: " + path_explanation['explanation'])
        
        # Add similarities
        if similarity_insights:
            similarities = []
            for insight in similarity_insights[:2]:  # Limit to 2
                similarities.append(insight['explanation'])
            
            if similarities:
                explanation.append("Similarities: " + " ".join(similarities))
        
        # Add contrasts
        if contrast_insights:
            contrasts = []
            for insight in contrast_insights[:2]:  # Limit to 2
                contrasts.append(insight['explanation'])
            
            if contrasts:
                explanation.append("Differences: " + " ".join(contrasts))
        
        return " ".join(explanation)
    
    def _generate_question_explanation(self, question: str,
                                     concepts: List[str],
                                     path_explanation: Dict,
                                     strategy_insights: Dict) -> str:
        """Generate explanation for philosophical question"""
        # Start with question restatement
        explanation = [f"Analyzing the question: {question}"]
        
        # Add concept exploration
        if concepts:
            explanation.append(f"Exploring relevant concepts: {', '.join(concepts)}")
        
        # Add exploration path
        if 'explanation' in path_explanation:
            explanation.append("Conceptual exploration: " + path_explanation['explanation'])
        
        # Add insights from different strategies
        for strategy, insights in strategy_insights.items():
            if insights:
                strategy_name = strategy.capitalize()
                insight_texts = []
                
                for insight in insights[:1]:  # Just top insight per strategy
                    insight_texts.append(insight['explanation'])
                
                if insight_texts:
                    explanation.append(f"{strategy_name} perspective: " + " ".join(insight_texts))
        
        # Add synthesis
        explanation.append("In synthesis, this question involves the interplay of multiple philosophical dimensions.")
        
        return " ".join(explanation)