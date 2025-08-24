"""
Philosophical Comprehension - Enables EVER to leverage philosophical reasoning for understanding statements
"""
from typing import Dict, List, Any, Tuple
import numpy as np
from src.reasoning.primitive_actions import PrimitiveActions
from src.reasoning.topographical_reasoning import TopographicalReasoning

class PhilosophicalComprehension:
    """Provides philosophical comprehension capabilities for EVER"""
    
    def __init__(self, energy_system, philosophical_reasoning):
        self.energy = energy_system
        self.reasoning = philosophical_reasoning
        self.primitives = self.reasoning.primitives
        
        # Comprehension patterns discovered through experience
        self.comprehension_patterns = []
        
        # Statement type classifiers
        self.statement_types = {
            'factual': self._detect_factual,
            'normative': self._detect_normative,
            'conceptual': self._detect_conceptual,
            'paradoxical': self._detect_paradoxical,
            'metaphorical': self._detect_metaphorical,
            'existential': self._detect_existential,
            'analytical': self._detect_analytical,
            'synthetic': self._detect_synthetic
        }
        
        # Statement types to philosophical approaches mapping
        self.approach_mapping = {
            'factual': ['deductive', 'inductive'],
            'normative': ['critical', 'pragmatic'],
            'conceptual': ['analytical', 'dialectical'],
            'paradoxical': ['dialectical', 'existential'],
            'metaphorical': ['analogical', 'hermeneutic'],
            'existential': ['existential', 'phenomenological'],
            'analytical': ['analytical', 'deductive'],
            'synthetic': ['synthetic', 'abductive']
        }
        
        # Track comprehension efficacy
        self.comprehension_efficacy = {}
    
    def comprehend_statement(self, statement: str, statement_energy: Dict,
                            context_energies: List[Dict] = None) -> Dict:
        """
        Comprehend a statement using philosophical reasoning
        
        Args:
            statement: The statement text
            statement_energy: Energy signature of the statement
            context_energies: Energy signatures from context
            
        Returns:
            Comprehension results including philosophical insights
        """
        # Detect statement types
        statement_types = self._classify_statement(statement, statement_energy)
        
        # Initialize comprehension results
        comprehension = {
            'statement': statement,
            'statement_types': statement_types,
            'base_energy': statement_energy,
            'philosophical_insights': [],
            'comprehension_pathways': [],
            'meta_understanding': {}
        }
        
        # Check for matching comprehension patterns
        pattern_match = self._find_matching_pattern(statement_energy, statement_types)
        
        if pattern_match:
            # Apply proven comprehension pattern
            comprehension['pattern_match'] = pattern_match['pattern_name']
            comprehension['comprehension_pathways'].append({
                'type': 'pattern_application',
                'pattern': pattern_match['pattern_name'],
                'actions': pattern_match['actions']
            })
            
            # Apply the pattern's action sequence
            comprehended_energy = self.reasoning.primitives.apply_sequence(
                pattern_match['actions'], statement_energy, context_energies)
            
            comprehension['comprehended_energy'] = comprehended_energy
            
            # Generate insights from this pattern
            insights = self._generate_insights_from_pattern(
                pattern_match, statement, comprehended_energy)
            
            comprehension['philosophical_insights'].extend(insights)
        else:
            # No matching pattern, use approach mapping
            relevant_approaches = self._get_relevant_approaches(statement_types)
            
            best_approach = None
            best_resonance = 0
            best_energy = None
            
            # Try each relevant philosophical approach
            for approach in relevant_approaches:
                # Apply this approach
                approach_energy = self.reasoning.find_reasoning_path(
                    statement_energy, approach, context_energies)
                
                # Calculate resonance improvement
                resonance = self._calculate_comprehension_resonance(
                    statement_energy, approach_energy, statement_types)
                
                if resonance > best_resonance:
                    best_resonance = resonance
                    best_approach = approach
                    best_energy = approach_energy
                
                # Record pathway
                comprehension['comprehension_pathways'].append({
                    'type': 'philosophical_approach',
                    'approach': approach,
                    'resonance': resonance
                })
            
            # Use best approach results
            if best_approach:
                comprehension['best_approach'] = best_approach
                comprehension['comprehended_energy'] = best_energy
                
                # Generate insights from best approach
                insights = self._generate_insights_from_approach(
                    best_approach, statement, best_energy)
                
                comprehension['philosophical_insights'].extend(insights)
            else:
                # Fallback if no approach worked well
                comprehension['comprehended_energy'] = statement_energy
        
        # Generate meta-understanding
        comprehension['meta_understanding'] = self._generate_meta_understanding(
            statement, statement_types, comprehension['philosophical_insights'])
        
        # Update comprehension efficacy
        self._update_comprehension_efficacy(statement_types, 
                                          len(comprehension['philosophical_insights']) > 0)
        
        # Learn from this comprehension experience
        self._learn_from_comprehension(statement_energy, comprehension)
        
        return comprehension
    
    def generate_philosophical_response(self, comprehension: Dict) -> Dict:
        """
        Generate a philosophical response based on comprehension results
        
        Args:
            comprehension: Results from comprehend_statement
            
        Returns:
            Response guidance with philosophical framing
        """
        # Extract key elements from comprehension
        statement_types = comprehension.get('statement_types', [])
        insights = comprehension.get('philosophical_insights', [])
        
        # Build philosophical response guidance
        response = {
            'philosophical_framing': self._select_philosophical_framing(statement_types),
            'key_insights': self._prioritize_insights(insights),
            'recommended_primitives': self._suggest_response_primitives(comprehension),
            'energy_guidance': self._generate_response_energy(comprehension)
        }
        
        return response
    
    def _classify_statement(self, statement: str, statement_energy: Dict) -> List[str]:
        """Classify statement into philosophical types"""
        detected_types = []
        
        # Check each statement type
        for type_name, detector in self.statement_types.items():
            if detector(statement, statement_energy):
                detected_types.append(type_name)
        
        # Ensure at least one type
        if not detected_types:
            # Default to factual and conceptual
            detected_types = ['factual', 'conceptual']
        
        return detected_types
    
    def _find_matching_pattern(self, statement_energy: Dict, 
                              statement_types: List[str]) -> Dict:
        """Find a matching comprehension pattern"""
        if not self.comprehension_patterns:
            return None
        
        best_match = None
        best_score = 0
        
        for pattern in self.comprehension_patterns:
            # Check type match
            type_match = len(set(pattern['statement_types']) & set(statement_types)) / \
                         max(len(pattern['statement_types']), len(statement_types))
            
            # Check energy signature match
            energy_match = self.reasoning._calculate_target_resonance(
                statement_energy, pattern['example_energy'])
            
            # Combined match score (weighted)
            match_score = type_match * 0.4 + energy_match * 0.6
            
            # Adjust by proven effectiveness
            match_score *= (0.5 + 0.5 * pattern.get('efficacy', 0.5))
            
            if match_score > best_score and match_score > 0.7:  # Good match threshold
                best_score = match_score
                best_match = pattern
        
        return best_match
    
    def _get_relevant_approaches(self, statement_types: List[str]) -> List[str]:
        """Get relevant philosophical approaches for these statement types"""
        approaches = set()
        
        # Collect approaches for each detected type
        for type_name in statement_types:
            if type_name in self.approach_mapping:
                approaches.update(self.approach_mapping[type_name])
        
        # Convert to list and sort by comprehension efficacy
        approach_list = list(approaches)
        approach_list.sort(key=lambda a: self.comprehension_efficacy.get(a, 0.5), reverse=True)
        
        return approach_list
    
    def _calculate_comprehension_resonance(self, original_energy: Dict, 
                                         comprehended_energy: Dict,
                                         statement_types: List[str]) -> float:
        """Calculate how well an energy signature represents comprehension"""
        # Resonance is measured by:
        # 1. Increased clarity (lower entropy)
        # 2. Appropriate abstraction level for the statement type
        # 3. Vector alignment with statement type expectations
        
        # Start with base resonance
        resonance = 0.5
        
        # Check entropy change (lower is better for comprehension)
        if 'entropy' in original_energy and 'entropy' in comprehended_energy:
            orig_entropy = original_energy['entropy'].get('value', 0.5)
            comp_entropy = comprehended_energy['entropy'].get('value', 0.5)
            
            # Reward entropy reduction
            if comp_entropy < orig_entropy:
                resonance += 0.2 * (orig_entropy - comp_entropy)
        
        # Check appropriate abstraction level (y-component of vector)
        if 'vector' in comprehended_energy:
            vector = comprehended_energy['vector'].get('value', [0.5, 0.5, 0.5])
            if len(vector) > 1:
                abstraction = vector[1]  # y-component
                
                # Conceptual statements should be more abstract
                if 'conceptual' in statement_types and abstraction > 0.6:
                    resonance += 0.1
                
                # Factual statements should be more concrete
                if 'factual' in statement_types and abstraction < 0.4:
                    resonance += 0.1
                
                # Existential statements should have high abstraction
                if 'existential' in statement_types and abstraction > 0.8:
                    resonance += 0.1
        
        # Check magnitude (strength of comprehension)
        if 'magnitude' in comprehended_energy:
            magnitude = comprehended_energy['magnitude'].get('value', 0.5)
            
            # Higher magnitude indicates stronger comprehension
            resonance += 0.1 * magnitude
        
        return min(1.0, resonance)
    
    def _generate_insights_from_pattern(self, pattern: Dict, statement: str,
                                      comprehended_energy: Dict) -> List[Dict]:
        """Generate philosophical insights from a pattern match"""
        insights = []
        
        # Base insight from pattern
        insights.append({
            'type': 'pattern_insight',
            'pattern': pattern['pattern_name'],
            'description': f"This statement fits the '{pattern['pattern_name']}' pattern",
            'confidence': 0.8
        })
        
        # Add pattern-specific insights
        for insight in pattern.get('insights', []):
            insights.append({
                'type': 'derived_insight',
                'description': insight,
                'confidence': 0.7
            })
        
        return insights
    
    def _generate_insights_from_approach(self, approach: str, statement: str,
                                       comprehended_energy: Dict) -> List[Dict]:
        """Generate philosophical insights using a philosophical approach"""
        insights = []
        
        # Generate insights based on approach
        if approach == 'dialectical':
            # Look for thesis-antithesis patterns
            insights.append({
                'type': 'dialectical_insight',
                'description': "This statement contains opposing elements that can be synthesized",
                'confidence': 0.6
            })
            
        elif approach == 'analytical':
            # Look for logical structure
            insights.append({
                'type': 'analytical_insight',
                'description': "This statement can be broken down into logical components",
                'confidence': 0.7
            })
            
        elif approach == 'phenomenological':
            # Focus on experiential qualities
            insights.append({
                'type': 'phenomenological_insight',
                'description': "This statement describes an experiential phenomenon",
                'confidence': 0.6
            })
            
        elif approach == 'existential':
            # Focus on meaning and purpose
            insights.append({
                'type': 'existential_insight',
                'description': "This statement relates to questions of meaning or purpose",
                'confidence': 0.6
            })
            
        elif approach == 'hermeneutic':
            # Focus on interpretation and context
            insights.append({
                'type': 'hermeneutic_insight',
                'description': "This statement requires contextual interpretation",
                'confidence': 0.7
            })
        
        # Energy signature specific insights
        if 'vector' in comprehended_energy:
            vector = comprehended_energy['vector'].get('value', [0.5, 0.5, 0.5])
            
            if len(vector) > 1 and vector[1] > 0.8:
                insights.append({
                    'type': 'abstraction_insight',
                    'description': "This statement operates at a high level of abstraction",
                    'confidence': 0.7
                })
        
        if 'entropy' in comprehended_energy:
            entropy = comprehended_energy['entropy'].get('value', 0.5)
            
            if entropy > 0.8:
                insights.append({
                    'type': 'complexity_insight',
                    'description': "This statement contains significant complexity or ambiguity",
                    'confidence': 0.7
                })
            elif entropy < 0.3:
                insights.append({
                    'type': 'clarity_insight',
                    'description': "This statement has a high degree of clarity and precision",
                    'confidence': 0.7
                })
        
        return insights
    
    def _generate_meta_understanding(self, statement: str, statement_types: List[str],
                                   insights: List[Dict]) -> Dict:
        """Generate meta-level understanding of the statement"""
        meta = {
            'philosophical_domain': self._determine_philosophical_domain(statement_types),
            'complexity_level': self._assess_complexity(insights),
            'primary_dimension': self._identify_primary_dimension(statement_types, insights)
        }
        
        return meta
    
    def _determine_philosophical_domain(self, statement_types: List[str]) -> str:
        """Determine the philosophical domain of a statement"""
        domain_mapping = {
            'factual': 'epistemology',
            'normative': 'ethics',
            'conceptual': 'metaphysics',
            'existential': 'existentialism',
            'analytical': 'logic',
            'paradoxical': 'logic',
            'metaphorical': 'aesthetics',
            'synthetic': 'epistemology'
        }
        
        # Count domains
        domain_counts = {}
        for type_name in statement_types:
            if type_name in domain_mapping:
                domain = domain_mapping[type_name]
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        # Find most common domain
        if domain_counts:
            return max(domain_counts.items(), key=lambda x: x[1])[0]
        else:
            return 'general philosophy'
    
    def _assess_complexity(self, insights: List[Dict]) -> str:
        """Assess the complexity level based on insights"""
        # Check for complexity indicators in insights
        has_complexity = any('complexity' in insight.get('type', '')
                            or 'complex' in insight.get('description', '').lower()
                            for insight in insights)
        
        has_paradox = any('paradox' in insight.get('description', '').lower()
                          for insight in insights)
        
        has_multiple_layers = len(insights) >= 3
        
        # Determine complexity level
        if has_paradox or (has_complexity and has_multiple_layers):
            return 'high'
        elif has_complexity or has_multiple_layers:
            return 'moderate'
        else:
            return 'low'
    
    def _identify_primary_dimension(self, statement_types: List[str],
                                  insights: List[Dict]) -> str:
        """Identify the primary philosophical dimension"""
        # Dimensions ordered from concrete to abstract
        dimensions = [
            'empirical',    # Based on observation
            'logical',      # Based on reasoning
            'ethical',      # Based on values
            'aesthetic',    # Based on beauty/art
            'existential',  # Based on existence/meaning
            'metaphysical'  # Based on fundamental reality
        ]
        
        # Check statement types
        dimension_scores = {dim: 0 for dim in dimensions}
        
        if 'factual' in statement_types:
            dimension_scores['empirical'] += 2
            
        if 'analytical' in statement_types:
            dimension_scores['logical'] += 2
            
        if 'normative' in statement_types:
            dimension_scores['ethical'] += 2
            
        if 'metaphorical' in statement_types:
            dimension_scores['aesthetic'] += 2
            
        if 'existential' in statement_types:
            dimension_scores['existential'] += 2
            
        if 'conceptual' in statement_types:
            dimension_scores['metaphysical'] += 1
            dimension_scores['logical'] += 1
            
        if 'paradoxical' in statement_types:
            dimension_scores['logical'] += 1
            dimension_scores['metaphysical'] += 1
            
        if 'synthetic' in statement_types:
            dimension_scores['empirical'] += 1
            dimension_scores['logical'] += 1
        
        # Check insights for additional clues
        for insight in insights:
            desc = insight.get('description', '').lower()
            
            if 'experience' in desc or 'observation' in desc:
                dimension_scores['empirical'] += 1
                
            if 'logic' in desc or 'reason' in desc:
                dimension_scores['logical'] += 1
                
            if 'value' in desc or 'ethical' in desc or 'moral' in desc:
                dimension_scores['ethical'] += 1
                
            if 'beauty' in desc or 'aesthetic' in desc or 'art' in desc:
                dimension_scores['aesthetic'] += 1
                
            if 'meaning' in desc or 'purpose' in desc or 'existence' in desc:
                dimension_scores['existential'] += 1
                
            if 'reality' in desc or 'being' in desc or 'nature' in desc:
                dimension_scores['metaphysical'] += 1
        
        # Return dimension with highest score
        return max(dimension_scores.items(), key=lambda x: x[1])[0]
    
    def _update_comprehension_efficacy(self, statement_types: List[str], 
                                     success: bool) -> None:
        """Update efficacy of philosophical approaches for statement types"""
        # Get approaches for these statement types
        approaches = set()
        for type_name in statement_types:
            if type_name in self.approach_mapping:
                approaches.update(self.approach_mapping[type_name])
        
        # Update efficacy
        for approach in approaches:
            current = self.comprehension_efficacy.get(approach, 0.5)
            
            if success:
                # Increase efficacy for successful comprehension
                new_efficacy = current + 0.02 * (1.0 - current)  # Diminishing returns
            else:
                # Decrease efficacy for unsuccessful comprehension
                new_efficacy = current - 0.01
            
            # Ensure within bounds
            new_efficacy = max(0.1, min(0.95, new_efficacy))
            self.comprehension_efficacy[approach] = new_efficacy
    
    def _learn_from_comprehension(self, statement_energy: Dict, 
                                comprehension: Dict) -> None:
        """Learn from this comprehension experience"""
        # If this was a successful comprehension, consider creating a new pattern
        if len(comprehension.get('philosophical_insights', [])) >= 2:
            # Extract pathway actions
            actions = []
            for pathway in comprehension.get('comprehension_pathways', []):
                if pathway.get('type') == 'philosophical_approach':
                    approach = pathway.get('approach')
                    if approach in self.primitives.philosophical_patterns:
                        actions = self.primitives.philosophical_patterns[approach]
                        break
                elif pathway.get('type') == 'pattern_application' and 'actions' in pathway:
                    actions = pathway['actions']
                    break
            
            if actions:
                # Check if this is a novel pattern
                is_novel = True
                for pattern in self.comprehension_patterns:
                    # Check similarity of statement types
                    type_overlap = set(pattern['statement_types']) & set(comprehension['statement_types'])
                    type_similarity = len(type_overlap) / max(len(pattern['statement_types']), 
                                                            len(comprehension['statement_types']))
                    
                    # Check similarity of action sequence
                    action_match = (pattern['actions'] == actions)
                    
                    if type_similarity > 0.7 and action_match:
                        is_novel = False
                        
                        # Update existing pattern efficacy
                        pattern['efficacy'] = pattern.get('efficacy', 0.5) + 0.05
                        pattern['efficacy'] = min(0.95, pattern['efficacy'])
                        
                        # Add insights if novel
                        existing_insights = {insight.get('description') for insight in pattern.get('insights', [])}
                        for insight in comprehension.get('philosophical_insights', []):
                            if insight.get('description') not in existing_insights:
                                pattern.setdefault('insights', []).append(insight.get('description'))
                        
                        break
                
                if is_novel:
                    # Create new pattern
                    pattern_name = f"Pattern_{len(self.comprehension_patterns) + 1}"
                    
                    # Extract insights
                    insights = [insight.get('description') for insight in 
                               comprehension.get('philosophical_insights', [])]
                    
                    new_pattern = {
                        'pattern_name': pattern_name,
                        'statement_types': comprehension['statement_types'],
                        'actions': actions,
                        'example_energy': statement_energy,
                        'insights': insights,
                        'efficacy': 0.6  # Start with moderate efficacy
                    }
                    
                    self.comprehension_patterns.append(new_pattern)
                    
                    # Limit total patterns
                    if len(self.comprehension_patterns) > 50:
                        # Remove least effective pattern
                        self.comprehension_patterns.sort(key=lambda p: p.get('efficacy', 0))
                        self.comprehension_patterns.pop(0)
    
    def _select_philosophical_framing(self, statement_types: List[str]) -> Dict:
        """Select philosophical framing for response"""
        framing = {
            'perspective': None,
            'approach': None,
            'dimension': None
        }
        
        # Map statement types to philosophical perspectives
        perspective_mapping = {
            'factual': ['empiricist', 'positivist'],
            'normative': ['ethicist', 'pragmatist'],
            'conceptual': ['rationalist', 'idealist'],
            'paradoxical': ['dialectician', 'mystic'],
            'metaphorical': ['hermeneuticist', 'phenomenologist'],
            'existential': ['existentialist', 'humanist'],
            'analytical': ['analytic', 'logician'],
            'synthetic': ['constructivist', 'pragmatist']
        }
        
        # Collect perspectives for detected types
        perspectives = []
        for type_name in statement_types:
            if type_name in perspective_mapping:
                perspectives.extend(perspective_mapping[type_name])
        
        # Select primary perspective (if any)
        if perspectives:
            framing['perspective'] = np.random.choice(perspectives)
        
        # Select approach based on statement types
        approaches = set()
        for type_name in statement_types:
            if type_name in self.approach_mapping:
                approaches.update(self.approach_mapping[type_name])
        
        if approaches:
            # Choose approach with highest efficacy
            approach_list = list(approaches)
            approach_list.sort(key=lambda a: self.comprehension_efficacy.get(a, 0.5), reverse=True)
            framing['approach'] = approach_list[0]
        
        # Determine primary dimension
        dimension_mapping = {
            'factual': 'empirical',
            'normative': 'ethical',
            'conceptual': 'logical',
            'paradoxical': 'metaphysical',
            'metaphorical': 'aesthetic',
            'existential': 'existential',
            'analytical': 'logical',
            'synthetic': 'empirical'
        }
        
        dimensions = [dimension_mapping.get(t) for t in statement_types if t in dimension_mapping]
        if dimensions:
            # Count dimensions
            dimension_counts = {}
            for dim in dimensions:
                dimension_counts[dim] = dimension_counts.get(dim, 0) + 1
            
            # Find most common dimension
            framing['dimension'] = max(dimension_counts.items(), key=lambda x: x[1])[0]
        
        return framing
    
    def _prioritize_insights(self, insights: List[Dict]) -> List[Dict]:
        """Prioritize philosophical insights for response"""
        if not insights:
            return []
        
        # Copy insights to avoid modifying original
        prioritized = insights.copy()
        
        # Sort by confidence
        prioritized.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Return top 3 insights
        return prioritized[:3]
    
    def _suggest_response_primitives(self, comprehension: Dict) -> List[str]:
        """Suggest primitive actions for response generation"""
        statement_types = comprehension.get('statement_types', [])
        
        # Default primitives
        primitives = ['shift_up', 'shift_right']
        
        # Adjust based on statement types
        if 'factual' in statement_types:
            primitives = ['shift_down', 'shift_right']  # More concrete, forward-moving
            
        elif 'conceptual' in statement_types:
            primitives = ['shift_up', 'expand']  # More abstract, expansive
            
        elif 'paradoxical' in statement_types:
            primitives = ['bifurcate', 'merge']  # Split then merge
            
        elif 'existential' in statement_types:
            primitives = ['shift_up', 'invert', 'expand']  # Abstract, negation, expansive
            
        elif 'normative' in statement_types:
            primitives = ['shift_up', 'resonate']  # Abstract, resonant
        
        # If we used a philosophical approach, suggest its primitives
        if 'best_approach' in comprehension:
            approach = comprehension['best_approach']
            if approach in self.primitives.philosophical_patterns:
                primitives = self.primitives.philosophical_patterns[approach]
        
        return primitives
    
    def _generate_response_energy(self, comprehension: Dict) -> Dict:
        """Generate energy guidance for response"""
        # Start with comprehended energy
        if 'comprehended_energy' in comprehension:
            response_energy = dict(comprehension['comprehended_energy'])
        else:
            response_energy = {}
        
        # Adjust based on meta-understanding
        meta = comprehension.get('meta_understanding', {})
        
        # Adjust complexity
        if 'complexity_level' in meta:
            if meta['complexity_level'] == 'high':
                # Reduce entropy for complex statements (clarify)
                if 'entropy' in response_energy:
                    response_energy['entropy']['value'] = max(0.2, 
                        response_energy['entropy'].get('value', 0.5) - 0.2)
                    
            elif meta['complexity_level'] == 'low':
                # Maintain simplicity
                if 'entropy' in response_energy:
                    response_energy['entropy']['value'] = max(0.2, 
                        response_energy['entropy'].get('value', 0.5) - 0.1)
        
        # Adjust abstraction based on primary dimension
        if 'primary_dimension' in meta and 'vector' in response_energy:
            vector = response_energy['vector'].get('value', [0.5, 0.5, 0.5])
            
            if len(vector) > 1:
                # Map dimensions to abstraction levels
                dimension_abstraction = {
                    'empirical': 0.3,      # More concrete
                    'logical': 0.5,        # Balanced
                    'ethical': 0.6,        # Slightly abstract
                    'aesthetic': 0.7,      # More abstract
                    'existential': 0.8,    # Highly abstract
                    'metaphysical': 0.9    # Very abstract
                }
                
                # Set y-component based on dimension
                target_abstraction = dimension_abstraction.get(meta['primary_dimension'], 0.5)
                vector[1] = 0.7 * vector[1] + 0.3 * target_abstraction
                
                response_energy['vector']['value'] = vector
        
        # Increase magnitude (confidence)
        if 'magnitude' in response_energy:
            response_energy['magnitude']['value'] = min(0.9, 
                response_energy['magnitude'].get('value', 0.5) + 0.1)
        
        return response_energy
    
    # Statement type detection methods
    
    def _detect_factual(self, statement: str, energy: Dict) -> bool:
        """Detect factual statements"""
        # Factual statements often have certain linguistic markers
        factual_markers = [
            ' is ', ' are ', ' was ', ' were ', ' has ', ' have ',
            ' exists', ' occurred', ' happened', ' contains',
            ' consists', ' comprises', ' equals'
        ]
        
        # Check for factual markers
        has_markers = any(marker in f" {statement} " for marker in factual_markers)
        
        # Check energy signature (factual statements tend to have higher magnitude, lower entropy)
        has_factual_energy = False
        if 'magnitude' in energy and 'entropy' in energy:
            magnitude = energy['magnitude'].get('value', 0.5)
            entropy = energy['entropy'].get('value', 0.5)
            
            has_factual_energy = magnitude > 0.6 and entropy < 0.4
        
        return has_markers or has_factual_energy
    
    def _detect_normative(self, statement: str, energy: Dict) -> bool:
        """Detect normative (should/ought) statements"""
        # Normative statements have specific markers
        normative_markers = [
            ' should ', ' ought ', ' must ', ' right ', ' wrong ',
            ' good ', ' bad ', ' better ', ' worse ', ' best ', ' worst ',
            ' moral', ' ethical', ' value', ' duty', ' obligation',
            ' responsibility', ' virtue', ' vice'
        ]
        
        # Check for normative markers
        return any(marker in f" {statement} " for marker in normative_markers)
    
    def _detect_conceptual(self, statement: str, energy: Dict) -> bool:
        """Detect conceptual statements"""
        # Conceptual statements often define or analyze concepts
        conceptual_markers = [
            ' means ', ' defined as ', ' refers to ', ' concept of ',
            ' idea of ', ' theory of ', ' notion of ', ' understanding of ',
            ' interpretation of ', ' definition of '
        ]
        
        # Check for conceptual markers
        has_markers = any(marker in f" {statement} " for marker in conceptual_markers)
        
        # Check energy signature (conceptual statements tend to have higher y-component in vector)
        has_conceptual_energy = False
        if 'vector' in energy:
            vector = energy['vector'].get('value', [0.5, 0.5, 0.5])
            if len(vector) > 1:
                has_conceptual_energy = vector[1] > 0.7  # High y-component (abstraction)
        
        return has_markers or has_conceptual_energy
    
    def _detect_paradoxical(self, statement: str, energy: Dict) -> bool:
        """Detect paradoxical statements"""
        # Paradoxical statements often contain contradiction markers
        paradox_markers = [
            ' paradox', ' contradiction', ' yet ', ' however ', ' but ',
            ' on the one hand', ' on the other hand', ' both ', ' neither ',
            ' and not ', ' while also ', ' simultaneously '
        ]
        
        # Check for paradox markers
        has_markers = any(marker in f" {statement} " for marker in paradox_markers)
        
        # Check energy signature (paradoxical statements tend to have high entropy)
        has_paradoxical_energy = False
        if 'entropy' in energy:
            entropy = energy['entropy'].get('value', 0.5)
            has_paradoxical_energy = entropy > 0.8
        
        return has_markers or has_paradoxical_energy
    
    def _detect_metaphorical(self, statement: str, energy: Dict) -> bool:
        """Detect metaphorical statements"""
        # Metaphorical statements often have specific markers
        metaphor_markers = [
            ' like ', ' as if ', ' as though ', ' metaphor', ' symbol',
            ' represents ', ' symbolizes ', ' mirrors ', ' reflects ',
            ' is a ', ' are a ', ' was a ', ' were a '
        ]
        
        # Check for metaphor markers
        return any(marker in f" {statement} " for marker in metaphor_markers)
    
    def _detect_existential(self, statement: str, energy: Dict) -> bool:
        """Detect existential statements"""
        # Existential statements concern meaning, purpose, existence
        existential_markers = [
            ' meaning ', ' purpose ', ' existence ', ' being ', ' life ',
            ' death ', ' freedom ', ' choice ', ' responsibility ',
            ' authenticity ', ' absurd', ' despair', ' anxiety ',
            ' what it means ', ' why we ', ' reason for '
        ]
        
        # Check for existential markers
        return any(marker in f" {statement} " for marker in existential_markers)
    
    def _detect_analytical(self, statement: str, energy: Dict) -> bool:
        """Detect analytical statements"""
        # Analytical statements concern logical analysis, definition
        analytical_markers = [
            ' necessarily ', ' by definition ', ' tautology ', ' logical ',
            ' entails ', ' implies ', ' if and only if ', ' equivalent to ',
            ' identical to ', ' same as ', ' defined as '
        ]
        
        # Check for analytical markers
        has_markers = any(marker in f" {statement} " for marker in analytical_markers)
        
        # Check energy signature (analytical statements tend to have low entropy, high frequency)
        has_analytical_energy = False
        if 'entropy' in energy and 'frequency' in energy:
            entropy = energy['entropy'].get('value', 0.5)
            frequency = energy['frequency'].get('value', 0.5)
            
            has_analytical_energy = entropy < 0.3 and frequency > 0.7
        
        return has_markers or has_analytical_energy
    
    def _detect_synthetic(self, statement: str, energy: Dict) -> bool:
        """Detect synthetic statements (empirical, a posteriori)"""
        # Synthetic statements depend on empirical evidence, experience
        synthetic_markers = [
            ' observed ', ' measured ', ' discovered ', ' found ',
            ' experiment ', ' evidence ', ' data ', ' empirical ',
            ' experience ', ' perception ', ' observation '
        ]
        
        # Check for synthetic markers
        return any(marker in f" {statement} " for marker in synthetic_markers)