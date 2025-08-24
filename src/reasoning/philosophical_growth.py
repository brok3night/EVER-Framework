"""
Philosophical Growth System - Enables EVER to develop reasoning abilities and form new connections
"""
import numpy as np
from typing import Dict, List, Set, Tuple
import datetime

class PhilosophicalGrowth:
    def __init__(self, philosophical_engine):
        self.philosophical = philosophical_engine
        
        # Track reasoning history
        self.reasoning_history = []
        
        # Track concept connections (network grows over time)
        self.concept_network = {}
        
        # Novel connection patterns discovered through experience
        self.discovered_patterns = []
        
        # Philosophical inclinations (evolve with experience)
        self.inclinations = {
            'abstraction_tendency': 0.5,  # Tendency toward abstract thinking
            'analytical_depth': 0.4,      # Depth of analytical processing
            'synthesis_drive': 0.6,       # Drive to synthesize disparate concepts
            'pattern_recognition': 0.5,   # Ability to recognize patterns
            'conceptual_curiosity': 0.7   # Curiosity about new concepts
        }
        
        # Reasoning efficacy tracking (improves with practice)
        self.reasoning_efficacy = {
            'dialectical': 0.4,
            'deductive': 0.5,
            'inductive': 0.4,
            'abductive': 0.3,
            'analogical': 0.4,
            'conceptual_blending': 0.5
        }
    
    def process_philosophical_experience(self, concepts: List[Dict], 
                                        reasoning_type: str, 
                                        result: Dict,
                                        feedback: Dict = None) -> Dict:
        """
        Process a philosophical reasoning experience and grow from it
        
        Args:
            concepts: The concepts involved in reasoning
            reasoning_type: Type of reasoning applied
            result: Result of the reasoning process
            feedback: Optional feedback on reasoning quality
        """
        # Record this experience
        timestamp = datetime.datetime.now().isoformat()
        experience = {
            'timestamp': timestamp,
            'reasoning_type': reasoning_type,
            'concept_count': len(concepts),
            'concept_names': [c.get('name', 'unnamed') for c in concepts],
            'result_name': result.get('name', 'unnamed_result'),
            'feedback': feedback
        }
        self.reasoning_history.append(experience)
        
        # Update concept network with new connections
        self._update_concept_network(concepts, result)
        
        # Look for novel patterns
        novel_pattern = self._identify_novel_pattern(concepts, result)
        if novel_pattern:
            self.discovered_patterns.append(novel_pattern)
        
        # Update reasoning efficacy based on experience and feedback
        self._update_reasoning_efficacy(reasoning_type, result, feedback)
        
        # Update philosophical inclinations based on experience
        self._update_inclinations(concepts, reasoning_type, result)
        
        # Generate growth insights
        growth_insights = self._generate_growth_insights()
        
        return {
            'experience_recorded': True,
            'reasoning_efficacy': self.reasoning_efficacy[reasoning_type],
            'network_size': len(self.concept_network),
            'novel_pattern_found': novel_pattern is not None,
            'growth_insights': growth_insights,
            'philosophical_maturity': self._calculate_philosophical_maturity()
        }
    
    def discover_new_connections(self, central_concept: Dict, depth: int = 2) -> List[Dict]:
        """
        Discover new connections between concepts through network exploration
        
        Args:
            central_concept: The concept to explore from
            depth: How many steps to explore
        """
        concept_name = central_concept.get('name')
        if not concept_name or concept_name not in self.concept_network:
            return []
        
        # Already discovered direct connections
        known_connections = set(self.concept_network[concept_name])
        
        # Find multi-step connections that might yield new insights
        multi_step_connections = self._find_multi_step_connections(concept_name, depth)
        
        # Discover unexpected connections through energy signature similarities
        unexpected_connections = self._find_unexpected_connections(central_concept)
        
        # Generate insights for each new connection
        connection_insights = []
        
        # Process multi-step connections
        for path in multi_step_connections:
            # Only consider paths that aren't direct known connections
            if path[-1] not in known_connections:
                # Generate insight for this connection
                insight = {
                    'type': 'multi_step',
                    'path': path,
                    'relation_type': self._infer_relation_type(central_concept, path[-1]),
                    'novelty_score': self._calculate_novelty(central_concept, path[-1])
                }
                connection_insights.append(insight)
        
        # Process unexpected connections
        for connection in unexpected_connections:
            if connection not in known_connections:
                insight = {
                    'type': 'unexpected_similarity',
                    'connection': connection,
                    'similarity_basis': self._explain_similarity(central_concept, connection),
                    'novelty_score': self._calculate_novelty(central_concept, connection)
                }
                connection_insights.append(insight)
        
        # Apply current philosophical inclinations to prioritize insights
        prioritized_insights = self._prioritize_by_inclinations(connection_insights)
        
        return prioritized_insights
    
    def generate_novel_perspective(self, concepts: List[Dict]) -> Dict:
        """
        Generate a novel philosophical perspective by applying learned patterns
        
        Args:
            concepts: Concepts to incorporate in the perspective
        """
        # Check for sufficient philosophical maturity
        maturity = self._calculate_philosophical_maturity()
        if maturity < 0.4:
            return {
                'success': False,
                'reason': 'Insufficient philosophical maturity',
                'maturity_level': maturity
            }
        
        # Select most appropriate reasoning approaches based on efficacy
        primary_approach = max(self.reasoning_efficacy.items(), key=lambda x: x[1])[0]
        
        # Apply primary reasoning approach
        primary_result = self.philosophical.apply_reasoning(primary_approach, concepts)
        
        # Apply secondary transformation using discovered patterns
        if self.discovered_patterns:
            # Choose most relevant pattern
            pattern = self._select_relevant_pattern(concepts, primary_result)
            
            # Apply pattern transformation
            transformed_result = self._apply_pattern_transformation(primary_result, pattern)
        else:
            transformed_result = primary_result
        
        # Infuse with current philosophical inclinations
        novel_perspective = self._infuse_with_inclinations(transformed_result)
        
        # Add derivation information
        novel_perspective['derivation'] = {
            'primary_approach': primary_approach,
            'pattern_applied': pattern['name'] if 'pattern' in locals() else None,
            'philosophical_maturity': maturity,
            'inclinations_applied': list(self.inclinations.keys())
        }
        
        return {
            'success': True,
            'perspective': novel_perspective,
            'novelty_score': self._assess_novelty(novel_perspective)
        }
    
    def _update_concept_network(self, concepts: List[Dict], result: Dict) -> None:
        """Update the concept network with new connections"""
        # Extract concept names
        concept_names = [c.get('name', f'concept_{i}') for i, c in enumerate(concepts)]
        result_name = result.get('name', 'result')
        
        # Add nodes if they don't exist
        for name in concept_names + [result_name]:
            if name not in self.concept_network:
                self.concept_network[name] = set()
        
        # Add connections between all concepts and the result
        for name in concept_names:
            # Connect to other input concepts
            for other_name in concept_names:
                if name != other_name:
                    self.concept_network[name].add(other_name)
            
            # Connect to result
            self.concept_network[name].add(result_name)
            self.concept_network[result_name].add(name)
    
    def _identify_novel_pattern(self, concepts: List[Dict], result: Dict) -> Dict:
        """Identify a novel pattern in the reasoning process"""
        # Analyze energy signature transformations
        input_signatures = [c.get('energy_signature', {}) for c in concepts]
        result_signature = result.get('energy_signature', {})
        
        # Skip if missing signatures
        if not input_signatures or not result_signature:
            return None
        
        # Look for consistent transformation patterns
        transformations = {}
        
        # For each energy property, analyze transformation
        for prop in result_signature:
            if prop not in result_signature or 'value' not in result_signature[prop]:
                continue
                
            result_value = result_signature[prop]['value']
            input_values = [sig.get(prop, {}).get('value') for sig in input_signatures if prop in sig]
            
            # Skip if missing input values
            if not input_values or any(v is None for v in input_values):
                continue
                
            # Detect transformation pattern
            if isinstance(result_value, (int, float)) and all(isinstance(v, (int, float)) for v in input_values):
                # Scalar transformation
                avg_input = sum(input_values) / len(input_values)
                
                # Calculate transformation factor
                if avg_input != 0:
                    factor = result_value / avg_input
                    
                    # Identify transformation type
                    if 0.95 < factor < 1.05:
                        transform_type = 'preservation'
                    elif factor > 1:
                        transform_type = 'amplification'
                    else:
                        transform_type = 'reduction'
                        
                    transformations[prop] = {
                        'type': transform_type,
                        'factor': factor
                    }
            
            elif isinstance(result_value, list) and all(isinstance(v, list) for v in input_values):
                # Vector transformation
                # Analyze each component
                if all(len(v) == len(result_value) for v in input_values):
                    component_transforms = []
                    
                    for i in range(len(result_value)):
                        input_components = [v[i] for v in input_values]
                        avg_component = sum(input_components) / len(input_components)
                        
                        if avg_component != 0:
                            factor = result_value[i] / avg_component
                            
                            if 0.95 < factor < 1.05:
                                transform_type = 'preservation'
                            elif factor > 1:
                                transform_type = 'amplification'
                            else:
                                transform_type = 'reduction'
                                
                            component_transforms.append({
                                'component': i,
                                'type': transform_type,
                                'factor': factor
                            })
                    
                    transformations[prop] = {
                        'type': 'vector_transformation',
                        'components': component_transforms
                    }
        
        # Check if this pattern is truly novel
        if transformations and not self._pattern_exists(transformations):
            # Create novel pattern
            pattern = {
                'name': f"Pattern_{len(self.discovered_patterns) + 1}",
                'transformations': transformations,
                'source_concepts': [c.get('name', 'unnamed') for c in concepts],
                'discovery_timestamp': datetime.datetime.now().isoformat()
            }
            
            return pattern
        
        return None
    
    def _pattern_exists(self, transformations: Dict) -> bool:
        """Check if a pattern already exists in discovered patterns"""
        for pattern in self.discovered_patterns:
            pattern_transforms = pattern.get('transformations', {})
            
            # Check if transformations match
            matches = True
            for prop, transform in transformations.items():
                if prop not in pattern_transforms:
                    matches = False
                    break
                    
                if transform.get('type') != pattern_transforms[prop].get('type'):
                    matches = False
                    break
                    
                if 'factor' in transform and 'factor' in pattern_transforms[prop]:
                    # Allow for some variation in factors
                    if abs(transform['factor'] - pattern_transforms[prop]['factor']) > 0.1:
                        matches = False
                        break
            
            if matches:
                return True
        
        return False
    
    def _update_reasoning_efficacy(self, reasoning_type: str, result: Dict, feedback: Dict) -> None:
        """Update reasoning efficacy based on experience and feedback"""
        # Start with small improvement from practice
        current_efficacy = self.reasoning_efficacy.get(reasoning_type, 0.5)
        practice_improvement = 0.01 * (1.0 - current_efficacy)  # Diminishing returns
        
        # Apply feedback if available
        feedback_adjustment = 0
        if feedback:
            if 'quality_rating' in feedback:
                # Scale from 0-10 to -0.05 to +0.05
                rating = feedback['quality_rating']
                feedback_adjustment = (rating - 5) / 100
        
        # Update efficacy
        new_efficacy = current_efficacy + practice_improvement + feedback_adjustment
        self.reasoning_efficacy[reasoning_type] = max(0.1, min(0.95, new_efficacy))
    
    def _update_inclinations(self, concepts: List[Dict], reasoning_type: str, result: Dict) -> None:
        """Update philosophical inclinations based on experience"""
        # Different reasoning types affect different inclinations
        if reasoning_type == 'dialectical' or reasoning_type == 'conceptual_blending':
            # These increase synthesis drive
            self.inclinations['synthesis_drive'] = min(0.95, self.inclinations['synthesis_drive'] + 0.01)
        
        if reasoning_type == 'deductive' or reasoning_type == 'inductive':
            # These increase analytical depth
            self.inclinations['analytical_depth'] = min(0.95, self.inclinations['analytical_depth'] + 0.01)
        
        if reasoning_type == 'analogical' or reasoning_type == 'abductive':
            # These increase pattern recognition
            self.inclinations['pattern_recognition'] = min(0.95, self.inclinations['pattern_recognition'] + 0.01)
        
        # Update abstraction tendency based on concept abstractions
        abstract_concepts = 0
        for concept in concepts:
            vector = concept.get('energy_signature', {}).get('vector', {}).get('value', [0.5, 0.5, 0.5])
            if isinstance(vector, list) and len(vector) > 1 and vector[1] > 0.7:
                # High y-component indicates abstraction
                abstract_concepts += 1
        
        if abstract_concepts > len(concepts) / 2:
            # Working with mostly abstract concepts increases abstraction tendency
            self.inclinations['abstraction_tendency'] = min(0.95, self.inclinations['abstraction_tendency'] + 0.01)
        else:
            # Working with concrete concepts decreases abstraction tendency
            self.inclinations['abstraction_tendency'] = max(0.05, self.inclinations['abstraction_tendency'] - 0.01)
        
        # Curiosity increases when discovering new connections
        if self._count_new_connections(concepts) > 0:
            self.inclinations['conceptual_curiosity'] = min(0.95, self.inclinations['conceptual_curiosity'] + 0.02)
    
    def _count_new_connections(self, concepts: List[Dict]) -> int:
        """Count new connections formed in this experience"""
        new_connections = 0
        
        concept_names = [c.get('name') for c in concepts if c.get('name')]
        for i, name1 in enumerate(concept_names):
            if name1 not in self.concept_network:
                new_connections += len(concept_names) - 1
                continue
                
            for name2 in concept_names[i+1:]:
                if name2 not in self.concept_network[name1]:
                    new_connections += 1
        
        return new_connections
    
    def _generate_growth_insights(self) -> List[str]:
        """Generate insights about philosophical growth"""
        insights = []
        
        # Check for recent improvements in reasoning efficacy
        if len(self.reasoning_history) >= 5:
            recent_types = [exp['reasoning_type'] for exp in self.reasoning_history[-5:]]
            most_common = max(set(recent_types), key=recent_types.count)
            
            if self.reasoning_efficacy[most_common] > 0.6:
                insights.append(f"Developing strength in {most_common} reasoning")
        
        # Check for network growth patterns
        if len(self.reasoning_history) >= 10:
            recent_concept_count = sum(exp['concept_count'] for exp in self.reasoning_history[-5:])
            older_concept_count = sum(exp['concept_count'] for exp in self.reasoning_history[-10:-5])
            
            if recent_concept_count > older_concept_count * 1.5:
                insights.append("Accelerating concept network expansion")
        
        # Check for inclination shifts
        high_inclinations = [k for k, v in self.inclinations.items() if v > 0.7]
        if high_inclinations:
            insights.append(f"Developing pronounced {high_inclinations[0]} tendency")
        
        # Check for novel pattern discovery rate
        if len(self.discovered_patterns) >= 3:
            insights.append("Building pattern recognition capabilities")
        
        return insights
    
    def _calculate_philosophical_maturity(self) -> float:
        """Calculate overall philosophical maturity level"""
        if not self.reasoning_history:
            return 0.1
        
        # Factors in maturity calculation
        factors = [
            # Experience breadth
            min(1.0, len(self.reasoning_history) / 100),
            
            # Reasoning diversity
            min(1.0, len(set(exp['reasoning_type'] for exp in self.reasoning_history)) / 6),
            
            # Average reasoning efficacy
            sum(self.reasoning_efficacy.values()) / len(self.reasoning_efficacy),
            
            # Network connectivity
            min(1.0, len(self.concept_network) / 100),
            
            # Pattern discovery
            min(1.0, len(self.discovered_patterns) / 10)
        ]
        
        # Weighted average
        weights = [0.3, 0.2, 0.2, 0.15, 0.15]
        maturity = sum(f * w for f, w in zip(factors, weights))
        
        return maturity
    
    def _find_multi_step_connections(self, start_concept: str, depth: int) -> List[List[str]]:
        """Find multi-step connections through the concept network"""
        if start_concept not in self.concept_network:
            return []
        
        # Use breadth-first search to find paths
        paths = []
        visited = {start_concept}
        queue = [(start_concept, [start_concept])]
        
        while queue:
            node, path = queue.pop(0)
            
            if len(path) > depth + 1:
                continue
                
            if node in self.concept_network:
                for neighbor in self.concept_network[node]:
                    if neighbor not in path:
                        new_path = path + [neighbor]
                        
                        if len(new_path) > 2:
                            paths.append(new_path)
                        
                        if len(new_path) <= depth + 1:
                            queue.append((neighbor, new_path))
        
        return paths
    
    def _find_unexpected_connections(self, concept: Dict) -> List[str]:
        """Find unexpected connections through energy signature similarities"""
        concept_sig = concept.get('energy_signature', {})
        if not concept_sig:
            return []
        
        # This would access the full concept database to find energy-similar concepts
        # In a real implementation, this would use the energy system to find similar concepts
        # For this example, we'll just return a placeholder
        return ["unexpected_connection_1", "unexpected_connection_2"]
    
    def _infer_relation_type(self, concept1: Dict, concept2: str) -> str:
        """Infer the type of relationship between concepts"""
        # In a real implementation, this would analyze the concepts to determine relationship type
        # For this example, we'll return a placeholder
        return "conceptual_association"
    
    def _calculate_novelty(self, concept1: Dict, concept2: str) -> float:
        """Calculate novelty score for a connection"""
        # In a real implementation, this would use sophisticated metrics
        # For this example, we'll return a placeholder
        return 0.8
    
    def _explain_similarity(self, concept1: Dict, concept2: str) -> str:
        """Explain why two concepts are similar"""
        # In a real implementation, this would analyze energy signatures
        # For this example, we'll return a placeholder
        return "Energy resonance in magnitude and vector orientation"
    
    def _prioritize_by_inclinations(self, insights: List[Dict]) -> List[Dict]:
        """Prioritize insights based on current philosophical inclinations"""
        # Score each insight based on alignment with inclinations
        scored_insights = []
        
        for insight in insights:
            score = 0
            
            # Different insight types align with different inclinations
            if insight['type'] == 'multi_step':
                score += self.inclinations['analytical_depth'] * 0.5
                score += self.inclinations['synthesis_drive'] * 0.3
            elif insight['type'] == 'unexpected_similarity':
                score += self.inclinations['pattern_recognition'] * 0.5
                score += self.inclinations['conceptual_curiosity'] * 0.3
            
            # Add novelty contribution
            score += insight['novelty_score'] * 0.2
            
            scored_insights.append((insight, score))
        
        # Sort by score
        scored_insights.sort(key=lambda x: x[1], reverse=True)
        
        # Return insights in priority order
        return [insight for insight, _ in scored_insights]
    
    def _select_relevant_pattern(self, concepts: List[Dict], result: Dict) -> Dict:
        """Select most relevant discovered pattern for these concepts"""
        # In a real implementation, this would match patterns to the concepts
        # For this example, we'll return a placeholder
        if self.discovered_patterns:
            return self.discovered_patterns[0]
        else:
            return {
                'name': 'default_pattern',
                'transformations': {}
            }
    
    def _apply_pattern_transformation(self, concept: Dict, pattern: Dict) -> Dict:
        """Apply a discovered pattern transformation to a concept"""
        # Copy the concept to avoid modifying the original
        transformed = concept.copy()
        transformed['energy_signature'] = concept.get('energy_signature', {}).copy()
        
        # Apply transformations from the pattern
        for prop, transform in pattern.get('transformations', {}).items():
            if prop in transformed['energy_signature'] and 'value' in transformed['energy_signature'][prop]:
                value = transformed['energy_signature'][prop]['value']
                
                if isinstance(value, (int, float)) and transform.get('type') in ['amplification', 'reduction', 'preservation']:
                    # Apply scalar transformation
                    factor = transform.get('factor', 1.0)
                    transformed['energy_signature'][prop]['value'] = value * factor
                
                elif isinstance(value, list) and transform.get('type') == 'vector_transformation':
                    # Apply vector transformation
                    components = transform.get('components', [])
                    
                    for component in components:
                        idx = component.get('component')
                        if idx < len(value):
                            factor = component.get('factor', 1.0)
                            value[idx] = value[idx] * factor
        
        # Add pattern application to derivation
        if 'derivation' not in transformed:
            transformed['derivation'] = {}
        
        transformed['derivation']['pattern_applied'] = pattern['name']
        
        return transformed
    
    def _infuse_with_inclinations(self, concept: Dict) -> Dict:
        """Infuse a concept with current philosophical inclinations"""
        # Copy the concept to avoid modifying the original
        infused = concept.copy()
        infused['energy_signature'] = concept.get('energy_signature', {}).copy()
        
        # Apply each inclination
        energy_sig = infused['energy_signature']
        
        # Abstraction tendency affects vector y-component
        if 'vector' in energy_sig and isinstance(energy_sig['vector'].get('value'), list):
            vector = energy_sig['vector']['value']
            if len(vector) > 1:
                # Shift y-component toward inclination
                tendency = self.inclinations['abstraction_tendency']
                vector[1] = vector[1] * 0.7 + tendency * 0.3
        
        # Analytical depth affects entropy
        if 'entropy' in energy_sig and isinstance(energy_sig['entropy'].get('value'), (int, float)):
            # Higher analytical depth reduces entropy
            depth = self.inclinations['analytical_depth']
            energy_sig['entropy']['value'] = energy_sig['entropy']['value'] * 0.7 + (1 - depth) * 0.3
        
        # Synthesis drive affects boundary width
        if 'boundary' in energy_sig and isinstance(energy_sig['boundary'].get('value'), list):
            boundary = energy_sig['boundary']['value']
            if len(boundary) >= 2:
                # Higher synthesis drive expands boundaries
                drive = self.inclinations['synthesis_drive']
                center = (boundary[0] + boundary[1]) / 2
                width = boundary[1] - boundary[0]
                new_width = width * 0.7 + drive * 0.6  # Max width expansion of 60%
                boundary[0] = max(0, center - new_width/2)
                boundary[1] = min(1, center + new_width/2)
        
        # Add inclination infusion to derivation
        if 'derivation' not in infused:
            infused['derivation'] = {}
        
        infused['derivation']['inclinations_infused'] = {k: v for k, v in self.inclinations.items()}
        
        return infused
    
    def _assess_novelty(self, perspective: Dict) -> float:
        """Assess the novelty of a generated perspective"""
        # In a real implementation, this would compare to previous perspectives
        # For this example, we'll use a placeholder calculation
        
        # Higher maturity enables more novelty
        base_novelty = self._calculate_philosophical_maturity() * 0.4
        
        # More discovered patterns enables more novelty
        pattern_factor = min(1.0, len(self.discovered_patterns) / 20) * 0.3
        
        # Network size contributes to novelty potential
        network_factor = min(1.0, len(self.concept_network) / 200) * 0.3
        
        # Combined novelty score
        return base_novelty + pattern_factor + network_factor