"""
Insight Evaluation - Evaluates the quality of philosophical insights
"""
from typing import Dict, List, Any, Tuple
import numpy as np

class InsightEvaluation:
    """Evaluates philosophical insights using multiple criteria"""
    
    def __init__(self, resonance_network, experiential_grounding):
        self.network = resonance_network
        self.grounding = experiential_grounding
        
        # Evaluation criteria
        self.evaluation_criteria = {
            'coherence': self._evaluate_coherence,
            'novelty': self._evaluate_novelty,
            'utility': self._evaluate_utility,
            'parsimony': self._evaluate_parsimony,
            'empirical_grounding': self._evaluate_empirical_grounding,
            'explanatory_power': self._evaluate_explanatory_power
        }
        
        # Weights for different criteria
        self.criteria_weights = {
            'coherence': 0.2,
            'novelty': 0.15,
            'utility': 0.2,
            'parsimony': 0.1,
            'empirical_grounding': 0.2,
            'explanatory_power': 0.15
        }
        
        # History of evaluated insights
        self.evaluated_insights = []
        
        # Learning feedback
        self.feedback_history = []
    
    def evaluate_insight(self, insight: Dict, context: Dict = None) -> Dict:
        """
        Evaluate a philosophical insight
        
        Args:
            insight: Insight to evaluate
            context: Optional context information
            
        Returns:
            Evaluation results
        """
        # Apply each evaluation criterion
        criterion_scores = {}
        
        for criterion, eval_func in self.evaluation_criteria.items():
            score = eval_func(insight, context)
            criterion_scores[criterion] = score
        
        # Calculate weighted overall score
        overall_score = 0.0
        for criterion, score in criterion_scores.items():
            weight = self.criteria_weights.get(criterion, 0.1)
            overall_score += score * weight
        
        # Create evaluation result
        evaluation = {
            'insight': insight,
            'criterion_scores': criterion_scores,
            'overall_score': overall_score
        }
        
        # Store evaluation
        self.evaluated_insights.append(evaluation)
        
        return evaluation
    
    def evaluate_insight_set(self, insights: List[Dict], 
                           context: Dict = None) -> Dict:
        """
        Evaluate a set of related insights
        
        Args:
            insights: List of insights to evaluate
            context: Optional context information
            
        Returns:
            Evaluation results
        """
        # Evaluate individual insights
        individual_evaluations = []
        
        for insight in insights:
            evaluation = self.evaluate_insight(insight, context)
            individual_evaluations.append(evaluation)
        
        # Calculate consistency across insights
        consistency = self._calculate_insight_consistency(individual_evaluations)
        
        # Calculate synergy between insights
        synergy = self._calculate_insight_synergy(individual_evaluations)
        
        # Calculate overall set score
        individual_scores = [eval_result['overall_score'] for eval_result in individual_evaluations]
        average_score = sum(individual_scores) / len(individual_scores) if individual_scores else 0.0
        
        # Adjust by consistency and synergy
        set_score = average_score * (1.0 + 0.3 * consistency + 0.2 * synergy)
        
        # Create set evaluation result
        set_evaluation = {
            'individual_evaluations': individual_evaluations,
            'consistency': consistency,
            'synergy': synergy,
            'set_score': set_score
        }
        
        return set_evaluation
    
    def record_feedback(self, insight_id: str, feedback_score: float,
                      feedback_type: str, comments: str = None) -> None:
        """
        Record external feedback on an insight
        
        Args:
            insight_id: Identifier for the insight
            feedback_score: Feedback score (0-1)
            feedback_type: Type of feedback
            comments: Optional comments
        """
        feedback = {
            'insight_id': insight_id,
            'feedback_score': feedback_score,
            'feedback_type': feedback_type,
            'comments': comments,
            'timestamp': np.datetime64('now')
        }
        
        self.feedback_history.append(feedback)
        
        # Update weights based on feedback
        self._update_weights_from_feedback(feedback)
    
    def get_top_insights(self, n: int = 10) -> List[Dict]:
        """
        Get top N insights by evaluation score
        
        Args:
            n: Number of insights to return
            
        Returns:
            List of top insights with evaluations
        """
        # Sort by overall score
        sorted_insights = sorted(
            self.evaluated_insights,
            key=lambda x: x['overall_score'],
            reverse=True
        )
        
        # Return top N
        return sorted_insights[:n]
    
    def _evaluate_coherence(self, insight: Dict, context: Dict = None) -> float:
        """Evaluate coherence of an insight"""
        # Check if insight has source and target concepts
        if ('source_concepts' not in insight or not insight['source_concepts'] or
            'target_concepts' not in insight or not insight['target_concepts']):
            return 0.5  # Default score for insights without clear concepts
        
        # Calculate coherence based on concept connections
        source_concepts = insight['source_concepts']
        target_concepts = insight['target_concepts']
        
        # Get network connections between these concepts
        connection_strengths = []
        
        for source in source_concepts:
            for target in target_concepts:
                if source in self.network.connections and target in self.network.connections[source]:
                    connection = self.network.connections[source][target]
                    connection_strengths.append(connection['strength'])
        
        # Calculate average connection strength
        if connection_strengths:
            return sum(connection_strengths) / len(connection_strengths)
        
        # If no direct connections, check for paths
        path_strengths = []
        
        for source in source_concepts:
            for target in target_concepts:
                path = self.network.find_resonance_path(source, target)
                if path:
                    # Calculate path strength
                    strength = 1.0
                    for _, conn_strength in path[1:]:  # Skip first node
                        strength *= conn_strength
                    
                    path_strengths.append(strength)
        
        if path_strengths:
            return sum(path_strengths) / len(path_strengths)
        
        return 0.3  # Low coherence if no connections found
    
    def _evaluate_novelty(self, insight: Dict, context: Dict = None) -> float:
        """Evaluate novelty of an insight"""
        # Check if this insight type has been seen before
        if not self.evaluated_insights:
            return 1.0  # First insight is maximally novel
        
        # Calculate similarity to previous insights
        similarities = []
        
        for prev_eval in self.evaluated_insights:
            prev_insight = prev_eval['insight']
            similarity = self._calculate_insight_similarity(insight, prev_insight)
            similarities.append(similarity)
        
        if not similarities:
            return 1.0
        
        # Novelty is inverse of maximum similarity
        max_similarity = max(similarities)
        return 1.0 - max_similarity
    
    def _evaluate_utility(self, insight: Dict, context: Dict = None) -> float:
        """Evaluate utility of an insight"""
        # Utility depends on potential applications
        utility = 0.5  # Default moderate utility
        
        # Check insight type
        if 'type' in insight:
            insight_type = insight['type']
            
            # Some types are inherently more useful
            if insight_type in ['causal_relationships', 'practical_consequences']:
                utility += 0.2  # More useful
            elif insight_type in ['conceptual_contrast', 'compositional_structure']:
                utility += 0.1  # Somewhat useful
        
        # Check for empirical grounding (more grounded = more useful)
        if 'source_concepts' in insight:
            grounding_strengths = []
            
            for concept_id in insight['source_concepts']:
                strength = self.grounding.calculate_grounding_strength(concept_id)
                grounding_strengths.append(strength)
            
            if grounding_strengths:
                avg_grounding = sum(grounding_strengths) / len(grounding_strengths)
                utility += 0.3 * avg_grounding
        
        return min(1.0, utility)
    
    def _evaluate_parsimony(self, insight: Dict, context: Dict = None) -> float:
        """Evaluate parsimony of an insight"""
        # Parsimony depends on simplicity and explicitness
        parsimony = 0.7  # Default good parsimony
        
        # Adjust based on complexity factors
        complexity_factors = 0
        
        # More concepts = more complex
        if 'source_concepts' in insight:
            complexity_factors += max(0, len(insight['source_concepts']) - 2)
        
        if 'target_concepts' in insight:
            complexity_factors += max(0, len(insight['target_concepts']) - 2)
        
        # Reduce parsimony for each complexity factor
        parsimony -= 0.1 * complexity_factors
        
        return max(0.1, parsimony)
    
    def _evaluate_empirical_grounding(self, insight: Dict, context: Dict = None) -> float:
        """Evaluate empirical grounding of an insight"""
        # Get empirical grounding information
        grounded_insight = self.grounding.ground_insight(insight)
        
        if 'grounding' in grounded_insight:
            return grounded_insight['grounding'].get('grounding_strength', 0.0)
        
        return 0.0  # No grounding
    
    def _evaluate_explanatory_power(self, insight: Dict, context: Dict = None) -> float:
        """Evaluate explanatory power of an insight"""
        # Explanatory power depends on:
        # 1. How many concepts it connects
        # 2. How disparate those concepts are
        
        explanatory_power = 0.5  # Default moderate power
        
        # More concepts = more explanatory
        num_concepts = 0
        
        if 'source_concepts' in insight:
            num_concepts += len(insight['source_concepts'])
        
        if 'target_concepts' in insight:
            num_concepts += len(insight['target_concepts'])
        
        # Adjust based on number of concepts
        if num_concepts <= 2:
            explanatory_power -= 0.1  # Less explanatory
        elif num_concepts >= 5:
            explanatory_power += 0.1  # More explanatory
        
        # Check conceptual distance
        if ('source_concepts' in insight and insight['source_concepts'] and
            'target_concepts' in insight and insight['target_concepts']):
            
            # Sample a source and target
            source = insight['source_concepts'][0]
            target = insight['target_concepts'][0]
            
            # Find path length
            path = self.network.find_resonance_path(source, target)
            
            if path:
                # Longer paths indicate more explanatory insights
                path_length = len(path)
                
                if path_length >= 4:
                    explanatory_power += 0.2  # Connects distant concepts
                elif path_length <= 2:
                    explanatory_power -= 0.1  # Connects close concepts
        
        return min(1.0, max(0.1, explanatory_power))
    
    def _calculate_insight_similarity(self, insight1: Dict, insight2: Dict) -> float:
        """Calculate similarity between two insights"""
        similarity = 0.0
        
        # Check if same type
        if 'type' in insight1 and 'type' in insight2:
            if insight1['type'] == insight2['type']:
                similarity += 0.3
        
        # Check for common concepts
        common_concepts = 0
        total_concepts = 0
        
        if 'source_concepts' in insight1 and 'source_concepts' in insight2:
            common_source = set(insight1['source_concepts']) & set(insight2['source_concepts'])
            common_concepts += len(common_source)
            total_concepts += len(set(insight1['source_concepts']) | set(insight2['source_concepts']))
        
        if 'target_concepts' in insight1 and 'target_concepts' in insight2:
            common_target = set(insight1['target_concepts']) & set(insight2['target_concepts'])
            common_concepts += len(common_target)
            total_concepts += len(set(insight1['target_concepts']) | set(insight2['target_concepts']))
        
        # Calculate Jaccard similarity of concepts
        if total_concepts > 0:
            concept_similarity = common_concepts / total_concepts
            similarity += 0.7 * concept_similarity
        
        return similarity
    
    def _calculate_insight_consistency(self, evaluations: List[Dict]) -> float:
        """Calculate consistency across a set of insights"""
        if not evaluations or len(evaluations) < 2:
            return 1.0  # Single insight is perfectly consistent with itself
        
        # Extract criterion scores
        criterion_variances = {}
        
        for criterion in self.evaluation_criteria.keys():
            scores = [eval_result['criterion_scores'].get(criterion, 0.0) 
                     for eval_result in evaluations]
            
            # Calculate variance
            variance = np.var(scores)
            criterion_variances[criterion] = variance
        
        # Calculate average variance
        avg_variance = sum(criterion_variances.values()) / len(criterion_variances)
        
        # Convert to consistency (inverse of variance)
        consistency = 1.0 - min(1.0, avg_variance * 5)
        
        return consistency
    
    def _calculate_insight_synergy(self, evaluations: List[Dict]) -> float:
        """Calculate synergy between insights"""
        if not evaluations or len(evaluations) < 2:
            return 0.0  # No synergy with a single insight
        
        # Extract insights
        insights = [eval_result['insight'] for eval_result in evaluations]
        
        # Check for complementary insight types
        types = [insight.get('type') for insight in insights if 'type' in insight]
        
        # Define complementary pairs
        complementary_pairs = [
            ('similarity', 'contrast'),
            ('abstraction_path', 'concretization_examples'),
            ('causal_relationships', 'practical_consequences'),
            ('compositional_structure', 'conceptual_contrast')
        ]
        
        # Count complementary pairs
        complementary_count = 0
        for type1, type2 in complementary_pairs:
            if type1 in types and type2 in types:
                complementary_count += 1
        
        # Calculate synergy score
        if complementary_count > 0:
            return min(1.0, 0.3 * complementary_count)
        
        return 0.0
    
    def _update_weights_from_feedback(self, feedback: Dict) -> None:
        """Update criterion weights based on feedback"""
        # Find the evaluated insight
        insight_id = feedback['insight_id']
        
        for eval_result in self.evaluated_insights:
            if 'id' in eval_result['insight'] and eval_result['insight']['id'] == insight_id:
                # Found matching insight
                
                # Calculate discrepancy between our evaluation and feedback
                discrepancy = feedback['feedback_score'] - eval_result['overall_score']
                
                if abs(discrepancy) < 0.1:
                    # Evaluation was close to feedback, no need to adjust
                    return
                
                # Find criterion with largest difference from feedback
                largest_diff = 0.0
                adjust_criterion = None
                
                for criterion, score in eval_result['criterion_scores'].items():
                    diff = abs(feedback['feedback_score'] - score)
                    if diff > largest_diff:
                        largest_diff = diff
                        adjust_criterion = criterion
                
                if adjust_criterion:
                    # Adjust weight for this criterion
                    adjustment = 0.05 * np.sign(discrepancy)
                    
                    # If feedback is higher than our score, and this criterion was high,
                    # increase its weight
                    if (discrepancy > 0 and 
                        eval_result['criterion_scores'][adjust_criterion] > eval_result['overall_score']):
                        self.criteria_weights[adjust_criterion] += adjustment
                    
                    # If feedback is lower than our score, and this criterion was high,
                    # decrease its weight
                    elif (discrepancy < 0 and 
                          eval_result['criterion_scores'][adjust_criterion] > eval_result['overall_score']):
                        self.criteria_weights[adjust_criterion] -= adjustment
                    
                    # Opposite adjustments for low criterion scores
                    elif (discrepancy > 0 and 
                          eval_result['criterion_scores'][adjust_criterion] < eval_result['overall_score']):
                        self.criteria_weights[adjust_criterion] -= adjustment
                    elif (discrepancy < 0 and 
                          eval_result['criterion_scores'][adjust_criterion] < eval_result['overall_score']):
                        self.criteria_weights[adjust_criterion] += adjustment
                    
                    # Normalize weights
                    total = sum(self.criteria_weights.values())
                    for criterion in self.criteria_weights:
                        self.criteria_weights[criterion] /= total
                
                break