"""
Topographical Reasoning System - Uses resonance to navigate conceptual landscapes
"""
from typing import Dict, List, Set, Tuple
import numpy as np
from src.reasoning.primitive_actions import PrimitiveActions

class TopographicalReasoning:
    """Implements reasoning as navigation through conceptual landscapes guided by resonance"""
    
    def __init__(self, energy_system):
        self.energy = energy_system
        self.primitives = PrimitiveActions(energy_system)
        
        # Topographical maps (discovered through experience)
        self.concept_maps = {}
        
        # Navigation history
        self.navigation_history = []
        
        # Successful pathways
        self.successful_pathways = []
    
    def navigate(self, starting_energy: Dict, target_energy: Dict = None,
                reasoning_intent: str = None, context_energies: List[Dict] = None) -> Dict:
        """
        Navigate from starting energy toward a target or guided by intent
        
        Args:
            starting_energy: Energy signature to start from
            target_energy: Target energy signature (optional)
            reasoning_intent: Type of reasoning intended (optional)
            context_energies: Other energy signatures for context
        """
        # Initialize navigation
        current_energy = dict(starting_energy)
        path = []
        
        # Maximum steps to prevent infinite loops
        max_steps = 10
        
        # Track resonance improvement
        initial_resonance = self._calculate_target_resonance(current_energy, target_energy)
        best_resonance = initial_resonance
        best_energy = dict(current_energy)
        
        # Begin navigation
        for step in range(max_steps):
            # Suggest next actions based on current energy and context
            actions = self.primitives.suggest_actions(
                current_energy, context_energies, reasoning_intent)
            
            if not actions:
                break
            
            # Apply best action
            next_energy = self.primitives.apply_action(actions[0], current_energy)
            
            # Record step
            path.append({
                'action': actions[0],
                'from_energy': current_energy,
                'to_energy': next_energy
            })
            
            # Update current energy
            current_energy = next_energy
            
            # Check if we've improved resonance with target
            if target_energy:
                resonance = self._calculate_target_resonance(current_energy, target_energy)
                
                # If improved, update best
                if resonance > best_resonance:
                    best_resonance = resonance
                    best_energy = dict(current_energy)
                
                # If we've reached high resonance, we can stop
                if resonance > 0.9:
                    break
        
        # Record navigation in history
        self.navigation_history.append({
            'starting_energy': starting_energy,
            'target_energy': target_energy,
            'intent': reasoning_intent,
            'path': path,
            'final_energy': current_energy,
            'resonance_improvement': best_resonance - initial_resonance
        })
        
        # If navigation was successful, record the pathway
        if best_resonance > initial_resonance + 0.2:  # Significant improvement
            self._record_successful_pathway(path, starting_energy, best_energy)
        
        # Return best energy achieved
        return best_energy if best_resonance > initial_resonance else current_energy
    
    def find_reasoning_path(self, concept_energy: Dict, philosophical_approach: str,
                           context_energies: List[Dict] = None) -> Dict:
        """
        Find a reasoning path using a specific philosophical approach
        
        Args:
            concept_energy: Energy signature to reason about
            philosophical_approach: Philosophical approach to use
            context_energies: Other energy signatures for context
        """
        # Get actions for this philosophical approach
        approach_actions = self.primitives.get_philosophical_actions(philosophical_approach)
        
        if not approach_actions:
            # Unknown approach, use general navigation
            return self.navigate(concept_energy, reasoning_intent="general", 
                               context_energies=context_energies)
        
        # Apply action sequence
        result = self.primitives.apply_sequence(
            approach_actions, concept_energy, context_energies)
        
        # Record in navigation history
        self.navigation_history.append({
            'starting_energy': concept_energy,
            'intent': philosophical_approach,
            'path': [{'action': action} for action in approach_actions],
            'final_energy': result
        })
        
        return result
    
    def explore_conceptual_space(self, seed_energy: Dict, 
                                exploration_paths: int = 3,
                                steps_per_path: int = 5) -> List[Dict]:
        """
        Explore conceptual space around a seed energy
        
        Args:
            seed_energy: Energy signature to start exploration from
            exploration_paths: Number of different paths to explore
            steps_per_path: Steps to take along each path
        """
        discovered_energies = []
        
        # Try different starting actions
        actions = self.primitives.suggest_actions(seed_energy)
        actions = actions[:exploration_paths]  # Limit to requested paths
        
        for action in actions:
            current = self.primitives.apply_action(action, seed_energy)
            path = [action]
            
            # Take additional steps
            for _ in range(steps_per_path - 1):
                next_actions = self.primitives.suggest_actions(current)
                if next_actions:
                    # Filter out actions already in path to avoid loops
                    new_actions = [a for a in next_actions if a not in path]
                    if new_actions:
                        next_action = new_actions[0]
                        current = self.primitives.apply_action(next_action, current)
                        path.append(next_action)
            
            # Add discovered energy
            discovered_energies.append({
                'energy': current,
                'path': path,
                'distance': len(path)
            })
        
        return discovered_energies
    
    def find_optimal_pathway(self, starting_energy: Dict, 
                            target_energy: Dict) -> List[str]:
        """
        Find the optimal pathway between energy signatures
        
        Args:
            starting_energy: Starting energy signature
            target_energy: Target energy signature
        """
        # Check for previously successful pathways
        pathway = self._find_matching_pathway(starting_energy, target_energy)
        
        if pathway:
            # Use known successful pathway
            return pathway
        
        # Perform new navigation
        self.navigate(starting_energy, target_energy)
        
        # Check if we found a good pathway
        pathway = self._find_matching_pathway(starting_energy, target_energy)
        
        if pathway:
            return pathway
        
        # If no good pathway found, use philosophical approaches
        approaches = list(self.primitives.philosophical_patterns.keys())
        best_approach = None
        best_resonance = 0
        
        for approach in approaches:
            # Try this approach
            result = self.find_reasoning_path(starting_energy, approach)
            
            # Check resonance with target
            resonance = self._calculate_target_resonance(result, target_energy)
            
            if resonance > best_resonance:
                best_resonance = resonance
                best_approach = approach
        
        if best_approach:
            return self.primitives.get_philosophical_actions(best_approach)
        
        # Fallback to basic moves
        return ['shift_up', 'shift_right', 'merge']
    
    def _calculate_target_resonance(self, current: Dict, target: Dict) -> float:
        """Calculate resonance between current and target energy signatures"""
        if not target:
            return 0.5  # No target, neutral resonance
        
        # Calculate similarity for common properties
        similarity_sum = 0.0
        weight_sum = 0.0
        
        for prop in ['magnitude', 'frequency', 'entropy', 'vector']:
            if prop in current and prop in target:
                val_current = current.get(prop, {}).get('value')
                val_target = target.get(prop, {}).get('value')
                
                if val_current is not None and val_target is not None:
                    # Property weight
                    weight = 1.0 if prop != 'vector' else 1.5
                    
                    if isinstance(val_current, (int, float)) and isinstance(val_target, (int, float)):
                        # Scalar similarity
                        similarity = 1.0 - min(1.0, abs(val_current - val_target))
                        similarity_sum += similarity * weight
                        weight_sum += weight
                    elif isinstance(val_current, list) and isinstance(val_target, list):
                        # Vector similarity
                        min_len = min(len(val_current), len(val_target))
                        if min_len > 0:
                            vector_similarity = 1.0 - sum(abs(val_current[i] - val_target[i]) 
                                                        for i in range(min_len)) / min_len
                            similarity_sum += vector_similarity * weight
                            weight_sum += weight
        
        # Return weighted average
        return similarity_sum / weight_sum if weight_sum > 0 else 0.0
    
    def _record_successful_pathway(self, path: List[Dict], 
                                 start_energy: Dict, end_energy: Dict) -> None:
        """Record a successful navigation pathway"""
        # Extract just the actions
        actions = [step['action'] for step in path]
        
        # Skip if empty
        if not actions:
            return
        
        # Record this pathway
        self.successful_pathways.append({
            'actions': actions,
            'start_signature': start_energy,
            'end_signature': end_energy,
            'usage_count': 1
        })
        
        # Limit total pathways
        if len(self.successful_pathways) > 100:
            # Remove least used pathway
            self.successful_pathways.sort(key=lambda x: x.get('usage_count', 0))
            self.successful_pathways.pop(0)
    
    def _find_matching_pathway(self, start_energy: Dict, target_energy: Dict) -> List[str]:
        """Find a matching pathway for this navigation task"""
        if not self.successful_pathways:
            return None
        
        best_pathway = None
        best_match = 0
        
        for pathway in self.successful_pathways:
            # Calculate similarity to start and end energies
            start_sim = self._calculate_target_resonance(
                start_energy, pathway['start_signature'])
            end_sim = self._calculate_target_resonance(
                target_energy, pathway['end_signature'])
            
            # Combined match score
            match_score = start_sim * 0.6 + end_sim * 0.4
            
            if match_score > best_match and match_score > 0.7:  # Good match threshold
                best_match = match_score
                best_pathway = pathway['actions']
                
                # Update usage count
                pathway['usage_count'] += 1
        
        return best_pathway