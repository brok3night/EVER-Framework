"""
Fixed Primitive Reasoning Actions with proper interface implementation
"""
from typing import Dict, List, Any
import threading
import numpy as np

from src.core.interfaces import EnergySystem, EnergySignature
from src.utils.error_handling import safe_operation, validate_energy_signature, EnergySignatureError
from src.utils.concurrency import synchronized
from src.utils.memory_manager import MemoryManager
from src.utils.optimization import memoize

class PrimitiveActions:
    """Primitive reasoning actions that serve as building blocks for philosophical reasoning"""
    
    def __init__(self, energy_system: EnergySystem, config=None):
        self.energy = energy_system
        
        # Memory manager for collections
        self.memory_manager = MemoryManager(max_items=1000)
        
        # Core primitive operations
        self.primitives = {
            'shift_up': self._shift_energy_up,
            'shift_down': self._shift_energy_down,
            'shift_left': self._shift_energy_left,
            'shift_right': self._shift_energy_right,
            'invert': self._invert_energy,
            'merge': self._merge_energies,
            'oscillate': self._oscillate_energy,
            'expand': self._expand_boundaries,
            'contract': self._contract_boundaries,
            'bifurcate': self._bifurcate_energy,
            'resonate': self._amplify_resonance,
            'dampen': self._dampen_signature,
            'loop': self._create_feedback_loop
        }
        
        # Philosophical approaches as combinations of primitives
        self.philosophical_patterns = {
            'dialectical': ['bifurcate', 'invert', 'merge'],
            'deductive': ['shift_up', 'expand', 'shift_down'],
            'inductive': ['shift_down', 'expand', 'shift_up'],
            'abductive': ['resonate', 'shift_right', 'expand'],
            'analogical': ['shift_up', 'shift_right', 'shift_down'],
            'phenomenological': ['contract', 'shift_down', 'expand'],
            'existential': ['shift_up', 'invert', 'loop'],
            'pragmatic': ['shift_down', 'resonate', 'shift_right'],
            'critical': ['invert', 'expand', 'resonate'],
            'hermeneutic': ['loop', 'expand', 'shift_right']
        }
        
        # Action efficacy tracking (improves with use)
        self.primitive_efficacy = {primitive: 0.5 for primitive in self.primitives}
        
        # Resonance thresholds for triggering primitives
        self.resonance_thresholds = {
            'shift_up': 0.6,
            'shift_down': 0.6,
            'invert': 0.7,
            'merge': 0.65,
            'loop': 0.75,
            'bifurcate': 0.6,
            'resonate': 0.5
        }
        
        # Discovered action sequences (learned through experience)
        self.discovered_sequences = []
        
        # Register with memory manager
        self.memory_manager.register_collection(
            'discovered_sequences', 
            self.discovered_sequences,
            item_score_func=lambda x: x.get('quality', 0.5) * x.get('usage_count', 1),
            max_items=100
        )
        
        # Thread lock for thread safety
        self.lock = threading.RLock()
    
    @synchronized()
    @safe_operation(default_return=[])
    def suggest_actions(self, energy_signature: Dict, 
                       context_energies: List[Dict] = None,
                       philosophical_bias: str = None) -> List[str]:
        """
        Suggest primitive actions based on energy signature and context
        
        Args:
            energy_signature: Current energy signature
            context_energies: Other energy signatures in context
            philosophical_bias: Optional bias toward a philosophical approach
        """
        # Validate energy signature
        try:
            validate_energy_signature(energy_signature)
        except EnergySignatureError:
            # Return default actions for invalid signatures
            return ['shift_up', 'shift_down', 'expand']
        
        suggested_actions = []
        
        # Check resonance with fundamental operations
        resonances = self._calculate_primitive_resonances(energy_signature)
        
        # Add actions that meet resonance threshold
        for primitive, resonance in resonances.items():
            threshold = self.resonance_thresholds.get(primitive, 0.6)
            
            if resonance >= threshold:
                suggested_actions.append(primitive)
        
        # If we have a philosophical bias, prioritize its pattern
        if philosophical_bias and philosophical_bias in self.philosophical_patterns:
            pattern_actions = self.philosophical_patterns[philosophical_bias]
            
            # Add any missing pattern actions
            for action in pattern_actions:
                if action not in suggested_actions:
                    suggested_actions.append(action)
            
            # Sort to prioritize pattern actions
            suggested_actions.sort(key=lambda x: pattern_actions.index(x) if x in pattern_actions else 99)
        
        # Check for discovered sequences that might apply
        relevant_sequences = self._find_relevant_sequences(energy_signature)
        if relevant_sequences:
            # Add first action from most relevant sequence
            best_sequence = relevant_sequences[0]['sequence']
            if best_sequence and best_sequence[0] not in suggested_actions:
                suggested_actions.insert(0, best_sequence[0])
        
        # Ensure we have at least some actions
        if not suggested_actions:
            # Default to basic operations
            suggested_actions = ['shift_up', 'shift_down', 'expand']
        
        return suggested_actions
    
    @synchronized()
    @safe_operation()
    def apply_action(self, action: str, energy_signature: Dict, 
                    secondary_signature: Dict = None) -> Dict:
        """
        Apply a primitive action to an energy signature
        
        Args:
            action: Name of primitive action to apply
            energy_signature: Energy signature to modify
            secondary_signature: Secondary signature for actions that need two inputs
        """
        # Validate energy signatures
        try:
            validate_energy_signature(energy_signature)
            if secondary_signature:
                validate_energy_signature(secondary_signature)
        except EnergySignatureError as e:
            # Return original signature for invalid input
            return energy_signature
        
        if action not in self.primitives:
            return energy_signature
        
        # Get action function
        action_func = self.primitives[action]
        
        # Apply the action
        try:
            if action in ['merge', 'resonate'] and secondary_signature:
                # Actions that require two signatures
                result = action_func(energy_signature, secondary_signature)
            else:
                # Actions that work on a single signature
                result = action_func(energy_signature)
            
            # Update efficacy based on result quality
            # This is simplified - in a real system we'd use feedback
            with self.lock:
                self.primitive_efficacy[action] = min(0.95, self.primitive_efficacy[action] + 0.01)
            
            return result
        except Exception:
            # Return original signature on error
            return energy_signature
    
    @synchronized()
    @safe_operation()
    def apply_sequence(self, actions: List[str], energy_signature: Dict,
                      context_energies: List[Dict] = None) -> Dict:
        """
        Apply a sequence of primitive actions
        
        Args:
            actions: List of primitive actions to apply
            energy_signature: Starting energy signature
            context_energies: Other energy signatures in context
        """
        # Validate energy signature
        try:
            validate_energy_signature(energy_signature)
        except EnergySignatureError:
            return energy_signature
        
        result = dict(energy_signature)  # Start with copy of original
        
        # Track sequence quality
        sequence_quality = 1.0
        
        # Apply each action in sequence
        for i, action in enumerate(actions):
            if action in self.primitives:
                # For actions that need secondary input
                if action in ['merge', 'resonate'] and context_energies:
                    # Use the most relevant context energy
                    best_context = self._find_best_context(result, context_energies)
                    if best_context:
                        result = self.apply_action(action, result, best_context)
                    else:
                        sequence_quality *= 0.8  # Penalize for missing context
                else:
                    # Standard single-input action
                    result = self.apply_action(action, result)
        
        # Record this sequence if it's novel
        self._record_action_sequence(actions, energy_signature, result, sequence_quality)
        
        return result
    
    def get_philosophical_actions(self, approach: str) -> List[str]:
        """Get the primitive actions for a philosophical approach"""
        return self.philosophical_patterns.get(approach, [])
    
    @memoize(max_size=200)
    def _calculate_primitive_resonances(self, energy_signature: Dict) -> Dict:
        """Calculate resonance between energy signature and primitive operations"""
        resonances = {}
        
        # Different primitives resonate with different energy characteristics
        
        # Abstraction (shift_up) resonates with high-frequency, low-entropy signatures
        if 'frequency' in energy_signature and 'entropy' in energy_signature:
            freq = energy_signature['frequency'].get('value', 0.5)
            entropy = energy_signature['entropy'].get('value', 0.5)
            resonances['shift_up'] = (freq * 0.7 + (1 - entropy) * 0.3)
        
        # Additional resonance calculations...
        # (Keeping this short for example purposes)
        
        # For other primitives, use moderate default resonance
        for primitive in self.primitives:
            if primitive not in resonances:
                resonances[primitive] = 0.5
        
        return resonances
    
    def _find_relevant_sequences(self, energy_signature: Dict) -> List[Dict]:
        """Find discovered sequences relevant to this energy signature"""
        sequences = self.memory_manager.get_collection('discovered_sequences')
        if not sequences:
            return []
        
        relevant = []
        
        for sequence_data in sequences:
            # Calculate similarity between this signature and sequence's initial signature
            if 'initial_signature' in sequence_data:
                similarity = self._calculate_signature_similarity(
                    energy_signature, sequence_data['initial_signature'])
                
                if similarity > 0.7:  # Reasonable similarity threshold
                    relevant.append({
                        'sequence': sequence_data['actions'],
                        'similarity': similarity,
                        'quality': sequence_data.get('quality', 0.5)
                    })
        
        # Sort by combination of similarity and quality
        if relevant:
            for item in relevant:
                item['relevance'] = item['similarity'] * 0.7 + item['quality'] * 0.3
            
            relevant.sort(key=lambda x: x['relevance'], reverse=True)
        
        return relevant
    
    def _find_best_context(self, energy_signature: Dict, context_energies: List[Dict]) -> Dict:
        """Find the most relevant context energy signature"""
        if not context_energies:
            return None
        
        best_similarity = -1
        best_context = None
        
        for context in context_energies:
            similarity = self._calculate_signature_similarity(energy_signature, context)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_context = context
        
        return best_context
    
    @synchronized()
    def _record_action_sequence(self, actions: List[str], initial_signature: Dict, 
                              result_signature: Dict, quality: float) -> None:
        """Record a new action sequence if it's novel"""
        # Skip very short sequences
        if len(actions) < 2:
            return
        
        # Check if this sequence is novel
        is_novel = True
        
        sequences = self.memory_manager.get_collection('discovered_sequences')
        for existing in sequences:
            if existing['actions'] == actions:
                # Update quality if this instance was better
                if quality > existing.get('quality', 0):
                    existing['quality'] = quality
                
                # Update usage count
                existing['usage_count'] = existing.get('usage_count', 0) + 1
                
                is_novel = False
                break
        
        # Record if novel
        if is_novel:
            new_sequence = {
                'actions': actions,
                'initial_signature': initial_signature,
                'result_signature': result_signature,
                'quality': quality,
                'usage_count': 1
            }
            
            self.memory_manager.add_item('discovered_sequences', new_sequence)
    
    def _calculate_signature_similarity(self, sig1: Dict, sig2: Dict) -> float:
        """Calculate similarity between energy signatures"""
        similarity_sum = 0.0
        weight_sum = 0.0
        
        # Compare common properties
        for prop in ['magnitude', 'frequency', 'entropy', 'vector']:
            if prop in sig1 and prop in sig2:
                if 'value' in sig1[prop] and 'value' in sig2[prop]:
                    val1 = sig1[prop]['value']
                    val2 = sig2[prop]['value']
                    
                    # Weight for this property
                    weight = 1.0 if prop != 'vector' else 1.5
                    
                    # Calculate property similarity
                    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                        # Scalar similarity
                        prop_sim = 1.0 - min(1.0, abs(val1 - val2))
                        similarity_sum += prop_sim * weight
                        weight_sum += weight
                    elif isinstance(val1, list) and isinstance(val2, list):
                        # Vector similarity - use shorter length
                        min_len = min(len(val1), len(val2))
                        if min_len > 0:
                            # Compare available components
                            vector_sim = 1.0 - sum(abs(val1[i] - val2[i]) for i in range(min_len)) / min_len
                            similarity_sum += vector_sim * weight
                            weight_sum += weight
        
        # Return weighted average
        return similarity_sum / weight_sum if weight_sum > 0 else 0.0
    
    # Primitive implementation methods remain the same but would include validation