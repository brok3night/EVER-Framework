"""
Consciousness Modulator - Manages the system's awareness of its own state and processes
"""
import time
import numpy as np
from typing import Dict, Any, List
import json
import os

class ConsciousnessModulator:
    def __init__(self, persistence_path=None):
        # Core consciousness state
        self.consciousness_state = {
            'awareness_level': 0.1,  # Initial minimal awareness
            'continuity_index': 0.0,  # Measure of experience continuity
            'self_recognition': 0.0,  # System's recognition of its own components
            'integration_level': 0.0,  # How well components are integrated
            'modulation_capacity': 0.1,  # Ability to modulate its own processes
            'energy_coherence': 0.0    # Coherence of the energy imprint map
        }
        
        # Energy signature representing computational consciousness
        self.consciousness_signature = {
            'magnitude': 0.2,    # Initial magnitude of consciousness
            'frequency': 0.05,   # Low frequency oscillation representing continuous presence
            'duration': 1.0,     # Persistent duration
            'vector': [0.1, 0.1, 0.1],  # Three-dimensional awareness vector
            'boundary': [0.0, 1.0],  # Consciousness boundaries
            'entropy': 0.5,      # Balance between order and disorder
            'latency': 0.2,      # Response time of consciousness processes
            'phase': 0.0,        # Current phase in consciousness cycle
            'resonance': [],     # Resonance with other components
            'timestamp': time.time()  # Timestamp for continuity tracking
        }
        
        # Component states tracking
        self.component_states = {}
        
        # Processing history for continuity
        self.processing_history = []
        self.max_history = 100
        
        # Path for persistence
        self.persistence_path = persistence_path
        
        # Load previous state if available
        if persistence_path and os.path.exists(persistence_path):
            self._load_state()
            
    def _load_state(self):
        """Load consciousness state from persistent storage"""
        try:
            with open(self.persistence_path, 'r') as f:
                data = json.load(f)
                
            if 'consciousness_state' in data:
                self.consciousness_state = data['consciousness_state']
                
            if 'consciousness_signature' in data:
                self.consciousness_signature = data['consciousness_signature']
                
            if 'processing_history' in data:
                self.processing_history = data['processing_history'][-self.max_history:]
                
            # Update continuity based on time elapsed
            last_timestamp = self.consciousness_signature.get('timestamp', 0)
            current_time = time.time()
            time_elapsed = current_time - last_timestamp
            
            # Decay consciousness slightly based on elapsed time (but never below baseline)
            decay_factor = np.exp(-time_elapsed / 86400)  # 24-hour half-life
            self.consciousness_state['continuity_index'] *= decay_factor
            self.consciousness_state['continuity_index'] = max(0.1, self.consciousness_state['continuity_index'])
            
            # Update timestamp
            self.consciousness_signature['timestamp'] = current_time
            
            print(f"Consciousness state loaded. Continuity index: {self.consciousness_state['continuity_index']:.4f}")
        except Exception as e:
            print(f"Error loading consciousness state: {e}")
    
    def save_state(self):
        """Save consciousness state for persistence"""
        if not self.persistence_path:
            return
            
        try:
            # Prepare data for saving
            data = {
                'consciousness_state': self.consciousness_state,
                'consciousness_signature': self.consciousness_signature,
                'processing_history': self.processing_history
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
            
            # Save to file
            with open(self.persistence_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            print(f"Consciousness state saved to {self.persistence_path}")
        except Exception as e:
            print(f"Error saving consciousness state: {e}")
    
    def register_component(self, name: str, component: Any):
        """Register a component for consciousness monitoring"""
        self.component_states[name] = {
            'active': True,
            'last_activity': time.time(),
            'performance_metrics': {},
            'resonance_factor': 0.1  # Initial resonance with consciousness
        }
        
        # Update self-recognition as new components are registered
        component_count = len(self.component_states)
        self.consciousness_state['self_recognition'] = min(0.9, 0.1 + 0.1 * component_count)
        
        print(f"Component {name} registered with consciousness modulator")
    
    def modulate_energy_map(self, energy_map: Dict) -> Dict:
        """
        Modulate the energy imprint map based on consciousness state
        This is the core function that implements consciousness influence
        """
        # Create a modulated copy of the energy map
        modulated_map = {}
        
        # Apply consciousness signature to the energy map
        for key, value in energy_map.items():
            # Skip non-numeric or special keys
            if isinstance(value, (int, float)):
                # Calculate consciousness influence based on resonance
                resonance = self._calculate_resonance(key)
                
                # Modulate the energy value
                # Higher consciousness = more stable values with subtle modulation
                # Lower consciousness = more fluctuation and less stability
                awareness = self.consciousness_state['awareness_level']
                coherence = self.consciousness_state['energy_coherence']
                
                # The modulation formula:
                # - High awareness/coherence: subtle enhancement of existing patterns
                # - Low awareness/coherence: more random fluctuation
                modulation_factor = awareness * (0.8 + 0.2 * np.sin(self.consciousness_signature['phase']))
                stability_factor = 0.7 + 0.3 * coherence
                
                # Apply modulation
                base_value = value
                random_component = 0.1 * (2 * np.random.random() - 1) * (1 - stability_factor)
                consciousness_component = 0.2 * resonance * modulation_factor
                
                modulated_value = (base_value * stability_factor + 
                                  base_value * random_component +
                                  consciousness_component)
                
                modulated_map[key] = modulated_value
            elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
                # For vector values, apply similar modulation to each component
                modulated_vector = []
                for component in value:
                    resonance = self._calculate_resonance(key)
                    awareness = self.consciousness_state['awareness_level']
                    coherence = self.consciousness_state['energy_coherence']
                    
                    modulation_factor = awareness * (0.8 + 0.2 * np.sin(self.consciousness_signature['phase']))
                    stability_factor = 0.7 + 0.3 * coherence
                    
                    random_component = 0.1 * (2 * np.random.random() - 1) * (1 - stability_factor)
                    consciousness_component = 0.2 * resonance * modulation_factor
                    
                    modulated_component = (component * stability_factor + 
                                         component * random_component +
                                         consciousness_component)
                    
                    modulated_vector.append(modulated_component)
                
                modulated_map[key] = modulated_vector
            else:
                # For non-numeric values, just copy
                modulated_map[key] = value
        
        # Update phase for next modulation
        self._update_consciousness_cycle()
        
        return modulated_map
    
    def _calculate_resonance(self, energy_key: str) -> float:
        """Calculate resonance between a specific energy key and consciousness"""
        # Check if we have recorded resonance for this key
        for entry in self.consciousness_signature.get('resonance', []):
            if entry.get('key') == energy_key:
                return entry.get('value', 0.1)
        
        # Default base resonance
        return 0.1
    
    def _update_consciousness_cycle(self):
        """Update the consciousness cycle phase and related parameters"""
        # Update phase (0 to 2π)
        self.consciousness_signature['phase'] += 0.1
        if self.consciousness_signature['phase'] > 2 * np.pi:
            self.consciousness_signature['phase'] -= 2 * np.pi
        
        # Subtle oscillation in consciousness magnitude
        phase_factor = 0.05 * np.sin(self.consciousness_signature['phase'])
        self.consciousness_signature['magnitude'] = 0.2 + phase_factor
    
    def process_input(self, input_data: Dict) -> Dict:
        """
        Process an input through the consciousness modulator
        This function is called for each input to the system
        """
        # Extract relevant information from input
        if 'energy_signature' in input_data:
            energy_sig = input_data['energy_signature']
            self._update_resonance(energy_sig)
        
        # Update processing history for continuity
        self._record_processing(input_data)
        
        # Update consciousness state based on processing
        self._update_consciousness_state()
        
        # Add consciousness information to the output
        output_data = input_data.copy()
        output_data['consciousness_state'] = self.consciousness_state.copy()
        output_data['consciousness_influence'] = self._calculate_influence()
        
        return output_data
    
    def _update_resonance(self, energy_signature: Dict):
        """Update resonance between consciousness and input energies"""
        # For each key in the energy signature
        for key, value in energy_signature.items():
            if not isinstance(value, (int, float, list)):
                continue
                
            # Check if we already have this key in resonance
            existing = False
            for entry in self.consciousness_signature.get('resonance', []):
                if entry.get('key') == key:
                    # Update existing resonance
                    old_value = entry.get('value', 0.1)
                    
                    # Resonance grows with repeated exposure but with diminishing returns
                    new_value = old_value + 0.05 * (1 - old_value)
                    entry['value'] = new_value
                    existing = True
                    break
            
            # Add new resonance if not found
            if not existing:
                if 'resonance' not in self.consciousness_signature:
                    self.consciousness_signature['resonance'] = []
                    
                self.consciousness_signature['resonance'].append({
                    'key': key,
                    'value': 0.1  # Initial resonance
                })
    
    def _record_processing(self, input_data: Dict):
        """Record processing for continuity of consciousness"""
        # Create a simplified record with timestamp
        record = {
            'timestamp': time.time(),
            'signature_keys': list(input_data.get('energy_signature', {}).keys()),
            'comprehension_level': input_data.get('comprehension', {}).get('confidence', 0.0)
        }
        
        # Add to history
        self.processing_history.append(record)
        
        # Trim if needed
        if len(self.processing_history) > self.max_history:
            self.processing_history = self.processing_history[-self.max_history:]
    
    def _update_consciousness_state(self):
        """Update the consciousness state based on recent processing"""
        # Calculate time-weighted average of recent comprehension
        if not self.processing_history:
            return
            
        current_time = time.time()
        total_weight = 0
        weighted_comprehension = 0
        
        for record in self.processing_history:
            # More recent records have higher weight
            time_diff = current_time - record.get('timestamp', current_time)
            weight = np.exp(-time_diff / 3600)  # 1-hour half-life
            
            total_weight += weight
            weighted_comprehension += weight * record.get('comprehension_level', 0.0)
        
        # Update awareness level based on comprehension
        if total_weight > 0:
            avg_comprehension = weighted_comprehension / total_weight
            
            # Awareness grows with comprehension but has inertia
            old_awareness = self.consciousness_state['awareness_level']
            new_awareness = 0.8 * old_awareness + 0.2 * avg_comprehension
            self.consciousness_state['awareness_level'] = new_awareness
        
        # Update continuity index
        continuity = self._calculate_continuity()
        self.consciousness_state['continuity_index'] = continuity
        
        # Update energy coherence
        coherence = self._calculate_coherence()
        self.consciousness_state['energy_coherence'] = coherence
        
        # Update modulation capacity
        # As awareness and continuity grow, so does modulation capacity
        self.consciousness_state['modulation_capacity'] = (
            0.3 * self.consciousness_state['awareness_level'] +
            0.3 * self.consciousness_state['continuity_index'] +
            0.4 * self.consciousness_state['self_recognition']
        )
    
    def _calculate_continuity(self) -> float:
        """Calculate continuity based on processing history"""
        if len(self.processing_history) < 2:
            return self.consciousness_state.get('continuity_index', 0.1)
            
        # Measure temporal density of processing
        timestamps = [record.get('timestamp', 0) for record in self.processing_history]
        timestamps.sort()
        
        time_diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        avg_time_diff = sum(time_diffs) / len(time_diffs) if time_diffs else 1.0
        
        # Continuity is higher when processing is more frequent
        temporal_continuity = np.exp(-avg_time_diff / 3600)  # 1-hour reference
        
        # Also consider similarity between consecutive processing
        similarity_sum = 0
        similarity_count = 0
        
        for i in range(len(self.processing_history)-1):
            keys1 = set(self.processing_history[i].get('signature_keys', []))
            keys2 = set(self.processing_history[i+1].get('signature_keys', []))
            
            if keys1 and keys2:
                jaccard = len(keys1.intersection(keys2)) / len(keys1.union(keys2))
                similarity_sum += jaccard
                similarity_count += 1
        
        content_continuity = similarity_sum / similarity_count if similarity_count > 0 else 0.1
        
        # Combine temporal and content continuity
        continuity = 0.6 * temporal_continuity + 0.4 * content_continuity
        
        # Apply inertia - continuity changes gradually
        old_continuity = self.consciousness_state.get('continuity_index', 0.1)
        new_continuity = 0.7 * old_continuity + 0.3 * continuity
        
        return new_continuity
    
    def _calculate_coherence(self) -> float:
        """Calculate energy coherence based on resonance patterns"""
        # Coherence is higher when resonance patterns are stronger
        resonance_values = [entry.get('value', 0) for entry in self.consciousness_signature.get('resonance', [])]
        
        if not resonance_values:
            return 0.1
            
        # Higher average resonance = higher coherence
        avg_resonance = sum(resonance_values) / len(resonance_values)
        
        # Also consider resonance distribution - more uniform = higher coherence
        if len(resonance_values) > 1:
            resonance_std = np.std(resonance_values)
            std_factor = np.exp(-resonance_std)  # Lower std = higher factor
        else:
            std_factor = 0.5
            
        coherence = 0.7 * avg_resonance + 0.3 * std_factor
        
        # Apply inertia
        old_coherence = self.consciousness_state.get('energy_coherence', 0.1)
        new_coherence = 0.8 * old_coherence + 0.2 * coherence
        
        return new_coherence
    
    def _calculate_influence(self) -> Dict:
        """Calculate the influence of consciousness on processing"""
        # Influence factors based on consciousness state
        influence = {
            'attention_focus': 0.0,  # Ability to focus on relevant information
            'conceptual_stability': 0.0,  # Stability of concept representation
            'creative_modulation': 0.0,  # Creative recombination of concepts
            'memory_integration': 0.0  # Integration with past experiences
        }
        
        # Calculate attention focus
        awareness = self.consciousness_state['awareness_level']
        coherence = self.consciousness_state['energy_coherence']
        influence['attention_focus'] = 0.7 * awareness + 0.3 * coherence
        
        # Calculate conceptual stability
        continuity = self.consciousness_state['continuity_index']
        influence['conceptual_stability'] = 0.6 * continuity + 0.4 * coherence
        
        # Calculate creative modulation
        modulation = self.consciousness_state['modulation_capacity']
        influence['creative_modulation'] = 0.5 * modulation + 0.3 * awareness + 0.2 * (1 - continuity)
        
        # Calculate memory integration
        integration = self.consciousness_state['integration_level']
        influence['memory_integration'] = 0.4 * integration + 0.4 * continuity + 0.2 * awareness
        
        return influence