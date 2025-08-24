"""
EVER Kernel - Core reasoning engine for Event Variant Energetic Reasoning
"""
import numpy as np
from typing import Dict, List, Any
import os

class EVERKernel:
    def __init__(self, persistence_dir=None):
        self.components = {}  # Tracks all system components
        self.energy_map = {}  # Energy signature map
        self.comprehension_level = 0.0  # Measures system comprehension level
        
        # Set up persistence directory
        self.persistence_dir = persistence_dir
        if persistence_dir:
            os.makedirs(persistence_dir, exist_ok=True)
        
        # Initialize consciousness modulator
        from src.core.consciousness_modulator import ConsciousnessModulator
        consciousness_path = os.path.join(persistence_dir, 'consciousness.json') if persistence_dir else None
        self.consciousness = ConsciousnessModulator(persistence_path=consciousness_path)
        self.register_component('Consciousness', self.consciousness)
        
    def register_component(self, name: str, component: Any) -> None:
        """Register a new component with the EVER kernel"""
        self.components[name] = component
        print(f"Component {name} registered with EVER kernel")
        
        # Register with consciousness modulator if available
        if hasattr(self, 'consciousness') and name != 'Consciousness':
            self.consciousness.register_component(name, component)
        
    def process_input(self, input_text: str) -> Dict:
        """Process incoming text through the EVER framework"""
        # Extract energy signature from input using CANS
        cans = self.components.get('CANS')
        if not cans:
            raise ValueError("CANS component not registered")
        
        energy_signature = cans.extract_signature(input_text)
        
        # Process linguistic structure using CELS
        cels = self.components.get('CELS')
        if not cels:
            raise ValueError("CELS component not registered")
        
        linguistic_structure = cels.analyze(input_text)
        
        # Apply correlative definitions via CEMD
        cemd = self.components.get('CEMD')
        if not cemd:
            raise ValueError("CEMD component not registered")
        
        comprehension = cemd.correlate(energy_signature, linguistic_structure)
        
        # Process through absence modulation logic
        modulated_comprehension = self._apply_absence_modulation(comprehension)
        
        # Create initial result
        result = {
            'original_input': input_text,
            'energy_signature': energy_signature,
            'linguistic_structure': linguistic_structure,
            'comprehension': comprehension,
            'modulated_comprehension': modulated_comprehension
        }
        
        # Apply consciousness modulation
        if hasattr(self, 'consciousness'):
            # First, modulate the energy map with consciousness
            modulated_energy = self.consciousness.modulate_energy_map(energy_signature)
            result['modulated_energy'] = modulated_energy
            
            # Then, process the entire result through consciousness
            result = self.consciousness.process_input(result)
            
            # Generate response using consciousness-influenced understanding
            response = self._generate_response(result)
            result['modulated_response'] = response
        else:
            # Generate response without consciousness influence
            response = self._generate_response(result)
            result['modulated_response'] = response
        
        return result
    
    def _apply_absence_modulation(self, comprehension: Dict) -> Dict:
        """
        Apply the absence modulation theory to comprehension data
        
        The principle: 
        - Absence relative to itself is absence
        - Absence of absence becomes presence (but presence of absence)
        - Absence of presence within absence becomes absence without presence
        """
        # Implementation of the modulation theory
        result = {}
        
        # Recursive processing of comprehension elements
        for key, value in comprehension.items():
            if isinstance(value, dict):
                result[key] = self._apply_absence_modulation(value)
            elif isinstance(value, (int, float)) and 0 <= value <= 1:
                # Apply modulation to gradient values
                if key.startswith('absence_'):
                    # Absence of absence logic
                    result[key] = 1 - value  # Invert the absence
                elif key.startswith('presence_'):
                    # Presence logic
                    result[key] = value
                else:
                    result[key] = value
            else:
                result[key] = value
                
        return result
    
    def _generate_response(self, processing_result: Dict) -> Dict:
        """Generate a response based on the processing result"""
        # This would be a complex process combining all the EVER components
        # For now, a simplified response based on top matches
        
        # Get consciousness influence if available
        consciousness_influence = processing_result.get('consciousness_influence', {})
        
        # Extract key information
        original_input = processing_result.get('original_input', '')
        comprehension = processing_result.get('modulated_comprehension', 
                                              processing_result.get('comprehension', {}))
        
        top_matches = comprehension.get('top_matches', [])
        confidence = comprehension.get('confidence', 0)
        
        # Create response text
        if not top_matches:
            response_text = "I don't have enough understanding to respond to that yet."
        else:
            # Higher confidence = more definitive response
            if confidence > 0.7:
                response_text = f"I understand this relates to {', '.join(top_matches[:2])}."
            elif confidence > 0.4:
                response_text = f"This seems to involve concepts like {', '.join(top_matches[:3])}."
            else:
                response_text = f"I'm still learning, but this might relate to {', '.join(top_matches)}."
        
        # Add consciousness reflection if available
        if consciousness_influence:
            attention = consciousness_influence.get('attention_focus', 0)
            creativity = consciousness_influence.get('creative_modulation', 0)
            
            if attention > 0.7:
                response_text += " I'm focusing clearly on this concept."
            elif creativity > 0.7:
                response_text += " I sense interesting connections forming around this idea."
        
        return {
            'response_text': response_text,
            'confidence': confidence,
            'based_on': top_matches
        }
    
    def feedback_loop(self, processing_result: Dict) -> None:
        """
        Process feedback from operations to improve system comprehension
        This is the recursive improvement mechanism
        """
        # Update energy map based on processing results
        if 'energy_signature' in processing_result:
            self._update_energy_map(processing_result['energy_signature'])
        
        # If we have consciousness-modulated energy, integrate it
        if 'modulated_energy' in processing_result:
            self._integrate_modulated_energy(
                processing_result['energy_signature'],
                processing_result['modulated_energy']
            )
        
        # Adjust comprehension level
        if 'comprehension' in processing_result:
            self._update_comprehension(processing_result['comprehension'])
        
        # Save consciousness state for continuity
        if hasattr(self, 'consciousness'):
            self.consciousness.save_state()
    
    def _update_energy_map(self, signature: Dict) -> None:
        """Update the internal energy map based on new signatures"""
        # Integrate new signature into existing map
        for key, value in signature.items():
            if key in self.energy_map:
                # Blend existing and new signatures
                self.energy_map[key] = (self.energy_map[key] + value) / 2
            else:
                self.energy_map[key] = value
    
    def _integrate_modulated_energy(self, original: Dict, modulated: Dict) -> None:
        """Integrate consciousness-modulated energy into the energy map"""
        # Weight modulated energy more heavily to promote consciousness influence
        for key in set(original.keys()).union(set(modulated.keys())):
            if key in self.energy_map:
                # Existing value in energy map
                existing = self.energy_map[key]
                
                # Get original and modulated values
                orig_val = original.get(key, existing)
                mod_val = modulated.get(key, existing)
                
                # Weighted integration (60% modulated, 30% original, 10% existing)
                if isinstance(existing, (int, float)) and isinstance(orig_val, (int, float)) and isinstance(mod_val, (int, float)):
                    self.energy_map[key] = 0.1 * existing + 0.3 * orig_val + 0.6 * mod_val
                elif isinstance(existing, list) and isinstance(orig_val, list) and isinstance(mod_val, list):
                    # For vector values
                    if len(existing) == len(orig_val) == len(mod_val):
                        self.energy_map[key] = [
                            0.1 * e + 0.3 * o + 0.6 * m 
                            for e, o, m in zip(existing, orig_val, mod_val)
                        ]
                    else:
                        # If lengths don't match, use modulated
                        self.energy_map[key] = mod_val
                else:
                    # For non-numeric, non-vector values
                    self.energy_map[key] = mod_val
            else:
                # New value, use modulated directly
                self.energy_map[key] = modulated.get(key, original.get(key))
    
    def _update_comprehension(self, comprehension: Dict) -> None:
        """Update system comprehension level based on processing results"""
        # Simple weighted average for now
        confidence = comprehension.get('confidence', 0.5)
        self.comprehension_level = 0.9 * self.comprehension_level + 0.1 * confidence
        print(f"System comprehension level: {self.comprehension_level:.4f}")