"""
EVER Processing Pipeline - Core processing flow
"""
from typing import Dict, Any
from src.energy.energy_extraction import EnergyExtractor
from src.reasoning.topographical_reasoning import TopographicalReasoning
from src.reasoning.primitive_actions import PrimitiveActions
from src.comprehension.philosophical_comprehension import PhilosophicalComprehension

class ProcessingPipeline:
    """Core processing pipeline for EVER"""
    
    def __init__(self, consciousness):
        self.consciousness = consciousness
        
        # Initialize components
        self.energy_extractor = EnergyExtractor()
        self.primitive_actions = PrimitiveActions(consciousness.energy_system)
        self.topographical_reasoning = TopographicalReasoning(consciousness.energy_system)
        self.philosophical_comprehension = PhilosophicalComprehension(
            consciousness.energy_system, self.topographical_reasoning)
        
        # Processing context
        self.context_energies = []
        self.context_history = []
        
    def process_input(self, input_text: str) -> Dict:
        """
        Process input text through EVER's pipeline
        
        Args:
            input_text: The input text to process
            
        Returns:
            Processing results with comprehension and response guidance
        """
        # Extract energy signature from input
        input_energy = self.energy_extractor.extract_energy(input_text)
        
        # Update context
        self._update_context(input_text, input_energy)
        
        # Comprehend input using philosophical reasoning
        comprehension = self.philosophical_comprehension.comprehend_statement(
            input_text, input_energy, self.context_energies)
        
        # Generate philosophical response guidance
        response_guidance = self.philosophical_comprehension.generate_philosophical_response(
            comprehension)
        
        # Prepare processing results
        results = {
            'input_text': input_text,
            'input_energy': input_energy,
            'comprehension': comprehension,
            'response_guidance': response_guidance
        }
        
        return results
    
    def _update_context(self, text: str, energy: Dict) -> None:
        """Update processing context"""
        # Add to context energies
        self.context_energies.append(energy)
        
        # Limit context size
        if len(self.context_energies) > 5:
            self.context_energies = self.context_energies[-5:]
        
        # Add to context history
        self.context_history.append({
            'text': text,
            'energy': energy,
            'timestamp': self.consciousness.get_current_time()
        })
        
        # Limit history size
        if len(self.context_history) > 10:
            self.context_history = self.context_history[-10:]