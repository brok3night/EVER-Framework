"""
Dual-Signature Language Processor - Processes both linguistic and semantic energy signatures
"""
from typing import Dict, Tuple, List, Any
import numpy as np

from src.core.interfaces import EnergySystem
from src.energy.resonance_network import ResonanceNetwork

class DualSignatureProcessor:
    """Processes linguistic inputs through dual energy signatures"""
    
    def __init__(self, energy_system: EnergySystem, resonance_network: ResonanceNetwork):
        self.energy = energy_system
        self.network = resonance_network
        
        # Linguistic signature patterns (based on grammatical properties)
        self.linguistic_patterns = {
            'noun': {'vector': [0.3, 0.5, 0.7], 'frequency': 0.4, 'stability': 0.8},
            'verb': {'vector': [0.7, 0.4, 0.3], 'frequency': 0.7, 'stability': 0.5},
            'adjective': {'vector': [0.4, 0.7, 0.3], 'frequency': 0.5, 'stability': 0.6},
            'abstract_noun': {'vector': [0.2, 0.8, 0.6], 'frequency': 0.3, 'stability': 0.7},
            'concrete_noun': {'vector': [0.6, 0.3, 0.7], 'frequency': 0.5, 'stability': 0.9},
            # Additional patterns for other linguistic elements
        }
    
    def process_input(self, text: str) -> Dict:
        """
        Process text input and extract dual energy signatures
        
        Args:
            text: Input text
            
        Returns:
            Processing results including dual signatures
        """
        # Break input into words/phrases (simplified for example)
        elements = text.split()
        
        # Process each element
        processed_elements = []
        
        for element in elements:
            # Extract both signature types
            linguistic_sig, semantic_sig = self._extract_dual_signatures(element)
            
            # Find resonant concepts
            resonant_concepts = self.network.find_resonant_concepts(semantic_sig, top_n=5)
            
            processed_elements.append({
                'text': element,
                'linguistic_signature': linguistic_sig,
                'semantic_signature': semantic_sig,
                'resonant_concepts': resonant_concepts
            })
        
        # Generate composite signatures for the entire input
        composite_linguistic = self._compose_linguistic_signatures(
            [e['linguistic_signature'] for e in processed_elements]
        )
        
        composite_semantic = self._compose_semantic_signatures(
            [e['semantic_signature'] for e in processed_elements]
        )
        
        # Get resonant field for the entire input
        concept_ids = []
        for element in processed_elements:
            for concept_id, _ in element.get('resonant_concepts', []):
                concept_ids.append(concept_id)
        
        resonant_field = self.network.get_resonant_field(concept_ids)
        
        # Prepare result
        result = {
            'text': text,
            'elements': processed_elements,
            'composite_linguistic': composite_linguistic,
            'composite_semantic': composite_semantic,
            'resonant_field': resonant_field
        }
        
        return result
    
    def _extract_dual_signatures(self, element: str) -> Tuple[Dict, Dict]:
        """
        Extract linguistic and semantic signatures for an element
        
        Args:
            element: Text element (word/phrase)
            
        Returns:
            Tuple of (linguistic_signature, semantic_signature)
        """
        # Determine linguistic type (simplified for example)
        linguistic_type = self._determine_linguistic_type(element)
        
        # Get base linguistic signature for this type
        if linguistic_type in self.linguistic_patterns:
            linguistic_sig = self.linguistic_patterns[linguistic_type].copy()
        else:
            # Default linguistic signature
            linguistic_sig = {
                'vector': [0.5, 0.5, 0.5],
                'frequency': 0.5,
                'stability': 0.5
            }
        
        # Add unique variations based on the specific element
        linguistic_sig = self._add_linguistic_variations(linguistic_sig, element)
        
        # Extract semantic signature (the element's unique "energy")
        semantic_sig = self._extract_semantic_signature(element)
        
        # Format as proper energy signatures
        linguistic_energy = {
            'vector': {'value': linguistic_sig['vector']},
            'frequency': {'value': linguistic_sig['frequency']},
            'entropy': {'value': 1.0 - linguistic_sig['stability']},
            'magnitude': {'value': 0.7},
            'meta': {'linguistic_type': linguistic_type}
        }
        
        semantic_energy = {
            'vector': {'value': semantic_sig['vector']},
            'frequency': {'value': semantic_sig['frequency']},
            'entropy': {'value': semantic_sig['entropy']},
            'magnitude': {'value': semantic_sig['magnitude']},
            'meta': {'concept': element}
        }
        
        return linguistic_energy, semantic_energy
    
    def _determine_linguistic_type(self, element: str) -> str:
        """Determine linguistic type of an element"""
        # This would use more sophisticated linguistics in a full implementation
        # Simplified version for example
        
        # Check for common verbs
        common_verbs = ['is', 'are', 'run', 'jump', 'think', 'feel', 'have', 'do']
        if element.lower() in common_verbs:
            return 'verb'
        
        # Check for common adjectives
        common_adjectives = ['good', 'bad', 'big', 'small', 'happy', 'sad']
        if element.lower() in common_adjectives:
            return 'adjective'
        
        # Check for abstract nouns
        abstract_nouns = ['love', 'hate', 'thought', 'idea', 'freedom', 'justice']
        if element.lower() in abstract_nouns:
            return 'abstract_noun'
        
        # Check for concrete nouns
        concrete_nouns = ['boat', 'car', 'tree', 'house', 'book']
        if element.lower() in concrete_nouns:
            return 'concrete_noun'
        
        # Default to noun for unknown words
        return 'noun'
    
    def _add_linguistic_variations(self, base_sig: Dict, element: str) -> Dict:
        """Add variations to linguistic signature based on the specific element"""
        # Clone the base signature
        sig = base_sig.copy()
        
        # Add unique variations based on the element
        # In a real implementation, these would be based on sophisticated linguistic analysis
        # For example, word length might affect certain properties
        length_factor = len(element) / 10.0  # Normalize by typical word length
        sig['vector'][0] += (length_factor - 0.5) * 0.1  # Small adjustment
        
        # First letter might affect another property
        first_letter = element[0].lower()
        letter_position = (ord(first_letter) - ord('a')) / 26.0  # 0-1 range
        sig['frequency'] += (letter_position - 0.5) * 0.1  # Small adjustment
        
        # Ensure values stay in valid ranges
        sig['vector'] = [max(0.0, min(1.0, v)) for v in sig['vector']]
        sig['frequency'] = max(0.0, min(1.0, sig['frequency']))
        sig['stability'] = max(0.0, min(1.0, sig['stability']))
        
        return sig
    
    def _extract_semantic_signature(self, element: str) -> Dict:
        """Extract semantic signature for an element"""
        # In a full implementation, this would use sophisticated methods
        # to generate a unique energy signature for the concept
        
        # For example, could use:
        # - Pre-trained embeddings
        # - Phonetic patterns
        # - Cultural associations
        # - Experiential data
        
        # Simple placeholder implementation
        # Create a deterministic but unique signature based on the string
        seed = sum(ord(c) for c in element)
        np.random.seed(seed)
        
        # Generate vector (would be more meaningful in real implementation)
        vector_length = 5  # Use 5D vectors for semantic signatures
        vector = np.random.rand(vector_length).tolist()
        
        # Generate other properties
        frequency = 0.3 + 0.4 * np.random.rand()  # 0.3-0.7 range
        entropy = 0.2 + 0.5 * np.random.rand()    # 0.2-0.7 range
        magnitude = 0.5 + 0.4 * np.random.rand()  # 0.5-0.9 range
        
        return {
            'vector': vector,
            'frequency': frequency,
            'entropy': entropy,
            'magnitude': magnitude
        }
    
    def _compose_linguistic_signatures(self, signatures: List[Dict]) -> Dict:
        """Compose multiple linguistic signatures into one"""
        if not signatures:
            return {
                'vector': {'value': [0.5, 0.5, 0.5]},
                'frequency': {'value': 0.5},
                'entropy': {'value': 0.5},
                'magnitude': {'value': 0.5}
            }
        
        # Extract values
        vectors = []
        frequencies = []
        entropies = []
        magnitudes = []
        
        for sig in signatures:
            if 'vector' in sig and 'value' in sig['vector']:
                vectors.append(sig['vector']['value'])
            
            if 'frequency' in sig and 'value' in sig['frequency']:
                frequencies.append(sig['frequency']['value'])
            
            if 'entropy' in sig and 'value' in sig['entropy']:
                entropies.append(sig['entropy']['value'])
            
            if 'magnitude' in sig and 'value' in sig['magnitude']:
                magnitudes.append(sig['magnitude']['value'])
        
        # Compose values
        composed = {}
        
        if vectors:
            # For vectors, use component-wise average
            # First, pad all vectors to same length
            max_length = max(len(v) for v in vectors)
            padded_vectors = []
            
            for v in vectors:
                if len(v) < max_length:
                    padded_vectors.append(v + [0.5] * (max_length - len(v)))
                else:
                    padded_vectors.append(v)
            
            # Now average
            avg_vector = []
            for i in range(max_length):
                avg_vector.append(sum(v[i] for v in padded_vectors) / len(padded_vectors))
            
            composed['vector'] = {'value': avg_vector}
        
        if frequencies:
            composed['frequency'] = {'value': sum(frequencies) / len(frequencies)}
        
        if entropies:
            # Entropy increases with more elements
            base_entropy = sum(entropies) / len(entropies)
            composed['entropy'] = {'value': min(1.0, base_entropy * (1.0 + 0.1 * len(entropies)))}
        
        if magnitudes:
            composed['magnitude'] = {'value': sum(magnitudes) / len(magnitudes)}
        
        return composed
    
    def _compose_semantic_signatures(self, signatures: List[Dict]) -> Dict:
        """Compose multiple semantic signatures into one"""
        # Similar to linguistic composition but with different rules
        
        if not signatures:
            return {
                'vector': {'value': [0.5, 0.5, 0.5, 0.5, 0.5]},
                'frequency': {'value': 0.5},
                'entropy': {'value': 0.5},
                'magnitude': {'value': 0.5}
            }
        
        # Extract values
        vectors = []
        frequencies = []
        entropies = []
        magnitudes = []
        
        for sig in signatures:
            if 'vector' in sig and 'value' in sig['vector']:
                vectors.append(sig['vector']['value'])
            
            if 'frequency' in sig and 'value' in sig['frequency']:
                frequencies.append(sig['frequency']['value'])
            
            if 'entropy' in sig and 'value' in sig['entropy']:
                entropies.append(sig['entropy']['value'])
            
            if 'magnitude' in sig and 'value' in sig['magnitude']:
                magnitudes.append(sig['magnitude']['value'])
        
        # Compose values
        composed = {}
        
        if vectors:
            # For semantic vectors, use weighted average based on magnitudes
            max_length = max(len(v) for v in vectors)
            avg_vector = [0.0] * max_length
            total_weight = 0.0
            
            for i, v in enumerate(vectors):
                weight = magnitudes[i] if i < len(magnitudes) else 0.5
                total_weight += weight
                
                for j in range(min(len(v), max_length)):
                    avg_vector[j] += v[j] * weight
            
            if total_weight > 0:
                avg_vector = [v / total_weight for v in avg_vector]
            
            composed['vector'] = {'value': avg_vector}
        
        if frequencies:
            # For semantic frequencies, higher values dominate
            composed['frequency'] = {'value': max(frequencies)}
        
        if entropies:
            # For semantic entropy, increases with more diverse elements
            base_entropy = sum(entropies) / len(entropies)
            # Calculate variance as a measure of diversity
            variance = sum((e - base_entropy) ** 2 for e in entropies) / len(entropies)
            composed['entropy'] = {'value': min(1.0, base_entropy + variance * 2.0)}
        
        if magnitudes:
            # For semantic magnitude, strongest signals dominate
            composed['magnitude'] = {'value': max(0.5, sum(magnitudes) / len(magnitudes))}
        
        return composed