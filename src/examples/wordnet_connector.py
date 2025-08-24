"""
Example of connecting EVER to WordNet
"""
import os
from nltk.corpus import wordnet as wn
import nltk
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.unified_energy import UnifiedEnergy
from core.consciousness_core import ConsciousnessCore
from connectors.dataset_connector import DatasetConnector

def ensure_wordnet():
    """Ensure WordNet is available"""
    try:
        wn.synsets('test')
    except LookupError:
        print("Downloading WordNet...")
        nltk.download('wordnet')

def map_synset_to_energy(synset):
    """Map a WordNet synset to an energy signature"""
    # Extract properties from synset
    name = synset.name()
    definition = synset.definition()
    examples = synset.examples()
    hypernyms = synset.hypernyms()
    hyponyms = synset.hyponyms()
    
    # Create energy signature based on synset properties
    energy = {
        'magnitude': {
            'value': min(1.0, 0.3 + 0.1 * len(examples) + 0.1 * len(hypernyms) + 0.1 * len(hyponyms))
        },
        'frequency': {
            'value': min(1.0, 0.2 + 0.2 * len(examples))
        },
        'entropy': {
            'value': min(1.0, 0.3 + 0.1 * len(hyponyms) / max(1, len(hypernyms)))
        },
        'vector': {
            'value': [
                0.5,  # x component (neutral)
                0.2 + 0.1 * len(hypernyms) - 0.1 * len(hyponyms),  # y component (higher = more abstract)
                0.5   # z component (neutral)
            ]
        },
        'identity': {
            'value': name
        }
    }
    
    return energy

def setup_wordnet_connector():
    """Set up EVER with WordNet connector"""
    # Ensure WordNet is available
    ensure_wordnet()
    
    # Initialize EVER components
    energy_system = UnifiedEnergy()
    consciousness = ConsciousnessCore()
    
    # Initialize connector
    connector = DatasetConnector(consciousness)
    
    # Connect to WordNet as a "virtual" dataset
    connector.connect_dataset(
        name="wordnet",
        location="nltk:wordnet",  # Custom location indicator
        format_type="api",  # Treat as API since we're using NLTK's API
        schema={
            "type": "lexical_database",
            "entities": ["synset", "lemma"]
        }
    )
    
    # Define custom query function for WordNet
    def query_wordnet(query):
        """Custom query function for WordNet"""
        results = []
        
        # Handle word lookup
        if 'word' in query:
            word = query['word']
            synsets = wn.synsets(word)
            
            for synset in synsets:
                results.append({
                    'name': synset.name(),
                    'pos': synset.pos(),
                    'definition': synset.definition(),
                    'examples': synset.examples(),
                    'lemmas': [lemma.name() for lemma in synset.lemmas()],
                    'hypernyms': [h.name() for h in synset.hypernyms()],
                    'hyponyms': [h.name() for h in synset.hyponyms()[:5]]  # Limit hyponyms
                })
        
        # Handle synset lookup
        elif 'synset' in query:
            synset_name = query['synset']
            try:
                synset = wn.synset(synset_name)
                results.append({
                    'name': synset.name(),
                    'pos': synset.pos(),
                    'definition': synset.definition(),
                    'examples': synset.examples(),
                    'lemmas': [lemma.name() for lemma in synset.lemmas()],
                    'hypernyms': [h.name() for h in synset.hypernyms()],
                    'hyponyms': [h.name() for h in synset.hyponyms()[:5]]  # Limit hyponyms
                })
            except:
                pass
        
        return results
    
    # Replace default accessor with custom function
    connector.connected_datasets['wordnet']['data_accessor'] = query_wordnet
    
    # Define energy mapping for WordNet
    def wordnet_energy_mapper(item):
        """Map WordNet item to energy signature"""
        # Create synthetic synset from item
        class SyntheticSynset:
            def __init__(self, item):
                self.item = item
            
            def name(self):
                return self.item.get('name', '')
            
            def definition(self):
                return self.item.get('definition', '')
            
            def examples(self):
                return self.item.get('examples', [])
            
            def hypernyms(self):
                return self.item.get('hypernyms', [])
            
            def hyponyms(self):
                return self.item.get('hyponyms', [])
        
        # Map to energy
        return map_synset_to_energy(SyntheticSynset(item))
    
    # Register energy mapper
    connector.define_energy_mapping('wordnet', wordnet_energy_mapper)
    
    return connector, energy_system, consciousness

def demo_wordnet_reasoning():
    """Demonstrate EVER reasoning with WordNet"""
    connector, energy_system, consciousness = setup_wordnet_connector()
    
    # Example questions
    questions = [
        "What is the difference between joy and happiness?",
        "How are rivers and streams related?",
        "What is the relationship between thinking and reasoning?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        result = connector.reason_across_datasets(question)
        
        print(f"Answer: {result['synthesis']['answer_text']}")
        
        # Show supporting items
        print("Supporting information:")
        for item in result['synthesis']['supporting_items']:
            print(f"- {item['item']['name']}: {item['item']['definition']}")
            if item['item']['examples']:
                print(f"  Example: {item['item']['examples'][0]}")

if __name__ == "__main__":
    demo_wordnet_reasoning()