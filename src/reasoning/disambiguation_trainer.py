"""
Disambiguation Trainer - Trains the deductive resolver on ambiguous words
"""
import os
import json
from typing import Dict, List
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.unified_energy import UnifiedEnergy
from reasoning.deductive_resolver import DeductiveResolver

def load_ambiguity_dataset(path: str) -> Dict:
    """Load ambiguity dataset from file"""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading ambiguity dataset: {e}")
        return {}

def create_sample_dataset(output_path: str) -> None:
    """Create a sample ambiguity dataset"""
    dataset = {
        "bank": {
            "contexts": [
                "I deposited money at the bank",
                "We sat by the river bank",
                "The plane had to bank sharply to avoid the storm"
            ],
            "senses": [
                "financial institution",
                "edge of river",
                "tilt or incline"
            ]
        },
        "light": {
            "contexts": [
                "The room was filled with light",
                "This backpack is very light",
                "I light the candle every evening"
            ],
            "senses": [
                "illumination",
                "not heavy",
                "ignite"
            ]
        },
        "ring": {
            "contexts": [
                "She wore a gold ring on her finger",
                "The telephone would not stop ringing",
                "They formed a ring around the campfire"
            ],
            "senses": [
                "circular band of jewelry",
                "sound of bell",
                "circular arrangement"
            ]
        },
        "run": {
            "contexts": [
                "I run five miles every morning",
                "Don't let the water run too long",
                "She will run for president next year"
            ],
            "senses": [
                "move quickly on foot",
                "flow",
                "compete for office"
            ]
        },
        "star": {
            "contexts": [
                "We could see every star in the sky that night",
                "She is a movie star now",
                "The students who star in the play are very talented"
            ],
            "senses": [
                "celestial body",
                "famous performer",
                "perform as main character"
            ]
        }
    }
    
    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save dataset
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Created sample ambiguity dataset with {len(dataset)} words at {output_path}")

def train_disambiguation(resolver: DeductiveResolver, dataset: Dict) -> Dict:
    """Train the deductive resolver on the ambiguity dataset"""
    results = {}
    
    for word, word_data in dataset.items():
        print(f"Training disambiguation for '{word}'...")
        
        contexts = word_data.get('contexts', [])
        senses = word_data.get('senses', [])
        
        # Skip if missing data
        if not contexts:
            results[word] = {
                'status': 'skipped',
                'reason': 'No contexts provided'
            }
            continue
        
        # Analyze ambiguity
        analysis = resolver.analyze_ambiguity(word, contexts)
        
        # Save analysis results
        results[word] = {
            'status': 'analyzed',
            'differentiating_properties': analysis.get('differentiating_properties', {}),
            'disambiguations': analysis.get('resolution', {}).get('disambiguations', [])
        }
        
        # Add senses to results if available
        if senses and len(senses) == len(contexts):
            results[word]['senses'] = senses
            
            # Add sense mapping to disambiguations
            disambiguations = results[word].get('disambiguations', [])
            for i, disambiguation in enumerate(disambiguations):
                contexts_in_disambiguation = disambiguation.get('contexts', [])
                sense_mapping = {}
                
                for context in contexts_in_disambiguation:
                    if context in contexts:
                        context_index = contexts.index(context)
                        if context_index < len(senses):
                            sense_mapping[context] = senses[context_index]
                
                disambiguation['sense_mapping'] = sense_mapping
                disambiguations[i] = disambiguation
            
            results[word]['disambiguations'] = disambiguations
    
    return results

def main():
    """Main function to run disambiguation training"""
    # Set up paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    ambiguity_path = os.path.join(data_dir, 'disambiguation', 'ambiguity_dataset.json')
    
    # Create directories
    os.makedirs(os.path.join(data_dir, 'disambiguation'), exist_ok=True)
    
    # Check if dataset exists, create sample if not
    if not os.path.exists(ambiguity_path):
        create_sample_dataset(ambiguity_path)
    
    # Load dataset
    dataset = load_ambiguity_dataset(ambiguity_path)
    
    if not dataset:
        print("No ambiguity dataset available. Exiting.")
        return
    
    # Initialize energy system
    energy_system = UnifiedEnergy()
    
    # Initialize deductive resolver
    resolver = DeductiveResolver(energy_system)
    
    # Train disambiguation
    results = train_disambiguation(resolver, dataset)
    
    # Save results
    results_path = os.path.join(data_dir, 'disambiguation', 'disambiguation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Disambiguation training complete. Results saved to {results_path}")
    
    # Print summary
    resolved_count = sum(1 for word_result in results.values() 
                        if word_result.get('status') == 'analyzed' and 
                        len(word_result.get('disambiguations', [])) > 0)
    
    print(f"Successfully disambiguated {resolved_count}/{len(dataset)} words")
    
    # Show example of a disambiguated word
    example_word = next((word for word in dataset.keys() 
                        if results.get(word, {}).get('status') == 'analyzed' and 
                        len(results.get(word, {}).get('disambiguations', [])) > 0), None)
    
    if example_word:
        print(f"\nExample disambiguation for '{example_word}':")
        disambiguations = results[example_word].get('disambiguations', [])
        for i, disambiguation in enumerate(disambiguations):
            print(f"  Disambiguation {i+1}:")
            print(f"    Contexts: {', '.join(disambiguation.get('contexts', []))}")
            print(f"    Property: {disambiguation.get('differentiating_property', '')}")
            print(f"    Explanation: {disambiguation.get('explanation', '')}")
            
            sense_mapping = disambiguation.get('sense_mapping', {})
            if sense_mapping:
                print(f"    Senses:")
                for context, sense in sense_mapping.items():
                    print(f"      '{context}' -> {sense}")

if __name__ == "__main__":
    main()