"""
Concept Behavior Demonstration - Demonstrates universal concept embodiment
"""
from src.universal.concept_embodiment import UniversalConceptEmbodiment

def demonstrate_concept_behavior():
    """Demonstrate concept behavior embodiment"""
    # This would use the actual EVER framework in a real implementation
    # Simplified for demonstration
    
    # Create mock network and primitives
    network = MockResonanceNetwork()
    primitives = MockDynamicPrimitives()
    
    # Create concept embodiment system
    embodiment = UniversalConceptEmbodiment(network, primitives)
    
    # Example concepts to embody
    concepts = [
        'badger',       # Yes, we can even embody a badger!
        'democracy',
        'water',
        'analysis',
        'creativity'
    ]
    
    # Embody each concept
    results = {}
    for concept in concepts:
        print(f"\nEmbodying concept: {concept}")
        embodiment_info = embodiment.embody_concept(concept)
        
        if 'error' in embodiment_info:
            print(f"Error: {embodiment_info['error']}")
            continue
        
        # Discover natural behaviors
        behaviors = embodiment.discover_natural_behaviors(concept)
        
        # Generate explanation
        explanation = embodiment.explain_concept_behavior(concept)
        print(explanation)
        
        # Test processing with concept
        test_input = f"This is a test input about {concept} and related topics."
        processing_result = embodiment.process_with_concept(test_input, concept)
        
        print(f"\nProcessing result for '{concept}':")
        print_processing_result(processing_result)
        
        results[concept] = {
            'embodiment': embodiment_info,
            'behaviors': behaviors,
            'explanation': explanation,
            'processing': processing_result
        }
    
    # Try blending concepts
    print("\n\nBlending concepts:")
    blend_pairs = [
        ('democracy', 'water'),
        ('analysis', 'creativity')
    ]
    
    for concept1, concept2 in blend_pairs:
        print(f"\nBlending '{concept1}' and '{concept2}':")
        blended = embodiment.blend_concept_behaviors([concept1, concept2])
        
        if 'error' in blended:
            print(f"Error: {blended['error']}")
            continue
        
        # Generate explanation
        explanation = embodiment.explain_concept_behavior(blended['concept_id'])
        print(explanation)
        
        # Test processing with blended concept
        test_input = f"This is a test input about {concept1}, {concept2}, and related topics."
        processing_result = embodiment.process_with_concept(test_input, blended['concept_id'])
        
        print(f"\nProcessing result for blended concept:")
        print_processing_result(processing_result)
        
        results[f"{concept1}+{concept2}"] = {
            'embodiment': blended,
            'explanation': explanation,
            'processing': processing_result
        }
    
    return results

def print_processing_result(result):
    """Print processing result in readable format"""
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    
    print(f"Concept: {result.get('concept_id', 'unknown')}")
    
    # Print stages
    if 'stages' in result:
        for stage, info in result['stages'].items():
            print(f"  {stage.capitalize()}: {info.get('behavior', 'default')}")
            if 'error' in info:
                print(f"    Error: {info['error']}")
    
    # Print output
    if 'output' in result:
        output = result['output']
        if isinstance(output, dict):
            for key, value in output.items():
                print(f"  {key}: {value}")
        else:
            print(f"  Output: {output}")

# Mock classes for demonstration
class MockResonanceNetwork:
    """Mock resonance network for demonstration"""
    
    def __init__(self):
        # Mock concepts with energy signatures
        self.concepts = {
            'badger': {
                'vector': {'value': [0.3, 0.2, 0.6]},  # Past-oriented, concrete, objective
                'frequency': {'value': 0.8},           # High frequency (active)
                'entropy': {'value': 0.4},             # Moderate entropy
                'magnitude': {'value': 0.7}            # High magnitude (strong presence)
            },
            'democracy': {
                'vector': {'value': [0.6, 0.7, 0.5]},  # Future-oriented, abstract, balanced
                'frequency': {'value': 0.5},           # Moderate frequency
                'entropy': {'value': 0.6},             # Moderate-high entropy
                'magnitude': {'value': 0.8}            # High magnitude
            },
            'water': {
                'vector': {'value': [0.5, 0.2, 0.7]},  # Balanced time, concrete, objective
                'frequency': {'value': 0.7},           # High frequency (fluid)
                'entropy': {'value': 0.3},             # Low entropy (clear)
                'magnitude': {'value': 0.6}            # Moderate magnitude
            },
            'analysis': {
                'vector': {'value': [0.4, 0.8, 0.8]},  # Past-oriented, abstract, objective
                'frequency': {'value': 0.3},           # Low frequency (stable)
                'entropy': {'value': 0.2},             # Low entropy (clear)
                'magnitude': {'value': 0.7}            # High magnitude
            },
            'creativity': {
                'vector': {'value': [0.8, 0.6, 0.3]},  # Future-oriented, mixed abstraction, subjective
                'frequency': {'value': 0.8},           # High frequency (dynamic)
                'entropy': {'value': 0.8},             # High entropy (complex)
                'magnitude': {'value': 0.7}            # High magnitude
            }
        }
        
        # Mock connections
        self.connections = {
            'badger': {
                'animal': {'type': 'is_a', 'strength': 0.9},
                'aggression': {'type': 'exhibits_behavior_of', 'strength': 0.8},
                'digging': {'type': 'performs', 'strength': 0.9}
            },
            'democracy': {
                'government': {'type': 'is_a', 'strength': 0.9},
                'voting': {'type': 'involves', 'strength': 0.9},
                'equality': {'type': 'promotes', 'strength': 0.8}
            },
            'water': {
                'fluid': {'type': 'is_a', 'strength': 0.9},
                'flowing': {'type': 'exhibits_behavior_of', 'strength': 0.8},
                'adaptation': {'type': 'enables', 'strength': 0.7}
            },
            'analysis': {
                'thinking': {'type': 'is_a', 'strength': 0.8},
                'decomposition': {'type': 'involves', 'strength': 0.9},
                'clarity': {'type': 'produces', 'strength': 0.8}
            },
            'creativity': {
                'thinking': {'type': 'is_a', 'strength': 0.7},
                'imagination': {'type': 'involves', 'strength': 0.9},
                'novelty': {'type': 'produces', 'strength': 0.9}
            }
        }
    
    def get_connected_concepts(self, concept_id):
        """Get connected concepts"""
        return self.connections.get(concept_id, {})
    
    def get_concept_context(self, concept_id):
        """Get concept context"""
        connected = self.get_connected_concepts(concept_id)
        return {
            'concept_id': concept_id,
            'strongest_connections': list(connected.items())[:3]
        }

class MockDynamicPrimitives:
    """Mock dynamic primitives for demonstration"""
    
    def __init__(self):
        pass
    
    def get_available_frameworks(self):
        """Get available frameworks"""
        return ['analytical', 'dialectical', 'phenomenological']

# Run demonstration if executed directly
if __name__ == "__main__":
    results = demonstrate_concept_behavior()
    print("\nDemonstration complete.")