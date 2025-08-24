"""
EVER Integration Test - Shows how components work together
"""
from src.energy.core_energy_fundamentals import CoreEnergyFundamentals
from src.explanation.energy_conceptual_bridge import EnergyConceptualBridge
from src.universal.concept_embodiment import UniversalConceptEmbodiment
from src.organization.multi_relational_hierarchies import MultiRelationalHierarchies

# Mock classes from test_suite.py
from src.testing.ever_test_suite import MockResonanceNetwork, MockDynamicPrimitives

def run_integration_demo():
    """Run integration demonstration"""
    print("EVER Integration Demonstration")
    print("=============================")
    
    # Initialize mock components
    print("\nInitializing components...")
    network = MockResonanceNetwork()
    primitives = MockDynamicPrimitives()
    
    # Initialize EVER components
    energy_fundamentals = CoreEnergyFundamentals()
    bridge = EnergyConceptualBridge(network, primitives)
    embodiment = UniversalConceptEmbodiment(network, primitives)
    hierarchies = MultiRelationalHierarchies(network)
    
    print("Components initialized.")
    
    # Step 1: Generate energy signatures
    print("\nStep 1: Generate energy signatures")
    print("---------------------------------")
    
    concepts = ["justice", "freedom", "equality", "democracy"]
    signatures = {}
    
    for concept in concepts:
        print(f"Generating signature for '{concept}'...")
        signatures[concept] = energy_fundamentals.generate_binary_signature(concept)
        
        # Print key properties
        vector = signatures[concept]['vector']['value']
        frequency = signatures[concept]['frequency']['value']
        entropy = signatures[concept]['entropy']['value']
        
        print(f"  Vector: [{', '.join(f'{v:.2f}' for v in vector[:3])}]")
        print(f"  Frequency: {frequency:.2f}, Entropy: {entropy:.2f}")
    
    print("Energy signatures generated.")
    
    # Step 2: Translate energy to concepts
    print("\nStep 2: Translate energy to concepts")
    print("----------------------------------")
    
    for concept, signature in signatures.items():
        print(f"Translating '{concept}' energy signature...")
        translation = bridge.translate_energy_signature(signature)
        print(f"  Description: {translation['description']}")
    
    # Step 3: Embody concepts
    print("\nStep 3: Embody concepts")
    print("---------------------")
    
    for concept in concepts:
        print(f"Embodying '{concept}'...")
        embodiment_info = embodiment.embody_concept(concept)
        
        # Get natural behaviors
        behaviors = embodiment.discover_natural_behaviors(concept)
        
        if 'dominant_behaviors' in behaviors:
            print("  Dominant behaviors:")
            for behavior, description in list(behaviors['dominant_behaviors'].items())[:2]:
                print(f"    - {behavior}: {description}")
    
    # Step 4: Process with embodied concepts
    print("\nStep 4: Process with embodied concepts")
    print("------------------------------------")
    
    test_input = "How do different political systems balance freedom and equality?"
    
    print(f"Processing input: '{test_input}'")
    
    for concept in ["democracy", "justice"]:
        print(f"Processing with '{concept}' embodiment...")
        result = embodiment.process_with_concept(test_input, concept)
        
        # Summarize processing stages
        if 'stages' in result:
            print("  Processing stages:")
            for stage, info in result['stages'].items():
                print(f"    - {stage}: {info.get('behavior', 'default')}")
    
    # Step 5: Blend concepts
    print("\nStep 5: Blend concepts")
    print("-------------------")
    
    print("Blending 'justice' and 'democracy'...")
    blended = embodiment.blend_concept_behaviors(['justice', 'democracy'])
    
    print(f"Created blended concept: '{blended['concept_id']}'")
    explanation = embodiment.explain_concept_behavior(blended['concept_id'])
    print(f"Blend explanation excerpt:\n  {explanation.split('\n')[0]}")
    
    # Process with blended concept
    print("Processing with blended concept...")
    blended_result = embodiment.process_with_concept(test_input, blended['concept_id'])
    
    if 'output' in blended_result:
        print(f"  Output type: {type(blended_result['output'])}")
    
    print("\nIntegration demonstration completed successfully.")
    return True

if __name__ == "__main__":
    run_integration_demo()