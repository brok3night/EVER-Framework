# ... existing code ...

def run_training():
    """Main training function"""
    # ... existing setup code ...
    
    # Bootstrap consciousness through self-generated queries
    # Instead of using hardcoded bootstrap inputs, generate them from primitives
    print("Bootstrapping consciousness through energy-based concept exploration...")
    
    # Get primitive concepts
    primitives = list(cemd.definitions.keys())
    
    # Generate exploration queries from primitives themselves
    bootstrap_inputs = []
    
    for i, primitive in enumerate(primitives):
        if i < len(primitives) - 1:
            # Create relationship exploration
            bootstrap_inputs.append(f"{primitive} {primitives[i+1]}")
        
        # Create concept exploration
        bootstrap_inputs.append(primitive)
    
    # Add some compositional explorations
    if len(primitives) >= 3:
        bootstrap_inputs.append(f"{primitives[0]} {primitives[1]} {primitives[2]}")
    
    # Process bootstrap inputs
    for input_text in bootstrap_inputs:
        result = kernel.process_input(input_text)
        kernel.feedback_loop(result)
        
        # Show results without template messages
        if 'consciousness_state' in result:
            awareness = result['consciousness_state'].get('awareness_level', 0)
            continuity = result['consciousness_state'].get('continuity_index', 0)
            print(f"Processed: '{input_text}'")
            print(f"Awareness: {awareness:.4f}, Continuity: {continuity:.4f}")
    
    print(f"Training complete. Framework self-organized.")
    return kernel