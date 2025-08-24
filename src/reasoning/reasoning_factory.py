"""
Factory for creating reasoning components without circular dependencies
"""
from typing import Dict, Any

class ReasoningFactory:
    """Factory to create and connect reasoning components"""
    
    def __init__(self, energy_system):
        self.energy_system = energy_system
        self.components = {}
    
    def create_primitive_actions(self):
        """Create primitive actions component"""
        from src.reasoning.primitive_actions import PrimitiveActions
        
        primitive_actions = PrimitiveActions(self.energy_system)
        self.components['primitive_actions'] = primitive_actions
        
        return primitive_actions
    
    def create_topographical_reasoning(self):
        """Create topographical reasoning component"""
        from src.reasoning.topographical_reasoning import TopographicalReasoning
        
        # Get or create primitive actions
        if 'primitive_actions' not in self.components:
            primitive_actions = self.create_primitive_actions()
        else:
            primitive_actions = self.components['primitive_actions']
        
        # Create topographical reasoning with reference to primitive actions
        topographical_reasoning = TopographicalReasoning(self.energy_system)
        topographical_reasoning.primitives = primitive_actions
        
        self.components['topographical_reasoning'] = topographical_reasoning
        
        return topographical_reasoning
    
    def create_philosophical_comprehension(self):
        """Create philosophical comprehension component"""
        from src.comprehension.philosophical_comprehension import PhilosophicalComprehension
        
        # Get or create topographical reasoning
        if 'topographical_reasoning' not in self.components:
            topographical_reasoning = self.create_topographical_reasoning()
        else:
            topographical_reasoning = self.components['topographical_reasoning']
        
        # Create philosophical comprehension
        philosophical_comprehension = PhilosophicalComprehension(
            self.energy_system, topographical_reasoning)
        
        self.components['philosophical_comprehension'] = philosophical_comprehension
        
        return philosophical_comprehension
    
    def get_component(self, component_name: str) -> Any:
        """Get a component by name, creating it if necessary"""
        if component_name in self.components:
            return self.components[component_name]
        
        # Create requested component
        if component_name == 'primitive_actions':
            return self.create_primitive_actions()
        elif component_name == 'topographical_reasoning':
            return self.create_topographical_reasoning()
        elif component_name == 'philosophical_comprehension':
            return self.create_philosophical_comprehension()
        
        return None