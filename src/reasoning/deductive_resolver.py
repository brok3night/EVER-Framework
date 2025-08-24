"""
Deductive Resolver - Implements deductive reasoning to resolve ambiguities
"""
import numpy as np
from typing import Dict, List, Set, Tuple, Any

class DeductiveResolver:
    def __init__(self, energy_system):
        self.energy = energy_system
        
        # Track logical assertions
        self.assertions = {}
        
        # Track logical rules
        self.rules = []
        
        # Track disambiguated concepts
        self.disambiguations = {}
    
    def add_assertion(self, subject: str, predicate: str, object_value: Any, confidence: float = 1.0) -> None:
        """Add a logical assertion to the knowledge base"""
        assertion_key = (subject, predicate)
        
        if assertion_key not in self.assertions:
            self.assertions[assertion_key] = []
            
        self.assertions[assertion_key].append({
            'value': object_value,
            'confidence': confidence
        })
    
    def add_rule(self, premises: List[Tuple[str, str, Any]], conclusion: Tuple[str, str, Any]) -> None:
        """Add a logical rule (if premises then conclusion)"""
        self.rules.append({
            'premises': premises,
            'conclusion': conclusion
        })
    
    def analyze_ambiguity(self, word: str, contexts: List[str]) -> Dict:
        """
        Analyze ambiguity for a word across different contexts
        Returns differentiating properties
        """
        # Skip if word not in energy system
        if word not in self.energy.definitions:
            return {'error': f"Word '{word}' not found in definitions"}
        
        # Process each context to get energy signatures
        context_signatures = []
        
        for context in contexts:
            # Create context by combining word with context
            full_context = f"{word} {context}"
            
            # Get energy signature for this context
            signature = self.energy.process_text(full_context, {})
            context_signatures.append({
                'context': context,
                'signature': signature
            })
        
        # Find differentiating properties
        differentiators = self._find_differentiating_properties(context_signatures)
        
        # Apply deductive reasoning to resolve ambiguity
        resolution = self._apply_deductive_reasoning(word, differentiators, context_signatures)
        
        return {
            'word': word,
            'contexts': contexts,
            'differentiating_properties': differentiators,
            'resolution': resolution
        }
    
    def _find_differentiating_properties(self, context_signatures: List[Dict]) -> Dict:
        """Find properties that differentiate between contexts"""
        if not context_signatures:
            return {}
            
        # Get all properties from first signature
        all_properties = set()
        for prop, details in context_signatures[0]['signature'].items():
            all_properties.add(prop)
        
        # Check each property for significant differences
        differentiators = {}
        
        for prop in all_properties:
            values = []
            
            # Extract values for this property across contexts
            for ctx_sig in context_signatures:
                if prop in ctx_sig['signature']:
                    prop_details = ctx_sig['signature'][prop]
                    
                    if isinstance(prop_details, dict) and 'value' in prop_details:
                        values.append(prop_details['value'])
                    else:
                        values.append(None)
            
            # Skip if missing values
            if None in values or len(values) != len(context_signatures):
                continue
                
            # Check if values are significantly different
            if all(isinstance(v, (int, float)) for v in values):
                # For numeric values
                min_val = min(values)
                max_val = max(values)
                
                # Consider property differentiating if range is significant
                if max_val - min_val > 0.3:
                    differentiators[prop] = {
                        'type': 'numeric',
                        'values': values,
                        'range': max_val - min_val
                    }
            elif all(isinstance(v, list) for v in values):
                # For vector values
                # Compare vector directions
                directions_differ = False
                
                # Check if all vectors have same length
                if all(len(v) == len(values[0]) for v in values):
                    # Calculate vector directions
                    directions = []
                    
                    for vec in values:
                        # Calculate unit vector
                        magnitude = sum(x**2 for x in vec)**0.5
                        if magnitude > 0:
                            unit_vec = [x/magnitude for x in vec]
                            directions.append(unit_vec)
                        else:
                            directions.append(vec)
                    
                    # Check if directions differ significantly
                    for i in range(len(directions)):
                        for j in range(i+1, len(directions)):
                            # Calculate dot product to measure similarity
                            dot_product = sum(a*b for a, b in zip(directions[i], directions[j]))
                            
                            # If dot product is less than 0.7, directions differ significantly
                            if dot_product < 0.7:
                                directions_differ = True
                                break
                
                if directions_differ:
                    differentiators[prop] = {
                        'type': 'vector',
                        'values': values
                    }
        
        return differentiators
    
    def _apply_deductive_reasoning(self, word: str, differentiators: Dict, 
                                   context_signatures: List[Dict]) -> Dict:
        """Apply deductive reasoning to resolve ambiguity"""
        # If no differentiators, can't resolve
        if not differentiators:
            return {
                'resolved': False,
                'reason': 'No differentiating properties found'
            }
        
        # Create logical assertions based on differentiators
        created_assertions = []
        
        for prop, diff_details in differentiators.items():
            for i, ctx_sig in enumerate(context_signatures):
                context = ctx_sig['context']
                
                if diff_details['type'] == 'numeric':
                    value = diff_details['values'][i]
                    
                    # Create assertion about this property in this context
                    assertion = (word, f"has_{prop}_in_context", (context, value))
                    self.add_assertion(word, f"has_{prop}_in_context", (context, value))
                    created_assertions.append(assertion)
                    
                    # Create comparative assertions
                    for j, other_ctx in enumerate(context_signatures):
                        if i != j:
                            other_context = other_ctx['context']
                            other_value = diff_details['values'][j]
                            
                            if value > other_value:
                                self.add_assertion(word, f"has_higher_{prop}", (context, other_context))
                                created_assertions.append((word, f"has_higher_{prop}", (context, other_context)))
                            else:
                                self.add_assertion(word, f"has_lower_{prop}", (context, other_context))
                                created_assertions.append((word, f"has_lower_{prop}", (context, other_context)))
        
        # Apply logical rules for disambiguation
        disambiguations = []
        
        # For each context pair, apply rules to distinguish meanings
        for i, ctx1 in enumerate(context_signatures):
            for j in range(i+1, len(context_signatures)):
                ctx2 = context_signatures[j]
                
                context1 = ctx1['context']
                context2 = ctx2['context']
                
                # Find distinguishing property with highest difference
                best_prop = None
                max_diff = 0
                
                for prop, diff_details in differentiators.items():
                    if diff_details['type'] == 'numeric':
                        diff = abs(diff_details['values'][i] - diff_details['values'][j])
                        
                        if diff > max_diff:
                            max_diff = diff
                            best_prop = prop
                
                if best_prop:
                    # Create disambiguation based on this property
                    disambiguation = {
                        'contexts': [context1, context2],
                        'differentiating_property': best_prop,
                        'difference': max_diff,
                        'explanation': f"The meaning of '{word}' in '{context1}' differs from '{context2}' primarily in terms of {best_prop}."
                    }
                    
                    # Add logical explanation
                    if diff_details['values'][i] > diff_details['values'][j]:
                        disambiguation['logical_form'] = f"∀x: ({word}(x) ∧ context({context1}, x)) → higher_{best_prop}(x) than ({word}(y) ∧ context({context2}, y))"
                    else:
                        disambiguation['logical_form'] = f"∀x: ({word}(x) ∧ context({context1}, x)) → lower_{best_prop}(x) than ({word}(y) ∧ context({context2}, y))"
                    
                    disambiguations.append(disambiguation)
        
        # Store disambiguations
        self.disambiguations[word] = disambiguations
        
        return {
            'resolved': len(disambiguations) > 0,
            'disambiguations': disambiguations,
            'created_assertions': created_assertions
        }
    
    def resolve_specific_ambiguity(self, word: str, context: str) -> Dict:
        """Resolve a specific word in a specific context"""
        # First check if we have disambiguations for this word
        if word not in self.disambiguations:
            # Try to analyze with this single context
            self.analyze_ambiguity(word, [context])
            
            # If still not disambiguated, can't resolve
            if word not in self.disambiguations:
                return {
                    'resolved': False,
                    'reason': 'No disambiguation data available'
                }
        
        # Check for assertions specific to this context
        context_assertions = []
        for (subj, pred), assertions in self.assertions.items():
            if subj == word and 'context' in pred:
                for assertion in assertions:
                    value = assertion['value']
                    if isinstance(value, tuple) and value[0] == context:
                        context_assertions.append({
                            'predicate': pred,
                            'value': value[1],
                            'confidence': assertion['confidence']
                        })
        
        # Get energy signature for this specific usage
        signature = self.energy.process_text(f"{word} {context}", {})
        
        return {
            'word': word,
            'context': context,
            'energy_signature': signature,
            'context_assertions': context_assertions,
            'resolved': len(context_assertions) > 0
        }