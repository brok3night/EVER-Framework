# ... existing code ...

def define_word(self, word, definition_text, related_words):
    """Define a word in terms of its energy signature and relationships"""
    # Create energy signature from related words
    energy_signature = {}
    
    # Instead of using preset templates for common definitions,
    # Generate energy signatures dynamically from the definition text itself
    def_words = definition_text.lower().split()
    def_energy = self._extract_base_signature(def_words, [])
    
    # Start with definition-based energy
    for prop, details in def_energy.items():
        if prop not in energy_signature:
            energy_signature[prop] = details.copy()
    
    # Blend with related word energies if available
    for related in related_words:
        if related in self.definitions:
            related_sig = self.definitions[related].get('energy_signature', {})
            
            for prop, details in related_sig.items():
                if prop not in energy_signature:
                    energy_signature[prop] = details.copy()
                else:
                    if isinstance(details['value'], (int, float)) and isinstance(energy_signature[prop]['value'], (int, float)):
                        energy_signature[prop]['value'] = (energy_signature[prop]['value'] + details['value']) / 2
                    elif isinstance(details['value'], list) and isinstance(energy_signature[prop]['value'], list):
                        if len(details['value']) == len(energy_signature[prop]['value']):
                            energy_signature[prop]['value'] = [
                                (a + b) / 2 for a, b in zip(energy_signature[prop]['value'], details['value'])
                            ]
    
    # Create definition
    self.definitions[word] = {
        'text': definition_text,
        'energy_signature': energy_signature,
        'related_words': related_words
    }
    
    # Create correlations
    if word not in self.correlations:
        self.correlations[word] = {}
        
    for related in related_words:
        if related in self.definitions:
            self.correlations[word][related] = 1.0
            
            if related not in self.correlations:
                self.correlations[related] = {}
            self.correlations[related][word] = 1.0