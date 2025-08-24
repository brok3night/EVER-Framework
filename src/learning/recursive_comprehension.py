def explore_exponential_connections(self, depth: int = 2) -> None:
    """Explore exponential connections between concepts using property-driven approach"""
    print(f"Beginning exponential connection exploration with depth {depth}")
    
    # Start with structurally significant words
    seed_words = self._select_seed_words_by_structure()
    
    # Track explored connections
    explored_connections = set()
    connection_insights = {}
    
    # For each seed word, explore connections
    for seed in seed_words:
        print(f"Exploring connections from seed word: {seed}")
        
        # Find paths from this seed using structural properties
        paths = self._find_structure_based_paths(seed, depth)
        
        for path in paths:
            # Create a path key
            path_key = " → ".join(path)
            
            # Skip if already explored
            if path_key in explored_connections:
                continue
                
            explored_connections.add(path_key)
            
            # Generate insight for this connection path
            insight = self._generate_structured_insight(path)
            connection_insights[path_key] = insight
            
            # Process this insight through consciousness
            self._process_through_consciousness(path_key, insight)
            
            # Create new emergent concept from significant paths
            if self._is_significant_path(path):
                self._create_emergent_concept(path, insight)
            
            print(f"Discovered connection: {path_key}")
            print(f"Insight: {insight[:100]}..." if len(insight) > 100 else f"Insight: {insight}")
    
    print(f"Explored {len(explored_connections)} unique connection paths")
    print(f"Consciousness awareness now: {self.consciousness.state.get('awareness_level', 0):.4f}")

def _select_seed_words_by_structure(self) -> List[str]:
    """Select seed words based on structural significance"""
    structural_scores = []
    
    for word in self.processing_history:
        if word not in self.energy.definitions:
            continue
            
        # Get structural properties
        structural_props = self.energy.definitions[word].get('structural_properties', {})
        
        # Calculate structural significance score
        score = 0.0
        
        # Connection richness
        if word in self.connection_graph:
            connection_count = len(self.connection_graph[word])
            score += 0.2 * min(1.0, connection_count / 10)
        
        # Directional strength
        if 'directional_strength' in structural_props:
            score += 0.2 * structural_props['directional_strength']
        
        # Concept scope (generalization)
        if 'concept_scope' in structural_props:
            score += 0.1 * structural_props['concept_scope']
        
        # Connection diversity (from entropy)
        if 'connection_diversity' in structural_props:
            score += 0.2 * structural_props['connection_diversity']
        
        # Harmonic relations
        if 'harmonic_relations' in structural_props and isinstance(structural_props['harmonic_relations'], list):
            score += 0.1 * min(1.0, len(structural_props['harmonic_relations']) / 5)
        
        # Memory persistence
        if 'memory_persistence' in structural_props:
            score += 0.1 * structural_props['memory_persistence']
        
        # Add comprehension level
        score += 0.1 * self.processing_history[word].get('comprehension_level', 0.5)
        
        structural_scores.append((word, score))
    
    # Sort by structural significance
    structural_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Take top significant words as seeds
    return [w[0] for w in structural_scores[:20]]

def _find_structure_based_paths(self, start_word: str, depth: int) -> List[List[str]]:
    """Find connection paths using structural properties"""
    if depth <= 0 or start_word not in self.energy.definitions:
        return []
        
    # Use guided exploration based on structural properties
    paths = []
    queue = [(start_word, [start_word])]
    visited = {start_word}
    
    while queue:
        current_word, current_path = queue.pop(0)
        
        # If at maximum depth, add path to results
        if len(current_path) >= depth + 1:
            paths.append(current_path)
            continue
            
        # Get structural properties
        structural_props = self.energy.definitions.get(current_word, {}).get('structural_properties', {})
        
        # Get potential connections prioritized by structural properties
        potential_connections = self._get_prioritized_connections(current_word, structural_props)
        
        # Explore connections
        for connected, priority in potential_connections:
            # Avoid cycles
            if connected in current_path:
                continue
                
            # Skip low priority connections sometimes
            if priority < 0.5 and np.random.random() > priority:
                continue
                
            new_path = current_path + [connected]
            
            # Always add paths of length 2 or more
            if len(new_path) > 2:
                paths.append(new_path)
            
            # Continue exploration if not at max depth
            if len(new_path) <= depth:
                queue.append((connected, new_path))
    
    return paths

def _get_prioritized_connections(self, word: str, structural_props: Dict) -> List[Tuple[str, float]]:
    """Get connections prioritized by structural properties"""
    connections = []
    
    # Start with basic connections from graph
    if word in self.connection_graph:
        for connected in self.connection_graph[word]:
            # Default priority
            priority = 0.5
            connections.append((connected, priority))
    
    # Enhance with structural connections
    for prop_name, prop_value in structural_props.items():
        # Dimensional connections
        if prop_name in ('x_dimension', 'y_dimension', 'z_dimension') and isinstance(prop_value, list):
            for connected in prop_value:
                connections.append((connected, 0.8))  # High priority
        
        # Hierarchical connections
        elif prop_name == 'hierarchical_relations' and isinstance(prop_value, dict):
            # Parent concepts
            if 'parent_concepts' in prop_value:
                for connected in prop_value['parent_concepts']:
                    connections.append((connected, 0.7))
            
            # Child concepts
            if 'child_concepts' in prop_value:
                for connected in prop_value['child_concepts']:
                    connections.append((connected, 0.7))
        
        # Harmonic relations
        elif prop_name == 'harmonic_relations' and isinstance(prop_value, list):
            for connected in prop_value:
                connections.append((connected, 0.9))  # Very high priority
        
        # Dissonant relations
        elif prop_name == 'dissonant_relations' and isinstance(prop_value, list):
            for connected in prop_value:
                connections.append((connected, 0.4))  # Lower priority
        
        # Creative connections
        elif prop_name == 'creative_connections' and isinstance(prop_value, list):
            for connected in prop_value:
                connections.append((connected, 0.6))
        
        # Temporal connections
        elif prop_name == 'temporal_connections' and isinstance(prop_value, list):
            for connected in prop_value:
                connections.append((connected, 0.7))
        
        # Oscillation group
        elif prop_name == 'oscillation_group' and isinstance(prop_value, list):
            for connected in prop_value:
                connections.append((connected, 0.8))
    
    # Remove duplicates while keeping highest priority
    connection_dict = {}
    for connected, priority in connections:
        if connected in connection_dict:
            connection_dict[connected] = max(connection_dict[connected], priority)
        else:
            connection_dict[connected] = priority
    
    # Convert back to list
    return [(c, p) for c, p in connection_dict.items()]

def _generate_structured_insight(self, path: List[str]) -> str:
    """Generate insight about a connection path using structural properties"""
    # Get energy signatures and structural properties for all words in path
    signatures = []
    structures = []
    
    for word in path:
        if word in self.energy.definitions:
            sig = self.energy.definitions[word].get('energy_signature', {})
            struct = self.energy.definitions[word].get('structural_properties', {})
            signatures.append(sig)
            structures.append(struct)
    
    # Skip if missing signatures
    if len(signatures) != len(path):
        return f"Connection between {' and '.join(path)}"
    
    # Analyze structural patterns
    pattern_insights = []
    
    # Check for organizational patterns
    org_levels = [s.get('organization_level', 0.5) for s in structures if 'organization_level' in s]
    if org_levels:
        if max(org_levels) - min(org_levels) > 0.5:
            pattern_insights.append("transition between order and chaos")
        elif np.mean(org_levels) > 0.7:
            pattern_insights.append("highly organized relationship")
        elif np.mean(org_levels) < 0.3:
            pattern_insights.append("entropic interconnection")
    
    # Check for hierarchical patterns
    hierarchical_count = sum(1 for s in structures if 'hierarchical_relations' in s)
    if hierarchical_count > len(path) / 2:
        pattern_insights.append("hierarchical structure")
    
    # Check for directional alignment
    dir_strengths = [s.get('directional_strength', 0) for s in structures if 'directional_strength' in s]
    if dir_strengths and np.mean(dir_strengths) > 0.7:
        pattern_insights.append("strong directional influence")
    
    # Check for phase relationships
    phases = [s.get('interaction_phase', 0) for s in structures if 'interaction_phase' in s]
    if len(phases) > 1:
        phase_diffs = [abs((phases[i] - phases[i+1]) % (2 * np.pi)) for i in range(len(phases)-1)]
        if all(diff < 0.5 or abs(diff - np.pi) < 0.5 for diff in phase_diffs):
            pattern_insights.append("harmonic phase alignment")
        else:
            pattern_insights.append("complex phase relationship")
    
    # Check for scope patterns
    scopes = [s.get('concept_scope', 0.5) for s in structures if 'concept_scope' in s]
    if scopes:
        if max(scopes) - min(scopes) > 0.5:
            pattern_insights.append("transition between specific and general")
        elif np.mean(scopes) > 0.7:
            pattern_insights.append("broad conceptual domain")
        elif np.mean(scopes) < 0.3:
            pattern_insights.append("highly specialized relationship")
    
    # Format insight based on patterns
    if pattern_insights:
        pattern_text = " and ".join(pattern_insights)
        insight = f"The connection from {path[0]} to {path[-1]} through {', '.join(path[1:-1]) if len(path) > 2 else 'direct relation'} demonstrates {pattern_text}."
        
        # Add additional structural details
        for i, word in enumerate(path):
            if i < len(structures) and structures[i]:
                key_props = []
                for prop, value in structures[i].items():
                    if prop in ('organization_level', 'directional_strength', 'concept_scope'):
                        key_props.append(f"{prop}: {value:.2f}")
                
                if key_props:
                    insight += f"\n{word} structural properties: {', '.join(key_props)}"
    else:
        # Use definitions to create insight
        definitions = []
        for word in path:
            if word in self.dictionary:
                def_text = self.dictionary[word].get('definition', '')
                if def_text:
                    definitions.append(f"{word}: {def_text}")
        
        if definitions:
            insight = f"Connection: {' → '.join(path)}\n" + "\n".join(definitions)
        else:
            insight = f"Connection between {' → '.join(path)}"
    
    return insight

def _is_significant_path(self, path: List[str]) -> bool:
    """Determine if a path is significant enough to form a new concept"""
    if len(path) < 3:
        return False
    
    # Check comprehension levels of words in path
    comprehension_levels = [self.processing_history.get(word, {}).get('comprehension_level', 0) for word in path]
    if np.mean(comprehension_levels) < 0.6:
        return False
    
    # Check structural properties
    has_strong_structure = False
    for word in path:
        if word in self.energy.definitions:
            struct = self.energy.definitions[word].get('structural_properties', {})
            
            # Check for strong structural properties
            if 'directional_strength' in struct and struct['directional_strength'] > 0.7:
                has_strong_structure = True
            
            if 'organization_level' in struct and struct['organization_level'] > 0.7:
                has_strong_structure = True
            
            if 'concept_scope' in struct and struct['concept_scope'] > 0.7:
                has_strong_structure = True
    
    if not has_strong_structure:
        return False
    
    # Check if the path reveals an interesting pattern
    # Look for alternating or progressive patterns in energy signatures
    signatures = []
    for word in path:
        if word in self.energy.definitions:
            sig = self.energy.definitions[word].get('energy_signature', {})
            signatures.append(sig)
    
    if len(signatures) == len(path):
        # Check for magnitude progression
        magnitudes = [sig.get('magnitude', {}).get('value', 0.5) for sig in signatures]
        diffs = [magnitudes[i+1] - magnitudes[i] for i in range(len(magnitudes)-1)]
        
        # Check if all differences have the same sign (consistent progression)
        if all(d > 0 for d in diffs) or all(d < 0 for d in diffs):
            return True
        
        # Check for oscillation pattern
        signs = [1 if d > 0 else -1 for d in diffs]
        sign_changes = sum(1 for i in range(len(signs)-1) if signs[i] != signs[i+1])
        
        if sign_changes >= len(signs) / 2:
            return True
    
    # Default to low probability of creating new concept
    return np.random.random() < 0.2  # 20% chance

def _create_emergent_concept(self, path: List[str], insight: str) -> None:
    """Create a new emergent concept from a significant path"""
    # Generate name for emergent concept
    if len(path) == 2:
        name = f"{path[0]}_{path[1]}"
    else:
        name = f"{path[0]}_{path[-1]}_connection"
    
    # Avoid duplicates
    if name in self.energy.definitions:
        name = f"{name}_{len(self.energy.definitions)}"
    
    # Generate definition from insight
    definition = f"Emergent concept representing the connection between {path[0]} and {path[-1]}"
    if len(path) > 2:
        definition += f" through {', '.join(path[1:-1])}"
    
    # Add insight as part of definition
    definition += f". {insight}"
    
    # Define word with all path elements as related words
    self.energy.define_word(name, definition, path)
    
    # Apply property-driven structure
    if name in self.energy.definitions:
        # Create merged energy signature from path elements
        merged_sig = {}
        for word in path:
            if word in self.energy.definitions:
                sig = self.energy.definitions[word].get('energy_signature', {})
                for prop, details in sig.items():
                    if prop not in merged_sig:
                        merged_sig[prop] = details.copy()
                    else:
                        if isinstance(details.get('value'), (int, float)) and isinstance(merged_sig[prop].get('value'), (int, float)):
                            merged_sig[prop]['value'] = (merged_sig[prop]['value'] + details['value']) / 2
        
        # Set emergent structural properties
        structural_props = {
            'emergent': True,
            'path_elements': path,
            'creation_time': np.datetime64('now')
        }
        
        # Add structural properties
        self.energy.definitions[name]['structural_properties'] = structural_props
        
        # Apply property-driven structure
        modified_def = self.property_structure.apply_property_structure(
            name, self.energy.definitions[name], self.energy)
        
        # Update definition
        self.energy.definitions[name] = modified_def
    
    # Add to processing history
    self.processing_history[name] = {
        'iterations': 1,
        'last_complexity': max(self.processing_history.get(word, {}).get('last_complexity', 0) for word in path) + 1,
        'comprehension_level': 0.5  # Start with moderate comprehension
    }
    
    # Add to connection graph
    if name not in self.connection_graph:
        self.connection_graph[name] = set()
    
    # Connect to all path elements
    for word in path:
        self.connection_graph[name].add(word)
        
        if word not in self.connection_graph:
            self.connection_graph[word] = set()
        self.connection_graph[word].add(name)
    
    print(f"Created emergent concept: {name}")