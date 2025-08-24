"""
Tiered Memory System - Manages concept memory across different memory tiers
"""
from typing import Dict, List, Any, Tuple, Set
import time
import numpy as np
import json
import os
import sqlite3

class TieredMemorySystem:
    """Manages memory across working, short-term, and long-term tiers"""
    
    def __init__(self, storage_dir: str = None):
        # Memory tiers
        self.working_memory = {}  # Active processing memory
        self.short_term_memory = {}  # Recently used concepts
        self.long_term_index = {}  # Index of concepts in long-term storage
        
        # Memory limits
        self.memory_limits = {
            'working': 50,    # Number of concepts in working memory
            'short_term': 500 # Number of concepts in short-term memory
        }
        
        # Access tracking
        self.access_timestamps = {}
        
        # Set up storage
        self.storage_dir = storage_dir
        if storage_dir:
            os.makedirs(storage_dir, exist_ok=True)
            self.long_term_db = os.path.join(storage_dir, 'long_term_memory.db')
            self._initialize_long_term_storage()
        else:
            self.long_term_db = None
        
        # Activation thresholds
        self.activation_thresholds = {
            'retrieval': 0.2,  # Minimum activation for retrieval
            'promotion': 0.6,  # Activation to promote to working memory
            'decay_rate': 0.05 # Rate of activation decay
        }
        
        # Context tracking
        self.current_context = {}
        self.context_history = []
        
        # Relationship tracking
        self.relationships = {}
    
    def store_concept(self, concept_id: str, concept_data: Dict,
                     tier: str = 'working', activation: float = 0.8) -> bool:
        """
        Store a concept in memory
        
        Args:
            concept_id: Identifier for the concept
            concept_data: Concept data including energy signature
            tier: Memory tier ('working', 'short_term', or 'long_term')
            activation: Initial activation level
        """
        # Record access timestamp
        current_time = time.time()
        self.access_timestamps[concept_id] = current_time
        
        # Update or add to relationships
        if 'related_concepts' in concept_data:
            self._update_relationships(concept_id, concept_data['related_concepts'])
        
        # Add activation to concept data
        concept_data['_memory'] = {
            'activation': activation,
            'last_access': current_time,
            'access_count': 1 if concept_id not in self.access_timestamps else 
                           concept_data.get('_memory', {}).get('access_count', 0) + 1
        }
        
        # Store in appropriate tier
        if tier == 'working':
            self.working_memory[concept_id] = concept_data
            
            # Check working memory limit
            if len(self.working_memory) > self.memory_limits['working']:
                self._manage_working_memory()
            
            return True
            
        elif tier == 'short_term':
            self.short_term_memory[concept_id] = concept_data
            
            # Check short-term memory limit
            if len(self.short_term_memory) > self.memory_limits['short_term']:
                self._manage_short_term_memory()
            
            return True
            
        elif tier == 'long_term' and self.storage_dir:
            # Store in long-term memory
            return self._store_in_long_term(concept_id, concept_data)
        
        return False
    
    def retrieve_concept(self, concept_id: str, activate: bool = True) -> Dict:
        """
        Retrieve a concept from memory
        
        Args:
            concept_id: Identifier for the concept
            activate: Whether to activate the concept on retrieval
        """
        # Check working memory first
        if concept_id in self.working_memory:
            concept = self.working_memory[concept_id]
            
            if activate:
                self._activate_concept(concept_id, concept, 0.2)
            
            return concept
        
        # Then check short-term memory
        if concept_id in self.short_term_memory:
            concept = self.short_term_memory[concept_id]
            
            if activate:
                self._activate_concept(concept_id, concept, 0.4)
                
                # Consider promoting to working memory
                if concept['_memory']['activation'] > self.activation_thresholds['promotion']:
                    self.working_memory[concept_id] = concept
                    del self.short_term_memory[concept_id]
            
            return concept
        
        # Finally check long-term memory
        if concept_id in self.long_term_index and self.storage_dir:
            concept = self._retrieve_from_long_term(concept_id)
            
            if concept:
                if activate:
                    self._activate_concept(concept_id, concept, 0.6)
                    
                    # Add to short-term memory
                    self.short_term_memory[concept_id] = concept
                
                return concept
        
        return None
    
    def query_by_energy(self, energy_signature: Dict, 
                        similarity_threshold: float = 0.7,
                        max_results: int = 10) -> List[Dict]:
        """
        Query memory by energy signature similarity
        
        Args:
            energy_signature: Energy signature to match
            similarity_threshold: Minimum similarity threshold
            max_results: Maximum number of results to return
        """
        results = []
        
        # Function to calculate similarity and add to results
        def process_concept(concept_id, concept_data):
            if 'energy_signature' in concept_data:
                similarity = self._calculate_energy_similarity(
                    energy_signature, concept_data['energy_signature'])
                
                if similarity >= similarity_threshold:
                    results.append({
                        'concept_id': concept_id,
                        'concept_data': concept_data,
                        'similarity': similarity
                    })
        
        # Search working memory
        for concept_id, concept_data in self.working_memory.items():
            process_concept(concept_id, concept_data)
        
        # Search short-term memory
        for concept_id, concept_data in self.short_term_memory.items():
            if concept_id not in self.working_memory:  # Avoid duplicates
                process_concept(concept_id, concept_data)
        
        # Search long-term memory index
        if self.storage_dir:
            long_term_results = self._query_long_term_by_energy(
                energy_signature, similarity_threshold)
            
            for result in long_term_results:
                if result['concept_id'] not in self.working_memory and \
                   result['concept_id'] not in self.short_term_memory:
                    results.append(result)
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Limit results
        if max_results > 0:
            results = results[:max_results]
        
        # Activate retrieved concepts
        for result in results:
            self._activate_concept(
                result['concept_id'], result['concept_data'], 0.3)
        
        return results
    
    def query_related(self, concept_id: str, relation_types: List[str] = None,
                     min_strength: float = 0.3, max_results: int = 10) -> List[Dict]:
        """
        Query for concepts related to a given concept
        
        Args:
            concept_id: Concept to find relations for
            relation_types: Types of relations to include (None = all)
            min_strength: Minimum relationship strength
            max_results: Maximum number of results to return
        """
        if concept_id not in self.relationships:
            return []
        
        results = []
        
        # Get related concepts
        for related_id, relation_info in self.relationships[concept_id].items():
            # Filter by relation type if specified
            if relation_types and relation_info['type'] not in relation_types:
                continue
            
            # Filter by strength
            if relation_info['strength'] < min_strength:
                continue
            
            # Retrieve related concept
            related_concept = self.retrieve_concept(related_id, activate=False)
            
            if related_concept:
                results.append({
                    'concept_id': related_id,
                    'concept_data': related_concept,
                    'relation_type': relation_info['type'],
                    'relation_strength': relation_info['strength']
                })
        
        # Sort by relationship strength
        results.sort(key=lambda x: x['relation_strength'], reverse=True)
        
        # Limit results
        if max_results > 0:
            results = results[:max_results]
        
        # Activate retrieved concepts
        for result in results:
            self._activate_concept(
                result['concept_id'], result['concept_data'], 0.2)
        
        return results
    
    def set_context(self, context_data: Dict) -> None:
        """
        Set the current context
        
        Args:
            context_data: Context information
        """
        # Archive current context if it exists
        if self.current_context:
            self.context_history.append({
                'context': self.current_context,
                'timestamp': time.time()
            })
            
            # Limit history size
            if len(self.context_history) > 10:
                self.context_history = self.context_history[-10:]
        
        # Set new context
        self.current_context = context_data
        
        # Activate concepts related to context
        if 'related_concepts' in context_data:
            for concept_id in context_data['related_concepts']:
                concept = self.retrieve_concept(concept_id)
                if concept:
                    self._activate_concept(concept_id, concept, 0.3)
    
    def get_context(self) -> Dict:
        """Get current context"""
        return self.current_context
    
    def get_context_history(self) -> List[Dict]:
        """Get context history"""
        return self.context_history
    
    def forget_concept(self, concept_id: str) -> bool:
        """
        Remove a concept from all memory tiers
        
        Args:
            concept_id: Identifier for the concept to forget
        """
        success = False
        
        # Remove from working memory
        if concept_id in self.working_memory:
            del self.working_memory[concept_id]
            success = True
        
        # Remove from short-term memory
        if concept_id in self.short_term_memory:
            del self.short_term_memory[concept_id]
            success = True
        
        # Remove from long-term memory
        if concept_id in self.long_term_index and self.storage_dir:
            self._remove_from_long_term(concept_id)
            success = True
        
        # Remove from relationships
        if concept_id in self.relationships:
            del self.relationships[concept_id]
            
            # Remove references from other relationships
            for other_id in list(self.relationships.keys()):
                if concept_id in self.relationships[other_id]:
                    del self.relationships[other_id][concept_id]
            
            success = True
        
        # Remove from access timestamps
        if concept_id in self.access_timestamps:
            del self.access_timestamps[concept_id]
        
        return success
    
    def decay_activations(self) -> None:
        """Decay activation levels of all concepts"""
        current_time = time.time()
        decay_rate = self.activation_thresholds['decay_rate']
        
        # Decay working memory
        for concept_id, concept in self.working_memory.items():
            if '_memory' in concept:
                # Time-based decay
                time_elapsed = current_time - concept['_memory']['last_access']
                decay_factor = 1.0 - min(1.0, decay_rate * time_elapsed / 3600)  # Decay over hours
                
                concept['_memory']['activation'] *= decay_factor
                
                # Ensure activation stays positive
                concept['_memory']['activation'] = max(0.01, concept['_memory']['activation'])
        
        # Decay short-term memory
        for concept_id, concept in self.short_term_memory.items():
            if '_memory' in concept:
                # Time-based decay
                time_elapsed = current_time - concept['_memory']['last_access']
                decay_factor = 1.0 - min(1.0, decay_rate * 2 * time_elapsed / 3600)  # Faster decay
                
                concept['_memory']['activation'] *= decay_factor
                
                # Ensure activation stays positive
                concept['_memory']['activation'] = max(0.01, concept['_memory']['activation'])
        
        # Manage memory after decay
        self._manage