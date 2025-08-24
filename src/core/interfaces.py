"""
Core interfaces for EVER components
"""
from typing import Dict, List, Any, Protocol
from abc import ABC, abstractmethod

class EnergySystem(ABC):
    """Interface defining required methods for energy systems"""
    
    @abstractmethod
    def extract_energy(self, input_data: Any) -> Dict:
        """Extract energy signature from input data"""
        pass
    
    @abstractmethod
    def compare_energies(self, energy1: Dict, energy2: Dict) -> float:
        """Compare two energy signatures and return similarity score"""
        pass
    
    @abstractmethod
    def transform_energy(self, energy: Dict, transformation: str, params: Dict = None) -> Dict:
        """Apply transformation to energy signature"""
        pass
    
    @abstractmethod
    def get_concept_energy(self, concept_id: str) -> Dict:
        """Retrieve energy signature for a concept"""
        pass
    
    @abstractmethod
    def store_concept_energy(self, concept_id: str, energy: Dict) -> bool:
        """Store energy signature for a concept"""
        pass

class Consciousness(ABC):
    """Interface defining required methods for consciousness"""
    
    @abstractmethod
    def get_current_time(self) -> float:
        """Get current system time"""
        pass
    
    @abstractmethod
    def get_energy_system(self) -> EnergySystem:
        """Get reference to energy system"""
        pass
    
    @abstractmethod
    def update_state(self, state_update: Dict) -> None:
        """Update consciousness state"""
        pass
    
    @abstractmethod
    def get_state(self) -> Dict:
        """Get current consciousness state"""
        pass
    
    @abstractmethod
    def register_event(self, event_type: str, event_data: Dict) -> None:
        """Register an event in consciousness"""
        pass

class EnergySignature:
    """Standard energy signature structure"""
    
    def __init__(self, data: Dict = None):
        # Initialize with default values
        self.data = {
            'magnitude': {'value': 0.5},
            'frequency': {'value': 0.5},
            'entropy': {'value': 0.5},
            'vector': {'value': [0.5, 0.5, 0.5]},
            'meta': {}
        }
        
        # Update with provided data if any
        if data:
            self.update(data)
    
    def update(self, data: Dict) -> None:
        """Update signature with new data"""
        for key, value in data.items():
            if key in self.data:
                if isinstance(value, dict):
                    self.data[key].update(value)
                else:
                    self.data[key] = value
            else:
                self.data[key] = value
    
    def get(self, key: str, default=None):
        """Get a value from the signature"""
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in the signature"""
        self.data[key] = value
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return self.data.copy()
    
    def __getitem__(self, key):
        """Allow dictionary-like access"""
        return self.data[key]
    
    def __setitem__(self, key, value):
        """Allow dictionary-like assignment"""
        self.data[key] = value
    
    def __contains__(self, key):
        """Allow 'in' operator"""
        return key in self.data
    
    def keys(self):
        """Return keys"""
        return self.data.keys()
    
    def values(self):
        """Return values"""
        return self.data.values()
    
    def items(self):
        """Return items"""
        return self.data.items()
    
    @staticmethod
    def validate(data: Dict) -> bool:
        """Validate an energy signature dictionary"""
        required_keys = ['magnitude', 'frequency', 'entropy', 'vector']
        
        # Check required keys
        if not all(key in data for key in required_keys):
            return False
        
        # Check that values are dictionaries with 'value' key
        for key in required_keys:
            if not isinstance(data[key], dict) or 'value' not in data[key]:
                return False
        
        # Check vector is a list with at least 3 elements
        vector = data['vector'].get('value')
        if not isinstance(vector, list) or len(vector) < 3:
            return False
        
        # Check numerical values
        for key in ['magnitude', 'frequency', 'entropy']:
            value = data[key].get('value')
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                return False
        
        return True