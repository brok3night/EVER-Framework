"""
Example of using ENDF format with EVER
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.unified_energy import UnifiedEnergy
from core.consciousness_core import ConsciousnessCore
from formats.endf_converter import ENDFConverter
from connectors.endf_connector import ENDFConnector

def main():
    """Demonstrate ENDF conversion and usage"""
    # Setup directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    # Sample dataset paths
    sample_json = os.path.join(data_dir, 'samples', 'planets.json')
    
    # Create sample data if it doesn't exist
    if not os.path.exists(sample_json):
        os.makedirs(os.path.dirname(sample_json), exist_ok=True)
        create_sample_data(sample_json)
    
    # Initialize EVER components
    energy_system = UnifiedEnergy()
    consciousness = ConsciousnessCore()
    
    # Create ENDF converter
    converter = ENDFConverter(os.path.join(data_dir, 'endf_datasets'))
    
    # Convert sample dataset
    endf_path = converter.convert_dataset(sample_json, 'json', 'planets')
    
    # Connect EVER to the ENDF dataset
    connector = ENDFConnector(consciousness)
    connector.connect(endf_path)
    
    # Query based on energy signature
    earth_signature = {
        'magnitude': {'value': 0.5},
        'frequency': {'value': 0.6},
        'vector': {'value': [0.5, 0.7, 0.5]}
    }
    
    print("\nQuerying for Earth-like planets:")
    results = connector.query_by_energy('planets', earth_signature)
    
    for result in results:
        print(f"Planet: {result['entity_data']['original_data']['name']}")
        print(f"Similarity: {result['similarity']:.2f}")
        print(f"Details: {result['entity_data']['original_data']}")
        print()
    
    # Query using consciousness
    consciousness.state['energy_signature'] = {
        'magnitude': {'value': 0.8},
        'frequency': {'value': 0.3},
        'vector': {'value': [0.3, 0.3, 0.8]}
    }
    
    print("\nQuerying based on consciousness state:")
    consciousness_results = connector.query_by_consciousness('planets')
    
    for result in consciousness_results:
        if 'entity_data' in result:
            print(f"Planet: {result['entity_data']['original_data']['name']}")
            print(f"Resonance/Similarity: {result.get('resonance', result.get('similarity', 0)):.2f}")
        else:
            print(f"Entity: {result['entity_id']}")
            print(f"Resonance: {result['resonance']:.2f}")

def create_sample_data(output_path):
    """Create sample planets dataset"""
    planets = [
        {
            "name": "Mercury",
            "type": "Terrestrial",
            "distance": 0.39,
            "mass": 0.055,
            "diameter": 0.38,
            "has_atmosphere": False,
            "moons": 0
        },
        {
            "name": "Venus",
            "type": "Terrestrial",
            "distance": 0.72,
            "mass": 0.815,
            "diameter": 0.95,
            "has_atmosphere": True,
            "moons": 0
        },
        {
            "name": "Earth",
            "type": "Terrestrial",
            "distance": 1.0,
            "mass": 1.0,
            "diameter": 1.0,
            "has_atmosphere": True,
            "moons": 1
        },
        {
            "name": "Mars",
            "type": "Terrestrial",
            "distance": 1.52,
            "mass": 0.107,
            "diameter": 0.53,
            "has_atmosphere": True,
            "moons": 2
        },
        {
            "name": "Jupiter",
            "type": "Gas Giant",
            "distance": 5.2,
            "mass": 317.8,
            "diameter": 11.2,
            "has_atmosphere": True,
            "moons": 79
        },
        {
            "name": "Saturn",
            "type": "Gas Giant",
            "distance": 9.58,
            "mass": 95.2,
            "diameter": 9.45,
            "has_atmosphere": True,
            "moons": 82
        },
        {
            "name": "Uranus",
            "type": "Ice Giant",
            "distance": 19.22,
            "mass": 14.5,
            "diameter": 4.0,
            "has_atmosphere": True,
            "moons": 27
        },
        {
            "name": "Neptune",
            "type": "Ice Giant",
            "distance": 30.05,
            "mass": 17.1,
            "diameter": 3.88,
            "has_atmosphere": True,
            "moons": 14
        }
    ]
    
    import json
    with open(output_path, 'w') as f:
        json.dump(planets, f, indent=2)
    
    print(f"Created sample planets dataset at {output_path}")

if __name__ == "__main__":
    main()