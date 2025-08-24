"""
Demonstration of EVER's knowledge connection capabilities
"""
import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.unified_energy import UnifiedEnergy
from src.core.consciousness_core import ConsciousnessCore
from src.connectors.knowledge_evr import KnowledgeEVRConnector

def demonstrate_knowledge_connection():
    """Show how EVER connects to knowledge sources"""
    # Initialize EVER components
    energy_system = UnifiedEnergy()
    consciousness = ConsciousnessCore()
    
    # Create knowledge connector
    connector = KnowledgeEVRConnector(consciousness)
    
    # Set up sample data directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "knowledge")
    os.makedirs(data_dir, exist_ok=True)
    
    # Create sample Wikipedia dataset
    wiki_path = os.path.join(data_dir, "sample_wikipedia.json")
    connector.create_sample_wikipedia(wiki_path, article_count=20)
    
    # Connect to Wikipedia
    print("\nConnecting to Wikipedia...")
    connector.connect_wikipedia(wiki_path)
    
    # Try to connect to WordNet if available
    try:
        print("\nConnecting to WordNet...")
        connector.connect_wordnet()
    except Exception as e:
        print(f"Couldn't connect to WordNet: {e}")
    
    # Query Wikipedia by text
    queries = [
        "What is quantum physics?",
        "Tell me about art and paintings",
        "Historical events in the 20th century"
    ]
    
    for query in queries:
        print(f"\n\nQuery: {query}")
        results = connector.query_by_text('wikipedia', query, limit=3)
        
        print(f"Wikipedia results:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['title']} (Similarity: {result['similarity']:.2f})")
            # You could print more details here
    
    # Query across knowledge sources
    print("\n\nCross-source query: 'What is relativity?'")
    cross_results = connector.query_across_sources("What is relativity?")
    
    for source, results in cross_results.items():
        print(f"\n{source.capitalize()} results:")
        for i, result in enumerate(results, 1):
            if source == 'wikipedia':
                print(f"  {i}. {result['title']} (Similarity: {result['similarity']:.2f})")
            elif source == 'wordnet':
                print(f"  {i}. {result['data']['name']} (Similarity: {result['similarity']:.2f})")
                print(f"     Definition: {result['data']['definition']}")
            else:
                print(f"  {i}. {result['id']} (Similarity: {result['similarity']:.2f})")

if __name__ == "__main__":
    demonstrate_knowledge_connection()