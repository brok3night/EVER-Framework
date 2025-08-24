# Connecting EVER to Knowledge Datasets

This guide explains how to connect EVER to AI training and knowledge datasets.

## Supported Knowledge Sources

The Knowledge EVR Connector currently supports these sources:

1. **Wikipedia** - Encyclopedia articles
2. **WordNet** - Lexical database of English
3. **Wikidata** - Structured knowledge base
4. **ConceptNet** - Common sense knowledge graph

## Getting Started

### Wikipedia Connection

```python
from ever.core import ConsciousnessCore
from ever.connectors import KnowledgeEVRConnector

# Initialize EVER
consciousness = ConsciousnessCore()
connector = KnowledgeEVRConnector(consciousness)

# Connect to Wikipedia dump
connector.connect_wikipedia('path/to/wikipedia.json')

# Query Wikipedia
results = connector.query_by_text('wikipedia', 'What is quantum physics?')