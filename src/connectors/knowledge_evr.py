"""
Knowledge EVR Connector - Specialized connector for AI training/knowledge datasets
"""
import os
import json
import tempfile
import logging
import numpy as np
from typing import Dict, List, Any, Optional

class KnowledgeEVRConnector:
    """Connects EVER to knowledge datasets like Wikipedia dumps"""
    
    # Supported knowledge sources
    SUPPORTED_SOURCES = {
        'wikipedia': {
            'description': 'Wikipedia article dumps',
            'structure': 'article-based'
        },
        'wikidata': {
            'description': 'Wikidata knowledge graph',
            'structure': 'entity-relation-entity'
        },
        'conceptnet': {
            'description': 'ConceptNet common sense knowledge',
            'structure': 'assertion-based'
        },
        'wordnet': {
            'description': 'WordNet lexical database',
            'structure': 'synset-based'
        }
    }
    
    def __init__(self, consciousness_core):
        """
        Initialize the Knowledge EVR connector
        
        Args:
            consciousness_core: The EVER consciousness core
        """
        self.consciousness = consciousness_core
        
        # Connected knowledge sources
        self.connected_sources = {}
        
        # Energy mapping cache
        self.energy_cache = {}
        self.cache_size = 5000
        
        # Set up logging
        self.logger = logging.getLogger('KnowledgeEVR')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def connect_wikipedia(self, dump_path: str, subset: str = 'all', 
                        max_articles: int = None) -> bool:
        """
        Connect to a Wikipedia dump
        
        Args:
            dump_path: Path to Wikipedia XML dump or processed JSON
            subset: Subset of articles to load ('all', 'featured', etc.)
            max_articles: Maximum number of articles to load (None = all)
        """
        try:
            self.logger.info(f"Connecting to Wikipedia dump: {dump_path}")
            
            # Create connection
            connection = {
                'type': 'wikipedia',
                'path': dump_path,
                'subset': subset,
                'max_articles': max_articles,
                'status': 'initializing',
                'indexed_articles': 0,
                'energy_map': {}
            }
            
            # Determine file type and process accordingly
            if dump_path.endswith('.xml') or dump_path.endswith('.xml.bz2'):
                self._process_wikipedia_xml(connection)
            elif dump_path.endswith('.json'):
                self._process_wikipedia_json(connection)
            else:
                self.logger.error(f"Unsupported Wikipedia dump format: {dump_path}")
                return False
            
            # Register connection
            self.connected_sources['wikipedia'] = connection
            
            self.logger.info(f"Connected to Wikipedia. Indexed {connection['indexed_articles']} articles.")
            return True
            
        except Exception as e:
            self.logger.error(f"Error connecting to Wikipedia: {e}")
            return False
    
    def connect_wikidata(self, dump_path: str, entity_types: List[str] = None) -> bool:
        """
        Connect to a Wikidata dump
        
        Args:
            dump_path: Path to Wikidata JSON dump
            entity_types: Types of entities to load (None = all)
        """
        try:
            self.logger.info(f"Connecting to Wikidata dump: {dump_path}")
            
            # Create connection
            connection = {
                'type': 'wikidata',
                'path': dump_path,
                'entity_types': entity_types,
                'status': 'initializing',
                'indexed_entities': 0,
                'energy_map': {}
            }
            
            # Process Wikidata dump
            self._process_wikidata(connection)
            
            # Register connection
            self.connected_sources['wikidata'] = connection
            
            self.logger.info(f"Connected to Wikidata. Indexed {connection['indexed_entities']} entities.")
            return True
            
        except Exception as e:
            self.logger.error(f"Error connecting to Wikidata: {e}")
            return False
    
    def connect_conceptnet(self, dump_path: str, languages: List[str] = ['en']) -> bool:
        """
        Connect to a ConceptNet dump
        
        Args:
            dump_path: Path to ConceptNet assertions dump
            languages: Languages to include
        """
        try:
            self.logger.info(f"Connecting to ConceptNet dump: {dump_path}")
            
            # Create connection
            connection = {
                'type': 'conceptnet',
                'path': dump_path,
                'languages': languages,
                'status': 'initializing',
                'indexed_assertions': 0,
                'energy_map': {}
            }
            
            # Process ConceptNet dump
            self._process_conceptnet(connection)
            
            # Register connection
            self.connected_sources['conceptnet'] = connection
            
            self.logger.info(f"Connected to ConceptNet. Indexed {connection['indexed_assertions']} assertions.")
            return True
            
        except Exception as e:
            self.logger.error(f"Error connecting to ConceptNet: {e}")
            return False
    
    def connect_wordnet(self) -> bool:
        """
        Connect to WordNet through NLTK
        """
        try:
            import nltk
            from nltk.corpus import wordnet as wn
            
            # Ensure WordNet is downloaded
            try:
                wn.synsets('test')
            except LookupError:
                self.logger.info("Downloading WordNet...")
                nltk.download('wordnet')
            
            self.logger.info("Connecting to WordNet")
            
            # Create connection
            connection = {
                'type': 'wordnet',
                'status': 'initializing',
                'indexed_synsets': 0,
                'energy_map': {}
            }
            
            # Process WordNet
            self._process_wordnet(connection)
            
            # Register connection
            self.connected_sources['wordnet'] = connection
            
            self.logger.info(f"Connected to WordNet. Indexed {connection['indexed_synsets']} synsets.")
            return True
            
        except ImportError:
            self.logger.error("NLTK is required for WordNet. Install with: pip install nltk")
            return False
        except Exception as e:
            self.logger.error(f"Error connecting to WordNet: {e}")
            return False
    
    def _process_wikipedia_xml(self, connection: Dict) -> None:
        """Process Wikipedia XML dump"""
        # In a real implementation, this would:
        # 1. Parse the Wikipedia XML dump (possibly using mwparserfromhell)
        # 2. Extract articles and convert to energy signatures
        # 3. Build energy map and indexes
        
        # Simplified implementation for demonstration
        self.logger.info("Processing Wikipedia XML dump (not fully implemented)")
        
        # Set connection status
        connection['status'] = 'connected'
        connection['indexed_articles'] = 0
        
        # Note: In a real implementation, we would process the actual dump
    
    def _process_wikipedia_json(self, connection: Dict) -> None:
        """Process Wikipedia JSON dump"""
        # Load JSON dump
        with open(connection['path'], 'r') as f:
            articles = json.load(f)
        
        # Apply max articles limit if specified
        max_articles = connection['max_articles']
        if max_articles:
            articles = articles[:max_articles]
        
        # Process each article
        for i, article in enumerate(articles):
            # Convert to energy signature
            article_id = article.get('id', str(i))
            energy_signature = self._wikipedia_article_to_energy(article)
            
            # Add to energy map
            connection['energy_map'][article_id] = {
                'title': article.get('title', ''),
                'energy': energy_signature
            }
        
        # Update connection stats
        connection['indexed_articles'] = len(connection['energy_map'])
        connection['status'] = 'connected'
    
    def _wikipedia_article_to_energy(self, article: Dict) -> Dict:
        """Convert Wikipedia article to energy signature"""
        # Extract article properties
        title = article.get('title', '')
        text = article.get('text', '')
        categories = article.get('categories', [])
        links = article.get('links', [])
        
        # Create base energy signature
        energy = {
            'magnitude': {
                'value': min(1.0, 0.3 + 0.001 * len(text))  # Larger articles = higher magnitude
            },
            'frequency': {
                'value': min(1.0, 0.2 + 0.01 * len(links))  # More links = higher frequency
            },
            'vector': {
                'value': [0.5, 0.5, 0.5]  # Neutral default
            },
            'entropy': {
                'value': 0.5  # Default entropy
            }
        }
        
        # Adjust vector based on categories
        if categories:
            # Simple category-based adjustments
            science_categories = ['science', 'physics', 'chemistry', 'biology', 'mathematics']
            arts_categories = ['art', 'music', 'literature', 'film', 'culture']
            history_categories = ['history', 'historical', 'ancient', 'medieval', 'century']
            
            # Convert categories to lowercase for matching
            categories_lower = [cat.lower() for cat in categories]
            
            # Adjust x-component: science vs. arts
            science_count = sum(1 for cat in categories_lower if any(sc in cat for sc in science_categories))
            arts_count = sum(1 for cat in categories_lower if any(ac in cat for ac in arts_categories))
            
            if science_count > arts_count:
                energy['vector']['value'][0] = 0.7  # More scientific
            elif arts_count > science_count:
                energy['vector']['value'][0] = 0.3  # More artistic
            
            # Adjust y-component: concrete vs. abstract
            concrete_words = ['physical', 'specific', 'concrete', 'particular']
            abstract_words = ['concept', 'theory', 'philosophy', 'abstract']
            
            concrete_count = sum(1 for word in concrete_words if word in text.lower())
            abstract_count = sum(1 for word in abstract_words if word in text.lower())
            
            if concrete_count > abstract_count:
                energy['vector']['value'][1] = 0.3  # More concrete
            elif abstract_count > concrete_count:
                energy['vector']['value'][1] = 0.7  # More abstract
            
            # Adjust z-component: historical vs. contemporary
            history_count = sum(1 for cat in categories_lower if any(hc in cat for hc in history_categories))
            
            if history_count > 0:
                energy['vector']['value'][2] = 0.3  # More historical
            else:
                energy['vector']['value'][2] = 0.7  # More contemporary
        
        return energy
    
    def _process_wikidata(self, connection: Dict) -> None:
        """Process Wikidata dump"""
        # Simplified implementation
        self.logger.info("Processing Wikidata dump (not fully implemented)")
        
        # Set connection status
        connection['status'] = 'connected'
        connection['indexed_entities'] = 0
    
    def _process_conceptnet(self, connection: Dict) -> None:
        """Process ConceptNet dump"""
        # Simplified implementation
        self.logger.info("Processing ConceptNet dump (not fully implemented)")
        
        # Set connection status
        connection['status'] = 'connected'
        connection['indexed_assertions'] = 0
    
    def _process_wordnet(self, connection: Dict) -> None:
        """Process WordNet"""
        try:
            from nltk.corpus import wordnet as wn
            
            # Get all synsets
            all_synsets = list(wn.all_synsets())
            
            # Process each synset
            for synset in all_synsets:
                # Convert to energy signature
                synset_id = synset.name()
                energy_signature = self._wordnet_synset_to_energy(synset)
                
                # Add to energy map
                connection['energy_map'][synset_id] = {
                    'name': synset_id,
                    'pos': synset.pos(),
                    'definition': synset.definition(),
                    'energy': energy_signature
                }
            
            # Update connection stats
            connection['indexed_synsets'] = len(connection['energy_map'])
            connection['status'] = 'connected'
            
        except Exception as e:
            self.logger.error(f"Error processing WordNet: {e}")
            connection['status'] = 'error'
    
    def _wordnet_synset_to_energy(self, synset) -> Dict:
        """Convert WordNet synset to energy signature"""
        # Extract synset properties
        name = synset.name()
        definition = synset.definition()
        examples = synset.examples()
        hypernyms = synset.hypernyms()
        hyponyms = synset.hyponyms()
        
        # Create energy signature
        energy = {
            'magnitude': {
                'value': min(1.0, 0.3 + 0.1 * len(examples) + 0.1 * len(hypernyms) + 0.1 * len(hyponyms))
            },
            'frequency': {
                'value': min(1.0, 0.2 + 0.2 * len(examples))
            },
            'entropy': {
                'value': min(1.0, 0.3 + 0.1 * len(hyponyms) / max(1, len(hypernyms)))
            },
            'vector': {
                'value': [
                    0.5,  # x component (neutral)
                    0.2 + 0.1 * len(hypernyms) - 0.1 * len(hyponyms),  # y component (higher = more abstract)
                    0.5   # z component (neutral)
                ]
            }
        }
        
        return energy
    
    def query_by_text(self, source: str, text: str, limit: int = 10) -> List[Dict]:
        """
        Query knowledge source using text as input
        
        Args:
            source: Knowledge source to query ('wikipedia', 'wikidata', etc.)
            text: Text to query
            limit: Maximum number of results to return
        """
        # Check if source is connected
        if source not in self.connected_sources:
            self.logger.error(f"Source {source} not connected")
            return []
        
        # Generate energy signature from text
        energy_signature = self._text_to_energy(text)
        
        # Query by energy
        return self.query_by_energy(source, energy_signature, limit=limit)
    
    def query_by_energy(self, source: str, energy_signature: Dict, 
                       similarity_threshold: float = 0.7, limit: int = 10) -> List[Dict]:
        """
        Query knowledge source using energy signature
        
        Args:
            source: Knowledge source to query
            energy_signature: Energy signature to match
            similarity_threshold: Minimum similarity threshold
            limit: Maximum number of results to return
        """
        # Check if source is connected
        if source not in self.connected_sources:
            self.logger.error(f"Source {source} not connected")
            return []
        
        connection = self.connected_sources[source]
        
        # Match against energy map
        results = []
        
        for entity_id, entity_data in connection['energy_map'].items():
            entity_energy = entity_data.get('energy', {})
            
            # Calculate similarity
            similarity = self._calculate_energy_similarity(energy_signature, entity_energy)
            
            # Add to results if above threshold
            if similarity >= similarity_threshold:
                result = {
                    'id': entity_id,
                    'similarity': similarity,
                    'data': entity_data
                }
                
                # Add source-specific fields
                if source == 'wikipedia':
                    result['title'] = entity_data.get('title', '')
                elif source == 'wordnet':
                    result['definition'] = entity_data.get('definition', '')
                    result['pos'] = entity_data.get('pos', '')
                
                results.append(result)
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Apply limit
        if limit > 0:
            results = results[:limit]
        
        return results
    
    def _text_to_energy(self, text: str) -> Dict:
        """Convert text to energy signature"""
        # Cache lookup
        if text in self.energy_cache:
            return self.energy_cache[text]
        
        # Simple text analysis for energy
        words = text.split()
        word_count = len(words)
        unique_words = len(set(words))
        
        # Basic energy signature
        energy = {
            'magnitude': {
                'value': min(1.0, 0.3 + 0.01 * word_count)
            },
            'frequency': {
                'value': min(1.0, 0.5 * (unique_words / max(1, word_count)))
            },
            'entropy': {
                'value': min(1.0, 0.3 + 0.7 * (unique_words / max(1, word_count)))
            },
            'vector': {
                'value': [0.5, 0.5, 0.5]  # Default neutral vector
            }
        }
        
        # Check for question words
        question_words = ['what', 'who', 'where', 'when', 'why', 'how']
        if any(text.lower().startswith(qw) for qw in question_words):
            energy['vector']['value'][1] = 0.7  # Questions have higher y-component
        
        # Cache result
        if len(self.energy_cache) >= self.cache_size:
            # Remove a random key if cache is full
            self.energy_cache.pop(next(iter(self.energy_cache)))
        
        self.energy_cache[text] = energy
        
        return energy
    
    def _calculate_energy_similarity(self, sig1: Dict, sig2: Dict) -> float:
        """Calculate similarity between energy signatures"""
        # Find common properties
        common_props = set(sig1.keys()) & set(sig2.keys())
        
        if not common_props:
            return 0.0
        
        similarity_sum = 0.0
        weight_sum = 0.0
        
        # Property weights
        weights = {
            'magnitude': 1.0,
            'frequency': 0.8,
            'entropy': 0.7,
            'vector': 1.2
        }
        
        for prop in common_props:
            # Skip properties without values
            if 'value' not in sig1.get(prop, {}) or 'value' not in sig2.get(prop, {}):
                continue
                
            val1 = sig1[prop]['value']
            val2 = sig2[prop]['value']
            weight = weights.get(prop, 1.0)
            
            # Calculate property similarity
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Scalar similarity
                diff = 1.0 - abs(val1 - val2) / max(1.0, abs(val1) + abs(val2))
                similarity_sum += diff * weight
                weight_sum += weight
            elif isinstance(val1, list) and isinstance(val2, list) and len(val1) == len(val2):
                # Vector similarity
                magnitude1 = sum(x**2 for x in val1)**0.5
                magnitude2 = sum(x**2 for x in val2)**0.5
                
                if magnitude1 > 0 and magnitude2 > 0:
                    dot_product = sum(x*y for x, y in zip(val1, val2))
                    cosine = dot_product / (magnitude1 * magnitude2)
                    sim = (cosine + 1) / 2  # Convert from [-1,1] to [0,1]
                    similarity_sum += sim * weight
                    weight_sum += weight
        
        # Return weighted average similarity
        return similarity_sum / weight_sum if weight_sum > 0 else 0.0
    
    def query_across_sources(self, text: str, sources: List[str] = None, 
                            limit_per_source: int = 5) -> Dict[str, List[Dict]]:
        """
        Query across multiple knowledge sources
        
        Args:
            text: Text to query
            sources: Sources to query (None = all connected)
            limit_per_source: Maximum results per source
        """
        # Use all connected sources if none specified
        if sources is None:
            sources = list(self.connected_sources.keys())
        else:
            # Filter to only connected sources
            sources = [s for s in sources if s in self.connected_sources]
        
        if not sources:
            self.logger.error("No valid sources to query")
            return {}
        
        # Generate energy signature from text
        energy_signature = self._text_to_energy(text)
        
        # Query each source
        results = {}
        
        for source in sources:
            source_results = self.query_by_energy(
                source, 
                energy_signature,
                similarity_threshold=0.6,  # Lower threshold for cross-source queries
                limit=limit_per_source
            )
            
            results[source] = source_results
        
        return results
    
    def create_sample_wikipedia(self, output_path: str, article_count: int = 20) -> str:
        """
        Create a sample Wikipedia dataset for testing
        
        Args:
            output_path: Path to save the sample dataset
            article_count: Number of sample articles to create
        """
        # Sample articles
        articles = []
        
        # Science articles
        science_articles = [
            {
                'id': 'physics_1',
                'title': 'Quantum Mechanics',
                'text': 'Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles.',
                'categories': ['Physics', 'Quantum mechanics', 'Science']
            },
            {
                'id': 'physics_2',
                'title': 'Theory of Relativity',
                'text': 'The theory of relativity usually encompasses two interrelated theories by Albert Einstein: special relativity and general relativity.',
                'categories': ['Physics', 'Relativity', 'Albert Einstein', 'Science']
            },
            {
                'id': 'biology_1',
                'title': 'DNA',
                'text': 'Deoxyribonucleic acid is a molecule composed of two polynucleotide chains that coil around each other to form a double helix carrying genetic instructions.',
                'categories': ['Biology', 'Genetics', 'Molecular biology', 'Science']
            }
        ]
        
        # History articles
        history_articles = [
            {
                'id': 'history_1',
                'title': 'World War II',
                'text': 'World War II was a global war that lasted from 1939 to 1945. It involved the vast majority of the world\'s countries.',
                'categories': ['History', '20th century', 'World War II', 'Military history']
            },
            {
                'id': 'history_2',
                'title': 'Ancient Egypt',
                'text': 'Ancient Egypt was a civilization of ancient North Africa, concentrated along the lower reaches of the Nile River.',
                'categories': ['History', 'Ancient Egypt', 'Ancient civilizations', 'Africa']
            }
        ]
        
        # Arts articles
        arts_articles = [
            {
                'id': 'art_1',
                'title': 'Mona Lisa',
                'text': 'The Mona Lisa is a half-length portrait painting by Italian artist Leonardo da Vinci.',
                'categories': ['Art', 'Leonardo da Vinci', 'Renaissance', 'Paintings']
            },
            {
                'id': 'music_1',
                'title': 'Ludwig van Beethoven',
                'text': 'Ludwig van Beethoven was a German composer and pianist whose music ranks amongst the most performed of the classical music repertoire.',
                'categories': ['Music', 'Composers', 'Classical music', 'German musicians']
            }
        ]
        
        # Combine and extend to desired count
        all_categories = [science_articles, history_articles, arts_articles]
        
        while len(articles) < article_count:
            for category in all_categories:
                for article in category:
                    if article not in articles:
                        articles.append(article)
                        if len(articles) >= article_count:
                            break
                if len(articles) >= article_count:
                    break
            
            # If we've used all articles but still need more, create variations
            if len(articles) < article_count and len(articles) == sum(len(c) for c in all_categories):
                # Create variations of existing articles
                for i, article in enumerate(articles.copy()):
                    if len(articles) >= article_count:
                        break
                    
                    variation = article.copy()
                    variation['id'] = f"{article['id']}_var{i}"
                    variation['title'] = f"{article['title']} (Variation)"
                    articles.append(variation)
        
        # Save to output path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(articles, f, indent=2)
        
        self.logger.info(f"Created sample Wikipedia dataset with {len(articles)} articles at {output_path}")
        return output_path