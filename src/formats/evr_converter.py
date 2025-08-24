"""
EVR Converter - Transforms various data sources into EVER's native .evr format
"""
import os
import json
import zipfile
import tempfile
import numpy as np
import hashlib
from typing import Dict, List, Any, Union, Tuple
import multiprocessing as mp
import logging
import sys

class EVRConverter:
    """Converts various data formats to EVER's native .evr format"""
    
    # Supported input formats with their handlers
    SUPPORTED_FORMATS = {
        # Structured data formats
        'json': {'extension': '.json', 'description': 'JSON data files'},
        'csv': {'extension': '.csv', 'description': 'CSV spreadsheets'},
        'xlsx': {'extension': '.xlsx', 'description': 'Excel workbooks'},
        'xml': {'extension': '.xml', 'description': 'XML documents'},
        'yaml': {'extension': '.yaml', 'description': 'YAML configuration files'},
        'toml': {'extension': '.toml', 'description': 'TOML configuration files'},
        
        # Database formats
        'sql': {'extension': '.sql', 'description': 'SQL database dumps'},
        'sqlite': {'extension': '.db', 'description': 'SQLite databases'},
        
        # Text formats
        'txt': {'extension': '.txt', 'description': 'Plain text files'},
        'md': {'extension': '.md', 'description': 'Markdown documents'},
        'pdf': {'extension': '.pdf', 'description': 'PDF documents (text extraction)'},
        
        # Web formats
        'html': {'extension': '.html', 'description': 'HTML documents'},
        'rss': {'extension': '.xml', 'description': 'RSS feeds'},
        
        # API formats
        'rest': {'extension': None, 'description': 'REST API endpoints'},
        'graphql': {'extension': None, 'description': 'GraphQL endpoints'},
        
        # Graph formats
        'graphdb': {'extension': None, 'description': 'Graph databases (Neo4j, etc.)'},
        'rdf': {'extension': '.rdf', 'description': 'Resource Description Framework'},
        
        # Semantic formats
        'owl': {'extension': '.owl', 'description': 'Web Ontology Language'},
        'ttl': {'extension': '.ttl', 'description': 'Turtle (RDF)'},
        
        # Specialized formats
        'jsonld': {'extension': '.jsonld', 'description': 'JSON-LD (Linked Data)'}
    }
    
    def __init__(self, output_dir: str = None, parallel: bool = True, log_level=logging.INFO):
        """
        Initialize the EVR converter
        
        Args:
            output_dir: Directory to store output files (defaults to current directory)
            parallel: Whether to use parallel processing for large datasets
            log_level: Logging level
        """
        self.output_dir = output_dir or os.getcwd()
        self.parallel = parallel
        self.processors = max(1, mp.cpu_count() - 1) if parallel else 1
        
        # Set up logging
        self.logger = logging.getLogger('EVRConverter')
        self.logger.setLevel(log_level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # EVR format version
        self.format_version = "1.0.0"
        
        # Track conversion metadata
        self.conversion_stats = {}
    
    def convert(self, source_path: str, format_type: str = None, dataset_name: str = None, 
                options: Dict = None) -> str:
        """
        Convert a dataset to EVR format
        
        Args:
            source_path: Path to source data
            format_type: Type of data format (auto-detected if None)
            dataset_name: Name for the dataset (derived from filename if None)
            options: Additional conversion options
            
        Returns:
            Path to the output .evr file
        """
        # Set default options
        options = options or {}
        
        # Auto-detect format if not specified
        if format_type is None:
            format_type = self._detect_format(source_path)
            self.logger.info(f"Auto-detected format: {format_type}")
        
        # Validate format
        if format_type not in self.SUPPORTED_FORMATS:
            supported = ", ".join(self.SUPPORTED_FORMATS.keys())
            raise ValueError(f"Unsupported format: {format_type}. Supported formats: {supported}")
        
        # Set dataset name if not provided
        if not dataset_name:
            dataset_name = os.path.basename(source_path).split('.')[0]
        
        # Create output filename
        output_path = os.path.join(self.output_dir, f"{dataset_name}.evr")
        
        self.logger.info(f"Converting {source_path} ({format_type}) to EVR format: {output_path}")
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Load and process data
            data = self._load_data(source_path, format_type, options)
            
            # Analyze dataset structure
            structure = self._analyze_structure(data, format_type)
            
            # Create metadata
            metadata = {
                'format_version': self.format_version,
                'dataset_name': dataset_name,
                'original_format': format_type,
                'original_path': source_path,
                'structure': structure,
                'schema': options.get('schema', {}),
                'entry_count': self._count_entries(data),
                'creation_timestamp': np.datetime64('now').astype(str),
                'energy_spectrum': {}  # Will be filled during conversion
            }
            
            # Process entities
            entities_dir = os.path.join(temp_dir, 'entities')
            os.makedirs(entities_dir, exist_ok=True)
            
            # Create indexes directory
            indexes_dir = os.path.join(temp_dir, 'indexes')
            os.makedirs(indexes_dir, exist_ok=True)
            
            # Process entities
            if isinstance(data, list):
                energy_map, entity_files = self._process_entity_list(data, entities_dir)
            elif isinstance(data, dict):
                energy_map, entity_files = self._process_entity_dict(data, entities_dir)
            else:
                raise ValueError(f"Unsupported data structure: {type(data)}")
            
            # Update metadata with energy spectrum
            metadata['energy_spectrum'] = self._calculate_energy_spectrum(energy_map)
            metadata['entity_files'] = [os.path.basename(f) for f in entity_files]
            
            # Write metadata file
            with open(os.path.join(temp_dir, 'evr_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Write energy map file
            with open(os.path.join(temp_dir, 'energy_map.json'), 'w') as f:
                json.dump(energy_map, f, indent=2)
            
            # Create energy index files
            self._create_energy_indexes(energy_map, indexes_dir)
            
            # Create the EVR file (zip archive)
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add metadata
                zipf.write(os.path.join(temp_dir, 'evr_metadata.json'), 'evr_metadata.json')
                
                # Add energy map
                zipf.write(os.path.join(temp_dir, 'energy_map.json'), 'energy_map.json')
                
                # Add all entity files
                for entity_file in entity_files:
                    zipf.write(entity_file, os.path.join('entities', os.path.basename(entity_file)))
                
                # Add all index files
                for root, _, files in os.walk(indexes_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.join('indexes', file)
                        zipf.write(file_path, arcname)
            
            self.logger.info(f"Conversion complete. EVR file created: {output_path}")
            
            # Update conversion stats
            self.conversion_stats[dataset_name] = {
                'source': source_path,
                'format': format_type,
                'output': output_path,
                'entity_count': len(entity_files),
                'size': os.path.getsize(output_path)
            }
            
            return output_path
    
    def _detect_format(self, path: str) -> str:
        """Auto-detect format based on file extension or content"""
        # Try to detect by extension first
        ext = os.path.splitext(path)[1].lower()
        
        for format_type, info in self.SUPPORTED_FORMATS.items():
            if info['extension'] == ext:
                return format_type
        
        # If that fails, try to detect by content
        if os.path.exists(path):
            # Check if it's JSON
            try:
                with open(path, 'r') as f:
                    json.load(f)
                return 'json'
            except:
                pass
            
            # Check if it's CSV
            try:
                with open(path, 'r') as f:
                    line = f.readline()
                    if ',' in line and len(line.split(',')) > 1:
                        return 'csv'
            except:
                pass
            
            # Check if it's XML
            try:
                with open(path, 'r') as f:
                    content = f.read(100)
                    if content.strip().startswith('<?xml') or '<' in content and '>' in content:
                        return 'xml'
            except:
                pass
        
        # Default to json if we can't detect
        return 'json'
    
    def _load_data(self, path: str, format_type: str, options: Dict) -> Any:
        """Load data based on format type"""
        # Handler methods for different formats
        handlers = {
            'json': self._load_json,
            'csv': self._load_csv,
            'xlsx': self._load_xlsx,
            'xml': self._load_xml,
            'yaml': self._load_yaml,
            'toml': self._load_toml,
            'sql': self._load_sql,
            'sqlite': self._load_sqlite,
            'txt': self._load_text,
            'md': self._load_markdown,
            'pdf': self._load_pdf,
            'html': self._load_html,
            'rss': self._load_rss,
            'rest': self._load_rest_api,
            'graphql': self._load_graphql,
            'graphdb': self._load_graph_db,
            'rdf': self._load_rdf,
            'owl': self._load_owl,
            'ttl': self._load_turtle,
            'jsonld': self._load_jsonld
        }
        
        # Get appropriate handler
        handler = handlers.get(format_type)
        
        if not handler:
            raise ValueError(f"No handler for format type: {format_type}")
        
        # Call handler with options
        return handler(path, options)
    
    def _load_json(self, path: str, options: Dict) -> Any:
        """Load data from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_csv(self, path: str, options: Dict) -> List[Dict]:
        """Load data from CSV file"""
        import csv
        
        data = []
        with open(path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        
        return data
    
    def _load_xlsx(self, path: str, options: Dict) -> List[Dict]:
        """Load data from Excel file"""
        try:
            import pandas as pd
            df = pd.read_excel(path, sheet_name=options.get('sheet_name', 0))
            return df.to_dict('records')
        except ImportError:
            self.logger.error("Pandas is required for Excel processing. Install with: pip install pandas openpyxl")
            raise
    
    # Implementation stubs for other formats
    # These would be fully implemented in the actual system
    
    def _load_xml(self, path: str, options: Dict) -> Dict:
        """Load data from XML file"""
        try:
            import xmltodict
            with open(path, 'r', encoding='utf-8') as f:
                return xmltodict.parse(f.read())
        except ImportError:
            self.logger.error("xmltodict is required for XML processing. Install with: pip install xmltodict")
            raise
    
    def _load_yaml(self, path: str, options: Dict) -> Dict:
        """Load data from YAML file"""
        try:
            import yaml
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except ImportError:
            self.logger.error("PyYAML is required for YAML processing. Install with: pip install pyyaml")
            raise
    
    def _load_toml(self, path: str, options: Dict) -> Dict:
        """Load data from TOML file"""
        try:
            import toml
            with open(path, 'r', encoding='utf-8') as f:
                return toml.load(f)
        except ImportError:
            self.logger.error("toml is required for TOML processing. Install with: pip install toml")
            raise
    
    def _load_sql(self, path: str, options: Dict) -> List[Dict]:
        """Load data from SQL dump"""
        self.logger.warning("SQL loading not fully implemented")
        return []
    
    def _load_sqlite(self, path: str, options: Dict) -> Dict:
        """Load data from SQLite database"""
        try:
            import sqlite3
            import pandas as pd
            
            conn = sqlite3.connect(path)
            
            # Get all tables
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            data = {}
            for table in tables:
                table_name = table[0]
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                data[table_name] = df.to_dict('records')
            
            conn.close()
            return data
        except ImportError:
            self.logger.error("Pandas is required for SQLite processing. Install with: pip install pandas")
            raise
    
    def _load_text(self, path: str, options: Dict) -> Dict:
        """Load data from text file"""
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into lines and paragraphs
        lines = content.splitlines()
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        return {
            'content': content,
            'lines': lines,
            'paragraphs': paragraphs,
            'word_count': len(content.split())
        }
    
    def _load_markdown(self, path: str, options: Dict) -> Dict:
        """Load data from Markdown file"""
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            import markdown
            html = markdown.markdown(content)
            
            return {
                'content': content,
                'html': html,
                'paragraphs': [p.strip() for p in content.split('\n\n') if p.strip()]
            }
        except ImportError:
            self.logger.warning("markdown package not available. Install with: pip install markdown")
            return self._load_text(path, options)
    
    def _load_pdf(self, path: str, options: Dict) -> Dict:
        """Load data from PDF file"""
        try:
            import PyPDF2
            
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                
                for page in reader.pages:
                    text += page.extract_text() + "\n\n"
            
            return {
                'content': text,
                'pages': [page.extract_text() for page in reader.pages],
                'page_count': len(reader.pages)
            }
        except ImportError:
            self.logger.error("PyPDF2 is required for PDF processing. Install with: pip install PyPDF2")
            raise
    
    # These methods would be more fully implemented in the actual system
    def _load_html(self, path: str, options: Dict) -> Dict:
        """Load data from HTML file"""
        self.logger.warning("HTML loading not fully implemented")
        return {'format': 'html'}
    
    def _load_rss(self, path: str, options: Dict) -> Dict:
        """Load data from RSS feed"""
        self.logger.warning("RSS loading not fully implemented")
        return {'format': 'rss'}
    
    def _load_rest_api(self, path: str, options: Dict) -> Dict:
        """Load data from REST API"""
        self.logger.warning("REST API loading not fully implemented")
        return {'format': 'rest'}
    
    def _load_graphql(self, path: str, options: Dict) -> Dict:
        """Load data from GraphQL endpoint"""
        self.logger.warning("GraphQL loading not fully implemented")
        return {'format': 'graphql'}
    
    def _load_graph_db(self, path: str, options: Dict) -> Dict:
        """Load data from graph database"""
        self.logger.warning("Graph DB loading not fully implemented")
        return {'format': 'graphdb'}
    
    def _load_rdf(self, path: str, options: Dict) -> Dict:
        """Load data from RDF file"""
        self.logger.warning("RDF loading not fully implemented")
        return {'format': 'rdf'}
    
    def _load_owl(self, path: str, options: Dict) -> Dict:
        """Load data from OWL file"""
        self.logger.warning("OWL loading not fully implemented")
        return {'format': 'owl'}
    
    def _load_turtle(self, path: str, options: Dict) -> Dict:
        """Load data from Turtle file"""
        self.logger.warning("Turtle loading not fully implemented")
        return {'format': 'ttl'}
    
    def _load_jsonld(self, path: str, options: Dict) -> Dict:
        """Load data from JSON-LD file"""
        self.logger.warning("JSON-LD loading not fully implemented")
        return {'format': 'jsonld'}
    
    # The rest of the implementation would be similar to the ENDF converter
    # but modified to work with .evr files instead of directories
    
    def _process_entity_list(self, entities: List[Dict], output_dir: str) -> Tuple[Dict, List[str]]:
        """Process a list of entities"""
        energy_map = {}
        entity_files = []
        
        # Process in parallel if enabled
        if self.parallel and len(entities) > 1000:
            # Implementation of parallel processing
            pass
        else:
            # Sequential processing
            for i, entity in enumerate(entities):
                # Generate entity ID if not present
                entity_id = entity.get('id', str(i))
                
                # Convert to EVR format
                evr_entity = self._convert_to_evr(entity)
                
                # Save entity file
                entity_file = os.path.join(output_dir, f"entity_{entity_id}.json")
                with open(entity_file, 'w') as f:
                    json.dump(evr_entity, f)
                
                # Add to energy map
                energy_map[entity_id] = evr_entity['energy_signature']
                entity_files.append(entity_file)
                
                # Log progress
                if (i+1) % 1000 == 0:
                    self.logger.info(f"Processed {i+1}/{len(entities)} entities")
        
        return energy_map, entity_files
    
    def _process_entity_dict(self, data: Dict, output_dir: str) -> Tuple[Dict, List[str]]:
        """Process a dictionary structure"""
        energy_map = {}
        entity_files = []
        
        # Process each top-level key as an entity
        for key, value in data.items():
            # Create entity from key-value pair
            entity = {
                'id': key,
                'value': value
            }
            
            # Convert to EVR format
            evr_entity = self._convert_to_evr(entity)
            
            # Save entity file
            entity_file = os.path.join(output_dir, f"entity_{key}.json")
            with open(entity_file, 'w') as f:
                json.dump(evr_entity, f)
            
            # Add to energy map
            energy_map[key] = evr_entity['energy_signature']
            entity_files.append(entity_file)
        
        return energy_map, entity_files
    
    def _convert_to_evr(self, entity: Dict) -> Dict:
        """Convert an entity to EVR format with energy signatures"""
        # Create EVR entity structure
        evr_entity = {
            'original_data': entity,
            'energy_signature': self._generate_energy_signature(entity),
            'structural_properties': self._extract_structural_properties(entity),
            'connections': self._identify_connections(entity),
            'evr_version': self.format_version
        }
        
        return evr_entity
    
    # Implementation of supporting methods would be similar to the ENDF converter
    
    def _generate_energy_signature(self, entity: Dict) -> Dict:
        """Generate energy signature for an entity"""
        # Simplified implementation - would be more sophisticated in practice
        return {
            'magnitude': {'value': 0.5},
            'frequency': {'value': 0.5},
            'vector': {'value': [0.5, 0.5, 0.5]}
        }
    
    def _extract_structural_properties(self, entity: Dict) -> Dict:
        """Extract structural properties from an entity"""
        # Simplified implementation
        return {'complexity': 0.5}
    
    def _identify_connections(self, entity: Dict) -> Dict:
        """Identify potential connections to other entities"""
        # Simplified implementation
        return {}
    
    def _analyze_structure(self, data: Any, format_type: str) -> Dict:
        """Analyze the structure of the dataset"""
        # Simplified implementation
        return {'type': str(type(data).__name__)}
    
    def _count_entries(self, data: Any) -> int:
        """Count entries in the dataset"""
        if isinstance(data, list):
            return len(data)
        elif isinstance(data, dict):
            return len(data)
        else:
            return 1
    
    def _calculate_energy_spectrum(self, energy_map: Dict) -> Dict:
        """Calculate the energy spectrum across all entities"""
        # Simplified implementation
        return {
            'magnitude': {'min': 0.0, 'max': 1.0, 'avg': 0.5},
            'frequency': {'min': 0.0, 'max': 1.0, 'avg': 0.5}
        }
    
    def _create_energy_indexes(self, energy_map: Dict, indexes_dir: str) -> None:
        """Create energy-based index files for efficient access"""
        # Simplified implementation - would create index files in the indexes_dir
        pass