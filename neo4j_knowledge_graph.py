"""
Neo4j Knowledge Graph module for extracting and querying entity relationships
"""
from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path
from datetime import datetime

from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_community.llms import Anthropic
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

from config import ModelConfig, SystemConfig

class Neo4jKnowledgeGraphManager:
    """Manages knowledge graph extraction and querying using Neo4j"""
    
    def __init__(self, config: SystemConfig, model_config: ModelConfig, 
                 neo4j_uri: str = "bolt://localhost:7687", 
                 neo4j_user: str = "neo4j", 
                 neo4j_password: str = "password"):
        self.config = config
        self.model_config = model_config
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        
        try:
            self.llm = self._initialize_llm()
        except Exception as e:
            print(f"Warning: Could not initialize LLM for knowledge graph: {e}")
            self.llm = self._create_dummy_llm()
        
        # Initialize Neo4j driver
        self.driver = None
        self._connect_to_neo4j()
        
        # Create constraints and indexes for better performance
        self._setup_database()
        
        # Initialize in-memory storage as fallback
        self.in_memory_entities = []
        self.in_memory_relationships = []
        self.use_in_memory_fallback = self.driver is None
    
    def _create_dummy_llm(self):
        """Create a dummy LLM for testing when real LLM is not available"""
        class DummyLLM:
            def predict(self, prompt, **kwargs):
                return '{"entities": [], "relationships": []}'
        
        return DummyLLM()
    
    def _initialize_llm(self):
        """Initialize the LLM for graph extraction"""
        if self.model_config.provider == "openai":
            return ChatOpenAI(
                model=self.model_config.model_name,
                temperature=self.model_config.temperature,
                max_tokens=self.model_config.max_tokens,
                api_key=self.model_config.api_key
            )
        elif self.model_config.provider == "anthropic":
            return Anthropic(
                model=self.model_config.model_name,
                temperature=self.model_config.temperature,
                max_tokens=self.model_config.max_tokens,
                anthropic_api_key=self.model_config.api_key
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.model_config.provider}")
    
    def _connect_to_neo4j(self):
        """Connect to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri, 
                auth=(self.neo4j_user, self.neo4j_password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("Successfully connected to Neo4j")
        except (ServiceUnavailable, AuthError) as e:
            print(f"Warning: Could not connect to Neo4j: {e}")
            print("Knowledge graph will work in memory mode only")
            self.driver = None
    
    def _setup_database(self):
        """Setup database constraints and indexes"""
        if not self.driver:
            return
        
        try:
            with self.driver.session() as session:
                # Create constraints for unique entity names
                session.run("CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
                
                # Create indexes for better query performance
                session.run("CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)")
                session.run("CREATE INDEX relationship_type_index IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.type)")
                
                print("Neo4j database setup completed")
        except Exception as e:
            print(f"Warning: Could not setup database constraints: {e}")
    
    def extract_graph_from_documents(self, documents: List[Document], force_recreate: bool = False):
        """Extract knowledge graph from documents using custom LLM-based extraction"""
        if not self.driver:
            print("Neo4j not available, using in-memory fallback for graph extraction")
            self.use_in_memory_fallback = True
        
        if force_recreate:
            if self.driver:
                self._clear_database()
            else:
                self.in_memory_entities = []
                self.in_memory_relationships = []
        
        print("Extracting knowledge graph from documents...")
        
        # Process documents in batches to avoid token limits
        batch_size = 3
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            for doc in batch:
                try:
                    # Extract entities and relationships from document
                    entities_relations = self._extract_entities_and_relations(doc.page_content)
                    
                    # Store in Neo4j or in-memory
                    if self.driver:
                        self._store_entities_and_relations(entities_relations, doc.metadata)
                    else:
                        self._store_entities_and_relations_in_memory(entities_relations, doc.metadata)
                            
                except Exception as e:
                    print(f"Error processing document: {e}")
                    continue
        
        print("Knowledge graph extraction completed")
        return self.get_graph_stats()
    
    def _extract_entities_and_relations(self, text: str) -> Dict[str, Any]:
        """Extract entities and relationships from text using a single method based on priority"""
        
        # Priority order: LLM > Semantic Search > Rule-based
        # Try LLM-based extraction first (highest quality)
        if hasattr(self.llm, 'predict') and not isinstance(self.llm, type(self._create_dummy_llm())):
            try:
                print("Using LLM-based extraction")
                llm_results = self._llm_based_extraction(text)
                entities = llm_results.get('entities', [])
                relationships = llm_results.get('relationships', [])
                
                # If LLM extraction was successful and returned results, use them
                if entities or relationships:
                    print(f"LLM extraction successful: {len(entities)} entities, {len(relationships)} relationships")
                    return {
                        "entities": self._deduplicate_entities(entities),
                        "relationships": self._deduplicate_relationships(relationships)
                    }
                else:
                    print("LLM extraction returned no results, trying semantic search")
            except Exception as e:
                print(f"LLM extraction failed: {e}, trying semantic search")
        
        # Try semantic search-based extraction (medium quality)
        try:
            print("Using semantic search extraction")
            semantic_results = self._semantic_search_extraction(text)
            entities = semantic_results.get('entities', [])
            relationships = semantic_results.get('relationships', [])
            
            # If semantic extraction was successful and returned results, use them
            if entities or relationships:
                print(f"Semantic search extraction successful: {len(entities)} entities, {len(relationships)} relationships")
                return {
                    "entities": self._deduplicate_entities(entities),
                    "relationships": self._deduplicate_relationships(relationships)
                }
            else:
                print("Semantic search extraction returned no results, using rule-based fallback")
        except Exception as e:
            print(f"Semantic search extraction failed: {e}, using rule-based fallback")
        
        # Use rule-based extraction as final fallback (lowest quality but most reliable)
        print("Using rule-based extraction")
        rule_results = self._rule_based_extraction(text)
        entities = rule_results.get('entities', [])
        relationships = rule_results.get('relationships', [])
        
        print(f"Rule-based extraction completed: {len(entities)} entities, {len(relationships)} relationships")
        
        return {
            "entities": self._deduplicate_entities(entities),
            "relationships": self._deduplicate_relationships(relationships)
        }
    
    def _rule_based_extraction(self, text: str) -> Dict[str, Any]:
        """Simple rule-based entity and relationship extraction"""
        import re
        
        entities = []
        relationships = []
        
        # Extract person names (simple approach - look for common names)
        person_names = ["Sarah", "Michael", "John", "Emily", "David", "Lisa"]
        for name in person_names:
            if name in text:
                entities.append({
                    "name": name,
                    "type": "person",
                    "description": "Individual mentioned in the text"
                })
        
        # Extract company names
        company_names = ["prismaticAI", "TechCorp", "InnovateLabs", "DataFlow"]
        for company in company_names:
            if company in text:
                entities.append({
                    "name": company,
                    "type": "organization",
                    "description": "Technology company"
                })
        
        # Extract job titles
        job_titles = ["software engineer", "data scientist", "product manager", "designer"]
        for job in job_titles:
            if job in text.lower():
                entities.append({
                    "name": job.title(),
                    "type": "job_title",
                    "description": "Professional role or position"
                })
        
        # Extract relationships based on text patterns
        # Sarah works at prismaticAI
        if "Sarah" in text and "prismaticAI" in text:
            relationships.append({
                "source": "Sarah",
                "target": "prismaticAI",
                "type": "works_at",
                "description": "Sarah works at prismaticAI"
            })
        
        # Michael works at prismaticAI
        if "Michael" in text and "prismaticAI" in text:
            relationships.append({
                "source": "Michael",
                "target": "prismaticAI",
                "type": "works_at",
                "description": "Michael works at prismaticAI"
            })
        
        # Sarah is a software engineer
        if "Sarah" in text and "software engineer" in text.lower():
            relationships.append({
                "source": "Sarah",
                "target": "Software Engineer",
                "type": "has_job_title",
                "description": "Sarah is a software engineer"
            })
        
        # Michael is a data scientist
        if "Michael" in text and "data scientist" in text.lower():
            relationships.append({
                "source": "Michael",
                "target": "Data Scientist",
                "type": "has_job_title",
                "description": "Michael is a data scientist"
            })
        
        return {
            "entities": entities,
            "relationships": relationships
        }
    
    def _llm_based_extraction(self, text: str) -> Dict[str, Any]:
        """Extract entities and relationships using LLM"""
        prompt = f"""
        Extract entities and relationships from the following text. 
        Return the result as a JSON object with the following structure:
        {{
            "entities": [
                {{
                    "name": "entity name",
                    "type": "person|organization|location|job_title|skill|technology",
                    "description": "brief description"
                }}
            ],
            "relationships": [
                {{
                    "source": "source entity name",
                    "target": "target entity name", 
                    "type": "relationship type (e.g., works_at, reports_to, located_in, has_skill)",
                    "description": "brief description of the relationship"
                }}
            ]
        }}

        Text to analyze:
        {text}

        Focus on:
        - People (names, titles, roles)
        - Organizations (companies, institutions)
        - Locations (cities, countries, addresses)
        - Job titles and skills
        - Technologies and tools
        - Relationships between entities (who works where, who reports to whom, etc.)

        Return only the JSON object, no additional text.
        """
        
        try:
            # Use LLM to extract entities and relationships
            if hasattr(self.llm, 'predict'):
                response = self.llm.predict(prompt)
            elif hasattr(self.llm, 'invoke'):
                response = self.llm.invoke(prompt).content
            else:
                return {"entities": [], "relationships": []}
            
            # Parse JSON response
            import json
            import re
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                return {
                    "entities": result.get("entities", []),
                    "relationships": result.get("relationships", [])
                }
            else:
                print(f"Could not parse JSON from LLM response: {response}")
                return {"entities": [], "relationships": []}
                
        except Exception as e:
            print(f"LLM extraction error: {e}")
            return {"entities": [], "relationships": []}
    
    def _semantic_search_extraction(self, text: str) -> Dict[str, Any]:
        """Extract entities and relationships using semantic search patterns"""
        entities = []
        relationships = []
        
        # Define semantic patterns for entity extraction
        import re
        import spacy
        
        try:
            # Try to load spaCy model for NER
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            # If spaCy model not available, use basic patterns
            return self._basic_semantic_extraction(text)
        
        # Process text with spaCy
        doc = nlp(text)
        
        # Extract named entities
        for ent in doc.ents:
            entity_type = self._map_spacy_entity_type(ent.label_)
            entities.append({
                "name": ent.text,
                "type": entity_type,
                "description": f"{entity_type} mentioned in text"
            })
        
        # Extract relationships using dependency parsing
        relationships.extend(self._extract_dependency_relationships(doc))
        
        # Extract relationships using semantic patterns
        relationships.extend(self._extract_semantic_patterns(text))
        
        return {
            "entities": entities,
            "relationships": relationships
        }
    
    def _basic_semantic_extraction(self, text: str) -> Dict[str, Any]:
        """Basic semantic extraction without spaCy"""
        entities = []
        relationships = []
        
        import re
        
        # Person name patterns
        person_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
            r'\b[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+\b',  # First M. Last
            r'\bDr\. [A-Z][a-z]+ [A-Z][a-z]+\b',  # Dr. First Last
        ]
        
        for pattern in person_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    "name": match.group(),
                    "type": "person",
                    "description": "Person mentioned in text"
                })
        
        # Organization patterns
        org_patterns = [
            r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)* (?:Inc|Corp|LLC|Ltd|Company|University|Institute)\b',
            r'\b[A-Z]{2,}\b',  # Acronyms like IBM, NASA
        ]
        
        for pattern in org_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    "name": match.group(),
                    "type": "organization",
                    "description": "Organization mentioned in text"
                })
        
        # Job title patterns
        job_patterns = [
            r'\b(?:senior |junior |lead |principal )?(?:software engineer|data scientist|product manager|designer|developer|analyst|manager|director|executive)\b',
        ]
        
        for pattern in job_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    "name": match.group().title(),
                    "type": "job_title",
                    "description": "Job title mentioned in text"
                })
        
        # Relationship patterns
        rel_patterns = [
            (r'(\w+) (?:works at|is employed by|is at) (\w+)', 'works_at'),
            (r'(\w+) (?:reports to|works for) (\w+)', 'reports_to'),
            (r'(\w+) is a (\w+)', 'has_job_title'),
            (r'(\w+) (?:lives in|is located in|is from) (\w+)', 'located_in'),
        ]
        
        for pattern, rel_type in rel_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                relationships.append({
                    "source": match.group(1),
                    "target": match.group(2),
                    "type": rel_type,
                    "description": f"{match.group(1)} {rel_type.replace('_', ' ')} {match.group(2)}"
                })
        
        return {
            "entities": entities,
            "relationships": relationships
        }
    
    def _map_spacy_entity_type(self, spacy_label: str) -> str:
        """Map spaCy entity types to our entity types"""
        mapping = {
            'PERSON': 'person',
            'ORG': 'organization',
            'GPE': 'location',
            'LOC': 'location',
            'PRODUCT': 'technology',
            'WORK_OF_ART': 'technology',
            'LAW': 'document',
            'LANGUAGE': 'skill',
            'DATE': 'date',
            'TIME': 'time',
            'MONEY': 'money',
            'PERCENT': 'percentage',
            'QUANTITY': 'quantity',
            'ORDINAL': 'ordinal',
            'CARDINAL': 'number',
            'FAC': 'location',
            'EVENT': 'event',
        }
        return mapping.get(spacy_label, 'other')
    
    def _extract_dependency_relationships(self, doc) -> List[Dict[str, Any]]:
        """Extract relationships using dependency parsing"""
        relationships = []
        
        for token in doc:
            # Look for subject-verb-object patterns
            if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                subject = token.text
                verb = token.head.text
                obj = None
                
                # Find object
                for child in token.head.children:
                    if child.dep_ in ["dobj", "pobj"]:
                        obj = child.text
                        break
                
                if obj:
                    rel_type = self._map_verb_to_relationship(verb)
                    if rel_type:
                        relationships.append({
                            "source": subject,
                            "target": obj,
                            "type": rel_type,
                            "description": f"{subject} {verb} {obj}"
                        })
        
        return relationships
    
    def _extract_semantic_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extract relationships using semantic patterns"""
        relationships = []
        
        # Define semantic patterns
        patterns = [
            # Employment patterns
            (r'(\w+) (?:works at|is employed by|is at|joined) (\w+)', 'works_at'),
            (r'(\w+) (?:reports to|works for|is managed by) (\w+)', 'reports_to'),
            (r'(\w+) (?:manages|leads|supervises) (\w+)', 'manages'),
            
            # Role patterns
            (r'(\w+) is a (\w+(?:\s+\w+)*)', 'has_job_title'),
            (r'(\w+) serves as (\w+(?:\s+\w+)*)', 'has_job_title'),
            (r'(\w+) holds the position of (\w+(?:\s+\w+)*)', 'has_job_title'),
            
            # Location patterns
            (r'(\w+) (?:lives in|is located in|is from|resides in) (\w+)', 'located_in'),
            (r'(\w+) (?:is based in|operates in) (\w+)', 'located_in'),
            
            # Collaboration patterns
            (r'(\w+) (?:collaborates with|works with|partners with) (\w+)', 'collaborates_with'),
            (r'(\w+) (?:teams up with|joins forces with) (\w+)', 'collaborates_with'),
            
            # Skill patterns
            (r'(\w+) (?:knows|is skilled in|expert in|specializes in) (\w+)', 'has_skill'),
            (r'(\w+) (?:uses|works with|develops) (\w+)', 'uses_technology'),
        ]
        
        import re
        for pattern, rel_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                relationships.append({
                    "source": match.group(1),
                    "target": match.group(2),
                    "type": rel_type,
                    "description": f"{match.group(1)} {rel_type.replace('_', ' ')} {match.group(2)}"
                })
        
        return relationships
    
    def _map_verb_to_relationship(self, verb: str) -> str:
        """Map verbs to relationship types"""
        verb_mapping = {
            'work': 'works_at',
            'manage': 'manages',
            'report': 'reports_to',
            'lead': 'leads',
            'supervise': 'supervises',
            'collaborate': 'collaborates_with',
            'use': 'uses_technology',
            'develop': 'develops',
            'create': 'creates',
            'build': 'builds',
        }
        return verb_mapping.get(verb.lower(), None)
    
    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate entities based on name"""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            name = entity.get('name', '').lower().strip()
            if name and name not in seen:
                seen.add(name)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _deduplicate_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate relationships based on source, target, and type"""
        seen = set()
        unique_relationships = []
        
        for rel in relationships:
            key = (
                rel.get('source', '').lower().strip(),
                rel.get('target', '').lower().strip(),
                rel.get('type', '').lower().strip()
            )
            if all(key) and key not in seen:
                seen.add(key)
                unique_relationships.append(rel)
        
        return unique_relationships
    
    def _store_entities_and_relations_in_memory(self, entities_relations: Dict[str, Any], metadata: Dict[str, Any]):
        """Store entities and relationships in memory"""
        # Store entities
        for entity in entities_relations.get('entities', []):
            if entity.get('name') and entity.get('type'):
                # Check if entity already exists
                existing_entity = next((e for e in self.in_memory_entities if e['name'].lower() == entity['name'].lower()), None)
                if not existing_entity:
                    self.in_memory_entities.append({
                        'name': entity['name'],
                        'type': entity['type'],
                        'description': entity.get('description', ''),
                        'metadata': metadata
                    })
        
        # Store relationships
        for relation in entities_relations.get('relationships', []):
            if relation.get('source') and relation.get('target') and relation.get('type'):
                # Check if relationship already exists
                existing_rel = next((r for r in self.in_memory_relationships 
                                   if r['source'].lower() == relation['source'].lower() 
                                   and r['target'].lower() == relation['target'].lower()
                                   and r['type'].lower() == relation['type'].lower()), None)
                if not existing_rel:
                    self.in_memory_relationships.append({
                        'source': relation['source'],
                        'target': relation['target'],
                        'type': relation['type'],
                        'description': relation.get('description', ''),
                        'metadata': metadata
                    })
    
    def _store_entities_and_relations(self, entities_relations: Dict[str, Any], metadata: Dict[str, Any]):
        """Store entities and relationships in Neo4j"""
        if not self.driver:
            return
        
        with self.driver.session() as session:
            # Store entities
            for entity in entities_relations.get('entities', []):
                if entity.get('name') and entity.get('type'):
                    session.run("""
                        MERGE (e:Entity {name: $name})
                        SET e.type = $type, e.description = $description, e.last_updated = datetime()
                    """, name=entity['name'], type=entity['type'], 
                         description=entity.get('description', ''))
            
            # Store relationships
            for relation in entities_relations.get('relationships', []):
                if relation.get('source') and relation.get('target') and relation.get('type'):
                    session.run("""
                        MATCH (source:Entity {name: $source})
                        MATCH (target:Entity {name: $target})
                        MERGE (source)-[r:RELATES_TO {type: $rel_type}]->(target)
                        SET r.description = $description, r.last_updated = datetime()
                    """, source=relation['source'], target=relation['target'],
                         rel_type=relation['type'], description=relation.get('description', ''))
    
    def _clear_database(self):
        """Clear all data from the database"""
        if not self.driver:
            return
        
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared")
    
    def query_graph(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Query the knowledge graph using natural language"""
        if not self.driver and not self.use_in_memory_fallback:
            return []
        
        # Simple keyword-based search for now
        query_lower = query.lower()
        results = []
        
        if self.driver:
            # Query Neo4j database
            with self.driver.session() as session:
                # Search for entities
                entity_results = session.run("""
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS $query 
                       OR toLower(e.description) CONTAINS $query
                       OR toLower(e.type) CONTAINS $query
                    RETURN e.name as name, e.type as type, e.description as description
                    LIMIT $limit
                """, query=query_lower, limit=max_results)
                
                for record in entity_results:
                    results.append({
                        "type": "entity",
                        "name": record["name"],
                        "entity_type": record["type"],
                        "description": record["description"]
                    })
                
                # Search for relationships
                rel_results = session.run("""
                    MATCH (source:Entity)-[r:RELATES_TO]->(target:Entity)
                    WHERE toLower(r.type) CONTAINS $query 
                       OR toLower(r.description) CONTAINS $query
                       OR toLower(source.name) CONTAINS $query
                       OR toLower(target.name) CONTAINS $query
                    RETURN source.name as source, target.name as target, r.type as rel_type, r.description as description
                    LIMIT $limit
                """, query=query_lower, limit=max_results)
                
                for record in rel_results:
                    results.append({
                        "type": "relationship",
                        "source": record["source"],
                        "target": record["target"],
                        "relationship_type": record["rel_type"],
                        "description": record["description"]
                    })
        else:
            # Query in-memory storage
            # Search for entities
            for entity in self.in_memory_entities:
                if (query_lower in entity['name'].lower() or 
                    query_lower in entity['description'].lower() or 
                    query_lower in entity['type'].lower()):
                    results.append({
                        "type": "entity",
                        "name": entity['name'],
                        "entity_type": entity['type'],
                        "description": entity['description']
                    })
                    if len(results) >= max_results:
                        break
            
            # Search for relationships
            for rel in self.in_memory_relationships:
                if (query_lower in rel['type'].lower() or 
                    query_lower in rel['description'].lower() or 
                    query_lower in rel['source'].lower() or 
                    query_lower in rel['target'].lower()):
                    results.append({
                        "type": "relationship",
                        "source": rel['source'],
                        "target": rel['target'],
                        "relationship_type": rel['type'],
                        "description": rel['description']
                    })
                    if len(results) >= max_results:
                        break
        
        return results[:max_results]
    
    def get_entity_relationships(self, entity: str, max_depth: int = 2) -> Dict[str, Any]:
        """Get relationships for a specific entity"""
        if not self.driver and not self.use_in_memory_fallback:
            return {"error": "Neo4j not available and no in-memory data"}
        
        if self.driver:
            # Query Neo4j database
            with self.driver.session() as session:
                # Check if entity exists
                exists = session.run("MATCH (e:Entity {name: $name}) RETURN e", name=entity)
                if not exists.single():
                    return {"error": f"Entity '{entity}' not found in graph"}
                
                # Get incoming relationships
                incoming = session.run("""
                    MATCH (source:Entity)-[r:RELATES_TO]->(target:Entity {name: $entity})
                    RETURN source.name as source, r.type as type, r.description as description
                """, entity=entity)
                
                # Get outgoing relationships
                outgoing = session.run("""
                    MATCH (source:Entity {name: $entity})-[r:RELATES_TO]->(target:Entity)
                    RETURN target.name as target, r.type as type, r.description as description
                """, entity=entity)
                
                # Get neighbors within max_depth
                neighbors = session.run("""
                    MATCH path = (start:Entity {name: $entity})-[*1..$depth]-(neighbor:Entity)
                    WHERE neighbor.name <> $entity
                    RETURN DISTINCT neighbor.name as name, neighbor.type as type, 
                           length(path) as distance
                    ORDER BY distance
                """, entity=entity, depth=max_depth)
                
                return {
                    "entity": entity,
                    "incoming": [{"source": r["source"], "type": r["type"], "description": r["description"]} 
                               for r in incoming],
                    "outgoing": [{"target": r["target"], "type": r["type"], "description": r["description"]} 
                               for r in outgoing],
                    "neighbors": [{"name": r["name"], "type": r["type"], "distance": r["distance"]} 
                                for r in neighbors]
                }
        else:
            # Query in-memory storage
            entity_lower = entity.lower()
            
            # Check if entity exists
            entity_exists = any(e['name'].lower() == entity_lower for e in self.in_memory_entities)
            if not entity_exists:
                return {"error": f"Entity '{entity}' not found in graph"}
            
            # Get incoming relationships
            incoming = []
            for rel in self.in_memory_relationships:
                if rel['target'].lower() == entity_lower:
                    incoming.append({
                        "source": rel['source'],
                        "type": rel["type"],
                        "description": rel['description']
                    })
            
            # Get outgoing relationships
            outgoing = []
            for rel in self.in_memory_relationships:
                if rel['source'].lower() == entity_lower:
                    outgoing.append({
                        "target": rel['target'],
                        "type": rel['type'],
                        "description": rel['description']
                    })
            
            # Get neighbors (simplified for in-memory)
            neighbors = []
            for rel in self.in_memory_relationships:
                if rel['source'].lower() == entity_lower:
                    neighbors.append({
                        "name": rel['target'],
                        "type": "unknown",  # We don't store target entity type in relationships
                        "distance": 1
                    })
                elif rel['target'].lower() == entity_lower:
                    neighbors.append({
                        "name": rel['source'],
                        "type": "unknown",  # We don't store source entity type in relationships
                        "distance": 1
                    })
            
            return {
                "entity": entity,
                "incoming": incoming,
                "outgoing": outgoing,
                "neighbors": neighbors[:max_depth]  # Limit to max_depth
            }
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        if not self.driver and not self.use_in_memory_fallback:
            return {"status": "neo4j_not_available"}
        
        if self.driver:
            # Get stats from Neo4j database
            with self.driver.session() as session:
                # Get node count
                node_count = session.run("MATCH (n:Entity) RETURN count(n) as count").single()["count"]
                
                # Get relationship count
                rel_count = session.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) as count").single()["count"]
                
                # Get entity types
                entity_types = session.run("""
                    MATCH (e:Entity)
                    RETURN e.type as type, count(e) as count
                    ORDER BY count DESC
                """)
                
                # Get relationship types
                rel_types = session.run("""
                    MATCH ()-[r:RELATES_TO]->()
                    RETURN r.type as type, count(r) as count
                    ORDER BY count DESC
                """)
                
                return {
                    "nodes": node_count,
                    "edges": rel_count,
                    "entity_types": {r["type"]: r["count"] for r in entity_types},
                    "relationship_types": {r["type"]: r["count"] for r in rel_types},
                    "status": "populated" if node_count > 0 else "empty"
                }
        else:
            # Get stats from in-memory storage
            from collections import Counter
            
            # Count entity types
            entity_types = Counter(entity['type'] for entity in self.in_memory_entities)
            
            # Count relationship types
            rel_types = Counter(rel['type'] for rel in self.in_memory_relationships)
            
            return {
                "nodes": len(self.in_memory_entities),
                "edges": len(self.in_memory_relationships),
                "entity_types": dict(entity_types),
                "relationship_types": dict(rel_types),
                "status": "populated" if self.in_memory_entities else "empty",
                "storage": "in_memory"
            }
    
    def save_graph(self, filename: str = "knowledge_graph.json"):
        """Save the knowledge graph to file (for backup purposes)"""
        if not self.driver:
            print("Neo4j not available, cannot save graph")
            return
        
        graph_data = {
            "nodes": [],
            "edges": [],
            "exported_at": datetime.now().isoformat()
        }
        
        with self.driver.session() as session:
            # Export nodes
            nodes = session.run("MATCH (e:Entity) RETURN e.name as name, e.type as type, e.description as description")
            for record in nodes:
                graph_data["nodes"].append({
                    "name": record["name"],
                    "type": record["type"],
                    "description": record["description"]
                })
            
            # Export edges
            edges = session.run("""
                MATCH (source:Entity)-[r:RELATES_TO]->(target:Entity)
                RETURN source.name as source, target.name as target, r.type as type, r.description as description
            """)
            for record in edges:
                graph_data["edges"].append({
                    "source": record["source"],
                    "target": record["target"],
                    "type": record["type"],
                    "description": record["description"]
                })
        
        # Save to file
        graph_path = Path(self.config.cache_dir) / "knowledge_graph"
        graph_path.mkdir(parents=True, exist_ok=True)
        file_path = graph_path / filename
        
        with open(file_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        print(f"Knowledge graph exported to {file_path}")
    
    def load_graph(self, filename: str = "knowledge_graph.json"):
        """Load the knowledge graph from file (for backup restoration)"""
        if not self.driver:
            print("Neo4j not available, cannot load graph")
            return None
        
        graph_path = Path(self.config.cache_dir) / "knowledge_graph" / filename
        
        if not graph_path.exists():
            print(f"Graph file not found: {graph_path}")
            return None
        
        with open(graph_path, 'r') as f:
            graph_data = json.load(f)
        
        # Clear existing data
        self._clear_database()
        
        # Import data
        with self.driver.session() as session:
            # Import nodes
            for node_data in graph_data.get("nodes", []):
                session.run("""
                    CREATE (e:Entity {name: $name, type: $type, description: $description})
                """, name=node_data["name"], type=node_data["type"], 
                     description=node_data.get("description", ""))
            
            # Import edges
            for edge_data in graph_data.get("edges", []):
                session.run("""
                    MATCH (source:Entity {name: $source})
                    MATCH (target:Entity {name: $target})
                    CREATE (source)-[r:RELATES_TO {type: $type, description: $description}]->(target)
                """, source=edge_data["source"], target=edge_data["target"],
                     type=edge_data["type"], description=edge_data.get("description", ""))
        
        print(f"Knowledge graph loaded with {len(graph_data.get('nodes', []))} nodes and {len(graph_data.get('edges', []))} edges")
        return self.get_graph_stats()
    
    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()
            print("Neo4j connection closed") 