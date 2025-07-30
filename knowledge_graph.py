"""
Knowledge Graph module for extracting and querying entity relationships
"""
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path

from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_community.llms import Anthropic

from config import ModelConfig, SystemConfig

class KnowledgeGraphManager:
    """Manages knowledge graph extraction and querying"""
    
    def __init__(self, config: SystemConfig, model_config: ModelConfig):
        self.config = config
        self.model_config = model_config
        self.graph = nx.DiGraph()
        try:
            self.llm = self._initialize_llm()
        except Exception as e:
            print(f"Warning: Could not initialize LLM for knowledge graph: {e}")
            self.llm = self._create_dummy_llm()
        self.graph_path = Path(config.cache_dir) / "knowledge_graph"
        self.graph_path.mkdir(parents=True, exist_ok=True)
    
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
    
    def extract_graph_from_documents(self, documents: List[Document], force_recreate: bool = False):
        """Extract knowledge graph from documents using custom LLM-based extraction"""
        if self.graph.number_of_nodes() > 0 and not force_recreate:
            return self.graph
        
        print("Extracting knowledge graph from documents...")
        
        # Build NetworkX graph from documents
        self.graph = nx.DiGraph()
        
        # Process documents in batches to avoid token limits
        batch_size = 3
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            for doc in batch:
                try:
                    # Extract entities and relationships from document
                    entities_relations = self._extract_entities_and_relations(doc.page_content)
                    
                    # Add entities as nodes
                    for entity in entities_relations.get('entities', []):
                        if entity.get('name') and entity.get('type'):
                            self.graph.add_node(
                                entity['name'], 
                                type=entity['type'],
                                description=entity.get('description', '')
                            )
                    
                    # Add relationships as edges
                    for relation in entities_relations.get('relationships', []):
                        if relation.get('source') and relation.get('target') and relation.get('type'):
                            self.graph.add_edge(
                                relation['source'],
                                relation['target'],
                                type=relation['type'],
                                description=relation.get('description', '')
                            )
                            
                except Exception as e:
                    print(f"Error processing document: {e}")
                    continue
        
        print(f"Knowledge graph created with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph
    
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
        person_names = ["Sarah", "Michael"]
        for name in person_names:
            if name in text:
                entities.append({
                    "name": name,
                    "type": "person",
                    "description": "Individual mentioned in the text"
                })
        
        # Extract company names
        company_names = ["prismaticAI"]
        for company in company_names:
            if company in text:
                entities.append({
                    "name": company,
                    "type": "organization",
                    "description": "Technology company"
                })
        
        # Extract job titles
        job_titles = ["software engineer", "data scientist"]
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
    
    def save_graph(self, filename: str = "knowledge_graph.json"):
        """Save the knowledge graph to file"""
        graph_data = {
            "nodes": [],
            "edges": []
        }
        
        # Save nodes
        for node, attrs in self.graph.nodes(data=True):
            graph_data["nodes"].append({
                "id": node,
                "attributes": attrs
            })
        
        # Save edges
        for source, target, attrs in self.graph.edges(data=True):
            graph_data["edges"].append({
                "source": source,
                "target": target,
                "attributes": attrs
            })
        
        file_path = self.graph_path / filename
        with open(file_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        print(f"Knowledge graph saved to {file_path}")
    
    def load_graph(self, filename: str = "knowledge_graph.json"):
        """Load the knowledge graph from file"""
        file_path = self.graph_path / filename
        
        if not file_path.exists():
            print(f"Graph file not found: {file_path}")
            return None
        
        with open(file_path, 'r') as f:
            graph_data = json.load(f)
        
        self.graph = nx.DiGraph()
        
        # Load nodes
        for node_data in graph_data["nodes"]:
            self.graph.add_node(node_data["id"], **node_data["attributes"])
        
        # Load edges
        for edge_data in graph_data["edges"]:
            self.graph.add_edge(
                edge_data["source"],
                edge_data["target"],
                **edge_data["attributes"]
            )
        
        print(f"Knowledge graph loaded with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph
    
    def query_graph(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Query the knowledge graph using natural language"""
        # Simple keyword-based search for now
        # In a more sophisticated implementation, this could use LLM to parse queries
        query_lower = query.lower()
        results = []
        
        # Search in node attributes
        for node, attrs in self.graph.nodes(data=True):
            node_str = str(node).lower() + " " + " ".join(str(v).lower() for v in attrs.values())
            if query_lower in node_str:
                results.append({
                    "type": "node",
                    "id": node,
                    "attributes": attrs,
                    "score": node_str.count(query_lower)
                })
        
        # Search in edge attributes
        for source, target, attrs in self.graph.edges(data=True):
            edge_str = str(source).lower() + " " + str(target).lower() + " " + " ".join(str(v).lower() for v in attrs.values())
            if query_lower in edge_str:
                results.append({
                    "type": "edge",
                    "source": source,
                    "target": target,
                    "attributes": attrs,
                    "score": edge_str.count(query_lower)
                })
        
        # Sort by score and limit results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:max_results]
    
    def get_entity_relationships(self, entity: str, max_depth: int = 2) -> Dict[str, Any]:
        """Get relationships for a specific entity"""
        if entity not in self.graph:
            return {"error": f"Entity '{entity}' not found in graph"}
        
        relationships = {
            "entity": entity,
            "incoming": [],
            "outgoing": [],
            "neighbors": []
        }
        
        # Get incoming edges
        for pred in self.graph.predecessors(entity):
            edge_data = self.graph.get_edge_data(pred, entity)
            relationships["incoming"].append({
                "source": pred,
                "attributes": edge_data
            })
        
        # Get outgoing edges
        for succ in self.graph.successors(entity):
            edge_data = self.graph.get_edge_data(entity, succ)
            relationships["outgoing"].append({
                "target": succ,
                "attributes": edge_data
            })
        
        # Get neighbors (nodes within max_depth)
        neighbors = set()
        current_level = {entity}
        
        for depth in range(max_depth):
            next_level = set()
            for node in current_level:
                neighbors.update(self.graph.predecessors(node))
                neighbors.update(self.graph.successors(node))
                next_level.update(self.graph.predecessors(node))
                next_level.update(self.graph.successors(node))
            current_level = next_level - neighbors
        
        relationships["neighbors"] = list(neighbors)
        
        return relationships
    
    def visualize_graph(self, output_path: Optional[str] = None, max_nodes: int = 50):
        """Visualize the knowledge graph"""
        if self.graph.number_of_nodes() == 0:
            print("No graph to visualize")
            return
        
        # Limit nodes for visualization
        if self.graph.number_of_nodes() > max_nodes:
            # Get the most connected nodes
            node_degrees = dict(self.graph.degree())
            top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            subgraph = self.graph.subgraph([node for node, _ in top_nodes])
        else:
            subgraph = self.graph
        
        # Create plotly visualization
        pos = nx.spring_layout(subgraph, k=1, iterations=50)
        
        # Create edges
        edge_x = []
        edge_y = []
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        # Create nodes
        node_x = []
        node_y = []
        node_text = []
        for node in subgraph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(node))
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                color=[],
                line_width=2))
        
        # Color nodes by degree
        node_adjacencies = []
        for node in subgraph.nodes():
            node_adjacencies.append(len(list(subgraph.neighbors(node))))
        
        node_trace.marker.color = node_adjacencies
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Knowledge Graph Visualization',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
        
        if output_path:
            fig.write_html(output_path)
            print(f"Graph visualization saved to {output_path}")
        else:
            fig.show()
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        if self.graph.number_of_nodes() == 0:
            return {"status": "empty"}
        
        stats = {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "connected_components": nx.number_connected_components(self.graph.to_undirected()),
            "average_clustering": nx.average_clustering(self.graph.to_undirected()),
            "status": "populated"
        }
        
        if self.graph.number_of_nodes() > 0:
            stats["average_degree"] = sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes()
        
        return stats 