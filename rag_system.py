"""
Main RAG System that orchestrates all components
"""
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime

from config import get_config, SystemConfig, ModelConfig, DataSourceConfig
from data_loader import DataSourceManager
from vector_store import VectorStoreManager
from knowledge_graph import KnowledgeGraphManager
from neo4j_knowledge_graph import Neo4jKnowledgeGraphManager
from llm_manager import LLMManager, RAGChain

class RAGSystem:
    """Main RAG system that orchestrates all components"""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or get_config()
        
        # Initialize components
        self.data_manager = DataSourceManager(self.config.data_sources)
        self.vector_manager = VectorStoreManager(self.config)
        self.llm_manager = LLMManager(self.config.models)
        
        # Initialize knowledge graph if enabled
        self.knowledge_graph = None
        if self.config.knowledge_graph_enabled:
            current_model = self.llm_manager.get_current_model()
            if current_model:
                print(f"Initializing knowledge graph with model: {current_model}")
                if self.config.knowledge_graph_type == "neo4j":
                    print(f"Initializing Neo4j knowledge graph with model: {current_model}")
                    self.knowledge_graph = Neo4jKnowledgeGraphManager(
                        self.config, 
                        self.config.models[current_model],
                        neo4j_uri=self.config.neo4j_config.uri,
                        neo4j_user=self.config.neo4j_config.user,
                        neo4j_password=self.config.neo4j_config.password
                    )
                else:
                    print(f"Initializing knowledge graph with model: {current_model}")
                    self.knowledge_graph = KnowledgeGraphManager(
                        self.config, 
                        self.config.models[current_model]
                    )
        
        # Initialize RAG chain
        self.rag_chain = RAGChain(self.llm_manager)
        
        # System state
        self.is_initialized = False
        self.documents = {}
        self.session_history = []
    
    def initialize(self, force_reload: bool = False) -> bool:
        """Initialize the RAG system"""
        try:
            print("Initializing RAG System...")
            
            # Load documents from all data sources
            print("Loading data sources...")
            self.documents = self.data_manager.load_all_sources()
            
            if not self.documents:
                print("Warning: No documents loaded from data sources")
                return False
            
            # Create vector store
            print("Creating vector store...")
            all_docs = self.data_manager.get_all_documents()
            self.vector_manager.create_vector_store(all_docs, force_recreate=force_reload)
            
            # Extract knowledge graph if enabled
            if self.knowledge_graph and self.config.knowledge_graph_enabled:
                print("Extracting knowledge graph...")
                self.knowledge_graph.extract_graph_from_documents(all_docs, force_recreate=force_reload)
                
                # Save knowledge graph
                self.knowledge_graph.save_graph()
            
            self.is_initialized = True
            print("RAG System initialized successfully!")
            return True
            
        except Exception as e:
            print(f"Error initializing RAG System: {str(e)}")
            return False
    
    def ask_question(self, question: str, use_knowledge_graph: bool = True, 
                    max_context_docs: int = 5) -> Dict[str, Any]:
        """Ask a question and get an answer using RAG"""
        if not self.is_initialized:
            raise ValueError("RAG System not initialized. Call initialize() first.")
        
        print(f"Processing question: {question}")
        
        # Get relevant documents from vector store
        context_docs = self.vector_manager.similarity_search_with_score(
            question, k=max_context_docs
        )
        
        # Convert to expected format
        context_data = []
        for doc, score in context_docs:
            context_data.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            })
        
        # Get knowledge graph insights if enabled
        kg_insights = None
        if use_knowledge_graph and self.knowledge_graph:
            kg_results = self.knowledge_graph.query_graph(question, max_results=3)
            if kg_results:
                kg_insights = {
                    "entities": [r for r in kg_results if r["type"] == "node"],
                    "relationships": [r for r in kg_results if r["type"] == "edge"]
                }
        
        # Generate answer using RAG
        rag_response = self.rag_chain.answer_with_sources(question, context_data)
        
        # Combine results
        response = {
            "question": question,
            "answer": rag_response["answer"],
            "sources": rag_response["sources"],
            "context_docs": len(context_data),
            "knowledge_graph_insights": kg_insights,
            "model_used": self.llm_manager.get_current_model(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to session history
        self.session_history.append(response)
        
        return response
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different LLM model"""
        success = self.llm_manager.switch_model(model_name)
        if success:
            # Reinitialize RAG chain with new model
            self.rag_chain = RAGChain(self.llm_manager)
            
            # Reinitialize knowledge graph if needed
            if self.knowledge_graph and self.config.knowledge_graph_enabled:
                self.knowledge_graph = KnowledgeGraphManager(
                    self.config, 
                    self.config.models[model_name]
                )
        
        return success
    
    def add_data_source(self, source: DataSourceConfig) -> bool:
        """Add a new data source"""
        try:
            self.data_manager.data_sources.append(source)
            
            # Load documents from new source
            loader = self.data_manager.loaders.get(source.name)
            if not loader:
                loader = self.data_manager.loaders[source.name] = self.data_manager.loaders[source.name] = None
            
            documents = self.data_manager.get_source_documents(source.name)
            if documents:
                self.vector_manager.add_documents(documents)
                print(f"Added {len(documents)} documents from {source.name}")
                return True
            else:
                print(f"No documents loaded from {source.name}")
                return False
                
        except Exception as e:
            print(f"Error adding data source {source.name}: {str(e)}")
            return False
    
    def remove_data_source(self, source_name: str) -> bool:
        """Remove a data source"""
        try:
            # Remove from data manager
            self.data_manager.data_sources = [
                source for source in self.data_manager.data_sources 
                if source.name != source_name
            ]
            
            # Remove documents from vector store (requires recreation)
            if source_name in self.data_manager.documents:
                del self.data_manager.documents[source_name]
                
                # Recreate vector store without the removed source
                all_docs = self.data_manager.get_all_documents()
                self.vector_manager.create_vector_store(all_docs, force_recreate=True)
            
            print(f"Removed data source: {source_name}")
            return True
            
        except Exception as e:
            print(f"Error removing data source {source_name}: {str(e)}")
            return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        stats = {
            "initialized": self.is_initialized,
            "current_model": self.llm_manager.get_current_model(),
            "available_models": self.llm_manager.get_available_models(),
            "data_sources": len(self.data_manager.data_sources),
            "loaded_documents": sum(len(docs) for docs in self.documents.values()),
            "session_questions": len(self.session_history),
            "knowledge_graph_enabled": self.config.knowledge_graph_enabled
        }
        
        # Add vector store stats
        vector_stats = self.vector_manager.get_stats()
        stats["vector_store"] = vector_stats
        
        # Add knowledge graph stats
        if self.knowledge_graph:
            kg_stats = self.knowledge_graph.get_graph_stats()
            stats["knowledge_graph"] = kg_stats
        
        # Add data source details
        source_details = {}
        for source in self.data_manager.data_sources:
            source_details[source.name] = {
                "type": source.type,
                "path": source.path,
                "format": source.format,
                "enabled": source.enabled,
                "document_count": len(self.documents.get(source.name, []))
            }
        stats["data_source_details"] = source_details
        
        return stats
    
    def save_session(self, filename: str = None) -> str:
        """Save current session to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_session_{timestamp}.json"
        
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config.dict(),
            "session_history": self.session_history,
            "system_stats": self.get_system_stats()
        }
        
        output_path = Path(self.config.output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"Session saved to {output_path}")
        return str(output_path)
    
    def load_session(self, filename: str) -> bool:
        """Load session from file"""
        try:
            file_path = Path(self.config.output_dir) / filename
            
            if not file_path.exists():
                print(f"Session file not found: {file_path}")
                return False
            
            with open(file_path, 'r') as f:
                session_data = json.load(f)
            
            # Load session history
            self.session_history = session_data.get("session_history", [])
            
            print(f"Session loaded from {file_path}")
            return True
            
        except Exception as e:
            print(f"Error loading session: {str(e)}")
            return False
    
    def visualize_knowledge_graph(self, output_path: str = None, max_nodes: int = 50):
        """Visualize the knowledge graph"""
        if not self.knowledge_graph:
            print("Knowledge graph not enabled")
            return
        
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(Path(self.config.output_dir) / f"knowledge_graph_{timestamp}.html")
        
        self.knowledge_graph.visualize_graph(output_path, max_nodes)
    
    def get_entity_relationships(self, entity: str, max_depth: int = 2) -> Dict[str, Any]:
        """Get relationships for a specific entity from knowledge graph"""
        if not self.knowledge_graph:
            return {"error": "Knowledge graph not enabled"}
        
        return self.knowledge_graph.get_entity_relationships(entity, max_depth)
    
    def clear_cache(self):
        """Clear all cached data"""
        try:
            # Clear vector store
            self.vector_manager.clear()
            
            # Clear knowledge graph
            if self.knowledge_graph:
                if hasattr(self.knowledge_graph, 'graph_path'):
                    # For JSON-based knowledge graph
                    import shutil
                    if self.knowledge_graph.graph_path.exists():
                        shutil.rmtree(self.knowledge_graph.graph_path)
                elif hasattr(self.knowledge_graph, '_clear_database'):
                    # For Neo4j-based knowledge graph
                    self.knowledge_graph._clear_database()
            
            # Clear session history
            self.session_history = []
            
            # Reset initialization
            self.is_initialized = False
            
            print("Cache cleared successfully")
            
        except Exception as e:
            print(f"Error clearing cache: {str(e)}")
    
    def close(self):
        """Close all connections and cleanup resources"""
        try:
            if self.knowledge_graph and hasattr(self.knowledge_graph, 'close'):
                self.knowledge_graph.close()
            print("RAG System closed successfully")
        except Exception as e:
            print(f"Error closing RAG System: {e}") 