"""
Test script for Neo4j Knowledge Graph functionality
"""
import os
from pathlib import Path
from config import get_config, SystemConfig, ModelConfig, Neo4jConfig
from neo4j_knowledge_graph import Neo4jKnowledgeGraphManager
from langchain.schema import Document

def test_neo4j_knowledge_graph():
    """Test the Neo4j knowledge graph functionality"""
    
    # Create test configuration
    config = SystemConfig(
        models={
            "test-model": ModelConfig(
                name="Test Model",
                provider="openai",
                api_key="test-key",
                model_name="gpt-3.5-turbo"
            )
        },
        data_sources=[],
        knowledge_graph_enabled=True,
        knowledge_graph_type="neo4j",
        neo4j_config=Neo4jConfig(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password"
        ),
        cache_dir="./cache",
        output_dir="./output"
    )
    
    # Create test documents
    test_documents = [
        Document(
            page_content="Sarah is a software engineer at prismaticAI. She has been working there for 3 years.",
            metadata={"source": "test1.txt", "author": "test"}
        ),
        Document(
            page_content="Michael is a data scientist at prismaticAI. He reports to Sarah.",
            metadata={"source": "test2.txt", "author": "test"}
        ),
        Document(
            page_content="prismaticAI is a technology company focused on AI solutions.",
            metadata={"source": "test3.txt", "author": "test"}
        )
    ]
    
    # Initialize Neo4j knowledge graph manager
    print("Initializing Neo4j Knowledge Graph Manager...")
    kg_manager = Neo4jKnowledgeGraphManager(
        config, 
        config.models["test-model"]
    )
    
    try:
        # Extract knowledge graph from documents
        print("\nExtracting knowledge graph from documents...")
        stats = kg_manager.extract_graph_from_documents(test_documents, force_recreate=True)
        print(f"Graph stats: {stats}")
        
        # Query the graph
        print("\nQuerying the knowledge graph...")
        results = kg_manager.query_graph("Sarah", max_results=5)
        print(f"Query results for 'Sarah': {results}")
        
        # Get entity relationships
        print("\nGetting entity relationships...")
        relationships = kg_manager.get_entity_relationships("Sarah", max_depth=2)
        print(f"Sarah's relationships: {relationships}")
        
        # Get graph statistics
        print("\nGetting graph statistics...")
        stats = kg_manager.get_graph_stats()
        print(f"Graph statistics: {stats}")
        
        # Export graph to JSON (for backup)
        print("\nExporting graph to JSON...")
        kg_manager.save_graph("test_knowledge_graph.json")
        
        print("\nNeo4j Knowledge Graph test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        print("This might be because Neo4j is not running or not accessible")
        print("To run Neo4j locally, you can use Docker:")
        print("docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest")
    
    finally:
        # Close the connection
        kg_manager.close()

def test_without_neo4j():
    """Test the knowledge graph manager when Neo4j is not available"""
    print("\n" + "="*50)
    print("Testing without Neo4j (should work in memory mode)")
    print("="*50)
    
    # Create test configuration with invalid Neo4j settings
    config = SystemConfig(
        models={
            "test-model": ModelConfig(
                name="Test Model",
                provider="openai",
                api_key="test-key",
                model_name="gpt-3.5-turbo"
            )
        },
        data_sources=[],
        knowledge_graph_enabled=True,
        knowledge_graph_type="neo4j",
        neo4j_config=Neo4jConfig(
            uri="bolt://invalid-host:7687",
            user="neo4j",
            password="wrong-password"
        ),
        cache_dir="./cache",
        output_dir="./output"
    )
    
    # Initialize Neo4j knowledge graph manager
    print("Initializing Neo4j Knowledge Graph Manager with invalid connection...")
    kg_manager = Neo4jKnowledgeGraphManager(
        config, 
        config.models["test-model"]
    )
    
    # Test that it gracefully handles the lack of Neo4j
    print(f"Driver available: {kg_manager.driver is not None}")
    
    # Test methods that should work even without Neo4j
    stats = kg_manager.get_graph_stats()
    print(f"Graph stats without Neo4j: {stats}")
    
    results = kg_manager.query_graph("test", max_results=5)
    print(f"Query results without Neo4j: {results}")
    
    kg_manager.close()

if __name__ == "__main__":
    print("Testing Neo4j Knowledge Graph Manager")
    print("="*50)
    
    # Test with Neo4j (if available)
    test_neo4j_knowledge_graph()
    
    # Test without Neo4j
    test_without_neo4j()
    
    print("\nAll tests completed!") 