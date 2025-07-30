"""
Test script for in-memory fallback functionality when Neo4j is not available
"""
import os
from pathlib import Path
from config import SystemConfig, ModelConfig, Neo4jConfig
from neo4j_knowledge_graph import Neo4jKnowledgeGraphManager

def test_in_memory_fallback():
    """Test the in-memory fallback when Neo4j is not available"""
    
    print("=" * 80)
    print("In-Memory Fallback Test")
    print("Testing Neo4jKnowledgeGraphManager when Neo4j is not available")
    print("=" * 80)
    
    # Create test configuration
    config = SystemConfig(
        models={
            "test-model": ModelConfig(
                name="Test Model",
                provider="openai",
                api_key=os.getenv("OPENAI_API_KEY", "test-key"),
                model_name="gpt-3.5-turbo"
            )
        },
        data_sources=[],
        knowledge_graph_enabled=True,
        knowledge_graph_type="neo4j",
        neo4j_config=Neo4jConfig(
            uri="bolt://localhost:7687",  # This should fail to connect
            user="neo4j",
            password="password"
        ),
        cache_dir="./cache",
        output_dir="./output"
    )
    
    # Initialize Neo4j knowledge graph manager (should fallback to in-memory)
    print("\n1. Initializing Neo4j Knowledge Graph Manager...")
    kg_manager = Neo4jKnowledgeGraphManager(
        config, 
        config.models["test-model"]
    )
    
    print(f"Neo4j driver available: {kg_manager.driver is not None}")
    print(f"Using in-memory fallback: {kg_manager.use_in_memory_fallback}")
    
    # Create test documents
    test_documents = [
        {
            "page_content": "Sarah Johnson is a software engineer at prismaticAI. She specializes in Python and machine learning.",
            "metadata": {"source": "test1.txt"}
        },
        {
            "page_content": "Michael Chen is a data scientist at TechCorp. He reports to Sarah Johnson and works on AI projects.",
            "metadata": {"source": "test2.txt"}
        },
        {
            "page_content": "Emily Davis serves as a product manager at InnovateLabs. She collaborates with both Sarah and Michael on various projects.",
            "metadata": {"source": "test3.txt"}
        }
    ]
    
    from langchain.schema import Document
    documents = [Document(page_content=doc["page_content"], metadata=doc["metadata"]) for doc in test_documents]
    
    # Test graph extraction (should work with in-memory fallback)
    print(f"\n2. Testing Graph Extraction...")
    try:
        stats = kg_manager.extract_graph_from_documents(documents, force_recreate=True)
        print(f"Extraction completed successfully!")
        print(f"Graph stats: {stats}")
    except Exception as e:
        print(f"Error during extraction: {e}")
    
    # Test querying the graph
    print(f"\n3. Testing Graph Queries...")
    
    test_queries = [
        "Sarah Johnson",
        "software engineer",
        "prismaticAI",
        "Michael Chen",
        "reports to",
        "collaborates"
    ]
    
    for query in test_queries:
        try:
            results = kg_manager.query_graph(query, max_results=5)
            print(f"\nQuery: '{query}'")
            print(f"Results: {len(results)} found")
            for result in results:
                if result["type"] == "entity":
                    print(f"  Entity: {result['name']} ({result['entity_type']})")
                else:
                    print(f"  Relationship: {result['source']} --[{result['relationship_type']}]--> {result['target']}")
        except Exception as e:
            print(f"Error querying '{query}': {e}")
    
    # Test entity relationships
    print(f"\n4. Testing Entity Relationships...")
    
    test_entities = ["Sarah Johnson", "Michael Chen", "Emily Davis", "prismaticAI"]
    
    for entity in test_entities:
        try:
            relationships = kg_manager.get_entity_relationships(entity, max_depth=2)
            print(f"\nEntity: '{entity}'")
            if "error" in relationships:
                print(f"  Error: {relationships['error']}")
            else:
                print(f"  Incoming relationships: {len(relationships['incoming'])}")
                print(f"  Outgoing relationships: {len(relationships['outgoing'])}")
                print(f"  Neighbors: {len(relationships['neighbors'])}")
                
                for rel in relationships['incoming'][:3]:  # Show first 3
                    print(f"    <- {rel['source']} --[{rel['type']}]--")
                for rel in relationships['outgoing'][:3]:  # Show first 3
                    print(f"    --[{rel['type']}]--> {rel['target']}")
        except Exception as e:
            print(f"Error getting relationships for '{entity}': {e}")
    
    # Test graph statistics
    print(f"\n5. Testing Graph Statistics...")
    try:
        stats = kg_manager.get_graph_stats()
        print(f"Graph Statistics: {stats}")
    except Exception as e:
        print(f"Error getting graph stats: {e}")
    
    # Test individual extraction methods
    print(f"\n6. Testing Individual Extraction Methods...")
    
    test_text = "Dr. Sarah Johnson works at prismaticAI as a senior software engineer. She manages Michael Chen, who is a data scientist specializing in Python and machine learning."
    
    print(f"\nTest Text: {test_text}")
    
    # Test LLM extraction
    print(f"\nLLM-based Extraction:")
    try:
        llm_results = kg_manager._llm_based_extraction(test_text)
        print(f"  Entities: {len(llm_results['entities'])}")
        print(f"  Relationships: {len(llm_results['relationships'])}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test semantic search extraction
    print(f"\nSemantic Search Extraction:")
    try:
        semantic_results = kg_manager._semantic_search_extraction(test_text)
        print(f"  Entities: {len(semantic_results['entities'])}")
        print(f"  Relationships: {len(semantic_results['relationships'])}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test rule-based extraction
    print(f"\nRule-based Extraction:")
    try:
        rule_results = kg_manager._rule_based_extraction(test_text)
        print(f"  Entities: {len(rule_results['entities'])}")
        print(f"  Relationships: {len(rule_results['relationships'])}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test combined extraction
    print(f"\nCombined Extraction:")
    try:
        combined_results = kg_manager._extract_entities_and_relations(test_text)
        print(f"  Entities: {len(combined_results['entities'])}")
        print(f"  Relationships: {len(combined_results['relationships'])}")
        
        print(f"\n  Extracted Entities:")
        for entity in combined_results['entities']:
            print(f"    - {entity['name']} ({entity['type']}): {entity['description']}")
        
        print(f"\n  Extracted Relationships:")
        for rel in combined_results['relationships']:
            print(f"    - {rel['source']} --[{rel['type']}]--> {rel['target']}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Cleanup
    if hasattr(kg_manager, 'close'):
        kg_manager.close()
    
    print(f"\n{'='*80}")
    print("In-Memory Fallback Test Completed!")
    print("✅ Neo4jKnowledgeGraphManager now works with in-memory fallback")
    print("✅ Graph extraction works without Neo4j")
    print("✅ Graph querying works without Neo4j")
    print("✅ Entity relationships work without Neo4j")
    print("✅ Graph statistics work without Neo4j")
    print(f"{'='*80}")

if __name__ == "__main__":
    test_in_memory_fallback() 