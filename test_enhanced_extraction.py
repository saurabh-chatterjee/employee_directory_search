"""
Test script for enhanced entity and relationship extraction
"""
import os
from pathlib import Path
from config import SystemConfig, ModelConfig, Neo4jConfig
from neo4j_knowledge_graph import Neo4jKnowledgeGraphManager

def test_enhanced_extraction():
    """Test the enhanced entity and relationship extraction"""
    
    print("=" * 60)
    print("Enhanced Entity and Relationship Extraction Test")
    print("=" * 60)
    
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
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password"
        ),
        cache_dir="./cache",
        output_dir="./output"
    )
    
    # Initialize Neo4j knowledge graph manager
    print("\n1. Initializing Neo4j Knowledge Graph Manager...")
    kg_manager = Neo4jKnowledgeGraphManager(
        config, 
        config.models["test-model"]
    )
    
    # Test texts with different complexity levels
    test_texts = [
        # Simple text
        "Sarah Johnson is a software engineer at prismaticAI.",
        
        # Medium complexity
        "Dr. Michael Chen is a senior data scientist at TechCorp. He reports to Sarah Johnson and specializes in machine learning algorithms. He lives in San Francisco and uses Python and TensorFlow for his work.",
        
        # Complex text with multiple relationships
        "Emily Davis serves as a product manager at InnovateLabs, where she leads a team of developers. She collaborates with David Wilson, a UX designer from DesignFlow Inc. Both Emily and David work on AI-powered customer service tools. Emily is based in New York while David operates from Boston. They frequently travel to meet clients in Los Angeles and Chicago.",
        
        # Technical text with skills and technologies
        "Alex Rodriguez is a principal software engineer at DataFlow Systems. He has expertise in Python, JavaScript, and React. Alex manages a team of junior developers and reports to the CTO, Jennifer Smith. He specializes in building scalable microservices and has worked with AWS, Docker, and Kubernetes. Alex mentors Sarah Johnson, who is learning machine learning techniques."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{'='*40}")
        print(f"Test Text {i}:")
        print(f"{'='*40}")
        print(f"Text: {text}")
        
        print(f"\nExtracting entities and relationships...")
        
        # Test the enhanced extraction
        try:
            results = kg_manager._extract_entities_and_relations(text)
            
            print(f"\nExtracted Entities ({len(results['entities'])}):")
            for entity in results['entities']:
                print(f"  - {entity['name']} ({entity['type']}): {entity['description']}")
            
            print(f"\nExtracted Relationships ({len(results['relationships'])}):")
            for rel in results['relationships']:
                print(f"  - {rel['source']} --[{rel['type']}]--> {rel['target']}")
                print(f"    Description: {rel['description']}")
            
        except Exception as e:
            print(f"Error during extraction: {e}")
    
    # Test individual extraction methods
    print(f"\n{'='*60}")
    print("Testing Individual Extraction Methods")
    print(f"{'='*60}")
    
    test_text = "Dr. Sarah Johnson works at prismaticAI as a senior software engineer. She manages Michael Chen, who is a data scientist specializing in Python and machine learning."
    
    print(f"\nTest Text: {test_text}")
    
    # Test LLM extraction
    print(f"\n1. LLM-based Extraction:")
    try:
        llm_results = kg_manager._llm_based_extraction(test_text)
        print(f"   Entities: {len(llm_results['entities'])}")
        print(f"   Relationships: {len(llm_results['relationships'])}")
        for entity in llm_results['entities']:
            print(f"     - {entity['name']} ({entity['type']})")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test semantic search extraction
    print(f"\n2. Semantic Search Extraction:")
    try:
        semantic_results = kg_manager._semantic_search_extraction(test_text)
        print(f"   Entities: {len(semantic_results['entities'])}")
        print(f"   Relationships: {len(semantic_results['relationships'])}")
        for entity in semantic_results['entities']:
            print(f"     - {entity['name']} ({entity['type']})")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test rule-based extraction
    print(f"\n3. Rule-based Extraction:")
    try:
        rule_results = kg_manager._rule_based_extraction(test_text)
        print(f"   Entities: {len(rule_results['entities'])}")
        print(f"   Relationships: {len(rule_results['relationships'])}")
        for entity in rule_results['entities']:
            print(f"     - {entity['name']} ({entity['type']})")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test deduplication
    print(f"\n4. Deduplication Test:")
    test_entities = [
        {"name": "Sarah", "type": "person", "description": "Test"},
        {"name": "sarah", "type": "person", "description": "Test 2"},
        {"name": "Michael", "type": "person", "description": "Test 3"},
    ]
    
    test_relationships = [
        {"source": "Sarah", "target": "prismaticAI", "type": "works_at", "description": "Test"},
        {"source": "sarah", "target": "prismaticAI", "type": "works_at", "description": "Test 2"},
        {"source": "Michael", "target": "Sarah", "type": "reports_to", "description": "Test 3"},
    ]
    
    unique_entities = kg_manager._deduplicate_entities(test_entities)
    unique_relationships = kg_manager._deduplicate_relationships(test_relationships)
    
    print(f"   Original entities: {len(test_entities)}, Unique: {len(unique_entities)}")
    print(f"   Original relationships: {len(test_relationships)}, Unique: {len(unique_relationships)}")
    
    kg_manager.close()
    print(f"\n{'='*60}")
    print("Enhanced Extraction Test Completed!")
    print(f"{'='*60}")

def test_extraction_performance():
    """Test extraction performance with different text sizes"""
    
    print(f"\n{'='*60}")
    print("Extraction Performance Test")
    print(f"{'='*60}")
    
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
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password"
        ),
        cache_dir="./cache",
        output_dir="./output"
    )
    
    kg_manager = Neo4jKnowledgeGraphManager(config, config.models["test-model"])
    
    # Test with different text sizes
    test_texts = [
        ("Short", "Sarah works at prismaticAI."),
        ("Medium", "Sarah Johnson is a software engineer at prismaticAI. She manages Michael Chen who is a data scientist."),
        ("Long", "Dr. Sarah Johnson serves as a senior software engineer at prismaticAI, where she leads a team of developers and data scientists. She manages Michael Chen, a data scientist who specializes in machine learning algorithms, and Emily Davis, a product manager focused on AI-powered solutions. The team collaborates with David Wilson, a UX designer from DesignFlow Inc., on various projects. Sarah is based in San Francisco while her team members are distributed across New York, Boston, and Los Angeles. She has expertise in Python, JavaScript, React, and cloud technologies including AWS and Docker.")
    ]
    
    import time
    
    for size, text in test_texts:
        print(f"\nTesting {size} text ({len(text)} characters):")
        
        start_time = time.time()
        try:
            results = kg_manager._extract_entities_and_relations(text)
            end_time = time.time()
            
            print(f"  Time: {end_time - start_time:.2f} seconds")
            print(f"  Entities: {len(results['entities'])}")
            print(f"  Relationships: {len(results['relationships'])}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    kg_manager.close()

if __name__ == "__main__":
    test_enhanced_extraction()
    test_extraction_performance() 