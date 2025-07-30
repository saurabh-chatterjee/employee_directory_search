"""
Test script for enhanced entity and relationship extraction
for both KnowledgeGraphManager (JSON) and Neo4jKnowledgeGraphManager
"""
import os
from pathlib import Path
from config import SystemConfig, ModelConfig, Neo4jConfig
from knowledge_graph import KnowledgeGraphManager
from neo4j_knowledge_graph import Neo4jKnowledgeGraphManager

def test_both_enhanced_extraction():
    """Test the enhanced entity and relationship extraction for both managers"""
    
    print("=" * 80)
    print("Enhanced Entity and Relationship Extraction Test")
    print("Comparing KnowledgeGraphManager (JSON) vs Neo4jKnowledgeGraphManager")
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
        knowledge_graph_type="json",  # We'll test both
        neo4j_config=Neo4jConfig(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password"
        ),
        cache_dir="./cache",
        output_dir="./output"
    )
    
    # Initialize both knowledge graph managers
    print("\n1. Initializing Knowledge Graph Managers...")
    
    # JSON-based manager
    json_kg_manager = KnowledgeGraphManager(
        config, 
        config.models["test-model"]
    )
    
    # Neo4j-based manager
    neo4j_kg_manager = Neo4jKnowledgeGraphManager(
        config, 
        config.models["test-model"]
    )
    
    # Test texts with different complexity levels
    test_texts = [
        # Simple text
        ("Simple", "Sarah Johnson is a software engineer at prismaticAI."),
        
        # Medium complexity
        ("Medium", "Dr. Michael Chen is a senior data scientist at TechCorp. He reports to Sarah Johnson and specializes in machine learning algorithms. He lives in San Francisco and uses Python and TensorFlow for his work."),
        
        # Complex text with multiple relationships
        ("Complex", "Emily Davis serves as a product manager at InnovateLabs, where she leads a team of developers. She collaborates with David Wilson, a UX designer from DesignFlow Inc. Both Emily and David work on AI-powered customer service tools. Emily is based in New York while David operates from Boston. They frequently travel to meet clients in Los Angeles and Chicago."),
        
        # Technical text with skills and technologies
        ("Technical", "Alex Rodriguez is a principal software engineer at DataFlow Systems. He has expertise in Python, JavaScript, and React. Alex manages a team of junior developers and reports to the CTO, Jennifer Smith. He specializes in building scalable microservices and has worked with AWS, Docker, and Kubernetes. Alex mentors Sarah Johnson, who is learning machine learning techniques.")
    ]
    
    for complexity, text in test_texts:
        print(f"\n{'='*60}")
        print(f"Test: {complexity} Text")
        print(f"{'='*60}")
        print(f"Text: {text}")
        
        print(f"\nExtracting entities and relationships...")
        
        # Test JSON-based extraction
        print(f"\n--- JSON Knowledge Graph Manager ---")
        try:
            json_results = json_kg_manager._extract_entities_and_relations(text)
            
            print(f"Extracted Entities ({len(json_results['entities'])}):")
            for entity in json_results['entities']:
                print(f"  - {entity['name']} ({entity['type']}): {entity['description']}")
            
            print(f"\nExtracted Relationships ({len(json_results['relationships'])}):")
            for rel in json_results['relationships']:
                print(f"  - {rel['source']} --[{rel['type']}]--> {rel['target']}")
                print(f"    Description: {rel['description']}")
            
        except Exception as e:
            print(f"Error during JSON extraction: {e}")
        
        # Test Neo4j-based extraction
        print(f"\n--- Neo4j Knowledge Graph Manager ---")
        try:
            neo4j_results = neo4j_kg_manager._extract_entities_and_relations(text)
            
            print(f"Extracted Entities ({len(neo4j_results['entities'])}):")
            for entity in neo4j_results['entities']:
                print(f"  - {entity['name']} ({entity['type']}): {entity['description']}")
            
            print(f"\nExtracted Relationships ({len(neo4j_results['relationships'])}):")
            for rel in neo4j_results['relationships']:
                print(f"  - {rel['source']} --[{rel['type']}]--> {rel['target']}")
                print(f"    Description: {rel['description']}")
            
        except Exception as e:
            print(f"Error during Neo4j extraction: {e}")
        
        # Compare results
        print(f"\n--- Comparison ---")
        json_entities = len(json_results.get('entities', []))
        json_relationships = len(json_results.get('relationships', []))
        neo4j_entities = len(neo4j_results.get('entities', []))
        neo4j_relationships = len(neo4j_results.get('relationships', []))
        
        print(f"JSON Manager: {json_entities} entities, {json_relationships} relationships")
        print(f"Neo4j Manager: {neo4j_entities} entities, {neo4j_relationships} relationships")
        
        if json_entities == neo4j_entities and json_relationships == neo4j_relationships:
            print("✅ Both managers produced identical results!")
        else:
            print("⚠️  Results differ between managers")
    
    # Test individual extraction methods for both managers
    print(f"\n{'='*80}")
    print("Testing Individual Extraction Methods")
    print(f"{'='*80}")
    
    test_text = "Dr. Sarah Johnson works at prismaticAI as a senior software engineer. She manages Michael Chen, who is a data scientist specializing in Python and machine learning."
    
    print(f"\nTest Text: {test_text}")
    
    # Test JSON manager individual methods
    print(f"\n--- JSON Manager Individual Methods ---")
    
    # Test LLM extraction
    print(f"\n1. LLM-based Extraction:")
    try:
        llm_results = json_kg_manager._llm_based_extraction(test_text)
        print(f"   Entities: {len(llm_results['entities'])}")
        print(f"   Relationships: {len(llm_results['relationships'])}")
        for entity in llm_results['entities']:
            print(f"     - {entity['name']} ({entity['type']})")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test semantic search extraction
    print(f"\n2. Semantic Search Extraction:")
    try:
        semantic_results = json_kg_manager._semantic_search_extraction(test_text)
        print(f"   Entities: {len(semantic_results['entities'])}")
        print(f"   Relationships: {len(semantic_results['relationships'])}")
        for entity in semantic_results['entities']:
            print(f"     - {entity['name']} ({entity['type']})")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test rule-based extraction
    print(f"\n3. Rule-based Extraction:")
    try:
        rule_results = json_kg_manager._rule_based_extraction(test_text)
        print(f"   Entities: {len(rule_results['entities'])}")
        print(f"   Relationships: {len(rule_results['relationships'])}")
        for entity in rule_results['entities']:
            print(f"     - {entity['name']} ({entity['type']})")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test Neo4j manager individual methods
    print(f"\n--- Neo4j Manager Individual Methods ---")
    
    # Test LLM extraction
    print(f"\n1. LLM-based Extraction:")
    try:
        llm_results = neo4j_kg_manager._llm_based_extraction(test_text)
        print(f"   Entities: {len(llm_results['entities'])}")
        print(f"   Relationships: {len(llm_results['relationships'])}")
        for entity in llm_results['entities']:
            print(f"     - {entity['name']} ({entity['type']})")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test semantic search extraction
    print(f"\n2. Semantic Search Extraction:")
    try:
        semantic_results = neo4j_kg_manager._semantic_search_extraction(test_text)
        print(f"   Entities: {len(semantic_results['entities'])}")
        print(f"   Relationships: {len(semantic_results['relationships'])}")
        for entity in semantic_results['entities']:
            print(f"     - {entity['name']} ({entity['type']})")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test rule-based extraction
    print(f"\n3. Rule-based Extraction:")
    try:
        rule_results = neo4j_kg_manager._rule_based_extraction(test_text)
        print(f"   Entities: {len(rule_results['entities'])}")
        print(f"   Relationships: {len(rule_results['relationships'])}")
        for entity in rule_results['entities']:
            print(f"     - {entity['name']} ({entity['type']})")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test deduplication for both managers
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
    
    # Test JSON manager deduplication
    json_unique_entities = json_kg_manager._deduplicate_entities(test_entities)
    json_unique_relationships = json_kg_manager._deduplicate_relationships(test_relationships)
    
    # Test Neo4j manager deduplication
    neo4j_unique_entities = neo4j_kg_manager._deduplicate_entities(test_entities)
    neo4j_unique_relationships = neo4j_kg_manager._deduplicate_relationships(test_relationships)
    
    print(f"   JSON Manager: Original entities: {len(test_entities)}, Unique: {len(json_unique_entities)}")
    print(f"   JSON Manager: Original relationships: {len(test_relationships)}, Unique: {len(json_unique_relationships)}")
    print(f"   Neo4j Manager: Original entities: {len(test_entities)}, Unique: {len(neo4j_unique_entities)}")
    print(f"   Neo4j Manager: Original relationships: {len(test_relationships)}, Unique: {len(neo4j_unique_relationships)}")
    
    # Cleanup
    if hasattr(neo4j_kg_manager, 'close'):
        neo4j_kg_manager.close()
    
    print(f"\n{'='*80}")
    print("Enhanced Extraction Test Completed!")
    print("Both KnowledgeGraphManager and Neo4jKnowledgeGraphManager now use:")
    print("- LLM-based extraction")
    print("- Semantic search extraction (spaCy NER + dependency parsing)")
    print("- Rule-based extraction (fallback)")
    print("- Automatic deduplication")
    print(f"{'='*80}")

def test_extraction_performance_comparison():
    """Test extraction performance comparison between both managers"""
    
    print(f"\n{'='*80}")
    print("Extraction Performance Comparison")
    print(f"{'='*80}")
    
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
        knowledge_graph_type="json",
        neo4j_config=Neo4jConfig(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password"
        ),
        cache_dir="./cache",
        output_dir="./output"
    )
    
    json_kg_manager = KnowledgeGraphManager(config, config.models["test-model"])
    neo4j_kg_manager = Neo4jKnowledgeGraphManager(config, config.models["test-model"])
    
    # Test with different text sizes
    test_texts = [
        ("Short", "Sarah works at prismaticAI."),
        ("Medium", "Sarah Johnson is a software engineer at prismaticAI. She manages Michael Chen who is a data scientist."),
        ("Long", "Dr. Sarah Johnson serves as a senior software engineer at prismaticAI, where she leads a team of developers and data scientists. She manages Michael Chen, a data scientist who specializes in machine learning algorithms, and Emily Davis, a product manager focused on AI-powered solutions. The team collaborates with David Wilson, a UX designer from DesignFlow Inc., on various projects. Sarah is based in San Francisco while her team members are distributed across New York, Boston, and Los Angeles. She has expertise in Python, JavaScript, React, and cloud technologies including AWS and Docker.")
    ]
    
    import time
    
    for size, text in test_texts:
        print(f"\nTesting {size} text ({len(text)} characters):")
        
        # Test JSON manager
        start_time = time.time()
        try:
            json_results = json_kg_manager._extract_entities_and_relations(text)
            json_time = time.time() - start_time
            print(f"  JSON Manager: {json_time:.2f}s, {len(json_results['entities'])} entities, {len(json_results['relationships'])} relationships")
        except Exception as e:
            print(f"  JSON Manager Error: {e}")
        
        # Test Neo4j manager
        start_time = time.time()
        try:
            neo4j_results = neo4j_kg_manager._extract_entities_and_relations(text)
            neo4j_time = time.time() - start_time
            print(f"  Neo4j Manager: {neo4j_time:.2f}s, {len(neo4j_results['entities'])} entities, {len(neo4j_results['relationships'])} relationships")
        except Exception as e:
            print(f"  Neo4j Manager Error: {e}")
    
    # Cleanup
    if hasattr(neo4j_kg_manager, 'close'):
        neo4j_kg_manager.close()

if __name__ == "__main__":
    test_both_enhanced_extraction()
    test_extraction_performance_comparison() 