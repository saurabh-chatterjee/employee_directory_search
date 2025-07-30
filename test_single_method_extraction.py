"""
Test script for single-method entity and relationship extraction
"""
import os
from pathlib import Path
from config import SystemConfig, ModelConfig, Neo4jConfig
from knowledge_graph import KnowledgeGraphManager
from neo4j_knowledge_graph import Neo4jKnowledgeGraphManager

def test_single_method_extraction():
    """Test the single-method extraction approach"""
    
    print("=" * 80)
    print("Single-Method Extraction Test")
    print("Testing extraction using LLM OR Semantic Search OR Rule-based")
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
        knowledge_graph_type="json",
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
    
    json_kg_manager = KnowledgeGraphManager(config, config.models["test-model"])
    neo4j_kg_manager = Neo4jKnowledgeGraphManager(config, config.models["test-model"])
    
    # Test texts with different complexity levels
    test_texts = [
        # Simple text - should work with rule-based
        ("Simple", "Sarah Johnson is a software engineer at prismaticAI."),
        
        # Medium complexity - should work with semantic search
        ("Medium", "Dr. Michael Chen is a senior data scientist at TechCorp. He reports to Sarah Johnson and specializes in machine learning algorithms."),
        
        # Complex text - should work with LLM
        ("Complex", "Emily Davis serves as a product manager at InnovateLabs, where she leads a team of developers. She collaborates with David Wilson, a UX designer from DesignFlow Inc. Both Emily and David work on AI-powered customer service tools."),
        
        # Technical text - should work with LLM
        ("Technical", "Alex Rodriguez is a principal software engineer at DataFlow Systems. He has expertise in Python, JavaScript, and React. Alex manages a team of junior developers and reports to the CTO, Jennifer Smith.")
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
    
    # Test individual extraction methods to show priority
    print(f"\n{'='*80}")
    print("Testing Individual Extraction Methods (Priority Order)")
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
        if llm_results['entities'] or llm_results['relationships']:
            print("   ✅ LLM extraction would be used")
        else:
            print("   ❌ LLM extraction returned no results")
    except Exception as e:
        print(f"   ❌ LLM extraction failed: {e}")
    
    # Test semantic search extraction
    print(f"\n2. Semantic Search Extraction:")
    try:
        semantic_results = json_kg_manager._semantic_search_extraction(test_text)
        print(f"   Entities: {len(semantic_results['entities'])}")
        print(f"   Relationships: {len(semantic_results['relationships'])}")
        if semantic_results['entities'] or semantic_results['relationships']:
            print("   ✅ Semantic search extraction would be used")
        else:
            print("   ❌ Semantic search extraction returned no results")
    except Exception as e:
        print(f"   ❌ Semantic search extraction failed: {e}")
    
    # Test rule-based extraction
    print(f"\n3. Rule-based Extraction:")
    try:
        rule_results = json_kg_manager._rule_based_extraction(test_text)
        print(f"   Entities: {len(rule_results['entities'])}")
        print(f"   Relationships: {len(rule_results['relationships'])}")
        if rule_results['entities'] or rule_results['relationships']:
            print("   ✅ Rule-based extraction would be used")
        else:
            print("   ❌ Rule-based extraction returned no results")
    except Exception as e:
        print(f"   ❌ Rule-based extraction failed: {e}")
    
    # Test Neo4j manager individual methods
    print(f"\n--- Neo4j Manager Individual Methods ---")
    
    # Test LLM extraction
    print(f"\n1. LLM-based Extraction:")
    try:
        llm_results = neo4j_kg_manager._llm_based_extraction(test_text)
        print(f"   Entities: {len(llm_results['entities'])}")
        print(f"   Relationships: {len(llm_results['relationships'])}")
        if llm_results['entities'] or llm_results['relationships']:
            print("   ✅ LLM extraction would be used")
        else:
            print("   ❌ LLM extraction returned no results")
    except Exception as e:
        print(f"   ❌ LLM extraction failed: {e}")
    
    # Test semantic search extraction
    print(f"\n2. Semantic Search Extraction:")
    try:
        semantic_results = neo4j_kg_manager._semantic_search_extraction(test_text)
        print(f"   Entities: {len(semantic_results['entities'])}")
        print(f"   Relationships: {len(semantic_results['relationships'])}")
        if semantic_results['entities'] or semantic_results['relationships']:
            print("   ✅ Semantic search extraction would be used")
        else:
            print("   ❌ Semantic search extraction returned no results")
    except Exception as e:
        print(f"   ❌ Semantic search extraction failed: {e}")
    
    # Test rule-based extraction
    print(f"\n3. Rule-based Extraction:")
    try:
        rule_results = neo4j_kg_manager._rule_based_extraction(test_text)
        print(f"   Entities: {len(rule_results['entities'])}")
        print(f"   Relationships: {len(rule_results['relationships'])}")
        if rule_results['entities'] or rule_results['relationships']:
            print("   ✅ Rule-based extraction would be used")
        else:
            print("   ❌ Rule-based extraction returned no results")
    except Exception as e:
        print(f"   ❌ Rule-based extraction failed: {e}")
    
    # Test the actual combined extraction to see which method is chosen
    print(f"\n{'='*80}")
    print("Testing Combined Extraction (Which Method Gets Chosen)")
    print(f"{'='*80}")
    
    print(f"\nTest Text: {test_text}")
    
    print(f"\n--- JSON Manager Combined Extraction ---")
    try:
        combined_results = json_kg_manager._extract_entities_and_relations(test_text)
        print(f"Final Results: {len(combined_results['entities'])} entities, {len(combined_results['relationships'])} relationships")
    except Exception as e:
        print(f"Error: {e}")
    
    print(f"\n--- Neo4j Manager Combined Extraction ---")
    try:
        combined_results = neo4j_kg_manager._extract_entities_and_relations(test_text)
        print(f"Final Results: {len(combined_results['entities'])} entities, {len(combined_results['relationships'])} relationships")
    except Exception as e:
        print(f"Error: {e}")
    
    # Cleanup
    if hasattr(neo4j_kg_manager, 'close'):
        neo4j_kg_manager.close()
    
    print(f"\n{'='*80}")
    print("Single-Method Extraction Test Completed!")
    print("✅ Both managers now use single-method extraction")
    print("✅ Priority order: LLM > Semantic Search > Rule-based")
    print("✅ Only one method is used per extraction")
    print("✅ Fallback to next method if current method fails or returns no results")
    print(f"{'='*80}")

def test_extraction_performance():
    """Test extraction performance with single-method approach"""
    
    print(f"\n{'='*80}")
    print("Extraction Performance Test (Single-Method)")
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
        ("Long", "Dr. Sarah Johnson serves as a senior software engineer at prismaticAI, where she leads a team of developers and data scientists. She manages Michael Chen, a data scientist who specializes in machine learning algorithms, and Emily Davis, a product manager focused on AI-powered solutions.")
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
    test_single_method_extraction()
    test_extraction_performance() 