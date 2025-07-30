#!/usr/bin/env python3
"""
Test script for knowledge graph extraction
"""

from knowledge_graph import KnowledgeGraphManager
from config import get_config

def test_knowledge_graph_extraction():
    """Test the rule-based knowledge graph extraction"""
    
    # Test text from test.txt
    test_text = """Sarah is an employee at prismaticAI, a leading technology company based in Westside Valley. She has been working there for the past three years as a software engineer.
Michael is also an employee at prismaticAI, where he works as a data scientist. He joined the company two years ago after completing his graduate studies.
prismaticAI is a well-known technology company that specializes in developing cutting-edge software solutions and artificial intelligence applications. The company has a diverse workforce of talented individuals from various backgrounds.
Both Sarah and Michael are highly skilled professionals who contribute significantly to prismaticAI's success. They work closely with their respective teams to develop innovative products and services that meet the evolving needs of the company's clients."""
    
    print("🧠 Testing Knowledge Graph Extraction")
    print("=" * 50)
    
    # Create knowledge graph manager
    config = get_config()
    kg_manager = KnowledgeGraphManager(config, config.models["gpt-3.5-turbo"])
    
    # Test rule-based extraction
    print("Testing rule-based extraction...")
    result = kg_manager._rule_based_extraction(test_text)
    
    print(f"Entities found: {len(result['entities'])}")
    for entity in result['entities']:
        print(f"  - {entity['name']} ({entity['type']}): {entity['description']}")
    
    print(f"\nRelationships found: {len(result['relationships'])}")
    for rel in result['relationships']:
        print(f"  - {rel['source']} --[{rel['type']}]--> {rel['target']}: {rel['description']}")
    
    # Test the full extraction method
    print("\nTesting full extraction method...")
    full_result = kg_manager._extract_entities_and_relations(test_text)
    
    print(f"Full extraction - Entities: {len(full_result['entities'])}")
    print(f"Full extraction - Relationships: {len(full_result['relationships'])}")
    
    return result

if __name__ == "__main__":
    test_knowledge_graph_extraction() 