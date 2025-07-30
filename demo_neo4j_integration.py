"""
Demonstration of Neo4j Knowledge Graph Integration with RAG System
"""
import os
from pathlib import Path
from config import SystemConfig, ModelConfig, Neo4jConfig, DataSourceConfig
from rag_system import RAGSystem
from langchain.schema import Document

def create_test_data():
    """Create test documents for demonstration"""
    test_documents = [
        Document(
            page_content="Sarah Johnson is a senior software engineer at prismaticAI. She has been working there for 3 years and specializes in machine learning algorithms.",
            metadata={"source": "employee_data.txt", "author": "HR"}
        ),
        Document(
            page_content="Michael Chen is a data scientist at prismaticAI. He reports to Sarah Johnson and works on predictive analytics projects.",
            metadata={"source": "employee_data.txt", "author": "HR"}
        ),
        Document(
            page_content="prismaticAI is a technology company focused on AI solutions. The company was founded in 2020 and has 50 employees.",
            metadata={"source": "company_info.txt", "author": "Marketing"}
        ),
        Document(
            page_content="Emily Davis is a product manager at prismaticAI. She leads the development of AI-powered customer service tools.",
            metadata={"source": "employee_data.txt", "author": "HR"}
        ),
        Document(
            page_content="David Wilson is a UX designer at prismaticAI. He collaborates with Emily Davis on product design.",
            metadata={"source": "employee_data.txt", "author": "HR"}
        )
    ]
    
    # Save test documents to file
    test_file = Path("test_employee_data.txt")
    with open(test_file, 'w') as f:
        for doc in test_documents:
            f.write(f"Source: {doc.metadata['source']}\n")
            f.write(f"Content: {doc.page_content}\n")
            f.write("-" * 50 + "\n")
    
    return test_file

def demonstrate_neo4j_integration():
    """Demonstrate the Neo4j knowledge graph integration"""
    
    print("=" * 60)
    print("Neo4j Knowledge Graph Integration Demonstration")
    print("=" * 60)
    
    # Create test data
    print("\n1. Creating test employee data...")
    test_file = create_test_data()
    print(f"   Test data saved to: {test_file}")
    
    # Create configuration with Neo4j
    print("\n2. Setting up RAG system with Neo4j knowledge graph...")
    
    config = SystemConfig(
        models={
            "gpt-3.5-turbo": ModelConfig(
                name="GPT-3.5 Turbo",
                provider="openai",
                api_key=os.getenv("OPENAI_API_KEY", "test-key"),
                model_name="gpt-3.5-turbo",
                temperature=0.1,
                max_tokens=2000
            )
        },
        data_sources=[
            DataSourceConfig(
                name="Employee Data",
                type="file",
                path=str(test_file),
                format="txt",
                chunk_size=500,
                chunk_overlap=100,
                enabled=True
            )
        ],
        knowledge_graph_enabled=True,
        knowledge_graph_type="neo4j",  # Use Neo4j instead of JSON
        neo4j_config=Neo4jConfig(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password"
        ),
        cache_dir="./cache",
        output_dir="./output"
    )
    
    # Initialize RAG system
    print("\n3. Initializing RAG system...")
    rag_system = RAGSystem(config)
    
    try:
        # Initialize the system
        success = rag_system.initialize(force_reload=True)
        if success:
            print("   ✓ RAG system initialized successfully")
        else:
            print("   ⚠ RAG system initialization had issues (this is normal without Neo4j)")
        
        # Get system statistics
        print("\n4. System statistics:")
        stats = rag_system.get_system_stats()
        for key, value in stats.items():
            if key != "data_source_details":
                print(f"   {key}: {value}")
        
        # Demonstrate knowledge graph queries
        print("\n5. Knowledge Graph Queries:")
        
        # Query for employees
        print("\n   Query: 'Sarah'")
        results = rag_system.knowledge_graph.query_graph("Sarah", max_results=5) if rag_system.knowledge_graph else []
        if results:
            for result in results:
                print(f"     - {result}")
        else:
            print("     (No results - Neo4j not available)")
        
        # Query for job titles
        print("\n   Query: 'software engineer'")
        results = rag_system.knowledge_graph.query_graph("software engineer", max_results=5) if rag_system.knowledge_graph else []
        if results:
            for result in results:
                print(f"     - {result}")
        else:
            print("     (No results - Neo4j not available)")
        
        # Get entity relationships
        print("\n6. Entity Relationships:")
        if rag_system.knowledge_graph:
            relationships = rag_system.get_entity_relationships("Sarah", max_depth=2)
            print(f"   Sarah's relationships: {relationships}")
        else:
            print("   (Knowledge graph not available)")
        
        # Demonstrate RAG question answering
        print("\n7. RAG Question Answering:")
        questions = [
            "Who is Sarah Johnson?",
            "What does prismaticAI do?",
            "Who reports to Sarah?",
            "What is Michael Chen's role?"
        ]
        
        for question in questions:
            print(f"\n   Q: {question}")
            try:
                answer = rag_system.ask_question(question, use_knowledge_graph=True)
                print(f"   A: {answer.get('answer', 'No answer available')}")
                if answer.get('sources'):
                    print(f"   Sources: {len(answer['sources'])} documents")
            except Exception as e:
                print(f"   Error: {e}")
        
        # Save session
        print("\n8. Saving session...")
        session_file = rag_system.save_session("neo4j_demo_session.json")
        print(f"   Session saved to: {session_file}")
        
        print("\n" + "=" * 60)
        print("Demonstration completed!")
        print("=" * 60)
        
        print("\nTo run with actual Neo4j database:")
        print("1. Start Docker Desktop")
        print("2. Run: docker-compose up -d")
        print("3. Wait for Neo4j to start (check http://localhost:7474)")
        print("4. Run this script again")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
    
    finally:
        # Cleanup
        if 'rag_system' in locals():
            rag_system.close()
        
        # Remove test file
        if test_file.exists():
            test_file.unlink()

def show_neo4j_benefits():
    """Show the benefits of using Neo4j over JSON storage"""
    
    print("\n" + "=" * 60)
    print("Neo4j vs JSON Storage Comparison")
    print("=" * 60)
    
    benefits = {
        "Complex Queries": [
            "Find all employees who work with Sarah",
            "Find the shortest path between two employees",
            "Find all employees in the same department"
        ],
        "Performance": [
            "Faster relationship traversal",
            "Better for large graphs",
            "Indexed queries"
        ],
        "Data Integrity": [
            "ACID compliance",
            "Constraint enforcement",
            "Relationship validation"
        ],
        "Scalability": [
            "Handles millions of nodes",
            "Distributed deployment",
            "Built-in clustering"
        ],
        "Advanced Features": [
            "Graph algorithms (PageRank, shortest path)",
            "Pattern matching",
            "Temporal queries"
        ]
    }
    
    for category, features in benefits.items():
        print(f"\n{category}:")
        for feature in features:
            print(f"  • {feature}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    demonstrate_neo4j_integration()
    show_neo4j_benefits() 