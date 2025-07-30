"""
Example usage of the RAG Knowledge Graph System
"""
import os
from pathlib import Path

from rag_system import RAGSystem
from config import DataSourceConfig, ModelConfig, get_config

def main():
    """Example usage of the RAG system"""
    
    print("🧠 RAG Knowledge Graph System - Example Usage")
    print("=" * 50)
    
    # Initialize the RAG system
    rag_system = RAGSystem()
    
    # Initialize the system
    print("\n1. Initializing the system...")
    success = rag_system.initialize()
    
    if not success:
        print("❌ Failed to initialize system. Check your configuration and data sources.")
        return
    
    print("✅ System initialized successfully!")
    
    # Display system stats
    print("\n2. System Statistics:")
    stats = rag_system.get_system_stats()
    print(f"   - Data sources: {stats['data_sources']}")
    print(f"   - Documents loaded: {stats['loaded_documents']}")
    print(f"   - Current model: {stats['current_model']}")
    print(f"   - Knowledge graph enabled: {stats['knowledge_graph_enabled']}")
    
    # Example questions
    example_questions = [
        "What is Python?",
        "How do I create a function in Python?",
        "What are the basic data types in Python?",
        "How does Python handle exceptions?",
        "What is the difference between a list and a tuple?"
    ]
    
    print("\n3. Asking example questions:")
    print("-" * 30)
    
    for i, question in enumerate(example_questions, 1):
        print(f"\nQ{i}: {question}")
        print("-" * 20)
        
        try:
            response = rag_system.ask_question(question, use_knowledge_graph=True)
            
            print(f"Answer: {response['answer'][:200]}...")
            print(f"Model used: {response['model_used']}")
            print(f"Sources: {len(response['sources'])} documents")
            
            if response.get('knowledge_graph_insights'):
                kg_insights = response['knowledge_graph_insights']
                if kg_insights.get('entities'):
                    print(f"Related entities: {len(kg_insights['entities'])}")
                if kg_insights.get('relationships'):
                    print(f"Related relationships: {len(kg_insights['relationships'])}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
    
    # Switch models example
    print("\n4. Model switching example:")
    available_models = rag_system.llm_manager.get_available_models()
    print(f"Available models: {available_models}")
    
    if len(available_models) > 1:
        # Switch to a different model
        new_model = available_models[1] if available_models[1] != rag_system.llm_manager.get_current_model() else available_models[0]
        print(f"Switching to model: {new_model}")
        rag_system.switch_model(new_model)
        
        # Ask a question with the new model
        test_question = "What is object-oriented programming?"
        print(f"Testing with new model: {test_question}")
        
        try:
            response = rag_system.ask_question(test_question)
            print(f"Answer: {response['answer'][:200]}...")
            print(f"Model used: {response['model_used']}")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    # Knowledge graph exploration
    if rag_system.knowledge_graph:
        print("\n5. Knowledge Graph Exploration:")
        kg_stats = rag_system.knowledge_graph.get_graph_stats()
        
        if kg_stats.get('status') == 'populated':
            print(f"   - Nodes: {kg_stats['nodes']}")
            print(f"   - Edges: {kg_stats['edges']}")
            print(f"   - Density: {kg_stats['density']:.3f}")
            
            # Search for entities
            search_query = "function"
            print(f"\n   Searching for entities related to '{search_query}':")
            results = rag_system.knowledge_graph.query_graph(search_query, max_results=3)
            
            for i, result in enumerate(results, 1):
                if result['type'] == 'node':
                    print(f"     {i}. Entity: {result['id']}")
                else:
                    print(f"     {i}. Relationship: {result['source']} → {result['target']}")
            
            # Get entity relationships
            if results:
                first_entity = results[0]['id'] if results[0]['type'] == 'node' else results[0]['source']
                print(f"\n   Exploring relationships for '{first_entity}':")
                relationships = rag_system.get_entity_relationships(first_entity, max_depth=1)
                
                if 'error' not in relationships:
                    print(f"     Incoming: {len(relationships['incoming'])}")
                    print(f"     Outgoing: {len(relationships['outgoing'])}")
                    print(f"     Neighbors: {len(relationships['neighbors'])}")
                else:
                    print(f"     {relationships['error']}")
    
    # Add new data source example
    print("\n6. Adding new data source example:")
    
    # Create a simple text file for demonstration
    demo_content = """
    Machine Learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed.
    
    There are three main types of machine learning:
    1. Supervised Learning: Learning from labeled data
    2. Unsupervised Learning: Finding patterns in unlabeled data
    3. Reinforcement Learning: Learning through interaction with environment
    
    Common machine learning algorithms include:
    - Linear Regression
    - Decision Trees
    - Random Forest
    - Support Vector Machines
    - Neural Networks
    """
    
    demo_file = Path("demo_ml.txt")
    with open(demo_file, "w") as f:
        f.write(demo_content)
    
    # Add as data source
    ml_source = DataSourceConfig(
        name="Machine Learning Demo",
        type="file",
        path=str(demo_file),
        format="txt",
        chunk_size=500,
        chunk_overlap=100
    )
    
    success = rag_system.add_data_source(ml_source)
    if success:
        print("   ✅ Added Machine Learning demo data source")
        
        # Ask a question about the new content
        ml_question = "What are the main types of machine learning?"
        print(f"   Testing with new data: {ml_question}")
        
        try:
            response = rag_system.ask_question(ml_question)
            print(f"   Answer: {response['answer'][:200]}...")
        except Exception as e:
            print(f"   Error: {str(e)}")
    else:
        print("   ❌ Failed to add data source")
    
    # Save session
    print("\n7. Saving session...")
    session_file = rag_system.save_session("example_session.json")
    print(f"   Session saved to: {session_file}")
    
    # Display final stats
    print("\n8. Final System Statistics:")
    final_stats = rag_system.get_system_stats()
    print(f"   - Questions asked: {final_stats['session_questions']}")
    print(f"   - Documents loaded: {final_stats['loaded_documents']}")
    print(f"   - Data sources: {final_stats['data_sources']}")
    
    # Clean up demo file
    if demo_file.exists():
        demo_file.unlink()
    
    print("\n✅ Example usage completed!")
    print("\nTo run the web interface, use: streamlit run app.py")

def interactive_mode():
    """Interactive mode for testing the system"""
    print("🧠 RAG System - Interactive Mode")
    print("=" * 40)
    
    rag_system = RAGSystem()
    
    # Initialize
    print("Initializing system...")
    if not rag_system.initialize():
        print("Failed to initialize system.")
        return
    
    print("System ready! Type 'quit' to exit, 'help' for commands.")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'help':
                print("Commands:")
                print("  ask <question> - Ask a question")
                print("  stats - Show system statistics")
                print("  models - List available models")
                print("  switch <model> - Switch to a different model")
                print("  kg <query> - Search knowledge graph")
                print("  quit - Exit")
            elif user_input.lower().startswith('ask '):
                question = user_input[4:]
                response = rag_system.ask_question(question)
                print(f"\nAnswer: {response['answer']}")
                print(f"Sources: {len(response['sources'])} documents")
            elif user_input.lower() == 'stats':
                stats = rag_system.get_system_stats()
                print(f"Documents: {stats['loaded_documents']}")
                print(f"Questions: {stats['session_questions']}")
                print(f"Current model: {stats['current_model']}")
            elif user_input.lower() == 'models':
                models = rag_system.llm_manager.get_available_models()
                current = rag_system.llm_manager.get_current_model()
                for model in models:
                    marker = " (current)" if model == current else ""
                    print(f"  {model}{marker}")
            elif user_input.lower().startswith('switch '):
                model = user_input[7:]
                if rag_system.switch_model(model):
                    print(f"Switched to {model}")
                else:
                    print(f"Failed to switch to {model}")
            elif user_input.lower().startswith('kg '):
                query = user_input[3:]
                if rag_system.knowledge_graph:
                    results = rag_system.knowledge_graph.query_graph(query)
                    print(f"Found {len(results)} results:")
                    for result in results[:3]:
                        if result['type'] == 'node':
                            print(f"  Entity: {result['id']}")
                        else:
                            print(f"  Relationship: {result['source']} → {result['target']}")
                else:
                    print("Knowledge graph not available")
            else:
                print("Unknown command. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("Goodbye!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        main() 