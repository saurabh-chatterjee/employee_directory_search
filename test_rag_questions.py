#!/usr/bin/env python3
"""
Test script for RAG questions about loaded documents
"""

from rag_system import RAGSystem
from config import get_config

def test_rag_questions():
    """Test RAG system with questions about loaded documents"""
    
    print("🧠 Testing RAG Questions About Loaded Documents")
    print("=" * 60)
    
    # Initialize the system
    config = get_config()
    rag_system = RAGSystem(config)
    rag_system.initialize()
    
    # Questions about the loaded documents
    questions = [
        "Who is Sarah?",
        "What does Michael do?",
        "What is prismaticAI?",
        "How long has Sarah been working at prismaticAI?",
        "What is Michael's role at the company?",
        "What type of company is prismaticAI?",
        "Where is prismaticAI located?",
        "What do Sarah and Michael work on together?"
    ]
    
    print("Asking questions about the loaded documents...\n")
    
    for i, question in enumerate(questions, 1):
        print(f"Q{i}: {question}")
        print("-" * 40)
        
        try:
            result = rag_system.ask_question(question)
            print(f"Answer: {result['answer']}")
            print(f"Model: {result.get('model_used', 'Unknown')}")
            print(f"Sources: {len(result.get('sources', []))} documents")
            if result.get('sources'):
                print("Source documents:")
                for j, source in enumerate(result['sources'], 1):
                    print(f"  {j}. {source['content'][:100]}...")
            print()
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    test_rag_questions() 