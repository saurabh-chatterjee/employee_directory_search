#!/usr/bin/env python3
"""
Simple test to debug RAG system step by step
"""

from rag_system import RAGSystem
from config import get_config

def test_simple_rag():
    """Test RAG system step by step"""
    
    print("🧠 Simple RAG Test")
    print("=" * 40)
    
    # Initialize the system
    config = get_config()
    rag_system = RAGSystem(config)
    rag_system.initialize()
    
    # Test question
    question = "Who is Sarah?"
    print(f"Question: {question}")
    
    # Step 1: Test vector store retrieval
    print(f"\n1. Testing vector store retrieval...")
    try:
        context_docs = rag_system.vector_manager.similarity_search_with_score(question, k=3)
        print(f"Retrieved {len(context_docs)} documents")
        
        for i, (doc, score) in enumerate(context_docs):
            print(f"  Doc {i+1}: {doc.page_content[:100]}... (score: {score:.3f})")
            
    except Exception as e:
        print(f"Error in vector store retrieval: {e}")
    
    # Step 2: Test RAG chain directly
    print(f"\n2. Testing RAG chain directly...")
    try:
        # Create context data manually
        context_data = []
        for doc, score in context_docs:
            context_data.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            })
        
        print(f"Context data: {len(context_data)} documents")
        for i, doc in enumerate(context_data):
            print(f"  Doc {i+1}: {doc['content'][:100]}...")
        
        # Test RAG chain
        rag_response = rag_system.rag_chain.answer_with_sources(question, context_data)
        print(f"RAG Response: {rag_response['answer']}")
        
    except Exception as e:
        print(f"Error in RAG chain: {e}")
    
    # Step 3: Test full ask_question
    print(f"\n3. Testing full ask_question...")
    try:
        result = rag_system.ask_question(question)
        print(f"Full Result: {result['answer']}")
        
    except Exception as e:
        print(f"Error in ask_question: {e}")

if __name__ == "__main__":
    test_simple_rag() 