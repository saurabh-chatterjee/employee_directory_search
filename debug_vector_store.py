#!/usr/bin/env python3
"""
Debug script for vector store and document retrieval
"""

from rag_system import RAGSystem
from config import get_config

def debug_vector_store():
    """Debug the vector store and document retrieval"""
    
    print("🔍 Debugging Vector Store and Document Retrieval")
    print("=" * 60)
    
    # Initialize the system
    config = get_config()
    rag_system = RAGSystem(config)
    rag_system.initialize()
    
    print(f"\n📊 System Status:")
    print(f"Documents loaded: {len(rag_system.documents)}")
    print(f"Vector store type: {rag_system.vector_manager.config.vector_store_type}")
    
    # Check what documents are loaded
    print(f"\n📄 Loaded Documents:")
    for source_name, docs in rag_system.documents.items():
        print(f"Source: {source_name}")
        print(f"  Number of documents: {len(docs)}")
        for i, doc in enumerate(docs):
            print(f"  Document {i+1}:")
            print(f"    Content: {doc.page_content[:100]}...")
            print(f"    Metadata: {doc.metadata}")
        print()
    
    # Test vector store retrieval
    print("🔍 Testing Vector Store Retrieval:")
    test_queries = ["Sarah", "Michael", "prismaticAI", "software engineer", "data scientist"]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        try:
            # Get similar documents
            similar_docs = rag_system.vector_manager.similarity_search(query, k=3)
            print(f"Found {len(similar_docs)} similar documents:")
            
            for j, doc in enumerate(similar_docs):
                print(f"  {j+1}. {doc.page_content[:100]}...")
                
        except Exception as e:
            print(f"Error: {e}")
    
    # Test RAG chain directly
    print(f"\n🧠 Testing RAG Chain:")
    test_question = "Who is Sarah?"
    print(f"Question: {test_question}")
    
    try:
        # Get similar documents for the question
        similar_docs = rag_system.vector_manager.similarity_search(test_question, k=3)
        print(f"Retrieved {len(similar_docs)} documents for context")
        
        if similar_docs:
            context = "\n".join([doc.page_content for doc in similar_docs])
            print(f"Context: {context[:200]}...")
            
            # Test LLM response
            response = rag_system.llm_manager.generate(f"Based on this context: {context}\n\nQuestion: {test_question}")
            print(f"LLM Response: {response}")
        else:
            print("No documents retrieved!")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_vector_store() 