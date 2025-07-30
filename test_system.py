"""
Simple test script to verify the RAG system components
"""
import os
import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from config import get_config, SystemConfig, ModelConfig, DataSourceConfig
        print("✅ Config module imported successfully")
    except ImportError as e:
        print(f"❌ Config module import failed: {e}")
        return False
    
    try:
        from data_loader import DataLoader, DataSourceManager
        print("✅ Data loader module imported successfully")
    except ImportError as e:
        print(f"❌ Data loader module import failed: {e}")
        return False
    
    try:
        from vector_store import VectorStoreManager
        print("✅ Vector store module imported successfully")
    except ImportError as e:
        print(f"❌ Vector store module import failed: {e}")
        return False
    
    try:
        from knowledge_graph import KnowledgeGraphManager
        print("✅ Knowledge graph module imported successfully")
    except ImportError as e:
        print(f"❌ Knowledge graph module import failed: {e}")
        return False
    
    try:
        from llm_manager import LLMManager, RAGChain
        print("✅ LLM manager module imported successfully")
    except ImportError as e:
        print(f"❌ LLM manager module import failed: {e}")
        return False
    
    try:
        from rag_system import RAGSystem
        print("✅ RAG system module imported successfully")
    except ImportError as e:
        print(f"❌ RAG system module import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration system"""
    print("\nTesting configuration...")
    
    try:
        from config import get_config, ModelConfig, DataSourceConfig
        
        config = get_config()
        print(f"✅ Configuration loaded: {len(config.models)} models, {len(config.data_sources)} data sources")
        
        # Test model config
        model_config = ModelConfig(
            name="Test Model",
            provider="openai",
            model_name="gpt-3.5-turbo",
            temperature=0.1
        )
        print(f"✅ Model config created: {model_config.name}")
        
        # Test data source config
        source_config = DataSourceConfig(
            name="Test Source",
            type="file",
            path="./test.txt",
            format="txt"
        )
        print(f"✅ Data source config created: {source_config.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_data_loader():
    """Test data loading functionality"""
    print("\nTesting data loader...")
    
    try:
        from data_loader import DataLoader, DataSourceManager
        from config import DataSourceConfig
        
        # Create test data source
        source_config = DataSourceConfig(
            name="Test Data",
            type="file",
            path="./test.txt",
            format="txt",
            chunk_size=500,
            chunk_overlap=100
        )
        
        # Test data loader
        loader = DataLoader(source_config)
        print(f"✅ Data loader created for {source_config.name}")
        
        # Test data source manager
        manager = DataSourceManager([source_config])
        print(f"✅ Data source manager created with {len(manager.data_sources)} sources")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        return False

def test_vector_store():
    """Test vector store functionality"""
    print("\nTesting vector store...")
    
    try:
        from vector_store import VectorStoreManager
        from config import get_config
        
        config = get_config()
        vector_manager = VectorStoreManager(config)
        print(f"✅ Vector store manager created with {config.vector_store_type} backend")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        return False

def test_llm_manager():
    """Test LLM manager functionality"""
    print("\nTesting LLM manager...")
    
    try:
        from llm_manager import LLMManager
        from config import get_config
        
        config = get_config()
        llm_manager = LLMManager(config.models)
        print(f"✅ LLM manager created with {len(config.models)} models")
        
        current_model = llm_manager.get_current_model()
        print(f"✅ Current model: {current_model}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM manager test failed: {e}")
        return False

def test_rag_system():
    """Test RAG system initialization"""
    print("\nTesting RAG system...")
    
    try:
        from rag_system import RAGSystem
        
        rag_system = RAGSystem()
        print("✅ RAG system created")
        
        # Test system stats
        stats = rag_system.get_system_stats()
        print(f"✅ System stats retrieved: {stats['data_sources']} data sources")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG system test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧠 RAG Knowledge Graph System - Component Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_data_loader,
        test_vector_store,
        test_llm_manager,
        test_rag_system
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Set up your API keys in a .env file")
        print("2. Run: streamlit run app.py")
        print("3. Or run: python example_usage.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon issues:")
        print("1. Missing dependencies - run: pip install -r requirements.txt")
        print("2. Missing API keys - create a .env file with your keys")
        print("3. Missing test.txt file - ensure it exists in the current directory")

if __name__ == "__main__":
    main() 