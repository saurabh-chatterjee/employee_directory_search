#!/usr/bin/env python3
"""
Test script for OpenAI API key functionality
"""

import os
from llm_manager import OpenAILLM
from config import ModelConfig

def test_openai_api():
    """Test OpenAI API key functionality"""
    
    print("🧠 Testing OpenAI API Key")
    print("=" * 50)
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ No OPENAI_API_KEY found in environment")
        print("Please set your API key:")
        print("export OPENAI_API_KEY='your_api_key_here'")
        return False
    
    print(f"✅ Found API key: {api_key[:10]}...")
    
    # Create model config
    config = ModelConfig(
        name="GPT-3.5 Turbo Test",
        provider="openai",
        api_key=api_key,
        model_name="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=100
    )
    
    # Test LLM initialization
    print("\nTesting LLM initialization...")
    try:
        llm = OpenAILLM(config)
        print("✅ LLM initialized successfully")
        
        # Test simple generation
        print("\nTesting simple generation...")
        response = llm.generate("Say 'Hello, World!'")
        print(f"✅ Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_openai_api() 