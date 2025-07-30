"""
LLM Manager module for handling different language models
"""
import os
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

import requests
import json
from langchain_community.llms import Anthropic, Ollama
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from config import ModelConfig

class BaseLLM(ABC):
    """Abstract base class for LLM implementations"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    def chat(self, messages: List[BaseMessage], **kwargs) -> str:
        """Generate response from chat messages"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        pass

class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        try:
            if config.api_key:
                # Use requests to call OpenAI API directly
                self.api_key = config.api_key
                self.model = config.model_name
                self.temperature = config.temperature
                self.max_tokens = config.max_tokens
                self.base_url = "https://api.openai.com/v1"
                print("✅ OpenAI API configured successfully")
            else:
                raise Exception("No API key provided")
        except Exception as e:
            print(f"Warning: Could not initialize OpenAI API: {e}")
            # Create a dummy LLM for testing
            self.api_key = None
            self.model = "dummy"
            self.temperature = 0.1
            self.max_tokens = 100
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI API directly"""
        try:
            if self.api_key:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                }
                
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    print(f"API Error: {response.status_code} - {response.text}")
                    return f"API Error: {response.status_code}"
            else:
                return f"Dummy response to: {prompt[:50]}..."
        except Exception as e:
            print(f"Error generating text: {e}")
            return f"Error: {e}"
    
    def chat(self, messages: List[BaseMessage], **kwargs) -> str:
        """Generate text from messages using OpenAI API directly"""
        try:
            if self.api_key:
                # Convert LangChain messages to OpenAI format
                openai_messages = []
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        openai_messages.append({"role": "user", "content": msg.content})
                    elif isinstance(msg, SystemMessage):
                        openai_messages.append({"role": "system", "content": msg.content})
                    else:
                        openai_messages.append({"role": "user", "content": str(msg.content)})
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": self.model,
                    "messages": openai_messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                }
                
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    print(f"API Error: {response.status_code} - {response.text}")
                    return f"API Error: {response.status_code}"
            else:
                return f"Dummy response to: {str(messages)[:50]}..."
        except Exception as e:
            print(f"Error generating messages: {e}")
            return f"Error: {e}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        return {
            "provider": "openai",
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }

class AnthropicLLM(BaseLLM):
    """Anthropic LLM implementation"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.llm = Anthropic(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            anthropic_api_key=config.api_key
        )
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        return self.llm.predict(prompt, **kwargs)
    
    def chat(self, messages: List[BaseMessage], **kwargs) -> str:
        """Generate response from chat messages"""
        return self.llm.predict_messages(messages, **kwargs).content
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        return {
            "provider": "anthropic",
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }

class OllamaLLM(BaseLLM):
    """Ollama LLM implementation for local models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.llm = Ollama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=config.base_url or "http://localhost:11434"
        )
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        return self.llm.predict(prompt, **kwargs)
    
    def chat(self, messages: List[BaseMessage], **kwargs) -> str:
        """Generate response from chat messages"""
        # Convert messages to prompt
        prompt = ""
        for message in messages:
            if isinstance(message, SystemMessage):
                prompt += f"System: {message.content}\n"
            elif isinstance(message, HumanMessage):
                prompt += f"Human: {message.content}\n"
            else:
                prompt += f"Assistant: {message.content}\n"
        prompt += "Assistant: "
        
        return self.llm.predict(prompt, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        return {
            "provider": "ollama",
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "base_url": self.config.base_url
        }

class LLMManager:
    """Manages different LLM providers and model switching"""
    
    def __init__(self, models: Dict[str, ModelConfig]):
        self.models = models
        self.current_model_name = list(models.keys())[0] if models else None
        self.current_llm = None
        self._initialize_current_model()
    
    def _initialize_current_model(self):
        """Initialize the current model"""
        if self.current_model_name and self.current_model_name in self.models:
            self.current_llm = self._create_llm(self.models[self.current_model_name])
    
    def _create_llm(self, config: ModelConfig) -> BaseLLM:
        """Create LLM instance based on provider"""
        if config.provider == "openai":
            return OpenAILLM(config)
        elif config.provider == "anthropic":
            return AnthropicLLM(config)
        elif config.provider == "local":
            return OllamaLLM(config)
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different model"""
        if model_name not in self.models:
            print(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
            return False
        
        self.current_model_name = model_name
        self.current_llm = self._create_llm(self.models[model_name])
        print(f"Switched to model: {model_name}")
        return True
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        return list(self.models.keys())
    
    def get_current_model(self) -> str:
        """Get current model name"""
        return self.current_model_name
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using current model"""
        if not self.current_llm:
            raise ValueError("No model initialized")
        return self.current_llm.generate(prompt, **kwargs)
    
    def chat(self, messages: List[BaseMessage], **kwargs) -> str:
        """Generate chat response using current model"""
        if not self.current_llm:
            raise ValueError("No model initialized")
        return self.current_llm.chat(messages, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current model"""
        if not self.current_llm:
            return {"error": "No model initialized"}
        return self.current_llm.get_model_info()
    
    def add_model(self, name: str, config: ModelConfig):
        """Add a new model configuration"""
        self.models[name] = config
        print(f"Added model: {name}")
    
    def remove_model(self, name: str) -> bool:
        """Remove a model configuration"""
        if name not in self.models:
            return False
        
        if self.current_model_name == name:
            # Switch to first available model
            available_models = [k for k in self.models.keys() if k != name]
            if available_models:
                self.switch_model(available_models[0])
            else:
                self.current_model_name = None
                self.current_llm = None
        
        del self.models[name]
        print(f"Removed model: {name}")
        return True

class RAGChain:
    """RAG chain for question answering"""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        
        # Define RAG prompt template
        self.rag_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are a helpful assistant that answers questions based on the provided context.
            
            Context:
            {context}
            
            Question: {question}
            
            Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to answer the question, please say so.
            
            Answer:"""
        )
        
        # Don't initialize chain with dummy LLM to avoid validation errors
        self.chain = None
    
    def answer_question(self, question: str, context: str) -> str:
        """Answer a question using RAG"""
        if not self.llm_manager.current_llm:
            raise ValueError("No model initialized")
        
        # Use the LLM directly for better control
        prompt = self.rag_prompt.format(context=context, question=question)
        return self.llm_manager.generate(prompt)
    
    def answer_with_sources(self, question: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Answer question with source information"""
        # Combine context from multiple documents
        context_parts = []
        sources = []
        
        for doc in context_docs:
            context_parts.append(doc.get("content", ""))
            sources.append({
                "source": doc.get("metadata", {}).get("source", "Unknown"),
                "score": doc.get("score", 0)
            })
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        answer = self.answer_question(question, context)
        
        return {
            "answer": answer,
            "sources": sources,
            "context": context[:500] + "..." if len(context) > 500 else context
        } 