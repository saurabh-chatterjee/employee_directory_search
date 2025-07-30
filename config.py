"""
Configuration file for the RAG Knowledge Graph System
"""
import os
from typing import Dict, List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class ModelConfig(BaseModel):
    """Configuration for different LLM models"""
    name: str
    provider: str  # "openai", "anthropic", "local", "huggingface"
    api_key: Optional[str] = None
    model_name: str
    temperature: float = 0.1
    max_tokens: int = 2000
    base_url: Optional[str] = None

class DataSourceConfig(BaseModel):
    """Configuration for data sources"""
    name: str
    type: str  # "file", "url", "database", "api"
    path: str
    format: str  # "txt", "pdf", "md", "html", "json"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    enabled: bool = True

class Neo4jConfig(BaseModel):
    """Configuration for Neo4j database"""
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"

class SystemConfig(BaseModel):
    """Main system configuration"""
    models: Dict[str, ModelConfig]
    data_sources: List[DataSourceConfig]
    vector_store_type: str = "chroma"  # "chroma", "faiss", "pinecone"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    knowledge_graph_enabled: bool = True
    knowledge_graph_type: str = "neo4j"  # "neo4j", "json"
    neo4j_config: Neo4jConfig = Neo4jConfig()
    cache_dir: str = "./cache"
    output_dir: str = "./output"

# Default model configurations
DEFAULT_MODELS = {
    "gpt-3.5-turbo": ModelConfig(
        name="GPT-3.5 Turbo",
        provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=2000
    )
}

# Default data sources (using local test file)
DEFAULT_DATA_SOURCES = [
    DataSourceConfig(
        name="Local Test Data",
        type="file",
        path="./test.txt",
        format="txt",
        chunk_size=500,
        chunk_overlap=100
    )
]

# Create default system configuration
DEFAULT_CONFIG = SystemConfig(
    models=DEFAULT_MODELS,
    data_sources=DEFAULT_DATA_SOURCES,
    vector_store_type="chroma",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    knowledge_graph_enabled=True,
    cache_dir="./cache",
    output_dir="./output"
)

def get_config() -> SystemConfig:
    """Get the current system configuration"""
    return DEFAULT_CONFIG

def update_model_config(model_name: str, config: ModelConfig):
    """Update a specific model configuration"""
    DEFAULT_CONFIG.models[model_name] = config

def add_data_source(source: DataSourceConfig):
    """Add a new data source"""
    DEFAULT_CONFIG.data_sources.append(source)

def remove_data_source(source_name: str):
    """Remove a data source by name"""
    DEFAULT_CONFIG.data_sources = [
        source for source in DEFAULT_CONFIG.data_sources 
        if source.name != source_name
    ] 