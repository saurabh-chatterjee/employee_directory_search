"""
Data loader module for handling various data sources
"""
import os
import requests
from typing import List, Dict, Any
from pathlib import Path
from urllib.parse import urlparse
import json

from langchain.document_loaders import (
    TextLoader, 
    PDFMinerLoader, 
    UnstructuredMarkdownLoader,
    WebBaseLoader,
    JSONLoader,
    CSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from bs4 import BeautifulSoup

from config import DataSourceConfig

class DataLoader:
    """Handles loading and processing of various data sources"""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_documents(self) -> List[Document]:
        """Load documents based on the data source configuration"""
        if self.config.type == "file":
            return self._load_file()
        elif self.config.type == "url":
            return self._load_url()
        elif self.config.type == "database":
            return self._load_database()
        elif self.config.type == "api":
            return self._load_api()
        else:
            raise ValueError(f"Unsupported data source type: {self.config.type}")
    
    def _load_file(self) -> List[Document]:
        """Load documents from file"""
        file_path = Path(self.config.path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if self.config.format.lower() == "txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
        elif self.config.format.lower() == "pdf":
            loader = PDFMinerLoader(str(file_path))
        elif self.config.format.lower() == "md":
            loader = UnstructuredMarkdownLoader(str(file_path))
        elif self.config.format.lower() == "json":
            loader = JSONLoader(
                file_path=str(file_path),
                jq_schema='.[]',
                text_content=False
            )
        elif self.config.format.lower() == "csv":
            loader = CSVLoader(str(file_path))
        else:
            raise ValueError(f"Unsupported file format: {self.config.format}")
        
        documents = loader.load()
        return self._split_documents(documents)
    
    def _load_url(self) -> List[Document]:
        """Load documents from URL"""
        try:
            loader = WebBaseLoader(self.config.path)
            documents = loader.load()
            return self._split_documents(documents)
        except Exception as e:
            raise Exception(f"Failed to load URL {self.config.path}: {str(e)}")
    
    def _load_database(self) -> List[Document]:
        """Load documents from database (placeholder for future implementation)"""
        # This would be implemented based on specific database requirements
        raise NotImplementedError("Database loading not yet implemented")
    
    def _load_api(self) -> List[Document]:
        """Load documents from API"""
        try:
            response = requests.get(self.config.path)
            response.raise_for_status()
            
            # Try to parse as JSON first
            try:
                data = response.json()
                # Convert JSON to text representation
                text_content = json.dumps(data, indent=2)
            except json.JSONDecodeError:
                # If not JSON, treat as text
                text_content = response.text
            
            document = Document(
                page_content=text_content,
                metadata={"source": self.config.path, "type": "api"}
            )
            
            return self._split_documents([document])
        except Exception as e:
            raise Exception(f"Failed to load API {self.config.path}: {str(e)}")
    
    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        return self.text_splitter.split_documents(documents)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the loaded data source"""
        return {
            "name": self.config.name,
            "type": self.config.type,
            "path": self.config.path,
            "format": self.config.format,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "enabled": self.config.enabled
        }

class DataSourceManager:
    """Manages multiple data sources"""
    
    def __init__(self, data_sources: List[DataSourceConfig]):
        self.data_sources = data_sources
        self.loaders = {}
        self.documents = {}
        
    def load_all_sources(self) -> Dict[str, List[Document]]:
        """Load all enabled data sources"""
        for source in self.data_sources:
            if source.enabled:
                try:
                    loader = DataLoader(source)
                    documents = loader.load_documents()
                    self.documents[source.name] = documents
                    self.loaders[source.name] = loader
                    print(f"Loaded {len(documents)} documents from {source.name}")
                except Exception as e:
                    print(f"Failed to load {source.name}: {str(e)}")
        
        return self.documents
    
    def get_all_documents(self) -> List[Document]:
        """Get all documents from all sources as a flat list"""
        all_docs = []
        for docs in self.documents.values():
            all_docs.extend(docs)
        return all_docs
    
    def get_source_documents(self, source_name: str) -> List[Document]:
        """Get documents from a specific source"""
        return self.documents.get(source_name, [])
    
    def get_source_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all sources"""
        metadata = {}
        for name, loader in self.loaders.items():
            metadata[name] = loader.get_metadata()
        return metadata 