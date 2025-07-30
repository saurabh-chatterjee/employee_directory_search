"""
Vector store module for document embeddings and similarity search
"""
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.vectorstores.base import VectorStore

from config import SystemConfig

class VectorStoreManager:
    """Manages vector stores for document embeddings"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.embeddings = self._initialize_embeddings()
        self.vector_store = None
        self.vector_store_path = Path(config.cache_dir) / "vector_store"
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
    
    def _initialize_embeddings(self):
        """Initialize the embedding model"""
        try:
            return HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except ImportError:
            # Fallback to a simple TF-IDF based approach
            print("Warning: sentence-transformers not available, using fallback embedding method")
            return self._create_fallback_embeddings()
    
    def _create_fallback_embeddings(self):
        """Create a simple fallback embedding method"""
        import numpy as np
        import re
        
        class FallbackEmbeddings:
            def __init__(self):
                self.dimension = 384  # Standard embedding dimension
                self.fitted = False
                self.vocabulary = set()
                self.word_to_idx = {}
                self.idx = 0
            
            def _preprocess_text(self, text):
                """Simple text preprocessing"""
                # Convert to lowercase and split into words
                words = re.findall(r'\b\w+\b', text.lower())
                return words
            
            def _build_vocabulary(self, texts):
                """Build vocabulary from all texts"""
                for text in texts:
                    words = self._preprocess_text(text)
                    for word in words:
                        if word not in self.vocabulary:
                            self.vocabulary.add(word)
                            self.word_to_idx[word] = self.idx
                            self.idx += 1
            
            def _text_to_vector(self, text):
                """Convert text to a simple vector representation"""
                words = self._preprocess_text(text)
                vector = np.zeros(self.dimension)
                
                for word in words:
                    if word in self.word_to_idx:
                        # Use word index to set a position in the vector
                        idx = self.word_to_idx[word] % self.dimension
                        vector[idx] += 1
                
                # Normalize the vector
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
                
                return vector
            
            def embed_documents(self, texts):
                """Create embeddings for documents"""
                if not self.fitted:
                    self._build_vocabulary(texts)
                    self.fitted = True
                
                embeddings = []
                for text in texts:
                    embedding = self._text_to_vector(text)
                    embeddings.append(embedding.tolist())
                
                return embeddings
            
            def embed_query(self, text):
                """Create embedding for a single query"""
                if not self.fitted:
                    # If not fitted, fit on this text
                    self._build_vocabulary([text])
                    self.fitted = True
                
                embedding = self._text_to_vector(text)
                return embedding.tolist()
        
        return FallbackEmbeddings()
    
    def create_vector_store(self, documents: List[Document], force_recreate: bool = False):
        """Create or load vector store from documents"""
        # Store documents for keyword search fallback
        self.all_documents = documents
        
        if self.vector_store is not None and not force_recreate:
            return self.vector_store
        
        if self.config.vector_store_type == "chroma":
            self.vector_store = self._create_chroma_store(documents, force_recreate)
        elif self.config.vector_store_type == "faiss":
            self.vector_store = self._create_faiss_store(documents, force_recreate)
        else:
            raise ValueError(f"Unsupported vector store type: {self.config.vector_store_type}")
        
        return self.vector_store
    
    def _create_chroma_store(self, documents: List[Document], force_recreate: bool = False):
        """Create Chroma vector store"""
        chroma_path = self.vector_store_path / "chroma"
        
        if chroma_path.exists() and not force_recreate:
            # Load existing store
            return Chroma(
                persist_directory=str(chroma_path),
                embedding_function=self.embeddings
            )
        else:
            # Create new store
            if chroma_path.exists():
                import shutil
                shutil.rmtree(chroma_path)
            
            store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(chroma_path)
            )
            store.persist()
            return store
    
    def _create_faiss_store(self, documents: List[Document], force_recreate: bool = False):
        """Create FAISS vector store"""
        faiss_path = self.vector_store_path / "faiss"
        
        if faiss_path.exists() and not force_recreate:
            # Load existing store
            return FAISS.load_local(str(faiss_path), self.embeddings)
        else:
            # Create new store
            store = FAISS.from_documents(documents, self.embeddings)
            store.save_local(str(faiss_path))
            return store
    
    def similarity_search(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[Document]:
        """Perform similarity search with keyword fallback"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_vector_store first.")
        
        try:
            # Try vector similarity first
            results = self.vector_store.similarity_search(query, k=k, filter=filter_dict)
            
            # If no results, fall back to keyword search
            if not results and hasattr(self, 'all_documents'):
                print(f"No vector results for '{query}', trying keyword search...")
                results = self._keyword_search(query, k=k)
            
            return results
        except Exception as e:
            print(f"Error in similarity search: {e}")
            # Fall back to keyword search
            if hasattr(self, 'all_documents'):
                return self._keyword_search(query, k=k)
            return []
    
    def _keyword_search(self, query: str, k: int = 5) -> List[Document]:
        """Simple keyword-based search as fallback"""
        if not hasattr(self, 'all_documents') or not self.all_documents:
            return []
        
        query_words = query.lower().split()
        scored_docs = []
        
        for doc in self.all_documents:
            doc_text = doc.page_content.lower()
            score = 0
            
            # Count matching words
            for word in query_words:
                if word in doc_text:
                    score += 1
            
            # Normalize by query length
            if len(query_words) > 0:
                score = score / len(query_words)
            
            if score > 0:
                scored_docs.append((doc, score))
        
        # Sort by score and return top k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:k]]
    
    def similarity_search_with_score(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[tuple]:
        """Perform similarity search with scores and keyword fallback"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_vector_store first.")
        
        try:
            # Try vector similarity first
            results = self.vector_store.similarity_search_with_score(query, k=k, filter=filter_dict)
            
            # If no results, fall back to keyword search
            if not results and hasattr(self, 'all_documents'):
                print(f"No vector results for '{query}', trying keyword search...")
                keyword_results = self._keyword_search(query, k=k)
                # Convert to format with scores
                results = [(doc, 1.0) for doc in keyword_results]  # Give perfect score for keyword matches
            
            return results
        except Exception as e:
            print(f"Error in similarity search with score: {e}")
            # Fall back to keyword search
            if hasattr(self, 'all_documents'):
                keyword_results = self._keyword_search(query, k=k)
                return [(doc, 1.0) for doc in keyword_results]
            return []
    
    def add_documents(self, documents: List[Document]):
        """Add new documents to the vector store"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_vector_store first.")
        
        if hasattr(self.vector_store, 'add_documents'):
            self.vector_store.add_documents(documents)
        else:
            # For stores that don't support adding documents, recreate
            all_docs = self.vector_store.docstore._dict.values()
            all_docs.extend(documents)
            self.create_vector_store(list(all_docs), force_recreate=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        if self.vector_store is None:
            return {"status": "not_initialized"}
        
        stats = {
            "type": self.config.vector_store_type,
            "embedding_model": self.config.embedding_model,
            "status": "initialized"
        }
        
        if hasattr(self.vector_store, 'docstore'):
            stats["document_count"] = len(self.vector_store.docstore._dict)
        
        return stats
    
    def clear(self):
        """Clear the vector store"""
        if self.vector_store_path.exists():
            import shutil
            shutil.rmtree(self.vector_store_path)
        self.vector_store = None 