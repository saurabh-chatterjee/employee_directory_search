# RAG Knowledge Graph System

A comprehensive Retrieval-Augmented Generation (RAG) system with knowledge graph capabilities that allows users to ask natural language questions over documentation and get intelligent, context-aware answers.

## 🚀 Features

### Core RAG Capabilities
- **Multi-source Document Loading**: Support for files (TXT, PDF, MD, JSON, CSV), URLs, and APIs
- **Vector-based Retrieval**: Advanced similarity search using embeddings
- **Context-aware Generation**: Intelligent answers based on retrieved context
- **Source Attribution**: Track and display information sources

### Knowledge Graph Integration
- **Entity Extraction**: Automatically extract entities and relationships from documents
- **Graph-based Querying**: Search and explore knowledge graphs
- **Relationship Analysis**: Understand connections between concepts
- **Visual Graph Representation**: Interactive graph visualizations
- **Neo4j Database Support**: Store entity relationships in a proper graph database
- **Advanced Graph Queries**: Complex relationship traversal and pattern matching

### Model Flexibility
- **Multi-LLM Support**: OpenAI GPT models, Anthropic Claude, local models (Ollama)
- **Dynamic Model Switching**: Change models on-the-fly
- **Configurable Parameters**: Temperature, max tokens, etc.
- **Provider Abstraction**: Unified interface for different LLM providers

### User Interface
- **Streamlit Web App**: Beautiful, interactive web interface
- **Real-time Querying**: Instant question-answering
- **System Monitoring**: Live statistics and performance metrics
- **Configuration Management**: Easy setup and customization

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Vector Store   │    │  Knowledge      │
│                 │    │                 │    │  Graph          │
│ • Files         │───▶│ • Chroma        │    │ • Entity        │
│ • URLs          │    │ • FAISS         │    │   Extraction    │
│ • APIs          │    │ • Pinecone      │    │ • Relationship  │
└─────────────────┘    └─────────────────┘    │   Mapping       │
                                              └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RAG System Core                              │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   LLM       │  │  RAG        │  │  Query      │            │
│  │  Manager    │  │  Chain      │  │  Processor  │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│   Web Interface │
│   (Streamlit)   │
└─────────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag-knowledge-graph-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   echo "ANTHROPIC_API_KEY=your_anthropic_api_key_here" >> .env
   ```

4. **Set up Neo4j (Optional - for advanced knowledge graph features)**
   ```bash
   # Using Docker Compose (recommended)
   docker-compose up -d
   
   # Or using Docker directly
   docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
   
   # Access Neo4j browser at http://localhost:7474
   # Default credentials: neo4j/password
   ```

5. **Run the system**
   ```bash
   # Web interface
   streamlit run app.py
   
   # Command line example
   python example_usage.py
   
   # Interactive mode
   python example_usage.py --interactive
   
   # Test Neo4j knowledge graph
   python test_neo4j_knowledge_graph.py
   ```

## 🎯 Usage

### Neo4j Knowledge Graph Configuration

The system supports two knowledge graph storage backends:

1. **JSON Storage (Default)**: Simple file-based storage for basic use cases
2. **Neo4j Database**: Advanced graph database for complex relationship queries

To use Neo4j for knowledge graph storage:

1. **Update configuration in `config.py`**:
   ```python
   from config import SystemConfig, Neo4jConfig
   
   config = SystemConfig(
       # ... other config ...
       knowledge_graph_type="neo4j",  # Use Neo4j instead of JSON
       neo4j_config=Neo4jConfig(
           uri="bolt://localhost:7687",
           user="neo4j",
           password="password"
       )
   )
   ```

2. **Benefits of Neo4j**:
   - Complex relationship queries
   - Graph traversal algorithms
   - Better performance for large graphs
   - ACID compliance
   - Built-in graph visualization

3. **Example Neo4j queries**:
   ```python
   # Get all employees at a company
   relationships = kg_manager.get_entity_relationships("prismaticAI", max_depth=1)
   
   # Find all software engineers
   results = kg_manager.query_graph("software engineer")
   
   # Get graph statistics
   stats = kg_manager.get_graph_stats()
   ```

### Web Interface

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Initialize the system**
   - Go to the Dashboard
   - Click "Initialize System"
   - Wait for data loading and processing

3. **Ask questions**
   - Navigate to "Ask Questions"
   - Enter your question
   - Configure options (model, knowledge graph, etc.)
   - Get instant answers with sources

4. **Explore features**
   - **System Stats**: Monitor performance and usage
   - **Configuration**: Add/remove data sources and models
   - **Knowledge Graph**: Explore entities and relationships

### Programmatic Usage

```python
from rag_system import RAGSystem

# Initialize system
rag = RAGSystem()
rag.initialize()

# Ask questions
response = rag.ask_question("What is Python?")
print(response['answer'])

# Switch models
rag.switch_model("gpt-4")

# Add data sources
from config import DataSourceConfig
new_source = DataSourceConfig(
    name="My Docs",
    type="file",
    path="./my_documents.txt",
    format="txt"
)
rag.add_data_source(new_source)
```

### Configuration

The system is highly configurable through the `config.py` file:

```python
# Add new models
from config import ModelConfig
new_model = ModelConfig(
    name="My Model",
    provider="openai",
    model_name="gpt-4",
    temperature=0.1
)

# Add data sources
from config import DataSourceConfig
new_source = DataSourceConfig(
    name="API Data",
    type="api",
    path="https://api.example.com/data",
    format="json"
)
```

## 📊 System Components

### 1. Data Loader (`data_loader.py`)
- **Multi-format support**: TXT, PDF, MD, JSON, CSV, HTML
- **Source types**: Files, URLs, APIs, Databases
- **Chunking**: Configurable text splitting with overlap
- **Error handling**: Robust loading with fallbacks

### 2. Vector Store (`vector_store.py`)
- **Multiple backends**: Chroma, FAISS, Pinecone
- **Embedding models**: Sentence transformers, OpenAI embeddings
- **Similarity search**: Configurable k-nearest neighbors
- **Persistence**: Save/load vector stores

### 3. Knowledge Graph (`knowledge_graph.py`)
- **Entity extraction**: LLM-powered entity recognition
- **Relationship mapping**: Automatic relationship discovery
- **Graph queries**: Natural language graph search
- **Visualization**: Interactive graph plots

### 4. LLM Manager (`llm_manager.py`)
- **Provider abstraction**: OpenAI, Anthropic, Ollama
- **Model switching**: Dynamic model selection
- **Parameter control**: Temperature, tokens, etc.
- **RAG chains**: Optimized question-answering

### 5. RAG System (`rag_system.py`)
- **Orchestration**: Coordinates all components
- **Session management**: Track conversations
- **Performance monitoring**: System statistics
- **Caching**: Efficient data reuse

## 🔧 Configuration Options

### Model Configuration
```python
{
    "name": "GPT-4",
    "provider": "openai",
    "model_name": "gpt-4",
    "temperature": 0.1,
    "max_tokens": 2000,
    "api_key": "your_key"
}
```

### Data Source Configuration
```python
{
    "name": "Python Docs",
    "type": "url",
    "path": "https://docs.python.org/3/tutorial/",
    "format": "html",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "enabled": True
}
```

### System Configuration
```python
{
    "vector_store_type": "chroma",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "knowledge_graph_enabled": True,
    "cache_dir": "./cache",
    "output_dir": "./output"
}
```

## 📈 Performance Features

- **Caching**: Vector stores and knowledge graphs are cached
- **Lazy loading**: Components initialize on-demand
- **Batch processing**: Efficient document processing
- **Memory optimization**: Configurable chunk sizes
- **Parallel processing**: Multi-threaded operations where possible

## 🔍 Example Queries

The system excels at various types of questions:

- **Definition queries**: "What is machine learning?"
- **How-to queries**: "How do I create a function in Python?"
- **Comparison queries**: "What's the difference between lists and tuples?"
- **Conceptual queries**: "Explain object-oriented programming"
- **Relationship queries**: "How does Python relate to data science?"

## 🛠️ Extending the System

### Adding New Data Sources
```python
class CustomLoader:
    def load_documents(self) -> List[Document]:
        # Your custom loading logic
        pass
```

### Adding New LLM Providers
```python
class CustomLLM(BaseLLM):
    def generate(self, prompt: str, **kwargs) -> str:
        # Your custom generation logic
        pass
```

### Adding New Vector Stores
```python
class CustomVectorStore:
    def similarity_search(self, query: str, k: int) -> List[Document]:
        # Your custom search logic
        pass
```

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure environment variables are set correctly
   - Check API key validity and permissions

2. **Memory Issues**
   - Reduce chunk sizes in data source configuration
   - Use smaller embedding models

3. **Slow Performance**
   - Enable caching
   - Use smaller context windows
   - Consider using FAISS instead of Chroma for large datasets

4. **Knowledge Graph Issues**
   - Ensure LLM has sufficient context for entity extraction
   - Check document quality and structure

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📚 References

- [LangChain Documentation](https://python.langchain.com/)
- [Chroma Vector Store](https://www.trychroma.com/)
- [NetworkX Graph Library](https://networkx.org/)
- [Streamlit Web Framework](https://streamlit.io/)

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review example usage files

---

**Built with ❤️ for intelligent document querying and knowledge discovery** 