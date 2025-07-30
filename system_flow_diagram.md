# RAG Knowledge Graph System - Flow Diagram

## System Architecture Overview

```mermaid
graph TB
    subgraph "User Interface Layer"
        UI[Streamlit Web App]
        CLI[Command Line Interface]
        API[REST API]
    end
    
    subgraph "Core System Layer"
        RAG[RAG System Orchestrator]
        CONFIG[Configuration Manager]
    end
    
    subgraph "Data Processing Layer"
        DL[Data Loader]
        DS[Data Source Manager]
        VS[Vector Store Manager]
        KG[Knowledge Graph Manager]
    end
    
    subgraph "AI/ML Layer"
        LLM[LLM Manager]
        RAG_CHAIN[RAG Chain]
        EMB[Embedding Models]
    end
    
    subgraph "Storage Layer"
        VDB[(Vector Database<br/>Chroma/FAISS)]
        KGD[(Knowledge Graph<br/>JSON/NetworkX)]
        CACHE[(Cache Directory)]
        OUTPUT[(Output Directory)]
    end
    
    subgraph "External Services"
        OPENAI[OpenAI API]
        ANTHROPIC[Anthropic API]
        OLLAMA[Ollama Local]
    end
    
    UI --> RAG
    CLI --> RAG
    API --> RAG
    
    RAG --> CONFIG
    RAG --> DL
    RAG --> VS
    RAG --> KG
    RAG --> LLM
    
    DL --> DS
    VS --> VDB
    KG --> KGD
    LLM --> RAG_CHAIN
    
    RAG_CHAIN --> EMB
    EMB --> VDB
    
    LLM --> OPENAI
    LLM --> ANTHROPIC
    LLM --> OLLAMA
    
    VS --> CACHE
    KG --> CACHE
    RAG --> OUTPUT
```

## Detailed System Flow

```mermaid
flowchart TD
    Start([User Starts System]) --> Init[Initialize RAG System]
    
    Init --> LoadConfig[Load Configuration]
    LoadConfig --> CreateComponents[Create System Components]
    
    CreateComponents --> DataLoader[Initialize Data Loader]
    CreateComponents --> VectorStore[Initialize Vector Store]
    CreateComponents --> LLMManager[Initialize LLM Manager]
    CreateComponents --> KnowledgeGraph[Initialize Knowledge Graph]
    
    DataLoader --> LoadDocs[Load Documents from Sources]
    LoadDocs --> ChunkDocs[Chunk Documents]
    ChunkDocs --> StoreDocs[Store in Vector Database]
    
    VectorStore --> CreateEmbeddings[Create Document Embeddings]
    CreateEmbeddings --> StoreVectors[Store Vectors in Chroma/FAISS]
    
    KnowledgeGraph --> ExtractEntities[Extract Entities & Relationships]
    ExtractEntities --> BuildGraph[Build Knowledge Graph]
    BuildGraph --> SaveGraph[Save Graph to JSON]
    
    LLMManager --> InitLLM[Initialize OpenAI/Anthropic/Ollama]
    
    StoreDocs --> Ready[System Ready]
    StoreVectors --> Ready
    SaveGraph --> Ready
    InitLLM --> Ready
    
    Ready --> UserQuestion{User Asks Question}
    
    UserQuestion --> RetrieveDocs[Retrieve Relevant Documents]
    RetrieveDocs --> VectorSearch[Vector Similarity Search]
    VectorSearch --> HasResults{Has Results?}
    
    HasResults -->|No| KeywordSearch[Keyword Search Fallback]
    HasResults -->|Yes| ProcessResults[Process Results]
    KeywordSearch --> ProcessResults
    
    ProcessResults --> QueryKG[Query Knowledge Graph]
    QueryKG --> CombineContext[Combine Document Context + KG Insights]
    
    CombineContext --> GeneratePrompt[Generate RAG Prompt]
    GeneratePrompt --> CallLLM[Call LLM API]
    CallLLM --> GetResponse[Get LLM Response]
    
    GetResponse --> FormatAnswer[Format Answer with Sources]
    FormatAnswer --> ReturnResult[Return Result to User]
    
    ReturnResult --> LogSession[Log Session History]
    LogSession --> End([End])
    
    style Start fill:#e1f5fe
    style Ready fill:#c8e6c9
    style End fill:#ffcdd2
    style UserQuestion fill:#fff3e0
    style VectorSearch fill:#f3e5f5
    style KeywordSearch fill:#f3e5f5
    style CallLLM fill:#e8f5e8
```

## Data Flow Diagram

```mermaid
graph LR
    subgraph "Input Sources"
        TXT[Text Files]
        PDF[PDF Files]
        MD[Markdown Files]
        HTML[Web Pages]
        JSON[JSON Files]
        CSV[CSV Files]
    end
    
    subgraph "Processing Pipeline"
        LOAD[Document Loader]
        CHUNK[Text Chunker]
        EMBED[Embedding Generator]
        EXTRACT[Entity Extractor]
        RELATE[Relationship Extractor]
    end
    
    subgraph "Storage"
        VECTORS[Vector Store]
        GRAPH[Knowledge Graph]
        METADATA[Document Metadata]
    end
    
    subgraph "Query Processing"
        QUERY[User Query]
        SEARCH[Similarity Search]
        CONTEXT[Context Assembly]
        LLM[LLM Generation]
        ANSWER[Final Answer]
    end
    
    TXT --> LOAD
    PDF --> LOAD
    MD --> LOAD
    HTML --> LOAD
    JSON --> LOAD
    CSV --> LOAD
    
    LOAD --> CHUNK
    CHUNK --> EMBED
    CHUNK --> EXTRACT
    
    EMBED --> VECTORS
    EXTRACT --> RELATE
    RELATE --> GRAPH
    CHUNK --> METADATA
    
    QUERY --> SEARCH
    SEARCH --> VECTORS
    SEARCH --> GRAPH
    VECTORS --> CONTEXT
    GRAPH --> CONTEXT
    CONTEXT --> LLM
    LLM --> ANSWER
```

## Component Interaction Flow

```mermaid
sequenceDiagram
    participant U as User
    participant R as RAG System
    participant D as Data Loader
    participant V as Vector Store
    participant K as Knowledge Graph
    participant L as LLM Manager
    participant O as OpenAI API
    
    U->>R: Initialize System
    R->>D: Load Documents
    D->>V: Create Embeddings
    V->>V: Store Vectors
    R->>K: Extract Entities
    K->>K: Build Graph
    R->>L: Initialize LLM
    L->>O: Test Connection
    R->>U: System Ready
    
    U->>R: Ask Question
    R->>V: Search Documents
    V->>V: Vector Search
    alt No Results
        V->>V: Keyword Search
    end
    V->>R: Return Documents
    R->>K: Query Graph
    K->>R: Return Insights
    R->>L: Generate Answer
    L->>O: API Call
    O->>L: Response
    L->>R: Formatted Answer
    R->>U: Final Answer
```

## Error Handling and Fallbacks

```mermaid
graph TD
    Start([System Start]) --> TryEmbed[Try Sentence Transformers]
    
    TryEmbed --> EmbedSuccess{Success?}
    EmbedSuccess -->|Yes| UseEmbed[Use HuggingFace Embeddings]
    EmbedSuccess -->|No| FallbackEmbed[Use Fallback Embeddings]
    
    UseEmbed --> TryVector[Try Vector Search]
    FallbackEmbed --> TryVector
    
    TryVector --> VectorSuccess{Success?}
    VectorSuccess -->|Yes| UseVector[Use Vector Results]
    VectorSuccess -->|No| KeywordFallback[Use Keyword Search]
    
    UseVector --> TryLLM[Try LLM API]
    KeywordFallback --> TryLLM
    
    TryLLM --> LLMSuccess{Success?}
    LLMSuccess -->|Yes| UseLLM[Use Real LLM]
    LLMSuccess -->|No| DummyLLM[Use Dummy LLM]
    
    UseLLM --> Success[System Operational]
    DummyLLM --> Success
    
    style Start fill:#e1f5fe
    style Success fill:#c8e6c9
    style FallbackEmbed fill:#fff3e0
    style KeywordFallback fill:#fff3e0
    style DummyLLM fill:#ffcdd2
```

## File Structure and Dependencies

```mermaid
graph TD
    subgraph "Core Modules"
        CONFIG[config.py]
        RAG[rag_system.py]
        DATA[data_loader.py]
        VECTOR[vector_store.py]
        KG[knowledge_graph.py]
        LLM[llm_manager.py]
    end
    
    subgraph "User Interface"
        APP[app.py]
        CLI[example_usage.py]
        TEST[test_system.py]
    end
    
    subgraph "Configuration"
        ENV[.env]
        REQ[requirements.txt]
        ENV_EX[env_example.txt]
    end
    
    subgraph "Data"
        TEST_TXT[test.txt]
        CACHE[cache/]
        OUTPUT[output/]
    end
    
    APP --> RAG
    CLI --> RAG
    TEST --> RAG
    
    RAG --> CONFIG
    RAG --> DATA
    RAG --> VECTOR
    RAG --> KG
    RAG --> LLM
    
    DATA --> TEST_TXT
    VECTOR --> CACHE
    KG --> CACHE
    RAG --> OUTPUT
    
    CONFIG --> ENV
    APP --> ENV
```

## Key Features and Capabilities

### ✅ **Implemented Features:**
- **Multi-format Document Loading**: TXT, PDF, MD, HTML, JSON, CSV
- **Intelligent Text Chunking**: Configurable chunk sizes and overlap
- **Vector Similarity Search**: ChromaDB and FAISS support
- **Keyword Search Fallback**: When vector search fails
- **Knowledge Graph Extraction**: Entity and relationship extraction
- **Multiple LLM Support**: OpenAI, Anthropic, Ollama
- **Robust Error Handling**: Multiple fallback mechanisms
- **Web Interface**: Streamlit-based UI
- **Session Management**: Query history and persistence
- **Extensible Architecture**: Easy to add new data sources and models

### 🔄 **System Workflow:**
1. **Initialization**: Load config, initialize components, load documents
2. **Document Processing**: Chunk, embed, and store documents
3. **Knowledge Extraction**: Build entity-relationship graph
4. **Query Processing**: Retrieve relevant context, generate answers
5. **Response Generation**: Combine context with LLM for final answer

### 🎯 **Use Cases:**
- **Document Q&A**: Ask questions about loaded documentation
- **Knowledge Discovery**: Explore entities and relationships
- **Research Assistant**: Multi-source information retrieval
- **Content Analysis**: Extract insights from large document collections 