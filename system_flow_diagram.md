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
    
    subgraph "Knowledge Graph Layer"
        JSON_KG[JSON Knowledge Graph<br/>File-based + NetworkX]
        NEO4J_KG[Neo4j Knowledge Graph<br/>Graph Database + In-Memory Fallback]
        EXTRACTION[Entity & Relationship Extraction<br/>Single-Method Priority: LLM → Semantic → Rule-based]
    end
    
    subgraph "AI/ML Layer"
        LLM[LLM Manager]
        RAG_CHAIN[RAG Chain]
        EMB[Embedding Models]
    end
    
    subgraph "Storage Layer"
        VDB[(Vector Database<br/>Chroma/FAISS)]
        JSON_STORE[(JSON Storage<br/>knowledge_graph.json)]
        NEO4J_DB[(Neo4j Database<br/>Graph Storage)]
        MEMORY[(In-Memory Storage<br/>Python Lists)]
        CACHE[(Cache Directory)]
        OUTPUT[(Output Directory)]
    end
    
    subgraph "External Services"
        OPENAI[OpenAI API]
        ANTHROPIC[Anthropic API]
        OLLAMA[Ollama Local]
        NEO4J_SERVICE[Neo4j Service]
    end
    
    UI --> RAG
    CLI --> RAG
    API --> RAG
    
    RAG --> CONFIG
    RAG --> DL
    RAG --> VS
    RAG --> KG
    
    KG --> JSON_KG
    KG --> NEO4J_KG
    KG --> EXTRACTION
    
    JSON_KG --> JSON_STORE
    NEO4J_KG --> NEO4J_DB
    NEO4J_KG --> MEMORY
    
    EXTRACTION --> JSON_KG
    EXTRACTION --> NEO4J_KG
    
    DL --> DS
    VS --> VDB
    LLM --> RAG_CHAIN
    
    RAG_CHAIN --> EMB
    EMB --> VDB
    
    LLM --> OPENAI
    LLM --> ANTHROPIC
    LLM --> OLLAMA
    
    NEO4J_KG --> NEO4J_SERVICE
    
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
    
    KnowledgeGraph --> KGType{Knowledge Graph Type?}
    KGType -->|JSON| JSONKG[Initialize JSON Knowledge Graph]
    KGType -->|Neo4j| NEO4JKG[Initialize Neo4j Knowledge Graph]
    
    NEO4JKG --> Neo4jAvailable{Neo4j Available?}
    Neo4jAvailable -->|Yes| UseNeo4j[Use Neo4j Database]
    Neo4jAvailable -->|No| UseMemory[Use In-Memory Fallback]
    
    DataLoader --> LoadDocs[Load Documents from Sources]
    LoadDocs --> ChunkDocs[Chunk Documents]
    ChunkDocs --> StoreDocs[Store in Vector Database]
    
    VectorStore --> CreateEmbeddings[Create Document Embeddings]
    CreateEmbeddings --> StoreVectors[Store Vectors in Chroma/FAISS]
    
    JSONKG --> ExtractEntitiesJSON[Extract Entities & Relationships]
    UseNeo4j --> ExtractEntitiesNeo4j[Extract Entities & Relationships]
    UseMemory --> ExtractEntitiesMemory[Extract Entities & Relationships]
    
    ExtractEntitiesJSON --> SingleMethodJSON[Single-Method Extraction<br/>LLM → Semantic → Rule-based]
    ExtractEntitiesNeo4j --> SingleMethodNeo4j[Single-Method Extraction<br/>LLM → Semantic → Rule-based]
    ExtractEntitiesMemory --> SingleMethodMemory[Single-Method Extraction<br/>LLM → Semantic → Rule-based]
    
    SingleMethodJSON --> BuildGraphJSON[Build Knowledge Graph]
    SingleMethodNeo4j --> BuildGraphNeo4j[Build Knowledge Graph]
    SingleMethodMemory --> BuildGraphMemory[Build Knowledge Graph]
    
    BuildGraphJSON --> SaveGraphJSON[Save Graph to JSON]
    BuildGraphNeo4j --> SaveGraphNeo4j[Save Graph to Neo4j]
    BuildGraphMemory --> SaveGraphMemory[Save Graph to Memory]
    
    LLMManager --> InitLLM[Initialize OpenAI/Anthropic/Ollama]
    
    StoreDocs --> Ready[System Ready]
    StoreVectors --> Ready
    SaveGraphJSON --> Ready
    SaveGraphNeo4j --> Ready
    SaveGraphMemory --> Ready
    InitLLM --> Ready
    
    Ready --> UserQuestion{User Asks Question}
    
    UserQuestion --> RetrieveDocs[Retrieve Relevant Documents]
    RetrieveDocs --> VectorSearch[Vector Similarity Search]
    VectorSearch --> HasResults{Has Results?}
    
    HasResults -->|No| KeywordSearch[Keyword Search Fallback]
    HasResults -->|Yes| ProcessResults[Process Results]
    KeywordSearch --> ProcessResults
    
    ProcessResults --> QueryKG[Query Knowledge Graph]
    QueryKG --> KGQueryType{Query Type?}
    KGQueryType -->|JSON| QueryJSON[Query JSON Graph]
    KGQueryType -->|Neo4j| QueryNeo4j[Query Neo4j Graph]
    KGQueryType -->|Memory| QueryMemory[Query In-Memory Graph]
    
    QueryJSON --> CombineContext[Combine Document Context + KG Insights]
    QueryNeo4j --> CombineContext
    QueryMemory --> CombineContext
    
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
    style UseMemory fill:#fff3e0
    style SingleMethodJSON fill:#e3f2fd
    style SingleMethodNeo4j fill:#e3f2fd
    style SingleMethodMemory fill:#e3f2fd
```

## Entity & Relationship Extraction Flow

```mermaid
flowchart TD
    DocumentText[Document Text] --> TryLLM[Try LLM Extraction]
    
    TryLLM --> LLMAvailable{LLM Available?}
    LLMAvailable -->|No| TrySemantic[Try Semantic Search]
    LLMAvailable -->|Yes| LLMProcess[Process with LLM]
    
    LLMProcess --> LLMResults{LLM Returns Results?}
    LLMResults -->|Yes| UseLLM[Use LLM Results]
    LLMResults -->|No| TrySemantic
    
    TrySemantic --> SemanticProcess[Process with spaCy NER + Dependency Parsing]
    SemanticProcess --> SemanticResults{Semantic Returns Results?}
    SemanticResults -->|Yes| UseSemantic[Use Semantic Results]
    SemanticResults -->|No| UseRuleBased[Use Rule-Based Extraction]
    
    UseLLM --> Deduplicate[Remove Duplicates]
    UseSemantic --> Deduplicate
    UseRuleBased --> Deduplicate
    
    Deduplicate --> StoreResults[Store Results]
    
    StoreResults --> StorageType{Storage Type?}
    StorageType -->|JSON| StoreJSON[Store in JSON File]
    StorageType -->|Neo4j| StoreNeo4j[Store in Neo4j Database]
    StorageType -->|Memory| StoreMemory[Store in Memory Lists]
    
    style DocumentText fill:#e1f5fe
    style UseLLM fill:#c8e6c9
    style UseSemantic fill:#fff3e0
    style UseRuleBased fill:#ffcdd2
    style StoreResults fill:#e8f5e8
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
        EXTRACT[Entity Extractor<br/>Single-Method Priority]
    end
    
    subgraph "Storage Options"
        VECTORS[Vector Store<br/>Chroma/FAISS]
        JSON_GRAPH[JSON Knowledge Graph<br/>File-based]
        NEO4J_GRAPH[Neo4j Knowledge Graph<br/>Database]
        MEMORY_GRAPH[In-Memory Graph<br/>Python Lists]
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
    EXTRACT --> JSON_GRAPH
    EXTRACT --> NEO4J_GRAPH
    EXTRACT --> MEMORY_GRAPH
    
    QUERY --> SEARCH
    SEARCH --> VECTORS
    SEARCH --> JSON_GRAPH
    SEARCH --> NEO4J_GRAPH
    SEARCH --> MEMORY_GRAPH
    VECTORS --> CONTEXT
    JSON_GRAPH --> CONTEXT
    NEO4J_GRAPH --> CONTEXT
    MEMORY_GRAPH --> CONTEXT
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
    participant N as Neo4j Manager
    participant J as JSON Manager
    participant L as LLM Manager
    participant O as OpenAI API
    
    U->>R: Initialize System
    R->>D: Load Documents
    D->>V: Create Embeddings
    V->>V: Store Vectors
    
    alt Knowledge Graph Type
        R->>N: Initialize Neo4j Manager
        N->>N: Check Neo4j Connection
        alt Neo4j Available
            N->>N: Use Neo4j Database
        else Neo4j Unavailable
            N->>N: Use In-Memory Fallback
        end
    else JSON Type
        R->>J: Initialize JSON Manager
    end
    
    R->>K: Extract Entities & Relationships
    K->>K: Single-Method Extraction (LLM → Semantic → Rule-based)
    K->>K: Store Results
    
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
    alt Neo4j Available
        K->>N: Query Neo4j
        N->>R: Return Insights
    else In-Memory
        K->>N: Query In-Memory
        N->>R: Return Insights
    else JSON
        K->>J: Query JSON
        J->>R: Return Insights
    end
    
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
    
    UseLLM --> TryNeo4j[Try Neo4j Connection]
    DummyLLM --> TryNeo4j
    
    TryNeo4j --> Neo4jSuccess{Success?}
    Neo4jSuccess -->|Yes| UseNeo4j[Use Neo4j Database]
    Neo4jSuccess -->|No| UseMemory[Use In-Memory Storage]
    
    UseNeo4j --> TryExtraction[Try Entity Extraction]
    UseMemory --> TryExtraction
    
    TryExtraction --> ExtractionSuccess{Success?}
    ExtractionSuccess -->|Yes| UseExtraction[Use Extraction Results]
    ExtractionSuccess -->|No| UseRuleBased[Use Rule-Based Fallback]
    
    UseExtraction --> Success[System Operational]
    UseRuleBased --> Success
    
    style Start fill:#e1f5fe
    style Success fill:#c8e6c9
    style FallbackEmbed fill:#fff3e0
    style KeywordFallback fill:#fff3e0
    style DummyLLM fill:#ffcdd2
    style UseMemory fill:#fff3e0
    style UseRuleBased fill:#ffcdd2
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
        NEO4J_KG[neo4j_knowledge_graph.py]
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
        DOCKER[docker-compose.yml]
    end
    
    subgraph "Data"
        TEST_TXT[test.txt]
        CACHE[cache/]
        OUTPUT[output/]
    end
    
    subgraph "Documentation"
        README[README.md]
        NEO4J_SUMMARY[NEO4J_INTEGRATION_SUMMARY.md]
        EXTRACTION_SUMMARY[ENHANCED_EXTRACTION_SUMMARY.md]
        SINGLE_METHOD[SINGLE_METHOD_EXTRACTION_SUMMARY.md]
        IN_MEMORY[IN_MEMORY_FALLBACK_SUMMARY.md]
    end
    
    APP --> RAG
    CLI --> RAG
    TEST --> RAG
    
    RAG --> CONFIG
    RAG --> DATA
    RAG --> VECTOR
    RAG --> KG
    RAG --> NEO4J_KG
    RAG --> LLM
    
    DATA --> TEST_TXT
    VECTOR --> CACHE
    KG --> CACHE
    NEO4J_KG --> CACHE
    RAG --> OUTPUT
    
    CONFIG --> ENV
    APP --> ENV
    NEO4J_KG --> DOCKER
```

## Key Features and Capabilities

### ✅ **Implemented Features:**
- **Multi-format Document Loading**: TXT, PDF, MD, HTML, JSON, CSV
- **Intelligent Text Chunking**: Configurable chunk sizes and overlap
- **Vector Similarity Search**: ChromaDB and FAISS support with keyword fallback
- **Dual Knowledge Graph Storage**: Neo4j (primary) + JSON (fallback)
- **In-Memory Fallback**: Neo4j unavailable → in-memory storage
- **Single-Method Extraction**: LLM → Semantic Search → Rule-based (priority-based)
- **Multiple LLM Support**: OpenAI, Anthropic, Ollama with fallback LLM
- **Robust Error Handling**: Multiple fallback mechanisms at every level
- **Web Interface**: Streamlit-based UI
- **Session Management**: Query history and persistence
- **Extensible Architecture**: Easy to add new data sources and models

### 🔄 **System Workflow:**
1. **Initialization**: Load config, initialize components, load documents
2. **Document Processing**: Chunk, embed, and store documents
3. **Knowledge Extraction**: Single-method entity/relationship extraction
4. **Storage Selection**: Neo4j (if available) or in-memory fallback
5. **Query Processing**: Retrieve relevant context, generate answers
6. **Response Generation**: Combine context with LLM for final answer

### 🎯 **Use Cases:**
- **Document Q&A**: Ask questions about loaded documentation
- **Knowledge Discovery**: Explore entities and relationships
- **Research Assistant**: Multi-source information retrieval
- **Content Analysis**: Extract insights from large document collections
- **Graph Analytics**: Query and visualize knowledge graphs

### 🛡️ **Robustness Features:**
- **Graceful Degradation**: System works even when components fail
- **Multiple Fallbacks**: LLM → Semantic → Rule-based extraction
- **Storage Flexibility**: Neo4j → In-Memory → JSON storage
- **Error Recovery**: Automatic fallback to alternative methods
- **Offline Capability**: Works without external services (except LLM) 