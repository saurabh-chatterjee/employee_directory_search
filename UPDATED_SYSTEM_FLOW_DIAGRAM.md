# Updated RAG Knowledge Graph System Flow Diagram

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           RAG KNOWLEDGE GRAPH SYSTEM                                │
│                                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│  │   CONFIGURATION │    │   DATA SOURCES  │    │   LLM MODELS    │                │
│  │                 │    │                 │    │                 │                │
│  │ • SystemConfig  │    │ • File Loaders  │    │ • OpenAI        │                │
│  │ • ModelConfig   │    │ • URL Loaders   │    │ • Anthropic     │                │
│  │ • DataSource    │    │ • PDF Loaders   │    │ • Ollama        │                │
│  │ • Neo4jConfig   │    │ • JSON Loaders  │    │ • Local Models  │                │
│  │ • Vector Config │    │ • CSV Loaders   │    │                 │                │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘                │
│           │                       │                       │                        │
│           └───────────────────────┼───────────────────────┘                        │
│                                   │                                                │
│  ┌─────────────────────────────────┼─────────────────────────────────────────────┐ │
│  │                        RAG SYSTEM (Main Orchestrator)                        │ │
│  │                                                                               │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │ │
│  │  │ DATA MANAGER    │  │ VECTOR MANAGER  │  │  LLM MANAGER    │              │ │
│  │  │                 │  │                 │  │                 │              │ │
│  │  │ • DataLoader    │  │ • Chroma Store  │  │ • Model Switch  │              │ │
│  │  │ • Text Splitter │  │ • FAISS Store   │  │ • API Handling  │              │ │
│  │  │ • Source Mgmt   │  │ • Embeddings    │  │ • Fallback LLM  │              │ │
│  │  │ • Document Mgmt │  │ • Similarity    │  │ • RAG Chain     │              │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘              │ │
│  │           │                       │                       │                  │ │
│  │           └───────────────────────┼───────────────────────┘                  │ │
│  │                                   │                                          │ │
│  │  ┌─────────────────────────────────┼─────────────────────────────────────────┐ │ │
│  │  │                    KNOWLEDGE GRAPH MANAGER                               │ │ │
│  │  │                                                                           │ │ │
│  │  │  ┌─────────────────┐                    ┌─────────────────┐              │ │ │
│  │  │  │ JSON MANAGER    │                    │  NEO4J MANAGER  │              │ │ │
│  │  │  │                 │                    │                 │              │ │ │
│  │  │  │ • File Storage  │                    │ • Graph Database│              │ │ │
│  │  │  │ • NetworkX      │                    │ • Cypher Queries│              │ │ │
│  │  │  │ • JSON Export   │                    │ • In-Memory     │              │ │ │
│  │  │  │ • Visualization │                    │   Fallback      │              │ │ │
│  │  │  └─────────────────┘                    └─────────────────┘              │ │ │
│  │  │                                                                           │ │ │
│  │  │  ┌─────────────────────────────────────────────────────────────────────┐ │ │ │
│  │  │  │              ENTITY & RELATIONSHIP EXTRACTION                       │ │ │ │
│  │  │  │                                                                     │ │ │ │
│  │  │  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │ │ │ │
│  │  │  │  │   LLM EXTRACTION│  │ SEMANTIC SEARCH │  │  RULE-BASED     │    │ │ │ │
│  │  │  │  │                 │  │   EXTRACTION    │  │   EXTRACTION    │    │ │ │ │
│  │  │  │  │ • OpenAI GPT    │  │ • spaCy NER     │  │ • Regex Patterns│    │ │ │ │
│  │  │  │  │ • JSON Prompts  │  │ • Dependency    │  │ • Named Entities│    │ │ │ │
│  │  │  │  │ • Context Aware │  │   Parsing       │  │ • Relationships │    │ │ │ │
│  │  │  │  │ • High Quality  │  │ • Semantic      │  │ • Fallback      │    │ │ │ │
│  │  │  │  │                 │  │   Patterns      │  │ • Reliable      │    │ │ │ │
│  │  │  │  └─────────────────┘  └─────────────────┘  └─────────────────┘    │ │ │ │
│  │  │  │                                                                     │ │ │ │
│  │  │  │  ┌─────────────────────────────────────────────────────────────────┐ │ │ │ │
│  │  │  │  │              SINGLE-METHOD PRIORITY EXTRACTION                  │ │ │ │ │
│  │  │  │  │                                                                 │ │ │ │ │
│  │  │  │  │  Priority: LLM → Semantic Search → Rule-based                   │ │ │ │ │
│  │  │  │  │  Fallback: Only if current method fails or returns no results   │ │ │ │ │
│  │  │  │  │  Deduplication: Automatic removal of duplicate entities/rels    │ │ │ │ │
│  │  │  │  └─────────────────────────────────────────────────────────────────┘ │ │ │ │
│  │  │  └─────────────────────────────────────────────────────────────────────┘ │ │ │
│  │  └─────────────────────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Detailed Component Flow

### 1. System Initialization Flow

```
┌─────────────────┐
│   START SYSTEM  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  LOAD CONFIG    │
│ • SystemConfig  │
│ • ModelConfig   │
│ • DataSources   │
│ • Neo4jConfig   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ INITIALIZE      │
│ COMPONENTS      │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ DATA MANAGER    │    │ VECTOR MANAGER  │    │  LLM MANAGER    │
│ • Load Sources  │    │ • Init Embeddings│   │ • Init Models    │
│ • Split Docs    │    │ • Create Store  │    │ • Set Current    │
│ • Process Files │    │ • Load Existing │    │ • Test Connection│
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              KNOWLEDGE GRAPH INITIALIZATION                     │
│                                                                 │
│  ┌─────────────────┐                    ┌─────────────────┐    │
│  │ JSON MANAGER    │                    │  NEO4J MANAGER  │    │
│  │                 │                    │                 │    │
│  │ • Load Existing │                    │ • Connect DB    │    │
│  │ • Create New    │                    │ • Setup Schema  │    │
│  │ • Init NetworkX │                    │ • Create Indexes│    │
│  │                 │                    │ • In-Memory     │    │
│  │                 │                    │   Fallback      │    │
│  └─────────────────┘                    └─────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────┐
│  EXTRACT KG     │
│ • Process Docs  │
│ • Extract Entities│
│ • Extract Rels  │
│ • Store Results │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ SYSTEM READY    │
└─────────────────┘
```

### 2. Question Processing Flow

```
┌─────────────────┐
│  USER QUESTION  │
└─────────┬───────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    QUESTION PROCESSING                          │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  │ VECTOR SEARCH   │    │  KG QUERY       │    │  LLM GENERATION │
│  │                 │    │                 │    │                 │
│  │ • Embed Query   │    │ • Query Entities│    │ • Combine Context│
│  │ • Similarity    │    │ • Query Rels    │    │ • Generate Answer│
│  │ • Top-K Docs    │    │ • Get Insights  │    │ • Format Response│
│  │ • Score Results │    │ • Filter Results│    │ • Add Sources    │
│  └─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
│            │                      │                      │
│            └──────────────────────┼──────────────────────┘
│                                   │
│  ┌─────────────────────────────────┼─────────────────────────────────────────┐
│  │                    CONTEXT COMBINATION                                    │
│  │                                                                           │
│  │ • Vector Store Results (Documents)                                       │
│  │ • Knowledge Graph Insights (Entities & Relationships)                    │
│  │ • Question Context                                                       │
│  │ • Model Configuration                                                    │
│  └─────────────────────────────────────────────────────────────────────────┘
│                                   │
│                                   ▼
│  ┌─────────────────────────────────────────────────────────────────────────┐
│  │                        RAG CHAIN EXECUTION                              │
│  │                                                                         │
│  │ • Format Context for LLM                                                │
│  │ • Generate Answer with Sources                                          │
│  │ • Include KG Insights                                                   │
│  │ • Format Final Response                                                 │
│  └─────────────────────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────┐
│  FINAL ANSWER   │
│ • Answer Text   │
│ • Source Docs   │
│ • KG Insights   │
│ • Model Used    │
│ • Timestamp     │
└─────────────────┘
```

### 3. Entity & Relationship Extraction Flow

```
┌─────────────────┐
│  DOCUMENT TEXT  │
└─────────┬───────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│              SINGLE-METHOD EXTRACTION PIPELINE                  │
│                                                                 │
│  ┌─────────────────┐                                            │
│  │  METHOD 1: LLM  │                                            │
│  │                 │                                            │
│  │ • Check LLM     │                                            │
│  │   Availability  │                                            │
│  │ • Send JSON     │                                            │
│  │   Prompt        │                                            │
│  │ • Parse Results │                                            │
│  │ • Validate JSON │                                            │
│  └─────────┬───────┘                                            │
│            │                                                    │
│            ▼                                                    │
│  ┌─────────────────┐                                            │
│  │  LLM SUCCESS?   │                                            │
│  │  Has Results?   │                                            │
│  └─────────┬───────┘                                            │
│            │                                                    │
│            ▼                                                    │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  │  YES - USE LLM  │    │  NO - TRY       │    │  FAILED - TRY   │
│  │                 │    │  SEMANTIC       │    │  SEMANTIC       │
│  │ • Return LLM    │    │  SEARCH         │    │  SEARCH         │
│  │   Results       │    │                 │    │                 │
│  │ • Deduplicate   │    │ • spaCy NER     │    │ • spaCy NER     │
│  │ • Format Output │    │ • Dependency    │    │ • Dependency    │
│  │                 │    │   Parsing       │    │   Parsing       │
│  │                 │    │ • Semantic      │    │ • Semantic      │
│  │                 │    │   Patterns      │    │   Patterns      │
│  └─────────────────┘    └─────────┬───────┘    └─────────┬───────┘
│                                 │                      │
│                                 ▼                      │
│  ┌─────────────────┐                                    │
│  │  SEMANTIC       │                                    │
│  │  SUCCESS?       │                                    │
│  │  Has Results?   │                                    │
│  └─────────┬───────┘                                    │
│            │                                            │
│            ▼                                            │
│  ┌─────────────────┐    ┌─────────────────┐            │
│  │  YES - USE      │    │  NO - USE       │            │
│  │  SEMANTIC       │    │  RULE-BASED     │            │
│  │                 │    │                 │            │
│  │ • Return        │    │ • Regex Patterns│            │
│  │   Semantic      │    │ • Named Entities│            │
│  │   Results       │    │ • Relationships │            │
│  │ • Deduplicate   │    │ • Fallback      │            │
│  │ • Format Output │    │ • Always Works  │            │
│  └─────────────────┘    └─────────────────┘            │
│                                                         │
│  ┌─────────────────────────────────────────────────────┐│
│  │              FINAL OUTPUT                           ││
│  │                                                     ││
│  │ • Entities: [{name, type, description}]             ││
│  │ • Relationships: [{source, target, type, desc}]     ││
│  │ • Method Used: LLM/Semantic/Rule-based              ││
│  │ • Deduplicated Results                              ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### 4. Knowledge Graph Storage Flow

```
┌─────────────────┐
│  EXTRACTION     │
│  RESULTS        │
└─────────┬───────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STORAGE SELECTION                            │
│                                                                 │
│  ┌─────────────────┐                    ┌─────────────────┐    │
│  │  NEO4J AVAILABLE│                    │  NEO4J UNAVAILABLE│   │
│  │                 │                    │                 │    │
│  │ • Connect DB    │                    │ • In-Memory      │    │
│  │ • Store Entities│                    │   Storage       │    │
│  │ • Store Rels    │                    │ • List Storage   │    │
│  │ • Create Indexes│                    │ • No Persistence │    │
│  │ • Cypher Queries│                    │ • Fast Access    │    │
│  └─────────┬───────┘                    └─────────┬───────┘    │
│            │                                      │
│            ▼                                      ▼
│  ┌─────────────────┐                    ┌─────────────────┐
│  │  NEO4J STORAGE  │                    │  IN-MEMORY      │
│  │                 │                    │  STORAGE        │
│  │ • MERGE Entity  │                    │ • Append Entity │
│  │ • MERGE Rel     │                    │ • Append Rel    │
│  │ • Add Metadata  │                    │ • Check Dups    │
│  │ • Update Index  │                    │ • Fast Query    │
│  └─────────────────┘                    └─────────────────┘
└─────────────────────────────────────────────────────────────────┘
          │                                      │
          └──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────┐
│  STORAGE        │
│  COMPLETE       │
└─────────────────┘
```

### 5. Query Processing Flow

```
┌─────────────────┐
│  KG QUERY       │
└─────────┬───────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    QUERY ROUTING                                │
│                                                                 │
│  ┌─────────────────┐                    ┌─────────────────┐    │
│  │  NEO4J STORAGE  │                    │  IN-MEMORY      │
│  │                 │                    │  STORAGE        │
│  │ • Cypher Query  │                    │ • List Search    │
│  │ • Graph Traversal│                   │ • String Match  │
│  │ • Complex Rel   │                    │ • Simple Filter │
│  │ • Performance   │                    │ • Fast Results  │
│  └─────────┬───────┘                    └─────────┬───────┘    │
│            │                                      │
│            ▼                                      ▼
│  ┌─────────────────┐                    ┌─────────────────┐
│  │  NEO4J QUERY    │                    │  MEMORY QUERY   │
│  │                 │                    │                 │
│  │ • MATCH Entity  │                    │ • Filter Entities│
│  │ • MATCH Rel     │                    │ • Filter Rels   │
│  │ • WHERE Clause  │                    │ • String Search │
│  │ • RETURN Results│                    │ • Return List   │
│  └─────────────────┘                    └─────────────────┘
└─────────────────────────────────────────────────────────────────┘
          │                                      │
          └──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────┐
│  QUERY RESULTS  │
│ • Entities      │
│ • Relationships │
│ • Metadata      │
└─────────────────┘
```

## Component Interactions

### Data Flow Summary

1. **Configuration Loading**: System loads configuration from `config.py`
2. **Data Loading**: `DataSourceManager` loads documents from various sources
3. **Vector Processing**: `VectorStoreManager` creates embeddings and stores them
4. **Knowledge Graph**: `KnowledgeGraphManager` extracts entities and relationships
5. **Question Processing**: System combines vector search and KG insights
6. **Answer Generation**: `RAGChain` generates final answer with sources

### Key Features

- **Single-Method Extraction**: Uses LLM OR Semantic Search OR Rule-based (priority-based)
- **In-Memory Fallback**: Neo4j unavailable → in-memory storage
- **Multiple LLM Support**: OpenAI, Anthropic, Ollama, Local models
- **Flexible Data Sources**: Files, URLs, PDFs, JSON, CSV
- **Vector Store Options**: Chroma, FAISS with fallback
- **Dual KG Storage**: Neo4j (primary) and JSON (fallback)

### Error Handling

- **LLM Failures**: Fallback to semantic search or rule-based
- **Neo4j Unavailable**: Automatic in-memory fallback
- **Vector Store Issues**: Fallback to keyword search
- **Data Source Errors**: Graceful degradation with available sources

This updated diagram accurately reflects the current state of the project with all the enhancements and improvements that have been implemented. 