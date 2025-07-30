# RAG Knowledge Graph System - Visual Flow

## High-Level System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RAG KNOWLEDGE GRAPH SYSTEM                           │
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                    │
│  │   CONFIG    │    │   DATA      │    │   LLM       │                    │
│  │             │    │  SOURCES    │    │  MODELS     │                    │
│  └─────────────┘    └─────────────┘    └─────────────┘                    │
│         │                   │                   │                          │
│         └───────────────────┼───────────────────┘                          │
│                             │                                              │
│  ┌───────────────────────────┼────────────────────────────────────────────┐ │
│  │                    RAG SYSTEM (Main Controller)                       │ │
│  │                                                                        │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │ │
│  │  │   DATA      │  │   VECTOR    │  │    LLM      │                   │ │
│  │  │ MANAGER     │  │  MANAGER    │  │  MANAGER    │                   │ │
│  │  │             │  │             │  │             │                   │ │
│  │  │ • Load Docs │  │ • Embeddings│  │ • Model Mgmt│                   │ │
│  │  │ • Split Text│  │ • Store     │  │ • API Calls │                   │ │
│  │  │ • Process   │  │ • Search    │  │ • RAG Chain │                   │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                   │ │
│  │         │                   │                   │                     │ │
│  │         └───────────────────┼───────────────────┘                     │ │
│  │                             │                                         │ │
│  │  ┌───────────────────────────┼──────────────────────────────────────┐ │ │
│  │  │              KNOWLEDGE GRAPH MANAGER                             │ │ │
│  │  │                                                                    │ │ │
│  │  │  ┌─────────────┐                    ┌─────────────┐               │ │ │
│  │  │  │   JSON      │                    │   NEO4J     │               │ │ │
│  │  │  │  MANAGER    │                    │  MANAGER    │               │ │ │
│  │  │  │             │                    │             │               │ │ │
│  │  │  │ • File Store│                    │ • Graph DB  │               │ │ │
│  │  │  │ • NetworkX  │                    │ • Cypher    │               │ │ │
│  │  │  │ • JSON      │                    │ • In-Memory │               │ │ │
│  │  │  │ • Visualize │                    │   Fallback  │               │ │ │
│  │  │  └─────────────┘                    └─────────────┘               │ │ │
│  │  │                                                                    │ │ │
│  │  │  ┌──────────────────────────────────────────────────────────────┐ │ │ │
│  │  │  │              ENTITY & RELATIONSHIP EXTRACTION                │ │ │ │
│  │  │  │                                                              │ │ │ │
│  │  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │ │ │ │
│  │  │  │  │     LLM     │  │  SEMANTIC   │  │   RULE      │          │ │ │ │
│  │  │  │  │ EXTRACTION  │  │  SEARCH     │  │  BASED      │          │ │ │ │
│  │  │  │  │             │  │ EXTRACTION  │  │ EXTRACTION  │          │ │ │ │
│  │  │  │  │ • GPT       │  │ • spaCy NER │  │ • Patterns  │          │ │ │ │
│  │  │  │  │ • JSON      │  │ • Dependency│  │ • Entities  │          │ │ │ │
│  │  │  │  │ • Context   │  │ • Semantic  │  │ • Relations │          │ │ │ │
│  │  │  │  │ • Quality   │  │ • Patterns  │  │ • Fallback  │          │ │ │ │
│  │  │  │  └─────────────┘  └─────────────┘  └─────────────┘          │ │ │ │
│  │  │  │                                                              │ │ │ │
│  │  │  │  ┌──────────────────────────────────────────────────────────┐ │ │ │ │
│  │  │  │  │              SINGLE-METHOD PRIORITY                      │ │ │ │ │
│  │  │  │  │                                                          │ │ │ │ │
│  │  │  │  │  LLM → Semantic Search → Rule-based                      │ │ │ │ │
│  │  │  │  │  (Only one method used per extraction)                   │ │ │ │ │
│  │  │  │  └──────────────────────────────────────────────────────────┘ │ │ │ │
│  │  │  └──────────────────────────────────────────────────────────────┘ │ │ │
│  │  └──────────────────────────────────────────────────────────────────┘ │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Main Process Flow

### 1. System Startup
```
START
  ↓
Load Configuration
  ↓
Initialize Components
  ↓
Load Data Sources
  ↓
Create Vector Store
  ↓
Extract Knowledge Graph
  ↓
SYSTEM READY
```

### 2. Question Processing
```
USER QUESTION
  ↓
Vector Search (Find relevant documents)
  ↓
Knowledge Graph Query (Find entities & relationships)
  ↓
Combine Context (Documents + KG insights)
  ↓
LLM Generation (Generate answer with sources)
  ↓
FINAL ANSWER
```

### 3. Entity Extraction (Single-Method)
```
DOCUMENT TEXT
  ↓
Try LLM Extraction
  ↓
[SUCCESS?] → YES → Use LLM Results
  ↓ NO
Try Semantic Search
  ↓
[SUCCESS?] → YES → Use Semantic Results
  ↓ NO
Use Rule-Based (Always works)
  ↓
DEDUPLICATE & STORE
```

### 4. Knowledge Graph Storage
```
EXTRACTION RESULTS
  ↓
[Neo4j Available?]
  ↓ YES                    ↓ NO
Store in Neo4j            Store in Memory
  ↓                        ↓
Graph Database            In-Memory Lists
  ↓                        ↓
Cypher Queries            Fast Search
  ↓                        ↓
COMPLETE                  COMPLETE
```

## Key Features Summary

### ✅ **Core Components**
- **RAG System**: Main orchestrator
- **Data Manager**: Handles file/URL/PDF loading
- **Vector Manager**: Embeddings and similarity search
- **LLM Manager**: Multiple model support (OpenAI, Anthropic, Ollama)
- **Knowledge Graph**: Dual storage (Neo4j + JSON)

### ✅ **Extraction Methods**
- **LLM**: Highest quality, context-aware
- **Semantic Search**: spaCy NER + dependency parsing
- **Rule-based**: Reliable fallback with patterns

### ✅ **Storage Options**
- **Neo4j**: Primary graph database
- **In-Memory**: Fallback when Neo4j unavailable
- **JSON**: File-based storage for JSON manager

### ✅ **Error Handling**
- **Graceful Fallbacks**: Each component has fallback options
- **Single-Method Priority**: LLM → Semantic → Rule-based
- **In-Memory Fallback**: Works without Neo4j
- **Multiple LLM Support**: Switch between models

### ✅ **Data Sources**
- **Files**: TXT, PDF, MD, JSON, CSV
- **URLs**: Web scraping
- **APIs**: REST endpoints
- **Databases**: Future implementation

## System Benefits

1. **Robust**: Multiple fallback mechanisms
2. **Flexible**: Configurable components
3. **Scalable**: Modular architecture
4. **Efficient**: Single-method extraction
5. **Reliable**: In-memory fallbacks
6. **Extensible**: Easy to add new features

This visual representation shows the current state of the project with all enhancements implemented, including the single-method extraction, in-memory fallbacks, and dual knowledge graph storage options. 