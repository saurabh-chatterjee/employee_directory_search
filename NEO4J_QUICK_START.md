# Neo4j Knowledge Graph Quick Start Guide

## 🚀 Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Neo4j (Choose One Option)

**Option A: Docker Compose (Recommended)**
```bash
docker-compose up -d
```

**Option B: Docker Direct**
```bash
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
```

**Option C: Neo4j Desktop**
- Download from https://neo4j.com/download/
- Create a new database
- Set password to "password"

### 3. Verify Neo4j is Running
- Open http://localhost:7474 in your browser
- Login with: `neo4j` / `password`
- You should see the Neo4j browser interface

## 📝 Basic Usage

### 1. Configure Your System
```python
from config import SystemConfig, Neo4jConfig
from rag_system import RAGSystem

# Create configuration with Neo4j
config = SystemConfig(
    knowledge_graph_type="neo4j",  # Use Neo4j instead of JSON
    neo4j_config=Neo4jConfig(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )
)

# Initialize RAG system
rag_system = RAGSystem(config)
rag_system.initialize()
```

### 2. Add Your Data
```python
from config import DataSourceConfig

# Add a data source
data_source = DataSourceConfig(
    name="My Documents",
    type="file",
    path="./my_documents.txt",
    format="txt",
    enabled=True
)

rag_system.add_data_source(data_source)
rag_system.initialize(force_reload=True)
```

### 3. Query Your Knowledge Graph
```python
# Natural language queries
results = rag_system.knowledge_graph.query_graph("software engineer")

# Get entity relationships
relationships = rag_system.get_entity_relationships("Sarah", max_depth=2)

# Get graph statistics
stats = rag_system.knowledge_graph.get_graph_stats()
```

### 4. Ask Questions
```python
# RAG with knowledge graph enhancement
answer = rag_system.ask_question(
    "Who works with Sarah?", 
    use_knowledge_graph=True
)
print(answer['answer'])
```

## 🔍 Advanced Queries

### 1. Direct Neo4j Queries
```python
# Access the Neo4j session directly
with rag_system.knowledge_graph.driver.session() as session:
    # Find all employees
    result = session.run("MATCH (e:Entity {type: 'person'}) RETURN e.name")
    for record in result:
        print(record["e.name"])
    
    # Find relationships
    result = session.run("""
        MATCH (source:Entity)-[r:RELATES_TO]->(target:Entity)
        WHERE r.type = 'works_at'
        RETURN source.name, target.name
    """)
    for record in result:
        print(f"{record['source.name']} works at {record['target.name']}")
```

### 2. Complex Graph Traversals
```python
# Find all colleagues within 2 hops
result = session.run("""
    MATCH path = (start:Entity {name: 'Sarah'})-[:RELATES_TO*1..2]-(colleague:Entity)
    WHERE colleague.type = 'person' AND colleague.name <> 'Sarah'
    RETURN DISTINCT colleague.name, length(path) as distance
    ORDER BY distance
""")
```

## 🛠️ Troubleshooting

### Common Issues

**1. Connection Refused**
```
Error: Couldn't connect to localhost:7687
```
**Solution**: Make sure Neo4j is running and accessible

**2. Authentication Failed**
```
Error: Authentication failed
```
**Solution**: Check username/password (default: neo4j/password)

**3. Database Not Found**
```
Error: Database does not exist
```
**Solution**: Create a database in Neo4j browser or use default "neo4j"

### Fallback Mode
If Neo4j is not available, the system automatically falls back to JSON storage:
```python
# Check if Neo4j is available
if rag_system.knowledge_graph.driver:
    print("Neo4j is connected")
else:
    print("Using JSON fallback mode")
```

## 📊 Monitoring

### 1. Graph Statistics
```python
stats = rag_system.knowledge_graph.get_graph_stats()
print(f"Nodes: {stats['nodes']}")
print(f"Edges: {stats['edges']}")
print(f"Entity Types: {stats['entity_types']}")
```

### 2. System Health
```python
system_stats = rag_system.get_system_stats()
print(f"Knowledge Graph: {system_stats['knowledge_graph']}")
```

## 🔄 Migration from JSON

### 1. Export Current Data
```python
# If using JSON storage
rag_system.knowledge_graph.save_graph("backup.json")
```

### 2. Switch to Neo4j
```python
# Update configuration
config.knowledge_graph_type = "neo4j"
rag_system = RAGSystem(config)
```

### 3. Import Data
```python
# Load data into Neo4j
rag_system.knowledge_graph.load_graph("backup.json")
```

## 🎯 Best Practices

### 1. Data Modeling
- Use consistent entity types (person, organization, location)
- Define clear relationship types (works_at, reports_to, located_in)
- Add descriptive properties to entities and relationships

### 2. Performance
- Use indexes for frequently queried properties
- Limit query depth for large graphs
- Use parameterized queries for better performance

### 3. Maintenance
- Regular backups using `save_graph()`
- Monitor graph statistics
- Clean up unused entities and relationships

## 📚 Examples

### Employee Directory
```python
# Sample employee data
documents = [
    "Sarah is a software engineer at TechCorp",
    "Michael reports to Sarah at TechCorp",
    "TechCorp is located in San Francisco"
]

# Extract and query
rag_system.knowledge_graph.extract_graph_from_documents(documents)
results = rag_system.knowledge_graph.query_graph("TechCorp employees")
```

### Research Paper Analysis
```python
# Extract authors, institutions, and citations
documents = [
    "Dr. Smith from MIT published a paper on AI",
    "The paper was cited by Dr. Johnson from Stanford",
    "MIT and Stanford collaborated on the research"
]

# Find research collaborations
relationships = rag_system.get_entity_relationships("MIT", max_depth=2)
```

## 🆘 Getting Help

### 1. Check Logs
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. Test Connection
```python
# Test Neo4j connection
from neo4j import GraphDatabase
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
with driver.session() as session:
    result = session.run("RETURN 1")
    print("Connection successful!")
```

### 3. Run Tests
```bash
python test_neo4j_knowledge_graph.py
python demo_neo4j_integration.py
```

## 📖 Next Steps

1. **Explore Neo4j Browser**: http://localhost:7474
2. **Learn Cypher**: Neo4j's query language
3. **Advanced Queries**: Graph algorithms and pattern matching
4. **Integration**: Connect with other data sources
5. **Visualization**: Create interactive graph visualizations 