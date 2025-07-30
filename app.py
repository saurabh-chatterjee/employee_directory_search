"""
Streamlit web application for the RAG Knowledge Graph System
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
from datetime import datetime

from rag_system import RAGSystem
from config import DataSourceConfig, ModelConfig, get_config

# Page configuration
st.set_page_config(
    page_title="RAG Knowledge Graph System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .response-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system (cached)"""
    return RAGSystem()

def main():
    """Main application function"""
    
    # Initialize RAG system
    rag_system = initialize_rag_system()
    
    # Sidebar
    st.sidebar.title("🧠 RAG System")
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["🏠 Dashboard", "❓ Ask Questions", "📊 System Stats", "⚙️ Configuration", "📈 Knowledge Graph"]
    )
    
    if page == "🏠 Dashboard":
        show_dashboard(rag_system)
    elif page == "❓ Ask Questions":
        show_question_interface(rag_system)
    elif page == "📊 System Stats":
        show_system_stats(rag_system)
    elif page == "⚙️ Configuration":
        show_configuration(rag_system)
    elif page == "📈 Knowledge Graph":
        show_knowledge_graph(rag_system)

def show_dashboard(rag_system):
    """Show the main dashboard"""
    st.markdown('<h1 class="main-header">RAG Knowledge Graph System</h1>', unsafe_allow_html=True)
    
    # Initialize system if not already done
    if not rag_system.is_initialized:
        if st.button("🚀 Initialize System"):
            with st.spinner("Initializing RAG System..."):
                success = rag_system.initialize()
                if success:
                    st.success("System initialized successfully!")
                    st.rerun()
                else:
                    st.error("Failed to initialize system. Check configuration.")
    
    # System status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("System Status", "✅ Ready" if rag_system.is_initialized else "⏳ Not Initialized")
    
    with col2:
        current_model = rag_system.llm_manager.get_current_model()
        st.metric("Current Model", current_model or "None")
    
    with col3:
        doc_count = sum(len(docs) for docs in rag_system.documents.values())
        st.metric("Documents Loaded", doc_count)
    
    with col4:
        st.metric("Questions Asked", len(rag_system.session_history))
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh System"):
            with st.spinner("Refreshing..."):
                rag_system.initialize(force_reload=True)
                st.success("System refreshed!")
                st.rerun()
    
    with col2:
        if st.button("💾 Save Session"):
            filename = rag_system.save_session()
            st.success(f"Session saved to {filename}")
    
    with col3:
        if st.button("🗑️ Clear Cache"):
            rag_system.clear_cache()
            st.success("Cache cleared!")
            st.rerun()
    
    # Recent questions
    if rag_system.session_history:
        st.subheader("📝 Recent Questions")
        
        for i, response in enumerate(reversed(rag_system.session_history[-5:])):
            with st.expander(f"Q: {response['question'][:50]}..."):
                st.write(f"**Answer:** {response['answer']}")
                st.write(f"**Model:** {response['model_used']}")
                st.write(f"**Sources:** {len(response['sources'])} documents")
                st.write(f"**Time:** {response['timestamp']}")

def show_question_interface(rag_system):
    """Show the question asking interface"""
    st.title("❓ Ask Questions")
    
    if not rag_system.is_initialized:
        st.warning("Please initialize the system first from the Dashboard.")
        return
    
    # Question input
    question = st.text_area("Enter your question:", height=100, placeholder="Ask any question about the loaded documentation...")
    
    # Options
    col1, col2 = st.columns(2)
    
    with col1:
        use_kg = st.checkbox("Use Knowledge Graph", value=True, help="Include knowledge graph insights in the response")
        max_docs = st.slider("Max Context Documents", 1, 10, 5)
    
    with col2:
        # Model selection
        available_models = rag_system.llm_manager.get_available_models()
        current_model = rag_system.llm_manager.get_current_model()
        
        selected_model = st.selectbox(
            "Select Model",
            available_models,
            index=available_models.index(current_model) if current_model in available_models else 0
        )
        
        if selected_model != current_model:
            rag_system.switch_model(selected_model)
            st.success(f"Switched to {selected_model}")
    
    # Ask question
    if st.button("🔍 Ask Question", type="primary"):
        if question.strip():
            with st.spinner("Processing your question..."):
                try:
                    response = rag_system.ask_question(
                        question, 
                        use_knowledge_graph=use_kg,
                        max_context_docs=max_docs
                    )
                    
                    # Display response
                    st.markdown('<div class="response-box">', unsafe_allow_html=True)
                    st.write("### Answer")
                    st.write(response['answer'])
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display sources
                    if response['sources']:
                        st.write("### Sources")
                        for i, source in enumerate(response['sources']):
                            st.write(f"{i+1}. **{source['source']}** (Score: {source['score']:.3f})")
                    
                    # Display knowledge graph insights
                    if response.get('knowledge_graph_insights'):
                        kg_insights = response['knowledge_graph_insights']
                        
                        if kg_insights.get('entities'):
                            st.write("### Related Entities")
                            for entity in kg_insights['entities']:
                                st.write(f"- **{entity['id']}**: {entity.get('attributes', {})}")
                        
                        if kg_insights.get('relationships'):
                            st.write("### Related Relationships")
                            for rel in kg_insights['relationships']:
                                st.write(f"- **{rel['source']}** → **{rel['target']}**: {rel.get('attributes', {})}")
                    
                    # Display metadata
                    with st.expander("Response Metadata"):
                        st.json(response)
                        
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")
        else:
            st.warning("Please enter a question.")

def show_system_stats(rag_system):
    """Show system statistics"""
    st.title("📊 System Statistics")
    
    if not rag_system.is_initialized:
        st.warning("Please initialize the system first from the Dashboard.")
        return
    
    # Get system stats
    stats = rag_system.get_system_stats()
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Data Sources", stats['data_sources'])
    
    with col2:
        st.metric("Documents Loaded", stats['loaded_documents'])
    
    with col3:
        st.metric("Questions Asked", stats['session_questions'])
    
    with col4:
        st.metric("Current Model", stats['current_model'])
    
    # Data source details
    st.subheader("📁 Data Sources")
    
    if stats['data_source_details']:
        source_data = []
        for name, details in stats['data_source_details'].items():
            source_data.append({
                'Name': name,
                'Type': details['type'],
                'Format': details['format'],
                'Enabled': '✅' if details['enabled'] else '❌',
                'Documents': details['document_count']
            })
        
        df_sources = pd.DataFrame(source_data)
        st.dataframe(df_sources, use_container_width=True)
    
    # Vector store stats
    st.subheader("🗄️ Vector Store")
    vector_stats = stats.get('vector_store', {})
    
    if vector_stats:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Type", vector_stats.get('type', 'Unknown'))
        
        with col2:
            st.metric("Status", vector_stats.get('status', 'Unknown'))
        
        with col3:
            st.metric("Documents", vector_stats.get('document_count', 0))
    
    # Knowledge graph stats
    if stats['knowledge_graph_enabled']:
        st.subheader("🕸️ Knowledge Graph")
        kg_stats = stats.get('knowledge_graph', {})
        
        if kg_stats and kg_stats.get('status') == 'populated':
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Nodes", kg_stats.get('nodes', 0))
            
            with col2:
                st.metric("Edges", kg_stats.get('edges', 0))
            
            with col3:
                st.metric("Density", f"{kg_stats.get('density', 0):.3f}")
            
            with col4:
                st.metric("Components", kg_stats.get('connected_components', 0))
            
            # Create graph visualization
            if st.button("📈 Visualize Knowledge Graph"):
                with st.spinner("Generating visualization..."):
                    rag_system.visualize_knowledge_graph()
                    st.success("Knowledge graph visualization generated!")
        else:
            st.info("Knowledge graph is empty or not available.")
    
    # Session history chart
    if rag_system.session_history:
        st.subheader("📈 Question History")
        
        # Create timeline
        timeline_data = []
        for response in rag_system.session_history:
            timeline_data.append({
                'Question': response['question'][:50] + "..." if len(response['question']) > 50 else response['question'],
                'Model': response['model_used'],
                'Sources': len(response['sources']),
                'Timestamp': response['timestamp']
            })
        
        df_timeline = pd.DataFrame(timeline_data)
        df_timeline['Timestamp'] = pd.to_datetime(df_timeline['Timestamp'])
        
        # Model usage chart
        model_counts = df_timeline['Model'].value_counts()
        fig_model = px.pie(values=model_counts.values, names=model_counts.index, title="Model Usage")
        st.plotly_chart(fig_model, use_container_width=True)

def show_configuration(rag_system):
    """Show configuration interface"""
    st.title("⚙️ Configuration")
    
    # Model configuration
    st.subheader("🤖 Model Configuration")
    
    available_models = rag_system.llm_manager.get_available_models()
    current_model = rag_system.llm_manager.get_current_model()
    
    # Model selection
    selected_model = st.selectbox(
        "Current Model",
        available_models,
        index=available_models.index(current_model) if current_model in available_models else 0
    )
    
    if selected_model != current_model:
        if st.button("Switch Model"):
            success = rag_system.switch_model(selected_model)
            if success:
                st.success(f"Switched to {selected_model}")
                st.rerun()
            else:
                st.error("Failed to switch model")
    
    # Add new model
    st.write("### Add New Model")
    
    with st.form("add_model"):
        model_name = st.text_input("Model Name")
        provider = st.selectbox("Provider", ["openai", "anthropic", "local"])
        model_type = st.text_input("Model Type (e.g., gpt-4, claude-3-sonnet)")
        api_key = st.text_input("API Key", type="password")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.1, 0.1)
        
        if st.form_submit_button("Add Model"):
            if model_name and model_type:
                config = ModelConfig(
                    name=model_name,
                    provider=provider,
                    model_name=model_type,
                    api_key=api_key if api_key else None,
                    temperature=temperature
                )
                rag_system.llm_manager.add_model(model_name, config)
                st.success(f"Added model: {model_name}")
                st.rerun()
    
    # Data source configuration
    st.subheader("📁 Data Source Configuration")
    
    # Add new data source
    st.write("### Add New Data Source")
    
    with st.form("add_data_source"):
        source_name = st.text_input("Source Name")
        source_type = st.selectbox("Type", ["file", "url", "api"])
        source_path = st.text_input("Path/URL")
        source_format = st.selectbox("Format", ["txt", "pdf", "md", "html", "json", "csv"])
        chunk_size = st.number_input("Chunk Size", 100, 2000, 1000)
        chunk_overlap = st.number_input("Chunk Overlap", 0, 500, 200)
        
        if st.form_submit_button("Add Data Source"):
            if source_name and source_path:
                source = DataSourceConfig(
                    name=source_name,
                    type=source_type,
                    path=source_path,
                    format=source_format,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                success = rag_system.add_data_source(source)
                if success:
                    st.success(f"Added data source: {source_name}")
                    st.rerun()
                else:
                    st.error(f"Failed to add data source: {source_name}")
    
    # Remove data source
    st.write("### Remove Data Source")
    
    if rag_system.data_manager.data_sources:
        source_to_remove = st.selectbox(
            "Select source to remove",
            [source.name for source in rag_system.data_manager.data_sources]
        )
        
        if st.button("Remove Selected Source"):
            success = rag_system.remove_data_source(source_to_remove)
            if success:
                st.success(f"Removed data source: {source_to_remove}")
                st.rerun()
            else:
                st.error(f"Failed to remove data source: {source_to_remove}")

def show_knowledge_graph(rag_system):
    """Show knowledge graph interface"""
    st.title("📈 Knowledge Graph")
    
    if not rag_system.is_initialized:
        st.warning("Please initialize the system first from the Dashboard.")
        return
    
    if not rag_system.knowledge_graph:
        st.warning("Knowledge graph is not enabled.")
        return
    
    # Graph statistics
    kg_stats = rag_system.knowledge_graph.get_graph_stats()
    
    if kg_stats.get('status') == 'populated':
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Nodes", kg_stats['nodes'])
        
        with col2:
            st.metric("Edges", kg_stats['edges'])
        
        with col3:
            st.metric("Density", f"{kg_stats['density']:.3f}")
        
        with col4:
            st.metric("Components", kg_stats['connected_components'])
        
        # Entity search
        st.subheader("🔍 Search Entities")
        
        search_query = st.text_input("Search for entities or relationships:")
        
        if search_query:
            results = rag_system.knowledge_graph.query_graph(search_query)
            
            if results:
                st.write(f"Found {len(results)} results:")
                
                for i, result in enumerate(results):
                    if result['type'] == 'node':
                        result_title = result['id']
                    else:
                        result_title = f"{result['source']} → {result['target']}"
                    
                    with st.expander(f"Result {i+1}: {result_title}"):
                        st.json(result)
            else:
                st.info("No results found.")
        
        # Entity relationships
        st.subheader("🔗 Entity Relationships")
        
        entity_name = st.text_input("Enter entity name to explore relationships:")
        
        if entity_name:
            relationships = rag_system.get_entity_relationships(entity_name)
            
            if 'error' not in relationships:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("### Incoming Relationships")
                    for rel in relationships['incoming']:
                        st.write(f"**{rel['source']}** → {entity_name}")
                
                with col2:
                    st.write("### Outgoing Relationships")
                    for rel in relationships['outgoing']:
                        st.write(f"{entity_name} → **{rel['target']}**")
                
                st.write("### Neighbors")
                for neighbor in relationships['neighbors']:
                    st.write(f"- {neighbor}")
            else:
                st.error(relationships['error'])
        
        # Visualization
        st.subheader("📊 Graph Visualization")
        
        max_nodes = st.slider("Max nodes to display", 10, 100, 50)
        
        if st.button("Generate Visualization"):
            with st.spinner("Generating knowledge graph visualization..."):
                rag_system.visualize_knowledge_graph(max_nodes=max_nodes)
                st.success("Visualization generated! Check the output directory.")
    else:
        st.info("Knowledge graph is empty. Please initialize the system with documents.")

if __name__ == "__main__":
    main() 