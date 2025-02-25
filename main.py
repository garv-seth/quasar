"""Main Streamlit application for QUASAR framework demonstration."""

import streamlit as st
import plotly.graph_objects as go
from quantum_agent_framework.integration import HybridComputation
from quantum_agent_framework.agents.web_agent import WebAgent
import asyncio
import json
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Page configuration
st.set_page_config(
    page_title="QUASAR: Quantum Search and Reasoning",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for improved UI
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .quantum-metrics {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stats-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
    }
</style>
""", unsafe_allow_html=True)

async def analyze_with_quantum(task: str):
    """Analyze content with quantum acceleration."""
    try:
        # Process results with quantum agent
        if not hasattr(st.session_state, 'web_agent'):
            st.error("Quantum agent not initialized. Please refresh the page.")
            return None

        result = await st.session_state.web_agent.analyze_content(task)
        return result

    except Exception as e:
        logging.error(f"Error during quantum analysis: {str(e)}")
        return {"error": str(e)}

def initialize_agents():
    """Initialize quantum agents with optimal configuration."""
    try:
        if 'hybrid_computer' not in st.session_state:
            st.session_state.hybrid_computer = HybridComputation(
                n_qubits=8,  # Optimized for balance between performance and quantum advantage
                use_quantum=True,
                use_azure=True
            )

        if 'web_agent' not in st.session_state:
            st.session_state.web_agent = WebAgent(
                optimizer=st.session_state.hybrid_computer.quantum_optimizer,
                preprocessor=st.session_state.hybrid_computer.quantum_preprocessor
            )
            logging.info("Successfully initialized quantum agents")
        return True
    except Exception as e:
        logging.error(f"Failed to initialize agents: {str(e)}")
        st.error("Error initializing quantum agents. Please try refreshing the page.")
        return False

def display_quantum_metrics(metrics: dict):
    """Display quantum processing metrics with visualizations."""
    try:
        st.markdown("### 🔄 Quantum Processing Metrics")

        # Display key metrics in columns
        cols = st.columns(4)
        with cols[0]:
            st.metric("Quantum Confidence", f"{metrics['quantum_confidence']:.1f}%")
        with cols[1]:
            st.metric("Qubits Used", metrics['circuit_stats']['n_qubits'])
        with cols[2]:
            st.metric("Circuit Depth", metrics['circuit_stats']['circuit_depth'])
        with cols[3]:
            st.metric("Processing Time", f"{metrics['processing_time_ms']}ms")

        # Create quantum advantage visualization
        st.markdown("#### 📊 Quantum Advantage Analysis")
        fig = go.Figure()

        # Add source relevance scores
        fig.add_trace(go.Bar(
            x=[f"Source {i+1}" for i in range(len(metrics['relevance_scores']))],
            y=metrics['relevance_scores'],
            name='Source Relevance',
            marker_color='rgb(55, 83, 109)'
        ))

        fig.update_layout(
            title='Quantum-Computed Source Relevance',
            xaxis_title='Source',
            yaxis_title='Relevance Score',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display technical details in expander
        with st.expander("🔍 Detailed Quantum Circuit Statistics"):
            st.json(metrics['circuit_stats'])

    except Exception as e:
        logging.error(f"Error displaying metrics: {str(e)}")
        st.warning("Unable to display some metrics. The analysis results are still valid.")

def main():
    st.title("⚛️ QUASAR: Quantum Search and Reasoning")
    st.markdown("""
    *Quantum-Accelerated AI Agent (Q3A) powered by Azure Quantum and IonQ*
    """)

    # Initialize agents with error handling
    if not initialize_agents():
        return

    # Main interface
    task = st.text_area(
        "What would you like me to analyze?",
        placeholder="Example: Analyze emerging trends in quantum computing applications",
        help="I can analyze any topic using quantum-accelerated processing"
    )

    if st.button("🚀 Analyze", disabled=not task):
        with st.spinner("🔄 Processing with quantum acceleration..."):
            try:
                # Create event loop for async execution
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Execute analysis with quantum acceleration
                result = loop.run_until_complete(analyze_with_quantum(task))

                if result and 'error' not in result:
                    # Display analysis results
                    st.markdown("### 📝 Analysis Results")
                    st.write(result['analysis'])

                    # Display quantum metrics
                    display_quantum_metrics(result['quantum_metrics'])

                    # Show sources
                    st.markdown("### 📚 Sources")
                    for url in result['quantum_metrics']['sources']:
                        st.markdown(f"- {url}")
                else:
                    st.error("An error occurred during analysis. Please try again.")

            except Exception as e:
                logging.error(f"Error during analysis: {str(e)}")
                st.error(f"An error occurred during analysis. Please try again.")

    # Sidebar for advanced settings
    with st.sidebar:
        st.header("⚙️ Advanced Settings")

        use_quantum = st.checkbox("Enable Quantum Acceleration", value=True)
        if use_quantum:
            n_qubits = st.slider("Number of Qubits", 4, 29, 8)
            st.info("Using 8 qubits provides optimal balance between speed and quantum advantage")

        st.markdown("---")
        st.markdown("""
        ### 🧪 About QUASAR
        **Q**uantum **U**nified **S**earch **A**nd **R**easoning

        This framework leverages:
        1. IonQ's quantum simulator
        2. Advanced quantum circuits for pattern recognition
        3. Realtime AI processing with GPT-4o-mini

        Powered by Azure Quantum & OpenAI
        """)

if __name__ == "__main__":
    main()