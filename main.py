"""Main Streamlit application for QUASAR framework demonstration."""

import streamlit as st
import plotly.graph_objects as go
from quantum_agent_framework.integration import HybridComputation
from quantum_agent_framework.agents.web_agent import WebAgent
import asyncio
import json

# Page configuration
st.set_page_config(
    page_title="QUASAR: Quantum AI Agent",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for cleaner UI
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .quantum-metrics {
        font-size: 0.8em;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

def initialize_agents():
    """Initialize quantum agents if not in session state."""
    if 'hybrid_computer' not in st.session_state:
        st.session_state.hybrid_computer = HybridComputation(
            n_qubits=4,
            use_quantum=True,
            use_azure=True
        )

    if 'web_agent' not in st.session_state:
        st.session_state.web_agent = WebAgent(
            optimizer=st.session_state.hybrid_computer.quantum_optimizer,
            preprocessor=st.session_state.hybrid_computer.quantum_preprocessor
        )

def main():
    st.title("⚛️ Quantum-Enhanced AI Agent")

    # Initialize agents
    initialize_agents()

    # Main interface
    task = st.text_area(
        "What would you like me to analyze?",
        placeholder="Example: Analyze current market trends in the AI sector",
        help="I can analyze market trends, technology news, and provide insights"
    )

    if st.button("Analyze", disabled=not task):
        with st.spinner("Processing with quantum acceleration..."):
            try:
                # Create event loop for async execution
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Execute analysis
                result = loop.run_until_complete(
                    st.session_state.web_agent.analyze_market_trends()
                )

                # Display analysis results
                st.markdown("### Analysis Results")
                st.write(result['analysis'])

                # Advanced metrics in expander
                with st.expander("Show Quantum Processing Details"):
                    metrics = result['quantum_metrics']

                    # Display quantum confidence
                    st.metric(
                        "Quantum Processing Confidence",
                        f"{metrics['quantum_confidence']:.1f}%"
                    )

                    # Plot relevance scores
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=[f"Source {i+1}" for i in range(len(metrics['relevance_scores']))],
                        y=metrics['relevance_scores'],
                        name='Source Relevance'
                    ))
                    fig.update_layout(
                        title='Quantum-Computed Source Relevance',
                        xaxis_title='Source',
                        yaxis_title='Relevance Score'
                    )
                    st.plotly_chart(fig)

                    # Additional metrics
                    st.json({
                        'processed_sources': metrics['processed_sources'],
                        'quantum_backend': 'Azure Quantum' if st.session_state.hybrid_computer.quantum_optimizer.use_azure else 'Local Simulator',
                        'qubits_used': st.session_state.hybrid_computer.n_qubits
                    })

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

    # Sidebar for advanced settings
    with st.sidebar:
        st.header("Advanced Settings")
        use_quantum = st.checkbox("Enable Quantum Acceleration", value=True)
        if use_quantum:
            n_qubits = st.slider("Number of Qubits", 2, 8, 4)
            st.info("More qubits = more processing power but slower execution")

        st.markdown("---")
        st.markdown("""
        ### About Quantum Processing
        This agent uses quantum computing to:
        1. Optimize information retrieval
        2. Process and analyze data
        3. Find patterns in market trends

        Powered by Azure Quantum
        """)

if __name__ == "__main__":
    main()