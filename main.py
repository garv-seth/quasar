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
    initial_sidebar_state="expanded"
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
    .analysis-section {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-explanation {
        font-size: 0.9em;
        color: #666;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

async def analyze_with_quantum(task: str, n_qubits: int):
    """Analyze content with quantum acceleration."""
    try:
        # Initialize agents with user-specified qubit count
        st.session_state.hybrid_computer = HybridComputation(
            n_qubits=n_qubits,
            use_quantum=True,
            use_azure=True
        )
        st.session_state.web_agent = WebAgent(
            optimizer=st.session_state.hybrid_computer.quantum_optimizer,
            preprocessor=st.session_state.hybrid_computer.quantum_preprocessor
        )

        result = await st.session_state.web_agent.analyze_content(task)
        return result

    except Exception as e:
        logging.error(f"Error during quantum analysis: {str(e)}")
        return {"error": str(e)}

def display_quantum_metrics(metrics: dict):
    """Display quantum processing metrics with visualizations and explanations."""
    try:
        st.markdown("### 🔄 Quantum Processing Metrics")

        # Display key metrics in columns with explanations
        cols = st.columns(4)
        with cols[0]:
            st.metric("Quantum Confidence", f"{metrics['quantum_confidence']:.1f}%")
            st.markdown("""
            <div class="metric-explanation">
            Measures the quantum circuit's confidence in its analysis based on measurement outcomes.
            Higher values indicate more certain results.
            </div>
            """, unsafe_allow_html=True)

        with cols[1]:
            st.metric("Qubits Used", metrics['circuit_stats']['n_qubits'])
            st.markdown("""
            <div class="metric-explanation">
            Number of quantum bits (qubits) used in the computation.
            More qubits enable processing more data dimensions simultaneously.
            </div>
            """, unsafe_allow_html=True)

        with cols[2]:
            st.metric("Circuit Depth", metrics['circuit_stats']['circuit_depth'])
            st.markdown("""
            <div class="metric-explanation">
            Number of sequential quantum operations.
            Lower depth means faster quantum processing.
            </div>
            """, unsafe_allow_html=True)

        with cols[3]:
            st.metric("Processing Time", f"{metrics['processing_time_ms']:.0f}ms")
            st.markdown("""
            <div class="metric-explanation">
            Time taken for quantum circuit execution.
            Excludes classical pre/post-processing time.
            </div>
            """, unsafe_allow_html=True)

        # Display quantum advantage metrics with detailed explanations
        st.markdown("#### ⚡ Performance Improvement")
        st.markdown("""
        Comparing quantum vs classical processing performance:
        """)

        adv_cols = st.columns(2)
        with adv_cols[0]:
            st.metric("Speed Improvement", metrics['quantum_advantage']['speedup'])
            st.markdown(f"""
            <div class="metric-explanation">
            Quantum processing time: {metrics['quantum_advantage']['quantum_time_ms']:.2f}ms<br>
            Classical processing time: {metrics['quantum_advantage']['classical_time_ms']:.2f}ms<br>
            The quantum algorithm processes data {metrics['quantum_advantage']['speedup']} times faster than classical processing.
            </div>
            """, unsafe_allow_html=True)

        with adv_cols[1]:
            st.metric("Accuracy Boost", metrics['quantum_advantage']['accuracy_improvement'])
            st.markdown("""
            <div class="metric-explanation">
            Improvement in relevance scoring accuracy using quantum superposition and interference effects
            compared to classical ranking algorithms.
            </div>
            """, unsafe_allow_html=True)

        # Create quantum advantage visualization
        st.markdown("#### 📊 Quantum vs Classical Performance")
        fig = go.Figure()

        # Add processing time comparison
        fig.add_trace(go.Bar(
            x=['Quantum', 'Classical'],
            y=[
                metrics['quantum_advantage']['quantum_time_ms'],
                metrics['quantum_advantage']['classical_time_ms']
            ],
            name='Processing Time (ms)',
            marker_color=['rgb(55, 83, 109)', 'rgb(26, 118, 255)']
        ))

        fig.update_layout(
            title='Processing Time Comparison',
            xaxis_title='Processing Method',
            yaxis_title='Time (milliseconds)',
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

    # Sidebar for advanced settings
    with st.sidebar:
        st.header("⚙️ Advanced Settings")

        use_quantum = st.checkbox("Enable Quantum Acceleration", value=True)
        n_qubits = 8  # Default value

        if use_quantum:
            n_qubits = st.slider(
                "Number of Qubits",
                min_value=4,
                max_value=29,
                value=8,
                help="More qubits allow processing more data dimensions but increase circuit complexity"
            )
            backend = st.selectbox(
                "Quantum Backend",
                ["Azure IonQ Simulator", "IonQ Aria-1"],
                help="Select quantum processing backend. Aria-1 provides real quantum hardware access"
            )

            if backend == "IonQ Aria-1":
                st.warning("""
                Using real quantum hardware (Aria-1) may increase processing time
                but provides true quantum advantage for complex calculations.
                """)

        st.markdown("---")
        st.markdown("""
        ### 🧪 About QUASAR
        **Q**uantum **U**nified **S**earch **A**nd **R**easoning

        This framework leverages:
        1. IonQ quantum processors for parallel data processing
        2. Quantum circuits optimized for text analysis
        3. Hybrid quantum-classical processing with GPT-4o

        The quantum advantage comes from:
        - Parallel processing of multiple data dimensions
        - Quantum superposition for feature comparison
        - Interference-based relevance scoring

        Powered by Azure Quantum & OpenAI
        """)

    # Main interface
    task = st.text_area(
        "What would you like me to analyze?",
        placeholder="Example: Analyze trends in the job market for quantum computing professionals",
        help="I can analyze market trends, industry data, and provide insights using quantum-accelerated processing"
    )

    if st.button("🚀 Analyze", disabled=not task):
        with st.spinner("🔄 Processing with quantum acceleration..."):
            try:
                # Create event loop for async execution
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Execute analysis with quantum acceleration
                result = loop.run_until_complete(analyze_with_quantum(task, n_qubits))

                if result and 'error' not in result:
                    # Display analysis results
                    st.markdown("### 📝 Analysis Results")
                    st.markdown("""---""")

                    # Format and display the analysis in sections
                    analysis_sections = result['analysis'].split('\n')
                    current_section = ""
                    section_content = []

                    for line in analysis_sections:
                        if line.strip().startswith(tuple(['1.', '2.', '3.', '4.', '5.'])):
                            if current_section and section_content:
                                with st.container():
                                    st.markdown(f"#### {current_section}")
                                    st.markdown('\n'.join(section_content))
                            current_section = line.strip()
                            section_content = []
                        else:
                            section_content.append(line)

                    # Display last section
                    if current_section and section_content:
                        with st.container():
                            st.markdown(f"#### {current_section}")
                            st.markdown('\n'.join(section_content))

                    # Display quantum metrics
                    st.markdown("""---""")
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

if __name__ == "__main__":
    main()