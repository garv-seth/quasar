"""Main Streamlit application for QUASAR framework demonstration."""

import streamlit as st
import plotly.graph_objects as go
from quantum_agent_framework.integration import HybridComputation
import asyncio
import logging
import re
from typing import Dict, Any, List

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
    .processing-type {
        color: #1e88e5;
        font-weight: bold;
        font-size: 1.1em;
        margin: 1rem 0;
    }
    .mathematical-result {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .source-section {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

async def analyze_with_quantum(task: str, n_qubits: int, use_quantum: bool = True) -> Dict[str, Any]:
    """Analyze content with quantum acceleration."""
    try:
        # Initialize hybrid computation system
        hybrid_computer = HybridComputation(
            n_qubits=n_qubits,
            use_quantum=use_quantum,
            use_azure=True
        )

        # Process the task
        result = await hybrid_computer.process_task(task)
        return result

    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
        return {"error": str(e)}

def display_quantum_metrics(metrics: Dict[str, Any]):
    """Display quantum processing metrics with enhanced visualization."""
    try:
        st.markdown('<div class="quantum-metrics">', unsafe_allow_html=True)
        st.markdown("### 🔄 Quantum Processing Metrics")

        # Display key metrics
        cols = st.columns(4)
        with cols[0]:
            st.metric("Processing Type", metrics.get('processing_type', 'Unknown'))
        with cols[1]:
            st.metric("Qubits Used", metrics.get('n_qubits', 0))
        with cols[2]:
            st.metric("Circuit Depth", metrics.get('circuit_depth', 0))
        with cols[3]:
            st.metric("Quantum Advantage", 
                     f"{metrics.get('quantum_advantage', '0')}x faster" 
                     if metrics.get('quantum_advantage') else "N/A")

        # Display specialized metrics for mathematical tasks
        if 'quantum_result' in metrics:
            st.markdown('<div class="mathematical-result">', unsafe_allow_html=True)
            st.markdown("#### 🧮 Mathematical Computation Results")

            # Display factorization or optimization results
            if 'factors' in metrics['quantum_result']:
                st.write("Factorization Results:", metrics['quantum_result']['factors'])
                st.write("Computation Time:", f"{metrics['quantum_result']['computation_time']:.2f} seconds")
            elif 'optimal_solution' in metrics['quantum_result']:
                st.write("Optimization Results:", metrics['quantum_result']['optimal_solution'])
                st.write("Convergence Status:", metrics['quantum_result']['convergence'])

            st.markdown('</div>', unsafe_allow_html=True)

        # Display sources section
        if 'sources' in metrics and metrics['sources']:
            st.markdown('<div class="source-section">', unsafe_allow_html=True)
            st.markdown("#### 📚 Academic and Government Sources")
            for source in metrics['sources']:
                st.markdown(f"- [{source['title']}]({source['url']})")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        logging.error(f"Error displaying metrics: {str(e)}")
        st.warning("Unable to display some metrics. The analysis results are still valid.")

def main():
    st.title("⚛️ QUASAR: Quantum Search and Reasoning")
    st.markdown("""
    *Quantum-Accelerated AI Agent (Q3A) powered by Azure Quantum and IonQ*

    Specialized in:
    - 🧮 Large number factorization
    - 📊 Resource optimization
    - 🔄 Parallel computing
    - 📚 Academic research integration
    """)

    # Sidebar for advanced settings
    with st.sidebar:
        st.header("⚙️ Quantum Settings")

        use_quantum = st.checkbox("Enable Quantum Acceleration", value=True)
        n_qubits = 8

        if use_quantum:
            n_qubits = st.slider(
                "Number of Qubits",
                min_value=4,
                max_value=29,
                value=8,
                help="More qubits allow processing larger numbers and more complex optimizations"
            )

            task_type = st.radio(
                "Task Type",
                ["Mathematical", "Optimization", "General"],
                help="Select the type of task to optimize quantum resource usage"
            )

            st.markdown("---")
            st.markdown("""
            ### 🧪 About QUASAR

            Specialized Capabilities:
            1. **Number Factorization**: Using Shor's algorithm
            2. **Resource Optimization**: Quantum-enhanced QAOA
            3. **Parallel Computing**: Quantum superposition
            4. **Academic Integration**: Direct access to research databases

            Powered by Azure Quantum & IonQ
            """)

    # Main interface
    task = st.text_area(
        "Enter your query:",
        placeholder="""Examples:
- Factor a large number: "Factor 15226050279225333605356183781326374297180681149613"
- Optimize resources: "Optimize distribution of 1000 items across 50 locations"
- General query: "Analyze recent quantum computing breakthroughs"
        """,
        help="For mathematical tasks, quantum computing will be used for exponential speedup"
    )

    if st.button("🚀 Process", disabled=not task):
        with st.spinner("🔄 Processing with quantum acceleration..."):
            try:
                # Create event loop for async execution
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Execute analysis
                result = loop.run_until_complete(analyze_with_quantum(task, n_qubits, use_quantum))

                if result and 'error' not in result:
                    # Display processing type
                    st.markdown(
                        f'<div class="processing-type">Processing Type: {result["task_type"].upper()}</div>',
                        unsafe_allow_html=True
                    )

                    # Display results based on task type
                    if result['task_type'] == 'mathematical':
                        st.markdown("### 📊 Mathematical Results")
                        st.json(result['quantum_result'])
                    else:
                        st.markdown("### 📝 Analysis Results")
                        st.write(result.get('classical_result', {}))

                    # Display quantum metrics
                    display_quantum_metrics(result)

                else:
                    st.error("An error occurred during analysis. Please try again.")

            except Exception as e:
                logging.error(f"Error during analysis: {str(e)}")
                st.error("An error occurred during analysis. Please try again.")

if __name__ == "__main__":
    main()