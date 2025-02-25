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
    .factorization-result {
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
            st.metric("Backend Used", metrics.get('backend', 'Unknown'))
        with cols[2]:
            st.metric("Circuit Depth", metrics.get('circuit_depth', 0))
        with cols[3]:
            st.metric("Quantum Advantage", 
                     f"{metrics.get('quantum_advantage', '0')}x faster" 
                     if metrics.get('quantum_advantage') else "N/A")

        # Display quantum result for factorization
        if 'quantum_result' in metrics and metrics['quantum_result'].get('factors'):
            st.markdown('<div class="factorization-result">', unsafe_allow_html=True)
            st.markdown("#### 🧮 Factorization Results")

            # Show all factors
            factors = metrics['quantum_result']['factors']
            st.success(f"All factors: {', '.join(map(str, factors))}")

            # Show computation details
            st.info(f"Computation method: {metrics['quantum_result'].get('method_used', 'Unknown')}")
            st.info(f"Backend used: {metrics['quantum_result'].get('backend', 'Unknown')}")

            # Show computation time
            st.write(f"Computation time: {metrics['quantum_result'].get('computation_time', 0):.4f} seconds")

            if 'explanation' in metrics['quantum_result']:
                st.markdown("#### 📝 Technical Details")
                st.write(metrics['quantum_result']['explanation'])

            st.markdown('</div>', unsafe_allow_html=True)

        # Display sources section if available
        if 'sources' in metrics and metrics['sources'] and metrics['sources'][0]['title'] != 'N/A':
            st.markdown('<div class="source-section">', unsafe_allow_html=True)
            st.markdown("#### 📚 Quantum Computing Research Sources")
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

                    # Display results
                    if result['task_type'] == 'factorization':
                        st.markdown("### 📊 Factorization Results")
                        if 'quantum_result' in result:
                            if result['quantum_result'].get('success', False):
                                factors = result['quantum_result']['factors']
                                factors_str = ", ".join(map(str, factors))
                                st.success(f"All factors: {factors_str}")

                                # Show computation method
                                method = result['quantum_result'].get('hardware', 'Unknown')
                                st.info(f"Computation method: {method}")

                                # Show computation time
                                time = result['quantum_result'].get('computation_time', 0)
                                st.write(f"Computation time: {time:.4f} seconds")
                            else:
                                st.warning("No factors found or computation failed")
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