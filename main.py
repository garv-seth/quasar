"""Main Streamlit application for QUASAR framework demonstration."""

import streamlit as st
import logging
import asyncio
from quantum_agent_framework.integration import HybridComputation

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
    .main { padding: 2rem; }
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
</style>
""", unsafe_allow_html=True)

def display_quantum_metrics(result: dict):
    """Display quantum processing metrics with enhanced visualization."""
    try:
        st.markdown('<div class="quantum-metrics">', unsafe_allow_html=True)
        st.markdown("### 🔄 Quantum Processing Metrics")

        # Display key metrics in columns
        cols = st.columns(4)
        with cols[0]:
            st.metric("Processing Type", result.get('method_used', 'Unknown').upper())
        with cols[1]:
            st.metric("Backend Used", result.get('backend', 'Unknown'))
        with cols[2]:
            st.metric("Circuit Depth", result.get('details', {}).get('circuit_depth', 0))
        with cols[3]:
            st.metric("Computation Time", f"{result.get('computation_time', 0):.4f}s")

        # Display factorization results if available
        if result.get('factors'):
            st.markdown('<div class="factorization-result">', unsafe_allow_html=True)
            st.markdown("#### 🧮 Factorization Results")

            # Show all factors
            factors = result['factors']
            st.success(f"All factors in ascending order: {', '.join(map(str, factors))}")

            # Show computation details
            st.info(f"Computation method: {result.get('method_used', 'Unknown').upper()}")
            st.info(f"Backend used: {result.get('backend', 'Unknown')}")

            if 'response' in result:
                st.markdown("#### 📝 Explanation")
                st.write(result['response'])

            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        logging.error(f"Error displaying metrics: {str(e)}")
        st.error("Error displaying metrics. The computation results are still valid.")

async def process_task(task: str, n_qubits: int = 8, use_quantum: bool = True) -> dict:
    """Process user task with quantum acceleration."""
    try:
        hybrid_computer = HybridComputation(
            n_qubits=n_qubits,
            use_quantum=use_quantum,
            use_azure=True
        )
        result = await hybrid_computer.process_task(task)
        logging.info(f"Task processing result: {result}")
        return result
    except Exception as e:
        logging.error(f"Error during task processing: {str(e)}")
        return {"error": str(e)}

def main():
    st.title("⚛️ QUASAR: Quantum Search and Reasoning")
    st.markdown("""
    *Quantum-Accelerated AI Agent (Q3A) powered by Azure Quantum and IonQ*

    Specialized in:
    - 🧮 Large number factorization
    - 📊 Resource optimization
    - 🔄 Parallel computing
    """)

    # Sidebar for quantum settings
    with st.sidebar:
        st.header("⚙️ Quantum Settings")
        use_quantum = st.checkbox("Enable Quantum Acceleration", value=True)
        n_qubits = st.slider(
            "Number of Qubits",
            min_value=4,
            max_value=29,
            value=8,
            help="More qubits allow processing larger numbers"
        )

    # Main interface
    task = st.text_area(
        "Enter your query:",
        placeholder="Examples:\n- Factor a number: 'Factor 25' or 'Find all factors of 3960'\n- Optimize resources: 'Optimize distribution of 1000 items'",
        help="For mathematical tasks, quantum computing will be used when advantageous"
    )

    if st.button("🚀 Process", disabled=not task):
        with st.spinner("🔄 Processing..."):
            try:
                # Create event loop for async execution
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Process task
                result = loop.run_until_complete(process_task(task, n_qubits, use_quantum))

                if result and 'error' not in result:
                    # Display results
                    display_quantum_metrics(result)
                else:
                    st.error(f"An error occurred: {result.get('error', 'Unknown error')}")

            except Exception as e:
                logging.error(f"Error: {str(e)}")
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()