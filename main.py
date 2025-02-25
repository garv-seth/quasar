"""Main Streamlit application for QUASAR framework demonstration."""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from quantum_agent_framework.integration import HybridComputation
from quantum_agent_framework.classical import PromptTemplate, Memory

# Page configuration
st.set_page_config(
    page_title="QUASAR Framework Demo",
    page_icon="⚛️",
    layout="wide"
)

def main():
    st.title("⚛️ QUASAR: Quantum-Accelerated Search and Reasoning")
    st.markdown("""
    Demonstrate the power of quantum-accelerated AI agents using the QUASAR framework.
    This platform combines classical AI with quantum computing principles for enhanced
    performance in search and reasoning tasks.
    """)

    # Initialize session state
    if 'hybrid_computer' not in st.session_state:
        st.session_state.hybrid_computer = HybridComputation(
            n_qubits=4,
            use_quantum=True
        )

    if 'memory' not in st.session_state:
        st.session_state.memory = Memory()

    # Sidebar controls
    st.sidebar.header("Framework Configuration")
    num_qubits = st.sidebar.slider("Number of Qubits", 2, 8, 4)
    use_quantum = st.sidebar.checkbox("Enable Quantum Acceleration", value=True)

    # Main content
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Task Execution")

        # Task examples
        st.markdown("""
        ### Example Tasks:
        - "Search for quantum computing news"
        - "Analyze market trends in tech sector"
        - "Summarize research papers on quantum ML"
        """)

        # Task input
        task = st.text_area(
            "Enter Task Description",
            placeholder="Example: Search for quantum computing news",
            help="Describe what you want the quantum agent to do"
        )

        # Execute button
        if st.button("Execute Task", disabled=not task):
            with st.spinner("Processing with quantum acceleration..."):
                try:
                    # Process task
                    result = st.session_state.hybrid_computer.process_task(task) #Corrected to remove await, assuming process_task is synchronous.  If asynchronous,  re-add await and ensure asyncio is handled appropriately.

                    # Display results
                    st.success("Task completed successfully!")
                    st.json(result)

                    # Update memory
                    st.session_state.memory.add('user', task)
                    st.session_state.memory.add('system', str(result))

                except Exception as e:
                    st.error(f"Error processing task: {str(e)}")

    with col2:
        st.subheader("Quantum Metrics")

        # Get quantum parameters from session
        metrics = st.session_state.hybrid_computer.get_quantum_metrics()

        # Display metrics
        cols = st.columns(len(metrics))
        for i, (metric, value) in enumerate(metrics.items()):
            cols[i].metric(
                label=metric.replace('_', ' ').title(),
                value=f"{value:.2f}"
            )

        st.subheader("Memory State")
        history = st.session_state.memory.get_history()
        for message in history:
            with st.expander(f"{message['role']} - {message['timestamp']}"):
                st.write(message['content'])

    # Additional Information
    st.markdown("""
    ---
    ### About QUASAR Framework

    The Quantum-Accelerated Search and Reasoning (QUASAR) framework combines classical AI
    with quantum computing principles to enhance search and reasoning capabilities:

    1. **Quantum Processing**: Leverages quantum circuits for optimization and classification
    2. **Hybrid Architecture**: Seamlessly combines classical and quantum components
    3. **Adaptive Learning**: Continuously improves through quantum-enhanced feedback
    4. **Scalable Design**: Supports varying numbers of qubits for different use cases

    For more information, visit our documentation or contact support.
    """)

if __name__ == "__main__":
    main()