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

def format_quantum_results(result: dict) -> str:
    """Format quantum processing results into user-friendly text."""
    if result.get('error'):
        return f"Error processing request: {result.get('message', 'Unknown error')}"

    # Get quantum confidence from optimization history
    confidence = 0.0
    if 'optimization_history' in result:
        # More negative value means higher confidence
        confidence = min(100, abs(result['optimization_history'][-1]) * 100)

    task_class = result.get('task_class', 0)
    task_types = ['Information Retrieval', 'Analysis']

    response = f"Task Type: {task_types[task_class]}\n"
    response += f"Processing Confidence: {confidence:.1f}%\n"

    if result.get('quantum_error'):
        response += "\nNote: Quantum processing encountered an issue and fell back to classical processing."

    return response

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
                    result = st.session_state.hybrid_computer.process_task(task)

                    # Display formatted results
                    st.success("Task processed successfully!")

                    # Show quantum processing details
                    with st.expander("Quantum Processing Details"):
                        st.write(format_quantum_results(result))

                        # Visualization of optimization progress
                        if 'optimization_history' in result:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                y=result['optimization_history'],
                                mode='lines',
                                name='Optimization Progress'
                            ))
                            fig.update_layout(
                                title='Quantum Optimization Progress',
                                xaxis_title='Step',
                                yaxis_title='Cost'
                            )
                            st.plotly_chart(fig)

                    # Update memory
                    st.session_state.memory.add('user', task)
                    st.session_state.memory.add('system', str(result))

                except Exception as e:
                    st.error(f"Error processing task: {str(e)}")

    with col2:
        st.subheader("Quantum System Status")

        # Get quantum parameters from session
        metrics = st.session_state.hybrid_computer.get_quantum_metrics()

        # Display metrics with proper formatting based on value type
        cols = st.columns(len(metrics))
        for i, (metric, value) in enumerate(metrics.items()):
            formatted_value = str(value) if isinstance(value, str) else f"{value:.2f}" if isinstance(value, float) else str(value)
            # Make metric names more user-friendly
            friendly_name = metric.replace('_', ' ').title()
            if metric == 'quantum_enabled':
                friendly_name = 'Quantum Mode'
                formatted_value = 'Enabled' if value > 0 else 'Disabled'
            elif metric == 'quantum_backend':
                friendly_name = 'Processing On'
                formatted_value = value.title()

            cols[i].metric(
                label=friendly_name,
                value=formatted_value
            )

        # Recent Activity
        st.subheader("Recent Activity")
        history = st.session_state.memory.get_history()
        for message in history[-5:]:  # Show only last 5 activities
            with st.expander(f"{message['role'].title()} - {message['timestamp']}"):
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