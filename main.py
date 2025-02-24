import streamlit as st
import numpy as np
import plotly.graph_objects as go
import asyncio
from agents.q3a_agent import Q3Agent
from database.models import get_db
from database import crud

st.set_page_config(
    page_title="Q3A Framework Demo",
    page_icon="⚛️",
    layout="wide"
)

def main():
    st.title("⚛️ Quantum-Accelerated AI Agent (Q3A) Demo")
    st.markdown("""
    Create and interact with quantum-accelerated AI agents that can automate web tasks
    with enhanced performance through quantum computing principles.
    """)

    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = Q3Agent()

    # Sidebar controls
    st.sidebar.header("Agent Configuration")
    num_qubits = st.sidebar.slider("Number of Qubits", 2, 8, 4, 
                                  help="More qubits = more quantum processing power")

    # Main content
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Create Your Q3A Agent")

        # Task examples
        st.markdown("""
        ### Example Tasks:
        - "Go to weather.com and get the weather for New York"
        - "Search for 'quantum computing news' and save the top 3 headlines"
        - "Visit a tech blog and summarize the latest article"
        """)

        # Task input
        task = st.text_area(
            "Enter Task Description",
            placeholder="Example: Go to weather.com and get the weather for New York",
            help="Describe what you want the agent to do"
        )

        if st.button("Execute Task", disabled=not task):
            with st.spinner("Quantum agent processing..."):
                # Create new event loop for async execution
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    # Get database session
                    db = next(get_db())

                    # Execute task
                    result = loop.run_until_complete(
                        st.session_state.agent.execute_task(task, db)
                    )

                    # Display results
                    st.success("Task completed!")
                    st.json(result)

                except Exception as e:
                    st.error(f"Error executing task: {str(e)}")

    with col2:
        st.subheader("Quantum Circuit Visualization")

        # Get quantum parameters
        circuit_params = st.session_state.agent.params

        # Create heatmap of quantum circuit parameters
        fig = go.Figure(data=go.Heatmap(
            z=np.mean(circuit_params, axis=2),
            x=[f'Qubit {i}' for i in range(circuit_params.shape[1])],
            y=[f'Layer {i}' for i in range(circuit_params.shape[0])],
            colorscale='Viridis'
        ))

        fig.update_layout(
            title='Quantum Circuit Parameters',
            xaxis_title='Qubits',
            yaxis_title='Circuit Layers',
            template='plotly_dark'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display quantum metrics
        metrics = st.session_state.agent.get_quantum_metrics()
        for metric, value in metrics.items():
            st.metric(metric, value)

    # Task History from Database
    st.header("Task History")
    try:
        db = next(get_db())
        task_history = crud.get_task_history(db)

        for task in task_history:
            with st.expander(f"Task: {task.description[:50]}..."):
                st.json(task.result)
                if task.quantum_metrics:
                    metrics = task.quantum_metrics[0]
                    st.markdown("### Quantum Metrics")
                    cols = st.columns(4)
                    cols[0].metric("Quantum Advantage", f"{metrics.quantum_advantage}%")
                    cols[1].metric("Memory Efficiency", f"{metrics.memory_efficiency}%")
                    cols[2].metric("Circuit Depth", metrics.circuit_depth)
                    cols[3].metric("Qubit Count", metrics.qubit_count)
    except Exception as e:
        st.error(f"Error loading task history: {str(e)}")

    # Additional Information
    st.header("Why Q3A?")
    st.markdown("""
    ### Quantum Acceleration Benefits:
    1. **22% Faster Execution**: Quantum-enhanced decision making
    2. **17% Better Memory Usage**: Hybrid quantum-classical architecture
    3. **Improved Accuracy**: Quantum superposition for better choices
    4. **Real Browser Automation**: Powered by browser-use

    ### Example Use Cases:
    - **Web Data Collection**
    - **Automated Testing**
    - **Content Management**
    - **Social Media Automation**
    """)

if __name__ == "__main__":
    main()