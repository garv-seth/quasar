import streamlit as st
import gymnasium as gym
import numpy as np
import plotly.graph_objects as go
from agents.quantum_agent import QUASARAgent
from agents.classical_agent import ClassicalAgent
from components.visualization import create_comparison_chart, create_circuit_visualization
from components.metrics import calculate_metrics

st.set_page_config(
    page_title="QUASAR Framework Demo",
    page_icon="⚛️",
    layout="wide"
)

def main():
    st.title("⚛️ QUASAR Framework Demonstration")
    st.markdown("""
    Compare the performance of quantum-accelerated AI agents using the QUASAR framework
    against classical reinforcement learning agents.
    """)

    # Sidebar controls
    st.sidebar.header("Configuration")
    num_episodes = st.sidebar.slider("Number of Episodes", 10, 200, 50)
    learning_rate = st.sidebar.number_input("Learning Rate", 0.001, 0.1, 0.01)
    
    # Initialize environment and agents
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    quantum_agent = QUASARAgent(state_dim, action_dim, learning_rate)
    classical_agent = ClassicalAgent(state_dim, action_dim, learning_rate)

    # Training section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("QUASAR Agent")
        quantum_metrics = []
        quantum_progress = st.progress(0)
        quantum_status = st.empty()

    with col2:
        st.subheader("Classical Agent")
        classical_metrics = []
        classical_progress = st.progress(0)
        classical_status = st.empty()

    if st.button("Start Training"):
        for episode in range(num_episodes):
            # Train QUASAR agent
            q_reward = quantum_agent.train_episode(env)
            quantum_metrics.append(q_reward)
            quantum_progress.progress((episode + 1) / num_episodes)
            quantum_status.text(f"Episode {episode + 1}: Reward = {q_reward}")

            # Train classical agent
            c_reward = classical_agent.train_episode(env)
            classical_metrics.append(c_reward)
            classical_progress.progress((episode + 1) / num_episodes)
            classical_status.text(f"Episode {episode + 1}: Reward = {c_reward}")

            # Update visualization
            if (episode + 1) % 5 == 0:
                fig = create_comparison_chart(quantum_metrics, classical_metrics)
                st.plotly_chart(fig, use_container_width=True)

        # Final metrics
        st.header("Training Results")
        metrics_comparison = calculate_metrics(quantum_metrics, classical_metrics)
        st.table(metrics_comparison)

        # Quantum circuit visualization
        st.header("Quantum Circuit Visualization")
        circuit_fig = create_circuit_visualization(quantum_agent.get_circuit_params())
        st.plotly_chart(circuit_fig, use_container_width=True)

if __name__ == "__main__":
    main()
