"""Main entry point for the Q3A: Quantum-Accelerated AI Agent."""

import os
import asyncio
import streamlit as st
import numpy as np
import pandas as pd
import time
import random
import math
import json
from datetime import datetime
from ai_agent import Q3AAgent

# Page configuration
st.set_page_config(
    page_title="QUASAR: Quantum AI Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5em;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 0.5em;
        background: linear-gradient(45deg, #1E3B70, #29539B);
        padding: 0.5em;
        border-radius: 10px;
    }
    .subheader {
        font-size: 1.5em;
        font-weight: bold;
        margin-top: 1em;
        margin-bottom: 0.5em;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1em;
        border-radius: 10px;
        margin-bottom: 1em;
    }
    .quantum-stats {
        font-size: 0.8em;
        color: #4a4a4a;
    }
    .quantum-badge {
        background-color: #29539B;
        color: white;
        padding: 0.2em 0.6em;
        border-radius: 10px;
        font-size: 0.8em;
        margin-right: 0.5em;
    }
    .classical-badge {
        background-color: #718096;
        color: white;
        padding: 0.2em 0.6em;
        border-radius: 10px;
        font-size: 0.8em;
        margin-right: 0.5em;
    }
    .agent-message {
        background-color: #f0f7ff;
        border-left: 3px solid #29539B;
        padding: 1em;
        margin: 0.5em 0;
        border-radius: 0 10px 10px 0;
    }
    .user-message {
        background-color: #f2f2f2;
        border-right: 3px solid #718096;
        padding: 1em;
        margin: 0.5em 0;
        border-radius: 10px 0 0 10px;
        text-align: right;
    }
    .task-result {
        background-color: #f0f8f7;
        border: 1px solid #ddd;
        padding: 1em;
        margin: 0.5em 0;
        border-radius: 10px;
    }
    .center {
        display: flex;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for the agent and conversation history
if 'agent' not in st.session_state:
    # Default to quantum enabled but check OpenAI API key
    use_openai = os.environ.get("OPENAI_API_KEY") is not None
    st.session_state.agent = Q3AAgent(use_quantum=True, n_qubits=8, use_openai=use_openai)
    st.session_state.show_debug = False
    st.session_state.tasks = []
    st.session_state.messages = []
    st.session_state.current_tab = "chat"

# Sidebar
with st.sidebar:
    st.title("QUASAR Framework")
    st.image("https://upload.wikimedia.org/wikipedia/commons/9/96/Quantum_circuit_compilation_for_nisq_gqo.png", width=250)
    
    st.markdown("### Agent Settings")
    
    # Quantum settings
    use_quantum = st.checkbox("Use Quantum Acceleration", value=True, key="use_quantum")
    n_qubits = st.slider("Number of Qubits", min_value=4, max_value=29, value=8, key="n_qubits")
    
    # Update agent if settings changed
    if use_quantum != st.session_state.agent.use_quantum or n_qubits != st.session_state.agent.n_qubits:
        st.session_state.agent.use_quantum = use_quantum
        st.session_state.agent.n_qubits = n_qubits
    
    st.markdown("### Advanced Settings")
    show_debug = st.checkbox("Show Debug Information", value=st.session_state.show_debug)
    st.session_state.show_debug = show_debug
    
    st.markdown("---")
    st.markdown("### Navigation")
    tab_options = ["Chat", "Tasks", "Performance"]
    selected_tab = st.radio("Select Interface", tab_options)
    st.session_state.current_tab = selected_tab.lower()
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **QUASAR: Quantum AI Agent Framework**  
    Quantum-Accelerated Search and AI Reasoning
    
    Version: 1.0.0  
    © 2025 Quantum Labs
    """)

# Main header
st.markdown('<div class="main-header">Q3A: Quantum-Accelerated AI Agent</div>', unsafe_allow_html=True)

# Content based on selected tab
if st.session_state.current_tab == "chat":
    st.subheader("Agent Chat Interface")
    
    st.markdown("""
    Interact with the Q3A agent to perform tasks with quantum acceleration.
    The agent can handle various tasks including:
    
    - **Enhanced Search**: Grover's algorithm provides quadratic speedup
    - **Factorization**: Shor's algorithm can factor large numbers efficiently
    - **Optimization**: QAOA provides advantages for constraint optimization problems
    """)
    
    # Display conversation history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='agent-message'>{message['content']}</div>", unsafe_allow_html=True)
    
    # Chat input
    with st.form(key="chat_form"):
        user_input = st.text_area("Enter your task or question:", height=100)
        cols = st.columns([1, 1, 4])
        with cols[0]:
            submit_button = st.form_submit_button("Send")
        with cols[1]:
            clear_button = st.form_submit_button("Clear Chat")
    
    # Form processing
    if clear_button:
        st.session_state.messages = []
        st.rerun()
        
    if submit_button and user_input:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Create a placeholder for the agent's response
        with st.spinner("Processing with Q3A Agent..."):
            # Process the task
            task_result = asyncio.run(st.session_state.agent.process_task(user_input))
            st.session_state.tasks.append(task_result)
            
            # Generate a user-friendly response based on task type
            if task_result["task_type"] == "search":
                response = f"### Search Results for: {task_result['result']['query']}\n\n"
                response += f"{task_result['result']['summary']}\n\n"
                
                for i, result in enumerate(task_result['result']['results'][:3]):
                    response += f"**Result {i+1}**: {result['title']}\n"
                    response += f"{result['content']}\n"
                    response += f"*Relevance: {result['relevance']:.1f}% - {result['processing']}*\n\n"
                    
                # Add performance metrics
                if st.session_state.show_debug:
                    response += "\n\n---\n\n"
                    response += f"**Processing Method**: {'Quantum-Accelerated' if task_result['use_quantum'] else 'Classical'}\n"
                    response += f"**Quantum Time**: {task_result['result']['quantum_time']:.4f}s\n"
                    response += f"**Classical Time**: {task_result['result']['classical_time']:.4f}s\n"
                    if task_result['use_quantum']:
                        response += f"**Speedup**: {task_result['result']['speedup']:.2f}x\n"
                    
            elif task_result["task_type"] == "factorization":
                response = f"### Factorization of {task_result['result']['number']}\n\n"
                response += f"{task_result['result']['explanation']}\n\n"
                
                # Add performance metrics
                if st.session_state.show_debug:
                    response += "\n\n---\n\n"
                    response += f"**Processing Method**: {'Quantum-Accelerated' if task_result['use_quantum'] else 'Classical'}\n"
                    response += f"**Quantum Time**: {task_result['result']['quantum_time']:.4f}s\n"
                    response += f"**Classical Time**: {task_result['result']['classical_time']:.4f}s\n"
                    if task_result['use_quantum']:
                        response += f"**Speedup**: {task_result['result']['speedup']:.2f}x\n"
                    
            elif task_result["task_type"] == "optimization":
                response = f"### Optimization Results\n\n"
                response += f"{task_result['result']['explanation']}\n\n"
                
                response += "**Solution:**\n"
                for key, value in task_result['result']['solution'].items():
                    response += f"- {key}: {value}\n"
                    
                response += f"\n**Objective Value**: {task_result['result']['objective_value']}\n"
                
                # Add performance metrics
                if st.session_state.show_debug:
                    response += "\n\n---\n\n"
                    response += f"**Processing Method**: {'Quantum-Accelerated' if task_result['use_quantum'] else 'Classical'}\n"
                    response += f"**Quantum Time**: {task_result['result']['quantum_time']:.4f}s\n"
                    response += f"**Classical Time**: {task_result['result']['classical_time']:.4f}s\n"
                    if task_result['use_quantum']:
                        response += f"**Speedup**: {task_result['result']['speedup']:.2f}x\n"
                    
            else:  # General task
                response = task_result['result']['response']
                
                # Add performance metrics
                if st.session_state.show_debug:
                    response += "\n\n---\n\n"
                    response += f"**Processing Method**: {'Quantum-Accelerated' if task_result['use_quantum'] else 'Classical'}\n"
                    response += f"**Quantum Time**: {task_result['result']['quantum_time']:.4f}s\n"
                    response += f"**Classical Time**: {task_result['result']['classical_time']:.4f}s\n"
                    if task_result['use_quantum']:
                        response += f"**Speedup**: {task_result['result']['speedup']:.2f}x\n"
            
            # Add agent response to history
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun to update the UI
        st.rerun()

elif st.session_state.current_tab == "tasks":
    st.subheader("Tasks History & Management")
    
    # Simple task history display
    if not st.session_state.tasks:
        st.info("No tasks have been processed yet. Try asking the agent to perform a task in the Chat tab.")
    else:
        st.write(f"Total tasks processed: {len(st.session_state.tasks)}")
        
        # Group by task type
        task_types = {}
        for task in st.session_state.tasks:
            task_type = task["task_type"]
            if task_type not in task_types:
                task_types[task_type] = []
            task_types[task_type].append(task)
        
        # Display by type
        for task_type, tasks in task_types.items():
            st.subheader(f"{task_type.capitalize()} Tasks ({len(tasks)})")
            for i, task in enumerate(tasks):
                with st.expander(f"Task {i+1}: {task['task'][:50]}...", expanded=False):
                    st.write(f"**ID**: {task['task_id']}")
                    st.write(f"**Type**: {task['task_type']}")
                    st.write(f"**Time**: {task['execution_time']:.4f}s")
                    st.write(f"**Quantum**: {'Yes' if task['use_quantum'] else 'No'}")
                    
                    # Task result display
                    st.markdown("##### Result")
                    result = task["result"]
                    # Format differently based on task type
                    if task["task_type"] == "search":
                        st.write(f"**Query**: {result['query']}")
                        st.write(f"**Summary**: {result['summary']}")
                        for j, r in enumerate(result['results'][:3]):
                            st.write(f"**Result {j+1}**: {r['title']}")
                    elif task["task_type"] == "factorization":
                        st.write(f"**Number**: {result['number']}")
                        st.write(f"**Factors**: {', '.join(map(str, result['factors']))}")
                        st.write(f"**Prime Factors**: {', '.join(map(str, result['prime_factors']))}")
                    elif task["task_type"] == "optimization":
                        st.write(f"**Objective Value**: {result['objective_value']}")
                        st.write("**Solution**:")
                        for k, v in result['solution'].items():
                            st.write(f"- {k}: {v}")
                    else:
                        st.write(result['response'])
                            
                    # Performance metrics
                    st.markdown("##### Performance Metrics")
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Quantum Time", f"{task['result']['quantum_time']:.4f}s")
                    with cols[1]:
                        st.metric("Classical Time", f"{task['result']['classical_time']:.4f}s")
                    with cols[2]:
                        if task['use_quantum']:
                            speedup = task['result'].get('speedup', 0)
                            st.metric("Speedup", f"{speedup:.2f}x")
                        else:
                            st.metric("Speedup", "N/A")

elif st.session_state.current_tab == "performance":
    st.subheader("Quantum-Classical Performance Comparison")
    
    # Get agent metrics
    metrics = st.session_state.agent.get_metrics()
    
    # Overall metrics
    st.markdown("### Overall Performance")
    
    cols = st.columns(3)
    with cols[0]:
        st.metric("Tasks Completed", metrics["tasks_completed"])
    with cols[1]:
        st.metric("Quantum Tasks", metrics["quantum_accelerated_tasks"])
    with cols[2]:
        st.metric("Classical Tasks", metrics["classical_tasks"])
    
    # Speedup metrics if any quantum tasks completed
    if metrics["quantum_accelerated_tasks"] > 0:
        st.metric("Average Quantum Speedup", f"{metrics['average_speedup']:.2f}x")
        
        # Simulated quantum advantage chart
        st.markdown("### Quantum vs. Classical Performance")
        
        # Create sample data for demonstration
        task_sizes = list(range(10, 110, 10))
        classical_times = [0.1 * x for x in task_sizes]
        quantum_times = [0.1 * x**0.5 for x in task_sizes]  # Square root speedup
        
        # Create a chart comparing quantum vs. classical
        chart_data = {"Task Size": task_sizes}
        chart_data["Classical Time (s)"] = classical_times
        chart_data["Quantum Time (s)"] = quantum_times
        
        chart_df = pd.DataFrame(chart_data)
        st.line_chart(chart_df.set_index("Task Size"))
        
        st.markdown("""
        *Note: This chart illustrates the theoretical advantage of quantum computing 
        for certain classes of problems. Real-world performance may vary based on 
        specific problem characteristics and available quantum hardware.*
        """)
        
        # Application domains
        st.markdown("### Quantum Advantage by Application Domain")
        
        domains = {
            "Search": "Quadratic speedup (O(√N) vs O(N))",
            "Factorization": "Exponential speedup (Polynomial vs Exponential)",
            "Optimization": "Potential exponential speedup for certain problems",
            "Machine Learning": "Potential quadratic speedup for specific algorithms",
            "Simulation": "Exponential speedup for quantum system simulation"
        }
        
        for domain, advantage in domains.items():
            st.markdown(f"**{domain}**: {advantage}")
    else:
        st.info("Complete some tasks with quantum acceleration to see performance metrics.")
        
# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888888; font-size: 0.8em;">
    Q3A: Quantum-Accelerated AI Agent | QUASAR Framework v1.0.0<br>
    Combining quantum computing with AI for enhanced capabilities
</div>
""", unsafe_allow_html=True)

def main():
    """Main function, not explicitly needed with Streamlit"""
    pass

if __name__ == "__main__":
    main()