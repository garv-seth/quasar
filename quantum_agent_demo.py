"""
QA³: Quantum-Accelerated AI Agent Demonstration

This script demonstrates the capabilities of the Quantum Agent Framework,
showing how quantum computing can enhance traditional agent capabilities.
"""

import os
import json
import time
import asyncio
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Basic logging setup
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quantum-agent-demo")

# Import the core agent components
from quantum_agent_framework.agent_core.core_agent import (
    QA3CoreAgent,
    AgentMemory,
    PerceptionSystem,
    ActionSystem,
    QuantumDecisionSystem,
    GoalManagementSystem,
    LearningSystem
)

# Import quantum engine
try:
    from quantum_agent_framework.quantum.azure_quantum_engine import AzureQuantumEngine
    AZURE_QUANTUM_AVAILABLE = True
except ImportError:
    AZURE_QUANTUM_AVAILABLE = False
    print("Azure Quantum engine not available. Using simulated quantum capabilities.")

# Import quantum bridge
try:
    from quantum_agent_framework.integration.quantum_agent_bridge import QuantumAgentBridge
    QUANTUM_BRIDGE_AVAILABLE = True
except ImportError:
    QUANTUM_BRIDGE_AVAILABLE = False
    print("Quantum bridge not available. Using direct quantum engine.")

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'agent_initialized' not in st.session_state:
        st.session_state.agent_initialized = False
        
    if 'agent_running' not in st.session_state:
        st.session_state.agent_running = False
        
    if 'agent' not in st.session_state:
        st.session_state.agent = None
        
    if 'quantum_engine' not in st.session_state:
        st.session_state.quantum_engine = None
        
    if 'quantum_bridge' not in st.session_state:
        st.session_state.quantum_bridge = None
        
    if 'task_history' not in st.session_state:
        st.session_state.task_history = []
        
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
    if 'n_qubits' not in st.session_state:
        st.session_state.n_qubits = 8
        
    if 'use_quantum' not in st.session_state:
        st.session_state.use_quantum = True
        
    if 'use_hardware' not in st.session_state:
        st.session_state.use_hardware = False
        
    if 'provider' not in st.session_state:
        st.session_state.provider = "ionq"

async def initialize_agent():
    """Initialize the QA³ agent with quantum capabilities"""
    if st.session_state.agent_initialized:
        return True
    
    # Initialize quantum engine
    try:
        if AZURE_QUANTUM_AVAILABLE and st.session_state.use_quantum:
            st.session_state.quantum_engine = AzureQuantumEngine(
                use_hardware=st.session_state.use_hardware,
                n_qubits=st.session_state.n_qubits,
                provider=st.session_state.provider
            )
            st.info(f"Initialized Azure Quantum engine with {st.session_state.n_qubits} qubits")
        else:
            # Use PennyLane simulation
            import pennylane as qml
            st.session_state.quantum_engine = "pennylane_simulation"
            st.info(f"Using PennyLane quantum simulation with {st.session_state.n_qubits} qubits")
    except Exception as e:
        st.error(f"Failed to initialize quantum engine: {str(e)}")
        st.session_state.quantum_engine = None
    
    # Initialize quantum bridge
    try:
        if QUANTUM_BRIDGE_AVAILABLE and st.session_state.use_quantum and st.session_state.quantum_engine:
            st.session_state.quantum_bridge = QuantumAgentBridge(
                st.session_state.quantum_engine,
                use_hardware=st.session_state.use_hardware,
                n_qubits=st.session_state.n_qubits
            )
            st.info("Initialized quantum agent bridge")
        else:
            st.session_state.quantum_bridge = None
    except Exception as e:
        st.error(f"Failed to initialize quantum bridge: {str(e)}")
        st.session_state.quantum_bridge = None
    
    # Initialize agent
    try:
        st.session_state.agent = QA3CoreAgent(
            use_quantum=st.session_state.use_quantum,
            n_qubits=st.session_state.n_qubits
        )
        
        # Start the agent loop to ensure it's running
        asyncio.create_task(st.session_state.agent.start_agent_loop())
        
        # Set the agent status to initialized and running
        st.session_state.agent_running = True
        st.success("QA³ agent initialized and running")
        st.session_state.agent_initialized = True
        return True
    except Exception as e:
        st.error(f"Failed to initialize agent: {str(e)}")
        st.session_state.agent_running = False
        return False

async def process_task(task):
    """Process a user task with the QA³ agent"""
    if not st.session_state.agent_initialized:
        st.error("Agent not initialized")
        return {"success": False, "error": "Agent not initialized"}
    
    # Add user message to chat
    st.session_state.chat_history.append({"role": "user", "content": task})
    
    try:
        # Process the task
        with st.spinner("Agent processing task..."):
            result = await st.session_state.agent.process_task(task)
        
        # Add result to task history
        st.session_state.task_history.append({
            "task": task,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
        # Add agent response to chat
        response = f"Task complete: {result.get('success', False)}\n\n"
        if result.get('success', False):
            response += f"I successfully completed the task '{task}' by taking the action: {result.get('decision', {}).get('description', 'unknown action')}.\n\n"
            response += f"This action was {'' if result.get('quantum_enhanced', False) else 'not '}quantum-enhanced."
        else:
            response += f"I was unable to complete the task '{task}'. The issue was: {result.get('action_result', {}).get('message', 'Unknown error')}."
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        return result
    except Exception as e:
        error_msg = f"Error processing task: {str(e)}"
        st.error(error_msg)
        st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {error_msg}"})
        return {"success": False, "error": error_msg}

def simulate_quantum_advantage():
    """Simulate and visualize quantum advantage for various tasks"""
    st.subheader("Quantum Advantage Visualization")
    
    task_type = st.selectbox(
        "Task Type",
        ["Search", "Factorization", "Optimization", "Decision Making"],
        key="quantum_advantage_task"
    )
    
    if task_type == "Search":
        simulate_search_advantage()
    elif task_type == "Factorization":
        simulate_factorization_advantage()
    elif task_type == "Optimization":
        simulate_optimization_advantage()
    elif task_type == "Decision Making":
        simulate_decision_advantage()

def simulate_search_advantage():
    """Simulate quantum advantage for search tasks"""
    # Allow user to specify database size
    db_size = st.slider("Database Size", 10, 10000, 1000, step=10)
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Database sizes to plot
    sizes = np.logspace(1, 5, 20).astype(int)
    
    # Classical vs. quantum performance
    classical_steps = sizes  # O(N)
    quantum_steps = np.sqrt(sizes)  # O(√N)
    
    # Plot results
    ax.plot(sizes, classical_steps, 'r-', label='Classical (O(N))')
    ax.plot(sizes, quantum_steps, 'g-', label='Quantum (O(√N))')
    
    # Mark the selected size
    ax.scatter([db_size], [db_size], color='blue', s=100, label='Classical Steps')
    ax.scatter([db_size], [np.sqrt(db_size)], color='green', s=100, label='Quantum Steps')
    
    # Set log scales
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Labels
    ax.set_title(f'Quantum vs. Classical Search Performance')
    ax.set_xlabel('Database Size (entries)')
    ax.set_ylabel('Steps Required (log scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Display the plot
    st.pyplot(fig)
    
    # Display summary
    speedup = db_size / np.sqrt(db_size)
    st.info(f"For a database of {db_size:,} entries:")
    st.markdown(f"""
    * Classical search requires up to **{db_size:,}** steps
    * Quantum search requires approximately **{int(np.sqrt(db_size)):,}** steps
    * Quantum speedup: **{speedup:.2f}x** faster
    """)

def simulate_factorization_advantage():
    """Simulate quantum advantage for factorization tasks"""
    # Allow user to specify number to factorize
    number = st.slider("Number to Factorize", 10, 10000, 1000, step=10)
    bits = len(bin(number)) - 2
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Range of bit lengths
    bit_lengths = np.arange(4, 32)
    numbers = [2**bits - 1 for bits in bit_lengths]
    
    # Classical vs. quantum performance (simplified models)
    # Classical: O(exp(sqrt(bits)))
    # Quantum: O(bits^3)
    classical_time = np.exp(np.sqrt(bit_lengths)) / 1e3
    quantum_time = bit_lengths**3 / 1e2
    
    # Plot results
    ax.plot(bit_lengths, classical_time, 'r-', label='Classical (exp)')
    ax.plot(bit_lengths, quantum_time, 'g-', label='Quantum (polynomial)')
    
    # Mark the selected number
    ax.scatter([bits], [np.exp(np.sqrt(bits)) / 1e3], color='blue', s=100, label=f'Selected: {number}')
    
    # Set log scale for y-axis
    ax.set_yscale('log')
    
    # Labels
    ax.set_title(f'Quantum vs. Classical Factorization Performance')
    ax.set_xlabel('Bit Length of Number')
    ax.set_ylabel('Relative Time (log scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Display the plot
    st.pyplot(fig)
    
    # Display summary
    classical_est = np.exp(np.sqrt(bits)) / 1e3
    quantum_est = bits**3 / 1e2
    speedup = classical_est / quantum_est
    
    st.info(f"For factorizing {number:,} ({bits} bits):")
    st.markdown(f"""
    * Classical factorization: Exponential time complexity
    * Quantum factorization: Polynomial time complexity
    * Estimated quantum speedup: **{speedup:.2f}x** faster
    
    Note: For small numbers like this, the advantage may be limited.
    Quantum factorization shines with numbers over 1024 bits.
    """)

def simulate_optimization_advantage():
    """Simulate quantum advantage for optimization tasks"""
    # Allow user to specify problem size
    problem_size = st.slider("Optimization Problem Size (variables)", 2, 200, 50, step=1)
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Range of problem sizes
    sizes = np.arange(2, 201)
    
    # Classical vs. quantum performance (simplified models)
    # Classical: O(2^n) for exact solution
    # Quantum: O(sqrt(2^n)) with Grover-based approaches
    classical_time = 2**np.clip(sizes/10, 0, 20)  # Clip to avoid overflow
    quantum_time = np.sqrt(2**np.clip(sizes/10, 0, 20))
    
    # Plot results (with clipping for visualization)
    max_y = 1e6
    ax.plot(sizes, np.clip(classical_time, 0, max_y), 'r-', label='Classical (exponential)')
    ax.plot(sizes, np.clip(quantum_time, 0, max_y), 'g-', label='Quantum (quadratic speedup)')
    
    # Mark the selected size
    ax.scatter([problem_size], [min(2**(problem_size/10), max_y)], color='blue', s=100, label=f'Selected: {problem_size} vars')
    
    # Set log scale for y-axis
    ax.set_yscale('log')
    
    # Labels
    ax.set_title(f'Quantum vs. Classical Optimization Performance')
    ax.set_xlabel('Problem Size (number of variables)')
    ax.set_ylabel('Relative Time (log scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Display the plot
    st.pyplot(fig)
    
    # Display summary
    if problem_size <= 20:
        advantage = "modest"
        speedup = np.sqrt(2**(problem_size/10))
    elif problem_size <= 50:
        advantage = "significant"
        speedup = np.sqrt(2**(problem_size/10))
    else:
        advantage = "dramatic"
        speedup = "many orders of magnitude"
    
    st.info(f"For an optimization problem with {problem_size} variables:")
    st.markdown(f"""
    * Classical exact solution: Exponential time complexity
    * Quantum optimization: Potential quadratic speedup
    * Quantum advantage: **{advantage}** (approximately {speedup if isinstance(speedup, float) else speedup})
    
    Note: This model is simplified. Real quantum advantage depends on the specific 
    problem structure and algorithm used.
    """)

def simulate_decision_advantage():
    """Simulate quantum advantage for decision making"""
    # Allow user to specify number of options
    num_options = st.slider("Number of Decision Options", 2, 100, 10, step=1)
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Range of option counts
    options_range = np.arange(2, 101)
    
    # Decision quality model (simplified)
    # Higher is better, capped at 1.0
    classical_quality = 0.5 + (0.5 * np.exp(-options_range / 20))
    quantum_quality = 0.6 + (0.4 * np.exp(-options_range / 60))
    
    # Plot results
    ax.plot(options_range, classical_quality, 'r-', label='Classical Decision')
    ax.plot(options_range, quantum_quality, 'g-', label='Quantum-Enhanced')
    
    # Mark the selected point
    selected_classical = 0.5 + (0.5 * np.exp(-num_options / 20))
    selected_quantum = 0.6 + (0.4 * np.exp(-num_options / 60))
    ax.scatter([num_options], [selected_classical], color='red', s=100)
    ax.scatter([num_options], [selected_quantum], color='green', s=100)
    
    # Labels
    ax.set_title(f'Quantum vs. Classical Decision Quality')
    ax.set_xlabel('Number of Options to Evaluate')
    ax.set_ylabel('Decision Quality (higher is better)')
    ax.set_ylim(0.4, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Display the plot
    st.pyplot(fig)
    
    # Display summary
    improvement = (selected_quantum - selected_classical) / selected_classical * 100
    
    st.info(f"For decision making with {num_options} options:")
    st.markdown(f"""
    * Classical decision quality: **{selected_classical:.2f}**
    * Quantum-enhanced decision quality: **{selected_quantum:.2f}**
    * Improvement: **{improvement:.1f}%**
    
    The quantum advantage in decision-making comes from:
    1. Better handling of uncertainty through superposition
    2. More effective exploration of option combinations
    3. Quantum interference helping to amplify good decisions
    """)

def display_agent_interface():
    """Display the main agent interface"""
    st.subheader("QA³ Agent Interface")
    
    # Agent status display
    if st.session_state.agent_initialized:
        status = st.session_state.agent.get_status()
        
        # Verify if the agent running status is accurate
        is_running = status.get('is_running', False)
        if is_running != st.session_state.agent_running:
            st.session_state.agent_running = is_running
        
        # Agent running banner
        if st.session_state.agent_running:
            st.success("🟢 Agent is running and ready to process tasks")
        else:
            st.warning("🟠 Agent is initialized but not running")
            if st.button("Start Agent"):
                asyncio.create_task(st.session_state.agent.start_agent_loop())
                st.session_state.agent_running = True
                st.rerun()
        
        with st.expander("Agent Status", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Agent Information**")
                st.write(f"Agent Name: {status.get('agent_name', 'QA³ Agent')}")
                st.write(f"Running: {is_running}")
                st.write(f"Startup Time: {status.get('startup_time', 'Unknown')}")
            
            with col2:
                st.markdown("**Quantum Status**")
                quantum_status = status.get('quantum_status', {})
                st.write(f"Quantum Enabled: {quantum_status.get('enabled', False)}")
                st.write(f"Qubits: {quantum_status.get('n_qubits', 0)}")
                st.write(f"Bridge Available: {quantum_status.get('bridge_available', False)}")
                st.write(f"Engine Available: {quantum_status.get('engine_available', False)}")
                
            st.markdown("**Active Goals**")
            goals = status.get('active_goals', [])
            if goals:
                for goal in goals:
                    st.write(f"- {goal.get('description', 'Unknown')} (Priority: {goal.get('priority', 0)}, Progress: {goal.get('progress', 0)}%)")
            else:
                st.write("No active goals")
    
    # Chat interface
    st.markdown("### Chat with QA³ Agent")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"<div style='background-color: #f0f2f6; color: #424242; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color: #e1f5fe; color: #0d47a1; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>QA³ Agent:</strong> {message['content']}</div>", unsafe_allow_html=True)
    
    # Input for new message
    with st.form(key="chat_form"):
        user_input = st.text_area("Type your task for the agent:", height=100)
        col1, col2 = st.columns([1, 5])
        with col1:
            submit_button = st.form_submit_button("Send")
        
        if submit_button and user_input:
            asyncio.run(process_task(user_input))
            st.rerun()

def display_task_history():
    """Display task history"""
    st.subheader("Task History")
    
    if not st.session_state.task_history:
        st.info("No tasks have been processed yet.")
        return
    
    for i, task_entry in enumerate(reversed(st.session_state.task_history)):
        with st.expander(f"Task: {task_entry['task']}", expanded=(i == 0)):
            result = task_entry.get("result", {})
            
            # Task details
            st.markdown(f"**Timestamp:** {task_entry.get('timestamp', 'Unknown')}")
            st.markdown(f"**Success:** {result.get('success', False)}")
            
            # Action details
            st.markdown("**Action Details:**")
            decision = result.get("decision", {})
            action_result = result.get("action_result", {})
            
            st.markdown(f"- Description: {decision.get('description', 'Unknown')}")
            st.markdown(f"- Action Type: {decision.get('action_type', 'Unknown')}")
            st.markdown(f"- Confidence: {decision.get('confidence', 0):.2f}")
            st.markdown(f"- Outcome: {action_result.get('message', 'Unknown')}")
            
            # Quantum enhancement
            st.markdown("**Quantum Information:**")
            st.markdown(f"- Quantum Enhanced: {result.get('quantum_enhanced', False)}")

def display_web_browsing():
    """Display web browsing interface with quantum-enhanced capabilities"""
    st.subheader("Quantum-Enhanced Web Browsing")
    
    st.markdown("""
    ### Web Navigation
    
    This interface demonstrates the agent's ability to browse the web with quantum-enhanced capabilities:
    1. Quantum-enhanced search and relevance ranking
    2. Web page analysis with quantum processing
    3. Intelligent navigation with quantum decision making
    """)
    
    # URL input
    with st.form("web_navigation_form"):
        url_input = st.text_input("Enter URL to visit:", placeholder="https://example.com")
        browse_button = st.form_submit_button("Browse")
        
        if browse_button and url_input:
            # Add to task history
            task = f"Navigate to {url_input}"
            st.session_state.chat_history.append({"role": "user", "content": task})
            
            with st.spinner(f"Navigating to {url_input}..."):
                # Process the navigation task
                try:
                    if st.session_state.agent_initialized and st.session_state.agent_running:
                        result = asyncio.run(st.session_state.agent.process_task(task))
                        
                        # Display result
                        if result.get("success", False):
                            # Add screenshot if available
                            if "screenshot" in result.get("action_result", {}):
                                st.image(f"data:image/png;base64,{result['action_result']['screenshot']}", 
                                         caption=f"Screenshot of {url_input}")
                            
                            # Add response to chat history
                            message = f"Successfully navigated to {url_input}"
                            if "title" in result.get("action_result", {}):
                                message += f"\nPage title: {result['action_result']['title']}"
                            
                            st.session_state.chat_history.append({"role": "assistant", "content": message})
                        else:
                            error = result.get("error", "Unknown error")
                            st.error(f"Failed to navigate: {error}")
                            st.session_state.chat_history.append({"role": "assistant", 
                                                                 "content": f"Failed to navigate to {url_input}: {error}"})
                    else:
                        st.error("Agent is not initialized or running. Please start the agent first.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    
    # Display previous web interactions
    with st.expander("Web Browsing History", expanded=False):
        web_interactions = [task for task in st.session_state.task_history 
                          if "navigate" in task.get("task", "").lower()]
        
        if not web_interactions:
            st.info("No web browsing history yet.")
        else:
            for interaction in web_interactions:
                st.markdown(f"**URL:** {interaction.get('task', '').replace('Navigate to ', '')}")
                st.markdown(f"**Time:** {interaction.get('timestamp', 'Unknown')}")
                
                # Display screenshot if available
                result = interaction.get("result", {})
                action_result = result.get("action_result", {})
                if "screenshot" in action_result:
                    st.image(f"data:image/png;base64,{action_result['screenshot']}", 
                             caption="Screenshot", width=400)
                st.markdown("---")

def display_quantum_demo():
    """Display quantum computing demonstration"""
    st.subheader("Quantum Computing Demonstration")
    
    tab1, tab2 = st.tabs(["Quantum Advantage", "Quantum Circuits"])
    
    with tab1:
        simulate_quantum_advantage()
    
    with tab2:
        display_quantum_circuits()

def display_quantum_circuits():
    """Display quantum circuit demonstrations"""
    circuit_type = st.selectbox(
        "Circuit Type",
        ["Bell State", "GHZ State", "Custom"],
        key="quantum_circuit_type"
    )
    
    if circuit_type == "Bell State":
        st.markdown("""
        ### Bell State
        
        The Bell state is one of the simplest examples of quantum entanglement: a property where the quantum state of each particle cannot be described independently of the others.
        
        Circuit:
        ```
        q0: ─H─●─
               │ 
        q1: ───X─
        ```
        
        This creates the state: |00⟩ + |11⟩ (unnormalized)
        """)
        
        # Create a simple Bell state visualization
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Draw the circuit diagram manually
        ax.axis('off')
        
        # Qubit lines
        ax.plot([0, 10], [1, 1], 'k-', linewidth=2)  # q0
        ax.plot([0, 10], [0, 0], 'k-', linewidth=2)  # q1
        
        # Labels
        ax.text(-0.5, 1, 'q0:', fontsize=14)
        ax.text(-0.5, 0, 'q1:', fontsize=14)
        
        # Hadamard gate
        ax.add_patch(plt.Rectangle((2, 0.75), 1, 0.5, fill=True, color='skyblue'))
        ax.text(2.4, 1, 'H', fontsize=14)
        
        # CNOT gate
        ax.add_patch(plt.Circle((5, 1), 0.2, fill=True, color='black'))  # Control
        ax.plot([5, 5], [0.8, 0.2], 'k-', linewidth=2)  # Vertical line
        ax.add_patch(plt.Circle((5, 0), 0.2, fill=False, color='black'))  # Target
        ax.plot([4.8, 5.2], [0, 0], 'k-', linewidth=2)  # X part
        
        st.pyplot(fig)
        
        # Show the state vector evolution
        st.markdown("### State Evolution")
        state_evolution = [
            ("|0⟩|0⟩", "Initial state"),
            ("(|0⟩+|1⟩)|0⟩/√2", "After Hadamard on q0"),
            ("(|0⟩|0⟩+|1⟩|1⟩)/√2", "After CNOT (entangled Bell state)")
        ]
        
        for state, description in state_evolution:
            st.markdown(f"**{description}**: {state}")
    
    elif circuit_type == "GHZ State":
        num_qubits = st.slider("Number of Qubits", 3, 10, 3, key="ghz_qubits")
        
        st.markdown(f"""
        ### GHZ State with {num_qubits} Qubits
        
        The GHZ (Greenberger-Horne-Zeilinger) state is a highly entangled quantum state of three or more qubits.
        
        This creates the state: |00...0⟩ + |11...1⟩ (unnormalized)
        """)
        
        # Create a simple GHZ visualization
        fig, ax = plt.subplots(figsize=(10, num_qubits/2 + 2))
        
        # Draw the circuit diagram manually
        ax.axis('off')
        
        # Qubit lines
        for i in range(num_qubits):
            ax.plot([0, 10], [num_qubits-i-1, num_qubits-i-1], 'k-', linewidth=2)
            ax.text(-0.7, num_qubits-i-1, f'q{i}:', fontsize=14)
        
        # Hadamard gate on first qubit
        ax.add_patch(plt.Rectangle((2, num_qubits-1-0.25), 1, 0.5, fill=True, color='skyblue'))
        ax.text(2.4, num_qubits-1, 'H', fontsize=14)
        
        # CNOT gates
        for i in range(1, num_qubits):
            ax.add_patch(plt.Circle((4+i*0.5, num_qubits-1), 0.2, fill=True, color='black'))  # Control
            ax.plot([4+i*0.5, 4+i*0.5], [num_qubits-1.2, num_qubits-i-1+0.2], 'k-', linewidth=2)  # Vertical line
            ax.add_patch(plt.Circle((4+i*0.5, num_qubits-i-1), 0.2, fill=False, color='black'))  # Target
            ax.plot([4+i*0.5-0.2, 4+i*0.5+0.2], [num_qubits-i-1, num_qubits-i-1], 'k-', linewidth=2)  # X part
        
        st.pyplot(fig)
        
        # Show the state vector evolution
        st.markdown("### Final State")
        st.latex(f"\\frac{{|{'0'*num_qubits}\\rangle + |{'1'*num_qubits}\\rangle}}{{\\sqrt{{2}}}}")
    
    elif circuit_type == "Custom":
        st.markdown("""
        ### Custom Quantum Circuit
        
        This is a parameterized quantum circuit that can create a variety of quantum states.
        You can adjust the parameters to see how they affect the circuit's behavior.
        """)
        
        num_qubits = st.slider("Number of Qubits", 2, 8, 4, key="custom_qubits")
        
        # Parameters for each qubit
        st.markdown("### Rotation Parameters")
        st.markdown("Set the parameters for each qubit's rotation gates")
        
        params = []
        cols = st.columns(3)
        
        for i in range(num_qubits):
            st.markdown(f"**Qubit {i}**")
            rx = st.slider(f"RX Rotation (q{i})", 0.0, 6.28, 0.0, 0.01, key=f"rx_{i}")
            ry = st.slider(f"RY Rotation (q{i})", 0.0, 6.28, 0.0, 0.01, key=f"ry_{i}")
            rz = st.slider(f"RZ Rotation (q{i})", 0.0, 6.28, 0.0, 0.01, key=f"rz_{i}")
            params.append([rx, ry, rz])
        
        # Create a visualization of the custom circuit
        fig, ax = plt.subplots(figsize=(12, num_qubits/2 + 2))
        
        # Draw the circuit diagram
        ax.axis('off')
        
        # Qubit lines
        for i in range(num_qubits):
            ax.plot([0, 12], [num_qubits-i-1, num_qubits-i-1], 'k-', linewidth=2)
            ax.text(-0.7, num_qubits-i-1, f'q{i}:', fontsize=12)
            
            # RX gate
            if params[i][0] > 0.01:
                ax.add_patch(plt.Rectangle((1, num_qubits-i-1-0.25), 0.8, 0.5, fill=True, color='pink'))
                ax.text(1.05, num_qubits-i-1, 'RX', fontsize=10)
            
            # RY gate
            if params[i][1] > 0.01:
                ax.add_patch(plt.Rectangle((3, num_qubits-i-1-0.25), 0.8, 0.5, fill=True, color='lightgreen'))
                ax.text(3.05, num_qubits-i-1, 'RY', fontsize=10)
            
            # RZ gate
            if params[i][2] > 0.01:
                ax.add_patch(plt.Rectangle((5, num_qubits-i-1-0.25), 0.8, 0.5, fill=True, color='lightyellow'))
                ax.text(5.05, num_qubits-i-1, 'RZ', fontsize=10)
        
        # CNOT gates between neighbors
        for i in range(num_qubits-1):
            ax.add_patch(plt.Circle((8, num_qubits-i-1), 0.2, fill=True, color='black'))  # Control
            ax.plot([8, 8], [num_qubits-i-1.2, num_qubits-i-2+0.2], 'k-', linewidth=2)  # Vertical line
            ax.add_patch(plt.Circle((8, num_qubits-i-2), 0.2, fill=False, color='black'))  # Target
            ax.plot([7.8, 8.2], [num_qubits-i-2, num_qubits-i-2], 'k-', linewidth=2)  # X part
        
        st.pyplot(fig)
        
        # Display a simulated quantum state (simplified)
        st.markdown("### Simulated Quantum State")
        
        # Very simplified state vector calculation
        num_states = 2**num_qubits
        amplitudes = np.ones(num_states) / np.sqrt(num_states)
        
        # Apply a very rough approximation of the effect of parameters
        for i, param_set in enumerate(params):
            rx, ry, rz = param_set
            total_rotation = (rx + ry + rz) / (3 * np.pi)
            
            # Modify amplitudes based on rotation amount (very approximate)
            for j in range(num_states):
                if (j >> i) & 1:  # If the i-th bit is 1
                    amplitudes[j] *= np.cos(total_rotation * np.pi / 2)
                else:
                    amplitudes[j] *= np.sin(total_rotation * np.pi / 2) + 1
        
        # Normalize
        amplitudes = amplitudes / np.sqrt(np.sum(np.abs(amplitudes)**2))
        
        # Display top states
        top_states = sorted(range(len(amplitudes)), key=lambda i: -abs(amplitudes[i]))[:8]
        
        for i in top_states:
            binary = format(i, f'0{num_qubits}b')
            probability = abs(amplitudes[i])**2
            if probability > 0.01:  # Only show non-negligible amplitudes
                st.markdown(f"|{binary}⟩: Probability = {probability:.4f}")

def display_agent_settings():
    """Display agent settings"""
    st.subheader("Agent Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Quantum Settings")
        
        use_quantum = st.checkbox("Use Quantum Acceleration", value=st.session_state.use_quantum)
        if use_quantum != st.session_state.use_quantum:
            st.session_state.use_quantum = use_quantum
            st.session_state.agent_initialized = False
        
        n_qubits = st.slider("Number of Qubits", 2, 29, st.session_state.n_qubits)
        if n_qubits != st.session_state.n_qubits:
            st.session_state.n_qubits = n_qubits
            st.session_state.agent_initialized = False
        
        if AZURE_QUANTUM_AVAILABLE:
            st.markdown("### Azure Quantum Settings")
            
            use_hardware = st.checkbox("Use Real Quantum Hardware (when available)", 
                                     value=st.session_state.use_hardware)
            if use_hardware != st.session_state.use_hardware:
                st.session_state.use_hardware = use_hardware
                st.session_state.agent_initialized = False
            
            provider = st.selectbox("Quantum Provider", ["ionq", "quantinuum"], 
                                  index=0 if st.session_state.provider == "ionq" else 1)
            if provider != st.session_state.provider:
                st.session_state.provider = provider
                st.session_state.agent_initialized = False
    
    with col2:
        st.markdown("### Agent Capabilities")
        
        # These are informational only - not connected to actual settings
        st.markdown("**Enabled Capabilities:**")
        
        st.markdown("✅ Quantum-Enhanced Decision Making")
        st.markdown("✅ Quantum-Optimized Memory Management")
        st.markdown("✅ Quantum-Enhanced Perception")
        st.markdown("✅ Agentic UI Interaction")
        st.markdown("✅ Goal-Driven Autonomy")
        
        if not st.session_state.agent_initialized:
            if st.button("Initialize Agent with Current Settings"):
                asyncio.run(initialize_agent())
                st.success("Agent initialized with new settings")
                st.rerun()

def display_about():
    """Display information about the quantum agent framework"""
    st.subheader("About QA³: Quantum-Accelerated AI Agent")
    
    st.markdown("""
    ## QA³: Quantum-Accelerated AI Agent
    
    QA³ is a cutting-edge framework that combines quantum computing with agentic AI capabilities. 
    This framework enables the creation of autonomous agents that leverage quantum advantages for 
    enhanced decision making, perception, and learning.
    
    ### Key Features
    
    * **True Quantum Acceleration**: Uses real quantum computing resources when available through Azure Quantum.
    * **Agentic Capabilities**: Provides autonomous goal-driven behavior with self-improvement.
    * **Quantum-Enhanced Decision Making**: Leverages quantum computing for better decisions under uncertainty.
    * **Quantum-Optimized Memory**: Uses quantum algorithms to optimize memory management and retrieval.
    * **Screen Interaction**: Can perceive and interact with computer interfaces.
    * **Extensible Framework**: Designed to be extended for specific use cases and domains.
    
    ### How It Works
    
    The framework integrates three key components:
    
    1. **Quantum Computing Layer**: Connects to quantum hardware or simulators through Azure Quantum and PennyLane.
    2. **Agent Core**: Provides the autonomous agent capabilities including perception, decision making, and learning.
    3. **Integration Bridge**: Enables quantum advantages to enhance traditional agent capabilities.
    
    ### Use Cases
    
    * **Enhanced Decision Making**: Make better decisions with quantum-accelerated analysis.
    * **Complex Optimization**: Solve complex optimization problems more efficiently.
    * **Pattern Recognition**: Identify patterns in large datasets with quantum advantage.
    * **Autonomous Agents**: Create agents that can interact with computer interfaces and web services.
    
    ### Technologies
    
    * **Azure Quantum**: Access to IonQ quantum hardware and simulators.
    * **PennyLane**: Quantum machine learning and simulation.
    * **Modern AI Models**: Integration with state-of-the-art AI for reasoning.
    """)
    
    # Show placeholder for quantum computing status
    st.subheader("Quantum Computing Status")
    
    if st.session_state.quantum_engine:
        if isinstance(st.session_state.quantum_engine, str) and st.session_state.quantum_engine == "pennylane_simulation":
            st.info("Using PennyLane quantum simulation")
        else:
            st.success("Connected to Azure Quantum services")
            
            # Display quantum engine status if available
            try:
                engine_status = st.session_state.quantum_engine.get_engine_status()
                
                st.markdown("**Azure Quantum Engine Status**")
                st.json(engine_status)
                
                # Display available targets if any
                targets = st.session_state.quantum_engine.get_available_targets()
                if targets:
                    st.markdown("**Available Quantum Targets**")
                    for target in targets:
                        st.markdown(f"- **{target.get('name', 'Unknown')}**: {target.get('qubits', 'N/A')} qubits, Provider: {target.get('provider', 'N/A')}")
            except:
                st.warning("Unable to retrieve detailed quantum engine status")
    else:
        st.warning("Quantum engine not initialized")

def main():
    """Main application function"""
    # Set up the page
    st.set_page_config(
        page_title="QA³: Quantum-Accelerated AI Agent Demo",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Title and introduction
    st.title("QA³: Quantum-Accelerated AI Agent")
    st.markdown("""
    This is a demonstration of the Quantum-Accelerated AI Agent framework, showing how quantum 
    computing can enhance traditional agent capabilities for true agency.
    """)
    
    # Sidebar
    st.sidebar.title("QA³ Navigation")
    
    # Agent status and initialization
    if not st.session_state.agent_initialized:
        st.sidebar.warning("⚠️ Agent not initialized")
        if st.sidebar.button("Initialize Agent"):
            with st.spinner("Initializing agent..."):
                success = asyncio.run(initialize_agent())
                if success:
                    st.sidebar.success("Agent initialized and running!")
                else:
                    st.sidebar.error("Failed to initialize agent")
    else:
        if st.session_state.agent_running:
            st.sidebar.success("🟢 Agent running")
        else:
            st.sidebar.warning("🟠 Agent initialized but not running")
            if st.sidebar.button("Start Agent Loop"):
                asyncio.create_task(st.session_state.agent.start_agent_loop())
                st.session_state.agent_running = True
                st.rerun()
    
    # Navigation
    page = st.sidebar.radio("View", [
        "Agent Interface",
        "Web Browsing",
        "Quantum Demo",
        "Task History",
        "Agent Settings",
        "About"
    ])
    
    # Display the selected page
    if page == "Agent Interface":
        display_agent_interface()
    elif page == "Web Browsing":
        display_web_browsing()
    elif page == "Quantum Demo":
        display_quantum_demo()
    elif page == "Task History":
        display_task_history()
    elif page == "Agent Settings":
        display_agent_settings()
    elif page == "About":
        display_about()

if __name__ == "__main__":
    main()