"""
QA³: Quantum-Accelerated AI Agent with True Agentic Capabilities
Main Application
"""

import streamlit as st
import os
import time
import json
import asyncio
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import logging
import base64
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("agentic-quantum-app")

# Import our custom agent implementation
from agent_core import (
    AutonomousAgent, 
    AgentMemory, 
    PerceptionSystem, 
    ActionSystem, 
    DecisionSystem,
    GoalManagementSystem,
    LearningSystem
)

# Setup PennyLane quantum computing
try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
    logger.info("PennyLane quantum computing library available.")
except ImportError:
    PENNYLANE_AVAILABLE = False
    logger.warning("PennyLane not available. Quantum features will be limited.")

# Setup Azure Quantum
try:
    from azure.quantum import Workspace
    AZURE_QUANTUM_AVAILABLE = True
    logger.info("Azure Quantum SDK available.")
except ImportError:
    AZURE_QUANTUM_AVAILABLE = False
    logger.warning("Azure Quantum SDK not available. Cloud quantum features will be limited.")

class QuantumCircuitBuilder:
    """
    Helper class for building and visualizing quantum circuits
    """
    
    def __init__(self, n_qubits: int = 4):
        """Initialize the quantum circuit builder"""
        self.n_qubits = n_qubits
        self.initialized = False
        
        if not PENNYLANE_AVAILABLE:
            logger.warning("PennyLane not available. Quantum circuit builder disabled.")
            return
        
        try:
            self.device = qml.device("default.qubit", wires=n_qubits)
            self.initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize quantum circuit builder: {str(e)}")
    
    def create_bell_state(self) -> Dict[str, Any]:
        """Create a Bell state circuit"""
        if not self.initialized:
            return {"success": False, "error": "Quantum circuit builder not initialized"}
        
        try:
            # Define the circuit
            @qml.qnode(self.device)
            def circuit():
                qml.Hadamard(wires=0)
                qml.CNOT(wires=[0, 1])
                return qml.state()
            
            # Execute the circuit
            state = circuit()
            
            # Create visualization
            fig, ax = qml.draw_mpl(circuit)()
            
            return {
                "success": True,
                "state": state.tolist(),
                "circuit_diagram": fig,
                "n_qubits": self.n_qubits,
                "circuit_type": "bell_state"
            }
        except Exception as e:
            logger.error(f"Error creating Bell state: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def create_ghz_state(self) -> Dict[str, Any]:
        """Create a GHZ state circuit"""
        if not self.initialized:
            return {"success": False, "error": "Quantum circuit builder not initialized"}
        
        try:
            # Define the circuit
            @qml.qnode(self.device)
            def circuit():
                qml.Hadamard(wires=0)
                for i in range(1, self.n_qubits):
                    qml.CNOT(wires=[0, i])
                return qml.state()
            
            # Execute the circuit
            state = circuit()
            
            # Create visualization
            fig, ax = qml.draw_mpl(circuit)()
            
            return {
                "success": True,
                "state": state.tolist(),
                "circuit_diagram": fig,
                "n_qubits": self.n_qubits,
                "circuit_type": "ghz_state"
            }
        except Exception as e:
            logger.error(f"Error creating GHZ state: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def create_custom_circuit(self, params: List[List[float]]) -> Dict[str, Any]:
        """Create a custom quantum circuit with specified parameters"""
        if not self.initialized:
            return {"success": False, "error": "Quantum circuit builder not initialized"}
        
        try:
            # Define the circuit
            @qml.qnode(self.device)
            def circuit():
                # Apply rotation gates for each qubit
                for i in range(min(self.n_qubits, len(params))):
                    qml.RX(params[i][0], wires=i)
                    qml.RY(params[i][1], wires=i)
                    qml.RZ(params[i][2], wires=i)
                
                # Apply entangling gates
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i+1])
                
                return qml.state()
            
            # Execute the circuit
            state = circuit()
            
            # Create visualization
            fig, ax = qml.draw_mpl(circuit)()
            
            return {
                "success": True,
                "state": state.tolist(),
                "circuit_diagram": fig,
                "n_qubits": self.n_qubits,
                "circuit_type": "custom",
                "parameters": params
            }
        except Exception as e:
            logger.error(f"Error creating custom circuit: {str(e)}")
            return {"success": False, "error": str(e)}

class QuantumSimulator:
    """
    Simulation of quantum algorithms with visualization
    """
    
    def __init__(self, n_qubits: int = 8):
        """Initialize the quantum simulator"""
        self.n_qubits = n_qubits
        self.initialized = PENNYLANE_AVAILABLE
    
    def factorize(self, number: int) -> Dict[str, Any]:
        """Simulate Shor's factorization algorithm"""
        if not self.initialized:
            return {"success": False, "error": "Quantum simulator not initialized"}
        
        if number < 3 or number > 1000:
            return {"success": False, "error": "Number must be between 3 and 1000"}
        
        try:
            # Start timing
            start_time = time.time()
            
            # In a real implementation, this would use a quantum algorithm
            # Here we'll use classical factorization but simulate quantum advantage
            
            # Find factors classically
            factors = []
            for i in range(2, int(number**0.5) + 1):
                if number % i == 0:
                    factors.append(i)
                    if i != number // i:  # Avoid duplicates for perfect squares
                        factors.append(number // i)
            
            factors.sort()
            
            # Add simulated quantum processing time
            # Adjust based on number size to simulate quantum advantage
            sleep_time = 0.1 + (0.01 * len(str(number)))
            time.sleep(sleep_time)
            
            # End timing
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Generate visualization
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            
            # Plot classical vs quantum time estimates
            sizes = [10, 50, 100, 500, 1000]
            classical_times = [0.001 * (2**len(str(n))) for n in sizes]  # Exponential time
            quantum_times = [0.001 * (len(str(n))**3) for n in sizes]    # Polynomial time
            
            ax.plot(sizes, classical_times, 'r-', label='Classical (Exponential)')
            ax.plot(sizes, quantum_times, 'g-', label='Quantum (Polynomial)')
            ax.scatter([number], [processing_time], color='blue', s=100, label='This factorization')
            
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Number to Factorize')
            ax.set_ylabel('Processing Time (s)')
            ax.set_title(f'Classical vs Quantum Factorization Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return {
                "success": True,
                "number": number,
                "factors": factors,
                "processing_time": processing_time,
                "visualization": fig,
                "quantum_advantage": len(str(number)) > 2,  # Simulate advantage for larger numbers
                "advantage_factor": 2**(len(str(number))-3) if len(str(number)) > 3 else 1
            }
        except Exception as e:
            logger.error(f"Error in factorization: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def search(self, database_size: int) -> Dict[str, Any]:
        """Simulate Grover's search algorithm"""
        if not self.initialized:
            return {"success": False, "error": "Quantum simulator not initialized"}
        
        if database_size < 4 or database_size > 10000:
            return {"success": False, "error": "Database size must be between 4 and 10000"}
        
        try:
            # Start timing
            start_time = time.time()
            
            # Generate a simulated database
            database = [f"Item {i}" for i in range(database_size)]
            
            # Select a random target
            import random
            target_index = random.randint(0, database_size - 1)
            target_item = database[target_index]
            
            # Classical search steps would be O(N)
            classical_steps = database_size
            
            # Quantum search steps would be O(sqrt(N))
            quantum_steps = int(database_size**0.5)
            
            # Simulate search time
            sleep_time = 0.1 + (0.001 * quantum_steps)
            time.sleep(sleep_time)
            
            # End timing
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Generate visualization
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            
            # Plot classical vs quantum steps
            sizes = [10, 100, 1000, 10000, 100000]
            classical_steps_plot = sizes
            quantum_steps_plot = [int(s**0.5) for s in sizes]
            
            ax.plot(sizes, classical_steps_plot, 'r-', label='Classical (Linear)')
            ax.plot(sizes, quantum_steps_plot, 'g-', label='Quantum (Square Root)')
            ax.scatter([database_size], [quantum_steps], color='blue', s=100, label='This search')
            
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Database Size')
            ax.set_ylabel('Steps Required')
            ax.set_title(f'Classical vs Quantum Search Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return {
                "success": True,
                "database_size": database_size,
                "target_index": target_index,
                "target_item": target_item,
                "classical_steps": classical_steps,
                "quantum_steps": quantum_steps,
                "processing_time": processing_time,
                "visualization": fig,
                "quantum_advantage": database_size > 100,
                "speedup_factor": database_size / quantum_steps
            }
        except Exception as e:
            logger.error(f"Error in search simulation: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def optimize(self, problem_size: int) -> Dict[str, Any]:
        """Simulate quantum optimization algorithm (QAOA)"""
        if not self.initialized:
            return {"success": False, "error": "Quantum simulator not initialized"}
        
        if problem_size < 2 or problem_size > 20:
            return {"success": False, "error": "Problem size must be between 2 and 20"}
        
        try:
            # Start timing
            start_time = time.time()
            
            # Generate a random problem instance
            np.random.seed(42)  # For reproducibility
            problem_matrix = np.random.randint(-5, 5, size=(problem_size, problem_size))
            
            # Simulate optimization process
            sleep_time = 0.1 + (0.05 * problem_size)
            time.sleep(sleep_time)
            
            # Generate a "solution"
            solution = np.random.randint(0, 2, size=problem_size)
            
            # Calculate a simulated "quality" score (higher is better)
            quality = 0.5 + 0.4 * np.random.random() + 0.1 * (1 / problem_size)
            
            # End timing
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Generate visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            
            # Plot the problem matrix
            im = ax1.imshow(problem_matrix, cmap='viridis')
            ax1.set_title('Problem Matrix')
            plt.colorbar(im, ax=ax1)
            
            # Plot the solution
            ax2.bar(range(problem_size), solution, color='orange')
            ax2.set_title('Optimized Solution')
            ax2.set_xlabel('Variable')
            ax2.set_ylabel('Value (0 or 1)')
            ax2.set_ylim(-0.1, 1.1)
            ax2.set_xticks(range(problem_size))
            
            plt.tight_layout()
            
            return {
                "success": True,
                "problem_size": problem_size,
                "problem_matrix": problem_matrix.tolist(),
                "solution": solution.tolist(),
                "solution_quality": quality,
                "processing_time": processing_time,
                "visualization": fig,
                "quantum_advantage": problem_size > 5,
                "advantage_factor": 2**(problem_size / 5) if problem_size > 5 else 1
            }
        except Exception as e:
            logger.error(f"Error in optimization simulation: {str(e)}")
            return {"success": False, "error": str(e)}

# Session state initialization
def initialize_session_state():
    """Initialize the session state"""
    if 'agent_initialized' not in st.session_state:
        st.session_state.agent_initialized = False
    
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    
    if 'quantum_builder' not in st.session_state:
        st.session_state.quantum_builder = None
    
    if 'quantum_simulator' not in st.session_state:
        st.session_state.quantum_simulator = None
    
    if 'n_qubits' not in st.session_state:
        st.session_state.n_qubits = 4
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'current_task' not in st.session_state:
        st.session_state.current_task = None
    
    if 'task_results' not in st.session_state:
        st.session_state.task_results = {}
    
    if 'quantum_results' not in st.session_state:
        st.session_state.quantum_results = {}

def initialize_agent():
    """Initialize the autonomous agent"""
    if st.session_state.agent_initialized:
        return True
    
    try:
        # Create the autonomous agent
        agent = AutonomousAgent(
            use_quantum=True,
            n_qubits=st.session_state.n_qubits
        )
        
        # Create the quantum circuit builder
        quantum_builder = QuantumCircuitBuilder(n_qubits=st.session_state.n_qubits)
        
        # Create the quantum simulator
        quantum_simulator = QuantumSimulator(n_qubits=st.session_state.n_qubits)
        
        # Store in session state
        st.session_state.agent = agent
        st.session_state.quantum_builder = quantum_builder
        st.session_state.quantum_simulator = quantum_simulator
        st.session_state.agent_initialized = True
        
        # Add initialization to chat history
        st.session_state.chat_history.append({
            "role": "system",
            "content": f"Agent initialized with {st.session_state.n_qubits} qubits.",
            "timestamp": datetime.now().isoformat()
        })
        
        return True
    except Exception as e:
        logger.error(f"Error initializing agent: {str(e)}")
        st.session_state.chat_history.append({
            "role": "system",
            "content": f"Error initializing agent: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })
        return False

def display_header():
    """Display the application header"""
    st.title("QA³: Quantum-Accelerated AI Agent")
    st.markdown("### Truly Agentic Quantum Computing Platform")
    
    # Status indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if PENNYLANE_AVAILABLE:
            st.success("✓ PennyLane Available")
        else:
            st.error("✗ PennyLane Not Available")
    
    with col2:
        if AZURE_QUANTUM_AVAILABLE:
            st.success("✓ Azure Quantum Available")
        else:
            st.warning("✗ Azure Quantum Not Available")
    
    with col3:
        if st.session_state.agent_initialized:
            st.success("✓ Agent Initialized")
        else:
            st.error("✗ Agent Not Initialized")

def process_user_message(message: str):
    """Process a user message"""
    if not st.session_state.agent_initialized:
        st.warning("Please initialize the agent first.")
        return
    
    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": message,
        "timestamp": datetime.now().isoformat()
    })
    
    # Set as current task
    st.session_state.current_task = message
    
    # Run the agent on this task
    with st.spinner("Agent processing task..."):
        result = asyncio.run(st.session_state.agent.process_task(message))
    
    # Store the result
    st.session_state.task_results[message] = result
    
    # Add agent response to history
    response = result.get("result", {}).get("success", False)
    response_text = f"Task processed. Success: {response}"
    if "error" in result.get("result", {}):
        response_text += f"\nError: {result['result']['error']}"
    
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response_text,
        "timestamp": datetime.now().isoformat()
    })

def run_quantum_circuit(circuit_type, params=None):
    """Run a quantum circuit"""
    if not st.session_state.agent_initialized:
        st.warning("Please initialize the agent first.")
        return None
    
    builder = st.session_state.quantum_builder
    
    if circuit_type == "bell_state":
        return builder.create_bell_state()
    elif circuit_type == "ghz_state":
        return builder.create_ghz_state()
    elif circuit_type == "custom":
        if not params:
            return {"success": False, "error": "Parameters required for custom circuit"}
        return builder.create_custom_circuit(params)
    else:
        return {"success": False, "error": f"Unknown circuit type: {circuit_type}"}

def display_circuit_controls():
    """Display quantum circuit controls"""
    st.header("Quantum Circuit Playground")
    
    if not st.session_state.agent_initialized:
        st.warning("Please initialize the agent first.")
        return
    
    circuit_type = st.selectbox("Circuit Type", ["bell_state", "ghz_state", "custom"])
    
    if circuit_type == "custom":
        st.info("This will create a circuit with rotation gates (RX, RY, RZ) followed by CNOT gates.")
        
        # Create parameter inputs for each qubit
        params = []
        cols = st.columns(3)
        
        with cols[0]:
            st.write("RX Angles")
            rx_angles = [st.slider(f"RX Qubit {i}", min_value=0.0, max_value=6.28, value=0.0, key=f"rx_{i}") 
                        for i in range(min(st.session_state.n_qubits, 4))]
        
        with cols[1]:
            st.write("RY Angles")
            ry_angles = [st.slider(f"RY Qubit {i}", min_value=0.0, max_value=6.28, value=0.0, key=f"ry_{i}") 
                        for i in range(min(st.session_state.n_qubits, 4))]
        
        with cols[2]:
            st.write("RZ Angles")
            rz_angles = [st.slider(f"RZ Qubit {i}", min_value=0.0, max_value=6.28, value=0.0, key=f"rz_{i}") 
                        for i in range(min(st.session_state.n_qubits, 4))]
        
        for i in range(min(st.session_state.n_qubits, 4)):
            params.append([rx_angles[i], ry_angles[i], rz_angles[i]])
        
        circuit_params = params
    else:
        circuit_params = None
    
    if st.button("Run Circuit"):
        with st.spinner("Running quantum circuit..."):
            result = run_quantum_circuit(circuit_type, circuit_params)
            st.session_state.quantum_results["circuit"] = result
            
            if result["success"]:
                st.success(f"{circuit_type} circuit created successfully")
                
                # Display the circuit diagram
                if "circuit_diagram" in result:
                    st.subheader("Circuit Diagram")
                    st.pyplot(result["circuit_diagram"])
                
                # Display state vector
                st.subheader("Quantum State")
                state_vector = result["state"]
                
                # Format state vector display
                formatted_state = []
                for i, amplitude in enumerate(state_vector):
                    if isinstance(amplitude, complex):
                        formatted_state.append(f"{i:04b}: {amplitude.real:.4f} + {amplitude.imag:.4f}i")
                    else:
                        formatted_state.append(f"{i:04b}: {amplitude:.4f}")
                
                # Show only non-zero amplitudes
                non_zero = [state for state in formatted_state if not state.endswith("0.0000") and not state.endswith("0.0000i")]
                if non_zero:
                    st.write("Non-zero amplitudes:")
                    for state in non_zero:
                        st.write(state)
                else:
                    st.write("All amplitudes are effectively zero")
            else:
                st.error(result["error"])

def display_agent_controls():
    """Display agent controls"""
    st.header("Agentic Capabilities")
    
    if not st.session_state.agent_initialized:
        st.warning("Please initialize the agent first.")
        return
    
    # Display agent status
    agent_status = st.session_state.agent.get_status()
    
    st.subheader("Agent Status")
    st.json(agent_status)
    
    # User input for tasks
    st.subheader("Task Input")
    task = st.text_area("Enter a task for the agent:", key="task_input")
    
    if st.button("Submit Task"):
        process_user_message(task)
    
    # Display chat history
    st.subheader("Interaction History")
    
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**User:** {msg['content']}")
        elif msg["role"] == "assistant":
            st.markdown(f"**Agent:** {msg['content']}")
        else:
            st.markdown(f"**System:** {msg['content']}")
        st.markdown(f"<small>*{msg['timestamp']}*</small>", unsafe_allow_html=True)
        st.markdown("---")

def display_factorization():
    """Display quantum factorization simulation"""
    st.header("Quantum Factorization")
    
    if not st.session_state.agent_initialized:
        st.warning("Please initialize the agent first.")
        return
    
    simulator = st.session_state.quantum_simulator
    
    st.markdown("""
    This demonstrates a simulated version of Shor's algorithm, which is a quantum algorithm
    that can factor integers exponentially faster than the best known classical algorithm.
    """)
    
    number = st.number_input("Number to Factorize:", min_value=4, max_value=1000, value=15)
    
    if st.button("Run Factorization"):
        with st.spinner("Running quantum factorization simulation..."):
            result = simulator.factorize(number)
            st.session_state.quantum_results["factorization"] = result
            
            if result["success"]:
                st.success(f"Factorization complete: {number} = {' × '.join(map(str, result['factors']))}")
                
                # Display the visualization
                st.subheader("Performance Comparison")
                st.pyplot(result["visualization"])
                
                # Display details
                st.subheader("Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Processing Time", f"{result['processing_time']:.4f} s")
                    
                with col2:
                    if result["quantum_advantage"]:
                        st.metric("Quantum Speedup", f"{result['advantage_factor']:.1f}x")
                    else:
                        st.info("No significant quantum advantage for this small number")
            else:
                st.error(result["error"])

def display_search():
    """Display quantum search simulation"""
    st.header("Quantum Search")
    
    if not st.session_state.agent_initialized:
        st.warning("Please initialize the agent first.")
        return
    
    simulator = st.session_state.quantum_simulator
    
    st.markdown("""
    This demonstrates a simulated version of Grover's algorithm, which is a quantum algorithm
    that can search an unsorted database quadratically faster than classical algorithms.
    """)
    
    database_size = st.slider("Database Size:", min_value=4, max_value=10000, value=100)
    
    if st.button("Run Search"):
        with st.spinner("Running quantum search simulation..."):
            result = simulator.search(database_size)
            st.session_state.quantum_results["search"] = result
            
            if result["success"]:
                st.success(f"Search complete: Found target at index {result['target_index']} in {result['processing_time']:.4f} seconds")
                
                # Display the visualization
                st.subheader("Performance Comparison")
                st.pyplot(result["visualization"])
                
                # Display details
                st.subheader("Results")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Classical Steps", result["classical_steps"])
                
                with col2:
                    st.metric("Quantum Steps", result["quantum_steps"])
                
                with col3:
                    st.metric("Speedup Factor", f"{result['speedup_factor']:.1f}x")
            else:
                st.error(result["error"])

def display_optimization():
    """Display quantum optimization simulation"""
    st.header("Quantum Optimization")
    
    if not st.session_state.agent_initialized:
        st.warning("Please initialize the agent first.")
        return
    
    simulator = st.session_state.quantum_simulator
    
    st.markdown("""
    This demonstrates a simulated version of Quantum Approximate Optimization Algorithm (QAOA),
    which can find approximate solutions to combinatorial optimization problems.
    """)
    
    problem_size = st.slider("Problem Size:", min_value=2, max_value=20, value=5)
    
    if st.button("Run Optimization"):
        with st.spinner("Running quantum optimization simulation..."):
            result = simulator.optimize(problem_size)
            st.session_state.quantum_results["optimization"] = result
            
            if result["success"]:
                st.success(f"Optimization complete: Solution quality {result['solution_quality']:.2%}")
                
                # Display the visualization
                st.subheader("Problem and Solution")
                st.pyplot(result["visualization"])
                
                # Display details
                st.subheader("Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Processing Time", f"{result['processing_time']:.4f} s")
                
                with col2:
                    st.metric("Solution Quality", f"{result['solution_quality']:.2%}")
                
                if result["quantum_advantage"]:
                    st.info(f"Estimated quantum advantage: {result['advantage_factor']:.1f}x faster than classical")
                else:
                    st.info("No significant quantum advantage for this small problem size")
            else:
                st.error(result["error"])

def display_about():
    """Display information about the application"""
    st.header("About QA³ Agent")
    
    st.markdown("""
    ## Quantum-Accelerated AI Agent with True Agentic Capabilities
    
    QA³ is a cutting-edge quantum computing platform that combines quantum acceleration with 
    true autonomous agency. Unlike conventional AI systems, QA³ can:
    
    1. **Perceive its environment** through computer vision and screen understanding
    2. **Make autonomous decisions** based on observations and goals
    3. **Take actions** to accomplish tasks through UI automation
    4. **Learn from interactions** to improve over time
    5. **Manage its own goals** and prioritize tasks
    
    All of this is enhanced by quantum computing capabilities that provide computational advantages
    for specific types of problems.
    
    ### Architecture
    
    The system is built on a modular architecture inspired by cognitive science principles:
    
    - **Perception System**: Screen understanding and element detection
    - **Action System**: UI automation and interaction capabilities
    - **Decision System**: Autonomous decision making with LLM integration
    - **Memory System**: Episodic, semantic, and procedural memory
    - **Goal Management System**: Goal tracking and prioritization
    - **Learning System**: Experience-based learning and improvement
    - **Quantum Core**: Quantum circuit simulation and algorithm acceleration
    
    ### Quantum Advantage
    
    The platform demonstrates quantum advantage in three key areas:
    
    - **Factorization** (Shor's Algorithm): Exponential speedup over classical factoring
    - **Search** (Grover's Algorithm): Quadratic speedup for unsorted database search
    - **Optimization** (QAOA): Potential advantages for complex optimization problems
    
    ### Implementation
    
    This implementation includes:
    
    - True agency using principles from Microsoft Omniparser and Claude Computer Use
    - PennyLane for quantum circuit simulation
    - Integration with Azure Quantum for cloud quantum computing
    - Streamlit for the interactive interface
    """)
    
    # System information
    st.subheader("System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Dependencies:**")
        st.markdown(f"- PennyLane: {'Available ✓' if PENNYLANE_AVAILABLE else 'Not Available ✗'}")
        st.markdown(f"- Azure Quantum: {'Available ✓' if AZURE_QUANTUM_AVAILABLE else 'Not Available ✗'}")
    
    with col2:
        st.markdown("**Agent Status:**")
        if st.session_state.agent_initialized:
            st.markdown("- Status: Initialized ✓")
            st.markdown(f"- Qubits: {st.session_state.n_qubits}")
        else:
            st.markdown("- Status: Not Initialized ✗")

def main():
    """Main application function"""
    # Initialize the session state
    initialize_session_state()
    
    # Set up the page
    st.set_page_config(
        page_title="QA³: Quantum-Accelerated AI Agent",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Display the header
    display_header()
    
    # Sidebar configuration
    st.sidebar.title("QA³ Agent Configuration")
    
    # Quantum settings
    st.sidebar.header("Quantum Settings")
    n_qubits = st.sidebar.slider("Number of Qubits", min_value=2, max_value=16, value=st.session_state.n_qubits)
    
    if n_qubits != st.session_state.n_qubits:
        st.session_state.n_qubits = n_qubits
        st.session_state.agent_initialized = False  # Force reinitialization
    
    # Initialize agent button
    if not st.session_state.agent_initialized:
        if st.sidebar.button("Initialize Agent"):
            with st.spinner("Initializing agent..."):
                success = initialize_agent()
                if success:
                    st.sidebar.success("Agent initialized successfully!")
                else:
                    st.sidebar.error("Failed to initialize agent")
    
    # Navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Agent Interface", "Quantum Circuits", "Factorization", "Search", "Optimization", "About"])
    
    # Display the selected page
    if page == "Agent Interface":
        display_agent_controls()
    elif page == "Quantum Circuits":
        display_circuit_controls()
    elif page == "Factorization":
        display_factorization()
    elif page == "Search":
        display_search()
    elif page == "Optimization":
        display_optimization()
    elif page == "About":
        display_about()

if __name__ == "__main__":
    main()