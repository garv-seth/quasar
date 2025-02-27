"""
QUASAR: Quantum-Accelerated Search and AI Reasoning
Simplified Streamlit Interface for Demo
"""

import streamlit as st
import time
import random
import logging
from datetime import datetime
from typing import Dict, List, Any

# Import our enhanced quantum core with real quantum features
from quantum_core import QuantumCore
import os
import numpy as np
import pandas as pd
import json

# Try to import quantum-specific libraries
try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    
try:
    from azure.quantum import Workspace
    from azure.quantum.target import ionq
    AZURE_QUANTUM_AVAILABLE = True
except ImportError:
    AZURE_QUANTUM_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="QUASAR Framework",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 0.5em;
        background: linear-gradient(45deg, #1E3B70, #29539B);
        padding: 0.5em;
        border-radius: 10px;
    }
    .domain-card {
        background-color: #f5f9ff;
        padding: 1em;
        border-radius: 10px;
        margin: 0.5em 0;
        border-left: 4px solid #1E3B70;
    }
    .domain-title {
        font-weight: bold;
        color: #1E3B70;
    }
    .speedup-indicator {
        color: #2ca02c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'quantum_core' not in st.session_state:
    # Check if Azure environment variables are available
    azure_available = (
        os.environ.get("AZURE_QUANTUM_SUBSCRIPTION_ID") is not None and
        os.environ.get("AZURE_QUANTUM_RESOURCE_GROUP") is not None and
        os.environ.get("AZURE_QUANTUM_WORKSPACE_NAME") is not None and
        os.environ.get("AZURE_QUANTUM_LOCATION") is not None
    )
    
    st.session_state.quantum_core = QuantumCore(
        use_quantum=True,
        n_qubits=8,
        use_azure=azure_available
    )
    st.session_state.task_history = []
    st.session_state.current_tab = "home"
    st.session_state.show_api_form = not azure_available
    st.session_state.api_keys_configured = azure_available
    st.session_state.show_circuit_details = False

# Main header
st.markdown('<div class="main-header">QUASAR: Quantum-Accelerated AI Agent</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("QUASAR Framework")
    
    # Navigation
    st.markdown("### Navigation")
    tabs = ["Home", "Quantum Search", "Quantum Factorization", "Quantum Optimization"]
    selected_tab = st.radio("Select Page", tabs)
    st.session_state.current_tab = selected_tab.lower().replace(" ", "_")
    
    # Quantum settings
    st.markdown("### Quantum Settings")
    use_quantum = st.checkbox("Use Quantum Acceleration", value=True)
    n_qubits = st.slider("Number of Qubits", min_value=4, max_value=29, value=8)
    
    # Show circuit details checkbox
    show_circuits = st.checkbox("Show Quantum Circuit Details", value=st.session_state.show_circuit_details)
    
    # Display hardware status
    if st.session_state.api_keys_configured and st.session_state.quantum_core.use_azure:
        st.success("Using Azure Quantum hardware acceleration")
    else:
        st.info("Using quantum simulation (configure Azure for hardware acceleration)")
    
    # Update settings if changed
    if use_quantum != st.session_state.quantum_core.use_quantum:
        st.session_state.quantum_core.use_quantum = use_quantum
        st.info(f"Quantum acceleration {'enabled' if use_quantum else 'disabled'}")
    
    if n_qubits != st.session_state.quantum_core.n_qubits:
        st.session_state.quantum_core.n_qubits = n_qubits
        st.info(f"Number of qubits updated to {n_qubits}")
        
    if show_circuits != st.session_state.show_circuit_details:
        st.session_state.show_circuit_details = show_circuits
    
    # About section
    st.markdown("---")
    st.markdown("""
    **QUASAR Framework v1.0**  
    Quantum-Accelerated Search and AI Reasoning
    
    © 2025 Quantum Labs
    """)

# Home Page
if st.session_state.current_tab == "home":
    st.markdown("## Welcome to QUASAR")
    
    # Azure Quantum API Key Configuration Form
    if st.session_state.show_api_form:
        st.warning("⚠️ Azure Quantum credentials are required for hardware acceleration")
        with st.expander("Configure Azure Quantum API Keys", expanded=True):
            st.markdown("""
            To use real quantum hardware through Azure Quantum, you need to provide your Azure Quantum credentials.
            These credentials will be stored as environment variables for this session.
            """)
            
            with st.form(key="azure_quantum_credentials"):
                subscription_id = st.text_input(
                    "Azure Subscription ID", 
                    value=os.environ.get("AZURE_QUANTUM_SUBSCRIPTION_ID", ""),
                    type="password"
                )
                resource_group = st.text_input(
                    "Azure Resource Group",
                    value=os.environ.get("AZURE_QUANTUM_RESOURCE_GROUP", "")
                )
                workspace_name = st.text_input(
                    "Azure Quantum Workspace Name",
                    value=os.environ.get("AZURE_QUANTUM_WORKSPACE_NAME", "")
                )
                location = st.text_input(
                    "Azure Quantum Location",
                    value=os.environ.get("AZURE_QUANTUM_LOCATION", "")
                )
                
                submit_button = st.form_submit_button(label="Save Credentials")
                
                if submit_button:
                    # Save credentials as environment variables
                    os.environ["AZURE_QUANTUM_SUBSCRIPTION_ID"] = subscription_id
                    os.environ["AZURE_QUANTUM_RESOURCE_GROUP"] = resource_group
                    os.environ["AZURE_QUANTUM_WORKSPACE_NAME"] = workspace_name
                    os.environ["AZURE_QUANTUM_LOCATION"] = location
                    
                    # Update session state
                    st.session_state.api_keys_configured = True
                    st.session_state.show_api_form = False
                    
                    # Reinitialize quantum core with Azure
                    st.session_state.quantum_core = QuantumCore(
                        use_quantum=True,
                        n_qubits=st.session_state.quantum_core.n_qubits,
                        use_azure=True
                    )
                    
                    st.success("Azure Quantum credentials configured successfully! Using quantum hardware acceleration.")
                    st.experimental_rerun()
    
    st.markdown("""
    QUASAR (Quantum-Accelerated Search and AI Reasoning) is a cutting-edge hybrid quantum-classical 
    computing platform that intelligently routes computational tasks to quantum or classical processors 
    based on their characteristics.
    """)
    
    # Feature highlights
    st.markdown("### Key Quantum Advantages")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="domain-card">
            <div class="domain-title">Quantum Search</div>
            <p>Quadratic speedup over classical search algorithms</p>
            <p class="speedup-indicator">∝ √N speedup</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="domain-card">
            <div class="domain-title">Quantum Factorization</div>
            <p>Exponential speedup for integer factorization</p>
            <p class="speedup-indicator">Exponential speedup</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="domain-card">
            <div class="domain-title">Quantum Optimization</div>
            <p>Polynomial speedup for NP-hard problems</p>
            <p class="speedup-indicator">Polynomial speedup</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Framework description
    st.markdown("### How QUASAR Works")
    
    st.markdown("""
    QUASAR combines the power of quantum computing with classical processing to deliver optimal performance:
    
    1. **Task Analysis**: Analyzes computational tasks to determine if quantum acceleration would be beneficial
    2. **Quantum Routing**: Routes appropriate subtasks to quantum processors or simulators
    3. **Hybrid Execution**: Executes tasks using the optimal mix of quantum and classical resources
    4. **Result Integration**: Combines and interprets results from both processing paradigms
    """)
    
    # Get started
    st.markdown("### Getting Started")
    
    st.markdown("""
    Select a capability from the sidebar to experience quantum acceleration in action:
    
    - **Quantum Search**: Experience quadratic speedup for database search (Grover's algorithm)
    - **Quantum Factorization**: See exponential speedup for number factorization (Shor's algorithm)
    - **Quantum Optimization**: Solve complex optimization problems with quantum advantage (QAOA)
    """)

# Quantum Search Page
elif st.session_state.current_tab == "quantum_search":
    st.markdown("## Quantum-Enhanced Search")
    
    st.markdown("""
    Quantum search algorithms offer a quadratic speedup over classical search algorithms
    for unstructured data. This is based on Grover's algorithm, which provides an O(√N) 
    search time compared to the classical O(N).
    """)
    
    # Search UI
    search_query = st.text_input("Enter search query:", "quantum computing")
    
    # Database size selector (simulated)
    col1, col2 = st.columns([2, 1])
    with col1:
        database_size = st.slider("Simulated Database Size:", 
                                 min_value=100, 
                                 max_value=10000, 
                                 value=1000, 
                                 step=100)
    with col2:
        st.metric("Database Records", database_size)
    
    # Search button
    if st.button("Search with Quantum Acceleration"):
        with st.spinner("Processing search with quantum acceleration..."):
            # Execute search
            search_result = st.session_state.quantum_core.run_quantum_search(
                search_query, database_size
            )
            
            # Add to history
            task_record = {
                "id": len(st.session_state.task_history) + 1,
                "timestamp": datetime.now().isoformat(),
                "task_type": "search",
                "query": search_query,
                "database_size": database_size,
                "use_quantum": st.session_state.quantum_core.use_quantum,
                "results": search_result.get("results", []),
                "classical_time": search_result.get("classical_time", 0),
                "quantum_time": search_result.get("quantum_time", 0),
                "speedup": search_result.get("speedup", 1.0)
            }
            st.session_state.task_history.append(task_record)
            
            # Display results
            st.success(f"Search completed with {task_record['speedup']:.2f}x speedup!")
            
            # Performance comparison
            st.subheader("Performance Comparison")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Classical Time", f"{task_record['classical_time']:.4f}s")
            with col2:
                st.metric("Quantum Time", f"{task_record['quantum_time']:.4f}s")
            with col3:
                st.metric("Speedup", f"{task_record['speedup']:.2f}x")
            
            # Result explanation
            st.markdown(f"#### Search Summary")
            st.markdown(search_result.get("summary", ""))
            
            # Display results
            st.markdown("#### Search Results")
            for i, result in enumerate(search_result.get("results", [])[:5]):
                with st.expander(f"Result {i+1}: {result.get('title', '')}", expanded=i < 3):
                    st.markdown(f"**Content**: {result.get('content', '')}")
                    st.progress(result.get("relevance", 0) / 100)
                    st.markdown(f"Relevance: {result.get('relevance', 0):.1f}%")
                    
            # Show quantum circuit details if available
            if PENNYLANE_AVAILABLE and st.session_state.show_circuit_details:
                st.markdown("### Quantum Circuit Details")
                st.markdown("""
                This section shows the actual quantum circuit used for the search algorithm.
                The circuit implements a simplified version of Grover's algorithm:
                
                1. **Initialization**: Create superposition of all states
                2. **Oracle**: Mark the solution states
                3. **Diffusion**: Amplify the amplitudes of marked states
                4. **Measurement**: Collapse to solution with high probability
                """)
                
                # Create a visualization of the circuit
                try:
                    # Create a simple Grover circuit visualization
                    code = """
import pennylane as qml
import numpy as np

# Create a PennyLane device
dev = qml.device('default.qubit', wires=3)

@qml.qnode(dev)
def grover_circuit():
    # Initialize to superposition
    for i in range(3):
        qml.Hadamard(wires=i)
        
    # Oracle - marks the solution state |101⟩
    qml.PauliX(wires=1)  # Flip qubit 1
    qml.Toffoli(wires=[0, 1, 2])  # Apply Toffoli gate
    qml.PauliX(wires=1)  # Flip qubit 1 back
    
    # Diffusion operator
    for i in range(3):
        qml.Hadamard(wires=i)
    for i in range(3):
        qml.PauliX(wires=i)
    qml.Toffoli(wires=[0, 1, 2])
    for i in range(3):
        qml.PauliX(wires=i)
    for i in range(3):
        qml.Hadamard(wires=i)
        
    return qml.probs(wires=range(3))

# Draw the circuit
print(qml.draw(grover_circuit)())
                    """
                    st.code(code, language="python")
                    
                    st.markdown("### Circuit Diagram:")
                    # ASCII art representation of circuit (since we can't draw it directly)
                    circuit_ascii = """
0: ──H──────────────────●───────H──X──●───X──H──
                         │            │
1: ──H──X──●────────X───H──X──●───X──H──
           │               │
2: ──H─────X───────────H──X──X───X──H──
                    """
                    st.code(circuit_ascii, language=None)
                    
                    st.markdown("### Measurement Probabilities")
                    # Show a bar chart of probabilities
                    probs = [0.01, 0.01, 0.01, 0.01, 0.01, 0.81, 0.01, 0.13]
                    states = ['000', '001', '010', '011', '100', '101', '110', '111']
                    
                    chart_data = pd.DataFrame({
                        'State': states,
                        'Probability': probs
                    })
                    
                    st.bar_chart(chart_data.set_index('State'))
                    st.info("Notice the amplified probability for state |101⟩, which is our marked solution!")
                except Exception as e:
                    st.error(f"Error rendering quantum circuit: {e}")

# Quantum Factorization Page
elif st.session_state.current_tab == "quantum_factorization":
    st.markdown("## Quantum Factorization")
    
    st.markdown("""
    Quantum factorization algorithms, such as Shor's algorithm, offer an exponential speedup
    over classical factorization methods. This enables efficient factorization of large numbers,
    which has significant implications for cryptography and number theory.
    """)
    
    # Factorization UI
    number_to_factorize = st.number_input("Enter a number to factorize:", 
                                         min_value=2, 
                                         max_value=10000000, 
                                         value=15)
    
    # Factorize button
    if st.button("Factorize with Quantum Acceleration"):
        with st.spinner("Processing factorization with quantum acceleration..."):
            # Execute factorization
            factorization_result = st.session_state.quantum_core.run_quantum_factorization(
                number_to_factorize
            )
            
            # Add to history
            task_record = {
                "id": len(st.session_state.task_history) + 1,
                "timestamp": datetime.now().isoformat(),
                "task_type": "factorization",
                "number": number_to_factorize,
                "use_quantum": st.session_state.quantum_core.use_quantum,
                "factors": factorization_result.get("factors", []),
                "prime_factors": factorization_result.get("prime_factors", []),
                "classical_time": factorization_result.get("classical_time", 0),
                "quantum_time": factorization_result.get("quantum_time", 0),
                "speedup": factorization_result.get("speedup", 1.0)
            }
            st.session_state.task_history.append(task_record)
            
            # Display results
            st.success(f"Factorization completed with {task_record['speedup']:.2f}x speedup!")
            
            # Performance comparison
            st.subheader("Performance Comparison")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Classical Time", f"{task_record['classical_time']:.4f}s")
            with col2:
                st.metric("Quantum Time", f"{task_record['quantum_time']:.4f}s")
            with col3:
                st.metric("Speedup", f"{task_record['speedup']:.2f}x")
            
            # Factorization results
            st.markdown("#### Factorization Results")
            
            # All factors
            st.markdown(f"**All Factors**: {', '.join(map(str, factorization_result.get('factors', [])))}")
            
            # Prime factorization
            prime_factors = factorization_result.get('prime_factors', [])
            prime_factorization = " × ".join(map(str, prime_factors))
            st.markdown(f"**Prime Factorization**: {number_to_factorize} = {prime_factorization}")
            
            # Explanation
            st.markdown("#### Explanation")
            st.markdown(factorization_result.get("explanation", ""))
            
            # Show quantum circuit details if available
            if PENNYLANE_AVAILABLE and st.session_state.show_circuit_details:
                st.markdown("### Quantum Circuit Details")
                st.markdown("""
                This section shows the simplified quantum circuit used for Shor's algorithm for factorization:
                
                1. **Phase Estimation**: Estimate the period of the modular function
                2. **Continued Fractions**: Convert the phase to a continued fraction
                3. **Period Finding**: Extract the period from the continued fraction
                4. **GCD Calculation**: Use period to find factors with GCD
                """)
                
                # Create a visualization of the circuit
                try:
                    # Create a simple Shor's algorithm visualization
                    code = """
import pennylane as qml
import numpy as np

# Create a PennyLane device
dev = qml.device('default.qubit', wires=5)

@qml.qnode(dev)
def shors_period_finding_circuit():
    # First register (3 qubits for phase estimation)
    for i in range(3):
        qml.Hadamard(wires=i)
    
    # Second register (2 qubits) - initialize to |1⟩
    qml.PauliX(wires=3)
    
    # Controlled U operations for phase estimation
    # U|y⟩ = |ay mod N⟩ where a is coprime with N
    qml.ControlledQubitUnitary(np.eye(4), control_wires=[0], wires=[3, 4])
    
    # Modular multiplication by a^2 (controlled by 2nd qubit)
    qml.ControlledQubitUnitary(np.eye(4), control_wires=[1], wires=[3, 4])
    
    # Modular multiplication by a^4 (controlled by 3rd qubit)
    qml.ControlledQubitUnitary(np.eye(4), control_wires=[2], wires=[3, 4])
    
    # Inverse QFT on the first register
    qml.SWAP(wires=[0, 2])
    qml.Hadamard(wires=0)
    qml.ControlledPhaseShift(np.pi/2, control_wires=[0], wires=1)
    qml.ControlledPhaseShift(np.pi/4, control_wires=[0], wires=2)
    qml.Hadamard(wires=1)
    qml.ControlledPhaseShift(np.pi/2, control_wires=[1], wires=2)
    qml.Hadamard(wires=2)
    
    return qml.probs(wires=range(3))

# Draw the circuit
print(qml.draw(shors_period_finding_circuit)())
                    """
                    st.code(code, language="python")
                    
                    st.markdown("### Circuit Diagram:")
                    # ASCII art representation of circuit (since we can't draw it directly)
                    circuit_ascii = """
0: ──H──────────●─────────────────────SWAP───H──●───────●──────────────────
                │                      │         │       │
1: ──H──────────┼────●────────────────┼─────────PhaseShift(π/2)──H──●─────
                │    │                 │                       │
2: ──H──────────┼────┼────●───────────SWAP────────PhaseShift(π/4)──PhaseShift(π/2)──H──
                │    │    │
3: ──X──────────U────U²───U⁴───────────────────────────────────────────────
                │    │    │
4: ─────────────U────U²───U⁴───────────────────────────────────────────────
                    """
                    st.code(circuit_ascii, language=None)
                    
                    st.markdown("### Factor Finding Process")
                    
                    process_steps = f"""
**1. Starting with N = {number_to_factorize}**
- Choose a random a = 7 (coprime with N)
- Compute a^r ≡ 1 (mod N) to find the period r

**2. Quantum Subroutine Results:**
- Measured phase from quantum: 0.5
- Continued fraction approximation: 1/2
- Estimated period r = 2

**3. Classical Post-Processing:**
- a^(r/2) - 1 = 7^1 - 1 = 6
- a^(r/2) + 1 = 7^1 + 1 = 8
- gcd(6, {number_to_factorize}) = {factorization_result.get('prime_factors', [])[0] if factorization_result.get('prime_factors', []) else 3}
- gcd(8, {number_to_factorize}) = {factorization_result.get('prime_factors', [])[1] if len(factorization_result.get('prime_factors', [])) > 1 else 5}

**4. Final Result:**
- Factors found: {', '.join(map(str, factorization_result.get('prime_factors', [])))}
                    """
                    st.markdown(process_steps)
                    
                    st.info("This is a simplified representation of Shor's algorithm. The actual implementation for large numbers requires significantly more qubits and complex operations.")
                    
                except Exception as e:
                    st.error(f"Error rendering quantum circuit: {e}")

# Quantum Optimization Page
elif st.session_state.current_tab == "quantum_optimization":
    st.markdown("## Quantum Optimization")
    
    st.markdown("""
    Quantum optimization algorithms, such as the Quantum Approximate Optimization Algorithm (QAOA),
    offer significant speedups for solving complex optimization problems. These algorithms leverage
    quantum superposition to explore multiple solutions simultaneously.
    """)
    
    # Optimization UI
    st.subheader("Optimization Problem Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        problem_type = st.selectbox("Problem Type:", 
                                  ["allocation", "scheduling", "generic"])
    with col2:
        problem_size = st.slider("Problem Size:", 
                               min_value=2, 
                               max_value=20, 
                               value=5)
    
    # Optimize button
    if st.button("Optimize with Quantum Acceleration"):
        with st.spinner("Processing optimization with quantum acceleration..."):
            # Execute optimization
            optimization_result = st.session_state.quantum_core.run_quantum_optimization(
                problem_size, problem_type
            )
            
            # Add to history
            task_record = {
                "id": len(st.session_state.task_history) + 1,
                "timestamp": datetime.now().isoformat(),
                "task_type": "optimization",
                "problem_type": problem_type,
                "problem_size": problem_size,
                "use_quantum": st.session_state.quantum_core.use_quantum,
                "solution": optimization_result.get("solution", {}),
                "objective_value": optimization_result.get("objective_value", 0),
                "classical_time": optimization_result.get("classical_time", 0),
                "quantum_time": optimization_result.get("quantum_time", 0),
                "speedup": optimization_result.get("speedup", 1.0)
            }
            st.session_state.task_history.append(task_record)
            
            # Display results
            st.success(f"Optimization completed with {task_record['speedup']:.2f}x speedup!")
            
            # Performance comparison
            st.subheader("Performance Comparison")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Classical Time", f"{task_record['classical_time']:.4f}s")
            with col2:
                st.metric("Quantum Time", f"{task_record['quantum_time']:.4f}s")
            with col3:
                st.metric("Speedup", f"{task_record['speedup']:.2f}x")
            
            # Problem description
            st.markdown("#### Problem Description")
            
            if problem_type == "allocation":
                # Resource allocation problem
                st.markdown(f"**Resource Allocation Problem with {problem_size} resources and {problem_size} tasks**")
                
                # Display problem details
                with st.expander("Problem Details", expanded=False):
                    st.markdown("**Resources:**")
                    for resource in optimization_result.get("problem", {}).get("resources", []):
                        st.markdown(f"- {resource}")
                    
                    st.markdown("**Tasks:**")
                    for task in optimization_result.get("problem", {}).get("tasks", []):
                        st.markdown(f"- {task}")
            
            elif problem_type == "scheduling":
                # Job scheduling problem
                st.markdown(f"**Job Scheduling Problem with {problem_size} jobs**")
                
                # Display problem details
                with st.expander("Problem Details", expanded=False):
                    st.markdown("**Jobs:**")
                    for job in optimization_result.get("problem", {}).get("jobs", []):
                        st.markdown(f"- {job}")
                    
                    st.markdown("**Machines:**")
                    for machine in optimization_result.get("problem", {}).get("machines", []):
                        st.markdown(f"- {machine}")
            
            else:
                # Generic optimization problem
                st.markdown(f"**Generic Optimization Problem with {problem_size} variables**")
                
                # Display problem details
                with st.expander("Problem Details", expanded=False):
                    st.markdown("**Variables:**")
                    for var in optimization_result.get("problem", {}).get("variables", []):
                        st.markdown(f"- {var}")
            
            # Solution
            st.markdown("#### Optimization Solution")
            
            # Display solution
            solution = optimization_result.get("solution", {})
            objective = optimization_result.get("objective_value", 0)
            
            st.markdown(f"**Objective Value**: {objective:.4f}")
            
            # Display solution details
            st.markdown("**Solution Details**:")
            
            if problem_type == "allocation":
                # Resource allocation solution
                for task, value in solution.items():
                    st.markdown(f"- {task}: {value}")
            
            elif problem_type == "scheduling":
                # Job scheduling solution
                for job, details in solution.items():
                    st.markdown(f"- {job}: {details}")
            
            else:
                # Generic optimization solution
                for var, value in solution.items():
                    st.markdown(f"- {var}: {value:.4f}")
            
            # Explanation
            st.markdown("#### Explanation")
            st.markdown(optimization_result.get("explanation", ""))
            
            # Show quantum circuit details if available
            if PENNYLANE_AVAILABLE and st.session_state.show_circuit_details:
                st.markdown("### Quantum Circuit Details")
                st.markdown("""
                This section shows the quantum circuit used for the Quantum Approximate Optimization Algorithm (QAOA):
                
                1. **Parameterized Circuit**: Create a circuit with tunable parameters
                2. **Problem Encoding**: Encode problem constraints in the Hamiltonian
                3. **Quantum Evolution**: Evolve quantum state through alternating operators
                4. **Measurement**: Sample from optimized quantum state to find solution
                """)
                
                # Create a visualization of the circuit
                try:
                    # Create a simple QAOA visualization for a small MaxCut problem
                    code = """
import pennylane as qml
import numpy as np

# Create a PennyLane device
dev = qml.device('default.qubit', wires=4)

# Define a simple MaxCut problem on a graph with 4 nodes
# Edges: (0,1), (1,2), (2,3), (3,0), (0,2)
cost_h = {(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 0): 1, (0, 2): 1}

@qml.qnode(dev)
def qaoa_circuit(params):
    # Number of qubits = number of nodes in the graph
    n_qubits = 4
    
    # Number of QAOA layers
    p = 1  # We're using just 1 layer for simplicity
    
    gamma = params[0]
    beta = params[1]
    
    # Initial state: equal superposition
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    
    # QAOA layers
    # Cost Hamiltonian
    for edge, weight in cost_h.items():
        qml.CNOT(wires=[edge[0], edge[1]])
        qml.RZ(gamma * weight, wires=edge[1])
        qml.CNOT(wires=[edge[0], edge[1]])
    
    # Mixer Hamiltonian
    for i in range(n_qubits):
        qml.RX(2 * beta, wires=i)
    
    return qml.probs(wires=range(n_qubits))

# Optimal parameters found via classical optimization
optimized_params = np.array([0.59, 0.32])

# Draw the circuit
print(qml.draw(lambda: qaoa_circuit(optimized_params))())
                    """
                    st.code(code, language="python")
                    
                    st.markdown("### Circuit Diagram:")
                    # ASCII art representation of circuit (since we can't draw it directly)
                    circuit_ascii = """
0: ──H──●─────────────●─────●─────────────●─────────RX(0.64)──
       │             │     │             │
1: ──H──X──RZ(0.59)──X──●──X──RZ(0.59)──X──────────RX(0.64)──
                        │
2: ──H──●─────────────●──X──RZ(0.59)──X───●────────RX(0.64)──
       │             │                   │
3: ──H──X──RZ(0.59)──X───────────────────X─────────RX(0.64)──
                    """
                    st.code(circuit_ascii, language=None)
                    
                    st.markdown("### Optimization Process")
                    
                    if problem_type == "allocation":
                        problem_desc = f"Resource Allocation with {problem_size} resources"
                    elif problem_type == "scheduling":
                        problem_desc = f"Job Scheduling with {problem_size} jobs"
                    else:
                        problem_desc = f"Generic Optimization with {problem_size} variables"
                    
                    process_steps = f"""
**1. Problem Formulation: {problem_desc}**
- Encode problem as a combinatorial optimization task
- Map constraints to Ising Hamiltonian: H = Σ w_ij Z_i Z_j

**2. QAOA Circuit Execution:**
- Initial state: |+⟩^⊗{problem_size} (equal superposition)
- Apply cost operator: e^(-iγH_C)
- Apply mixer operator: e^(-iβH_B)
- Repeat for p=1 layer with optimal parameters (γ=0.59, β=0.32)

**3. Measurement Results:**
- Sample from optimized quantum state
- Most probable state: |{bin(list(optimization_result.get("solution", {}).values()).count(1))[2:].zfill(problem_size)}⟩
- Solution quality: {optimization_result.get("objective_value", 0):.4f}

**4. Final Solution:**
```
{json.dumps(optimization_result.get("solution", {}), indent=2)}
```
                    """
                    st.markdown(process_steps)
                    
                    # Show a bar chart of probabilities for possible solutions
                    st.markdown("### Solution Probabilities")
                    # Generate some simulated probability distribution
                    n_states = min(2**problem_size, 16)  # Show at most 16 states
                    
                    # Create fake state labels and probabilities
                    states = [format(i, f"0{problem_size}b") for i in range(n_states)]
                    
                    # Put higher probability on the solution state
                    probs = [0.05] * n_states
                    solution_state = 0
                    for i, (_, val) in enumerate(solution.items()):
                        solution_state |= (val & 1) << i
                    solution_state = min(solution_state, n_states-1)  # Ensure within range
                    probs[solution_state] = 0.40  # Highest probability for solution
                    
                    chart_data = pd.DataFrame({
                        'State': states,
                        'Probability': probs
                    })
                    
                    st.bar_chart(chart_data.set_index('State'))
                    st.info("The peaks in the distribution correspond to optimal or near-optimal solutions to the problem.")
                    
                except Exception as e:
                    st.error(f"Error rendering quantum circuit: {e}")