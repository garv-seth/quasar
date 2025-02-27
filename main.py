"""
QUASAR: Quantum-Accelerated Search and AI Reasoning
Main Streamlit Interface
"""

import streamlit as st
import time
import random
import os
import logging
from datetime import datetime
from typing import Dict, List, Any

# Import our quantum core
from quantum_core import QuantumCore

# Try to import optional dependencies
try:
    import numpy as np
except ImportError:
    # Create a simple numpy substitute for basic functions
    class np:
        @staticmethod
        def sqrt(x):
            return x ** 0.5
            
        @staticmethod
        def array(x):
            return x

try:
    import pandas as pd
except ImportError:
    # Create a simple pandas DataFrame substitute
    class pd:
        @staticmethod
        def DataFrame(data):
            return data

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
    .subheader {
        font-size: 1.5rem;
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
    .metric-card {
        background-color: #f0f7ff;
        padding: 1em;
        border-radius: 10px;
        margin: 0.5em;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 1.8em;
        font-weight: bold;
        color: #1E3B70;
    }
    .metric-label {
        font-size: 0.9em;
        color: #666;
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
        os.environ.get("AZURE_QUANTUM_WORKSPACE_NAME") is not None and
        os.environ.get("AZURE_QUANTUM_SUBSCRIPTION_ID") is not None and
        os.environ.get("AZURE_QUANTUM_RESOURCE_GROUP") is not None and
        os.environ.get("AZURE_QUANTUM_LOCATION") is not None
    )
    
    st.session_state.quantum_core = QuantumCore(
        use_quantum=True,
        n_qubits=8,
        use_azure=azure_available
    )
    
    st.session_state.task_history = []
    st.session_state.current_tab = "home"
    st.session_state.show_quantum_details = False

# Sidebar
with st.sidebar:
    st.title("QUASAR Framework")
    
    # Navigation
    st.markdown("### Navigation")
    tabs = ["Home", "Quantum Search", "Quantum Factorization", "Quantum Optimization", "Task History", "About"]
    selected_tab = st.radio("Select Page", tabs)
    st.session_state.current_tab = selected_tab.lower().replace(" ", "_")
    
    # Quantum settings
    st.markdown("### Quantum Settings")
    use_quantum = st.checkbox("Use Quantum Acceleration", value=True)
    n_qubits = st.slider("Number of Qubits", min_value=4, max_value=29, value=8)
    show_quantum_details = st.checkbox("Show Quantum Details", value=st.session_state.show_quantum_details)
    
    # Update settings if changed
    if use_quantum != st.session_state.quantum_core.use_quantum:
        st.session_state.quantum_core.use_quantum = use_quantum
        st.info(f"Quantum acceleration {'enabled' if use_quantum else 'disabled'}")
    
    if n_qubits != st.session_state.quantum_core.n_qubits:
        st.session_state.quantum_core.n_qubits = n_qubits
        st.info(f"Number of qubits updated to {n_qubits}")
    
    st.session_state.show_quantum_details = show_quantum_details
    
    # About section
    st.markdown("---")
    st.markdown("""
    **QUASAR Framework v1.0**  
    Quantum-Accelerated Search and AI Reasoning
    
    © 2025 Quantum Labs
    """)

# Main header
st.markdown('<div class="main-header">QUASAR: Quantum-Accelerated AI Agent</div>', unsafe_allow_html=True)

# Home Page
if st.session_state.current_tab == "home":
    st.markdown("## Welcome to QUASAR")
    
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
    
    # Azure Quantum status
    azure_available = (
        os.environ.get("AZURE_QUANTUM_WORKSPACE_NAME") is not None and
        os.environ.get("AZURE_QUANTUM_SUBSCRIPTION_ID") is not None and
        os.environ.get("AZURE_QUANTUM_RESOURCE_GROUP") is not None and
        os.environ.get("AZURE_QUANTUM_LOCATION") is not None
    )
    
    st.markdown("### Quantum Backends")
    
    if azure_available:
        st.success("✅ Azure Quantum connection available")
        st.markdown(f"Workspace: `{os.environ.get('AZURE_QUANTUM_WORKSPACE_NAME')}`")
        st.markdown(f"Location: `{os.environ.get('AZURE_QUANTUM_LOCATION')}`")
    else:
        st.warning("⚠️ Azure Quantum connection not configured")
        st.markdown("""
        For Azure Quantum integration, set the following environment variables:
        - `AZURE_QUANTUM_WORKSPACE_NAME`
        - `AZURE_QUANTUM_SUBSCRIPTION_ID`
        - `AZURE_QUANTUM_RESOURCE_GROUP`
        - `AZURE_QUANTUM_LOCATION`
        """)
    
    # Display task history summary if available
    if st.session_state.task_history:
        st.markdown("### Recent Activity")
        
        # Count by task type
        task_types = {}
        for task in st.session_state.task_history:
            task_type = task.get("task_type", "unknown")
            if task_type not in task_types:
                task_types[task_type] = 0
            task_types[task_type] += 1
        
        # Display summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tasks", len(st.session_state.task_history))
        with col2:
            avg_speedup = sum(task.get("speedup", 1.0) for task in st.session_state.task_history) / max(1, len(st.session_state.task_history))
            st.metric("Average Speedup", f"{avg_speedup:.2f}x")
        with col3:
            quantum_tasks = sum(1 for task in st.session_state.task_history if task.get("use_quantum", False))
            st.metric("Quantum Tasks", f"{quantum_tasks}/{len(st.session_state.task_history)}")
        
        # Show last 3 tasks
        st.markdown("#### Last 3 Tasks")
        for task in st.session_state.task_history[-3:]:
            if task.get("task_type") == "search":
                st.markdown(f"🔍 Search: '{task.get('query', '')}' ({task.get('speedup', 1.0):.2f}x speedup)")
            elif task.get("task_type") == "factorization":
                st.markdown(f"🔢 Factorized: {task.get('number', 0)} ({task.get('speedup', 1.0):.2f}x speedup)")
            elif task.get("task_type") == "optimization":
                st.markdown(f"⚙️ Optimization: {task.get('problem_type', '')} ({task.get('speedup', 1.0):.2f}x speedup)")

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
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{database_size}</div>
            <div class="metric-label">Database Records</div>
        </div>
        """, unsafe_allow_html=True)
    
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
            
            # Show quantum details if enabled
            if st.session_state.show_quantum_details:
                st.markdown("#### Quantum Implementation Details")
                st.markdown("""
                **Grover's Algorithm Overview**:
                
                1. Initialize qubits in superposition
                2. Apply oracle for the search condition
                3. Apply diffusion operator
                4. Repeat steps 2-3 approximately √N times
                5. Measure to get the result
                
                This provides a quadratic speedup over classical search algorithms.
                """)
                
                # Add visualization or detailed metrics
                st.markdown("##### Quantum Advantage Scaling")
                sizes = [10, 100, 1000, 10000, 100000]
                classical = [size/1000 for size in sizes]
                quantum = [np.sqrt(size)/1000 for size in sizes]
                
                chart_data = pd.DataFrame({
                    'Database Size': sizes,
                    'Classical (seconds)': classical,
                    'Quantum (seconds)': quantum
                })
                st.line_chart(chart_data, x='Database Size')

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
            
            # Show quantum details if enabled
            if st.session_state.show_quantum_details and factorization_result.get("circuit_results") is not None:
                st.markdown("#### Quantum Circuit Results")
                
                # Display circuit probabilities as a bar chart
                circuit_results = factorization_result.get("circuit_results")
                if circuit_results:
                    states = [f"{i:0{len(circuit_results)-1}b}" for i in range(len(circuit_results))]
                    probs_df = pd.DataFrame({
                        "State": states,
                        "Probability": circuit_results
                    })
                    st.bar_chart(probs_df.set_index("State"))
                
                st.markdown("##### Shor's Algorithm Overview")
                st.markdown("""
                Shor's algorithm for factoring an integer N:
                
                1. Choose a random number a < N
                2. Compute gcd(a, N). If it's not 1, we've found a factor.
                3. Use quantum period finding to find the period r of f(x) = a^x mod N
                4. If r is odd, go back to step 1
                5. Compute gcd(a^(r/2) ± 1, N) to find the factors
                
                The quantum part (period finding) gives the exponential speedup over classical methods.
                """)
                
                # Add visualization of speedup
                st.markdown("##### Quantum Advantage Scaling")
                bit_sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
                classical = [0.001 * (2 ** (bits/8)) for bits in bit_sizes]
                quantum = [0.001 * (bits ** 2) for bits in bit_sizes]
                
                chart_data = pd.DataFrame({
                    'Number Size (bits)': bit_sizes,
                    'Classical (seconds)': classical,
                    'Quantum (seconds)': quantum
                })
                st.line_chart(chart_data, x='Number Size (bits)')

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
                    
                    st.markdown("**Requirements:**")
                    for task, reqs in optimization_result.get("problem", {}).get("requirements", {}).items():
                        st.markdown(f"- {task}: {reqs}")
            
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
                    
                    st.markdown("**Constraints:**")
                    for i, constraint in enumerate(optimization_result.get("problem", {}).get("constraints", [])):
                        st.markdown(f"- Constraint {i+1}: {constraint}")
            
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
            
            # Show quantum details if enabled
            if st.session_state.show_quantum_details:
                st.markdown("#### Quantum Implementation Details")
                
                if optimization_result.get("circuit_results") is not None:
                    # Display circuit probabilities as a bar chart
                    circuit_results = optimization_result.get("circuit_results")
                    if circuit_results:
                        states = [f"{i:0{len(circuit_results)-1}b}" for i in range(len(circuit_results))]
                        probs_df = pd.DataFrame({
                            "State": states,
                            "Probability": circuit_results
                        })
                        st.bar_chart(probs_df.set_index("State"))
                
                st.markdown("##### QAOA Overview")
                st.markdown("""
                The Quantum Approximate Optimization Algorithm (QAOA) works as follows:
                
                1. Encode the optimization problem into a cost Hamiltonian
                2. Prepare an initial state in superposition
                3. Alternately apply the cost Hamiltonian and mixing Hamiltonian
                4. Optimize the circuit parameters to minimize the expected cost
                5. Measure to obtain a solution
                
                QAOA can provide polynomial speedup for many NP-hard problems.
                """)
                
                # Add visualization of scaling
                st.markdown("##### Optimization Performance by Problem Size")
                sizes = [5, 10, 15, 20, 25, 30, 35, 40]
                classical = [0.001 * (1.5 ** size) for size in sizes]
                quantum = [0.001 * (size ** 2.5) for size in sizes]
                
                chart_data = pd.DataFrame({
                    'Problem Size': sizes,
                    'Classical (seconds)': classical,
                    'Quantum (seconds)': quantum
                })
                st.line_chart(chart_data, x='Problem Size')

# Task History Page
elif st.session_state.current_tab == "task_history":
    st.markdown("## Task History")
    
    if not st.session_state.task_history:
        st.info("No tasks have been executed yet. Try using the Quantum Search, Factorization, or Optimization features.")
    else:
        # Summary metrics
        n_tasks = len(st.session_state.task_history)
        avg_speedup = sum(task.get("speedup", 1.0) for task in st.session_state.task_history) / n_tasks
        quantum_tasks = sum(1 for task in st.session_state.task_history if task.get("use_quantum", False))
        
        # Display summary
        st.markdown("### Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Tasks", n_tasks)
        with col2:
            st.metric("Quantum Tasks", f"{quantum_tasks}/{n_tasks}")
        with col3:
            st.metric("Classical Tasks", f"{n_tasks - quantum_tasks}/{n_tasks}")
        with col4:
            st.metric("Average Speedup", f"{avg_speedup:.2f}x")
        
        # Group by task type
        task_types = {}
        for task in st.session_state.task_history:
            task_type = task.get("task_type", "unknown")
            if task_type not in task_types:
                task_types[task_type] = []
            task_types[task_type].append(task)
        
        # Display task history by type
        for task_type, tasks in task_types.items():
            st.markdown(f"### {task_type.capitalize()} Tasks ({len(tasks)})")
            
            for i, task in enumerate(tasks):
                # Create a descriptive title based on task type
                if task_type == "search":
                    title = f"Search for '{task.get('query', '')}'"
                elif task_type == "factorization":
                    title = f"Factorize {task.get('number', 0)}"
                elif task_type == "optimization":
                    title = f"{task.get('problem_type', '').capitalize()} Optimization (size {task.get('problem_size', 0)})"
                else:
                    title = f"Task {task.get('id', i+1)}"
                
                # Add speedup to title
                title += f" ({task.get('speedup', 1.0):.2f}x speedup)"
                
                with st.expander(title, expanded=i == len(tasks) - 1):
                    # Common metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Classical Time", f"{task.get('classical_time', 0):.4f}s")
                    with col2:
                        st.metric("Quantum Time", f"{task.get('quantum_time', 0):.4f}s")
                    with col3:
                        st.metric("Speedup", f"{task.get('speedup', 1.0):.2f}x")
                    
                    # Task-specific details
                    if task_type == "search":
                        st.markdown(f"**Query**: {task.get('query', '')}")
                        st.markdown(f"**Database Size**: {task.get('database_size', 0)}")
                        
                        # Results
                        st.markdown("**Top Results**:")
                        for j, result in enumerate(task.get("results", [])[:3]):
                            st.markdown(f"{j+1}. {result.get('title', '')} ({result.get('relevance', 0):.1f}%)")
                    
                    elif task_type == "factorization":
                        st.markdown(f"**Number**: {task.get('number', 0)}")
                        st.markdown(f"**Factors**: {', '.join(map(str, task.get('factors', [])))}")
                        st.markdown(f"**Prime Factors**: {', '.join(map(str, task.get('prime_factors', [])))}")
                    
                    elif task_type == "optimization":
                        st.markdown(f"**Problem Type**: {task.get('problem_type', '')}")
                        st.markdown(f"**Problem Size**: {task.get('problem_size', 0)}")
                        st.markdown(f"**Objective Value**: {task.get('objective_value', 0):.4f}")
                        
                        # Solution summary
                        st.markdown("**Solution Overview**:")
                        solution = task.get("solution", {})
                        if len(solution) <= 5:
                            for key, value in solution.items():
                                st.markdown(f"- {key}: {value}")
                        else:
                            st.markdown(f"Solution with {len(solution)} variables")
                    
                    # Timestamp
                    st.markdown(f"**Timestamp**: {task.get('timestamp', '')}")

# About Page
elif st.session_state.current_tab == "about":
    st.markdown("## About QUASAR Framework")
    
    st.markdown("""
    ### QUASAR: Quantum-Accelerated Search and AI Reasoning
    
    QUASAR is a cutting-edge hybrid quantum-classical computing platform that enables intelligent
    task optimization by leveraging the unique advantages of quantum computing.
    
    #### Key Components
    
    - **PennyLane Integration**: For quantum circuit simulation and optimization
    - **Azure Quantum Support**: For access to IonQ Aria-1 quantum hardware
    - **Hybrid Computation Engine**: For intelligent quantum/classical task routing
    - **Quantum Algorithms**: For search, factorization, and optimization
    
    #### Quantum Advantages
    
    1. **Quantum Search (Grover's Algorithm)**
       - **Classical Complexity**: O(N)
       - **Quantum Complexity**: O(√N)
       - **Speedup**: Quadratic
       - **Applications**: Database search, pattern matching, cryptanalysis
    
    2. **Quantum Factorization (Shor's Algorithm)**
       - **Classical Complexity**: O(2^(n^(1/3)))
       - **Quantum Complexity**: O(n^3)
       - **Speedup**: Exponential
       - **Applications**: Cryptography, number theory
    
    3. **Quantum Optimization (QAOA)**
       - **Classical Complexity**: Often exponential for exact solutions
       - **Quantum Complexity**: Polynomial for approximate solutions
       - **Speedup**: Polynomial to exponential depending on problem
       - **Applications**: Resource allocation, scheduling, logistics, portfolio optimization
    
    #### Framework Architecture
    
    QUASAR employs a modular architecture with specialized components for quantum and classical processing:
    
    ```
    ┌────────────────┐          ┌─────────────────────┐
    │                │          │                     │
    │  User          │◄────────►│  Task Router        │
    │  Interface     │          │  & Analysis         │
    │                │          │                     │
    └────────────────┘          └──────────┬──────────┘
                                           │
                                           ▼
                       ┌─────────────────────────────────────┐
                       │                                     │
           ┌───────────┤          Task Engine               ├───────────┐
           │           │                                     │           │
           │           └─────────────────────────────────────┘           │
           ▼                                                             ▼
    ┌─────────────────┐                                          ┌─────────────────┐
    │                 │                                          │                 │
    │  Classical      │◄─────────── Task-specific ──────────────►│  Quantum        │
    │  Compute        │             Optimization                 │  Compute        │
    │                 │                                          │                 │
    └────────┬────────┘                                          └────────┬────────┘
             │                                                            │
             └────────────────────────►┌───────────────┐◄────────────────┘
                                       │               │
                                       │  Results      │
                                       │  Integration  │
                                       │               │
                                       └───────────────┘
    ```
    
    #### References
    
    - Grover, L. K. (1996). A fast quantum mechanical algorithm for database search.
    - Shor, P. W. (1994). Algorithms for quantum computation: discrete logarithms and factoring.
    - Farhi, E., Goldstone, J., & Gutmann, S. (2014). A Quantum Approximate Optimization Algorithm.
    
    #### Acknowledgments
    
    QUASAR builds upon research from Microsoft Quantum, IBM Quantum, Google Quantum AI, and the broader
    quantum computing research community.
    
    © 2025 Quantum Labs
    """)

# Run the app
def main():
    """Main function"""
    pass

if __name__ == "__main__":
    main()