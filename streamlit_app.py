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

# Import our simplified quantum core
from simple_quantum_core import QuantumCore

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
    st.session_state.quantum_core = QuantumCore(
        use_quantum=True,
        n_qubits=8,
        use_azure=False
    )
    st.session_state.task_history = []
    st.session_state.current_tab = "home"

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
    
    # Update settings if changed
    if use_quantum != st.session_state.quantum_core.use_quantum:
        st.session_state.quantum_core.use_quantum = use_quantum
        st.info(f"Quantum acceleration {'enabled' if use_quantum else 'disabled'}")
    
    if n_qubits != st.session_state.quantum_core.n_qubits:
        st.session_state.quantum_core.n_qubits = n_qubits
        st.info(f"Number of qubits updated to {n_qubits}")
    
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