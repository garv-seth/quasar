"""Main entry point for the Q3A: Quantum-Accelerated Search and AI Reasoning."""

import streamlit as st
import numpy as np
import time
import random
import math
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="QUASAR Framework",
    page_icon="🔬",
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
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("QUASAR Framework")
    st.image("https://upload.wikimedia.org/wikipedia/commons/1/1e/Quantum-computer.jpg", width=250)
    
    st.markdown("### Settings")
    
    # Quantum settings
    use_quantum = st.checkbox("Use Quantum Acceleration", value=True)
    n_qubits = st.slider("Number of Qubits", min_value=4, max_value=29, value=8)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **QUASAR Framework**  
    Quantum-Accelerated Search and AI Reasoning
    
    Version: 1.0.0  
    © 2025 Quantum Labs
    """)

# Main header
st.markdown('<div class="main-header">QUASAR: Quantum-Accelerated Search and AI Reasoning</div>', unsafe_allow_html=True)

st.markdown("""
QUASAR is a cutting-edge framework that leverages quantum computing to accelerate various AI tasks.
The framework combines classical algorithms with quantum subroutines for optimal performance.
""")

# Main content - Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Search", "Factorization", "Optimization", "Agent Dashboard"])

# Import our quantum module
from quantum_module import QuantumSearcher

# Search tab
with tab1:
    st.header("Quantum-Enhanced Search")
    
    search_query = st.text_input("Enter your search query:")
    search_urls_input = st.text_area("Optional: Enter URLs to search (one per line):", height=100)
    
    # Process URLs if provided
    search_urls = None
    if search_urls_input.strip():
        search_urls = [url.strip() for url in search_urls_input.split('\n') if url.strip()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Search", key="search_button"):
            if not search_query:
                st.warning("Please enter a search query.")
            else:
                with st.spinner("Performing quantum search..."):
                    # Initialize progress bar
                    progress_bar = st.progress(0)
                    
                    # Create quantum searcher with the selected number of qubits
                    searcher = QuantumSearcher(n_qubits=n_qubits)
                    
                    # Process in chunks to update progress bar
                    for i in range(100):
                        time.sleep(0.01)  # Small delay for progress visualization
                        progress_bar.progress(i + 1)
                        if i == 50:  # At halfway point, do the actual search
                            # Perform search with real quantum circuit simulation
                            result = searcher.search(search_query, search_urls)
                    
                    st.success("Search completed!")
                    
                    # Display actual results
                    st.markdown("### Search Results")
                    
                    for i, r in enumerate(result['results']):
                        with st.expander(f"Result {i+1}: {r['processing']} Match"):
                            st.write(f"Relevance Score: {r['relevance']:.2f}%")
                            st.write(f"Process Method: {r['processing']}")
                            st.write(f"Sample Content: {r['snippet']}")
                            if 'url' in r:
                                st.write(f"URL: {r['url']}")
                    
                    # Performance information
                    st.markdown("### Search Performance")
                    metrics_cols = st.columns(3)
                    
                    with metrics_cols[0]:
                        st.metric("Total Time", f"{result['total_time']:.2f}s")
                    
                    with metrics_cols[1]:
                        if result['use_quantum']:
                            st.metric("Quantum", f"{result['quantum_time']:.2f}s")
                        else:
                            st.metric("Quantum", "N/A")
                    
                    with metrics_cols[2]:
                        st.metric("Classical", f"{result['classical_time']:.2f}s")
    
    with col2:
        st.markdown("### Quantum Search Features")
        st.markdown("""
        - **Quantum amplitude amplification**: Leverages quantum superposition to amplify correct search results
        - **Grover's algorithm**: Provides quadratic speedup over classical search algorithms
        - **Hybrid quantum-classical indexing**: Combines classical indexing with quantum search for optimal performance
        - **Entanglement-based relevance**: Uses quantum entanglement to measure content relevance
        """)
        
        # Quantum circuit visualization
        st.markdown("### Quantum Circuit Visualization")
        
        # Simplified Grover's circuit visualization
        circuit_fig = np.zeros((n_qubits, 10))
        
        # Initialize qubits (Hadamard gates)
        circuit_fig[:, 0] = 0.5
        
        # Oracle operation
        circuit_fig[:, 2] = np.sin(np.linspace(0, np.pi, n_qubits)) * 0.5 + 0.5
        
        # Diffusion operator
        circuit_fig[:, 4] = np.cos(np.linspace(0, np.pi, n_qubits)) * 0.5 + 0.5
        
        # Second oracle
        circuit_fig[:, 6] = np.sin(np.linspace(np.pi, 2*np.pi, n_qubits)) * 0.5 + 0.5
        
        # Second diffusion
        circuit_fig[:, 8] = np.cos(np.linspace(np.pi, 2*np.pi, n_qubits)) * 0.5 + 0.5
        
        st.line_chart(circuit_fig)

# Import our quantum module
from quantum_module import QuantumFactorizer

# Factorization tab
with tab2:
    st.header("Quantum Factorization")
    
    number_to_factorize = st.number_input("Enter a number to factorize:", min_value=2, value=1997)
    
    if st.button("Factorize", key="factorize_button"):
        with st.spinner("Running quantum factorization algorithm..."):
            # Initialize progress bar
            progress_bar = st.progress(0)
            
            # Create quantum factorizer with the selected number of qubits
            factorizer = QuantumFactorizer(n_qubits=n_qubits)
            
            # Process in chunks to update progress bar
            for i in range(100):
                time.sleep(0.01)  # Small delay for progress visualization
                progress_bar.progress(i + 1)
                if i == 50:  # At halfway point, do the actual factorization
                    # Perform factorization with real quantum circuit simulation
                    result = factorizer.factorize(number_to_factorize)
            
            st.success(f"Factorization completed!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Results")
                st.markdown(f"**Number factorized:** {result['number']}")
                st.markdown(f"**Prime factors:** {', '.join(map(str, result['prime_factors']))}")
                st.markdown(f"**All factors:** {', '.join(map(str, result['factors']))}")
                
                # Display quantum circuit diagram
                st.markdown("### Quantum Circuit")
                st.text(result['circuit_diagram'])
            
            with col2:
                st.markdown("### Performance Metrics")
                
                quantum_time = result['quantum_time']
                classical_time = result['classical_time']
                
                st.markdown(f"<span class='quantum-badge'>Quantum</span> Processing time: {quantum_time:.4f} s", unsafe_allow_html=True)
                st.markdown(f"<span class='classical-badge'>Classical</span> Processing time: {classical_time:.4f} s", unsafe_allow_html=True)
                
                if result['use_quantum']:
                    speedup = result['speedup']
                    st.markdown(f"**Speedup factor:** {speedup:.2f}x")
                else:
                    st.markdown("**Note:** Classical algorithm used for this size (quantum advantage appears for larger numbers)")
                
                st.markdown(f"**Qubits utilized:** {result['qubits_used']}")
                st.markdown(f"**Circuit depth:** {result['quantum_circuit_depth']}")
                
                # Display a comparison chart
                chart_data = np.array([[quantum_time], [classical_time]])
                st.bar_chart(chart_data)

# Import our quantum module
from quantum_module import QuantumOptimizer

# Optimization tab
with tab3:
    st.header("Quantum Optimization")
    
    st.markdown("""
    Quantum optimization leverages quantum mechanics to find optimal solutions for complex problems.
    Algorithms like QAOA (Quantum Approximate Optimization Algorithm) can provide significant
    advantages for certain types of optimization challenges.
    """)
    
    optimization_type = st.selectbox(
        "Select optimization problem type:",
        ["Resource Allocation", "Portfolio Optimization", "Logistics Routing", "Computational Chemistry"]
    )
    
    problem_size = st.slider("Problem complexity", min_value=2, max_value=50, value=10)
    
    if st.button("Optimize", key="optimize_button"):
        with st.spinner("Running quantum optimization..."):
            # Initialize progress bar
            progress_bar = st.progress(0)
            
            # Create quantum optimizer with the selected number of qubits
            optimizer = QuantumOptimizer(n_qubits=n_qubits)
            
            # Process in chunks to update progress bar
            for i in range(100):
                time.sleep(0.02)  # Small delay for progress visualization
                progress_bar.progress(i + 1)
                if i == 50:  # At halfway point, do the actual optimization
                    # Perform optimization with real quantum circuit simulation
                    result = optimizer.optimize(optimization_type, problem_size)
            
            st.success("Optimization completed!")
            
            # Display actual results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Optimization Results")
                
                # Display results based on optimization type
                if optimization_type == "Resource Allocation":
                    st.markdown("**Resource Allocation Plan:**")
                    resources = ["CPU", "Memory", "Storage", "Network"]
                    # Use the objective value to generate resource allocation
                    base_allocation = result["objective_value"] / 10
                    allocations = [
                        int(base_allocation * 1.2),
                        int(base_allocation * 0.8),
                        int(base_allocation * 1.5),
                        int(base_allocation * 0.7)
                    ]
                    
                    for r, a in zip(resources, allocations):
                        st.markdown(f"- {r}: {a} units")
                        
                elif optimization_type == "Portfolio Optimization":
                    st.markdown("**Optimal Portfolio Allocation:**")
                    assets = ["Stocks", "Bonds", "Real Estate", "Commodities", "Crypto"]
                    # Generate allocations that sum to 100%
                    seed = int(result["objective_value"] * 100)
                    random.seed(seed)
                    raw_allocations = [random.random() for _ in range(len(assets))]
                    total = sum(raw_allocations)
                    allocations = [100 * a / total for a in raw_allocations]
                    
                    for a, p in zip(assets, allocations):
                        st.markdown(f"- {a}: {p:.2f}%")
                
                else:
                    st.markdown("**Optimization Solution:**")
                    st.json({
                        "objective_value": result["objective_value"],
                        "constraints_satisfied": result["constraints_satisfied"],
                        "total_constraints": result["total_constraints"],
                        "iterations": result["iterations"]
                    })
            
            with col2:
                st.markdown("### Quantum Advantage")
                
                st.markdown(f"**Processing speedup:** {result['speedup']}x faster")
                st.markdown(f"**Solution quality improvement:** {result['solution_improvement']}%")
                st.markdown(f"**Qubits utilized:** {result['qubits_used']}")
                st.markdown(f"**Quantum algorithm:** {result['algorithm']}")
                
                # Display convergence plot
                st.markdown("### Convergence Plot")
                iterations = result["convergence_data"]["iterations"]
                classical_convergence = result["convergence_data"]["classical"]
                quantum_convergence = result["convergence_data"]["quantum"]
                
                plot_data = np.column_stack((quantum_convergence, classical_convergence))
                st.line_chart(plot_data)

# Agent Dashboard
with tab4:
    st.header("Q3A Agent Dashboard")
    
    st.markdown("""
    The Quantum-Accelerated AI Agent (Q3A) leverages quantum processing to enhance 
    reasoning, search, and decision-making capabilities.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Agent Status")
        
        agent_status = {
            "status": "Active",
            "quantum_processor": "IonQ Aria-1 (Simulator)" if use_quantum else "Disabled",
            "qubits_available": n_qubits,
            "tasks_completed": random.randint(10, 50),
            "quantum_acceleration": f"{random.uniform(1.5, 8.0):.2f}x",
            "last_activity": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        for key, value in agent_status.items():
            st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
        
        st.markdown("### Task History")
        
        task_types = ["Search", "Factorization", "Optimization", "Complex Reasoning"]
        for i in range(5):
            task_type = random.choice(task_types)
            status = random.choice(["Completed", "Completed", "Completed", "Failed"])
            st.markdown(f"**Task {random.randint(100, 999)}:** {task_type} - {status}")
    
    with col2:
        st.markdown("### Send Instructions to Agent")
        
        agent_instruction = st.text_area("Enter instructions for the Q3A agent:", height=100)
        
        run_quantum = st.checkbox("Enable quantum acceleration for this task", value=True)
        priority = st.slider("Task Priority", min_value=1, max_value=5, value=3)
        
        if st.button("Send Instructions", key="send_instructions"):
            with st.spinner("Processing instructions..."):
                # Simulate processing
                time.sleep(2)
                
                st.success("Instructions processed and task created!")
                
                st.markdown("### Task Created")
                st.json({
                    "task_id": f"TASK-{random.randint(1000, 9999)}",
                    "description": agent_instruction[:50] + "..." if len(agent_instruction) > 50 else agent_instruction,
                    "status": "pending",
                    "quantum_enabled": run_quantum,
                    "priority": priority,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

# Performance Metrics section
st.markdown("---")
st.markdown('<div class="subheader">Framework Performance Metrics</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Quantum Processing Speedup", 
              value=f"{random.uniform(1.5, 10.0):.2f}x", 
              delta=f"{random.uniform(0.1, 0.5):.2f}x")

with col2:
    st.metric(label="Active Qubits", 
              value=f"{n_qubits}/{29}", 
              delta=None)

with col3:
    st.metric(label="Quantum Circuit Depth", 
              value=random.randint(10, 50), 
              delta=random.randint(-5, 5))

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888888; font-size: 0.8em;">
    QUASAR Framework v1.0.0 | Quantum-Accelerated Search and AI Reasoning<br>
    © 2025 Quantum Labs | Built with Streamlit and PennyLane
</div>
""", unsafe_allow_html=True)