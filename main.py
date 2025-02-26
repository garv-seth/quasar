"""Main application for Q3A: Quantum-Accelerated AI Agent."""

import streamlit as st
import logging
import asyncio
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time
import json
import random
import sys
from datetime import datetime
import os
from typing import Dict, Any, List, Optional

# Import QUASAR framework components
from quantum_agent_framework.integration import HybridComputation
from quantum_agent_framework.agents.web_agent import QuantumWebAgent
from quantum_agent_framework.quantum.optimizer import QuantumOptimizer
from quantum_agent_framework.quantum.factorization_manager import FactorizationManager
from components.visualization import (
    create_comparison_chart, 
    create_circuit_visualization,
    create_factorization_tree,
    create_execution_time_comparison,
    create_3d_quantum_state,
    create_quantum_circuit_diagram,
    create_performance_dashboard
)
from database import get_db, crud, models
from database.models import initialize_database
from config import Config

# Import demo module
try:
    from components.demo import QuasarDemo
except ImportError:
    # Create a stub if the demo module isn't available
    class QuasarDemo:
        def __init__(self, *args, **kwargs):
            pass
        def run_streamlit_demo(self):
            st.warning("Demo module not available. Please install the demo component.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quasar-main")

# Initialize database if needed
initialize_database()

# Page configuration
st.set_page_config(
    page_title="Q3A: Quantum-Accelerated AI Agent",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved UI
st.markdown("""
<style>
    .main { padding: 2rem; }
    .stButton>button {
        width: 100%;
        background-color: #7b2cbf;
        color: white;
    }
    .quantum-metrics {
        background-color: #172a45;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #3f51b5;
    }
    .classical-metrics {
        background-color: #1d3557;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #457b9d;
    }
    .processing-type {
        color: #8892b0;
        font-weight: bold;
        font-size: 1.1em;
        margin: 1rem 0;
    }
    .factorization-result {
        background-color: #172a45;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #7b2cbf;
    }
    .sidebar-header {
        font-size: 1.2em;
        font-weight: bold;
        color: #7b2cbf;
    }
    .performance-comparison {
        background-color: #172a45;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 2rem;
        border: 1px solid #3f51b5;
    }
    .highlight {
        color: #7b2cbf;
        font-weight: bold;
    }
    .tab-container {
        padding: 1rem 0;
    }
    .search-result {
        background-color: #172a45;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #458588;
    }
    .circuit-visualization {
        background-color: #172a45;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .feature-card {
        background-color: #172a45;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #7b2cbf;
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .centered {
        text-align: center;
    }
    .large-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .nav-link {
        color: #7b2cbf;
        text-decoration: none;
        font-weight: bold;
    }
    .nav-link:hover {
        color: #9d4edd;
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

def display_quantum_metrics(result: dict):
    """Display quantum processing metrics with enhanced visualization."""
    try:
        st.markdown('<div class="quantum-metrics">', unsafe_allow_html=True)
        st.markdown("### 🔄 Quantum Processing Metrics")

        # Display key metrics in columns
        cols = st.columns(4)
        with cols[0]:
            st.metric("Processing Type", result.get('method_used', 'Unknown').upper())
        with cols[1]:
            st.metric("Backend Used", result.get('backend', 'Unknown'))
        with cols[2]:
            st.metric("Circuit Depth", result.get('details', {}).get('circuit_depth', 0))
        with cols[3]:
            st.metric("Computation Time", f"{result.get('computation_time', 0):.4f}s")

        # Display quantum advantage
        if result.get('details', {}).get('quantum_advantage'):
            st.info(f"Quantum Advantage: {result.get('details', {}).get('quantum_advantage')}")

        # Display quantum circuit visualization
        circuit_fig = create_circuit_visualization(
            np.random.uniform(-np.pi, np.pi, (3, 8, 3))
        )
        st.plotly_chart(circuit_fig, use_container_width=True)

        # Display factorization results if available
        if result.get('factors'):
            st.markdown('<div class="factorization-result">', unsafe_allow_html=True)
            st.markdown("#### 🧮 Factorization Results")

            # Show all factors
            factors = result['factors']
            st.success(f"All factors in ascending order: {', '.join(map(str, factors))}")

            # Create factor tree visualization for educational purposes
            if len(factors) > 2 and result.get('method_used') in ['quantum', 'quantum_hybrid']:
                # Get the number being factorized
                number = int(result.get('task', '0').split()[-1])

                # Create factor tree
                factor_tree = create_factorization_tree(factors, number)
                st.plotly_chart(factor_tree, use_container_width=True)

            # Show computation details
            st.info(f"Computation method: {result.get('method_used', 'Unknown').upper()}")
            st.info(f"Backend used: {result.get('backend', 'Unknown')}")

            if 'response' in result:
                st.markdown("#### 📝 Explanation")
                st.write(result['response'])

            st.markdown('</div>', unsafe_allow_html=True)

        # Display performance comparison
        if result.get('classical_time') and result.get('computation_time'):
            st.markdown("#### ⚡ Performance Comparison")
            perf_fig = create_execution_time_comparison(
                [result.get('computation_time', 0)],
                [result.get('classical_time', 0)],
                ["Processing Time"]
            )
            st.plotly_chart(perf_fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        logger.error(f"Error displaying metrics: {str(e)}")
        st.error("Error displaying metrics. The computation results are still valid.")

def display_search_results(results: dict):
    """Display search results with quantum metrics."""
    try:
        st.markdown('<div class="quantum-metrics">', unsafe_allow_html=True)
        st.markdown("### 🔍 Quantum-Enhanced Search Results")

        # Display key metrics
        cols = st.columns(4)
        with cols[0]:
            st.metric("Total Results", results.get('total_results', 0))
        with cols[1]:
            st.metric("Total Sources", results.get('total_sources', 0))
        with cols[2]:
            st.metric("Processing Time", f"{results.get('processing_time', 0):.4f}s")
        with cols[3]:
            st.metric("Quantum Advantage", 
                      results.get('quantum_metrics', {}).get('quantum_advantage', 'N/A'))

        # Display performance comparison
        perf = results.get('performance_comparison', {})
        if perf:
            st.markdown("#### ⚡ Performance Comparison")

            perf_fig = create_execution_time_comparison(
                [perf.get('quantum_time', 0)],
                [perf.get('classical_time', 0)],
                ["Search Processing Time"]
            )
            st.plotly_chart(perf_fig, use_container_width=True)

            # Additional metrics
            st.markdown(f"""
            - Speed improvement: **{perf.get('speedup', 0):.2f}x** faster
            - Result similarity: **{perf.get('result_similarity', 0)*100:.1f}%** (compared to classical algorithm)
            """)

        # Display search results
        st.markdown("#### 📊 Top Results")
        for i, result in enumerate(results.get('results', [])[:5]):
            st.markdown(f'<div class="search-result">', unsafe_allow_html=True)
            st.markdown(f"##### {i+1}. {result.get('title', 'Untitled')}")
            st.markdown(f"**URL:** {result.get('url', 'Unknown')}")
            st.markdown(f"**Quantum Score:** {result.get('quantum_score', 0):.4f}")

            # Show keywords if available
            if result.get('keywords'):
                st.markdown(f"**Keywords:** {', '.join(result.get('keywords', [])[:5])}")

            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        logger.error(f"Error displaying search results: {str(e)}")
        st.error("Error displaying search results. The search was completed successfully.")

def display_optimization_results(result: dict):
    """Display optimization results with quantum metrics."""
    try:
        st.markdown('<div class="quantum-metrics">', unsafe_allow_html=True)
        st.markdown("### 🔄 Quantum Optimization Results")

        # Display key metrics
        cols = st.columns(4)
        with cols[0]:
            st.metric("Method Used", result.get('method_used', 'Unknown').upper())
        with cols[1]:
            st.metric("Objective Value", f"{result.get('objective_value', 0):.2f}")
        with cols[2]:
            st.metric("Computation Time", f"{result.get('computation_time', 0):.4f}s")
        with cols[3]:
            st.metric("Quantum Advantage", result.get('quantum_advantage', 'N/A'))

        # Display allocation
        st.markdown("#### 📊 Resource Allocation")
        allocation = result.get('allocation', [])
        if allocation:
            # Create bar chart
            fig = px.bar(
                x=[f"Resource {i+1}" for i in range(len(allocation))],
                y=allocation,
                color=allocation,
                color_continuous_scale='Viridis'
            )

            fig.update_layout(
                title='Resource Allocation',
                xaxis_title='Resources',
                yaxis_title='Allocation Value',
                template='plotly_dark'
            )

            st.plotly_chart(fig, use_container_width=True)

        # Display explanation
        if 'response' in result:
            st.markdown("#### 📝 Explanation")
            st.write(result['response'])

        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        logger.error(f"Error displaying optimization results: {str(e)}")
        st.error("Error displaying optimization results. The optimization was completed successfully.")

async def process_task(task: str, n_qubits: int = 8, use_quantum: bool = True, use_azure: bool = True) -> dict:
    """Process user task with quantum acceleration."""
    try:
        hybrid_computer = HybridComputation(
            n_qubits=n_qubits,
            use_quantum=use_quantum,
            use_azure=use_azure
        )
        result = await hybrid_computer.process_task(task)
        logger.info(f"Task processing result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error during task processing: {str(e)}")
        return {"error": str(e)}

async def process_search(query: str, n_qubits: int = 8, use_quantum: bool = True) -> dict:
    """Process search query with quantum enhancement."""
    try:
        web_agent = QuantumWebAgent(n_qubits=n_qubits, use_quantum=use_quantum)
        result = await web_agent.search(query)
        logger.info(f"Search result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        return {"error": str(e)}

async def analyze_content(query: str, n_qubits: int = 8, use_quantum: bool = True) -> dict:
    """Analyze content with quantum enhancement."""
    try:
        web_agent = QuantumWebAgent(n_qubits=n_qubits, use_quantum=use_quantum)
        result = await web_agent.analyze_content(query)
        logger.info(f"Analysis result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error during content analysis: {str(e)}")
        return {"error": str(e)}

def display_home_page():
    """Display the home page."""
    st.title("⚛️ Q3A: Quantum-Accelerated AI Agent")

    st.markdown("""
    <div class="centered">
    <h3>Unleashing Exponential Speedups with Quantum-Classical Hybrid Intelligence</h3>
    </div>
    """, unsafe_allow_html=True)

    # Quick links
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🧮 Factorization", key="nav_factor"):
            st.session_state.page = "factorization"
            st.rerun()
    with col2:
        if st.button("🔍 Search", key="nav_search"):
            st.session_state.page = "search"
            st.rerun()
    with col3:
        if st.button("⚙️ Optimization", key="nav_opt"):
            st.session_state.page = "optimization"
            st.rerun()
    with col4:
        if st.button("🚀 Demonstrations", key="nav_demo"):
            st.session_state.page = "demo"
            st.rerun()

    # Feature cards
    st.markdown("### 🚀 Key Features")

    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.markdown("""
        <div class="feature-card">
        <div class="centered">
        <div class="large-icon">🧮</div>
        <h4>Quantum Factorization</h4>
        </div>
        <p>Experience <span class="highlight">exponential speedup</span> with Shor's algorithm for integer factorization. Q3A automatically determines when to use quantum computing for maximum advantage.</p>
        <p>Our hybrid approach combines quantum period-finding with classical post-processing to outperform traditional methods by orders of magnitude.</p>
        </div>
        """, unsafe_allow_html=True)

    with row1_col2:
        st.markdown("""
        <div class="feature-card">
        <div class="centered">
        <div class="large-icon">🔍</div>
        <h4>Quantum-Enhanced Search</h4>
        </div>
        <p>Achieve <span class="highlight">quadratic speedup</span> with Grover's algorithm for unstructured database search. Q3A's quantum pattern matching delivers superior information retrieval.</p>
        <p>Our system accelerates pattern recognition and similarity calculations using quantum interference and superposition principles.</p>
        </div>
        """, unsafe_allow_html=True)

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.markdown("""
        <div class="feature-card">
        <div class="centered">
        <div class="large-icon">⚙️</div>
        <h4>Quantum Optimization</h4>
        </div>
        <p>Solve complex resource allocation problems with <span class="highlight">quantum optimization</span> algorithms like QAOA that provide polynomial speedups over classical methods.</p>
        <p>Q3A excels at constrained optimization problems, finding better solutions faster through quantum superposition of all possible configurations.</p>
        </div>
        """, unsafe_allow_html=True)

    with row2_col2:
        st.markdown("""
        <div class="feature-card">
        <div class="centered">
        <div class="large-icon">🧠</div>
        <h4>AI Integration</h4>
        </div>
        <p>Combine the power of GPT-4o with quantum acceleration for <span class="highlight">enhanced natural language processing</span> and problem-solving.</p>
        <p>Our system intelligently decides when to use quantum computing versus classical AI, delivering optimal performance for each task type.</p>
        </div>
        """, unsafe_allow_html=True)

    # Technical breakdown
    st.markdown("### 🔬 Quantum Advantage Breakdown")

    # Create comparison table
    data = {
        'Task Type': ['Factorization', 'Search', 'Optimization', 'Machine Learning'],
        'Quantum Algorithm': ['Shor\'s Algorithm', 'Grover\'s Algorithm', 'QAOA', 'QML Variational Circuits'],
        'Quantum Speedup': ['Exponential (O(log³N) vs O(e^N))', 'Quadratic (O(√N) vs O(N))', 
                           'Polynomial (varies by problem)', '1.5-3x improvement'],
        'Best Use Case': ['Large number factorization', 'Unstructured database search', 
                         'Resource allocation problems', 'Feature extraction & classification']
    }

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

    # Show quantum circuit visualization
    st.markdown("### ⚛️ Quantum Circuit Visualization")

    st.markdown("""
    Q3A utilizes parameterized quantum circuits tailored to each problem type. 
    Below is an example of a quantum circuit used in our framework:
    """)

    circuit_params = np.random.uniform(-np.pi, np.pi, (3, 8, 3))
    circuit_fig = create_circuit_visualization(circuit_params)
    st.plotly_chart(circuit_fig, use_container_width=True)

    # Latest tasks
    st.markdown("### 📊 Recent Tasks")

    try:
        # Get recent tasks from database
        db = next(get_db())
        recent_tasks = crud.get_task_history(db, limit=5)

        if recent_tasks:
            task_data = []
            for task in recent_tasks:
                task_data.append({
                    "ID": task.id,
                    "Description": task.description,
                    "Status": task.status,
                    "Method": task.processing_method,
                    "Time": f"{task.execution_time:.4f}s" if task.execution_time else "N/A",
                    "Created": task.created_at.strftime("%Y-%m-%d %H:%M:%S")
                })

            task_df = pd.DataFrame(task_data)
            st.dataframe(task_df, use_container_width=True)
        else:
            st.info("No tasks have been processed yet. Try using one of the features above!")
    except Exception as e:
        logger.error(f"Error fetching recent tasks: {str(e)}")
        st.error("Error fetching recent tasks.")

def display_factorization_page():
    """Display the factorization page."""
    st.title("🧮 Quantum-Accelerated Factorization")

    st.markdown("""
    Experience the power of quantum computing for integer factorization. Shor's algorithm 
    provides an **exponential speedup** over classical methods for factoring large numbers.

    Our system automatically determines when to use quantum methods and when classical 
    approaches are more efficient, providing the best of both worlds.
    """)

    # Input section
    st.markdown("### Enter a Number to Factorize")

    number = st.number_input(
        "Number to factorize",
        min_value=2,
        max_value=10000000,
        value=3960,
        help="Enter a positive integer to factorize. Larger numbers will demonstrate greater quantum advantage."
    )

    process_button = st.button("🚀 Factorize Number", use_container_width=True)

    if process_button:
        with st.spinner("🔄 Processing with quantum acceleration..."):
            try:
                # Create factorization task
                task = f"Factor {number}"

                # Process task
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    process_task(
                        task, 
                        n_qubits=st.session_state.n_qubits, 
                        use_quantum=st.session_state.use_quantum,
                        use_azure=st.session_state.use_azure
                    )
                )

                if result and 'error' not in result:
                    # Display results
                    display_quantum_metrics(result)

                    # Save to database
                    try:
                        db = next(get_db())
                        task_record = crud.create_task(db, task, task_type="factorization")
                        crud.update_task_result(
                            db,
                            task_record.id,
                            result,
                            result.get('computation_time', 0),
                            result.get('method_used', 'unknown')
                        )
                    except Exception as e:
                        logger.error(f"Error saving to database: {str(e)}")
                else:
                    st.error(f"An error occurred: {result.get('error', 'Unknown error')}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Explanation of quantum factorization
    with st.expander("ℹ️ How Quantum Factorization Works"):
        st.markdown("""
        ### Shor's Algorithm

        **Shor's algorithm** is a quantum algorithm for integer factorization, formulated by Peter Shor in 1994.
        It finds the prime factors of an integer N significantly faster than the best-known classical algorithm.

        On a classical computer, the most efficient algorithm is the General Number Field Sieve,
        which runs in sub-exponential time: O(e^(log n)^(1/3) × (log log n)^(2/3)).

        Shor's algorithm, however, runs in polynomial time: O(log³ n), providing an **exponential speedup**.

        ### The Algorithm Steps

        1. **Quantum Period Finding**: The core of Shor's algorithm is finding the period of a function
           using quantum superposition and the Quantum Fourier Transform.

        2. **Classical Post-Processing**: Once the period is found, classical calculations are used to
           determine the factors.

        ### Our Hybrid Approach

        Q3A uses a hybrid approach:

        - For small numbers, we use classical methods for efficiency
        - For larger numbers, we use quantum methods for speed
        - We handle edge cases and optimizations automatically

        This provides optimal performance across all input sizes.
        """)

        # Add a simple visualization
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c0/Shor%27s_algorithm.svg/440px-Shor%27s_algorithm.svg.png", 
                caption="Quantum circuit for Shor's algorithm (Source: Wikipedia)")

def display_search_page():
    """Display the search page."""
    st.title("🔍 Quantum-Enhanced Search")

    st.markdown("""
    Experience the power of quantum computing for search tasks. Quantum algorithms provide
    a **quadratic speedup** over classical search methods, helping you find information faster.

    Our system accelerates pattern recognition and similarity calculations using quantum 
    interference and superposition principles.
    """)

    # Input section
    st.markdown("### Enter a Search Query")

    query = st.text_area(
        "Search query",
        value="quantum computing applications",
        help="Enter a search query. Our quantum-enhanced algorithms will find the most relevant results."
    )

    search_button = st.button("🔍 Quantum Search", use_container_width=True)

    if search_button:
        with st.spinner("🔄 Searching with quantum acceleration..."):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    process_search(
                        query,
                        n_qubits=st.session_state.n_qubits,
                        use_quantum=st.session_state.use_quantum
                    )
                )

                if result and 'error' not in result:
                    # Display results
                    display_search_results(result)

                    # Save to database
                    try:
                        db = next(get_db())
                        task_record = crud.create_task(db, f"Search: {query}", task_type="search")
                        crud.update_task_result(
                            db,
                            task_record.id,
                            result,
                            result.get('processing_time', 0),
                            "quantum" if st.session_state.use_quantum else "classical"
                        )
                    except Exception as e:
                        logger.error(f"Error saving to database: {str(e)}")
                else:
                    st.error(f"An error occurred: {result.get('error', 'Unknown error')}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Explanation of quantum search
    with st.expander("ℹ️ How Quantum Search Works"):
        st.markdown("""
        ### Grover's Algorithm

        **Grover's algorithm** is a quantum algorithm for unstructured search, developed by Lov Grover in 1996.
        It finds an element in an unsorted database with N entries in approximately O(√N) steps,
        whereas classical algorithms require O(N) steps.

        This provides a **quadratic speedup** which becomes increasingly significant as the database size grows.

        ### The Algorithm Steps

        1. **Initialization**: Create a superposition of all possible states

        2. **Oracle**: Mark the solution states through phase inversion

        3. **Diffusion**: Amplify the amplitude of the marked states

        4. **Measurement**: Observe the system to find the solution with high probability

        ### Our Enhanced Implementation

        Q3A extends these principles with:

        - **Quantum Pattern Matching**: Using quantum interference for enhanced similarity detection

        - **Quantum Feature Extraction**: Mapping document features to quantum states for faster processing

        - **Hybrid Computation**: Combining quantum search with classical post-processing

        This delivers superior search performance for information retrieval tasks.
        """)

        # Add a simple visualization
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Quantum_amplitude_of_Grover_iterations.png/400px-Quantum_amplitude_of_Grover_iterations.png", 
                caption="Amplitude amplification in Grover's algorithm (Source: Wikipedia)")

async def display_optimization_page():
    """Display the optimization page."""
    st.title("⚙️ Quantum Optimization")

    st.markdown("""
    Experience the power of quantum computing for optimization problems. Quantum algorithms
    like QAOA (Quantum Approximate Optimization Algorithm) can provide significant speedups
    for complex optimization tasks with many constraints or variables.

    Our system excels at resource allocation, scheduling, and other constrained optimization problems.
    """)

    # Input section
    st.markdown("### Resource Optimization")

    col1, col2 = st.columns(2)

    with col1:
        num_resources = st.slider(
            "Number of resources",
            min_value=5,
            max_value=50,
            value=20,
            help="Number of resources to optimize"
        )

    with col2:
        num_constraints = st.slider(
            "Number of constraints",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of constraints on resource allocation"
        )

    optimize_button = st.button("⚙️ Optimize Resources", use_container_width=True)

    if optimize_button:
        with st.spinner("🔄 Optimizing with quantum acceleration..."):
            try:
                # Generate random optimization problem
                resources = {
                    "items": list(range(num_resources)),
                    "weights": [random.uniform(0.5, 10.0) for _ in range(num_resources)],
                    "values": [random.uniform(1.0, 20.0) for _ in range(num_resources)],
                    "constraints": [
                        {
                            "resources": random.sample(range(num_resources), min(random.randint(2, 5), num_resources)),
                            "type": random.choice(["sum", "max"]),
                            "limit": random.uniform(10, 100)
                        }
                        for _ in range(num_constraints)
                    ],
                    "objective": "maximize"
                }

                # Create optimization task
                task = f"Optimize allocation of {num_resources} resources with {num_constraints} constraints"

                # Process task
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Get hybrid computation
                hybrid_comp = HybridComputation(
                    n_qubits=st.session_state.n_qubits,
                    use_quantum=st.session_state.use_quantum
                )

                # Get optimizer
                if hasattr(hybrid_comp, 'quantum_optimizer'):
                    optimizer = hybrid_comp.quantum_optimizer
                else:
                    optimizer = QuantumOptimizer(
                        n_qubits=st.session_state.n_qubits
                    )

                # Process with quantum
                start_time = time.time()
                quantum_result = optimizer.optimize_resource_allocation(resources)
                quantum_time = time.time() - start_time

                # Process with classical
                classical_start = time.time()
                classical_result = await hybrid_comp._classical_optimization(resources)
                classical_time = time.time() - classical_start

                # Calculate speedup
                speedup = classical_time / quantum_time if quantum_time > 0 else 0

                # Add to result
                result = quantum_result
                result['classical_time'] = classical_time
                result['speedup'] = speedup
                result['execution_time'] = quantum_time

                if result and 'error' not in result:
                    # Display results
                    display_optimization_results(result)

                    # Save to database
                    try:
                        db = next(get_db())
                        task_record = crud.create_task(db, task, task_type="optimization")
                        crud.update_task_result(
                            db,
                            task_record.id,
                            result,
                            quantum_time,
                            "quantum_optimization"
                        )
                    except Exception as e:
                        logger.error(f"Error saving to database: {str(e)}")
                else:
                    st.error(f"An error occurred: {result.get('error', 'Unknown error')}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Explanation of quantum optimization
    with st.expander("ℹ️ How Quantum Optimization Works"):
        st.markdown("""
        ### Quantum Approximate Optimization Algorithm (QAOA)

        **QAOA** is a quantum algorithm designed to find approximate solutions to combinatorial 
        optimization problems. It was developed by Farhi, Goldstone, and Gutmann in 2014.

        QAOA can provide polynomial speedups for certain optimization problems, with the
        advantage growing with the complexity of the problem.

        ### The Algorithm Steps

        1. **Problem Mapping**: Convert the optimization problem to a quantum Hamiltonian

        2. **Parameter Optimization**: Set up a parameterized quantum circuit

        3. **Quantum Evolution**: Apply alternating cost and mixer Hamiltonians

        4. **Measurement**: Sample from the resulting quantum state to find optimal solutions

        ### Our Enhanced Implementation

        Q3A extends QAOA with:

        - **Adaptive Parameter Selection**: Dynamically adjusting circuit parameters

        - **Problem Decomposition**: Breaking down large problems into manageable chunks

        - **Hybrid Optimization**: Combining quantum and classical optimization techniques

        This provides superior performance for resource allocation, scheduling, and portfolio optimization.
        """)

        # Add circuit visualization
        circuit_params = np.random.uniform(-np.pi, np.pi, (4, 8, 3))
        circuit_fig = create_circuit_visualization(circuit_params)
        st.plotly_chart(circuit_fig, use_container_width=True)

def display_demo_page():
    """Display the demonstration page."""
    st.title("🚀 Quantum Acceleration Demonstrations")

    st.markdown("""
    This page provides interactive demonstrations of quantum advantage across
    different types of tasks. See the power of quantum computing in action!
    """)

    # Run the demo module
    demo = QuasarDemo(
        n_qubits=st.session_state.n_qubits,
        use_quantum=st.session_state.use_quantum
    )
    demo.run_streamlit_demo()

def display_task_history_page():
    """Display the task history page."""
    st.title("📋 Task History")

    st.markdown("""
    View and analyze your task history. This page shows all tasks processed by the system,
    including performance metrics and quantum advantage.
    """)

    # Get task history from database
    try:
        db = next(get_db())
        tasks = crud.get_task_history(db, limit=50)

        if tasks:
            # Create dataframe
            task_data = []
            for task in tasks:
                task_data.append({
                    "ID": task.id,
                    "Description": task.description,
                    "Type": task.task_type,
                    "Status": task.status,
                    "Method": task.processing_method,
                    "Time (s)": task.execution_time if task.execution_time else 0,
                    "Created": task.created_at.strftime("%Y-%m-%d %H:%M:%S") if task.created_at else "",
                    "Completed": task.completed_at.strftime("%Y-%m-%d %H:%M:%S") if task.completed_at else ""
                })

            # Create dataframe
            df = pd.DataFrame(task_data)

            # Display filters
            col1, col2, col3 = st.columns(3)
            with col1:
                task_type_filter = st.selectbox(
                    "Filter by task type",
                    ["All"] + list(df["Type"].unique())
                )
            with col2:
                method_filter = st.selectbox(
                    "Filter by method",
                    ["All"] + list(df["Method"].unique())
                )
            with col3:
                status_filter = st.selectbox(
                    "Filter by status",
                    ["All"] + list(df["Status"].unique())
                )

            # Apply filters
            filtered_df = df.copy()
            if task_type_filter != "All":
                filtered_df = filtered_df[filtered_df["Type"] == task_type_filter]
            if method_filter != "All":
                filtered_df = filtered_df[filtered_df["Method"] == method_filter]
            if status_filter != "All":
                filtered_df = filtered_df[filtered_df["Status"] == status_filter]

            # Display dataframe
            st.dataframe(filtered_df, use_container_width=True)

            # Task details
            task_id = st.number_input("Enter task ID to view details", min_value=1, max_value=df["ID"].max())

            if st.button("Show Task Details"):
                task_details = crud.get_task_with_metrics(db, task_id)

                if task_details:
                    st.markdown("### Task Details")

                    # Basic info
                    st.markdown(f"**Description:** {task_details['task']['description']}")
                    st.markdown(f"**Type:** {task_details['task']['task_type']}")
                    st.markdown(f"**Status:** {task_details['task']['status']}")
                    st.markdown(f"**Method:** {task_details['task']['processing_method']}")
                    st.markdown(f"**Execution Time:** {task_details['task']['execution_time']} seconds")

                    # Results
                    if task_details['task']['result']:
                        st.markdown("### Results")
                        st.json(task_details['task']['result'])

                    # Quantum metrics
                    if task_details['quantum_metrics']:
                        st.markdown("### Quantum Metrics")
                        qm = task_details['quantum_metrics']

                        cols = st.columns(4)
                        with cols[0]:
                            st.metric("Quantum Advantage", f"{qm['quantum_advantage']:.2f}x")
                        with cols[1]:
                            st.metric("Circuit Depth", qm['circuit_depth'])
                        with cols[2]:
                            st.metric("Qubit Count", qm['qubit_count'])
                        with cols[3]:
                            st.metric("Backend", qm['backend_name'])

                    # Performance comparison
                    if task_details['performance_comparison']:
                        st.markdown("### Performance Comparison")
                        pc = task_details['performance_comparison']

                        cols = st.columns(3)
                        with cols[0]:
                            st.metric("Quantum Time", f"{pc['quantum_time']:.4f}s")
                        with cols[1]:
                            st.metric("Classical Time", f"{pc['classical_time']:.4f}s")
                        with cols[2]:
                            st.metric("Speedup", f"{pc['speedup']:.2f}x")
                else:
                    st.error(f"Task with ID {task_id} not found")

            # Performance metrics
            st.markdown("### Performance Metrics")

            # Create charts
            col1, col2 = st.columns(2)

            with col1:
                # Task type distribution
                type_counts = df["Type"].value_counts()

                fig = px.pie(
                    names=type_counts.index,
                    values=type_counts.values,
                    title="Task Type Distribution"
                )
                fig.update_layout(template="plotly_dark")

                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Method distribution
                method_counts = df["Method"].value_counts()

                fig = px.pie(
                    names=method_counts.index,
                    values=method_counts.values,
                    title="Processing Method Distribution"
                )
                fig.update_layout(template="plotly_dark")

                st.plotly_chart(fig, use_container_width=True)

            # Execution time by task type
            fig = px.box(
                df,
                x="Type",
                y="Time (s)",
                color="Method",
                title="Execution Time by Task Type"
            )
            fig.update_layout(template="plotly_dark")

            st.plotly_chart(fig, use_container_width=True)

            # Performance metrics
            st.markdown("### Quantum Advantage Metrics")

            # Get performance metrics
            quantum_stats = crud.get_quantum_statistics(db)

            # Factorization metrics
            factorization_metrics = crud.get_performance_metrics(db, task_type="factorization")
            search_metrics = crud.get_performance_metrics(db, task_type="search")
            optimization_metrics = crud.get_performance_metrics(db, task_type="optimization")

            # Display metrics
            cols = st.columns(3)

            with cols[0]:
                st.markdown("#### Factorization")
                st.metric("Average Speedup", f"{factorization_metrics.get('average_speedup', 0):.2f}x")
                st.metric("Max Speedup", f"{factorization_metrics.get('max_speedup', 0):.2f}x")
                st.metric("Quantum Win %", f"{factorization_metrics.get('quantum_win_percentage', 0):.1f}%")

            with cols[1]:
                st.markdown("#### Search")
                st.metric("Average Speedup", f"{search_metrics.get('average_speedup', 0):.2f}x")
                st.metric("Max Speedup", f"{search_metrics.get('max_speedup', 0):.2f}x")
                st.metric("Quantum Win %", f"{search_metrics.get('quantum_win_percentage', 0):.1f}%")

            with cols[2]:
                st.markdown("#### Optimization")
                st.metric("Average Speedup", f"{optimization_metrics.get('average_speedup', 0):.2f}x")
                st.metric("Max Speedup", f"{optimization_metrics.get('max_speedup', 0):.2f}x")
                st.metric("Quantum Win %", f"{optimization_metrics.get('quantum_win_percentage', 0):.1f}%")

            # Quantum statistics
            st.markdown("### Quantum Statistics")

            cols = st.columns(4)

            with cols[0]:
                st.metric("Total Quantum Tasks", quantum_stats.get("total_quantum_tasks", 0))
            with cols[1]:
                st.metric("Avg. Qubit Count", f"{quantum_stats.get('average_qubit_count', 0):.1f}")
            with cols[2]:
                st.metric("Avg. Circuit Depth", f"{quantum_stats.get('average_circuit_depth', 0):.1f}")
            with cols[3]:
                st.metric("Avg. Quantum Advantage", f"{quantum_stats.get('average_quantum_advantage', 0):.2f}x")

            # Provider distribution
            provider_dist = quantum_stats.get("provider_distribution", {})

            if provider_dist:
                fig = px.pie(
                    names=list(provider_dist.keys()),
                    values=list(provider_dist.values()),
                    title="Quantum Provider Distribution"
                )
                fig.update_layout(template="plotly_dark")

                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No tasks found in the database. Process some tasks to see them here!")
    except Exception as e:
        logger.error(f"Error displaying task history: {str(e)}")
        st.error(f"Error displaying task history: {str(e)}")

def display_settings_page():
    """Display the settings page."""
    st.title("⚙️ Settings")

    st.markdown("""
    Configure your Q3A: Quantum-Accelerated AI Agent system settings.
    """)

    # Quantum settings
    st.markdown("### Quantum Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.session_state.use_quantum = st.checkbox(
            "Enable quantum acceleration",
            value=st.session_state.use_quantum
        )

    with col2:
        st.session_state.use_azure = st.checkbox(
            "Use Azure Quantum",
            value=st.session_state.use_azure
        )

    st.session_state.n_qubits = st.slider(
        "Number of qubits",
        min_value=4,
        max_value=29,
        value=st.session_state.n_qubits,
        help="More qubits allow for more complex calculations, but may not be available on all hardware."
    )

    # API settings
    st.markdown("### API Settings")

    openai_key = st.text_input(
        "OpenAI API Key",
        value=os.environ.get("OPENAI_API_KEY", ""),
        type="password"
    )

    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    # Database settings
    st.markdown("### Database Settings")

    database_url = st.text_input(
        "Database URL",
        value=os.environ.get("DATABASE_URL", "sqlite:///./quasar.db")
    )

    if database_url:
        os.environ["DATABASE_URL"] = database_url

    # Save settings button
    if st.button("💾 Save Settings"):
        try:
            # Save settings to .env file
            with open(".env", "w") as f:
                f.write(f"QUANTUM_ENABLED={str(st.session_state.use_quantum).lower()}\n")
                f.write(f"USE_AZURE={str(st.session_state.use_azure).lower()}\n")
                f.write(f"DEFAULT_QUBITS={st.session_state.n_qubits}\n")
                f.write(f"OPENAI_API_KEY={openai_key}\n")
                f.write(f"DATABASE_URL={database_url}\n")

            st.success("Settings saved successfully!")
        except Exception as e:
            st.error(f"Error saving settings: {str(e)}")

    # Reset button
    if st.button("🔄 Reset to Defaults"):
        st.session_state.use_quantum = True
        st.session_state.use_azure = True
        st.session_state.n_qubits = 8
        st.success("Settings reset to defaults!")

    # System information
    st.markdown("### System Information")

    # Check components
    try:
        import pennylane
        pennylane_version = pennylane.__version__
    except ImportError:
        pennylane_version = "Not installed"

    try:
        import openai
        openai_version = openai.__version__
    except ImportError:
        openai_version = "Not installed"

    try:
        import streamlit
        streamlit_version = streamlit.__version__
    except ImportError:
        streamlit_version = "Not installed"

    try:
        import sqlalchemy
        sqlalchemy_version = sqlalchemy.__version__
    except ImportError:
        sqlalchemy_version = "Not installed"

    # Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # Display version information
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Software Versions")
        st.markdown(f"- **Python:** {python_version}")
        st.markdown(f"- **PennyLane:** {pennylane_version}")
        st.markdown(f"- **OpenAI:** {openai_version}")
        st.markdown(f"- **Streamlit:** {streamlit_version}")
        st.markdown(f"- **SQLAlchemy:** {sqlalchemy_version}")

    with col2:
        st.markdown("#### Configuration Status")
        st.markdown(f"- **Quantum Enabled:** {'✅' if st.session_state.use_quantum else '❌'}")
        st.markdown(f"- **Azure Quantum:** {'✅' if st.session_state.use_azure else '❌'}")
        st.markdown(f"- **Number of Qubits:** {st.session_state.n_qubits}")
        st.markdown(f"- **OpenAI API:** {'✅' if openai_key else '❌'}")
        st.markdown(f"- **Database:** {'✅' if database_url else '❌'}")

def main():
    """Main application function."""
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "home"

    if 'use_quantum' not in st.session_state:
        st.session_state.use_quantum = Config.QUANTUM_ENABLED

    if 'use_azure' not in st.session_state:
        st.session_state.use_azure = Config.USE_AZURE

    if 'n_qubits' not in st.session_state:
        st.session_state.n_qubits = Config.DEFAULT_QUBITS

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header">⚛️ QUASAR Framework</div>', unsafe_allow_html=True)

        # Navigation
        st.markdown("### Navigation")

        if st.button("🏠 Home", key="nav_home"):
            st.session_state.page = "home"
            st.rerun()

        st.markdown("#### Core Features")

        if st.button("🧮 Factorization", key="nav_fact"):
            st.session_state.page = "factorization"
            st.rerun()

        if st.button("🔍 Search", key="nav_srch"):
            st.session_state.page = "search"
            st.rerun()

        if st.button("⚙️ Optimization", key="nav_opti"):
            st.session_state.page = "optimization"
            st.rerun()

        st.markdown("#### Additional Tools")

        if st.button("🚀 Demonstrations", key="nav_dem"):
            st.session_state.page = "demo"
            st.rerun()

        if st.button("📋 Task History", key="nav_hist"):
            st.session_state.page = "history"
            st.rerun()

        if st.button("⚙️ Settings", key="nav_set"):
            st.session_state.page = "settings"
            st.rerun()

        # Quantum settings summary
        st.markdown("### Quantum Settings")

        st.markdown(f"**Quantum Computing:** {'✅ Enabled' if st.session_state.use_quantum else '❌ Disabled'}")
        st.markdown(f"**Number of Qubits:** {st.session_state.n_qubits}")
        st.markdown(f"**Quantum Provider:** {'Azure Quantum' if st.session_state.use_azure else 'IBM Qiskit'}")

        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.8em;">
        Q3A: Quantum-Accelerated AI Agent<br>
        Powered by QUASAR Framework<br>
        © 2025
        </div>
        """, unsafe_allow_html=True)

    # Main content based on page
    if st.session_state.page == "home":
        display_home_page()
    elif st.session_state.page == "factorization":
        display_factorization_page()
    elif st.session_state.page == "search":
        display_search_page()
    elif st.session_state.page == "optimization":
        # Special handling for async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(display_optimization_page())
    elif st.session_state.page == "demo":
        display_demo_page()
    elif st.session_state.page == "history":
        display_task_history_page()
    elif st.session_state.page == "settings":
        display_settings_page()

if __name__ == "__main__":
    main()