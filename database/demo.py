"""Demonstration module for the QUASAR framework."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
import random
from datetime import datetime

from quantum_agent_framework.integration import HybridComputation
from quantum_agent_framework.agents.web_agent import QuantumWebAgent
from quantum_agent_framework.quantum.optimizer import QuantumOptimizer
from quantum_agent_framework.agents.quantum_agent import QuantumRLAgent
from components.visualization import (create_comparison_chart,
                                      create_circuit_visualization,
                                      create_factorization_tree,
                                      create_3d_quantum_state,
                                      create_execution_time_comparison)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quasar-demo")


class QuasarDemo:
    """Demonstration class for the QUASAR framework."""

    def __init__(self, n_qubits: int = 8, use_quantum: bool = True):
        """Initialize the demonstration."""
        self.n_qubits = n_qubits
        self.use_quantum = use_quantum

        # Initialize components
        self.hybrid_comp = HybridComputation(n_qubits=n_qubits,
                                             use_quantum=use_quantum)

        # Initialize web agent for search demos
        self.web_agent = QuantumWebAgent(n_qubits=n_qubits,
                                         use_quantum=use_quantum)

        # Initialize optimizer for optimization demos
        if hasattr(self.hybrid_comp, 'quantum_optimizer'):
            self.optimizer = self.hybrid_comp.quantum_optimizer
        else:
            self.optimizer = QuantumOptimizer(n_qubits=n_qubits,
                                              use_azure=False)

        # Store results
        self.results = {}

    # Factorization demonstration
    async def demonstrate_factorization(self,
                                        numbers: List[int]) -> Dict[str, Any]:
        """
        Demonstrate factorization with quantum and classical methods.

        Args:
            numbers: List of numbers to factorize

        Returns:
            Dict with demonstration results
        """
        results = []

        for number in numbers:
            # Process with quantum acceleration
            quantum_start = time.time()
            quantum_result = await self.hybrid_comp.process_task(
                f"Factor {number}")
            quantum_time = time.time() - quantum_start

            # Save quantum result
            q_result = {
                "number":
                number,
                "factors":
                quantum_result.get("factors", []),
                "method":
                quantum_result.get("method_used", "unknown"),
                "backend":
                quantum_result.get("backend", "Unknown"),
                "execution_time":
                quantum_time,
                "circuit_depth":
                quantum_result.get("details", {}).get("circuit_depth", 0),
                "speedup":
                quantum_result.get("details", {}).get("quantum_advantage",
                                                      "Unknown")
            }

            # Process with classical methods
            # Disable quantum to force classical processing
            use_quantum_backup = self.hybrid_comp.use_quantum
            self.hybrid_comp.use_quantum = False

            classical_start = time.time()
            classical_result = await self.hybrid_comp.process_task(
                f"Factor {number}")
            classical_time = time.time() - classical_start

            # Restore quantum setting
            self.hybrid_comp.use_quantum = use_quantum_backup

            # Save classical result
            c_result = {
                "number": number,
                "factors": classical_result.get("factors", []),
                "method": classical_result.get("method_used", "unknown"),
                "execution_time": classical_time
            }

            # Calculate speedup
            speedup = classical_time / quantum_time if quantum_time > 0 else 0

            # Save combined result
            results.append({
                "number": number,
                "quantum_result": q_result,
                "classical_result": c_result,
                "speedup": speedup,
                "factors":
                q_result["factors"]  # Should be the same for both methods
            })

        # Save overall results
        self.results["factorization"] = results

        return {
            "results":
            results,
            "quantum_times":
            [r["quantum_result"]["execution_time"] for r in results],
            "classical_times":
            [r["classical_result"]["execution_time"] for r in results],
            "speedups": [r["speedup"] for r in results],
            "numbers":
            numbers
        }

    # Search demonstration
    async def demonstrate_search(self, queries: List[str]) -> Dict[str, Any]:
        """
        Demonstrate quantum-enhanced search.

        Args:
            queries: List of search queries

        Returns:
            Dict with demonstration results
        """
        results = []

        for query in queries:
            # Search with quantum acceleration
            quantum_start = time.time()
            quantum_result = await self.web_agent.search(query,
                                                         use_quantum=True)
            quantum_time = time.time() - quantum_start

            # Search with classical methods
            classical_start = time.time()
            classical_result = await self.web_agent.search(query,
                                                           use_quantum=False)
            classical_time = time.time() - classical_start

            # Calculate speedup
            speedup = classical_time / quantum_time if quantum_time > 0 else 0

            # Calculate result similarity (how many of the same URLs appear in both result sets)
            q_urls = [
                r.get("url", "") for r in quantum_result.get("results", [])
            ]
            c_urls = [
                r.get("url", "") for r in classical_result.get("results", [])
            ]

            common_urls = set(q_urls).intersection(set(c_urls))
            similarity = len(common_urls) / max(
                len(q_urls), len(c_urls)) if max(len(q_urls),
                                                 len(c_urls)) > 0 else 0

            # Save results
            results.append({
                "query": query,
                "quantum_result": {
                    "time": quantum_time,
                    "results_count": len(quantum_result.get("results", [])),
                    "top_results": quantum_result.get("results", [])[:3],
                    "quantum_metrics":
                    quantum_result.get("quantum_metrics", {})
                },
                "classical_result": {
                    "time": classical_time,
                    "results_count": len(classical_result.get("results", [])),
                    "top_results": classical_result.get("results", [])[:3]
                },
                "speedup": speedup,
                "result_similarity": similarity
            })

        # Save overall results
        self.results["search"] = results

        return {
            "results": results,
            "quantum_times": [r["quantum_result"]["time"] for r in results],
            "classical_times":
            [r["classical_result"]["time"] for r in results],
            "speedups": [r["speedup"] for r in results],
            "result_similarities": [r["result_similarity"] for r in results],
            "queries": queries
        }

    # Optimization demonstration
    async def demonstrate_optimization(
            self,
            resource_counts: List[int],
            constraint_counts: List[int] = None) -> Dict[str, Any]:
        """
        Demonstrate quantum optimization.

        Args:
            resource_counts: List of resource counts to optimize
            constraint_counts: Optional list of constraint counts

        Returns:
            Dict with demonstration results
        """
        if constraint_counts is None:
            constraint_counts = [max(1, rc // 3) for rc in resource_counts]

        results = []

        for rc, cc in zip(resource_counts, constraint_counts):
            # Generate a random resource allocation problem
            resources = self._generate_optimization_problem(rc, cc)

            # Optimize with quantum
            quantum_start = time.time()
            quantum_result = self.optimizer.optimize_resource_allocation(
                resources)
            quantum_time = time.time() - quantum_start

            # Optimize with classical methods
            classical_start = time.time()
            classical_result = await self.hybrid_comp._classical_optimization(
                resources)
            classical_time = time.time() - classical_start

            # Calculate speedup
            speedup = classical_time / quantum_time if quantum_time > 0 else 0

            # Compare solution quality
            q_objective = quantum_result.get("objective_value", 0)
            c_objective = classical_result.get("objective_value", 0)

            # Higher is better
            quality_diff = (q_objective - c_objective) / max(
                q_objective, c_objective) if max(q_objective,
                                                 c_objective) > 0 else 0

            # Save results
            results.append({
                "resource_count": rc,
                "constraint_count": cc,
                "quantum_result": {
                    "time": quantum_time,
                    "allocation": quantum_result.get("allocation", []),
                    "objective_value": q_objective,
                    "method": quantum_result.get("method_used", "unknown")
                },
                "classical_result": {
                    "time": classical_time,
                    "allocation": classical_result.get("allocation", []),
                    "objective_value": c_objective,
                    "method": classical_result.get("method_used", "unknown")
                },
                "speedup": speedup,
                "quality_improvement": quality_diff
            })

        # Save overall results
        self.results["optimization"] = results

        return {
            "results": results,
            "quantum_times": [r["quantum_result"]["time"] for r in results],
            "classical_times":
            [r["classical_result"]["time"] for r in results],
            "speedups": [r["speedup"] for r in results],
            "quality_improvements":
            [r["quality_improvement"] for r in results],
            "resource_counts": resource_counts
        }

    # Reinforcement learning demonstration
    async def demonstrate_reinforcement_learning(
            self,
            episodes: int = 50,
            env_name: str = "CartPole-v1") -> Dict[str, Any]:
        """
        Demonstrate quantum-enhanced reinforcement learning.

        Args:
            episodes: Number of episodes to train
            env_name: Name of the environment

        Returns:
            Dict with demonstration results
        """
        try:
            import gymnasium as gym
        except ImportError:
            return {
                "error":
                "Gymnasium not installed. Please install with 'pip install gymnasium'"
            }

        # Create environment
        env = gym.make(env_name)

        # Get state and action dimensions
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n if hasattr(
            env.action_space, 'n') else env.action_space.shape[0]

        # Create quantum RL agent
        quantum_agent = QuantumRLAgent(state_dim=state_dim,
                                       action_dim=action_dim,
                                       n_qubits=self.n_qubits,
                                       use_quantum=self.use_quantum)

        # Create classical RL agent
        classical_agent = QuantumRLAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            use_quantum=False  # Force classical processing
        )

        # Train both agents
        quantum_rewards = []
        classical_rewards = []
        quantum_times = []
        classical_times = []

        for episode in range(episodes):
            # Train quantum agent
            quantum_start = time.time()
            quantum_result = await asyncio.to_thread(
                quantum_agent.train_episode, env)
            quantum_time = time.time() - quantum_start

            # Train classical agent
            classical_start = time.time()
            classical_result = await asyncio.to_thread(
                classical_agent.train_episode, env)
            classical_time = time.time() - classical_start

            # Record results
            quantum_rewards.append(quantum_result["total_reward"])
            classical_rewards.append(classical_result["total_reward"])
            quantum_times.append(quantum_time)
            classical_times.append(classical_time)

        # Calculate average speedup
        avg_speedup = sum(classical_times) / sum(quantum_times) if sum(
            quantum_times) > 0 else 0

        # Calculate sample efficiency (episodes to reach 90% of max reward)
        quantum_efficiency = self._calculate_sample_efficiency(quantum_rewards)
        classical_efficiency = self._calculate_sample_efficiency(
            classical_rewards)

        efficiency_improvement = (
            classical_efficiency - quantum_efficiency
        ) / classical_efficiency if classical_efficiency > 0 else 0

        # Get final performance metrics
        quantum_metrics = quantum_agent.get_performance_metrics()
        classical_metrics = classical_agent.get_performance_metrics()

        # Save results
        results = {
            "env_name": env_name,
            "episodes": episodes,
            "quantum_rewards": quantum_rewards,
            "classical_rewards": classical_rewards,
            "quantum_times": quantum_times,
            "classical_times": classical_times,
            "avg_speedup": avg_speedup,
            "efficiency_improvement": efficiency_improvement,
            "quantum_metrics": quantum_metrics,
            "classical_metrics": classical_metrics
        }

        self.results["reinforcement_learning"] = results

        return results

    # Helper methods
    def _generate_optimization_problem(
            self, resource_count: int,
            constraint_count: int) -> Dict[str, Any]:
        """Generate a random resource allocation problem."""
        # Generate random weights for resources
        weights = [random.uniform(0.5, 10.0) for _ in range(resource_count)]

        # Generate random values for resources
        values = [random.uniform(1.0, 20.0) for _ in range(resource_count)]

        # Generate random constraints
        constraints = []
        for _ in range(constraint_count):
            # Random subset of resources involved in this constraint
            involved_resources = random.sample(
                range(resource_count), min(random.randint(2, 5),
                                           resource_count))

            # Random constraint type (sum, max, min)
            constraint_type = random.choice(["sum", "max"])

            # Random limit value
            limit = sum([weights[i] for i in involved_resources
                         ]) * random.uniform(0.4, 0.8)

            constraints.append({
                "resources": involved_resources,
                "type": constraint_type,
                "limit": limit
            })

        return {
            "items": list(range(resource_count)),
            "weights": weights,
            "values": values,
            "constraints": constraints,
            "objective": "maximize"
        }

    def _calculate_sample_efficiency(self, rewards: List[float]) -> int:
        """Calculate sample efficiency (episodes to reach 90% of max reward)."""
        if not rewards:
            return 0

        max_reward = max(rewards)
        target = 0.9 * max_reward

        for i, reward in enumerate(rewards):
            if reward >= target:
                return i + 1

        return len(rewards)

    # Streamlit demonstration UI
    def run_streamlit_demo(self):
        """Run the demonstration in Streamlit."""
        st.title("Q3A: Quantum Acceleration Demonstration")

        st.markdown("""
        This demonstration showcases the quantum advantage provided by the QUASAR framework
        across different types of tasks:

        1. **Factorization**: Quantum computing provides exponential speedup
        2. **Search**: Quantum computing offers quadratic speedup
        3. **Optimization**: Quantum algorithms provide polynomial speedup
        4. **Reinforcement Learning**: Quantum-enhanced RL shows improved sample efficiency
        """)

        # Create tabs
        tabs = st.tabs([
            "🧮 Factorization", "🔍 Search", "⚙️ Optimization",
            "🤖 Reinforcement Learning"
        ])

        # Factorization demo
        with tabs[0]:
            st.header("Quantum-Accelerated Factorization")

            st.markdown("""
            Factorization is where quantum computing shows its most dramatic advantage.
            Shor's algorithm provides **exponential speedup** over classical methods.

            Let's compare quantum vs. classical factorization for different numbers:
            """)

            # Input
            col1, col2 = st.columns(2)
            with col1:
                custom_number = st.number_input("Enter a number to factorize",
                                                min_value=4,
                                                max_value=1000000,
                                                value=3960)

            with col2:
                run_demo = st.button("🚀 Run Factorization Demo",
                                     key="factorization_demo")

            if run_demo:
                with st.spinner("Running factorization demonstration..."):
                    # Run demo with some standard numbers plus custom number
                    numbers = [15, 21, 35, 57, custom_number]
                    numbers = sorted(list(
                        set(numbers)))  # Remove duplicates and sort

                    # Run demo
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    results = loop.run_until_complete(
                        self.demonstrate_factorization(numbers))

                    # Show results
                    st.subheader("Results")

                    # Create comparison chart
                    fig = create_execution_time_comparison(
                        results["quantum_times"], results["classical_times"],
                        [str(n) for n in results["numbers"]])
                    st.plotly_chart(fig, use_container_width=True)

                    # Show detailed results
                    for result in results["results"]:
                        with st.expander(
                                f"Factorization of {result['number']}"):
                            cols = st.columns(2)

                            # Quantum results
                            with cols[0]:
                                st.markdown("#### Quantum Method")
                                st.markdown(
                                    f"**Method:** {result['quantum_result']['method']}"
                                )
                                st.markdown(
                                    f"**Backend:** {result['quantum_result']['backend']}"
                                )
                                st.markdown(
                                    f"**Time:** {result['quantum_result']['execution_time']:.6f} seconds"
                                )
                                st.markdown(
                                    f"**Circuit Depth:** {result['quantum_result']['circuit_depth']}"
                                )

                            # Classical results
                            with cols[1]:
                                st.markdown("#### Classical Method")
                                st.markdown(
                                    f"**Method:** {result['classical_result']['method']}"
                                )
                                st.markdown(
                                    f"**Time:** {result['classical_result']['execution_time']:.6f} seconds"
                                )
                                st.markdown(
                                    f"**Speedup:** {result['speedup']:.2f}x")

                            # Show factors
                            st.markdown("#### Factors")
                            st.markdown(
                                f"{', '.join(map(str, result['factors']))}")

                            # Show factor tree
                            if len(result['factors']) > 2:
                                try:
                                    factor_tree = create_factorization_tree(
                                        result['factors'], result['number'])
                                    st.plotly_chart(factor_tree,
                                                    use_container_width=True)
                                except Exception as e:
                                    st.error(
                                        f"Could not create factor tree: {str(e)}"
                                    )

                    # Show quantum circuit visualization for the last result
                    if results["results"]:
                        st.subheader("Quantum Circuit Visualization")
                        circuit_params = np.random.uniform(
                            -np.pi, np.pi, (3, self.n_qubits, 3))
                        circuit_fig = create_circuit_visualization(
                            circuit_params)
                        st.plotly_chart(circuit_fig, use_container_width=True)

        # Search demo
        with tabs[1]:
            st.header("Quantum-Enhanced Search")

            st.markdown("""
            Quantum search algorithms like Grover's algorithm provide a **quadratic speedup**
            over classical search algorithms. For a database of size N, classical algorithms
            require O(N) operations, while quantum algorithms need only O(√N) operations.

            Let's compare quantum vs. classical search performance:
            """)

            # Input
            col1, col2 = st.columns(2)
            with col1:
                search_query = st.text_input(
                    "Enter a search query",
                    value="quantum computing applications")

            with col2:
                run_search_demo = st.button("🚀 Run Search Demo",
                                            key="search_demo")

            if run_search_demo:
                with st.spinner("Running search demonstration..."):
                    # Run demo with some standard queries plus custom query
                    queries = [
                        "quantum computing applications", "shor's algorithm",
                        "quantum machine learning", search_query
                    ]
                    queries = list(dict.fromkeys(
                        queries))  # Remove duplicates while preserving order

                    # Run demo
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    results = loop.run_until_complete(
                        self.demonstrate_search(queries))

                    # Show results
                    st.subheader("Results")

                    # Create comparison chart
                    fig = create_execution_time_comparison(
                        results["quantum_times"], results["classical_times"],
                        results["queries"])
                    st.plotly_chart(fig, use_container_width=True)

                    # Show result similarities
                    similarity_fig = px.bar(
                        x=results["queries"],
                        y=results["result_similarities"],
                        labels={
                            "x": "Query",
                            "y": "Result Similarity"
                        },
                        title=
                        "Result Similarity Between Quantum and Classical Search",
                        color=results["result_similarities"],
                        color_continuous_scale="Viridis")
                    similarity_fig.update_layout(
                        yaxis_title="Similarity (higher is better)",
                        template="plotly_dark")
                    st.plotly_chart(similarity_fig, use_container_width=True)

                    # Show detailed results
                    for result in results["results"]:
                        with st.expander(
                                f"Search results for '{result['query']}'"):
                            cols = st.columns(2)

                            # Quantum results
                            with cols[0]:
                                st.markdown("#### Quantum Search")
                                st.markdown(
                                    f"**Time:** {result['quantum_result']['time']:.6f} seconds"
                                )
                                st.markdown(
                                    f"**Results Count:** {result['quantum_result']['results_count']}"
                                )

                                # Show top results
                                st.markdown("**Top Results:**")
                                for i, res in enumerate(
                                        result['quantum_result']
                                    ['top_results']):
                                    st.markdown(
                                        f"{i+1}. **{res.get('title', 'Unknown')}**  \n"
                                        f"URL: {res.get('url', 'Unknown')}  \n"
                                        f"Score: {res.get('quantum_score', 0):.4f}"
                                    )

                            # Classical results
                            with cols[1]:
                                st.markdown("#### Classical Search")
                                st.markdown(
                                    f"**Time:** {result['classical_result']['time']:.6f} seconds"
                                )
                                st.markdown(
                                    f"**Results Count:** {result['classical_result']['results_count']}"
                                )
                                st.markdown(
                                    f"**Speedup:** {result['speedup']:.2f}x")

                                # Show top results
                                st.markdown("**Top Results:**")
                                for i, res in enumerate(
                                        result['classical_result']
                                    ['top_results']):
                                    st.markdown(
                                        f"{i+1}. **{res.get('title', 'Unknown')}**  \n"
                                        f"URL: {res.get('url', 'Unknown')}  \n"
                                        f"Score: {res.get('classical_score', 0):.4f}"
                                    )

                    # Show quantum circuit visualization for search
                    st.subheader("Quantum Search Circuit")
                    st.markdown("""
                    This visualization shows a simplified representation of Grover's algorithm
                    quantum circuit for search acceleration.
                    """)
                    circuit_params = np.random.uniform(-np.pi, np.pi,
                                                       (2, self.n_qubits, 3))
                    circuit_fig = create_circuit_visualization(circuit_params)
                    st.plotly_chart(circuit_fig, use_container_width=True)

        # Optimization demo
        with tabs[2]:
            st.header("Quantum Optimization")

            st.markdown("""
            Quantum optimization algorithms like QAOA (Quantum Approximate Optimization Algorithm)
            can provide significant speedups for complex optimization problems, particularly
            those with many constraints or variables.

            Let's compare quantum vs. classical optimization performance:
            """)

            # Input
            col1, col2 = st.columns(2)
            with col1:
                resource_count = st.slider("Number of resources to optimize",
                                           min_value=5,
                                           max_value=50,
                                           value=20)
                constraint_count = st.slider("Number of constraints",
                                             min_value=1,
                                             max_value=10,
                                             value=3)

            with col2:
                run_opt_demo = st.button("🚀 Run Optimization Demo",
                                         key="opt_demo")

            if run_opt_demo:
                with st.spinner("Running optimization demonstration..."):
                    # Run demo with different resource counts
                    resource_counts = [5, 10, 20, resource_count]
                    resource_counts = sorted(list(
                        set(resource_counts)))  # Remove duplicates and sort

                    constraint_counts = [
                        max(1, rc // 5) for rc in resource_counts
                    ]
                    constraint_counts[
                        -1] = constraint_count  # Use user-specified constraint count for the last one

                    # Run demo
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    results = loop.run_until_complete(
                        self.demonstrate_optimization(resource_counts,
                                                      constraint_counts))

                    # Show results
                    st.subheader("Results")

                    # Create comparison chart
                    fig = create_execution_time_comparison(
                        results["quantum_times"], results["classical_times"], [
                            f"{rc} resources, {cc} constraints" for rc, cc in
                            zip(results["resource_counts"], constraint_counts)
                        ])
                    st.plotly_chart(fig, use_container_width=True)

                    # Show quality improvements
                    quality_fig = px.bar(
                        x=[
                            f"{rc} resources"
                            for rc in results["resource_counts"]
                        ],
                        y=results["quality_improvements"],
                        labels={
                            "x": "Problem Size",
                            "y": "Solution Quality Improvement"
                        },
                        title="Quantum vs. Classical Solution Quality",
                        color=results["quality_improvements"],
                        color_continuous_scale="Viridis")
                    quality_fig.update_layout(
                        yaxis_title="Quality Improvement (%)",
                        template="plotly_dark")
                    st.plotly_chart(quality_fig, use_container_width=True)

                    # Show detailed results
                    for i, result in enumerate(results["results"]):
                        with st.expander(
                                f"{result['resource_count']} resources, {result['constraint_count']} constraints"
                        ):
                            cols = st.columns(2)

                            # Quantum results
                            with cols[0]:
                                st.markdown("#### Quantum Optimization")
                                st.markdown(
                                    f"**Method:** {result['quantum_result']['method']}"
                                )
                                st.markdown(
                                    f"**Time:** {result['quantum_result']['time']:.6f} seconds"
                                )
                                st.markdown(
                                    f"**Objective Value:** {result['quantum_result']['objective_value']:.4f}"
                                )

                                # Show allocation
                                st.markdown("**Resource Allocation:**")
                                if result['quantum_result']['allocation']:
                                    # Create allocation visualization
                                    allocation = result['quantum_result'][
                                        'allocation']
                                    fig = px.bar(
                                        x=[
                                            f"Resource {i}"
                                            for i in range(len(allocation))
                                        ],
                                        y=allocation,
                                        labels={
                                            "x": "Resource",
                                            "y": "Allocation Value"
                                        },
                                        title="Quantum Resource Allocation",
                                        color=allocation,
                                        color_continuous_scale="Viridis")
                                    st.plotly_chart(fig,
                                                    use_container_width=True)

                            # Classical results
                            with cols[1]:
                                st.markdown("#### Classical Optimization")
                                st.markdown(
                                    f"**Method:** {result['classical_result']['method']}"
                                )
                                st.markdown(
                                    f"**Time:** {result['classical_result']['time']:.6f} seconds"
                                )
                                st.markdown(
                                    f"**Objective Value:** {result['classical_result']['objective_value']:.4f}"
                                )
                                st.markdown(
                                    f"**Speedup:** {result['speedup']:.2f}x")

                                # Show allocation
                                st.markdown("**Resource Allocation:**")
                                if result['classical_result']['allocation']:
                                    # Create allocation visualization
                                    allocation = result['classical_result'][
                                        'allocation']
                                    fig = px.bar(
                                        x=[
                                            f"Resource {i}"
                                            for i in range(len(allocation))
                                        ],
                                        y=allocation,
                                        labels={
                                            "x": "Resource",
                                            "y": "Allocation Value"
                                        },
                                        title="Classical Resource Allocation",
                                        color=allocation,
                                        color_continuous_scale="Blues")
                                    st.plotly_chart(fig,
                                                    use_container_width=True)

                    # Show QAOA circuit
                    st.subheader("QAOA Circuit Visualization")
                    st.markdown("""
                    This visualization shows a simplified representation of the Quantum Approximate
                    Optimization Algorithm (QAOA) circuit used for quantum optimization.
                    """)
                    circuit_params = np.random.uniform(-np.pi, np.pi,
                                                       (4, self.n_qubits, 3))
                    circuit_fig = create_circuit_visualization(circuit_params)
                    st.plotly_chart(circuit_fig, use_container_width=True)

        # Reinforcement learning demo
        with tabs[3]:
            st.header("Quantum-Enhanced Reinforcement Learning")

            st.markdown("""
            Quantum-enhanced reinforcement learning can provide benefits in:
            1. **Sample Efficiency**: Learning from fewer experiences
            2. **Training Speed**: Faster convergence to optimal policies
            3. **Exploration Efficiency**: Better exploration of complex state spaces

            Let's compare quantum vs. classical RL performance:
            """)

            # Input
            col1, col2 = st.columns(2)
            with col1:
                episode_count = st.slider("Number of training episodes",
                                          min_value=10,
                                          max_value=100,
                                          value=50)
                environment = st.selectbox(
                    "Environment",
                    ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"],
                    index=0)

            with col2:
                run_rl_demo = st.button("🚀 Run RL Demo", key="rl_demo")

            if run_rl_demo:
                try:
                    with st.spinner(
                            "Running reinforcement learning demonstration..."):
                        # Run demo
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        results = loop.run_until_complete(
                            self.demonstrate_reinforcement_learning(
                                episodes=episode_count, env_name=environment))

                        if "error" in results:
                            st.error(results["error"])
                        else:
                            # Show results
                            st.subheader("Results")

                            # Create reward comparison chart
                            reward_fig = go.Figure()

                            # Add quantum rewards
                            reward_fig.add_trace(
                                go.Scatter(y=results["quantum_rewards"],
                                           name='Quantum RL Agent',
                                           line=dict(color='#7b2cbf', width=2),
                                           mode='lines+markers'))

                            # Add classical rewards
                            reward_fig.add_trace(
                                go.Scatter(y=results["classical_rewards"],
                                           name='Classical RL Agent',
                                           line=dict(color='#2cb5e8', width=2),
                                           mode='lines+markers'))

                            # Update layout
                            reward_fig.update_layout(
                                title=f'Reward Comparison ({environment})',
                                xaxis_title='Episode',
                                yaxis_title='Reward',
                                template='plotly_dark',
                                hovermode='x unified')

                            st.plotly_chart(reward_fig,
                                            use_container_width=True)

                            # Create time comparison chart
                            time_fig = go.Figure()

                            # Add quantum times
                            time_fig.add_trace(
                                go.Scatter(y=results["quantum_times"],
                                           name='Quantum RL Agent',
                                           line=dict(color='#7b2cbf', width=2),
                                           mode='lines+markers'))

                            # Add classical times
                            time_fig.add_trace(
                                go.Scatter(y=results["classical_times"],
                                           name='Classical RL Agent',
                                           line=dict(color='#2cb5e8', width=2),
                                           mode='lines+markers'))

                            # Update layout
                            time_fig.update_layout(
                                title='Computation Time Comparison',
                                xaxis_title='Episode',
                                yaxis_title='Time (seconds)',
                                template='plotly_dark',
                                hovermode='x unified')

                            st.plotly_chart(time_fig, use_container_width=True)

                            # Show metrics
                            st.subheader("Performance Metrics")

                            metrics_cols = st.columns(2)

                            with metrics_cols[0]:
                                st.markdown("#### Quantum RL Agent")
                                st.markdown(
                                    f"**Average Reward:** {np.mean(results['quantum_rewards']):.2f}"
                                )
                                st.markdown(
                                    f"**Max Reward:** {np.max(results['quantum_rewards']):.2f}"
                                )
                                st.markdown(
                                    f"**Average Time per Episode:** {np.mean(results['quantum_times']):.6f} seconds"
                                )

                                # Show quantum advantage metrics
                                st.markdown("#### Quantum Advantage")
                                st.markdown(
                                    f"**Time Speedup:** {results['avg_speedup']:.2f}x"
                                )
                                st.markdown(
                                    f"**Sample Efficiency Improvement:** {results['efficiency_improvement']*100:.2f}%"
                                )

                                # Get additional metrics from the quantum agent
                                for key, value in results[
                                        "quantum_metrics"].get(
                                            "quantum_advantage", {}).items():
                                    st.markdown(
                                        f"**{key.replace('_', ' ').title()}:** {value}"
                                    )

                            with metrics_cols[1]:
                                st.markdown("#### Classical RL Agent")
                                st.markdown(
                                    f"**Average Reward:** {np.mean(results['classical_rewards']):.2f}"
                                )
                                st.markdown(
                                    f"**Max Reward:** {np.max(results['classical_rewards']):.2f}"
                                )
                                st.markdown(
                                    f"**Average Time per Episode:** {np.mean(results['classical_times']):.6f} seconds"
                                )

                            # Show quantum circuit visualization
                            st.subheader("Quantum RL Circuit")
                            st.markdown("""
                            This visualization shows a simplified representation of the variational
                            quantum circuit used by the quantum RL agent for action selection.
                            """)
                            circuit_params = np.random.uniform(
                                -np.pi, np.pi, (3, self.n_qubits, 3))
                            circuit_fig = create_circuit_visualization(
                                circuit_params)
                            st.plotly_chart(circuit_fig,
                                            use_container_width=True)

                            # Show quantum state visualization
                            st.subheader("Quantum State Visualization")
                            state_vector = np.zeros(2**min(4, self.n_qubits))
                            state_vector[0] = 0.7071  # |0⟩ component
                            state_vector[1] = 0.7071  # |1⟩ component

                            state_fig = create_3d_quantum_state(state_vector)
                            st.plotly_chart(state_fig,
                                            use_container_width=True)
                except Exception as e:
                    st.error(f"Error running RL demonstration: {str(e)}")

        # Quantum performance dashboard
        st.header("Quantum Advantage Dashboard")

        if self.results:
            # Create a combined performance dashboard
            st.markdown("""
            This dashboard summarizes the quantum advantage demonstrated across different tasks.
            """)

            # Create summary dataframe
            data = {
                'Task Type': [],
                'Quantum Time (s)': [],
                'Classical Time (s)': [],
                'Speedup': [],
                'Theoretical Speedup': [],
                'Best Use Case': []
            }

            # Add factorization results
            if "factorization" in self.results:
                for result in self.results["factorization"]:
                    data['Task Type'].append(f"Factor {result['number']}")
                    data['Quantum Time (s)'].append(
                        result['quantum_result']['execution_time'])
                    data['Classical Time (s)'].append(
                        result['classical_result']['execution_time'])
                    data['Speedup'].append(result['speedup'])
                    data['Theoretical Speedup'].append("Exponential")
                    data['Best Use Case'].append("Large number factorization")

            # Add search results
            if "search" in self.results:
                for result in self.results["search"]:
                    data['Task Type'].append(
                        f"Search: {result['query'][:20]}...")
                    data['Quantum Time (s)'].append(
                        result['quantum_result']['time'])
                    data['Classical Time (s)'].append(
                        result['classical_result']['time'])
                    data['Speedup'].append(result['speedup'])
                    data['Theoretical Speedup'].append("Quadratic")
                    data['Best Use Case'].append("Large database search")

            # Add optimization results
            if "optimization" in self.results:
                for result in self.results["optimization"]:
                    data['Task Type'].append(
                        f"Optimize {result['resource_count']} resources")
                    data['Quantum Time (s)'].append(
                        result['quantum_result']['time'])
                    data['Classical Time (s)'].append(
                        result['classical_result']['time'])
                    data['Speedup'].append(result['speedup'])
                    data['Theoretical Speedup'].append("Polynomial")
                    data['Best Use Case'].append(
                        "Complex constraint satisfaction")

            # Add reinforcement learning results if available
            if "reinforcement_learning" in self.results:
                rl_results = self.results["reinforcement_learning"]
                data['Task Type'].append(f"RL: {rl_results['env_name']}")
                data['Quantum Time (s)'].append(
                    np.mean(rl_results['quantum_times']))
                data['Classical Time (s)'].append(
                    np.mean(rl_results['classical_times']))
                data['Speedup'].append(rl_results['avg_speedup'])
                data['Theoretical Speedup'].append("Polynomial")
                data['Best Use Case'].append("Complex policy learning")

            # Create dataframe
            df = pd.DataFrame(data)

            # Format numeric columns
            df['Quantum Time (s)'] = df['Quantum Time (s)'].map(
                '{:.6f}'.format)
            df['Classical Time (s)'] = df['Classical Time (s)'].map(
                '{:.6f}'.format)
            df['Speedup'] = df['Speedup'].map('{:.2f}x'.format)

            # Display table
            st.dataframe(df, use_container_width=True)

            # Create speedup visualization
            speedup_data = {
                'Task': data['Task Type'],
                'Speedup': [float(s.replace('x', '')) for s in df['Speedup']],
                'Type':
                [t.split(':')[0].split(' ')[0] for t in data['Task Type']]
            }

            speedup_fig = px.bar(
                speedup_data,
                x='Task',
                y='Speedup',
                color='Type',
                labels={"Speedup": "Speedup Factor (higher is better)"},
                title="Quantum Speedup Across Tasks",
                color_discrete_map={
                    'Factor': '#7b2cbf',
                    'Search': '#2cb5e8',
                    'Optimize': '#ff9800',
                    'RL': '#4caf50'
                })

            speedup_fig.update_layout(xaxis_title="Task",
                                      yaxis_title="Speedup Factor (x)",
                                      template="plotly_dark")

            st.plotly_chart(speedup_fig, use_container_width=True)
        else:
            st.info(
                "Run at least one demonstration to see the performance dashboard."
            )


# Run the demo if executed directly
if __name__ == "__main__":
    # Initialize the demo
    demo = QuasarDemo(n_qubits=8, use_quantum=True)

    # Run the Streamlit demo
    demo.run_streamlit_demo()
