"""
QA³: Quantum-Accelerated AI Agent - Unified Interface

This streamlined interface provides a single entry point for the QA³ agent,
with a simple interface for searching across 20+ sources with quantum acceleration.
"""

import os
import json
import time
import asyncio
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simple-quantum-agent")

# Import Streamlit for UI
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Try to import the enhanced QA³ agent
try:
    from quantum_agent_framework.qa3_agent_enhanced import QA3AgentEnhanced
    ENHANCED_AGENT_AVAILABLE = True
except ImportError:
    ENHANCED_AGENT_AVAILABLE = False
    logger.warning("Enhanced QA³ agent not available. Falling back to standard agent.")

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'agent_initialized' not in st.session_state:
        st.session_state.agent_initialized = False
        
    if 'agent' not in st.session_state:
        st.session_state.agent = None
        
    if 'task_history' not in st.session_state:
        st.session_state.task_history = []
        
    if 'n_qubits' not in st.session_state:
        st.session_state.n_qubits = 8
        
    if 'use_quantum' not in st.session_state:
        st.session_state.use_quantum = True
        
    if 'query' not in st.session_state:
        st.session_state.query = ""
        
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None

async def initialize_agent():
    """Initialize the QA³ agent"""
    if st.session_state.agent_initialized:
        return True
    
    try:
        # Create and initialize the enhanced agent
        st.session_state.agent = QA3AgentEnhanced(
            use_quantum=st.session_state.use_quantum,
            n_qubits=st.session_state.n_qubits
        )
        
        # Initialize agent
        await st.session_state.agent.initialize()
        
        st.session_state.agent_initialized = True
        
        return True
            
    except Exception as e:
        st.error(f"Failed to initialize agent: {str(e)}")
        logger.error(f"Agent initialization failed: {str(e)}", exc_info=True)
        return False

async def process_task(task):
    """Process a user task with the QA³ agent"""
    if not st.session_state.agent_initialized:
        success = await initialize_agent()
        if not success:
            st.error("Agent not initialized")
            return {"success": False, "error": "Agent not initialized"}
    
    try:
        # Process the task with the agent
        with st.spinner(f"Processing: {task}"):
            # Process the task
            result = await st.session_state.agent.process_task(task)
            
            # Record the result in task history
            record = {
                "task": task,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.task_history.append(record)
            
        return result
    except Exception as e:
        error_msg = f"Error processing task: {str(e)}"
        logger.error(f"Exception in process_task: {str(e)}", exc_info=True)
        st.error(error_msg)
        return {"success": False, "error": error_msg}

async def perform_search(query, use_quantum=True):
    """Perform a search using the agent"""
    if not st.session_state.agent_initialized:
        success = await initialize_agent()
        if not success:
            st.error("Agent not initialized")
            return {"success": False, "error": "Agent not initialized"}
    
    if not query:
        st.warning("Please enter a search query")
        return {"success": False, "error": "Empty search query"}
    
    # Format search task
    task = f"Search: {query}"
    
    try:
        # Update quantum setting before search
        st.session_state.agent.use_quantum = use_quantum
        
        # Use the agent to process the search
        with st.spinner(f"Searching for: {query}" + (" (with quantum acceleration)" if use_quantum else "")):
            result = await st.session_state.agent.process_task(task)
            
            # Store search results in session state
            st.session_state.search_results = result
            
        return result
    except Exception as e:
        error_msg = f"Error during search: {str(e)}"
        logger.error(f"Exception in perform_search: {str(e)}", exc_info=True)
        st.error(error_msg)
        return {"success": False, "error": error_msg}

def display_unifiedInterface():
    """Display the unified interface for the QA³ agent"""
    st.title("QA³: Quantum-Accelerated AI Agent")
    
    # Initializing the agent in the background
    if not st.session_state.agent_initialized:
        with st.status("Initializing QA³ agent...", expanded=True) as status:
            asyncio.run(initialize_agent())
            if st.session_state.agent_initialized:
                status.update(label="QA³ agent ready!", state="complete")
    
    # Search interface
    st.header("Quantum-Enhanced Search")
    
    with st.form(key="unified_search_form"):
        query = st.text_input("Enter your search query:", key="unified_query_input")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            use_quantum = st.checkbox("Enable quantum acceleration", value=st.session_state.use_quantum)
        
        with col2:
            submit_button = st.form_submit_button("🔍 Search")
        
        if submit_button and query:
            # Execute search
            result = asyncio.run(perform_search(query, use_quantum))
            
            # Store query for later
            st.session_state.query = query
            
            # Force a rerun to update results display
            st.rerun()
    
    # Display search help
    st.caption("Enter any query to search across 20+ sources including academic papers, news sites, and more.")
    
    # Display search results
    if st.session_state.search_results:
        results = st.session_state.search_results
        
        # Display search information
        if results.get("success", False):
            method = "quantum-enhanced" if results.get("quantum_enhanced", False) else "classical"
            source_count = results.get("source_count", 1)
            result_count = results.get("result_count", 0)
            execution_time = results.get("execution_time", 0)
            speedup = results.get("speedup_factor", 0)
            
            # Create performance metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Results Found", f"{result_count} results")
            with col2:
                st.metric("Sources", f"{source_count} sources")
            with col3:
                if results.get("quantum_enhanced", False) and speedup > 0:
                    st.metric("Quantum Speedup", f"{speedup:.2f}x")
                else:
                    st.metric("Execution Time", f"{execution_time:.3f}s")
            
            # Display comprehensive summary if available
            if "comprehensive_summary" in results:
                st.subheader("Search Results Summary")
                st.markdown(results["comprehensive_summary"])
            
            # Display search results
            if "search_results" in results and results["search_results"]:
                with st.expander("Detailed Results", expanded=False):
                    for i, result in enumerate(results["search_results"]):
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.markdown(f"### {i+1}. {result.get('title', 'Untitled')}")
                            if "url" in result:
                                st.markdown(f"[{result.get('url', '')}]({result.get('url', '')})")
                            if "snippet" in result:
                                st.markdown(result["snippet"])
                            if "source" in result:
                                st.caption(f"Source: {result['source']}")
                        with col2:
                            if "relevance_score" in result:
                                score = result["relevance_score"]
                                st.metric("Relevance", f"{score:.2f}")
                            if "rank_change" in result:
                                change = result["rank_change"]
                                if change != 0:
                                    direction = "↑" if change > 0 else "↓"
                                    st.caption(f"Quantum rerank: {direction} {abs(change)}")
                        st.divider()
        else:
            st.error(f"Search failed: {results.get('error', 'Unknown error')}")

def display_quantum_explanation():
    """Display explanation of quantum acceleration"""
    with st.expander("How quantum acceleration works", expanded=False):
        st.markdown("""
        ### Quantum Acceleration in QA³

        **Quantum-Enhanced Search** uses quantum computing principles to accelerate search and improve result ranking:

        1. **Grover's Algorithm inspired approach**: Quantum search achieves quadratic speedup O(√N) vs. classical O(N)
        2. **Quantum Feature Encoding**: Maps search result features to quantum states
        3. **Quantum Circuit Execution**: Processes multiple result features simultaneously
        4. **Quantum Amplitude Amplification**: Enhances relevance of best results
        5. **Quantum Measurement**: Collapses superpositions to obtain final rankings

        This gives QA³ the ability to search deeper and provide better-ranked results than classical approaches.
        """)
        
        # Display quantum search speedup visualization
        st.subheader("Quantum Search Speedup")
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Database sizes to plot
        sizes = np.logspace(1, 5, 20).astype(int)
        
        # Classical vs. quantum performance
        classical_steps = sizes  # O(N)
        quantum_steps = np.sqrt(sizes)  # O(√N)
        
        # Plot results
        ax.plot(sizes, classical_steps, 'r-', label='Classical (O(N))')
        ax.plot(sizes, quantum_steps, 'g-', label='Quantum (O(√N))')
        
        # Set log scales
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Labels
        ax.set_title('Quantum vs. Classical Search Performance')
        ax.set_xlabel('Database Size (entries)')
        ax.set_ylabel('Steps Required (log scale)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Display the plot
        st.pyplot(fig)

def main():
    """Main application function"""
    # Set page title and layout
    st.set_page_config(
        page_title="QA³: Quantum-Accelerated AI Agent",
        page_icon="⚛",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Display unified interface
    display_unifiedInterface()
    
    # Display quantum explanation
    display_quantum_explanation()
    
    # Display footer
    st.markdown("---")
    st.caption("QA³: Quantum-Accelerated AI Agent | Powered by PennyLane, Streamlit, and Python")

if __name__ == "__main__":
    main()