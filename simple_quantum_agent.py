"""
QA³: Quantum-Accelerated AI Agent - Unified Interface

This streamlined interface provides a conversational AI agent with 
quantum-enhanced deep search capabilities.
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
        
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    if 'current_view' not in st.session_state:
        st.session_state.current_view = "chat"
        
    # Initialize PWA status
    if 'pwa_status' not in st.session_state:
        st.session_state.pwa_status = {
            "installed": False,
            "canInstall": False
        }

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

async def process_message(message):
    """Process a user message with the QA³ agent"""
    if not st.session_state.agent_initialized:
        success = await initialize_agent()
        if not success:
            return {
                "success": False, 
                "error": "Agent not initialized", 
                "response": "I'm having trouble connecting to my quantum capabilities. Please try again later."
            }
    
    try:
        # Process the message with the agent
        with st.spinner("Thinking..."):
            # Process the message
            result = await st.session_state.agent.process_task(message)
            
            # Record the result in task history
            record = {
                "task": message,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.task_history.append(record)
            
        return result
    except Exception as e:
        error_msg = f"Error processing message: {str(e)}"
        logger.error(f"Exception in process_message: {str(e)}", exc_info=True)
        return {
            "success": False, 
            "error": error_msg,
            "response": "I encountered an error while processing your message. Please try again."
        }

async def perform_search(query, use_quantum=True, deep_search=False):
    """Perform a search using the agent"""
    if not st.session_state.agent_initialized:
        success = await initialize_agent()
        if not success:
            st.error("Agent not initialized")
            return {"success": False, "error": "Agent not initialized"}
    
    if not query:
        st.warning("Please enter a search query")
        return {"success": False, "error": "Empty search query"}
    
    # Format search task with optional deep search flag
    task = f"Search: {query}"
    if deep_search:
        task = f"Deep search: {query}"
    
    try:
        # Update quantum setting before search
        st.session_state.agent.use_quantum = use_quantum
        
        # Use the agent to process the search
        with st.spinner(f"Searching for: {query}" + 
                      (" (with quantum acceleration)" if use_quantum else "") +
                      (" (deep search)" if deep_search else "")):
            result = await st.session_state.agent.process_task(task)
            
            # Store search results in session state
            st.session_state.search_results = result
            
        return result
    except Exception as e:
        error_msg = f"Error during search: {str(e)}"
        logger.error(f"Exception in perform_search: {str(e)}", exc_info=True)
        st.error(error_msg)
        return {"success": False, "error": error_msg}

def display_chat_interface():
    """Display the chat interface for the QA³ agent"""
    st.title("QA³: Quantum-Accelerated AI Assistant")
    
    # Initialize agent in the background if needed
    if not st.session_state.agent_initialized:
        asyncio.run(initialize_agent())
    
    # Chat container
    chat_container = st.container()
    
    # Display existing messages
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Handle new message input
    if prompt := st.chat_input("Ask me anything or type a query to search..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("⚛️ Thinking...")
            
            # Process the message
            result = asyncio.run(process_message(prompt))
            
            if result.get("success", False):
                response = result.get("response", "I processed your request but couldn't generate a response.")
                message_placeholder.markdown(response)
                
                # Check if search was performed
                if "search_results" in result and result["search_results"]:
                    st.session_state.search_results = result
                    
                    # Show a deep search button if it seems like a search query
                    if result.get("is_search_query", False) and not result.get("deep_search", False):
                        if st.button("🔍 Perform Deep Search", key="deep_search_button"):
                            with st.spinner("Performing deep search..."):
                                deep_result = asyncio.run(perform_search(prompt, use_quantum=True, deep_search=True))
                                if deep_result.get("success", False):
                                    st.session_state.search_results = deep_result
                                    
                                    # Add search result to message
                                    if "comprehensive_summary" in deep_result:
                                        deep_summary = f"""
                                        ## Deep Search Results
                                        
                                        {deep_result["comprehensive_summary"]}
                                        
                                        *Search performed across {deep_result.get('source_count', 0)} sources with quantum acceleration.*
                                        """
                                        message_placeholder.markdown(response + "\n\n" + deep_summary)
                    
                    # Show search metrics
                    if result.get("is_search_query", False):
                        method = "quantum-enhanced" if result.get("quantum_enhanced", False) else "classical"
                        deep = "deep " if result.get("deep_search", False) else ""
                        sources = result.get("source_count", 0)
                        results_count = result.get("result_count", 0)
                        
                        st.caption(f"*{deep}Search performed with {method} processing across {sources} sources, finding {results_count} results*")
            else:
                message_placeholder.markdown(f"I'm sorry, I encountered an error: {result.get('error', 'Unknown error')}")
            
            # Add AI response to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result.get("response", "Error: No response generated")
            })

def display_search_interface():
    """Display the search interface for the QA³ agent"""
    st.title("Quantum-Enhanced Search")
    
    with st.form(key="search_form"):
        query = st.text_input("Enter your search query:", key="search_query_input")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            use_quantum = st.checkbox("Enable quantum acceleration", value=st.session_state.use_quantum)
        
        with col2:
            deep_search = st.checkbox("Enable deep search", value=False)
        
        with col3:
            submit_button = st.form_submit_button("🔍 Search")
        
        if submit_button and query:
            # Execute search
            result = asyncio.run(perform_search(query, use_quantum, deep_search))
            
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
            deep = "deep " if results.get("deep_search", False) else ""
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
            
def display_sidebar():
    """Display the sidebar with navigation and controls"""
    with st.sidebar:
        st.title("QA³ Agent")
        
        # Navigation
        st.subheader("Navigation")
        view = st.radio("Select View", ["Chat", "Search"], 
                       index=0 if st.session_state.current_view == "chat" else 1)
        
        if view == "Chat" and st.session_state.current_view != "chat":
            st.session_state.current_view = "chat"
            st.rerun()
        elif view == "Search" and st.session_state.current_view != "search":
            st.session_state.current_view = "search"
            st.rerun()
        
        # Quantum Settings
        st.subheader("Quantum Settings")
        n_qubits = st.slider("Number of Qubits", min_value=4, max_value=16, value=st.session_state.n_qubits)
        if n_qubits != st.session_state.n_qubits:
            st.session_state.n_qubits = n_qubits
            # Reset agent to apply new settings
            st.session_state.agent_initialized = False
            asyncio.run(initialize_agent())
        
        # Display agent status
        if st.session_state.agent_initialized:
            st.success("Agent Ready")
        else:
            st.warning("Agent Initializing...")
            
        # PWA Installation
        st.subheader("Install App")
        if st.session_state.pwa_status.get("canInstall", False):
            if st.button("📱 Install as App", key="pwa-install-button"):
                st.markdown("Click the button above to install the app on your device.")
        elif st.session_state.pwa_status.get("installed", False):
            st.success("App Installed")
        else:
            st.info("This app can be installed on your device for offline access.")
            
        # Reset chat
        if st.button("🔄 Reset Chat"):
            st.session_state.messages = []
            st.rerun()

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
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Add PWA Script
    st.markdown("""
    <script src="/pwa.js" defer></script>
    <link rel="manifest" href="/manifest.json">
    <meta name="theme-color" content="#7B68EE">
    """, unsafe_allow_html=True)
    
    # Display the sidebar
    display_sidebar()
    
    # Display the appropriate interface based on the current view
    if st.session_state.current_view == "chat":
        display_chat_interface()
    else:
        display_search_interface()
    
    # Display quantum explanation
    display_quantum_explanation()
    
    # Display footer
    st.markdown("---")
    st.caption("QA³: Quantum-Accelerated AI Agent | Powered by PennyLane, Streamlit, and Python")

if __name__ == "__main__":
    main()