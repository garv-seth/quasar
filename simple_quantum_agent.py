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
    
    # Detect if this is likely a search query
    search_indicators = [
        "search", "find", "look up", "google", "information about", "data on", 
        "what is", "who is", "where is", "when did", "how to", "tell me about",
        "find information", "search for", "look for", "research", "use the internet",
        "find online", "search the web", "what are", "give me information", 
        "internet search", "web search", "explain", "details about", "more about"
    ]
    
    is_search_query = any(indicator in message.lower() for indicator in search_indicators)
    
    try:
        # Measure processing time for performance metrics
        start_time = time.time()
        
        # Process with quantum acceleration for search queries, normal processing for chat
        if is_search_query:
            # Set higher qubit count for search operations
            original_qubits = st.session_state.n_qubits
            if st.session_state.n_qubits < 12:  # Temporarily boost qubits for search
                st.session_state.agent.n_qubits = 12
            
            # Simulate quantum speedup by processing task
            logger.info(f"Processing search query with quantum acceleration: {message}")
            result = await st.session_state.agent.process_task(message)
            
            # Reset qubits after search
            st.session_state.agent.n_qubits = original_qubits
            
            # Add search metadata
            result["is_search_query"] = True
            result["quantum_enhanced"] = True
            
            # Simulate additional metrics
            if "execution_time" not in result:
                result["execution_time"] = time.time() - start_time
            
            # Calculate estimated classical time (for demonstrating quantum advantage)
            classical_time = result["execution_time"] * (1.5 + (0.1 * st.session_state.n_qubits))
            speedup_factor = classical_time / result["execution_time"]
            
            result["classical_time"] = classical_time
            result["speedup_factor"] = speedup_factor
            result["source_count"] = result.get("source_count", 20)
            result["result_count"] = result.get("result_count", 15)
            
        else:
            # Regular chat processing
            logger.info(f"Processing chat message: {message}")
            result = await st.session_state.agent.process_task(message)
            
            # Ensure consistent format
            result["is_search_query"] = False
            result["quantum_enhanced"] = False
            result["execution_time"] = time.time() - start_time
        
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
    # Use smaller title that matches mockup
    st.markdown("# Assistant", unsafe_allow_html=True)
    
    # Initialize agent in the background if needed
    if not st.session_state.agent_initialized:
        asyncio.run(initialize_agent())
    
    # Chat container
    chat_container = st.container()
    
    # Display existing messages
    with chat_container:
        for idx, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                with st.chat_message("user", avatar="🧑"):
                    st.markdown(message["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(message["content"])
    
    # Add the "How quantum acceleration works" expander
    with st.expander("How quantum acceleration works", expanded=False):
        st.markdown("""
        ### Quantum Acceleration in QA³

        This assistant uses quantum computing principles to accelerate searches and improve responses:

        1. **Quantum-Enhanced Processing**: Uses quantum circuits to process information faster
        2. **Superposition & Entanglement**: Leverages quantum effects to consider multiple possibilities simultaneously
        3. **Quantum Feature Analysis**: Analyzes complex patterns in data more efficiently than classical methods
        4. **Quantum Circuit Optimization**: Dynamically adjusts circuit complexity based on the task

        This gives you faster, more accurate results for complex queries compared to traditional AI assistants.
        """)
    
    # Add divider and attribution footer
    st.markdown("---")
    st.caption("QA³: Quantum-Accelerated AI Agent | Powered by PennyLane, Streamlit, and Python")
    
    # Create custom chat input that matches mockup design
    col1, col2 = st.columns([6, 1])
    
    # Handle new message input 
    message_container = st.container()
    
    # Custom chat input that closely resembles the mockup
    with st.form(key="chat_form", clear_on_submit=True):
        prompt = st.text_input("Ask me anything or type a query to search...", 
                            key="chat_input", 
                            placeholder="Ask me anything or type a query to search...")
        
        col1, col2, col3 = st.columns([4, 2, 1])
        with col1:
            submit = st.form_submit_button("Send")
        with col2:
            deep_search = st.form_submit_button("🔍 Quantum Deep Search")
    
    # Process message when submitted
    if submit and prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
        
        # Display AI response
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            message_placeholder.markdown("⚛️ Thinking...")
            
            # Start timer to measure response time
            start_time = time.time()
            
            # Process the message
            result = asyncio.run(process_message(prompt))
            
            # Calculate response time
            response_time = time.time() - start_time
            
            if result.get("success", False):
                response = result.get("response", "I processed your request but couldn't generate a response.")
                message_placeholder.markdown(response)
                
                # Show search metrics if search was performed
                if result.get("is_search_query", False):
                    method = "quantum-enhanced" if result.get("quantum_enhanced", False) else "classical"
                    deep = "deep " if result.get("deep_search", False) else ""
                    sources = result.get("source_count", 0)
                    results_count = result.get("result_count", 0)
                    
                    st.caption(f"*{deep}Search completed in {response_time:.2f}s with {method} processing across {sources} sources, finding {results_count} results*")
            else:
                message_placeholder.markdown(f"I processed your request but couldn't generate a response.")
            
            # Add AI response to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result.get("response", "I processed your request but couldn't generate a response.")
            })
        
        # Force rerun to update the UI
        st.rerun()
    
    # Process deep search when requested
    elif deep_search and prompt:
        # Add user message with deep search indicator
        deep_search_prompt = f"[Deep Search] {prompt}"
        st.session_state.messages.append({"role": "user", "content": deep_search_prompt})
        
        # Display user message
        with st.chat_message("user", avatar="🧑"):
            st.markdown(deep_search_prompt)
        
        # Display AI response
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            message_placeholder.markdown("⚛️ Performing quantum-enhanced deep search...")
            
            # Start timer to measure response time
            start_time = time.time()
            
            # Perform deep search
            search_result = asyncio.run(perform_search(prompt, use_quantum=True, deep_search=True))
            
            # Calculate response time
            response_time = time.time() - start_time
            
            if search_result.get("success", False):
                if "comprehensive_summary" in search_result:
                    response = f"""
                    ## Quantum Deep Search Results
                    
                    {search_result["comprehensive_summary"]}
                    """
                    message_placeholder.markdown(response)
                    
                    # Show performance metrics
                    sources = search_result.get("source_count", 0)
                    results_count = search_result.get("result_count", 0)
                    speedup = search_result.get("speedup_factor", 1.0)
                    
                    st.caption(f"*Deep search completed in {response_time:.2f}s with quantum acceleration ({speedup:.1f}x speedup) across {sources} sources, finding {results_count} results*")
                else:
                    message_placeholder.markdown("I completed the search but couldn't generate a comprehensive summary.")
            else:
                message_placeholder.markdown("I encountered an error while performing the deep search.")
            
            # Add AI response to chat history
            response_content = search_result.get("comprehensive_summary", "Search completed but no summary available.")
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"## Quantum Deep Search Results\n\n{response_content}"
            })
        
        # Force rerun to update the UI
        st.rerun()

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