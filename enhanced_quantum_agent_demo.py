"""
Enhanced QA³: Quantum-Accelerated AI Agent Demo

This demo showcases the enhanced QA³ agent with:
1. Deep search across 20+ sources with quantum acceleration
2. Browser automation with natural language task processing
3. Task history tracking and analysis
4. PWA support for offline functionality
"""

import os
import json
import time
import asyncio
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enhanced-quantum-agent-demo")

# Import Streamlit for UI
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Try to import the PWA integration if available
try:
    from quantum_agent_framework.pwa_integration import initialize_pwa, get_pwa_controls
    PWA_AVAILABLE = True
except ImportError:
    PWA_AVAILABLE = False
    logger.warning("PWA integration not available. PWA features will be disabled.")

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
        
    if 'agent_running' not in st.session_state:
        st.session_state.agent_running = False
        
    if 'agent' not in st.session_state:
        st.session_state.agent = None
        
    if 'task_history' not in st.session_state:
        st.session_state.task_history = []
        
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
    if 'n_qubits' not in st.session_state:
        st.session_state.n_qubits = 8
        
    if 'use_quantum' not in st.session_state:
        st.session_state.use_quantum = True
        
    if 'view' not in st.session_state:
        st.session_state.view = "chat"  # Default view: chat, search, browser, settings
        
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
        
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
        
    if 'browser_url' not in st.session_state:
        st.session_state.browser_url = ""
        
    if 'browser_screenshot' not in st.session_state:
        st.session_state.browser_screenshot = None
        
    if 'browser_content' not in st.session_state:
        st.session_state.browser_content = None
        
    if 'browser_history' not in st.session_state:
        st.session_state.browser_history = []
        
    if 'deep_search' not in st.session_state:
        st.session_state.deep_search = True

async def initialize_agent():
    """Initialize the QA³ agent"""
    if st.session_state.agent_initialized:
        return True
    
    try:
        with st.status("Initializing Enhanced QA³ agent...", expanded=True) as status:
            st.write("Setting up agent components...")
            
            # Check if enhanced agent is available
            if ENHANCED_AGENT_AVAILABLE:
                st.write("Using enhanced QA³ agent implementation")
                
                # Create and initialize the enhanced agent
                st.session_state.agent = QA3AgentEnhanced(
                    use_quantum=st.session_state.use_quantum,
                    n_qubits=st.session_state.n_qubits,
                    use_claude=True,
                    use_openai=True
                )
                
                # Initialize agent
                await st.session_state.agent.initialize()
                
                # Start agent loop
                await st.session_state.agent.start_agent_loop()
                
                st.session_state.agent_initialized = True
                st.session_state.agent_running = True
                
                st.write("✅ Enhanced QA³ agent ready!")
                status.update(label="Enhanced QA³ agent initialized successfully!", state="complete")
                
            else:
                # Fall back to standard agent
                st.write("⚠️ Enhanced agent not available. Using standard QA³ agent.")
                
                # Try to import standard agent
                try:
                    from quantum_agent_framework.qa3_agent import QA3Agent
                    
                    # Create and initialize standard agent
                    st.session_state.agent = QA3Agent(
                        use_quantum=st.session_state.use_quantum,
                        n_qubits=st.session_state.n_qubits,
                        use_claude=True
                    )
                    
                    # Start agent loop
                    await st.session_state.agent.start_agent_loop()
                    
                    st.session_state.agent_initialized = True
                    st.session_state.agent_running = True
                    
                    st.write("✅ Standard QA³ agent ready!")
                    status.update(label="Standard QA³ agent initialized successfully!", state="complete")
                    
                except ImportError:
                    st.error("Could not initialize any agent implementation. Please check your installation.")
                    return False
            
            st.success("Agent initialized and ready!")
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
    
    # Add user message to chat
    st.session_state.chat_history.append({"role": "user", "content": task})
    
    try:
        # Process the task with the agent
        with st.spinner(f"Agent processing task: {task}"):
            # Process the task
            result = await st.session_state.agent.process_task(task)
            
            # Record the result in task history
            record = {
                "task": task,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.task_history.append(record)
            
            # Add agent response to chat
            response = result.get("summary", f"Task complete: {result.get('success', False)}")
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
        return result
    except Exception as e:
        error_msg = f"Error processing task: {str(e)}"
        logger.error(f"Exception in process_task: {str(e)}", exc_info=True)
        st.error(error_msg)
        st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {error_msg}"})
        return {"success": False, "error": error_msg}

async def perform_search(query, deep_search=True):
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
    if deep_search:
        task = f"Deep search: {query}"
    else:
        task = f"Search: {query}"
    
    try:
        # Use the agent to process the search
        with st.spinner(f"Searching for: {query}"):
            result = await st.session_state.agent.process_task(task)
            
            # Store search results in session state
            st.session_state.search_results = result
            
            # Don't add to chat history for explicit search
            
        return result
    except Exception as e:
        error_msg = f"Error during search: {str(e)}"
        logger.error(f"Exception in perform_search: {str(e)}", exc_info=True)
        st.error(error_msg)
        return {"success": False, "error": error_msg}

async def process_web_task(task):
    """Process a web browsing task"""
    if not st.session_state.agent_initialized:
        success = await initialize_agent()
        if not success:
            st.error("Agent not initialized")
            return {"success": False, "error": "Agent not initialized"}
    
    if not task:
        st.warning("Please enter a web task")
        return {"success": False, "error": "Empty web task"}
    
    try:
        # Use the agent to process the web task
        with st.spinner(f"Processing web task: {task}"):
            result = await st.session_state.agent.process_task(task)
            
            # Update browser state if applicable
            if result.get("success", False):
                if "url" in result:
                    st.session_state.browser_url = result["url"]
                
                if "screenshot" in result and result["screenshot"]:
                    st.session_state.browser_screenshot = result["screenshot"]
                
                if "content" in result:
                    st.session_state.browser_content = result["content"]
            
            # Add to task history
            record = {
                "task": task,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.task_history.append(record)
            
            # Add to chat history
            response = result.get("summary", f"Web task complete: {result.get('success', False)}")
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
        return result
    except Exception as e:
        error_msg = f"Error processing web task: {str(e)}"
        logger.error(f"Exception in process_web_task: {str(e)}", exc_info=True)
        st.error(error_msg)
        st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {error_msg}"})
        return {"success": False, "error": error_msg}

def display_chat_interface():
    """Display the chat interface"""
    st.header("Chat with QA³ Agent")
    
    # Display agent status
    if st.session_state.agent_initialized and st.session_state.agent_running:
        st.success("Agent is running and ready to process tasks")
    elif st.session_state.agent_initialized:
        st.warning("Agent is initialized but not running")
    else:
        st.warning("Agent is not initialized. Please initialize the agent.")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**QA³ Agent:** {message['content']}")
    
    # Chat input
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area("Type your task for the agent:", key="chat_input", height=100)
        submit_button = st.form_submit_button("Send")
        
        if submit_button and user_input:
            # Process user input
            asyncio.run(process_task(user_input))
            
            # Force a rerun to update chat display
            st.rerun()

def display_search_interface():
    """Display the search interface"""
    st.header("Quantum-Enhanced Deep Search")
    
    with st.form(key="search_form"):
        query = st.text_input("Search query:", key="search_query_input")
        col1, col2 = st.columns([3, 1])
        with col1:
            deep_search = st.checkbox("Use deep search (20+ sources)", value=st.session_state.deep_search)
        with col2:
            submit_button = st.form_submit_button("Search")
        
        if submit_button and query:
            # Update preferences
            st.session_state.deep_search = deep_search
            
            # Execute search
            asyncio.run(perform_search(query, deep_search))
            
            # Force a rerun to update results display
            st.rerun()
    
    # Display search results
    if st.session_state.search_results:
        results = st.session_state.search_results
        
        # Display search information
        if results.get("success", False):
            method = "quantum-enhanced" if results.get("quantum_enhanced", False) else "classical"
            source_count = results.get("source_count", 1)
            result_count = results.get("result_count", 0)
            execution_time = results.get("execution_time", 0)
            ranking_method = results.get("ranking_method", "unknown")
            
            st.info(f"Found {result_count} results from {source_count} sources using {method} search in {execution_time:.3f} seconds. Ranking method: {ranking_method}")
            
            # Display comprehensive summary if available
            if "comprehensive_summary" in results:
                with st.expander("Comprehensive Summary", expanded=True):
                    st.markdown(results["comprehensive_summary"])
            
            # Display search results
            if "search_results" in results and results["search_results"]:
                with st.expander("Search Results", expanded=False):
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
            
            # Display source distribution if available
            if "source_distribution" in results:
                with st.expander("Source Distribution", expanded=False):
                    source_dist = results["source_distribution"]
                    # Create bar chart of source distribution
                    if source_dist:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        sources = list(source_dist.keys())
                        counts = list(source_dist.values())
                        y_pos = np.arange(len(sources))
                        ax.barh(y_pos, counts)
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(sources)
                        ax.invert_yaxis()  # Labels read top-to-bottom
                        ax.set_xlabel('Number of Results')
                        ax.set_title('Results by Source Type')
                        st.pyplot(fig)
        else:
            st.error(f"Search failed: {results.get('error', 'Unknown error')}")

def display_browser_interface():
    """Display the web browsing interface"""
    st.header("Quantum-Enhanced Web Browsing")
    
    # Web task input
    with st.form(key="web_form"):
        web_task = st.text_input("Enter a web task (e.g., 'Find jobs at Microsoft', 'Go to example.com'):")
        submitted = st.form_submit_button("Execute")
        
        if submitted and web_task:
            # Execute web task
            asyncio.run(process_web_task(web_task))
            
            # Force a rerun to update browser display
            st.rerun()
    
    # Browser history
    if hasattr(st.session_state, 'browser_history') and st.session_state.browser_history:
        with st.expander("Web Browsing History", expanded=False):
            for i, entry in enumerate(reversed(st.session_state.browser_history[-10:])):  # Show last 10 entries
                st.write(f"{entry.get('timestamp', 'Unknown time')}: {entry.get('url', 'Unknown URL')}")
    
    # Current browser status
    if hasattr(st.session_state, 'browser_url') and st.session_state.browser_url:
        st.subheader("Current Page")
        st.info(f"URL: {st.session_state.browser_url}")
        
        # Display screenshot if available
        if st.session_state.browser_screenshot:
            st.image(st.session_state.browser_screenshot, caption="Browser Screenshot")
        else:
            st.caption("No screenshot available")
        
        # Display page content or analysis
        with st.expander("Page Analysis", expanded=False):
            if hasattr(st.session_state, 'browser_content') and st.session_state.browser_content:
                st.text_area("Page Content", st.session_state.browser_content, height=300)
            else:
                st.caption("No page content available")

def display_task_history():
    """Display task history"""
    st.header("Task History")
    
    if not st.session_state.task_history:
        st.info("No tasks have been executed yet.")
        return
    
    # Sort tasks by timestamp (newest first)
    sorted_tasks = sorted(st.session_state.task_history, 
                          key=lambda x: x.get("timestamp", ""), 
                          reverse=True)
    
    # Display tasks
    for i, task_record in enumerate(sorted_tasks):
        task = task_record.get("task", "Unknown task")
        result = task_record.get("result", {})
        timestamp = task_record.get("timestamp", "Unknown time")
        
        # Format timestamp
        try:
            dt = datetime.fromisoformat(timestamp)
            formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            formatted_time = timestamp
        
        # Task success status
        success = result.get("success", False)
        status_icon = "✅" if success else "❌"
        
        # Display task in expander
        with st.expander(f"{status_icon} {task} ({formatted_time})", expanded=(i == 0)):
            # Display task details
            st.caption(f"Task type: {result.get('task_type', 'unknown')}")
            
            # Display execution time if available
            if "execution_time" in result:
                st.caption(f"Execution time: {result['execution_time']:.3f} seconds")
            
            # Display quantum info if available
            if "quantum_enhanced" in result:
                quantum_status = "Quantum-enhanced" if result["quantum_enhanced"] else "Classical"
                st.caption(f"Processing: {quantum_status}")
            
            # Display task summary
            if "summary" in result:
                st.markdown("### Summary")
                st.markdown(result["summary"])
            
            # Display detailed results based on task type
            if result.get("task_type") == "search" and "search_results" in result:
                with st.expander("Search Results", expanded=False):
                    for j, search_result in enumerate(result["search_results"][:5]):  # Show top 5
                        st.markdown(f"**{j+1}. {search_result.get('title', 'Untitled')}**")
                        if "snippet" in search_result:
                            st.markdown(search_result["snippet"])
                        st.caption(f"Source: {search_result.get('source', 'Unknown')}")
                        st.divider()
            
            elif result.get("task_type") == "web_task":
                if "url" in result:
                    st.caption(f"URL: {result['url']}")
                if "screenshot" in result and result["screenshot"]:
                    st.image(result["screenshot"], caption="Browser Screenshot")
            
            # Display error if task failed
            if not success and "error" in result:
                st.error(f"Error: {result['error']}")

def display_settings():
    """Display settings interface"""
    st.header("QA³ Agent Settings")
    
    # Quantum computing settings
    st.subheader("Quantum Computing Settings")
    use_quantum = st.checkbox("Use quantum acceleration", value=st.session_state.use_quantum)
    n_qubits = st.slider("Number of qubits", min_value=4, max_value=29, value=st.session_state.n_qubits)
    
    # Agent settings
    st.subheader("Agent Settings")
    use_claude = st.checkbox("Use Claude AI (if available)", value=True)
    use_openai = st.checkbox("Use OpenAI (if available)", value=True)
    
    # Apply settings button
    if st.button("Apply Settings"):
        # Store new settings
        st.session_state.use_quantum = use_quantum
        st.session_state.n_qubits = n_qubits
        
        # Reset agent if settings changed
        if st.session_state.agent_initialized:
            st.warning("Settings changed. Restarting agent...")
            
            # Stop current agent
            if hasattr(st.session_state, 'agent') and st.session_state.agent:
                if hasattr(st.session_state.agent, 'stop_agent_loop'):
                    asyncio.run(st.session_state.agent.stop_agent_loop())
                if hasattr(st.session_state.agent, 'close'):
                    asyncio.run(st.session_state.agent.close())
            
            # Reset agent state
            st.session_state.agent = None
            st.session_state.agent_initialized = False
            st.session_state.agent_running = False
            
            # Initialize new agent
            asyncio.run(initialize_agent())
            
            st.success("Agent restarted with new settings!")
            st.rerun()

def display_quantum_demo():
    """Display quantum computing demonstration"""
    st.header("Quantum Computing Demonstration")
    
    demo_type = st.selectbox(
        "Select demonstration",
        ["Search Algorithm", "Factorization", "Optimization"]
    )
    
    if demo_type == "Search Algorithm":
        display_search_demo()
    elif demo_type == "Factorization":
        display_factorization_demo()
    elif demo_type == "Optimization":
        display_optimization_demo()

def display_search_demo():
    """Display quantum search algorithm demo"""
    st.subheader("Quantum vs. Classical Search")
    
    # Allow user to specify database size
    db_size = st.slider("Database Size", 10, 10000, 1000, step=10)
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Database sizes to plot
    sizes = np.logspace(1, 5, 20).astype(int)
    
    # Classical vs. quantum performance
    classical_steps = sizes  # O(N)
    quantum_steps = np.sqrt(sizes)  # O(√N)
    
    # Plot results
    ax.plot(sizes, classical_steps, 'r-', label='Classical (O(N))')
    ax.plot(sizes, quantum_steps, 'g-', label='Quantum (O(√N))')
    
    # Mark the selected size
    ax.scatter([db_size], [db_size], color='blue', s=100, label='Classical Steps')
    ax.scatter([db_size], [np.sqrt(db_size)], color='green', s=100, label='Quantum Steps')
    
    # Set log scales
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Labels
    ax.set_title(f'Quantum vs. Classical Search Performance')
    ax.set_xlabel('Database Size (entries)')
    ax.set_ylabel('Steps Required (log scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Display the plot
    st.pyplot(fig)
    
    # Display summary
    speedup = db_size / np.sqrt(db_size)
    st.info(f"For a database of {db_size:,} entries:")
    st.markdown(f"""
    * Classical search requires up to **{db_size:,}** steps
    * Quantum search requires approximately **{int(np.sqrt(db_size)):,}** steps
    * Quantum speedup: **{speedup:.2f}x** faster
    """)

def display_factorization_demo():
    """Display quantum factorization demo"""
    st.subheader("Quantum vs. Classical Factorization")
    
    # Allow user to specify number to factorize
    number = st.slider("Number to Factorize", 10, 10000, 1000, step=10)
    bits = len(bin(number)) - 2
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Range of bit lengths
    bit_lengths = np.arange(4, 32)
    
    # Classical vs. quantum performance (simplified models)
    # Classical: O(exp(sqrt(bits)))
    # Quantum: O(bits^3)
    classical_time = np.exp(np.sqrt(bit_lengths)) / 1e3
    quantum_time = bit_lengths**3 / 1e2
    
    # Plot results
    ax.plot(bit_lengths, classical_time, 'r-', label='Classical (exp)')
    ax.plot(bit_lengths, quantum_time, 'g-', label='Quantum (polynomial)')
    
    # Mark the selected number
    ax.scatter([bits], [np.exp(np.sqrt(bits)) / 1e3], color='blue', s=100, label=f'Selected: {number}')
    
    # Set log scale for y-axis
    ax.set_yscale('log')
    
    # Labels
    ax.set_title(f'Quantum vs. Classical Factorization Performance')
    ax.set_xlabel('Bit Length of Number')
    ax.set_ylabel('Relative Time (log scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Display the plot
    st.pyplot(fig)
    
    # Display summary
    classical_est = np.exp(np.sqrt(bits)) / 1e3
    quantum_est = bits**3 / 1e2
    speedup = classical_est / quantum_est
    
    st.info(f"For factorizing {number:,} ({bits} bits):")
    st.markdown(f"""
    * Classical factorization: Exponential time complexity
    * Quantum factorization: Polynomial time complexity
    * Estimated quantum speedup: **{speedup:.2f}x** faster
    
    Note: For small numbers like this, the advantage may be limited.
    Quantum factorization shines with numbers over 1024 bits.
    """)

def display_optimization_demo():
    """Display quantum optimization demo"""
    st.subheader("Quantum vs. Classical Optimization")
    
    # Allow user to specify problem size
    problem_size = st.slider("Optimization Problem Size (variables)", 2, 200, 50, step=1)
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Range of problem sizes
    sizes = np.arange(2, 201)
    
    # Classical vs. quantum performance (simplified models)
    # Classical: O(2^n) for exact solution
    # Quantum: O(sqrt(2^n)) with Grover-based approaches
    classical_time = 2**np.clip(sizes/10, 0, 20)  # Clip to avoid overflow
    quantum_time = np.sqrt(2**np.clip(sizes/10, 0, 20))
    
    # Plot results (with clipping for visualization)
    max_y = 1e6
    ax.plot(sizes, np.clip(classical_time, 0, max_y), 'r-', label='Classical (exponential)')
    ax.plot(sizes, np.clip(quantum_time, 0, max_y), 'g-', label='Quantum (quadratic speedup)')
    
    # Mark the selected size
    ax.scatter([problem_size], [min(2**(problem_size/10), max_y)], color='blue', s=100, label=f'Selected: {problem_size} vars')
    
    # Set log scale for y-axis
    ax.set_yscale('log')
    
    # Labels
    ax.set_title(f'Quantum vs. Classical Optimization Performance')
    ax.set_xlabel('Problem Size (number of variables)')
    ax.set_ylabel('Relative Time (log scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Display the plot
    st.pyplot(fig)
    
    # Display summary
    if problem_size <= 20:
        classical_est = 2**(problem_size/10)
        quantum_est = np.sqrt(2**(problem_size/10))
    else:
        classical_est = 2**2  # Avoid overflow, just show "very large"
        quantum_est = np.sqrt(2**2)
    
    speedup = classical_est / quantum_est
    
    st.info(f"For optimization with {problem_size} variables:")
    st.markdown(f"""
    * Classical optimization: Exponential time complexity
    * Quantum optimization: Quadratic speedup for many algorithms
    * Estimated quantum speedup: **{speedup:.2f}x** faster
    
    Note: The actual speedup depends on the specific optimization problem and algorithm.
    """)

def display_about():
    """Display information about the QA³ agent"""
    st.header("About QA³ Agent")
    
    st.markdown("""
    ## QA³: Quantum-Accelerated AI Agent
    
    QA³ (Quantum-Accelerated AI Agent) is a cutting-edge hybrid quantum-classical computing platform
    that provides quantum-enhanced search, browsing, and data processing capabilities.
    
    ### Key Features
    
    - **Quantum-Enhanced Search**: Search across 20+ sources with quantum acceleration
    - **Natural Language Interface**: Communicate with the agent in natural language
    - **Autonomous Web Browsing**: Complete complex web tasks autonomously
    - **Task History**: Track and analyze task results
    - **PWA Support**: Use offline and install as a standalone application
    
    ### Quantum Advantage
    
    The agent uses quantum computing principles to accelerate search and optimization tasks,
    providing real performance gains over classical algorithms. This is achieved through:
    
    - **Grover's Algorithm**: Quantum search with quadratic speedup O(√N) vs. O(N)
    - **Quantum Feature Encoding**: Mapping data to quantum states for enhanced processing
    - **Quantum Circuit Learning**: Circuit-based machine learning for relevance ranking
    
    ### Technical Information
    
    The agent is built using:
    
    - **PennyLane**: Quantum circuit simulation and optimization
    - **Azure Quantum IonQ**: Real quantum hardware integration (when available)
    - **Streamlit**: Interactive web interface
    - **LangChain**: Agent architecture and orchestration
    """)
    
    # Display agent status if available
    if st.session_state.agent_initialized and hasattr(st.session_state.agent, 'get_status'):
        with st.expander("Agent Status", expanded=False):
            try:
                status = st.session_state.agent.get_status()
                st.json(status)
            except Exception as e:
                st.error(f"Error getting agent status: {str(e)}")

def display_sidebar():
    """Display sidebar with navigation and controls"""
    with st.sidebar:
        st.title("QA³ Agent")
        
        # Navigation
        st.subheader("Navigation")
        
        # View selector
        view = st.radio("Select View", [
            "Chat", "Search", "Web Browsing", "Task History", 
            "Settings", "Quantum Demo", "About"
        ])
        
        # Update session state
        if view == "Chat":
            st.session_state.view = "chat"
        elif view == "Search":
            st.session_state.view = "search"
        elif view == "Web Browsing":
            st.session_state.view = "browser"
        elif view == "Task History":
            st.session_state.view = "history"
        elif view == "Settings":
            st.session_state.view = "settings"
        elif view == "Quantum Demo":
            st.session_state.view = "quantum_demo"
        elif view == "About":
            st.session_state.view = "about"
        
        # Agent initialization
        st.subheader("Agent Control")
        
        if not st.session_state.agent_initialized:
            if st.button("Initialize Agent"):
                asyncio.run(initialize_agent())
                st.rerun()
        else:
            st.success("Agent initialized")
            
            if st.session_state.agent_running:
                st.info("Agent running")
                
                if st.button("Stop Agent"):
                    if hasattr(st.session_state.agent, 'stop_agent_loop'):
                        asyncio.run(st.session_state.agent.stop_agent_loop())
                        st.session_state.agent_running = False
                        st.rerun()
            else:
                st.warning("Agent stopped")
                
                if st.button("Start Agent"):
                    if hasattr(st.session_state.agent, 'start_agent_loop'):
                        asyncio.run(st.session_state.agent.start_agent_loop())
                        st.session_state.agent_running = True
                        st.rerun()
        
        # PWA controls if available
        if PWA_AVAILABLE:
            st.subheader("PWA Controls")
            get_pwa_controls()
        
        # Agent status
        if st.session_state.agent_initialized:
            st.subheader("Agent Status")
            
            try:
                # Display core metrics
                quantum = "Enabled" if st.session_state.use_quantum else "Disabled"
                st.caption(f"Quantum: {quantum}")
                st.caption(f"Qubits: {st.session_state.n_qubits}")
                
                # Get status from agent if available
                if hasattr(st.session_state.agent, 'get_status'):
                    status = st.session_state.agent.get_status()
                    if "tasks_completed" in status:
                        st.caption(f"Tasks: {status['tasks_completed']}")
                    if "searches_performed" in status:
                        st.caption(f"Searches: {status['searches_performed']}")
                    if "web_interactions" in status:
                        st.caption(f"Web tasks: {status['web_interactions']}")
                    if "quantum_operations" in status:
                        st.caption(f"Quantum ops: {status['quantum_operations']}")
            except Exception as e:
                st.caption(f"Error: {str(e)}")

def main():
    """Main application"""
    # Set page title and layout
    st.set_page_config(
        page_title="QA³: Quantum-Accelerated AI Agent",
        page_icon="⚛",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize PWA if available
    if PWA_AVAILABLE:
        initialize_pwa()
    
    # Initialize session state
    initialize_session_state()
    
    # Display sidebar
    display_sidebar()
    
    # Display content based on selected view
    if st.session_state.view == "chat":
        display_chat_interface()
    elif st.session_state.view == "search":
        display_search_interface()
    elif st.session_state.view == "browser":
        display_browser_interface()
    elif st.session_state.view == "history":
        display_task_history()
    elif st.session_state.view == "settings":
        display_settings()
    elif st.session_state.view == "quantum_demo":
        display_quantum_demo()
    elif st.session_state.view == "about":
        display_about()

if __name__ == "__main__":
    main()