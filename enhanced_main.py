"""
Enhanced Q3A: Quantum-Accelerated AI Agent with Browser Automation
Main Streamlit Interface
"""

import os
import asyncio
import streamlit as st
import numpy as np
import pandas as pd
import time
import json
from datetime import datetime
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Import enhanced Q3A agent
from quantum_agent_framework.agents.enhanced_q3a_agent import EnhancedQ3AAgent

# Page configuration
st.set_page_config(
    page_title="QUASAR: Quantum AI Agent with Browser Automation",
    page_icon="🧠",
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
    .agent-message {
        background-color: #f0f7ff;
        border-left: 3px solid #29539B;
        padding: 1em;
        margin: 0.5em 0;
        border-radius: 0 10px 10px 0;
    }
    .user-message {
        background-color: #f2f2f2;
        border-right: 3px solid #718096;
        padding: 1em;
        margin: 0.5em 0;
        border-radius: 10px 0 0 10px;
        text-align: right;
    }
    .task-result {
        background-color: #f0f8f7;
        border: 1px solid #ddd;
        padding: 1em;
        margin: 0.5em 0;
        border-radius: 10px;
    }
    .center {
        display: flex;
        justify-content: center;
    }
    .browser-preview {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 0.5em;
        margin: 1em 0;
        background-color: #f8f9fa;
    }
    .api-key-form {
        background-color: #f9f9f9;
        padding: 1em;
        border-radius: 10px;
        margin-bottom: 1em;
        border: 1px solid #eee;
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
    .browse-history-item {
        border-bottom: 1px solid #eee;
        padding: 0.5em 0;
    }
    .screenshot-container {
        text-align: center;
        margin: 1em 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for the agent and conversation history
async def initialize_agent_if_needed():
    """Initialize the enhanced Q3A agent if not already in session state"""
    if 'agent' not in st.session_state:
        # Check for API keys
        use_claude = os.environ.get("ANTHROPIC_API_KEY") is not None
        use_openai = os.environ.get("OPENAI_API_KEY") is not None
        
        # Default to quantum enabled with appropriate AI based on available keys
        st.session_state.agent = EnhancedQ3AAgent(
            use_quantum=True, 
            n_qubits=8, 
            use_azure=True,
            use_claude=use_claude
        )
        
        # Initialize agent's async components
        await st.session_state.agent.initialize()
        
        # Set default session state values
        st.session_state.show_debug = False
        st.session_state.tasks = []
        st.session_state.messages = []
        st.session_state.current_tab = "chat"
        st.session_state.browse_history = []
        st.session_state.last_screenshot = None

# Function to display API key configuration form
def display_api_key_form():
    """Display a form to configure API keys if not already set"""
    with st.expander("Configure API Keys", expanded=not (os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))):
        st.markdown("### API Key Configuration")
        
        # OpenAI API key
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        openai_key_input = st.text_input(
            "OpenAI API Key",
            value=openai_key if openai_key else "",
            type="password",
            help="Enter your OpenAI API key to enable GPT-4o integration"
        )
        
        # Anthropic API key
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        anthropic_key_input = st.text_input(
            "Anthropic API Key",
            value=anthropic_key if anthropic_key else "",
            type="password",
            help="Enter your Anthropic API key to enable Claude 3.7 Sonnet integration"
        )
        
        # Save keys button
        if st.button("Save API Keys"):
            # Set environment variables
            if openai_key_input:
                os.environ["OPENAI_API_KEY"] = openai_key_input
            if anthropic_key_input:
                os.environ["ANTHROPIC_API_KEY"] = anthropic_key_input
            
            # Reinitialize agent
            if 'agent' in st.session_state:
                del st.session_state.agent
            
            st.success("API keys saved! Reinitializing agent...")
            st.rerun()

# Function to process user instructions
async def process_user_instruction(instruction):
    """Process a user instruction with the enhanced Q3A agent"""
    if not instruction:
        return None
    
    with st.spinner("Processing with Enhanced Q3A Agent..."):
        # Process the instruction
        result = await st.session_state.agent.process_user_instruction(instruction)
        
        # Add result to task history
        st.session_state.tasks.append(result)
        
        # Update browse history if it was a browsing task
        if result.get("requires_browsing", False) and result.get("result", {}).get("success", False):
            browse_result = result.get("result", {})
            
            # Add to browse history
            browse_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "url": browse_result.get("url", ""),
                "title": browse_result.get("title", ""),
                "summary": browse_result.get("summary", ""),
                "success": browse_result.get("success", False),
                "screenshot": None  # Will be updated if available
            }
            
            st.session_state.browse_history.append(browse_entry)
            
            # Store most recent screenshot if available
            if browse_result.get("success") and browse_result.get("has_screenshot"):
                # This would be captured in a real implementation
                # Here we'd extract the screenshot from the browser
                pass
    
    return result

# Sidebar
def display_sidebar():
    """Display the sidebar with settings and navigation"""
    with st.sidebar:
        st.title("QUASAR Framework")
        st.image("https://upload.wikimedia.org/wikipedia/commons/9/96/Quantum_circuit_compilation_for_nisq_gqo.png", width=250)
        
        st.markdown("### Agent Settings")
        
        # Quantum settings
        use_quantum = st.checkbox("Use Quantum Acceleration", value=True, key="use_quantum")
        n_qubits = st.slider("Number of Qubits", min_value=4, max_value=29, value=8, key="n_qubits")
        
        # AI model settings
        st.markdown("### AI Model Settings")
        use_claude = st.checkbox("Use Claude 3.7 Sonnet", value=os.environ.get("ANTHROPIC_API_KEY") is not None, key="use_claude")
        
        # Update agent if settings changed
        if 'agent' in st.session_state:
            if (use_quantum != st.session_state.agent.use_quantum or 
                n_qubits != st.session_state.agent.n_qubits or
                use_claude != st.session_state.agent.use_claude):
                st.session_state.agent.use_quantum = use_quantum
                st.session_state.agent.n_qubits = n_qubits
                st.session_state.agent.use_claude = use_claude
        
        st.markdown("### Advanced Settings")
        show_debug = st.checkbox("Show Debug Information", value=st.session_state.get("show_debug", False))
        st.session_state.show_debug = show_debug
        
        # Download task history button
        if st.session_state.get("tasks") and len(st.session_state.tasks) > 0:
            if st.button("Download Task History"):
                task_json = json.dumps(st.session_state.tasks, indent=2)
                b64 = base64.b64encode(task_json.encode()).decode()
                href = f'<a href="data:application/json;base64,{b64}" download="q3a_tasks.json">Download JSON</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Navigation")
        tab_options = ["Chat", "Browser", "Tasks", "Performance"]
        selected_tab = st.radio("Select Interface", tab_options)
        st.session_state.current_tab = selected_tab.lower()
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        **QUASAR: Quantum AI Agent Framework**  
        Quantum-Accelerated Search and AI Reasoning
        
        Version: 1.0.0  
        © 2025 Quantum Labs
        """)

# Function to display the chat interface
def display_chat_interface():
    """Display the main chat interface"""
    st.subheader("Quantum-Enhanced AI Agent Chat")
    
    st.markdown("""
    Interact with the Q3A agent to perform tasks with quantum acceleration.
    The agent can handle various tasks including:
    
    - **Enhanced Search**: Grover's algorithm provides quadratic speedup
    - **Factorization**: Shor's algorithm can factor large numbers efficiently
    - **Optimization**: QAOA provides advantages for constraint optimization problems
    - **Web Browsing**: Quantum-enhanced content analysis and prioritization
    """)
    
    # Display conversation history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='agent-message'>{message['content']}</div>", unsafe_allow_html=True)
    
    # Chat input
    with st.form(key="chat_form"):
        user_input = st.text_area("Enter your instruction or question:", height=100)
        cols = st.columns([1, 1, 4])
        with cols[0]:
            submit_button = st.form_submit_button("Send")
        with cols[1]:
            clear_button = st.form_submit_button("Clear Chat")
    
    # Form processing
    if clear_button:
        st.session_state.messages = []
        st.rerun()
        
    if submit_button and user_input:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Process instruction asynchronously
        result = asyncio.run(process_user_instruction(user_input))
        
        if result:
            # Add agent response to history
            response = result.get("result", {}).get("response", "I couldn't process your instruction.")
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Show screenshot if available and it was a browsing task
            if (result.get("requires_browsing", False) and 
                result.get("result", {}).get("success", True) and 
                result.get("result", {}).get("has_screenshot", False)):
                # This is where we would display the screenshot
                pass
        
        # Rerun to update the UI
        st.rerun()

# Function to display browser interface
def display_browser_interface():
    """Display the browser interface with history and controls"""
    st.subheader("Quantum-Enhanced Web Browser")
    
    st.markdown("""
    This interface allows you to interact with the quantum-enhanced browser.
    The browser uses quantum computing to analyze web content, prioritize elements,
    and enhance the search and retrieval capabilities.
    """)
    
    # URL navigation form
    with st.form(key="browser_form"):
        url_input = st.text_input("Enter URL to browse:", placeholder="e.g., example.com or full URL")
        
        cols = st.columns([1, 5])
        with cols[0]:
            browse_button = st.form_submit_button("Browse")
    
    # Process URL navigation
    if browse_button and url_input:
        # Create a browse instruction
        browse_instruction = f"Browse to {url_input} and analyze the content"
        
        # Add to messages
        st.session_state.messages.append({"role": "user", "content": browse_instruction})
        
        # Process instruction asynchronously
        result = asyncio.run(process_user_instruction(browse_instruction))
        
        if result:
            # Add agent response to history
            response = result.get("result", {}).get("response", "I couldn't browse to the specified URL.")
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun to update the UI
        st.rerun()
    
    # Display browse history
    if st.session_state.get("browse_history"):
        st.markdown("### Browse History")
        
        for i, entry in enumerate(reversed(st.session_state.browse_history)):
            if i >= 10:  # Only show last 10 entries
                break
                
            with st.expander(f"{entry['title']} - {entry['url']}", expanded=i==0):
                st.markdown(f"**Visited**: {entry['timestamp']}")
                
                if entry.get("summary"):
                    st.markdown(f"**Summary**: {entry['summary']}")
                
                # This is where we would display the screenshot
                if entry.get("screenshot"):
                    with st.container():
                        st.markdown("<div class='screenshot-container'>Screenshot</div>", unsafe_allow_html=True)
                        st.image(entry["screenshot"], use_column_width=True)
                
                # Re-browse button
                if st.button(f"Browse Again", key=f"rebrowse_{i}"):
                    # Create a browse instruction
                    browse_instruction = f"Browse to {entry['url']} and analyze the content"
                    
                    # Add to messages
                    st.session_state.messages.append({"role": "user", "content": browse_instruction})
                    
                    # Process instruction asynchronously
                    result = asyncio.run(process_user_instruction(browse_instruction))
                    
                    if result:
                        # Add agent response to history
                        response = result.get("result", {}).get("response", "I couldn't browse to the specified URL.")
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Rerun to update the UI
                    st.rerun()
    else:
        st.info("No browsing history yet. Enter a URL above to start browsing.")

# Function to display task history
def display_task_history():
    """Display the task history and details"""
    st.subheader("Tasks History & Management")
    
    # Simple task history display
    if not st.session_state.tasks:
        st.info("No tasks have been processed yet. Try asking the agent to perform a task in the Chat tab.")
    else:
        st.write(f"Total tasks processed: {len(st.session_state.tasks)}")
        
        # Group by task type
        task_types = {}
        for task in st.session_state.tasks:
            task_type = task["task_type"]
            if task_type not in task_types:
                task_types[task_type] = []
            task_types[task_type].append(task)
        
        # Display by type
        for task_type, tasks in task_types.items():
            st.subheader(f"{task_type.capitalize()} Tasks ({len(tasks)})")
            for i, task in enumerate(tasks):
                with st.expander(f"Task {i+1}: {task['task'][:50]}...", expanded=False):
                    st.write(f"**ID**: {task.get('task_id', 'N/A')}")
                    st.write(f"**Type**: {task['task_type']}")
                    st.write(f"**Time**: {task['execution_time']:.4f}s")
                    st.write(f"**Quantum**: {'Yes' if task['use_quantum'] else 'No'}")
                    st.write(f"**Browsing Required**: {'Yes' if task.get('requires_browsing', False) else 'No'}")
                    
                    # Task result display
                    st.markdown("##### Result")
                    result = task.get("result", {})
                    
                    # Format differently based on task type
                    if task["task_type"] == "search":
                        st.write(f"**Query**: {result.get('query', 'N/A')}")
                        if result.get('summary'):
                            st.write(f"**Summary**: {result['summary']}")
                        
                        if result.get('results'):
                            for j, r in enumerate(result['results'][:3]):
                                st.write(f"**Result {j+1}**: {r.get('title', 'No title')}")
                    
                    elif task["task_type"] == "factorization":
                        st.write(f"**Number**: {result.get('number', 'N/A')}")
                        if result.get('factors'):
                            st.write(f"**Factors**: {', '.join(map(str, result['factors']))}")
                        if result.get('prime_factors'):
                            st.write(f"**Prime Factors**: {', '.join(map(str, result['prime_factors']))}")
                    
                    elif task["task_type"] == "optimization":
                        st.write(f"**Objective Value**: {result.get('objective_value', 'N/A')}")
                        if result.get('solution'):
                            st.write("**Solution**:")
                            for k, v in result['solution'].items():
                                st.write(f"- {k}: {v}")
                    else:
                        # General or browser tasks
                        st.write(result.get('response', 'No detailed result available'))
                            
                    # Performance metrics
                    if st.session_state.show_debug:
                        st.markdown("##### Performance Metrics")
                        cols = st.columns(3)
                        with cols[0]:
                            st.metric("Quantum Time", f"{result.get('quantum_time', 0):.4f}s")
                        with cols[1]:
                            st.metric("Classical Time", f"{result.get('classical_time', 0):.4f}s")
                        with cols[2]:
                            if task['use_quantum'] and result.get('speedup'):
                                st.metric("Speedup", f"{result['speedup']:.2f}x")
                            else:
                                st.metric("Speedup", "N/A")

# Function to display performance metrics
def display_performance_metrics():
    """Display agent performance metrics and statistics"""
    st.subheader("Quantum-Classical Performance Metrics")
    
    # Get agent metrics
    metrics = {}
    if 'agent' in st.session_state:
        metrics = st.session_state.agent.get_metrics()
    
    if not metrics:
        st.info("No performance metrics available yet. Try performing some tasks first.")
        return
    
    # Overall metrics
    st.markdown("### Overall Performance")
    
    # Display key metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Total Tasks</div>
        </div>
        """.format(metrics.get("tasks_completed", 0)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Quantum Tasks</div>
        </div>
        """.format(metrics.get("quantum_accelerated_tasks", 0)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Browser Sessions</div>
        </div>
        """.format(metrics.get("browser_sessions", 0)), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.2f}x</div>
            <div class="metric-label">Avg Quantum Speedup</div>
        </div>
        """.format(metrics.get("average_quantum_speedup", 1.0)), unsafe_allow_html=True)
    
    # AI usage metrics
    st.markdown("### AI Model Usage")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Claude Interactions</div>
        </div>
        """.format(metrics.get("claude_interactions", 0)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">OpenAI Interactions</div>
        </div>
        """.format(metrics.get("openai_interactions", 0)), unsafe_allow_html=True)
    
    with col3:
        success_rate = 0
        if metrics.get("successful_interactions", 0) + metrics.get("failed_interactions", 0) > 0:
            success_rate = (metrics.get("successful_interactions", 0) / 
                           (metrics.get("successful_interactions", 0) + metrics.get("failed_interactions", 0)) * 100)
        
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.1f}%</div>
            <div class="metric-label">AI Success Rate</div>
        </div>
        """.format(success_rate), unsafe_allow_html=True)
    
    # Simulated quantum advantage chart if quantum tasks completed
    if metrics.get("quantum_accelerated_tasks", 0) > 0:
        st.markdown("### Quantum vs. Classical Performance")
        
        # Create sample data for demonstration
        task_sizes = list(range(10, 110, 10))
        classical_times = [0.1 * x for x in task_sizes]
        quantum_times = [0.1 * x**0.5 for x in task_sizes]  # Square root speedup
        
        # Create a chart comparing quantum vs. classical
        chart_data = {"Task Size": task_sizes}
        chart_data["Classical Time (s)"] = classical_times
        chart_data["Quantum Time (s)"] = quantum_times
        
        chart_df = pd.DataFrame(chart_data)
        st.line_chart(chart_df.set_index("Task Size"))
        
        st.markdown("""
        *Note: This chart illustrates the theoretical advantage of quantum computing 
        for certain classes of problems. Real-world performance may vary based on 
        specific problem characteristics and available quantum hardware.*
        """)
        
        # Application domains
        st.markdown("### Quantum Advantage by Application Domain")
        
        domains = {
            "Search": "Quadratic speedup (O(√N) vs O(N))",
            "Factorization": "Exponential speedup (Polynomial vs Exponential)",
            "Optimization": "Potential exponential speedup for certain problems",
            "Machine Learning": "Potential quadratic speedup for specific algorithms",
            "Simulation": "Exponential speedup for quantum system simulation"
        }
        
        for domain, advantage in domains.items():
            st.markdown(f"**{domain}**: {advantage}")

# Main function
def main():
    """Main application function"""
    # Run async initialization inside a sync function
    asyncio.run(initialize_agent_if_needed())
    
    # Main header
    st.markdown('<div class="main-header">Q3A: Quantum-Accelerated AI Agent</div>', unsafe_allow_html=True)
    
    # Display API key configuration
    display_api_key_form()
    
    # Display sidebar
    display_sidebar()
    
    # Content based on selected tab
    if st.session_state.current_tab == "chat":
        display_chat_interface()
    elif st.session_state.current_tab == "browser":
        display_browser_interface()
    elif st.session_state.current_tab == "tasks":
        display_task_history()
    elif st.session_state.current_tab == "performance":
        display_performance_metrics()

# Run the application
if __name__ == "__main__":
    main()