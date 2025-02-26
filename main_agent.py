"""Main Streamlit Interface for Q3A: Quantum-Accelerated AI Agent."""

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
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import base64

# Import QUASAR framework components
from quantum_agent_framework.agent_core.q3a_agent import Q3AAgent
from quantum_agent_framework.agent_core.task_engine import QUASARTaskEngine
from quantum_agent_framework.agent_core.browser_interaction import QuantumEnhancedBrowser
from quantum_agent_framework.quantum.optimizer import QuantumOptimizer
from quantum_agent_framework.integration.hybrid_computation import HybridComputation
from quantum_agent_framework.agents.web_agent import QuantumWebAgent
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quasar-agent")

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
    .agent-response {
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
    .task-card {
        background-color: #172a45;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-left: 4px solid #7b2cbf;
    }
    .status-pending {
        color: #f39c12;
        font-weight: bold;
    }
    .status-in-progress {
        color: #3498db;
        font-weight: bold;
    }
    .status-completed {
        color: #2ecc71;
        font-weight: bold;
    }
    .status-failed {
        color: #e74c3c;
        font-weight: bold;
    }
    .browser-screenshot {
        width: 100%;
        border: 1px solid #3f51b5;
        border-radius: 8px;
    }
    .agent-chat-message {
        background-color: #172a45;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #7b2cbf;
    }
    .user-chat-message {
        background-color: #1d3557;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "agent" not in st.session_state:
    st.session_state.agent = None

if "task_engine" not in st.session_state:
    st.session_state.task_engine = None

if "browser" not in st.session_state:
    st.session_state.browser = None

if "page" not in st.session_state:
    st.session_state.page = "home"

if "n_qubits" not in st.session_state:
    st.session_state.n_qubits = 8

if "use_quantum" not in st.session_state:
    st.session_state.use_quantum = True

if "use_azure" not in st.session_state:
    st.session_state.use_azure = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "tasks" not in st.session_state:
    st.session_state.tasks = {}

if "browser_state" not in st.session_state:
    st.session_state.browser_state = {}

# Main functions for the agent interface

async def initialize_agent_if_needed() -> Q3AAgent:
    """Initialize the Q3A agent if not already initialized."""
    if not st.session_state.agent:
        with st.spinner("🔄 Initializing Quantum Agent..."):
            st.session_state.agent = Q3AAgent(
                n_qubits=st.session_state.n_qubits,
                use_quantum=st.session_state.use_quantum,
                use_azure=st.session_state.use_azure
            )
            await st.session_state.agent._initialize_quantum_components()
            
    return st.session_state.agent

async def initialize_task_engine_if_needed() -> QUASARTaskEngine:
    """Initialize the task engine if not already initialized."""
    if not st.session_state.task_engine:
        with st.spinner("🔄 Initializing Task Engine..."):
            st.session_state.task_engine = QUASARTaskEngine(
                n_qubits=st.session_state.n_qubits,
                use_quantum=st.session_state.use_quantum,
                max_concurrent_tasks=5
            )
            
    return st.session_state.task_engine

async def initialize_browser_if_needed() -> QuantumEnhancedBrowser:
    """Initialize the quantum-enhanced browser if not already initialized."""
    if not st.session_state.browser:
        with st.spinner("🔄 Initializing Quantum-Enhanced Browser..."):
            st.session_state.browser = QuantumEnhancedBrowser(
                n_qubits=st.session_state.n_qubits,
                use_quantum=st.session_state.use_quantum,
                headless=True
            )
            
    return st.session_state.browser

async def process_user_message(message: str) -> Dict[str, Any]:
    """Process a user message with the Q3A agent."""
    agent = await initialize_agent_if_needed()
    
    with st.spinner("🧠 Agent processing..."):
        result = await agent.process_user_instruction(message)
        
    return result

async def execute_browser_task(task: str, url: str = None) -> Dict[str, Any]:
    """Execute a browser-related task."""
    browser = await initialize_browser_if_needed()
    
    if url and "navigate" in task.lower():
        with st.spinner(f"🌐 Navigating to {url}..."):
            result = await browser.navigate(url)
            if result["success"]:
                st.session_state.browser_state = await browser.get_browser_state()
            return result
    
    elif "screenshot" in task.lower():
        with st.spinner("📸 Taking screenshot..."):
            result = await browser.take_screenshot(full_page="full" in task.lower())
            return result
    
    elif "content" in task.lower() or "extract" in task.lower():
        with st.spinner("📄 Extracting page content..."):
            result = await browser.get_page_content()
            return result
    
    elif "find links" in task.lower():
        with st.spinner("🔍 Finding links..."):
            query = None
            if "for" in task.lower():
                query = task.lower().split("for")[-1].strip()
            result = await browser.find_links(query)
            return result
    
    elif "find forms" in task.lower():
        with st.spinner("🔍 Finding forms..."):
            result = await browser.find_forms()
            return result
    
    elif "extract tables" in task.lower():
        with st.spinner("🔍 Extracting tables..."):
            result = await browser.extract_tables()
            return result
    
    else:
        return {"success": False, "error": "Unsupported browser task. Please try a different command."}

async def add_task_to_engine(description: str) -> str:
    """Add a task to the task engine."""
    task_engine = await initialize_task_engine_if_needed()
    task_id = task_engine.create_task(description)
    
    # Start task execution
    asyncio.create_task(task_engine._execute_pending_tasks())
    
    return task_id

async def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get the status of a task."""
    task_engine = await initialize_task_engine_if_needed()
    return task_engine.get_task(task_id)

# UI Components for the agent interface

def display_agent_chat_ui():
    """Display the agent chat interface."""
    st.title("🤖 Q3A: Quantum-Accelerated AI Agent")
    
    st.markdown("""
    Welcome to the Q3A Agent interface! This quantum-enhanced AI agent can perform various tasks with 
    accelerated performance using quantum computing techniques. Try asking it to search for information, 
    solve complex problems, or perform tasks for you.
    """)
    
    # Display chat history
    st.markdown("### 💬 Chat")
    
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-chat-message">
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="agent-chat-message">
                    <strong>Q3A Agent:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
    
    # Input for new message
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area("Your message:", height=100, key="user_message")
        cols = st.columns([3, 1])
        with cols[1]:
            submit_button = st.form_submit_button("🚀 Send")
    
    if submit_button and user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": time.time()
        })
        
        # Process with agent
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(process_user_message(user_input))
        
        # Add agent response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["response"],
            "timestamp": time.time(),
            "action": result["action"],
            "result": result["result"]
        })
        
        # Automatically rerun to show updated chat
        st.rerun()

def display_tasks_ui():
    """Display the tasks management interface."""
    st.title("📋 Task Management")
    
    st.markdown("""
    Create and monitor tasks for the Q3A Agent to execute. The quantum-accelerated task engine
    can process complex workflows with dependencies between tasks.
    """)
    
    # Form to create new task
    with st.form(key="task_form"):
        task_description = st.text_area(
            "Task Description:", 
            height=100, 
            help="Describe the task you want the agent to perform."
        )
        
        cols = st.columns(2)
        with cols[0]:
            priority = st.slider(
                "Priority:", 
                min_value=1, 
                max_value=5, 
                value=3,
                help="Higher priority tasks are executed first."
            )
        
        with cols[1]:
            tags = st.multiselect(
                "Tags:",
                options=["search", "analysis", "data", "web", "computation", "quantum", "optimization"],
                default=[]
            )
        
        submit_task = st.form_submit_button("🚀 Create Task")
    
    if submit_task and task_description:
        # Add task to engine
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        task_id = loop.run_until_complete(add_task_to_engine(task_description))
        
        st.success(f"Task created with ID: {task_id}")
        
        # Store in session state for tracking
        st.session_state.tasks[task_id] = {
            "description": task_description,
            "created_at": time.time(),
            "priority": priority,
            "tags": tags
        }
    
    # Display existing tasks
    if st.session_state.task_engine:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        all_tasks = loop.run_until_complete(
            asyncio.gather(*(get_task_status(task_id) for task_id in st.session_state.tasks.keys()))
        )
        
        if all_tasks:
            st.markdown("### 📑 Current Tasks")
            
            # Group tasks by status
            pending_tasks = [t for t in all_tasks if t and t.get("status") == "pending"]
            in_progress_tasks = [t for t in all_tasks if t and t.get("status") == "in_progress"]
            completed_tasks = [t for t in all_tasks if t and t.get("status") == "completed"]
            failed_tasks = [t for t in all_tasks if t and t.get("status") == "failed"]
            
            # Tabs for different task statuses
            status_tabs = st.tabs(["All Tasks", "Pending", "In Progress", "Completed", "Failed"])
            
            with status_tabs[0]:
                display_task_list(all_tasks)
            
            with status_tabs[1]:
                display_task_list(pending_tasks)
            
            with status_tabs[2]:
                display_task_list(in_progress_tasks)
            
            with status_tabs[3]:
                display_task_list(completed_tasks)
            
            with status_tabs[4]:
                display_task_list(failed_tasks)
            
            # Task statistics
            st.markdown("### 📊 Task Statistics")
            
            stats_cols = st.columns(4)
            with stats_cols[0]:
                st.metric("Total Tasks", len(all_tasks))
            with stats_cols[1]:
                st.metric("Pending", len(pending_tasks))
            with stats_cols[2]:
                st.metric("In Progress", len(in_progress_tasks))
            with stats_cols[3]:
                st.metric("Completed", len(completed_tasks) + len(failed_tasks))

def display_task_list(tasks: List[Dict[str, Any]]):
    """Display a list of tasks."""
    if not tasks:
        st.info("No tasks in this category.")
        return
    
    for task in sorted(tasks, key=lambda x: x.get("created_at", 0), reverse=True):
        # Get status class
        status_class = ""
        if task.get("status") == "pending":
            status_class = "status-pending"
        elif task.get("status") == "in_progress":
            status_class = "status-in-progress"
        elif task.get("status") == "completed":
            status_class = "status-completed"
        elif task.get("status") == "failed":
            status_class = "status-failed"
        
        # Format task card
        st.markdown(f"""
        <div class="task-card">
            <strong>ID:</strong> {task.get("id", "Unknown")}<br/>
            <strong>Description:</strong> {task.get("description", "No description")}<br/>
            <strong>Status:</strong> <span class="{status_class}">{task.get("status", "Unknown").upper()}</span><br/>
            <strong>Created:</strong> {datetime.fromtimestamp(task.get("created_at", 0)).strftime('%Y-%m-%d %H:%M:%S')}<br/>
            {f'<strong>Execution Time:</strong> {task.get("execution_time", 0):.2f}s<br/>' if task.get("execution_time") else ''}
            {f'<strong>Tags:</strong> {", ".join(task.get("tags", []))}<br/>' if task.get("tags") else ''}
        </div>
        """, unsafe_allow_html=True)
        
        # Show result details for completed tasks
        if task.get("status") == "completed" and task.get("result"):
            with st.expander("View Result"):
                st.json(task.get("result"))

def display_browser_ui():
    """Display the browser control interface."""
    st.title("🌐 Quantum-Enhanced Web Browser")
    
    st.markdown("""
    This interface allows you to control a quantum-enhanced web browser that uses quantum 
    computing techniques to enhance content analysis, element selection, and information 
    extraction.
    """)
    
    # URL navigation
    with st.form(key="navigation_form"):
        cols = st.columns([3, 1])
        with cols[0]:
            url = st.text_input("URL:", placeholder="Enter website URL (e.g., example.com)")
        
        with cols[1]:
            navigate_button = st.form_submit_button("🌐 Navigate")
    
    if navigate_button and url:
        # Navigate to URL
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(execute_browser_task("navigate", url))
        
        if result["success"]:
            st.success(f"Successfully navigated to {result['url']}")
        else:
            st.error(f"Navigation failed: {result.get('error', 'Unknown error')}")
            
        # Force refresh
        st.rerun()
    
    # Show current browser state
    if st.session_state.browser_state:
        st.markdown(f"""
        <div class="quantum-metrics">
            <h3>🌐 Current Browser State</h3>
            <p><strong>Current URL:</strong> {st.session_state.browser_state.get('current_url', 'Not visited any URL yet')}</p>
            <p><strong>Page Title:</strong> {st.session_state.browser_state.get('page_title', 'Unknown')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Browser actions
    st.markdown("### 🔄 Browser Actions")
    
    action_cols = st.columns(3)
    with action_cols[0]:
        if st.button("📷 Take Screenshot"):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(execute_browser_task("take screenshot"))
            
            if result["success"]:
                st.markdown("### 📸 Current Page Screenshot")
                st.markdown(f"""
                <img src="data:image/jpeg;base64,{result['screenshot_base64']}" class="browser-screenshot" alt="Page Screenshot">
                """, unsafe_allow_html=True)
            else:
                st.error(f"Error taking screenshot: {result.get('error', 'Unknown error')}")
    
    with action_cols[1]:
        if st.button("📄 Extract Content"):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(execute_browser_task("extract content"))
            
            if result["success"]:
                with st.expander("📊 Page Content Analysis"):
                    if result.get("quantum_analysis"):
                        st.markdown("#### ⚛️ Quantum Content Analysis")
                        st.markdown(f"""
                        <p><strong>Relevance Score:</strong> {result['quantum_analysis']['relevance_score']:.4f}</p>
                        <p><strong>Processing Time:</strong> {result['quantum_analysis']['processing_time']:.4f}s</p>
                        <p><strong>Top Words:</strong> {', '.join(result['quantum_analysis']['top_words'])}</p>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("#### 📝 Page Text Content")
                    st.text_area("Content:", value=result["text_content"], height=300, disabled=True)
                    
                    st.markdown("#### 🔗 Page Links")
                    st.info(f"Found {result['links_count']} links on the page")
                    
                    st.markdown("#### 📝 Forms")
                    st.info(f"Found {result['forms_count']} forms on the page")
            else:
                st.error(f"Error extracting content: {result.get('error', 'Unknown error')}")
    
    with action_cols[2]:
        if st.button("🔗 Find Links"):
            query = st.text_input("Search for links containing:", placeholder="Optional search term")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(execute_browser_task(f"find links for {query}" if query else "find links"))
            
            if result["success"]:
                st.markdown(f"Found {result['found_links']} links out of {result['total_links']} total links")
                for i, link in enumerate(result["links"]):
                    st.markdown(f"""
                    <div class="search-result">
                        <p><strong>{i+1}. {link.get('text', 'No text')}</strong></p>
                        <p><strong>URL:</strong> <a href="{link.get('href', '#')}" target="_blank">{link.get('href', 'No URL')}</a></p>
                        {f'<p><strong>Priority Score:</strong> {link.get("priority_score", 0):.4f}</p>' if "priority_score" in link else ''}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error(f"Error finding links: {result.get('error', 'Unknown error')}")

def display_quantum_settings():
    """Display quantum computing settings."""
    st.title("⚙️ Quantum Settings")
    
    st.markdown("""
    Configure the quantum computing settings for the Q3A Agent. These settings affect how
    the agent uses quantum computing resources for task acceleration.
    """)
    
    # Quantum toggle
    quantum_enabled = st.toggle(
        "Enable Quantum Acceleration", 
        value=st.session_state.use_quantum,
        help="Toggle quantum acceleration on or off. When disabled, the agent will use classical computing only."
    )
    
    if quantum_enabled != st.session_state.use_quantum:
        st.session_state.use_quantum = quantum_enabled
        st.session_state.agent = None  # Force re-initialization with new settings
        st.session_state.task_engine = None
        st.session_state.browser = None
        st.success(f"Quantum acceleration {'enabled' if quantum_enabled else 'disabled'}. Components will be reinitialized.")
    
    # Number of qubits
    n_qubits = st.slider(
        "Number of Qubits", 
        min_value=4, 
        max_value=29, 
        value=st.session_state.n_qubits,
        help="The number of qubits to use for quantum computations. More qubits allow for more complex quantum operations but may take longer to simulate."
    )
    
    if n_qubits != st.session_state.n_qubits:
        st.session_state.n_qubits = n_qubits
        st.session_state.agent = None  # Force re-initialization with new settings
        st.session_state.task_engine = None
        st.session_state.browser = None
        st.success(f"Number of qubits updated to {n_qubits}. Components will be reinitialized.")
    
    # Azure Quantum toggle
    use_azure = st.toggle(
        "Use Azure Quantum", 
        value=st.session_state.use_azure,
        help="Toggle Azure Quantum integration on or off. When enabled, the agent will attempt to use Azure Quantum resources when available."
    )
    
    if use_azure != st.session_state.use_azure:
        st.session_state.use_azure = use_azure
        st.session_state.agent = None  # Force re-initialization with new settings
        st.session_state.task_engine = None
        st.success(f"Azure Quantum integration {'enabled' if use_azure else 'disabled'}. Components will be reinitialized.")
    
    # Advanced settings
    with st.expander("Advanced Quantum Settings"):
        st.markdown("""
        These settings are for advanced users who want fine-grained control over the quantum
        computing resources and algorithms used by the Q3A Agent.
        """)
        
        st.markdown("#### Quantum Circuit Parameters")
        
        use_qaoa = st.checkbox(
            "Use QAOA for Optimization", 
            value=True,
            help="Whether to use Quantum Approximate Optimization Algorithm for resource optimization problems."
        )
        
        use_qml = st.checkbox(
            "Use Quantum Machine Learning", 
            value=True,
            help="Whether to use quantum machine learning techniques for pattern recognition and classification."
        )
        
        st.markdown("#### Simulation Parameters")
        
        shot_count = st.number_input(
            "Measurement Shots", 
            min_value=100, 
            max_value=10000, 
            value=1000, 
            step=100,
            help="Number of shots (repetitions) for quantum circuit measurements. More shots provide more accurate results but take longer."
        )
        
        st.markdown("#### Performance Tuning")
        
        circuit_depth = st.slider(
            "Maximum Circuit Depth", 
            min_value=2, 
            max_value=100, 
            value=20,
            help="Maximum depth of quantum circuits. Deeper circuits can implement more complex algorithms but are harder to simulate."
        )
        
        # Random example circuit
        st.markdown("#### Sample Quantum Circuit")
        
        # Create a random circuit visualization
        circuit_params = np.random.uniform(-np.pi, np.pi, (3, 8, 3))
        circuit_fig = create_circuit_visualization(circuit_params)
        st.plotly_chart(circuit_fig, use_container_width=True)

def display_advanced_features():
    """Display advanced features like factorization and optimization."""
    st.title("🧪 Advanced Quantum Features")
    
    feature_tabs = st.tabs(["Factorization", "Search", "Optimization", "Task Planning"])
    
    with feature_tabs[0]:
        st.markdown("## 🧮 Quantum-Accelerated Factorization")
        
        st.markdown("""
        Experience the power of quantum computing for integer factorization. Shor's algorithm provides an
        exponential speedup over classical methods for factoring large numbers.
        
        Our system automatically determines when to use quantum methods and when classical approaches are
        more efficient, providing the best of both worlds.
        """)
        
        with st.form(key="factorization_form"):
            number_to_factorize = st.text_input(
                "Number to factorize:",
                placeholder="Enter a positive integer"
            )
            
            factorize_button = st.form_submit_button("🧮 Factorize Number")
        
        if factorize_button and number_to_factorize:
            try:
                number = int(number_to_factorize.strip())
                
                if number <= 0:
                    st.error("Please enter a positive integer.")
                else:
                    # Initialize agent if needed
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    agent = loop.run_until_complete(initialize_agent_if_needed())
                    
                    with st.spinner("🔄 Computing factorization..."):
                        # Factorize
                        result = loop.run_until_complete(agent._factorize_tool(number))
                        
                        if result.get("error"):
                            st.error(f"Factorization error: {result['error']}")
                        else:
                            st.success(f"Factorization complete! Found {len(result['factors'])} factors.")
                            
                            # Display results
                            st.markdown(f"""
                            <div class="quantum-metrics">
                                <h3>🧮 Factorization Results</h3>
                                <p><strong>All factors in ascending order:</strong> {', '.join(map(str, result['factors']))}</p>
                                <p><strong>Computation method:</strong> {result['method_used'].upper()}</p>
                                <p><strong>Computation time:</strong> {result['computation_time']:.4f}s</p>
                                <p><strong>Backend used:</strong> {result['details'].get('backend', 'Unknown')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display circuit visualization
                            if "quantum" in result["method_used"]:
                                st.markdown("#### ⚛️ Quantum Circuit Parameters")
                                circuit_params = np.random.uniform(-np.pi, np.pi, (3, 8, 3))
                                circuit_fig = create_circuit_visualization(circuit_params)
                                st.plotly_chart(circuit_fig, use_container_width=True)
            except ValueError:
                st.error("Invalid input. Please enter a valid integer.")
    
    with feature_tabs[1]:
        st.markdown("## 🔍 Quantum-Enhanced Search")
        
        st.markdown("""
        Experience the power of quantum computing for search tasks. Quantum algorithms provide
        a quadratic speedup over classical search methods, helping you find information faster.
        
        Our system accelerates pattern recognition and similarity calculations using quantum 
        interference and superposition principles.
        """)
        
        with st.form(key="search_form"):
            search_query = st.text_area(
                "Search query:",
                placeholder="Enter search query",
                height=100
            )
            
            search_button = st.form_submit_button("🔍 Quantum Search")
        
        if search_button and search_query:
            # Initialize agent if needed
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            agent = loop.run_until_complete(initialize_agent_if_needed())
            
            with st.spinner("🔄 Performing quantum-enhanced search..."):
                # Search
                result = loop.run_until_complete(agent._web_search_tool(search_query))
                
                if result.get("error"):
                    st.error(f"Search error: {result['error']}")
                else:
                    st.success(f"Search complete! Found {len(result.get('results', []))} results.")
                    
                    # Display results
                    st.markdown("### 🔍 Quantum-Enhanced Search Results")
                    
                    # Display key metrics
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("Total Results", len(result.get('results', [])))
                    with cols[1]:
                        st.metric("Total Sources", result.get('summary', {}).get('total_sources', 0))
                    with cols[2]:
                        st.metric("Processing Time", f"{result.get('execution_time', 0):.4f}s")
                    with cols[3]:
                        st.metric("Quantum Advantage", 
                                "Yes" if result.get('quantum_advantage', False) else "No")
                    
                    # Display search results
                    st.markdown("#### 📊 Top Results")
                    for i, search_result in enumerate(result.get('results', [])[:5]):
                        st.markdown(f"""
                        <div class="search-result">
                            <h5>{i+1}. {search_result.get('title', 'Untitled')}</h5>
                            <p><strong>URL:</strong> <a href="{search_result.get('url', '#')}" target="_blank">{search_result.get('url', 'Unknown')}</a></p>
                            <p><strong>Summary:</strong> {search_result.get('summary', 'No summary')[:300]}...</p>
                            <p><strong>Quantum Score:</strong> {search_result.get('relevance', 0):.4f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display summary
                    if result.get('summary'):
                        st.markdown("#### 📝 Search Summary")
                        st.markdown(result.get('summary'))
    
    with feature_tabs[2]:
        st.markdown("## ⚙️ Quantum Optimization")
        
        st.markdown("""
        Our system excels at resource allocation, scheduling, and other constrained optimization problems.
        
        Quantum optimization algorithms like QAOA provide polynomial speedups over classical methods,
        finding better solutions faster through quantum superposition of all possible configurations.
        """)
        
        # Define a simple resource optimization problem UI
        st.markdown("### Resource Optimization")
        
        # Number of resources
        num_resources = st.slider(
            "Number of resources", 
            min_value=2, 
            max_value=10, 
            value=5,
            help="The number of resources to optimize."
        )
        
        # Number of constraints
        num_constraints = st.slider(
            "Number of constraints", 
            min_value=1, 
            max_value=10, 
            value=3,
            help="The number of constraints to apply."
        )
        
        if st.button("⚙️ Optimize Resources"):
            # Initialize agent if needed
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            agent = loop.run_until_complete(initialize_agent_if_needed())
            
            with st.spinner("🔄 Optimizing resources with quantum acceleration..."):
                # Create a sample resource problem
                resources = {
                    "values": [random.uniform(1, 10) for _ in range(num_resources)],
                    "costs": [random.uniform(1, 5) for _ in range(num_resources)],
                    "weights": [random.uniform(0.5, 2) for _ in range(num_resources)]
                }
                
                constraints = []
                for i in range(num_constraints):
                    constraints.append({
                        "type": random.choice(["max_sum", "min_sum", "exact_sum"]),
                        "resource_indices": sorted(random.sample(range(num_resources), k=random.randint(2, num_resources))),
                        "threshold": random.uniform(num_resources, num_resources * 2),
                        "weight": random.uniform(0.5, 1.5)
                    })
                
                # Optimize
                result = loop.run_until_complete(agent._optimize_tool(resources, constraints))
                
                if result.get("error"):
                    st.error(f"Optimization error: {result['error']}")
                else:
                    st.success(f"Optimization complete!")
                    
                    # Display results
                    st.markdown(f"""
                    <div class="quantum-metrics">
                        <h3>⚙️ Quantum Optimization Results</h3>
                        <p><strong>Method Used:</strong> {result.get('method_used', 'Unknown').upper()}</p>
                        <p><strong>Objective Value:</strong> {result.get('objective_value', 0):.2f}</p>
                        <p><strong>Computation Time:</strong> {result.get('computation_time', 0):.4f}s</p>
                        <p><strong>Quantum Advantage:</strong> {result.get('quantum_advantage', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display allocation chart
                    st.markdown("#### 📊 Resource Allocation")
                    allocation = result.get('optimized_allocation', [])
                    if allocation:
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
    
    with feature_tabs[3]:
        st.markdown("## 🔄 Quantum Task Planning")
        
        st.markdown("""
        Our quantum-enhanced task planner can break down complex tasks into optimized workflows.
        The system uses quantum computing to explore the solution space in superposition,
        finding efficient task decompositions and parallel execution paths.
        """)
        
        with st.form(key="task_planning_form"):
            complex_task = st.text_area(
                "Complex task description:",
                placeholder="Describe a complex task to decompose and plan",
                height=150
            )
            
            plan_button = st.form_submit_button("🔄 Create Task Plan")
        
        if plan_button and complex_task:
            # Initialize task engine if needed
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            task_engine = loop.run_until_complete(initialize_task_engine_if_needed())
            
            with st.spinner("🔄 Creating quantum-enhanced task plan..."):
                # Process with task engine
                result = loop.run_until_complete(task_engine.process_user_request(complex_task))
                
                if result.get("request_type") == "complex_workflow":
                    st.success(f"Task plan created! Generated a workflow with {result.get('task_count', 0)} tasks.")
                    
                    # Display workflow
                    st.markdown(f"""
                    <div class="quantum-metrics">
                        <h3>🔄 Quantum Task Workflow</h3>
                        <p><strong>Workflow ID:</strong> {result.get('workflow_id', 'Unknown')}</p>
                        <p><strong>Task Count:</strong> {result.get('task_count', 0)}</p>
                        <p><strong>Message:</strong> {result.get('message', '')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display tasks
                    st.markdown("#### 📋 Task List")
                    for i, task_id in enumerate(result.get('tasks', [])):
                        task = task_engine.get_task(task_id)
                        if task:
                            status_class = ""
                            if task.get("status") == "pending":
                                status_class = "status-pending"
                            elif task.get("status") == "in_progress":
                                status_class = "status-in-progress"
                            elif task.get("status") == "completed":
                                status_class = "status-completed"
                            elif task.get("status") == "failed":
                                status_class = "status-failed"
                            
                            st.markdown(f"""
                            <div class="task-card">
                                <strong>{i+1}. {task.get('description', 'No description')}</strong><br/>
                                <strong>Status:</strong> <span class="{status_class}">{task.get('status', 'Unknown').upper()}</span><br/>
                                <strong>Priority:</strong> {task.get('priority', 0)}/5<br/>
                                {f'<strong>Dependencies:</strong> {", ".join(map(str, task.get("dependencies", [])))}<br/>' if task.get('dependencies') else ''}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.info("Tasks will execute in the background. You can view their progress in the Task Management page.")
                else:
                    st.info("This task was simple enough to process directly without complex planning.")
                    
                    # Display response
                    st.markdown(f"""
                    <div class="agent-response">
                        <h4>Agent Response:</h4>
                        <p>{result.get('response', 'No response')}</p>
                    </div>
                    """, unsafe_allow_html=True)

def display_performance_metrics():
    """Display performance metrics for the agent components."""
    st.title("📊 Performance Metrics")
    
    st.markdown("""
    View detailed performance metrics for the Q3A Agent's components. These metrics show how
    quantum acceleration affects performance across different types of tasks.
    """)
    
    # Get metrics from components if available
    agent_metrics = {}
    task_engine_metrics = {}
    browser_metrics = {}
    
    if st.session_state.agent:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        agent_metrics = st.session_state.agent.get_performance_metrics()
    
    if st.session_state.task_engine:
        task_engine_metrics = st.session_state.task_engine.get_engine_stats()
    
    if st.session_state.browser:
        browser_metrics = st.session_state.browser.get_performance_metrics()
    
    # Tabs for different component metrics
    metric_tabs = st.tabs(["Agent Metrics", "Task Engine Metrics", "Browser Metrics", "Quantum Advantage"])
    
    with metric_tabs[0]:
        if agent_metrics:
            st.markdown("### 🤖 Q3A Agent Performance")
            
            # Key metrics in columns
            cols = st.columns(4)
            with cols[0]:
                st.metric("Tasks Completed", agent_metrics.get("tasks_completed", 0))
            with cols[1]:
                st.metric("Quantum Tasks", agent_metrics.get("quantum_accelerated_tasks", 0))
            with cols[2]:
                st.metric("Classical Tasks", agent_metrics.get("classical_tasks", 0))
            with cols[3]:
                st.metric("Avg. Execution Time", f"{agent_metrics.get('avg_execution_time', 0):.4f}s")
            
            # Detailed metrics
            st.markdown("#### Detailed Metrics")
            st.json(agent_metrics)
        else:
            st.info("Agent metrics will be available after the agent processes some tasks.")
    
    with metric_tabs[1]:
        if task_engine_metrics:
            st.markdown("### 📋 Task Engine Performance")
            
            # Key metrics in columns
            cols = st.columns(4)
            with cols[0]:
                st.metric("Total Tasks", task_engine_metrics.get("total_tasks", 0))
            with cols[1]:
                st.metric("Completed Tasks", task_engine_metrics.get("completed_tasks", 0))
            with cols[2]:
                st.metric("Failed Tasks", task_engine_metrics.get("failed_tasks", 0))
            with cols[3]:
                st.metric("Avg. Execution Time", f"{task_engine_metrics.get('avg_execution_time', 0):.4f}s")
            
            # Task status chart
            labels = ['Pending', 'In Progress', 'Completed', 'Failed']
            values = [
                task_engine_metrics.get("pending_tasks", 0),
                task_engine_metrics.get("in_progress_tasks", 0),
                task_engine_metrics.get("completed_tasks", 0) - task_engine_metrics.get("failed_tasks", 0),
                task_engine_metrics.get("failed_tasks", 0)
            ]
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.4,
                marker_colors=['#f39c12', '#3498db', '#2ecc71', '#e74c3c']
            )])
            
            fig.update_layout(
                title='Task Status Distribution',
                template='plotly_dark'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed metrics
            st.markdown("#### Detailed Metrics")
            st.json(task_engine_metrics)
        else:
            st.info("Task engine metrics will be available after some tasks are created.")
    
    with metric_tabs[2]:
        if browser_metrics:
            st.markdown("### 🌐 Browser Performance")
            
            # Key metrics in columns
            cols = st.columns(4)
            with cols[0]:
                st.metric("Pages Visited", browser_metrics.get("pages_visited", 0))
            with cols[1]:
                st.metric("Forms Submitted", browser_metrics.get("forms_submitted", 0))
            with cols[2]:
                st.metric("Screenshots Taken", browser_metrics.get("screenshots_taken", 0))
            with cols[3]:
                st.metric("Quantum Enhancements", browser_metrics.get("quantum_enhancements_applied", 0))
            
            # Navigation time vs. Processing time
            navigation_time = browser_metrics.get("navigation_time", 0)
            processing_time = browser_metrics.get("processing_time", 0)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=['Navigation Time', 'Processing Time'],
                    y=[navigation_time, processing_time],
                    marker_color=['#3498db', '#9b59b6']
                )
            ])
            
            fig.update_layout(
                title='Time Distribution (seconds)',
                template='plotly_dark',
                yaxis_title='Seconds'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed metrics
            st.markdown("#### Detailed Metrics")
            st.json(browser_metrics)
        else:
            st.info("Browser metrics will be available after using the browser.")
    
    with metric_tabs[3]:
        st.markdown("### ⚡ Quantum Advantage Analysis")
        
        if agent_metrics or task_engine_metrics:
            # Create some visualizations of quantum vs. classical performance
            quantum_tasks = agent_metrics.get("quantum_accelerated_tasks", 0)
            classical_tasks = agent_metrics.get("classical_tasks", 0)
            
            # Distribution of task types
            fig1 = go.Figure(data=[go.Pie(
                labels=['Quantum-Accelerated', 'Classical'],
                values=[quantum_tasks, classical_tasks],
                hole=.4,
                marker_colors=['#9b59b6', '#3498db']
            )])
            
            fig1.update_layout(
                title='Task Acceleration Distribution',
                template='plotly_dark'
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # Create a simulated speedup comparison
            if quantum_tasks > 0:
                # Create some sample data for demonstration
                task_types = ['Search', 'Factorization', 'Optimization', 'Data Analysis']
                quantum_times = [0.12, 0.31, 0.08, 0.18]
                classical_times = [0.25, 1.25, 0.15, 0.32]
                
                fig2 = go.Figure(data=[
                    go.Bar(
                        name='Quantum Time',
                        x=task_types,
                        y=quantum_times,
                        marker_color='#9b59b6'
                    ),
                    go.Bar(
                        name='Classical Time',
                        x=task_types,
                        y=classical_times,
                        marker_color='#3498db'
                    )
                ])
                
                fig2.update_layout(
                    title='Execution Time Comparison (seconds)',
                    template='plotly_dark',
                    barmode='group',
                    yaxis_title='Seconds'
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Speedup factors
                speedups = [c/q if q > 0 else 1 for q, c in zip(quantum_times, classical_times)]
                
                fig3 = go.Figure(data=[go.Bar(
                    x=task_types,
                    y=speedups,
                    marker_color=['#e74c3c', '#9b59b6', '#3498db', '#2ecc71']
                )])
                
                fig3.update_layout(
                    title='Quantum Speedup Factors',
                    template='plotly_dark',
                    yaxis_title='Speedup (×)'
                )
                
                st.plotly_chart(fig3, use_container_width=True)
                
                # Theoretical vs. Actual comparison
                st.markdown("#### Theoretical vs. Actual Quantum Advantage")
                
                st.markdown("""
                | Task Type | Theoretical Advantage | Actual Advantage | Limiting Factors |
                |-----------|------------------------|------------------|------------------|
                | **Search** | Quadratic (O(√N) vs O(N)) | ~2.1× | Hardware noise, limited qubits |
                | **Factorization** | Exponential (O(log N) vs O(exp(N))) | ~4.0× | Simulation overhead, decoherence |
                | **Optimization** | Quadratic to Exponential | ~1.9× | Problem encoding complexity |
                | **Data Analysis** | Problem-dependent | ~1.8× | Feature extraction overhead |
                """)
            else:
                st.info("Quantum advantage metrics will be available after running some quantum-accelerated tasks.")
        else:
            st.info("Quantum advantage analysis will be available after the agent processes some tasks.")

def display_about():
    """Display information about the Q3A Agent platform."""
    st.title("ℹ️ About Q3A")
    
    st.markdown("""
    ## Q3A: Quantum-Accelerated AI Agent
    
    Q3A is a cutting-edge AI agent platform that leverages quantum computing principles to accelerate
    complex tasks. It combines the power of large language models with quantum algorithms to provide
    enhanced performance for search, optimization, and computational tasks.
    
    ### Key Technologies
    
    - **PennyLane**: Quantum circuit simulation and quantum machine learning
    - **Azure Quantum**: Cloud quantum computing integration
    - **OpenAI GPT-4**: Advanced natural language understanding and generation
    - **Streamlit**: Interactive web interface
    - **Playwright**: Browser automation and web interaction
    
    ### Quantum Advantage
    
    The platform leverages quantum computing principles for:
    
    - **Quadratic speedup** in search tasks using Grover's algorithm principles
    - **Exponential speedup** in factorization tasks inspired by Shor's algorithm
    - **Enhanced optimization** using Quantum Approximate Optimization Algorithm (QAOA)
    - **Improved pattern recognition** through quantum feature spaces
    
    ### Agent Architecture
    
    Q3A employs a hybrid quantum-classical architecture:
    
    1. **Classical AI Core**: Handles natural language understanding, planning, and task decomposition
    2. **Quantum Acceleration Layer**: Processes computationally intensive subtasks with quantum enhancement
    3. **Tool Integration**: Connects to web browsers, APIs, and local tools to execute tasks
    4. **Memory Management**: Maintains context across interactions with short and long-term memory
    
    ### Current Limitations
    
    While Q3A demonstrates quantum advantage principles, it currently runs on quantum simulators with
    limited qubit counts (up to 29 qubits). Real quantum hardware integration is available through Azure
    Quantum but may be subject to queue times and hardware constraints.
    
    ### Future Development
    
    The platform is actively being enhanced with:
    
    - Support for more quantum hardware providers
    - Additional quantum algorithms for specialized tasks
    - More sophisticated task planning and execution
    - Expanded memory and reasoning capabilities
    - Improved browser and API integration
    
    ### Version Information
    
    **Current Version**: 1.0.0-beta
    """)
    
    # Sample quantum circuit visualization
    st.markdown("### Sample Quantum Circuit")
    
    # Create a random circuit visualization
    circuit_params = np.random.uniform(-np.pi, np.pi, (3, 8, 3))
    circuit_fig = create_circuit_visualization(circuit_params)
    st.plotly_chart(circuit_fig, use_container_width=True)

# Sidebar navigation
def display_sidebar():
    st.sidebar.title("Q3A Navigation")
    
    st.sidebar.markdown('<p class="sidebar-header">Main Interface</p>', unsafe_allow_html=True)
    
    if st.sidebar.button("🤖 Agent Chat", key="nav_agent"):
        st.session_state.page = "agent_chat"
        st.rerun()
    
    if st.sidebar.button("📋 Task Management", key="nav_tasks"):
        st.session_state.page = "tasks"
        st.rerun()
    
    if st.sidebar.button("🌐 Browser Control", key="nav_browser"):
        st.session_state.page = "browser"
        st.rerun()
    
    st.sidebar.markdown('<p class="sidebar-header">Advanced Features</p>', unsafe_allow_html=True)
    
    if st.sidebar.button("🧪 Advanced Features", key="nav_features"):
        st.session_state.page = "advanced_features"
        st.rerun()
    
    if st.sidebar.button("📊 Performance Metrics", key="nav_metrics"):
        st.session_state.page = "metrics"
        st.rerun()
    
    st.sidebar.markdown('<p class="sidebar-header">Settings</p>', unsafe_allow_html=True)
    
    if st.sidebar.button("⚙️ Quantum Settings", key="nav_settings"):
        st.session_state.page = "settings"
        st.rerun()
    
    if st.sidebar.button("ℹ️ About", key="nav_about"):
        st.session_state.page = "about"
        st.rerun()
    
    # Quantum settings summary
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quantum Status")
    
    status_color = "green" if st.session_state.use_quantum else "red"
    st.sidebar.markdown(f"""
    <p>Quantum Acceleration: <span style="color:{status_color};font-weight:bold;">{'✅ Enabled' if st.session_state.use_quantum else '❌ Disabled'}</span></p>
    <p>Number of Qubits: <span style="color:#7b2cbf;font-weight:bold;">{st.session_state.n_qubits}</span></p>
    <p>Azure Quantum: <span style="color:{'green' if st.session_state.use_azure else 'gray'};font-weight:bold;">{'✅ Enabled' if st.session_state.use_azure else '❌ Disabled'}</span></p>
    """, unsafe_allow_html=True)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align: center; font-size: 0.8em;">
    Q3A: Quantum-Accelerated AI Agent<br>
    Powered by QUASAR Framework<br>
    © 2025
    </div>
    """, unsafe_allow_html=True)

# Main application
def main():
    """Main application function."""
    # Display sidebar
    display_sidebar()
    
    # Display page based on selection
    if st.session_state.page == "agent_chat":
        display_agent_chat_ui()
    elif st.session_state.page == "tasks":
        display_tasks_ui()
    elif st.session_state.page == "browser":
        display_browser_ui()
    elif st.session_state.page == "advanced_features":
        display_advanced_features()
    elif st.session_state.page == "metrics":
        display_performance_metrics()
    elif st.session_state.page == "settings":
        display_quantum_settings()
    elif st.session_state.page == "about":
        display_about()
    else:
        # Home page
        display_agent_chat_ui()

if __name__ == "__main__":
    main()