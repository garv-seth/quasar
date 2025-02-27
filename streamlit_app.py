"""
QA³: Quantum-Accelerated AI Agent with True Agentic Capabilities
Enhanced Streamlit Interface with Advanced Browser Automation and Computer Vision
"""

import os
import streamlit as st
import asyncio
import time
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import base64
import random

# Import agent components with proper error handling
try:
    from enhanced_autonomous_agent import EnhancedAutonomousAgent, run_async
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False
    st.error("Enhanced Autonomous Agent not available. Some features may be limited.")

# Quantum components with proper error handling
try:
    import pennylane as qml
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False

# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'agent_initialized' not in st.session_state:
        st.session_state.agent_initialized = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_task' not in st.session_state:
        st.session_state.current_task = None
    if 'task_results' not in st.session_state:
        st.session_state.task_results = []
    if 'api_keys_set' not in st.session_state:
        st.session_state.api_keys_set = False
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Chat"

# Async initialization of the agent
async def initialize_agent_async():
    """Initialize the agent asynchronously"""
    if not AGENT_AVAILABLE:
        return {"success": False, "error": "Agent not available"}
        
    try:
        # Create agent with specified settings
        agent = EnhancedAutonomousAgent(
            use_quantum=st.session_state.get('use_quantum', True),
            n_qubits=st.session_state.get('n_qubits', 8),
            use_web_automation=st.session_state.get('use_web_automation', True),
            use_vision=st.session_state.get('use_vision', True),
            use_claude=st.session_state.get('use_claude', True)
        )
        
        # Initialize agent
        init_result = await agent.initialize()
        
        if init_result.get('success', False):
            st.session_state.agent = agent
            st.session_state.agent_initialized = True
            return {"success": True}
        else:
            return {"success": False, "error": "Failed to initialize agent", "details": init_result}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

# Function to run async functions from Streamlit
def initialize_agent():
    """Initialize the agent (wrapper for async initialization)"""
    with st.status("Initializing agent...", expanded=True) as status:
        result = run_async(initialize_agent_async())
        
        if result.get('success', False):
            status.update(label="Agent initialized successfully!", state="complete", expanded=False)
            return True
        else:
            error_msg = result.get('error', 'Unknown error')
            status.update(label=f"Failed to initialize agent: {error_msg}", state="error", expanded=True)
            st.error(f"Initialization error: {error_msg}")
            if 'details' in result:
                st.json(result['details'])
            return False

# Process user messages
async def process_message_async(message):
    """Process a user message asynchronously"""
    if not st.session_state.agent_initialized or not st.session_state.agent:
        return {
            "success": False,
            "error": "Agent not initialized",
            "response": "Please initialize the agent first."
        }
        
    try:
        # Record message
        st.session_state.messages.append({"role": "user", "content": message})
        
        # Process message as a task
        result = await st.session_state.agent.process_task(message)
        
        # Store task result
        st.session_state.task_results.append(result)
        st.session_state.current_task = result
        
        # Create response message
        response = result.get('summary', 'Task processed but no summary available')
        
        # Add response to messages
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        return {
            "success": True,
            "response": response,
            "result": result
        }
        
    except Exception as e:
        error_message = f"Error processing message: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_message})
        
        return {
            "success": False,
            "error": str(e),
            "response": error_message
        }

def process_message(message):
    """Process a user message (wrapper for async processing)"""
    with st.status("Processing task...", expanded=True) as status:
        result = run_async(process_message_async(message))
        
        if result.get('success', False):
            status.update(label="Task completed!", state="complete", expanded=False)
        else:
            error_msg = result.get('error', 'Unknown error')
            status.update(label=f"Task failed: {error_msg}", state="error", expanded=True)
            
        return result

# Display interface components
def display_chat_interface():
    """Display the main chat interface"""
    st.header("QA³ Agent Chat")
    
    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
    # Get user input
    if user_input := st.chat_input("Enter your request or task..."):
        # Process user input
        process_message(user_input)
        
        # Force UI refresh
        st.rerun()

def display_task_details():
    """Display details of the current or most recent task"""
    st.header("Current Task Details")
    
    if not st.session_state.current_task:
        st.info("No task has been executed yet. Use the chat interface to give the agent a task.")
        return
        
    # Display task information
    task = st.session_state.current_task
    
    # Summary card
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Task Summary")
            st.write(task.get("task", "No task description"))
            st.markdown(f"**Status**: {'✅ Success' if task.get('success', False) else '❌ Failed'}")
            if task.get("error"):
                st.error(f"Error: {task['error']}")
                
        with col2:
            # Display timing information
            execution_time = task.get("execution_time", 0)
            st.metric("Execution Time", f"{execution_time:.2f}s")
            
            # Display counts
            steps = task.get("steps", [])
            successful_steps = sum(1 for step in steps if step.get("success", False))
            
            st.metric("Steps", f"{successful_steps}/{len(steps)}")
            
    # Display analysis if available
    if "analysis" in task:
        with st.expander("Task Analysis", expanded=False):
            analysis = task["analysis"]
            st.markdown(f"**Task Type**: {analysis.get('task_type', 'Unknown')}")
            
            st.markdown("**Goals**:")
            for goal in analysis.get("goals", []):
                st.markdown(f"- {goal}")
                
            st.markdown("**Required Tools**:")
            for tool in analysis.get("required_tools", []):
                st.markdown(f"- {tool}")
                
            st.markdown("**Approaches**:")
            for approach in analysis.get("approaches", []):
                st.markdown(f"- {approach}")
                
    # Display execution plan if available
    if "plan" in task:
        with st.expander("Execution Plan", expanded=False):
            plan = task["plan"]
            
            if "original_plan" in plan:
                st.markdown("**Original Plan**:")
                st.markdown(f"```\n{plan['original_plan']}\n```")
            else:
                st.markdown("**Steps**:")
                for step in plan.get("steps", []):
                    st.markdown(f"**Step {step.get('step_number')}**: {step.get('description')}")
                    
                    if step.get("tool"):
                        st.markdown(f"Tool: `{step.get('tool')}`")
                        
                    if step.get("parameters"):
                        st.markdown("Parameters:")
                        for key, value in step.get("parameters", {}).items():
                            st.markdown(f"- {key}: `{value}`")
                            
                    st.markdown(f"Expected outcome: {step.get('expected_outcome', '')}")
                    
    # Display execution steps
    st.subheader("Execution Steps")
    
    for i, step in enumerate(task.get("steps", [])):
        with st.expander(f"Step {step.get('step_number')}: {step.get('description')}", 
                       expanded=step.get("error") is not None):
            
            # Status and timing
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Status**: {'✅ Success' if step.get('success', False) else '❌ Failed'}")
            with col2:
                execution_time = step.get("execution_time", 0)
                st.markdown(f"**Time**: {execution_time:.2f}s")
                
            # Display tool information if available
            if step.get("tool"):
                st.markdown(f"**Tool**: `{step.get('tool')}`")
                
                st.markdown("**Parameters**:")
                for key, value in step.get("parameters", {}).items():
                    st.markdown(f"- {key}: `{value}`")
                    
            # Display error if failed
            if not step.get("success", False) and step.get("error"):
                st.error(f"Error: {step.get('error')}")
                
            # Display output if available
            if "output" in step:
                with st.expander("Output Details", expanded=False):
                    output = step["output"]
                    
                    # Handle screenshot specially
                    if output.get("screenshot"):
                        st.image(
                            "data:image/png;base64," + output["screenshot"],
                            caption="Screenshot from this step",
                            use_column_width=True
                        )
                        
                    # Display other outputs as JSON
                    clean_output = {k: v for k, v in output.items() if k != "screenshot"}
                    if clean_output:
                        st.json(clean_output)

def display_agent_status():
    """Display agent status and metrics"""
    st.header("Agent Status")
    
    if not st.session_state.agent_initialized or not st.session_state.agent:
        st.warning("Agent not initialized. Please initialize the agent to view status.")
        return
        
    # Get agent status
    agent_status = run_async(lambda: st.session_state.agent.get_status())
    
    # Display agent information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Status", "Active" if agent_status.get("initialized", False) else "Inactive")
        
    with col2:
        performance = agent_status.get("performance", {})
        success_rate = performance.get("success_rate", 0) * 100
        st.metric("Task Success Rate", f"{success_rate:.1f}%")
        
    with col3:
        uptime = performance.get("uptime", 0)
        uptime_str = f"{int(uptime/3600)}h {int((uptime%3600)/60)}m {int(uptime%60)}s"
        st.metric("Uptime", uptime_str)
        
    # Component status
    st.subheader("Component Status")
    
    components = agent_status.get("components", {})
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**AI Components:**")
        st.markdown(f"- OpenAI Client: {'✅ Active' if components.get('openai_client', False) else '❌ Inactive'}")
        st.markdown(f"- Anthropic Client: {'✅ Active' if components.get('anthropic_client', False) else '❌ Inactive'}")
        
    with col2:
        st.markdown("**Automation Components:**")
        st.markdown(f"- Web Agent: {'✅ Active' if components.get('web_agent', False) else '❌ Inactive'}")
        st.markdown(f"- Vision System: {'✅ Active' if components.get('vision_system', False) else '❌ Inactive'}")
        
    # Quantum capabilities
    st.subheader("Quantum Capabilities")
    
    quantum = agent_status.get("quantum", {})
    if quantum:
        st.markdown(f"**Provider**: {quantum.get('provider', 'Not available')}")
        st.markdown(f"**Qubits**: {quantum.get('n_qubits', 0)}")
        st.markdown(f"**Device Type**: {quantum.get('device_type', 'Not available')}")
    else:
        st.warning("Quantum capabilities not available.")
        
    # Memory metrics
    st.subheader("Memory Status")
    
    memory = agent_status.get("memory", {})
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Short-term Items", memory.get("short_term_items", 0))
    with col2:
        st.metric("Working Memory Items", memory.get("working_memory_items", 0))
    with col3:
        st.metric("Long-term Items", memory.get("long_term_items", 0))
        
    # Tools
    st.subheader("Available Tools")
    
    tools = agent_status.get("available_tools", [])
    
    if tools:
        col1, col2 = st.columns(2)
        
        for i, tool in enumerate(tools):
            if i % 2 == 0:
                col1.markdown(f"- {tool}")
            else:
                col2.markdown(f"- {tool}")
    else:
        st.info("No tools available.")
        
    # Web metrics if available
    web_metrics = agent_status.get("web_metrics")
    if web_metrics:
        with st.expander("Web Interaction Metrics"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Pages Visited", web_metrics.get("pages_visited", 0))
                
            with col2:
                st.metric("Interactions", web_metrics.get("interactions", 0))
                
            with col3:
                st.metric("Errors", web_metrics.get("errors", 0))
                
            # Navigation history
            if "navigation_history_count" in web_metrics and web_metrics["navigation_history_count"] > 0:
                st.markdown(f"**Current URL**: {web_metrics.get('current_url', 'None')}")
                
def display_api_config():
    """Display API configuration interface"""
    st.header("API Configuration")
    
    with st.form("api_keys_form"):
        # OpenAI API key
        openai_api_key = st.text_input(
            "OpenAI API Key", 
            value=os.environ.get("OPENAI_API_KEY", ""), 
            type="password",
            help="Required for AI reasoning capabilities"
        )
        
        # Anthropic API key (optional)
        anthropic_api_key = st.text_input(
            "Anthropic API Key (Optional)", 
            value=os.environ.get("ANTHROPIC_API_KEY", ""), 
            type="password",
            help="Optional: Enables Claude models for enhanced reasoning"
        )
        
        # Azure Quantum settings (optional)
        st.subheader("Azure Quantum Settings (Optional)")
        
        azure_quantum_subscription_id = st.text_input(
            "Azure Quantum Subscription ID",
            value=os.environ.get("AZURE_QUANTUM_SUBSCRIPTION_ID", ""),
            help="Required for Azure Quantum hardware access"
        )
        
        azure_quantum_resource_group = st.text_input(
            "Azure Quantum Resource Group",
            value=os.environ.get("AZURE_QUANTUM_RESOURCE_GROUP", ""),
            help="Required for Azure Quantum hardware access"
        )
        
        azure_quantum_workspace_name = st.text_input(
            "Azure Quantum Workspace Name",
            value=os.environ.get("AZURE_QUANTUM_WORKSPACE_NAME", ""),
            help="Required for Azure Quantum hardware access"
        )
        
        # Submit button
        submit_button = st.form_submit_button("Save API Keys")
        
        if submit_button:
            # Set environment variables
            os.environ["OPENAI_API_KEY"] = openai_api_key
            
            if anthropic_api_key:
                os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
                
            if azure_quantum_subscription_id:
                os.environ["AZURE_QUANTUM_SUBSCRIPTION_ID"] = azure_quantum_subscription_id
                
            if azure_quantum_resource_group:
                os.environ["AZURE_QUANTUM_RESOURCE_GROUP"] = azure_quantum_resource_group
                
            if azure_quantum_workspace_name:
                os.environ["AZURE_QUANTUM_WORKSPACE_NAME"] = azure_quantum_workspace_name
                
            # Update session state
            st.session_state.api_keys_set = True
            
            # Reinitialize agent if it was already initialized
            if st.session_state.agent_initialized:
                st.info("API keys updated. Reinitializing agent...")
                
                # Close the current agent
                if st.session_state.agent:
                    run_async(st.session_state.agent.close())
                    
                # Reset initialization state
                st.session_state.agent = None
                st.session_state.agent_initialized = False
                
                # Initialize new agent
                initialize_agent()
            else:
                st.success("API keys saved successfully!")
                
    # Agent initialization
    if not st.session_state.agent_initialized:
        st.subheader("Agent Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.use_quantum = st.checkbox("Use Quantum Acceleration", value=True)
            st.session_state.use_web_automation = st.checkbox("Enable Web Automation", value=True)
            
        with col2:
            st.session_state.use_vision = st.checkbox("Enable Computer Vision", value=True)
            st.session_state.use_claude = st.checkbox("Use Claude if available", value=True)
            
        st.session_state.n_qubits = st.slider("Number of Qubits", min_value=4, max_value=16, value=8)
        
        if st.button("Initialize Agent", disabled=not st.session_state.api_keys_set and not openai_api_key):
            initialize_agent()
    else:
        if st.button("Reinitialize Agent"):
            # Close the current agent
            if st.session_state.agent:
                run_async(st.session_state.agent.close())
                
            # Reset initialization state
            st.session_state.agent = None
            st.session_state.agent_initialized = False
            
            # Initialize new agent
            initialize_agent()

def display_about():
    """Display information about the agent"""
    st.header("About QA³ Agent")
    
    st.markdown("""
    ## Quantum-Accelerated AI Agent with True Agentic Capabilities
    
    The QA³ (Quantum-Accelerated AI Agent) combines advanced quantum computing capabilities with true agentic behavior, enabling autonomous interaction with digital interfaces.
    
    ### Key Components:
    
    #### 1. Autonomous Decision System
    - Goal-driven behavior with hierarchical task decomposition
    - Self-evaluation and learning from experience
    - Strategic decision-making with quantum enhancement
    
    #### 2. Computer Vision System
    - Screen understanding for UI element detection
    - Visual analysis of interfaces
    - OCR capabilities for text extraction
    
    #### 3. Web Interaction Agent
    - Autonomous browsing capabilities
    - Form filling and interaction
    - Search and information extraction
    
    #### 4. Quantum Enhancement
    - Decision optimization using quantum circuits
    - Uncertainty modeling with quantum superposition
    - Leveraging quantum parallelism for enhanced processing
    
    ### Quantum Advantage
    
    The agent uses quantum computing to enhance several key capabilities:
    
    - **Decision-making**: Evaluating multiple options simultaneously
    - **Search optimization**: Accelerating information discovery
    - **Pattern recognition**: Improving detection of complex patterns
    
    ### Development Status
    
    This system is a prototype demonstrating the integration of quantum computing with agentic AI. It represents an early exploration of how quantum capabilities can enhance autonomous systems.
    """)

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="QA³: Quantum-Accelerated AI Agent",
        page_icon="🔮",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar with information and settings
    with st.sidebar:
        st.title("QA³ Agent")
        st.markdown("Quantum-Accelerated AI Agent with True Agentic Capabilities")
        
        # Navigation
        st.subheader("Navigation")
        tab_options = ["Chat", "Task Details", "Status", "API Configuration", "About"]
        selected_tab = st.radio("Select Tab", tab_options, index=tab_options.index(st.session_state.current_tab))
        st.session_state.current_tab = selected_tab
        
        # Quantum status
        st.subheader("System Status")
        
        if not QUANTUM_AVAILABLE:
            st.warning("⚠️ Quantum libraries not available")
        else:
            st.success("✅ Quantum libraries available")
            
        if not st.session_state.api_keys_set and "OPENAI_API_KEY" not in os.environ:
            st.warning("⚠️ API keys not configured")
        else:
            st.success("✅ API keys configured")
            
        if not st.session_state.agent_initialized:
            st.warning("⚠️ Agent not initialized")
        else:
            st.success("✅ Agent initialized")
            
        # License information
        st.markdown("---")
        st.markdown("© 2025 Quantum Agent Research")
        
    # Main content area
    if st.session_state.current_tab == "Chat":
        display_chat_interface()
    elif st.session_state.current_tab == "Task Details":
        display_task_details()
    elif st.session_state.current_tab == "Status":
        display_agent_status()
    elif st.session_state.current_tab == "API Configuration":
        display_api_config()
    elif st.session_state.current_tab == "About":
        display_about()
        
    # Show warning if agent not initialized for tabs that need it
    if not st.session_state.agent_initialized and st.session_state.current_tab in ["Chat", "Task Details", "Status"]:
        st.warning("Agent not initialized. Please go to the API Configuration tab to configure and initialize the agent.")

if __name__ == "__main__":
    main()