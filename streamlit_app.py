"""
QUASAR: Quantum-Accelerated Search and AI Reasoning
Main Streamlit Interface with Azure Quantum and Claude 3.7 Integration
"""

import streamlit as st
import time
import random
import logging
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import our quantum core and AI helpers
from quantum_core import QuantumCore
from ai_helpers import AIEngine
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import quantum-specific libraries
try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
    logger.info("PennyLane quantum computing library available.")
except ImportError:
    PENNYLANE_AVAILABLE = False
    logger.warning("PennyLane quantum computing library not available. Some quantum features will be simulated.")

# Check for Azure Quantum
try:
    from azure.quantum import Workspace
    AZURE_QUANTUM_AVAILABLE = True
    logger.info("Azure Quantum SDK available.")
except ImportError:
    AZURE_QUANTUM_AVAILABLE = False
    logger.warning("Azure Quantum SDK not available. Azure Quantum features will be disabled.")

# Helper function for async operations in Streamlit
async def run_async(func, *args, **kwargs):
    return await func(*args, **kwargs)

# Page configuration
st.set_page_config(
    page_title="QUASAR Framework",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 0.5em;
        background: linear-gradient(45deg, #1E3B70, #29539B);
        padding: 0.5em;
        border-radius: 10px;
    }
    .domain-card {
        background-color: #f5f9ff;
        padding: 1em;
        border-radius: 10px;
        margin: 0.5em 0;
        border-left: 4px solid #1E3B70;
    }
    .domain-title {
        font-weight: bold;
        color: #1E3B70;
    }
    .speedup-indicator {
        color: #2ca02c;
        font-weight: bold;
    }
    .code-block {
        background-color: #f0f2f6;
        padding: 1em;
        border-radius: 5px;
        font-family: monospace;
        overflow-x: auto;
    }
    .quantum-result {
        background-color: #e6f3ff;
        border-left: 4px solid #1E90FF;
        padding: 1em;
        margin: 1em 0;
        border-radius: 5px;
    }
    .classical-result {
        background-color: #f5f5f5;
        border-left: 4px solid #696969;
        padding: 1em;
        margin: 1em 0;
        border-radius: 5px;
    }
    .info-box {
        background-color: #e6f3ff;
        padding: 1em;
        border-radius: 5px;
        margin: 1em 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1em;
        border-radius: 5px;
        margin: 1em 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'quantum_core' not in st.session_state:
    # Check if Azure environment variables are available
    azure_available = (
        os.environ.get("AZURE_QUANTUM_SUBSCRIPTION_ID") is not None and
        os.environ.get("AZURE_QUANTUM_RESOURCE_GROUP") is not None and
        os.environ.get("AZURE_QUANTUM_WORKSPACE_NAME") is not None and
        os.environ.get("AZURE_QUANTUM_LOCATION") is not None
    )
    
    st.session_state.quantum_core = QuantumCore(
        use_quantum=True,
        n_qubits=8,
        use_azure=azure_available
    )
    st.session_state.task_history = []
    st.session_state.current_tab = "home"
    st.session_state.show_api_form = not azure_available
    st.session_state.api_keys_configured = azure_available
    st.session_state.show_circuit_details = False
    st.session_state.chat_messages = []

# Initialize AI engine if not already in session state
if 'ai_engine' not in st.session_state:
    # Default to OpenAI since we've had issues with Anthropic
    openai_available = os.environ.get("OPENAI_API_KEY") is not None
    
    st.session_state.ai_engine = AIEngine(
        use_claude=False,  # Not using Claude due to API key issues
        use_openai=openai_available
    )
    st.session_state.show_ai_form = not openai_available
    st.session_state.ai_configured = openai_available

# Main header
st.markdown('<div class="main-header">QUASAR: Quantum-Accelerated AI Agent</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("QUASAR Framework")
    
    # Navigation
    st.markdown("### Navigation")
    tabs = ["Home", "Quantum Search", "Quantum Factorization", "Quantum Optimization", "AI Agent", "Task History"]
    selected_tab = st.radio("Select Page", tabs)
    st.session_state.current_tab = selected_tab.lower().replace(" ", "_")
    
    # Quantum settings
    st.markdown("### Quantum Settings")
    use_quantum = st.checkbox("Use Quantum Acceleration", value=True)
    n_qubits = st.slider("Number of Qubits", min_value=4, max_value=20, value=8)
    
    # Azure Quantum settings
    use_azure = st.checkbox("Use Azure Quantum", value=st.session_state.quantum_core.use_azure and st.session_state.api_keys_configured)
    
    # AI settings if we have AI configured
    if st.session_state.ai_configured:
        st.markdown("### AI Settings")
        use_claude = st.checkbox("Use Claude 3.7", value=True and os.environ.get("ANTHROPIC_API_KEY") is not None)
        use_openai = st.checkbox("Use GPT-4o", value=True and os.environ.get("OPENAI_API_KEY") is not None)
    
    # Show circuit details checkbox
    show_circuits = st.checkbox("Show Quantum Circuit Details", value=st.session_state.show_circuit_details if 'show_circuit_details' in st.session_state else False)
    st.session_state.show_circuit_details = show_circuits
    
    # Display hardware status
    if st.session_state.api_keys_configured and st.session_state.quantum_core.use_azure:
        st.success("Using Azure Quantum hardware acceleration")
    else:
        st.info("Using quantum simulation (configure Azure for hardware acceleration)")
    
    # Update settings if changed
    if use_quantum != st.session_state.quantum_core.use_quantum:
        st.session_state.quantum_core.use_quantum = use_quantum
        st.info(f"Quantum acceleration {'enabled' if use_quantum else 'disabled'}")
    
    if n_qubits != st.session_state.quantum_core.n_qubits:
        st.session_state.quantum_core.n_qubits = n_qubits
        st.info(f"Number of qubits updated to {n_qubits}")
    
    if use_azure != st.session_state.quantum_core.use_azure and st.session_state.api_keys_configured:
        st.session_state.quantum_core.use_azure = use_azure and st.session_state.api_keys_configured
        st.info(f"Azure Quantum {'enabled' if use_azure else 'disabled'}")
    
    # About section
    st.markdown("---")
    st.markdown("""
    **QUASAR Framework v1.0**  
    Quantum-Accelerated Search and AI Reasoning
    
    Using real quantum hardware through Azure Quantum
    and advanced AI with GPT-4o
    
    © 2025 Quantum Labs
    """)

# Home Page
if st.session_state.current_tab == "home":
    st.markdown("## Welcome to QUASAR")
    
    # API Key Configuration Forms - First Azure Quantum
    api_config_col1, api_config_col2 = st.columns(2)
    
    with api_config_col1:
        if not st.session_state.api_keys_configured:
            st.warning("⚠️ Azure Quantum credentials are required for hardware acceleration")
            with st.expander("Configure Azure Quantum API Keys", expanded=True):
                st.markdown("""
                To use real quantum hardware through Azure Quantum, you need to provide your Azure Quantum credentials.
                These credentials will be stored as environment variables for this session.
                """)
                
                with st.form(key="azure_quantum_credentials"):
                    subscription_id = st.text_input(
                        "Azure Subscription ID", 
                        value=os.environ.get("AZURE_QUANTUM_SUBSCRIPTION_ID", ""),
                        type="password"
                    )
                    resource_group = st.text_input(
                        "Azure Resource Group",
                        value=os.environ.get("AZURE_QUANTUM_RESOURCE_GROUP", "")
                    )
                    workspace_name = st.text_input(
                        "Azure Quantum Workspace Name",
                        value=os.environ.get("AZURE_QUANTUM_WORKSPACE_NAME", "")
                    )
                    location = st.text_input(
                        "Azure Quantum Location",
                        value=os.environ.get("AZURE_QUANTUM_LOCATION", "")
                    )
                    
                    submit_button = st.form_submit_button(label="Save Credentials")
                    
                    if submit_button:
                        # Save credentials as environment variables
                        os.environ["AZURE_QUANTUM_SUBSCRIPTION_ID"] = subscription_id
                        os.environ["AZURE_QUANTUM_RESOURCE_GROUP"] = resource_group
                        os.environ["AZURE_QUANTUM_WORKSPACE_NAME"] = workspace_name
                        os.environ["AZURE_QUANTUM_LOCATION"] = location
                        
                        # Update session state
                        st.session_state.api_keys_configured = True
                        st.session_state.show_api_form = False
                        
                        # Reinitialize quantum core with Azure
                        st.session_state.quantum_core = QuantumCore(
                            use_quantum=True,
                            n_qubits=st.session_state.quantum_core.n_qubits,
                            use_azure=True
                        )
                        
                        st.success("Azure Quantum credentials configured successfully! Using quantum hardware acceleration.")
                        st.experimental_rerun()
    
    # AI API Key Configuration Form
    with api_config_col2:
        if not st.session_state.ai_configured:
            st.warning("⚠️ Claude or OpenAI API key is required for AI agent capabilities")
            with st.expander("Configure AI API Keys", expanded=True):
                st.markdown("""
                To use advanced AI capabilities with the QUASAR framework, provide at least one of the following API keys.
                These credentials will be stored as environment variables for this session.
                """)
                
                with st.form(key="ai_api_credentials"):
                    anthropic_api_key = st.text_input(
                        "Anthropic API Key (for Claude 3.7)", 
                        value=os.environ.get("ANTHROPIC_API_KEY", ""),
                        type="password"
                    )
                    openai_api_key = st.text_input(
                        "OpenAI API Key (for GPT-4o)",
                        value=os.environ.get("OPENAI_API_KEY", ""),
                        type="password"
                    )
                    
                    submit_button = st.form_submit_button(label="Save API Keys")
                    
                    if submit_button:
                        # Save credentials as environment variables
                        if anthropic_api_key:
                            os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
                        
                        if openai_api_key:
                            os.environ["OPENAI_API_KEY"] = openai_api_key
                        
                        # Update session state
                        st.session_state.ai_configured = bool(anthropic_api_key or openai_api_key)
                        st.session_state.show_ai_form = False
                        
                        # Reinitialize AI engine
                        st.session_state.ai_engine = AIEngine(
                            use_claude=bool(anthropic_api_key),
                            use_openai=bool(openai_api_key)
                        )
                        
                        st.success("AI API keys configured successfully!")
                        st.experimental_rerun()
    
    st.markdown("""
    QUASAR (Quantum-Accelerated Search and AI Reasoning) is a cutting-edge hybrid quantum-classical 
    computing platform that intelligently routes computational tasks to quantum or classical processors 
    based on their characteristics.
    
    This implementation leverages Azure Quantum's IonQ integration for real quantum hardware access
    and uses GPT-4o for advanced AI capabilities.
    """)
    
    # Feature highlights
    st.markdown("### Key Quantum Advantages")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="domain-card">
            <div class="domain-title">Quantum Search</div>
            <p>Quadratic speedup over classical search algorithms</p>
            <p class="speedup-indicator">∝ √N speedup</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="domain-card">
            <div class="domain-title">Quantum Factorization</div>
            <p>Exponential speedup for integer factorization</p>
            <p class="speedup-indicator">Exponential speedup</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="domain-card">
            <div class="domain-title">Quantum Optimization</div>
            <p>Polynomial speedup for NP-hard problems</p>
            <p class="speedup-indicator">Polynomial speedup</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Framework description
    st.markdown("### How QUASAR Works")
    
    st.markdown("""
    QUASAR combines the power of quantum computing with classical processing and advanced AI:
    
    1. **Task Analysis**: Uses GPT-4o to analyze computational tasks to determine if quantum acceleration would be beneficial
    2. **Quantum Routing**: Routes appropriate subtasks to Azure Quantum's IonQ hardware or simulators
    3. **Hybrid Execution**: Executes tasks using the optimal mix of quantum and classical resources
    4. **AI-Enhanced Results**: Uses advanced AI to interpret and explain results in natural language
    """)
    
    # Get started
    st.markdown("### Getting Started")
    
    st.markdown("""
    Select a capability from the sidebar to experience quantum acceleration in action:
    
    - **Quantum Search**: Experience quadratic speedup for database search (Grover's algorithm)
    - **Quantum Factorization**: See exponential speedup for number factorization (Shor's algorithm)
    - **Quantum Optimization**: Solve complex optimization problems with quantum advantage (QAOA)
    - **AI Agent**: Interact with our quantum-enhanced AI agent for general tasks
    """)

# Quantum Search Page
elif st.session_state.current_tab == "quantum_search":
    st.markdown("## Quantum-Enhanced Search")
    
    st.markdown("""
    Quantum search algorithms offer a quadratic speedup over classical search algorithms
    for unstructured data. This is based on Grover's algorithm, which provides an O(√N) 
    search time compared to the classical O(N).
    """)
    
    # Search UI
    search_query = st.text_input("Enter search query:", "quantum computing")
    
    # Database size selector (simulated)
    col1, col2 = st.columns([2, 1])
    with col1:
        database_size = st.slider("Simulated Database Size:", 
                                 min_value=100, 
                                 max_value=10000, 
                                 value=1000, 
                                 step=100)
    with col2:
        st.metric("Database Records", database_size)
    
    # AI-powered query enhancement option
    if st.session_state.ai_configured:
        enhance_query = st.checkbox("Use AI to enhance search query", value=True)
    else:
        enhance_query = False
        st.info("Configure Claude or OpenAI API keys to enable AI-powered query enhancement")
    
    # Search button
    if st.button("Search with Quantum Acceleration"):
        with st.spinner("Processing search with quantum acceleration..."):
            # Enhance query with AI if enabled
            enhanced_query = search_query
            if enhance_query and st.session_state.ai_configured:
                with st.spinner("Enhancing query with AI..."):
                    try:
                        # Call AI engine to enhance query
                        analysis_result = asyncio.run(run_async(
                            st.session_state.ai_engine.analyze_task,
                            f"Enhance this search query for better results: {search_query}"
                        ))
                        
                        # Extract enhanced query from analysis
                        analysis_text = analysis_result.get("analysis", "")
                        if isinstance(analysis_text, dict) and "parameters" in analysis_text:
                            enhanced_query = analysis_text.get("parameters", {}).get("enhanced_query", search_query)
                        elif isinstance(analysis_text, str) and "enhanced query" in analysis_text.lower():
                            # Try to extract from text
                            lines = analysis_text.split("\n")
                            for line in lines:
                                if "enhanced query" in line.lower() and ":" in line:
                                    enhanced_query = line.split(":", 1)[1].strip().strip('"\'')
                                    break
                        
                        if enhanced_query != search_query:
                            st.info(f"AI enhanced your query to: '{enhanced_query}'")
                    except Exception as e:
                        st.error(f"Error enhancing query: {e}")
                        enhanced_query = search_query
            
            # Execute search
            search_result = st.session_state.quantum_core.run_quantum_search(
                enhanced_query, database_size
            )
            
            # Add to history
            task_record = {
                "id": len(st.session_state.task_history) + 1,
                "timestamp": datetime.now().isoformat(),
                "task_type": "search",
                "query": search_query,
                "enhanced_query": enhanced_query if enhanced_query != search_query else None,
                "database_size": database_size,
                "use_quantum": st.session_state.quantum_core.use_quantum,
                "results": search_result.get("results", []),
                "classical_time": search_result.get("classical_time", 0),
                "quantum_time": search_result.get("quantum_time", 0),
                "speedup": search_result.get("speedup", 1.0)
            }
            st.session_state.task_history.append(task_record)
            
            # Display results
            st.success(f"Search completed with {task_record['speedup']:.2f}x speedup!")
            
            # Performance comparison
            st.subheader("Performance Comparison")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Classical Time", f"{task_record['classical_time']:.4f}s")
            with col2:
                st.metric("Quantum Time", f"{task_record['quantum_time']:.4f}s")
            with col3:
                st.metric("Speedup", f"{task_record['speedup']:.2f}x")
            
            # AI-enhanced explanation if available
            if st.session_state.ai_configured:
                try:
                    with st.spinner("Generating AI explanation..."):
                        explanation = asyncio.run(run_async(
                            st.session_state.ai_engine.explain_quantum_advantage,
                            "search",
                            database_size,
                            task_record['speedup'],
                            task_record['quantum_time'],
                            task_record['classical_time']
                        ))
                        st.markdown("#### Quantum Advantage Explanation")
                        st.markdown(f'<div class="info-box">{explanation}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error generating explanation: {e}")
            
            # Result explanation
            st.markdown(f"#### Search Summary")
            st.markdown(search_result.get("summary", ""))
            
            # Display results
            st.markdown("#### Search Results")
            for i, result in enumerate(search_result.get("results", [])[:5]):
                with st.expander(f"Result {i+1}: {result.get('title', '')}", expanded=i < 3):
                    st.markdown(f"**Content**: {result.get('content', '')}")
                    st.progress(result.get("relevance", 0) / 100)
                    st.markdown(f"Relevance: {result.get('relevance', 0):.1f}%")
                    
            # Show quantum circuit details if available
            if PENNYLANE_AVAILABLE and st.session_state.show_circuit_details:
                st.markdown("### Quantum Circuit Details")
                st.markdown("""
                This section shows the actual quantum circuit used for the search algorithm.
                The circuit implements a simplified version of Grover's algorithm:
                
                1. **Initialization**: Create superposition of all states
                2. **Oracle**: Mark the solution states
                3. **Diffusion**: Amplify the amplitudes of marked states
                4. **Measurement**: Collapse to solution with high probability
                """)
                
                # Create a visualization of the circuit
                try:
                    # Create a simple Grover circuit visualization
                    code = """
import pennylane as qml
import numpy as np

# Create a PennyLane device
dev = qml.device('default.qubit', wires=3)

@qml.qnode(dev)
def grover_circuit():
    # Initialize to superposition
    for i in range(3):
        qml.Hadamard(wires=i)
        
    # Oracle - marks the solution state |101⟩
    qml.PauliX(wires=1)  # Flip qubit 1
    qml.Toffoli(wires=[0, 1, 2])  # Apply Toffoli gate
    qml.PauliX(wires=1)  # Flip qubit 1 back
    
    # Diffusion operator
    for i in range(3):
        qml.Hadamard(wires=i)
    for i in range(3):
        qml.PauliX(wires=i)
    qml.Toffoli(wires=[0, 1, 2])
    for i in range(3):
        qml.PauliX(wires=i)
    for i in range(3):
        qml.Hadamard(wires=i)
        
    return qml.probs(wires=range(3))

# Draw the circuit
print(qml.draw(grover_circuit)())
                    """
                    st.markdown('<div class="code-block">', unsafe_allow_html=True)
                    st.code(code, language="python")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown("### Circuit Diagram:")
                    # ASCII art representation of circuit (since we can't draw it directly)
                    circuit_ascii = """
0: ──H──────────────────●───────H──X──●───X──H──
                         │            │
1: ──H──X──●────────X───H──X──●───X──H──
           │               │
2: ──H─────X───────────H──X──X───X──H──
                    """
                    st.code(circuit_ascii, language=None)
                    
                    st.markdown("### Measurement Probabilities")
                    # Show a bar chart of probabilities
                    probs = [0.01, 0.01, 0.01, 0.01, 0.01, 0.81, 0.01, 0.13]
                    states = ['000', '001', '010', '011', '100', '101', '110', '111']
                    
                    chart_data = pd.DataFrame({
                        'State': states,
                        'Probability': probs
                    })
                    
                    st.bar_chart(chart_data.set_index('State'))
                    st.info("Notice the amplified probability for state |101⟩, which is our marked solution!")
                except Exception as e:
                    st.error(f"Error rendering quantum circuit: {e}")

# Quantum Factorization Page
elif st.session_state.current_tab == "quantum_factorization":
    st.markdown("## Quantum Factorization")
    
    st.markdown("""
    Quantum factorization algorithms, such as Shor's algorithm, offer an exponential speedup
    over classical algorithms for integer factorization. This is one of the most powerful
    known quantum advantages, with implications for cryptography and number theory.
    """)
    
    # Factorization UI
    number_to_factorize = st.number_input(
        "Enter a number to factorize:", 
        min_value=2, 
        max_value=1000000,
        value=21,
        step=1
    )
    
    # Factorize button
    if st.button("Factorize with Quantum Acceleration"):
        with st.spinner("Processing factorization with quantum acceleration..."):
            # Execute factorization
            factorization_result = st.session_state.quantum_core.run_quantum_factorization(
                number_to_factorize
            )
            
            # Add to history
            task_record = {
                "id": len(st.session_state.task_history) + 1,
                "timestamp": datetime.now().isoformat(),
                "task_type": "factorization",
                "number": number_to_factorize,
                "bit_length": number_to_factorize.bit_length(),
                "use_quantum": st.session_state.quantum_core.use_quantum,
                "factors": factorization_result.get("factors", []),
                "prime_factors": factorization_result.get("prime_factors", []),
                "classical_time": factorization_result.get("classical_time", 0),
                "quantum_time": factorization_result.get("quantum_time", 0),
                "speedup": factorization_result.get("speedup", 1.0)
            }
            st.session_state.task_history.append(task_record)
            
            # Display results
            st.success(f"Factorization completed with {task_record['speedup']:.2f}x speedup!")
            
            # Performance comparison
            st.subheader("Performance Comparison")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Classical Time", f"{task_record['classical_time']:.6f}s")
            with col2:
                st.metric("Quantum Time", f"{task_record['quantum_time']:.6f}s")
            with col3:
                st.metric("Speedup", f"{task_record['speedup']:.2f}x")
            
            # AI-enhanced explanation if available
            if st.session_state.ai_configured:
                try:
                    with st.spinner("Generating AI explanation..."):
                        explanation = asyncio.run(run_async(
                            st.session_state.ai_engine.explain_quantum_advantage,
                            "factorization",
                            number_to_factorize.bit_length(),
                            task_record['speedup'],
                            task_record['quantum_time'],
                            task_record['classical_time']
                        ))
                        st.markdown("#### Quantum Advantage Explanation")
                        st.markdown(f'<div class="info-box">{explanation}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error generating explanation: {e}")
            
            # Factorization results
            st.markdown("#### Factorization Results")
            
            # Prime factorization
            prime_factors = factorization_result.get("prime_factors", [])
            prime_factorization = " × ".join(map(str, prime_factors))
            st.markdown(f"**Prime factorization**: {number_to_factorize} = {prime_factorization}")
            
            # All factors
            factors = factorization_result.get("factors", [])
            factors_str = ", ".join(map(str, sorted(factors)))
            st.markdown(f"**All factors**: {factors_str}")
            
            # Explanation
            st.markdown("#### Detailed Explanation")
            st.markdown(factorization_result.get("explanation", ""))
            
            # Show quantum circuit details if available
            if PENNYLANE_AVAILABLE and st.session_state.show_circuit_details and number_to_factorize <= 15:
                st.markdown("### Quantum Circuit Details")
                st.markdown("""
                This section shows a simplified quantum circuit inspired by Shor's algorithm concepts:
                
                1. **Quantum Fourier Transform**: Prepares superposition for period finding
                2. **Modular Exponentiation**: Applies operations based on the number properties
                3. **Inverse QFT**: Extracts the period information
                4. **Classical Post-Processing**: Converts period to factors
                """)
                
                # Create a visualization of a simplified Shor-inspired circuit
                try:
                    code = f"""
import pennylane as qml
import numpy as np

# Create a PennyLane device
dev = qml.device('default.qubit', wires=4)

@qml.qnode(dev)
def shors_demo_circuit():
    # We're factoring {number_to_factorize}
    
    # First register: period finding register
    for i in range(2):
        qml.Hadamard(wires=i)
    
    # Second register: computational register
    # Initialize to |1⟩
    qml.PauliX(wires=2)
    
    # Apply controlled modular multiplication operations
    # In a real Shor's implementation, this would be much more complex
    # This is just a conceptual demo
    qml.ctrl(qml.PauliX(wires=3), control=0)
    qml.ctrl(qml.SWAP(wires=[2, 3]), control=1)
    
    # Apply inverse QFT to first register
    qml.Hadamard(wires=0)
    qml.ctrl(qml.PhaseShift(-np.pi/2, wires=0), control=1)
    qml.Hadamard(wires=1)
    
    return qml.probs(wires=range(4))

# Draw the circuit
print(qml.draw(shors_demo_circuit)())
                    """
                    st.markdown('<div class="code-block">', unsafe_allow_html=True)
                    st.code(code, language="python")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show plot of circuit results if available
                    if factorization_result.get("circuit_results") is not None:
                        st.markdown("### Circuit Measurement Probabilities")
                        
                        # Get probabilities and create plot
                        probs = factorization_result.get("circuit_results", [])
                        if len(probs) > 0:
                            states = [f"{i:0{len(bin(len(probs)-1))-2}b}" for i in range(len(probs))]
                            
                            chart_data = pd.DataFrame({
                                'State': states,
                                'Probability': probs
                            })
                            
                            st.bar_chart(chart_data.set_index('State'))
                        
                except Exception as e:
                    st.error(f"Error rendering quantum circuit: {e}")

# Quantum Optimization Page
elif st.session_state.current_tab == "quantum_optimization":
    st.markdown("## Quantum Optimization")
    
    st.markdown("""
    Quantum optimization algorithms like QAOA (Quantum Approximate Optimization Algorithm)
    can provide polynomial speedups for certain NP-hard optimization problems such as
    resource allocation, scheduling, and routing.
    """)
    
    # Optimization UI
    st.subheader("Configure Optimization Problem")
    
    problem_type = st.selectbox(
        "Problem Type:",
        options=["resource_allocation", "scheduling", "portfolio_optimization"],
        index=0
    )
    
    problem_size = st.slider(
        "Problem Size (dimensions):", 
        min_value=2, 
        max_value=20, 
        value=5
    )
    
    # Problem type descriptions
    problem_descriptions = {
        "resource_allocation": """
        **Resource Allocation Problem**: 
        Determining the optimal allocation of resources (e.g., compute nodes, memory) to tasks.
        The quantum algorithm helps find an allocation that maximizes utility while respecting constraints.
        """,
        "scheduling": """
        **Scheduling Problem**: 
        Finding the optimal order and timing for a set of tasks, considering dependencies and resource constraints.
        The quantum algorithm helps find a schedule that minimizes total completion time.
        """,
        "portfolio_optimization": """
        **Portfolio Optimization Problem**: 
        Selecting an optimal mix of assets to maximize returns while controlling risk.
        The quantum algorithm helps find a portfolio allocation that balances risk and return.
        """
    }
    
    # Display problem description
    st.markdown(problem_descriptions.get(problem_type, ""))
    
    # Optimize button
    if st.button("Optimize with Quantum Acceleration"):
        with st.spinner("Processing optimization with quantum acceleration..."):
            # Execute optimization
            optimization_result = st.session_state.quantum_core.run_quantum_optimization(
                problem_size, problem_type
            )
            
            # Add to history
            task_record = {
                "id": len(st.session_state.task_history) + 1,
                "timestamp": datetime.now().isoformat(),
                "task_type": "optimization",
                "problem_type": problem_type,
                "problem_size": problem_size,
                "use_quantum": st.session_state.quantum_core.use_quantum,
                "solution": optimization_result.get("solution", {}),
                "objective_value": optimization_result.get("objective_value", 0),
                "classical_time": optimization_result.get("classical_time", 0),
                "quantum_time": optimization_result.get("quantum_time", 0),
                "speedup": optimization_result.get("speedup", 1.0)
            }
            st.session_state.task_history.append(task_record)
            
            # Display results
            st.success(f"Optimization completed with {task_record['speedup']:.2f}x speedup!")
            
            # Performance comparison
            st.subheader("Performance Comparison")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Classical Time", f"{task_record['classical_time']:.4f}s")
            with col2:
                st.metric("Quantum Time", f"{task_record['quantum_time']:.4f}s")
            with col3:
                st.metric("Speedup", f"{task_record['speedup']:.2f}x")
            
            # AI-enhanced explanation if available
            if st.session_state.ai_configured:
                try:
                    with st.spinner("Generating AI explanation..."):
                        explanation = asyncio.run(run_async(
                            st.session_state.ai_engine.explain_quantum_advantage,
                            "optimization",
                            problem_size,
                            task_record['speedup'],
                            task_record['quantum_time'],
                            task_record['classical_time']
                        ))
                        st.markdown("#### Quantum Advantage Explanation")
                        st.markdown(f'<div class="info-box">{explanation}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error generating explanation: {e}")
            
            # Optimization results
            st.markdown("#### Optimization Results")
            
            # Objective value
            objective_value = optimization_result.get("objective_value", 0)
            st.metric("Objective Value", f"{objective_value:.4f}", 
                     delta=f"{optimization_result.get('improvement', 0):.2f}%")
            
            # Solution visualization depends on problem type
            st.markdown("#### Solution Visualization")
            
            if problem_type == "resource_allocation":
                # Resource allocation visualization
                solution = optimization_result.get("solution", {})
                resources = solution.get("resources", {})
                allocations = solution.get("allocations", {})
                
                # Create allocation bar chart
                if allocations:
                    allocation_data = []
                    for task, resource_allocs in allocations.items():
                        for resource, amount in resource_allocs.items():
                            allocation_data.append({
                                "Task": task,
                                "Resource": resource,
                                "Amount": amount
                            })
                    
                    df = pd.DataFrame(allocation_data)
                    fig = px.bar(df, x="Task", y="Amount", color="Resource", barmode="group")
                    st.plotly_chart(fig)
            
            elif problem_type == "scheduling":
                # Scheduling visualization (Gantt chart)
                solution = optimization_result.get("solution", {})
                schedule = solution.get("schedule", {})
                
                if schedule:
                    # Create Gantt chart data
                    gantt_data = []
                    for task, timing in schedule.items():
                        gantt_data.append({
                            "Task": task,
                            "Start": timing.get("start", 0),
                            "End": timing.get("end", 1),
                            "Resource": timing.get("resource", "Unknown")
                        })
                    
                    df = pd.DataFrame(gantt_data)
                    fig = px.timeline(df, x_start="Start", x_end="End", y="Task", color="Resource")
                    # Make the graph look like a Gantt chart
                    fig.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig)
            
            elif problem_type == "portfolio_optimization":
                # Portfolio visualization (pie chart)
                solution = optimization_result.get("solution", {})
                portfolio = solution.get("portfolio", {})
                
                if portfolio:
                    # Create pie chart
                    portfolio_data = []
                    for asset, weight in portfolio.items():
                        portfolio_data.append({
                            "Asset": asset,
                            "Weight": weight
                        })
                    
                    df = pd.DataFrame(portfolio_data)
                    fig = px.pie(df, values="Weight", names="Asset")
                    st.plotly_chart(fig)
            
            # Explanation
            st.markdown("#### Solution Explanation")
            st.markdown(optimization_result.get("explanation", ""))
            
            # Show quantum circuit details if available
            if PENNYLANE_AVAILABLE and st.session_state.show_circuit_details:
                st.markdown("### Quantum Circuit Details")
                st.markdown("""
                This section shows a simplified quantum circuit inspired by QAOA (Quantum Approximate Optimization Algorithm):
                
                1. **Preparation**: Initialize in equal superposition
                2. **Cost Unitary**: Encode problem constraints
                3. **Mixer Unitary**: Explore solution space
                4. **Iteration**: Repeat Cost and Mixer steps
                5. **Measurement**: Sample best solution
                """)
                
                # Create a visualization of a simplified QAOA circuit
                try:
                    code = f"""
import pennylane as qml
import numpy as np

# Create a PennyLane device
dev = qml.device('default.qubit', wires={problem_size})

# QAOA for a {problem_size}-variable problem
@qml.qnode(dev)
def qaoa_circuit(gamma, beta):
    # Prepare initial state (equal superposition)
    for i in range({problem_size}):
        qml.Hadamard(wires=i)
    
    # Alternating cost and mixer unitaries
    # Cost unitary - encodes the problem constraints
    for i in range({problem_size}-1):
        qml.CNOT(wires=[i, i+1])
        qml.RZ(gamma, wires=i+1)
        qml.CNOT(wires=[i, i+1])
    
    # Mixer unitary - explores the solution space
    for i in range({problem_size}):
        qml.RX(beta, wires=i)
    
    return qml.probs(wires=range({problem_size}))

# Example execution with random parameters
gamma, beta = 0.5, 0.3
result = qaoa_circuit(gamma, beta)
print(qml.draw(qaoa_circuit)(gamma, beta))
                    """
                    st.markdown('<div class="code-block">', unsafe_allow_html=True)
                    st.code(code, language="python")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show QAOA energy landscape if available
                    if "energy_landscape" in optimization_result:
                        st.markdown("### QAOA Energy Landscape")
                        energy_landscape = optimization_result.get("energy_landscape", [])
                        
                        if energy_landscape:
                            # Create and display heatmap
                            df = pd.DataFrame(energy_landscape)
                            fig = px.imshow(df, 
                                           labels=dict(x="Mixer Parameter (β)", y="Cost Parameter (γ)", color="Energy"),
                                           color_continuous_scale="Viridis")
                            st.plotly_chart(fig)
                    
                except Exception as e:
                    st.error(f"Error rendering quantum circuit: {e}")

# AI Agent Page
elif st.session_state.current_tab == "ai_agent":
    st.markdown("## Quantum-Enhanced AI Agent")
    
    st.markdown("""
    Interact with our quantum-enhanced AI agent, which can leverage quantum computing
    capabilities for appropriate tasks. The agent can understand natural language instructions
    and route computational tasks to quantum or classical processors based on the nature of the task.
    """)
    
    # Check if AI is configured
    if not st.session_state.ai_configured:
        st.warning("⚠️ Configure Claude or OpenAI API keys to enable the AI agent capabilities")
        with st.expander("Configure AI API Keys", expanded=True):
            st.markdown("""
            To use the AI agent capabilities, provide at least one of the following API keys.
            These credentials will be stored as environment variables for this session.
            """)
            
            with st.form(key="ai_api_credentials_agent"):
                anthropic_api_key = st.text_input(
                    "Anthropic API Key (for Claude 3.7)", 
                    value=os.environ.get("ANTHROPIC_API_KEY", ""),
                    type="password"
                )
                openai_api_key = st.text_input(
                    "OpenAI API Key (for GPT-4o)",
                    value=os.environ.get("OPENAI_API_KEY", ""),
                    type="password"
                )
                
                submit_button = st.form_submit_button(label="Save API Keys")
                
                if submit_button:
                    # Save credentials as environment variables
                    if anthropic_api_key:
                        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
                    
                    if openai_api_key:
                        os.environ["OPENAI_API_KEY"] = openai_api_key
                    
                    # Update session state
                    st.session_state.ai_configured = bool(anthropic_api_key or openai_api_key)
                    st.session_state.show_ai_form = False
                    
                    # Reinitialize AI engine
                    st.session_state.ai_engine = AIEngine(
                        use_claude=bool(anthropic_api_key),
                        use_openai=bool(openai_api_key)
                    )
                    
                    st.success("AI API keys configured successfully!")
                    st.experimental_rerun()
    else:
        # Display chat interface
        st.markdown("### Chat with the Quantum-AI Agent")
        
        # Initialize chat messages if not already in session state
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        
        # Display chat messages
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Get user input
        user_input = st.chat_input("Ask about quantum computing or provide a computational task...")
        
        if user_input:
            # Add user message to chat
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Process user input with AI agent
            with st.chat_message("assistant"):
                with st.spinner("The quantum-enhanced AI agent is processing your request..."):
                    try:
                        # First analyze the task
                        analysis_result = asyncio.run(run_async(
                            st.session_state.ai_engine.analyze_task,
                            user_input
                        ))
                        
                        analysis = analysis_result.get("analysis", {})
                        
                        # Check if this is a computational task that can benefit from quantum computing
                        if isinstance(analysis, dict) and analysis.get("quantum_advantage", False):
                            task_type = analysis.get("task_type", "general")
                            
                            # Process with appropriate quantum method based on task type
                            if task_type == "search":
                                st.markdown("I'll use quantum search algorithms to process this request.")
                                
                                # Extract parameters
                                parameters = analysis.get("parameters", {})
                                query = parameters.get("query", user_input)
                                database_size = parameters.get("database_size", 1000)
                                
                                # Execute search
                                search_result = st.session_state.quantum_core.run_quantum_search(
                                    query, database_size
                                )
                                
                                # Generate summary
                                summary = asyncio.run(run_async(
                                    st.session_state.ai_engine.summarize_results,
                                    "search",
                                    search_result
                                ))
                                
                                # Display results
                                st.markdown(f"**Search Results**")
                                st.markdown(summary)
                                
                                # Show top results
                                for i, result in enumerate(search_result.get("results", [])[:3]):
                                    st.markdown(f"**Result {i+1}**: {result.get('title', '')}")
                                    st.markdown(f"{result.get('content', '')}")
                                    st.markdown(f"Relevance: {result.get('relevance', 0):.1f}%")
                                
                                # Add to session history
                                quantum_response = f"I processed your search request using quantum algorithms with a {search_result.get('speedup', 1.0):.2f}x speedup.\n\n{summary}"
                                
                            elif task_type == "factorization":
                                st.markdown("I'll use quantum factorization algorithms to process this request.")
                                
                                # Extract parameters
                                parameters = analysis.get("parameters", {})
                                number = int(parameters.get("number", 21))
                                
                                # Execute factorization
                                factorization_result = st.session_state.quantum_core.run_quantum_factorization(number)
                                
                                # Generate summary
                                summary = asyncio.run(run_async(
                                    st.session_state.ai_engine.summarize_results,
                                    "factorization",
                                    factorization_result
                                ))
                                
                                # Display results
                                st.markdown(f"**Factorization Results**")
                                st.markdown(summary)
                                
                                # Show factors
                                prime_factors = factorization_result.get("prime_factors", [])
                                prime_factorization = " × ".join(map(str, prime_factors))
                                st.markdown(f"**Prime factorization**: {number} = {prime_factorization}")
                                
                                # Add to session history
                                quantum_response = f"I factorized the number {number} using quantum algorithms with a {factorization_result.get('speedup', 1.0):.2f}x speedup.\n\n{summary}"
                                
                            elif task_type == "optimization":
                                st.markdown("I'll use quantum optimization algorithms to process this request.")
                                
                                # Extract parameters
                                parameters = analysis.get("parameters", {})
                                problem_type = parameters.get("problem_type", "resource_allocation")
                                problem_size = int(parameters.get("problem_size", 5))
                                
                                # Execute optimization
                                optimization_result = st.session_state.quantum_core.run_quantum_optimization(
                                    problem_size, problem_type
                                )
                                
                                # Generate summary
                                summary = asyncio.run(run_async(
                                    st.session_state.ai_engine.summarize_results,
                                    "optimization",
                                    optimization_result
                                ))
                                
                                # Display results
                                st.markdown(f"**Optimization Results**")
                                st.markdown(summary)
                                
                                # Show objective value
                                objective_value = optimization_result.get("objective_value", 0)
                                st.markdown(f"**Objective value**: {objective_value:.4f}")
                                
                                # Add to session history
                                quantum_response = f"I solved the {problem_type} optimization problem using quantum algorithms with a {optimization_result.get('speedup', 1.0):.2f}x speedup.\n\n{summary}"
                                
                            else:
                                # General task
                                st.markdown(f"I'm analyzing your request to determine if quantum computing would be beneficial...")
                                
                                # Generate response with AI
                                ai_response = asyncio.run(run_async(
                                    st.session_state.ai_engine.clients["claude"].messages.create if st.session_state.ai_engine.use_claude else st.session_state.ai_engine.clients["openai"].chat.completions.create,
                                    model=st.session_state.ai_engine.claude_model if st.session_state.ai_engine.use_claude else st.session_state.ai_engine.openai_model,
                                    max_tokens=1000,
                                    messages=[
                                        {"role": "system", "content": "You are a quantum computing expert assistant. You can help explain quantum computing concepts and answer questions about quantum algorithms, applications, and hardware. If presented with a computational task, you should consider whether quantum computing offers an advantage."},
                                        {"role": "user", "content": user_input}
                                    ]
                                ))
                                
                                if st.session_state.ai_engine.use_claude:
                                    quantum_response = ai_response.content[0].text
                                else:
                                    quantum_response = ai_response.choices[0].message.content
                                
                                st.markdown(quantum_response)
                        
                        else:
                            # Not a quantum task, use regular AI response
                            ai_response = asyncio.run(run_async(
                                st.session_state.ai_engine.clients["claude"].messages.create if st.session_state.ai_engine.use_claude else st.session_state.ai_engine.clients["openai"].chat.completions.create,
                                model=st.session_state.ai_engine.claude_model if st.session_state.ai_engine.use_claude else st.session_state.ai_engine.openai_model,
                                max_tokens=1000,
                                messages=[
                                    {"role": "system", "content": "You are a quantum computing expert assistant. You can help explain quantum computing concepts and answer questions about quantum algorithms, applications, and hardware. If presented with a computational task, you should consider whether quantum computing offers an advantage."},
                                    {"role": "user", "content": user_input}
                                ]
                            ))
                            
                            if st.session_state.ai_engine.use_claude:
                                quantum_response = ai_response.content[0].text
                            else:
                                quantum_response = ai_response.choices[0].message.content
                            
                            st.markdown(quantum_response)
                            
                    except Exception as e:
                        st.error(f"Error processing request: {e}")
                        quantum_response = f"I encountered an error while processing your request: {str(e)}"
                        st.markdown(quantum_response)
            
            # Add assistant response to chat history
            st.session_state.chat_messages.append({"role": "assistant", "content": quantum_response})

# Task History Page
elif st.session_state.current_tab == "task_history":
    st.markdown("## Task History")
    
    st.markdown("""
    View the history of all quantum-accelerated tasks you've run in this session.
    This helps you track performance and compare speedups across different tasks.
    """)
    
    if len(st.session_state.task_history) == 0:
        st.info("No tasks have been run yet. Try running a quantum search, factorization, or optimization task.")
    else:
        # Display task history
        for i, task in enumerate(reversed(st.session_state.task_history)):
            with st.expander(f"Task {task['id']}: {task['task_type'].capitalize()} ({task['timestamp']})", expanded=i == 0):
                # Task details based on type
                if task['task_type'] == "search":
                    st.markdown(f"**Query**: {task['query']}")
                    if task.get('enhanced_query'):
                        st.markdown(f"**Enhanced Query**: {task['enhanced_query']}")
                    st.markdown(f"**Database Size**: {task['database_size']} records")
                elif task['task_type'] == "factorization":
                    st.markdown(f"**Number**: {task['number']}")
                    st.markdown(f"**Bit Length**: {task['bit_length']} bits")
                    st.markdown(f"**Factors**: {', '.join(map(str, task['factors']))}")
                elif task['task_type'] == "optimization":
                    st.markdown(f"**Problem Type**: {task['problem_type']}")
                    st.markdown(f"**Problem Size**: {task['problem_size']} dimensions")
                    st.markdown(f"**Objective Value**: {task.get('objective_value', 0):.4f}")
                
                # Performance metrics
                st.markdown("#### Performance Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Classical Time", f"{task['classical_time']:.6f}s")
                with col2:
                    st.metric("Quantum Time", f"{task['quantum_time']:.6f}s")
                with col3:
                    st.metric("Speedup", f"{task['speedup']:.2f}x")
                
                # Quantum usage
                st.markdown(f"**Quantum Acceleration**: {'Enabled' if task['use_quantum'] else 'Disabled'}")
                
                # Create visualization
                st.markdown("#### Performance Visualization")
                
                # Bar chart comparing classical vs quantum time
                chart_data = pd.DataFrame({
                    'Method': ['Classical', 'Quantum'],
                    'Time (s)': [task['classical_time'], task['quantum_time']]
                })
                
                fig = px.bar(chart_data, x='Method', y='Time (s)', 
                            title=f"Performance Comparison for {task['task_type'].capitalize()} Task",
                            color='Method', 
                            color_discrete_map={'Classical': '#A9A9A9', 'Quantum': '#1E90FF'})
                st.plotly_chart(fig)

# Main function
def main():
    # Everything is already set up and running
    pass

if __name__ == "__main__":
    main()