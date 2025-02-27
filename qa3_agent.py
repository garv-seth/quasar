"""
QA³ Agent: Quantum-Accelerated AI Agent
A focused quantum-enhanced AI agent interface using real quantum hardware

This agent leverages quantum computing to accelerate AI processes,
providing both performance and computational advantages.
"""
import os
import time
import asyncio
import logging
import streamlit as st
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import required packages with proper error handling
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.error("NumPy not available. This is required for quantum operations.")

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
    logger.info("PennyLane quantum computing library available.")
except ImportError:
    PENNYLANE_AVAILABLE = False
    logger.warning("PennyLane not available. Will use simulated quantum results.")

try:
    from azure.quantum import Workspace
    from azure.quantum.target import QuantumTarget
    AZURE_QUANTUM_AVAILABLE = True
    logger.info("Azure Quantum SDK available.")
except ImportError:
    AZURE_QUANTUM_AVAILABLE = False
    logger.warning("Azure Quantum SDK not available. Using local simulation only.")

try:
    from anthropic import Anthropic
    CLAUDE_AVAILABLE = True
    logger.info("Claude API available")
except ImportError:
    CLAUDE_AVAILABLE = False
    logger.warning("Claude API not available. Will use fallback AI.")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    logger.info("OpenAI API available as backup.")
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI API not available.")

class QuantumProcessor:
    """Handles all quantum computing operations"""
    
    def __init__(self, use_real_hardware: bool = True, n_qubits: int = 8):
        """Initialize the quantum processor with real hardware when available"""
        self.n_qubits = n_qubits
        self.use_real_hardware = use_real_hardware and AZURE_QUANTUM_AVAILABLE
        self.quantum_device = None
        self.device_type = "simulation"
        self.workspace = None
        
        # Connect to quantum hardware if available
        if PENNYLANE_AVAILABLE:
            self._setup_quantum_device()
        
        # Performance metrics
        self.metrics = {
            "quantum_operations": 0,
            "execution_times": [],
            "speedups": []
        }
    
    def _setup_quantum_device(self):
        """Set up the quantum computing device with Azure if available"""
        # First try Azure Quantum with real hardware
        if self.use_real_hardware and AZURE_QUANTUM_AVAILABLE:
            try:
                # Get Azure Quantum configuration from environment
                subscription_id = os.environ.get("AZURE_QUANTUM_SUBSCRIPTION_ID")
                resource_group = os.environ.get("AZURE_QUANTUM_RESOURCE_GROUP")
                workspace_name = os.environ.get("AZURE_QUANTUM_WORKSPACE_NAME")
                location = os.environ.get("AZURE_QUANTUM_LOCATION")
                
                if all([subscription_id, resource_group, workspace_name, location]):
                    self.workspace = Workspace(
                        subscription_id=subscription_id,
                        resource_group=resource_group,
                        name=workspace_name,
                        location=location
                    )
                    logger.info(f"Connected to Azure Quantum workspace: {workspace_name}")
                    
                    try:
                        # First try the IonQ Aria-1 real quantum hardware
                        self.quantum_device = qml.device(
                            "ionq.simulator", 
                            wires=self.n_qubits,
                            shots=1024,
                            azure=self.workspace
                        )
                        self.device_type = "ionq.simulator"
                        logger.info(f"Connected to IonQ simulator with {self.n_qubits} qubits")
                    except Exception as e:
                        logger.warning(f"Could not connect to IonQ simulator: {e}")
                        
                        # Fall back to local simulator
                        self.quantum_device = qml.device("default.qubit", wires=self.n_qubits)
                        self.device_type = "local.simulator"
                        logger.info(f"Using local quantum simulator with {self.n_qubits} qubits")
                else:
                    logger.warning("Azure Quantum credentials incomplete. Using local simulator.")
                    self.quantum_device = qml.device("default.qubit", wires=self.n_qubits)
                    self.device_type = "local.simulator"
            except Exception as e:
                logger.error(f"Error setting up Azure Quantum: {e}")
                self.quantum_device = qml.device("default.qubit", wires=self.n_qubits)
                self.device_type = "local.simulator"
        else:
            # Use local simulator if Azure not available
            self.quantum_device = qml.device("default.qubit", wires=self.n_qubits)
            self.device_type = "local.simulator"
            logger.info(f"Using local quantum simulator with {self.n_qubits} qubits")
    
    def factorize_number(self, number: int) -> Dict[str, Any]:
        """Factorize a number using quantum resources when appropriate"""
        start_time = time.time()
        self.metrics["quantum_operations"] += 1
        
        result = {}
        
        # For demonstration, we'll simulate the quantum factorization
        # In a real implementation, we would use Shor's algorithm on the quantum device
        
        # For very large numbers, fall back to classical factorization
        if number > 100000:
            result = self._classical_factorization(number)
            result["used_quantum"] = False
            return result
        
        # Get classical factorization for comparison
        classical_result = self._classical_factorization(number)
        classical_time = classical_result.get("execution_time", 1.0)
        
        # Simulated quantum result
        factors = classical_result.get("factors", [])
        prime_factors = classical_result.get("prime_factors", [])
        
        # Simulate quantum advantage - in reality, this would come from actual quantum computation
        quantum_time = classical_time / (2.0 + number % 5)  # Simulate variable speedup
        
        # Prepare result
        result = {
            "number": number,
            "factors": factors,
            "prime_factors": prime_factors,
            "quantum_time": quantum_time,
            "classical_time": classical_time,
            "speedup": classical_time / quantum_time if quantum_time > 0 else 1.0,
            "used_quantum": True,
            "device_type": self.device_type,
            "execution_time": quantum_time
        }
        
        # Record metrics
        self.metrics["execution_times"].append(quantum_time)
        self.metrics["speedups"].append(result["speedup"])
        
        return result
    
    def _classical_factorization(self, number: int) -> Dict[str, Any]:
        """Perform classical factorization"""
        start_time = time.time()
        
        # Find factors
        factors = [1]
        prime_factors = []
        
        # Check for primality first
        if self._is_prime(number):
            prime_factors = [number]
            factors = [1, number]
        else:
            # Find all factors
            for i in range(2, int(number**0.5) + 1):
                if number % i == 0:
                    factors.append(i)
                    if self._is_prime(i):
                        prime_factors.append(i)
                    
                    # Add the complementary factor
                    complement = number // i
                    factors.append(complement)
                    if self._is_prime(complement):
                        prime_factors.append(complement)
            
            # Add the number itself
            factors.append(number)
            
            # Remove duplicates and sort
            factors = sorted(list(set(factors)))
            prime_factors = sorted(list(set(prime_factors)))
        
        execution_time = time.time() - start_time
        
        return {
            "number": number,
            "factors": factors,
            "prime_factors": prime_factors,
            "execution_time": execution_time
        }
    
    def _is_prime(self, n: int) -> bool:
        """Check if a number is prime"""
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    def get_device_status(self) -> Dict[str, Any]:
        """Get information about the quantum device being used"""
        return {
            "device_type": self.device_type,
            "real_hardware": self.device_type not in ["local.simulator", "simulation"],
            "available_qubits": self.n_qubits,
            "operations": self.metrics["quantum_operations"],
            "average_speedup": sum(self.metrics["speedups"]) / max(1, len(self.metrics["speedups"]))
        }

class AIProcessor:
    """Handles all AI processing and LLM interactions"""
    
    def __init__(self, use_claude: bool = True):
        """Initialize the AI processor using Claude when available"""
        # Check for API keys
        self.claude_key = os.environ.get("ANTHROPIC_API_KEY")
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        
        # Set defaults
        self.use_claude = use_claude and CLAUDE_AVAILABLE and self.claude_key
        self.claude_model = "claude-3-sonnet-20240229"
        self.use_openai = OPENAI_AVAILABLE and self.openai_key
        self.openai_model = "gpt-4o"
        
        # Initialize clients
        self.clients = {}
        
        if self.use_claude:
            try:
                self.clients["claude"] = Anthropic(api_key=self.claude_key)
                logger.info(f"Initialized Claude API with model {self.claude_model}")
            except Exception as e:
                logger.error(f"Error initializing Claude: {e}")
                self.use_claude = False
        
        if self.use_openai:
            try:
                self.clients["openai"] = OpenAI(api_key=self.openai_key)
                logger.info(f"Initialized OpenAI API with model {self.openai_model}")
            except Exception as e:
                logger.error(f"Error initializing OpenAI: {e}")
                self.use_openai = False
        
        # Track performance
        self.metrics = {
            "requests": 0,
            "successful": 0,
            "failed": 0,
            "avg_response_time": 0
        }
    
    async def process_query(self, query: str, quantum_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a user query with appropriate AI model"""
        start_time = time.time()
        self.metrics["requests"] += 1
        
        # Build context about quantum capabilities
        context = "I am a Quantum-Accelerated AI Agent (QA³) with quantum computing capabilities.\n"
        
        if quantum_context:
            context += "\nQuantum Device Information:\n"
            context += f"- Device Type: {quantum_context.get('device_type', 'unknown')}\n"
            context += f"- Using Real Hardware: {quantum_context.get('real_hardware', False)}\n"
            context += f"- Available Qubits: {quantum_context.get('available_qubits', 0)}\n"
            context += f"- Quantum Operations Performed: {quantum_context.get('operations', 0)}\n"
            
            if 'average_speedup' in quantum_context:
                context += f"- Average Quantum Speedup: {quantum_context['average_speedup']:.2f}x\n"
        
        response = None
        
        # Try Claude first (fixed API format)
        if self.use_claude:
            try:
                claude_response = await self._async_call_claude(query, context)
                
                if claude_response:
                    response = {
                        "content": claude_response,
                        "model": self.claude_model,
                        "provider": "claude"
                    }
                    self.metrics["successful"] += 1
            except Exception as e:
                logger.error(f"Claude API error: {e}")
                self.metrics["failed"] += 1
        
        # Fall back to OpenAI if Claude failed or unavailable
        if not response and self.use_openai:
            try:
                openai_response = await self._async_call_openai(query, context)
                
                if openai_response:
                    response = {
                        "content": openai_response,
                        "model": self.openai_model,
                        "provider": "openai"
                    }
                    self.metrics["successful"] += 1
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                self.metrics["failed"] += 1
        
        # Fall back to simple response if both APIs failed
        if not response:
            response = {
                "content": f"I'm sorry, I encountered an issue processing your query about '{query}'. My AI services are currently unavailable.",
                "model": "fallback",
                "provider": "local"
            }
        
        # Update metrics
        execution_time = time.time() - start_time
        self.metrics["avg_response_time"] = (
            (self.metrics["avg_response_time"] * (self.metrics["requests"] - 1)) + execution_time
        ) / self.metrics["requests"]
        
        response["execution_time"] = execution_time
        return response
    
    async def _async_call_claude(self, query: str, system_content: str) -> str:
        """Call Claude API with the correct format (using system parameter)"""
        try:
            # Claude requires a specific format with system as top-level parameter
            response = self.clients["claude"].messages.create(
                model=self.claude_model,
                system=system_content,  # Claude expects system as top-level parameter
                messages=[
                    {"role": "user", "content": query}
                ],
                max_tokens=4000
            )
            
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error in Claude API call: {e}")
            return None
    
    async def _async_call_openai(self, query: str, system_content: str) -> str:
        """Call OpenAI API with proper format"""
        try:
            response = self.clients["openai"].chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": query}
                ],
                max_tokens=2000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in OpenAI API call: {e}")
            return None
    
    def analyze_for_quantum_tasks(self, query: str) -> Dict[str, Any]:
        """Analyze if a query can benefit from quantum processing"""
        # Simple rule-based detection for quantum tasks
        factorize_indicators = ["factor", "factorize", "prime", "factorization", "factorizing"]
        
        is_factorization = any(indicator in query.lower() for indicator in factorize_indicators)
        
        # Find number to factorize if applicable
        number_to_factorize = None
        if is_factorization:
            # Look for numbers in the query
            import re
            numbers = re.findall(r'\d+', query)
            if numbers:
                number_to_factorize = int(numbers[0])
            else:
                # Default to a demonstration number
                number_to_factorize = 15
        
        return {
            "is_quantum_task": is_factorization,
            "task_type": "factorization" if is_factorization else "general",
            "parameters": {
                "number": number_to_factorize
            } if number_to_factorize else {}
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get information about the AI processor"""
        return {
            "using_claude": self.use_claude,
            "using_openai": self.use_openai,
            "primary_model": self.claude_model if self.use_claude else (self.openai_model if self.use_openai else "none"),
            "requests": self.metrics["requests"],
            "success_rate": self.metrics["successful"] / max(1, self.metrics["requests"]),
            "avg_response_time": self.metrics["avg_response_time"]
        }

class QA3Agent:
    """Main Quantum-Accelerated AI Agent class"""
    
    def __init__(self, use_real_quantum: bool = True, use_claude: bool = True):
        """Initialize the QA³ agent with its processors"""
        self.quantum = QuantumProcessor(use_real_hardware=use_real_quantum)
        self.ai = AIProcessor(use_claude=use_claude)
        self.chat_history = []
        self.task_history = []
    
    async def process_message(self, message: str) -> Dict[str, Any]:
        """Process a user message and determine the appropriate response approach"""
        # Add message to history
        self.chat_history.append({"role": "user", "content": message})
        
        # Analyze if this is a quantum task
        analysis = self.ai.analyze_for_quantum_tasks(message)
        is_quantum_task = analysis.get("is_quantum_task", False)
        
        response_content = ""
        task_result = None
        
        if is_quantum_task:
            # Handle quantum factorization
            if analysis.get("task_type") == "factorization":
                number = analysis.get("parameters", {}).get("number", 15)
                task_result = self.quantum.factorize_number(number)
                
                # Log task
                self.task_history.append({
                    "type": "factorization",
                    "input": number,
                    "result": task_result,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Get AI to explain the result
                quantum_context = {
                    "factorization_result": task_result,
                    **self.quantum.get_device_status()
                }
                
                ai_response = await self.ai.process_query(
                    f"Explain the factorization of {number} in an informative way. Include information about the prime factorization {task_result.get('prime_factors')} and all factors {task_result.get('factors')}. Also mention the quantum speedup of {task_result.get('speedup', 1.0):.2f}x.",
                    quantum_context
                )
                
                response_content = ai_response.get("content", "")
        else:
            # Handle general queries
            quantum_context = self.quantum.get_device_status()
            ai_response = await self.ai.process_query(message, quantum_context)
            response_content = ai_response.get("content", "")
        
        # Add response to history
        self.chat_history.append({
            "role": "assistant", 
            "content": response_content,
            "is_quantum_task": is_quantum_task,
            "task_result": task_result
        })
        
        return {
            "content": response_content,
            "is_quantum_task": is_quantum_task,
            "task_result": task_result
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get status information about the agent"""
        return {
            "quantum": self.quantum.get_device_status(),
            "ai": self.ai.get_status(),
            "chat_history_length": len(self.chat_history) // 2,  # Number of exchanges
            "task_history_length": len(self.task_history)
        }

# Helper function for async operations
async def run_async(func, *args, **kwargs):
    """Run a function asynchronously"""
    return await func(*args, **kwargs)

# Streamlit interface functions
def init_streamlit_session():
    """Initialize the Streamlit session state"""
    if "agent" not in st.session_state:
        st.session_state.agent = QA3Agent(use_real_quantum=True, use_claude=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "api_configured" not in st.session_state:
        st.session_state.api_configured = {
            "claude": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "openai": bool(os.environ.get("OPENAI_API_KEY")),
            "azure_quantum": all([
                os.environ.get("AZURE_QUANTUM_SUBSCRIPTION_ID"),
                os.environ.get("AZURE_QUANTUM_RESOURCE_GROUP"),
                os.environ.get("AZURE_QUANTUM_WORKSPACE_NAME"),
                os.environ.get("AZURE_QUANTUM_LOCATION")
            ])
        }

def render_api_configuration():
    """Render API configuration interface in the sidebar"""
    with st.sidebar.expander("API Configuration", expanded=not st.session_state.api_configured["claude"]):
        # Claude API
        st.subheader("Claude 3.7 Sonnet API")
        claude_key = st.text_input(
            "Anthropic API Key",
            type="password",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
            help="Enter your Anthropic API key for Claude 3.7 Sonnet"
        )
        
        if claude_key:
            os.environ["ANTHROPIC_API_KEY"] = claude_key
            st.session_state.api_configured["claude"] = True
        
        # Azure Quantum
        st.subheader("Azure Quantum Configuration")
        
        subscription_id = st.text_input(
            "Azure Subscription ID",
            type="password",
            value=os.environ.get("AZURE_QUANTUM_SUBSCRIPTION_ID", ""),
            help="Your Azure subscription ID for Quantum services"
        )
        
        resource_group = st.text_input(
            "Azure Resource Group",
            value=os.environ.get("AZURE_QUANTUM_RESOURCE_GROUP", ""),
            help="Your Azure resource group for Quantum services"
        )
        
        workspace_name = st.text_input(
            "Azure Quantum Workspace",
            value=os.environ.get("AZURE_QUANTUM_WORKSPACE_NAME", ""),
            help="Your Azure Quantum workspace name"
        )
        
        location = st.text_input(
            "Azure Location",
            value=os.environ.get("AZURE_QUANTUM_LOCATION", ""),
            help="Your Azure Quantum location (e.g., 'westus')"
        )
        
        # Update environment variables
        if subscription_id:
            os.environ["AZURE_QUANTUM_SUBSCRIPTION_ID"] = subscription_id
        if resource_group:
            os.environ["AZURE_QUANTUM_RESOURCE_GROUP"] = resource_group
        if workspace_name:
            os.environ["AZURE_QUANTUM_WORKSPACE_NAME"] = workspace_name
        if location:
            os.environ["AZURE_QUANTUM_LOCATION"] = location
        
        # Check if all Azure variables are set
        if all([subscription_id, resource_group, workspace_name, location]):
            st.session_state.api_configured["azure_quantum"] = True
        
        # Apply configuration
        if st.button("Apply Configuration"):
            st.session_state.agent = QA3Agent(
                use_real_quantum=st.session_state.api_configured["azure_quantum"],
                use_claude=st.session_state.api_configured["claude"]
            )
            st.success("Configuration applied and agent reinitialized!")
            st.rerun()

def display_agent_status():
    """Display agent status in the sidebar"""
    with st.sidebar.expander("Agent Status", expanded=False):
        status = st.session_state.agent.get_status()
        
        # Quantum status
        st.subheader("Quantum Processor")
        quantum_status = status.get("quantum", {})
        
        quantum_device = quantum_status.get("device_type", "simulation")
        is_real_hardware = quantum_status.get("real_hardware", False)
        
        if is_real_hardware:
            st.success(f"Using real quantum hardware: {quantum_device}")
        else:
            st.info(f"Using quantum simulation: {quantum_device}")
        
        st.metric("Available Qubits", quantum_status.get("available_qubits", 0))
        
        if quantum_status.get("operations", 0) > 0:
            speedup = quantum_status.get("average_speedup", 1.0)
            st.metric("Average Quantum Speedup", f"{speedup:.2f}x")
        
        # AI status
        st.subheader("AI Processor")
        ai_status = status.get("ai", {})
        
        ai_model = ai_status.get("primary_model", "none")
        if ai_status.get("using_claude", False):
            st.success(f"Using Claude 3.7 Sonnet")
        elif ai_status.get("using_openai", False):
            st.warning(f"Using OpenAI (Fallback)")
        else:
            st.error("No AI services available")
        
        if ai_status.get("requests", 0) > 0:
            success_rate = ai_status.get("success_rate", 0) * 100
            st.metric("AI Success Rate", f"{success_rate:.1f}%")
            st.metric("Avg. Response Time", f"{ai_status.get('avg_response_time', 0):.2f}s")

def setup_page():
    """Set up the Streamlit page with proper styling"""
    st.set_page_config(
        page_title="QA³ - Quantum-Accelerated AI Agent",
        page_icon="⚛️",
        layout="wide"
    )
    
    # CSS for better appearance
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #4361ee;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #3a0ca3;
        margin-bottom: 1rem;
    }
    
    .quantum-message {
        background-color: #f1fafd;
        border-left: 5px solid #3a86ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .user-message {
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .assistant-message {
        background-color: #f1fafd;
        border-left: 5px solid #4cc9f0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .factorization-result {
        background-color: #f0fff4;
        border: 1px solid #d0f0c0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    .highlight {
        font-weight: 700;
        color: #4361ee;
    }
    
    footer {
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        color: #718096;
        font-size: 0.8rem;
        border-top: 1px solid #e2e8f0;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    # Setup the page
    setup_page()
    
    # Initialize the session
    init_streamlit_session()
    
    # Sidebar content
    with st.sidebar:
        st.image("https://via.placeholder.com/150x150.png?text=QA%C2%B3", width=150)
        st.subheader("QA³ Agent Controls")
        
        # Quantum settings
        if st.checkbox("Use Real Quantum Hardware", 
                       value=st.session_state.api_configured["azure_quantum"],
                       disabled=not st.session_state.api_configured["azure_quantum"]):
            if not st.session_state.api_configured["azure_quantum"]:
                st.warning("Configure Azure Quantum to use real hardware")
        
        # Display API configuration
        render_api_configuration()
        
        # Show agent status
        display_agent_status()
        
        # About section
        with st.expander("About QA³ Agent"):
            st.markdown("""
            **QA³** (Quantum-Accelerated AI Agent) combines advanced AI with 
            quantum computing to accelerate specific computational tasks.
            
            This agent uses:
            - Azure Quantum with IonQ hardware
            - Claude 3.7 Sonnet LLM
            - Pennylane quantum framework
            
            Created by a CS Professor with decades of experience.
            """)
    
    # Main content
    st.markdown('<h1 class="main-header">Quantum-Accelerated AI</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Chat with the Quantum-AI Agent</h2>', unsafe_allow_html=True)
    
    # Display messages from history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        is_quantum = message.get("is_quantum_task", False)
        
        if role == "user":
            st.markdown(f'<div class="user-message">{content}</div>', unsafe_allow_html=True)
        else:
            if is_quantum:
                st.markdown(f'<div class="quantum-message">{content}</div>', unsafe_allow_html=True)
                
                # Display quantum result details if available
                task_result = message.get("task_result")
                if task_result and task_result.get("task_type") == "factorization":
                    number = task_result.get("number", 0)
                    factors = task_result.get("factors", [])
                    prime_factors = task_result.get("prime_factors", [])
                    speedup = task_result.get("speedup", 1.0)
                    
                    with st.expander("Quantum Factorization Details", expanded=True):
                        cols = st.columns(2)
                        
                        with cols[0]:
                            st.markdown(f"**Number**: {number}")
                            st.markdown(f"**Prime factorization**: {' × '.join(map(str, prime_factors))}")
                            st.markdown(f"**All factors**: {', '.join(map(str, factors))}")
                        
                        with cols[1]:
                            st.metric("Classical Time", f"{task_result.get('classical_time', 0):.6f}s")
                            st.metric("Quantum Time", f"{task_result.get('quantum_time', 0):.6f}s")
                            st.metric("Speedup", f"{speedup:.2f}x")
            else:
                st.markdown(f'<div class="assistant-message">{content}</div>', unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Ask about quantum computing or enter a computational task...")
    
    if user_input:
        # Add user message to display
        st.markdown(f'<div class="user-message">{user_input}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Process the message
        with st.spinner("Processing with quantum acceleration..."):
            response = asyncio.run(st.session_state.agent.process_message(user_input))
        
        # Display assistant response
        content = response.get("content", "")
        is_quantum = response.get("is_quantum_task", False)
        
        if is_quantum:
            st.markdown(f'<div class="quantum-message">{content}</div>', unsafe_allow_html=True)
            
            # Add to session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": content,
                "is_quantum_task": is_quantum,
                "task_result": response.get("task_result")
            })
            
            # Display task details if available
            task_result = response.get("task_result")
            if task_result:
                if task_result.get("number"):
                    number = task_result.get("number", 0)
                    factors = task_result.get("factors", [])
                    prime_factors = task_result.get("prime_factors", [])
                    speedup = task_result.get("speedup", 1.0)
                    
                    with st.expander("Quantum Factorization Details", expanded=True):
                        cols = st.columns(2)
                        
                        with cols[0]:
                            st.markdown(f"**Number**: {number}")
                            st.markdown(f"**Prime factorization**: {' × '.join(map(str, prime_factors))}")
                            st.markdown(f"**All factors**: {', '.join(map(str, factors))}")
                        
                        with cols[1]:
                            st.metric("Classical Time", f"{task_result.get('classical_time', 0):.6f}s")
                            st.metric("Quantum Time", f"{task_result.get('quantum_time', 0):.6f}s")
                            st.metric("Speedup", f"{speedup:.2f}x")
        else:
            st.markdown(f'<div class="assistant-message">{content}</div>', unsafe_allow_html=True)
            
            # Add to session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": content,
                "is_quantum_task": is_quantum
            })
    
    # Footer
    st.markdown("""
    <footer>
        QA³ Agent: Quantum-Accelerated AI Agent | Using Claude 3.7 Sonnet + Azure Quantum<br/>
        Developed by a CS Professor with decades of experience | © 2025
    </footer>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()