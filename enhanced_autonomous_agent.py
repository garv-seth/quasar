"""
Enhanced Autonomous Agent: Integrating Quantum Computing with True Agency

This agent combines quantum computing capabilities with advanced autonomous behavior, 
enabling true agentic interactions with digital systems.
"""

import os
import logging
import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import base64
import re
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enhanced-autonomous-agent")

# Import framework components with proper error handling
try:
    from web_interaction_agent import WebInteractionAgent
    WEB_AGENT_AVAILABLE = True
except ImportError:
    WEB_AGENT_AVAILABLE = False
    logger.warning("Web interaction agent not available.")

try:
    from vision_system import VisionSystem
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    logger.warning("Vision system not available.")

try:
    from autonomous_decision_system import GoalManager, DecisionMaker, SelfEvaluationSystem
    DECISION_SYSTEM_AVAILABLE = True
except ImportError:
    DECISION_SYSTEM_AVAILABLE = False
    logger.warning("Autonomous decision system not available.")

# Quantum components with proper error handling
try:
    import pennylane as qml
    from pennylane import numpy as np
    QUANTUM_AVAILABLE = True
    logger.info("PennyLane quantum computing library available.")
except ImportError:
    QUANTUM_AVAILABLE = False
    logger.warning("PennyLane not available. Using classical processing only.")

# Azure Quantum with proper error handling
try:
    from azure.quantum import Workspace
    AZURE_QUANTUM_AVAILABLE = True
    logger.info("Azure Quantum SDK available.")
except ImportError:
    AZURE_QUANTUM_AVAILABLE = False
    logger.warning("Azure Quantum SDK not available. Using local simulation only.")

# AI models with proper error handling
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI API not available.")

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic API not available.")

# Memory system for the agent
class AgentMemory:
    """Enhanced memory system for autonomous agent"""
    
    def __init__(self, max_short_term_items: int = 20):
        """Initialize agent memory systems"""
        self.short_term = []  # Recent interactions and observations
        self.working_memory = {}  # Current task context
        self.long_term = []  # Important learned information
        self.web_history = []  # Web navigation history
        self.screen_observations = []  # Screenshots and visual observations
        self.max_short_term = max_short_term_items
        
    def add_interaction(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add an interaction to short-term memory with proper timestamp"""
        interaction = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.short_term.append(interaction)
        
        # Trim short-term memory if it exceeds the limit
        if len(self.short_term) > self.max_short_term:
            self.short_term = self.short_term[-self.max_short_term:]
            
    def add_to_working_memory(self, key: str, value: Any):
        """Store information in working memory"""
        self.working_memory[key] = value
        
    def get_from_working_memory(self, key: str, default: Any = None) -> Any:
        """Retrieve information from working memory"""
        return self.working_memory.get(key, default)
        
    def add_to_long_term(self, item: Dict[str, Any]):
        """Store important information in long-term memory"""
        if not isinstance(item, dict):
            item = {"content": str(item)}
            
        if "timestamp" not in item:
            item["timestamp"] = datetime.now().isoformat()
            
        self.long_term.append(item)
        
    def add_web_visit(self, url: str, title: str, content_summary: str, metadata: Optional[Dict[str, Any]] = None):
        """Record web browsing history with metadata"""
        visit = {
            "url": url,
            "title": title,
            "content_summary": content_summary,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.web_history.append(visit)
        
    def add_screen_observation(self, screenshot_base64: str, elements_detected: List[Dict[str, Any]], description: str):
        """Record screen observation with detected UI elements"""
        observation = {
            "screenshot_base64": screenshot_base64,
            "elements_detected": elements_detected,
            "description": description,
            "timestamp": datetime.now().isoformat()
        }
        
        self.screen_observations.append(observation)
        
        # Keep only the latest 10 observations to save memory
        if len(self.screen_observations) > 10:
            self.screen_observations = self.screen_observations[-10:]
            
    def get_conversation_context(self, include_web_history: bool = True, max_items: int = 10) -> str:
        """Generate a context summary for decision making"""
        context = "Recent interactions:\n"
        
        # Add recent interactions from short-term memory
        for item in self.short_term[-max_items:]:
            role = item["role"]
            content = item["content"]
            timestamp = item["timestamp"].split("T")[1][:8]  # Extract time portion
            context += f"[{timestamp}] {role}: {content[:200]}...\n"
            
        # Add recent web visits if requested
        if include_web_history and self.web_history:
            context += "\nRecent web visits:\n"
            for visit in self.web_history[-5:]:
                timestamp = visit["timestamp"].split("T")[1][:8]
                context += f"[{timestamp}] {visit['title']} - {visit['url']}\n"
                
        # Add key working memory items
        if self.working_memory:
            context += "\nCurrent context:\n"
            for key, value in self.working_memory.items():
                if isinstance(value, (str, int, float, bool)):
                    context += f"- {key}: {value}\n"
                elif isinstance(value, dict):
                    context += f"- {key}: {json.dumps(value)[:100]}...\n"
                elif isinstance(value, list):
                    context += f"- {key}: List with {len(value)} items\n"
                    
        return context
        
    def clear_working_memory(self):
        """Clear working memory for new tasks"""
        self.working_memory = {}
        
    def get_relevant_memories(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Simple retrieval of relevant memories based on keyword matching"""
        # In a real implementation, this would use embeddings and semantic search
        query_terms = set(query.lower().split())
        scored_memories = []
        
        # Score long-term memories by term overlap
        for memory in self.long_term:
            content = memory.get("content", "").lower()
            score = sum(1 for term in query_terms if term in content)
            if score > 0:
                scored_memories.append((score, memory))
                
        # Sort by score (descending) and return top results
        scored_memories.sort(reverse=True, key=lambda x: x[0])
        return [memory for score, memory in scored_memories[:max_results]]


class QuantumEnhancedDecisionSystem:
    """
    Quantum-enhanced decision system for autonomous behavior
    
    This component integrates quantum computing to enhance decision-making
    capabilities:
    1. Quantum-accelerated option evaluation
    2. Uncertainty modeling with quantum superposition
    3. Decision optimization using quantum circuits
    """
    
    def __init__(self, 
                n_qubits: int = 8, 
                use_quantum: bool = True, 
                use_azure: bool = False):
        """
        Initialize the quantum-enhanced decision system
        
        Args:
            n_qubits: Number of qubits to use for quantum circuits
            use_quantum: Whether to use quantum processing
            use_azure: Whether to use Azure Quantum (if available)
        """
        self.n_qubits = n_qubits
        self.use_quantum = use_quantum and QUANTUM_AVAILABLE
        self.use_azure = use_azure and AZURE_QUANTUM_AVAILABLE
        
        # Metrics
        self.decisions_made = 0
        self.quantum_evaluations = 0
        self.classical_evaluations = 0
        self.quantum_advantage_metrics = []
        
        # Initialize decision components
        self.goal_manager = GoalManager() if DECISION_SYSTEM_AVAILABLE else None
        self.decision_maker = DecisionMaker() if DECISION_SYSTEM_AVAILABLE else None
        self.self_evaluation = SelfEvaluationSystem() if DECISION_SYSTEM_AVAILABLE else None
        
        # Initialize quantum devices if available
        self._initialize_quantum_devices()
        
    def _initialize_quantum_devices(self):
        """Initialize quantum devices if available"""
        if not self.use_quantum:
            self.quantum_device = None
            return
            
        try:
            if self.use_azure and AZURE_QUANTUM_AVAILABLE:
                # Try to use Azure Quantum IonQ device if credentials are available
                subscription_id = os.environ.get("AZURE_QUANTUM_SUBSCRIPTION_ID")
                resource_group = os.environ.get("AZURE_QUANTUM_RESOURCE_GROUP")
                workspace_name = os.environ.get("AZURE_QUANTUM_WORKSPACE_NAME")
                
                if subscription_id and resource_group and workspace_name:
                    self.workspace = Workspace(
                        subscription_id=subscription_id,
                        resource_group=resource_group,
                        name=workspace_name
                    )
                    self.quantum_device = qml.device("microsoft.ionq", wires=self.n_qubits, shots=1000, workspace=self.workspace)
                    logger.info(f"Azure Quantum IonQ device initialized with {self.n_qubits} qubits")
                else:
                    # Fall back to simulation
                    self.quantum_device = qml.device("default.qubit", wires=self.n_qubits)
                    logger.info(f"Quantum simulator initialized with {self.n_qubits} qubits (Azure credentials not available)")
            else:
                # Use local simulator
                self.quantum_device = qml.device("default.qubit", wires=self.n_qubits)
                logger.info(f"Quantum simulator initialized with {self.n_qubits} qubits")
                
        except Exception as e:
            logger.error(f"Error initializing quantum device: {str(e)}")
            self.quantum_device = None
            self.use_quantum = False
            
    async def evaluate_options(self, 
                            options: List[Dict[str, Any]], 
                            criteria: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate decision options using quantum or classical methods
        
        Args:
            options: List of options to evaluate
            criteria: List of criteria to use for evaluation
            
        Returns:
            List of options with evaluation scores
        """
        if self.use_quantum and self.quantum_device is not None:
            return await self._quantum_evaluate_options(options, criteria)
        else:
            return self._classical_evaluate_options(options, criteria)
            
    async def _quantum_evaluate_options(self, 
                                    options: List[Dict[str, Any]], 
                                    criteria: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate options using quantum circuits for improved decision making"""
        start_time = time.time()
        evaluated_options = []
        
        try:
            # Define quantum circuit for decision evaluation
            @qml.qnode(self.quantum_device)
            def decision_circuit(weights, option_features):
                # Encode option features into quantum states
                for i, feature in enumerate(option_features):
                    if i < self.n_qubits:
                        qml.RY(feature * np.pi, wires=i)
                        
                # Apply entangling layers
                for layer in range(2):
                    # Entangle qubits
                    for i in range(self.n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                        
                    # Apply weighted rotations
                    for i in range(self.n_qubits):
                        weight_idx = layer * self.n_qubits + i
                        if weight_idx < len(weights):
                            qml.RY(weights[weight_idx], wires=i)
                            
                # Measure all qubits
                return [qml.expval(qml.PauliZ(i)) for i in range(min(len(criteria), self.n_qubits))]
                
            # Process each option
            for option in options:
                # Convert option attributes to numerical features
                option_features = []
                for criterion in criteria:
                    criterion_id = criterion["id"]
                    if criterion_id in option.get("attributes", {}):
                        # Normalize value to [0, 1] range
                        value = option["attributes"][criterion_id]
                        if isinstance(value, (int, float)):
                            normalized_value = max(0.0, min(1.0, float(value) / criterion.get("max_value", 1.0)))
                            option_features.append(normalized_value)
                        else:
                            option_features.append(0.5)  # Default for non-numeric values
                    else:
                        option_features.append(0.5)  # Default value
                        
                # Pad features if needed
                while len(option_features) < self.n_qubits:
                    option_features.append(0.0)
                    
                # Generate random weights for the circuit (in a real system, these would be learned)
                weights = np.random.uniform(0, 2 * np.pi, size=2 * self.n_qubits)
                
                # Evaluate option using quantum circuit
                criterion_scores = decision_circuit(weights, option_features)
                
                # Map quantum values from [-1, 1] to [0, 1] range
                normalized_scores = [(score + 1) / 2 for score in criterion_scores]
                
                # Add evaluation to option
                option["evaluation"] = {}
                for i, criterion in enumerate(criteria):
                    if i < len(normalized_scores):
                        criterion_id = criterion["id"]
                        option["evaluation"][criterion_id] = {
                            "score": float(normalized_scores[i]),
                            "method": "quantum"
                        }
                        
                # Calculate overall score as weighted average
                total_weight = sum(criterion.get("weight", 1.0) for criterion in criteria)
                weighted_sum = sum(
                    option["evaluation"][criterion["id"]]["score"] * criterion.get("weight", 1.0)
                    for criterion in criteria
                    if criterion["id"] in option["evaluation"]
                )
                
                option["overall_score"] = weighted_sum / total_weight if total_weight > 0 else 0.0
                evaluated_options.append(option)
                
            # Update metrics
            self.quantum_evaluations += 1
            processing_time = time.time() - start_time
            
            return evaluated_options
            
        except Exception as e:
            logger.error(f"Error in quantum evaluation: {str(e)}")
            # Fall back to classical evaluation
            return self._classical_evaluate_options(options, criteria)
            
    def _classical_evaluate_options(self, 
                                 options: List[Dict[str, Any]], 
                                 criteria: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate options using classical methods"""
        start_time = time.time()
        evaluated_options = []
        
        for option in options:
            # Add evaluation to option
            option["evaluation"] = {}
            
            for criterion in criteria:
                criterion_id = criterion["id"]
                
                # Get criterion value if available
                value = option.get("attributes", {}).get(criterion_id, 0.5)
                
                # Calculate score based on criterion type
                if criterion.get("type") == "maximize":
                    score = min(1.0, max(0.0, value / criterion.get("max_value", 1.0)))
                elif criterion.get("type") == "minimize":
                    score = 1.0 - min(1.0, max(0.0, value / criterion.get("max_value", 1.0)))
                else:
                    # For boolean or categorical criteria
                    score = float(value) if isinstance(value, (int, float)) else 0.5
                    
                option["evaluation"][criterion_id] = {
                    "score": score,
                    "method": "classical"
                }
                
            # Calculate overall score as weighted average
            total_weight = sum(criterion.get("weight", 1.0) for criterion in criteria)
            weighted_sum = sum(
                option["evaluation"][criterion["id"]]["score"] * criterion.get("weight", 1.0)
                for criterion in criteria
                if criterion["id"] in option["evaluation"]
            )
            
            option["overall_score"] = weighted_sum / total_weight if total_weight > 0 else 0.0
            evaluated_options.append(option)
            
        # Update metrics
        self.classical_evaluations += 1
        processing_time = time.time() - start_time
        
        return evaluated_options
        
    async def make_decision(self, 
                         options: List[Dict[str, Any]], 
                         criteria: List[Dict[str, Any]],
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a decision by evaluating options and selecting the best one
        
        Args:
            options: List of options to choose from
            criteria: Criteria for evaluation
            context: Current context information
            
        Returns:
            Dictionary with the selected option and decision metadata
        """
        start_time = time.time()
        
        # Check if we have decision maker from autonomous_decision_system
        if DECISION_SYSTEM_AVAILABLE and self.decision_maker:
            # Convert to DecisionOptions format
            from autonomous_decision_system import DecisionOptions
            decision_options = DecisionOptions()
            
            # Add criteria
            for criterion in criteria:
                decision_options.add_criterion(criterion["id"], criterion.get("weight", 1.0))
                
            # Add options
            for option in options:
                decision_options.add_option(
                    option_id=option["id"],
                    description=option["description"],
                    actions=option.get("actions", [])
                )
                
                # Add evaluations if available
                if "evaluation" in option:
                    for criterion_id, evaluation in option["evaluation"].items():
                        decision_options.evaluate_option(
                            option_id=option["id"],
                            criterion=criterion_id,
                            score=evaluation["score"],
                            explanation=evaluation.get("explanation", "")
                        )
                        
            # Make decision using the decision maker
            decision_result = await self.decision_maker.make_decision(decision_options)
            
            # Update metrics
            self.decisions_made += 1
            
            return decision_result
            
        else:
            # Evaluate options if not already evaluated
            if any("evaluation" not in option for option in options):
                evaluated_options = await self.evaluate_options(options, criteria)
            else:
                evaluated_options = options
                
            # Select the best option
            if evaluated_options:
                best_option = max(evaluated_options, key=lambda x: x.get("overall_score", 0))
            else:
                # Default option if none available
                best_option = {
                    "id": "default",
                    "description": "Default option (no valid options available)",
                    "overall_score": 0.0
                }
                
            decision = {
                "success": True,
                "selected_option": best_option,
                "score": best_option.get("overall_score", 0),
                "decision_time": datetime.now().isoformat(),
                "processing_method": "quantum" if self.use_quantum else "classical",
                "processing_time": time.time() - start_time,
                "all_options": evaluated_options
            }
            
            # Update metrics
            self.decisions_made += 1
            
            return decision
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for the decision system"""
        return {
            "decisions_made": self.decisions_made,
            "quantum_evaluations": self.quantum_evaluations,
            "classical_evaluations": self.classical_evaluations,
            "quantum_device_type": "Azure Quantum IonQ" if self.use_quantum and self.use_azure else "Local simulator" if self.use_quantum else "None",
            "n_qubits": self.n_qubits if self.use_quantum else 0,
            "use_quantum": self.use_quantum,
            "active_goals": len(self.goal_manager.active_goals) if self.goal_manager else 0,
            "completed_goals": len(self.goal_manager.completed_goals) if self.goal_manager else 0
        }


class EnhancedAutonomousAgent:
    """
    Enhanced autonomous agent with true agentic capabilities
    
    This agent combines:
    1. Web interaction capabilities
    2. Computer vision for screen understanding
    3. Goal-driven autonomous behavior
    4. Quantum-enhanced decision making
    5. Memory management and learning
    """
    
    def __init__(self, 
                use_quantum: bool = True, 
                n_qubits: int = 8,
                use_web_automation: bool = True,
                use_vision: bool = True,
                use_claude: bool = True):
        """
        Initialize the enhanced autonomous agent
        
        Args:
            use_quantum: Whether to use quantum computing
            n_qubits: Number of qubits to use
            use_web_automation: Whether to use web automation
            use_vision: Whether to use computer vision
            use_claude: Whether to use Claude instead of GPT
        """
        # Initialize components
        self.memory = AgentMemory()
        self.web_agent = None if not use_web_automation or not WEB_AGENT_AVAILABLE else WebInteractionAgent()
        self.vision_system = None if not use_vision or not VISION_AVAILABLE else VisionSystem()
        self.decision_system = QuantumEnhancedDecisionSystem(n_qubits=n_qubits, use_quantum=use_quantum)
        
        # Initialize AI clients
        self.openai_client = None
        self.anthropic_client = None
        self.use_claude = use_claude and ANTHROPIC_AVAILABLE
        self._initialize_ai_clients()
        
        # Status tracking
        self.initialized = False
        self.current_task = None
        self.tools = {}
        
        # Metrics
        self.tasks_processed = 0
        self.successful_tasks = 0
        self.average_task_time = 0
        self.start_time = time.time()
        
    def _initialize_ai_clients(self):
        """Initialize AI client connections"""
        # Initialize OpenAI client
        if OPENAI_AVAILABLE:
            try:
                self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {str(e)}")
                
        # Initialize Anthropic client if preferred and available
        if self.use_claude and ANTHROPIC_AVAILABLE:
            try:
                self.anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
                logger.info("Anthropic client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {str(e)}")
                self.use_claude = False
                
    async def initialize(self):
        """Initialize all agent components that require async setup"""
        initialization_results = []
        
        # Initialize web agent if available
        if self.web_agent:
            try:
                web_init_result = await self.web_agent.initialize(headless=True)
                initialization_results.append({
                    "component": "web_agent",
                    "success": web_init_result.get("success", False),
                    "message": web_init_result.get("message", "Unknown result")
                })
            except Exception as e:
                logger.error(f"Failed to initialize web agent: {str(e)}")
                initialization_results.append({
                    "component": "web_agent",
                    "success": False,
                    "message": str(e)
                })
                
        # Register available tools
        await self._register_tools()
        
        # Set initialized flag based on results
        self.initialized = all(result["success"] for result in initialization_results) or not initialization_results
        
        logger.info(f"Enhanced autonomous agent initialized (status: {self.initialized})")
        
        return {
            "success": self.initialized,
            "results": initialization_results
        }
        
    async def _register_tools(self):
        """Register all available tools for the agent to use"""
        
        # Web interaction tools
        if self.web_agent:
            self.register_tool(
                "navigate",
                "Navigate to a specified URL",
                self.web_agent.navigate
            )
            
            self.register_tool(
                "search",
                "Perform a web search using a search engine",
                self.web_agent.search
            )
            
            self.register_tool(
                "click_element",
                "Click on an element on the current page",
                self.web_agent.click_element
            )
            
            self.register_tool(
                "fill_form",
                "Fill a form field on the current page",
                self.web_agent.fill_form
            )
            
            self.register_tool(
                "submit_form",
                "Submit a form on the current page",
                self.web_agent.submit_form
            )
            
            self.register_tool(
                "extract_structured_data",
                "Extract structured data from the current page",
                self.web_agent.extract_structured_data
            )
            
            self.register_tool(
                "take_screenshot",
                "Take a screenshot of the current page",
                self.web_agent.take_screenshot
            )
            
        # Vision tools
        if self.vision_system:
            self.register_tool(
                "analyze_screenshot",
                "Analyze a screenshot to detect UI elements",
                self.vision_system.analyze_screenshot
            )
            
    def register_tool(self, name: str, description: str, func):
        """Register a new tool for the agent to use"""
        self.tools[name] = {
            "name": name,
            "description": description,
            "function": func
        }
        
    async def execute_tool(self, tool_name: str, *args, **kwargs) -> Dict[str, Any]:
        """Execute a registered tool by name"""
        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found"
            }
            
        tool = self.tools[tool_name]
        try:
            # Execute the tool function
            result = await tool["function"](*args, **kwargs)
            
            # Record the tool usage in memory
            self.memory.add_interaction(
                role="agent",
                content=f"Used tool: {tool_name}",
                metadata={
                    "tool": tool_name,
                    "args": args,
                    "kwargs": kwargs,
                    "success": result.get("success", False)
                }
            )
            
            return result
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name
            }
            
    async def process_task(self, task: str) -> Dict[str, Any]:
        """
        Process a user task autonomously
        
        This is the main entry point for handling tasks. The agent will:
        1. Analyze the task
        2. Plan necessary actions
        3. Execute actions using tools
        4. Report results
        """
        task_start_time = time.time()
        self.current_task = task
        
        # Record task in memory
        self.memory.add_interaction(
            role="user",
            content=task,
            metadata={"type": "task"}
        )
        
        # Clear working memory for new task
        self.memory.clear_working_memory()
        self.memory.add_to_working_memory("current_task", task)
        
        # Initialize result structure
        result = {
            "task": task,
            "success": False,
            "steps": [],
            "summary": "",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "execution_time": 0
        }
        
        try:
            # Analyze the task
            analysis = await self._analyze_task(task)
            result["analysis"] = analysis
            
            # Plan the task execution
            plan = await self._create_plan(task, analysis)
            result["plan"] = plan
            
            # Execute the plan
            step_results = []
            for step in plan.get("steps", []):
                step_result = await self._execute_step(step)
                step_results.append(step_result)
                result["steps"].append(step_result)
                
                # Check if step failed and stop if critical
                if not step_result.get("success", False) and step.get("critical", False):
                    logger.warning(f"Critical step failed: {step.get('description')}")
                    break
                    
            # Generate summary
            success = all(step.get("success", False) for step in step_results)
            summary = await self._generate_result_summary(task, step_results, success)
            
            # Update result
            result["success"] = success
            result["summary"] = summary
            result["end_time"] = datetime.now().isoformat()
            result["execution_time"] = time.time() - task_start_time
            
            # Update agent metrics
            self.tasks_processed += 1
            if success:
                self.successful_tasks += 1
                
            # Update average task time
            total_task_time = self.average_task_time * (self.tasks_processed - 1) + result["execution_time"]
            self.average_task_time = total_task_time / self.tasks_processed if self.tasks_processed > 0 else 0
            
            # Record result in memory
            self.memory.add_interaction(
                role="agent",
                content=f"Task completed: {summary}",
                metadata={
                    "success": success,
                    "execution_time": result["execution_time"]
                }
            )
            
            # Store important results in long-term memory
            self.memory.add_to_long_term({
                "type": "task_result",
                "task": task,
                "success": success,
                "summary": summary,
                "execution_time": result["execution_time"]
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing task: {str(e)}")
            result["success"] = False
            result["error"] = str(e)
            result["end_time"] = datetime.now().isoformat()
            result["execution_time"] = time.time() - task_start_time
            
            # Update agent metrics
            self.tasks_processed += 1
            
            # Record error in memory
            self.memory.add_interaction(
                role="agent",
                content=f"Error processing task: {str(e)}",
                metadata={"error": str(e)}
            )
            
            return result
            
    async def _analyze_task(self, task: str) -> Dict[str, Any]:
        """
        Analyze a task to determine appropriate action plan
        Uses AI reasoning to understand the task and requirements
        """
        if self.use_claude and self.anthropic_client:
            return await self._analyze_task_with_claude(task)
        elif self.openai_client:
            return await self._analyze_task_with_openai(task)
        else:
            return self._analyze_task_basic(task)
            
    async def _analyze_task_with_claude(self, task: str) -> Dict[str, Any]:
        """Analyze task using Claude"""
        try:
            # Create prompt for Claude
            system_prompt = """You are an expert AI assistant helping analyze tasks for an autonomous agent with:
1. Web browsing capabilities (navigate, search, click, fill forms)
2. Computer vision for screen understanding
3. Quantum computing for enhanced decision-making
4. Goal-oriented autonomous reasoning

Analyze the task to determine:
1. The type of task (web_search, data_analysis, interface_interaction, quantum_computation, or general)
2. Specific sub-goals needed to accomplish the task
3. Required tools and resources
4. Appropriate approaches to solve the task
5. Potential challenges and solutions

Your analysis should be detailed and structured."""

            user_prompt = f"Please analyze this task: {task}\n\nInclude a structured analysis with task type, goals, required tools, and approaches."
            
            # Call Claude API
            response = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Parse response
            analysis_text = response.content[0].text
            
            # Extract structured information using regex
            task_type_match = re.search(r"Task Type:\s*(\w+)", analysis_text, re.IGNORECASE)
            task_type = task_type_match.group(1).lower() if task_type_match else "general"
            
            goals_section = re.search(r"(?:Goals|Sub-goals):(.*?)(?:\n\n|\n#|\Z)", analysis_text, re.DOTALL)
            goals = []
            if goals_section:
                goals_text = goals_section.group(1)
                goal_items = re.findall(r"\n\s*[-*]\s*(.*?)(?=\n\s*[-*]|\Z)", goals_text, re.DOTALL)
                goals = [goal.strip() for goal in goal_items if goal.strip()]
                
            tools_section = re.search(r"(?:Tools|Required Tools):(.*?)(?:\n\n|\n#|\Z)", analysis_text, re.DOTALL)
            tools = []
            if tools_section:
                tools_text = tools_section.group(1)
                tool_items = re.findall(r"\n\s*[-*]\s*(.*?)(?=\n\s*[-*]|\Z)", tools_text, re.DOTALL)
                tools = [tool.strip() for tool in tool_items if tool.strip()]
                
            approaches_section = re.search(r"(?:Approaches|Approach):(.*?)(?:\n\n|\n#|\Z)", analysis_text, re.DOTALL)
            approaches = []
            if approaches_section:
                approaches_text = approaches_section.group(1)
                approach_items = re.findall(r"\n\s*[-*]\s*(.*?)(?=\n\s*[-*]|\Z)", approaches_text, re.DOTALL)
                approaches = [approach.strip() for approach in approach_items if approach.strip()]
                
            challenges_section = re.search(r"(?:Challenges|Potential Challenges):(.*?)(?:\n\n|\n#|\Z)", analysis_text, re.DOTALL)
            challenges = []
            if challenges_section:
                challenges_text = challenges_section.group(1)
                challenge_items = re.findall(r"\n\s*[-*]\s*(.*?)(?=\n\s*[-*]|\Z)", challenges_text, re.DOTALL)
                challenges = [challenge.strip() for challenge in challenge_items if challenge.strip()]
                
            # Assemble structured analysis
            analysis = {
                "task_type": task_type,
                "goals": goals,
                "required_tools": tools,
                "approaches": approaches,
                "challenges": challenges,
                "use_quantum": "quantum" in task_type.lower() or any("quantum" in approach.lower() for approach in approaches),
                "full_analysis": analysis_text,
                "method": "claude"
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing task with Claude: {str(e)}")
            # Fall back to basic analysis
            return self._analyze_task_basic(task)
            
    async def _analyze_task_with_openai(self, task: str) -> Dict[str, Any]:
        """Analyze task using OpenAI"""
        try:
            # Create prompt for OpenAI
            system_prompt = """You are an expert AI assistant helping analyze tasks for an autonomous agent with:
1. Web browsing capabilities (navigate, search, click, fill forms)
2. Computer vision for screen understanding
3. Quantum computing for enhanced decision-making
4. Goal-oriented autonomous reasoning

Analyze the task to determine:
1. The type of task (web_search, data_analysis, interface_interaction, quantum_computation, or general)
2. Specific sub-goals needed to accomplish the task
3. Required tools and resources
4. Appropriate approaches to solve the task
5. Potential challenges and solutions

Your analysis should be detailed and structured."""

            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Please analyze this task: {task}"}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
            # Parse response
            analysis_text = response.choices[0].message.content
            
            # Extract structured information using regex
            task_type_match = re.search(r"Task Type:\s*(\w+)", analysis_text, re.IGNORECASE)
            task_type = task_type_match.group(1).lower() if task_type_match else "general"
            
            goals_section = re.search(r"(?:Goals|Sub-goals):(.*?)(?:\n\n|\n#|\Z)", analysis_text, re.DOTALL)
            goals = []
            if goals_section:
                goals_text = goals_section.group(1)
                goal_items = re.findall(r"\n\s*[-*]\s*(.*?)(?=\n\s*[-*]|\Z)", goals_text, re.DOTALL)
                goals = [goal.strip() for goal in goal_items if goal.strip()]
                
            tools_section = re.search(r"(?:Tools|Required Tools):(.*?)(?:\n\n|\n#|\Z)", analysis_text, re.DOTALL)
            tools = []
            if tools_section:
                tools_text = tools_section.group(1)
                tool_items = re.findall(r"\n\s*[-*]\s*(.*?)(?=\n\s*[-*]|\Z)", tools_text, re.DOTALL)
                tools = [tool.strip() for tool in tool_items if tool.strip()]
                
            approaches_section = re.search(r"(?:Approaches|Approach):(.*?)(?:\n\n|\n#|\Z)", analysis_text, re.DOTALL)
            approaches = []
            if approaches_section:
                approaches_text = approaches_section.group(1)
                approach_items = re.findall(r"\n\s*[-*]\s*(.*?)(?=\n\s*[-*]|\Z)", approaches_text, re.DOTALL)
                approaches = [approach.strip() for approach in approach_items if approach.strip()]
                
            challenges_section = re.search(r"(?:Challenges|Potential Challenges):(.*?)(?:\n\n|\n#|\Z)", analysis_text, re.DOTALL)
            challenges = []
            if challenges_section:
                challenges_text = challenges_section.group(1)
                challenge_items = re.findall(r"\n\s*[-*]\s*(.*?)(?=\n\s*[-*]|\Z)", challenges_text, re.DOTALL)
                challenges = [challenge.strip() for challenge in challenge_items if challenge.strip()]
                
            # Assemble structured analysis
            analysis = {
                "task_type": task_type,
                "goals": goals,
                "required_tools": tools,
                "approaches": approaches,
                "challenges": challenges,
                "use_quantum": "quantum" in task_type.lower() or any("quantum" in approach.lower() for approach in approaches),
                "full_analysis": analysis_text,
                "method": "openai"
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing task with OpenAI: {str(e)}")
            # Fall back to basic analysis
            return self._analyze_task_basic(task)
            
    def _analyze_task_basic(self, task: str) -> Dict[str, Any]:
        """Basic task analysis without AI models"""
        # Simple keyword-based analysis
        task_lower = task.lower()
        
        # Determine task type
        if any(keyword in task_lower for keyword in ["search", "find", "look up", "google"]):
            task_type = "web_search"
        elif any(keyword in task_lower for keyword in ["analyze", "data", "statistics", "metrics"]):
            task_type = "data_analysis"
        elif any(keyword in task_lower for keyword in ["click", "navigate", "fill", "form", "button"]):
            task_type = "interface_interaction"
        elif any(keyword in task_lower for keyword in ["quantum", "qubits", "superposition"]):
            task_type = "quantum_computation"
        else:
            task_type = "general"
            
        # Simple goals extraction
        goals = [f"Complete the task: {task}"]
        
        # Determine required tools
        tools = []
        if task_type == "web_search":
            tools.append("search")
            tools.append("navigate")
        elif task_type == "interface_interaction":
            tools.append("navigate")
            tools.append("click_element")
            tools.append("fill_form")
            tools.append("submit_form")
        elif task_type == "quantum_computation":
            tools.append("quantum_processing")
            
        # Simple approaches
        approaches = [f"Direct approach to {task_type} task"]
        
        # Simple challenges
        challenges = ["Handling unexpected content or errors"]
        
        return {
            "task_type": task_type,
            "goals": goals,
            "required_tools": tools,
            "approaches": approaches,
            "challenges": challenges,
            "use_quantum": "quantum" in task_type.lower(),
            "method": "basic"
        }
        
    async def _create_plan(self, task: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create a plan for executing the task based on analysis"""
        if self.use_claude and self.anthropic_client:
            return await self._create_plan_with_claude(task, analysis)
        elif self.openai_client:
            return await self._create_plan_with_openai(task, analysis)
        else:
            return self._create_plan_basic(task, analysis)
            
    async def _create_plan_with_claude(self, task: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create a plan using Claude"""
        try:
            # Prepare tools information
            tools_info = "\n".join([f"- {name}: {tool['description']}" 
                                for name, tool in self.tools.items()])
            
            system_prompt = f"""You are an expert AI planner helping create execution plans for an autonomous agent.

Available tools:
{tools_info}

Your task is to create a detailed, step-by-step plan for executing the task.
Each step should include:
1. Step description
2. Tool to use (if applicable)
3. Parameters for the tool
4. Expected outcome
5. Whether the step is critical (failure stops execution)

Make the plan concrete, specific, and executable by an autonomous agent."""

            # Create user prompt with task and analysis
            user_prompt = f"""Task: {task}

Analysis:
- Type: {analysis.get('task_type', 'general')}
- Goals: {', '.join(analysis.get('goals', []))}
- Required tools: {', '.join(analysis.get('required_tools', []))}
- Approaches: {', '.join(analysis.get('approaches', []))}

Please create a detailed execution plan with specific steps."""
            
            # Call Claude API
            response = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Parse response
            plan_text = response.content[0].text
            
            # Extract steps using regex
            steps = []
            step_pattern = r"(?:Step|)\s*(\d+):\s*(.*?)(?=(?:Step|)\s*\d+:|$)"
            step_matches = re.finditer(step_pattern, plan_text, re.DOTALL)
            
            for match in step_matches:
                step_number = int(match.group(1))
                step_content = match.group(2).strip()
                
                # Extract tool and parameters
                tool_match = re.search(r"Tool:\s*([\w_]+)", step_content, re.IGNORECASE)
                tool = tool_match.group(1) if tool_match else None
                
                params_match = re.search(r"Parameters:\s*(.*?)(?=Expected|$)", step_content, re.DOTALL | re.IGNORECASE)
                params_text = params_match.group(1).strip() if params_match else ""
                
                # Parse parameters
                params = {}
                if params_text:
                    param_pattern = r"[-*]?\s*(\w+):\s*(.*?)(?=[-*]?\s*\w+:|$)"
                    param_matches = re.finditer(param_pattern, params_text, re.DOTALL)
                    for param_match in param_matches:
                        param_name = param_match.group(1).strip()
                        param_value = param_match.group(2).strip()
                        params[param_name] = param_value
                        
                # Extract expected outcome
                outcome_match = re.search(r"Expected outcome:\s*(.*?)(?=Critical:|$)", step_content, re.DOTALL | re.IGNORECASE)
                expected_outcome = outcome_match.group(1).strip() if outcome_match else ""
                
                # Extract critical flag
                critical_match = re.search(r"Critical:\s*(yes|no|true|false)", step_content, re.IGNORECASE)
                critical = critical_match.group(1).lower() in ["yes", "true"] if critical_match else False
                
                # Extract description (first line or everything before Tool:)
                desc_match = re.search(r"^(.*?)(?=Tool:|$)", step_content, re.DOTALL)
                description = desc_match.group(1).strip() if desc_match else step_content
                
                steps.append({
                    "step_number": step_number,
                    "description": description,
                    "tool": tool,
                    "parameters": params,
                    "expected_outcome": expected_outcome,
                    "critical": critical
                })
                
            # Sort steps by number
            steps.sort(key=lambda x: x["step_number"])
            
            return {
                "steps": steps,
                "original_plan": plan_text,
                "method": "claude"
            }
            
        except Exception as e:
            logger.error(f"Error creating plan with Claude: {str(e)}")
            # Fall back to basic plan
            return self._create_plan_basic(task, analysis)
            
    async def _create_plan_with_openai(self, task: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create a plan using OpenAI"""
        try:
            # Prepare tools information
            tools_info = "\n".join([f"- {name}: {tool['description']}" 
                                for name, tool in self.tools.items()])
            
            system_prompt = f"""You are an expert AI planner helping create execution plans for an autonomous agent.

Available tools:
{tools_info}

Your task is to create a detailed, step-by-step plan for executing the task.
Each step should include:
1. Step description
2. Tool to use (if applicable)
3. Parameters for the tool
4. Expected outcome
5. Whether the step is critical (failure stops execution)

Make the plan concrete, specific, and executable by an autonomous agent."""
            
            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"""Task: {task}

Analysis:
- Type: {analysis.get('task_type', 'general')}
- Goals: {', '.join(analysis.get('goals', []))}
- Required tools: {', '.join(analysis.get('required_tools', []))}
- Approaches: {', '.join(analysis.get('approaches', []))}

Please create a detailed execution plan with specific steps."""}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
            # Parse response
            plan_text = response.choices[0].message.content
            
            # Extract steps using regex
            steps = []
            step_pattern = r"(?:Step|)\s*(\d+):\s*(.*?)(?=(?:Step|)\s*\d+:|$)"
            step_matches = re.finditer(step_pattern, plan_text, re.DOTALL)
            
            for match in step_matches:
                step_number = int(match.group(1))
                step_content = match.group(2).strip()
                
                # Extract tool and parameters
                tool_match = re.search(r"Tool:\s*([\w_]+)", step_content, re.IGNORECASE)
                tool = tool_match.group(1) if tool_match else None
                
                params_match = re.search(r"Parameters:\s*(.*?)(?=Expected|$)", step_content, re.DOTALL | re.IGNORECASE)
                params_text = params_match.group(1).strip() if params_match else ""
                
                # Parse parameters
                params = {}
                if params_text:
                    param_pattern = r"[-*]?\s*(\w+):\s*(.*?)(?=[-*]?\s*\w+:|$)"
                    param_matches = re.finditer(param_pattern, params_text, re.DOTALL)
                    for param_match in param_matches:
                        param_name = param_match.group(1).strip()
                        param_value = param_match.group(2).strip()
                        params[param_name] = param_value
                        
                # Extract expected outcome
                outcome_match = re.search(r"Expected outcome:\s*(.*?)(?=Critical:|$)", step_content, re.DOTALL | re.IGNORECASE)
                expected_outcome = outcome_match.group(1).strip() if outcome_match else ""
                
                # Extract critical flag
                critical_match = re.search(r"Critical:\s*(yes|no|true|false)", step_content, re.IGNORECASE)
                critical = critical_match.group(1).lower() in ["yes", "true"] if critical_match else False
                
                # Extract description (first line or everything before Tool:)
                desc_match = re.search(r"^(.*?)(?=Tool:|$)", step_content, re.DOTALL)
                description = desc_match.group(1).strip() if desc_match else step_content
                
                steps.append({
                    "step_number": step_number,
                    "description": description,
                    "tool": tool,
                    "parameters": params,
                    "expected_outcome": expected_outcome,
                    "critical": critical
                })
                
            # Sort steps by number
            steps.sort(key=lambda x: x["step_number"])
            
            return {
                "steps": steps,
                "original_plan": plan_text,
                "method": "openai"
            }
            
        except Exception as e:
            logger.error(f"Error creating plan with OpenAI: {str(e)}")
            # Fall back to basic plan
            return self._create_plan_basic(task, analysis)
            
    def _create_plan_basic(self, task: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create a basic plan without AI models"""
        task_type = analysis.get("task_type", "general")
        steps = []
        
        if task_type == "web_search":
            # Basic web search plan
            search_terms = task.replace("search", "").replace("for", "").strip()
            steps = [
                {
                    "step_number": 1,
                    "description": f"Search for information about: {search_terms}",
                    "tool": "search",
                    "parameters": {
                        "query": search_terms,
                        "search_engine": "google"
                    },
                    "expected_outcome": "Search results with relevant information",
                    "critical": True
                },
                {
                    "step_number": 2,
                    "description": "Navigate to the most relevant result",
                    "tool": "navigate",
                    "parameters": {
                        "url": "{result_from_step_1}"
                    },
                    "expected_outcome": "Web page with detailed information",
                    "critical": False
                },
                {
                    "step_number": 3,
                    "description": "Extract structured data from the page",
                    "tool": "extract_structured_data",
                    "parameters": {},
                    "expected_outcome": "Structured information from the page",
                    "critical": False
                }
            ]
        elif task_type == "interface_interaction":
            steps = [
                {
                    "step_number": 1,
                    "description": "Navigate to the target website",
                    "tool": "navigate",
                    "parameters": {
                        "url": "https://example.com"  # This would be replaced with actual URL
                    },
                    "expected_outcome": "Web page loaded successfully",
                    "critical": True
                },
                {
                    "step_number": 2,
                    "description": "Take screenshot to analyze the interface",
                    "tool": "take_screenshot",
                    "parameters": {},
                    "expected_outcome": "Screenshot of the current page",
                    "critical": False
                },
                {
                    "step_number": 3,
                    "description": "Analyze screenshot to detect UI elements",
                    "tool": "analyze_screenshot",
                    "parameters": {
                        "screenshot_base64": "{result_from_step_2.screenshot}"
                    },
                    "expected_outcome": "Detected UI elements on the page",
                    "critical": False
                },
                {
                    "step_number": 4,
                    "description": "Click on relevant element",
                    "tool": "click_element",
                    "parameters": {
                        "selector": "button"  # This would be replaced with actual selector
                    },
                    "expected_outcome": "Interaction with the selected element",
                    "critical": False
                }
            ]
        else:
            # Generic plan for any task
            steps = [
                {
                    "step_number": 1,
                    "description": f"Analyze the task: {task}",
                    "tool": None,
                    "parameters": {},
                    "expected_outcome": "Understanding of the task requirements",
                    "critical": True
                },
                {
                    "step_number": 2,
                    "description": "Gather relevant information",
                    "tool": "search",
                    "parameters": {
                        "query": task,
                        "search_engine": "google"
                    },
                    "expected_outcome": "Relevant information found",
                    "critical": False
                },
                {
                    "step_number": 3,
                    "description": "Process the information",
                    "tool": None,
                    "parameters": {},
                    "expected_outcome": "Information processed for the task",
                    "critical": False
                }
            ]
            
        return {
            "steps": steps,
            "method": "basic"
        }
        
    async def _execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step in the plan"""
        step_number = step.get("step_number", 0)
        description = step.get("description", "")
        tool_name = step.get("tool")
        parameters = step.get("parameters", {})
        expected_outcome = step.get("expected_outcome", "")
        
        logger.info(f"Executing step {step_number}: {description}")
        
        start_time = time.time()
        
        # Record step execution in memory
        self.memory.add_interaction(
            role="agent",
            content=f"Executing step {step_number}: {description}",
            metadata={"step": step}
        )
        
        # Initialize result structure
        result = {
            "step_number": step_number,
            "description": description,
            "tool": tool_name,
            "parameters": parameters,
            "expected_outcome": expected_outcome,
            "success": False,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "execution_time": 0
        }
        
        try:
            # Process parameter references to previous steps
            processed_parameters = {}
            for key, value in parameters.items():
                if isinstance(value, str) and "{result_from_step_" in value:
                    # Extract step number from reference
                    ref_match = re.search(r"{result_from_step_(\d+)(?:\.(\w+))?}", value)
                    if ref_match:
                        ref_step = int(ref_match.group(1))
                        ref_key = ref_match.group(2) if ref_match.group(2) else None
                        
                        # Get value from working memory
                        step_result = self.memory.get_from_working_memory(f"step_{ref_step}_result")
                        if step_result:
                            if ref_key:
                                # Reference to specific result key
                                ref_value = step_result.get(ref_key)
                                processed_parameters[key] = ref_value
                            else:
                                # Reference to entire result
                                processed_parameters[key] = step_result
                        else:
                            processed_parameters[key] = value  # Keep original if not found
                else:
                    processed_parameters[key] = value
                    
            # Execute step based on whether it uses a tool
            if tool_name and tool_name in self.tools:
                # Execute using the specified tool
                tool_result = await self.execute_tool(tool_name, **processed_parameters)
                
                # Update result with tool execution result
                result.update({
                    "success": tool_result.get("success", False),
                    "output": tool_result,
                    "error": tool_result.get("error") if not tool_result.get("success", False) else None
                })
            else:
                # No tool - this is a logical/thinking step
                result.update({
                    "success": True,
                    "output": {"message": f"Logical step completed: {description}"}
                })
                
            # Store step result in working memory
            self.memory.add_to_working_memory(f"step_{step_number}_result", result)
            
            # Update execution time and end time
            result["end_time"] = datetime.now().isoformat()
            result["execution_time"] = time.time() - start_time
            
            # Record step completion in memory
            self.memory.add_interaction(
                role="agent",
                content=f"Step {step_number} {'completed successfully' if result['success'] else 'failed'}",
                metadata={"step_result": result}
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing step {step_number}: {str(e)}")
            
            # Update result with error information
            result.update({
                "success": False,
                "error": str(e),
                "end_time": datetime.now().isoformat(),
                "execution_time": time.time() - start_time
            })
            
            # Store step result in working memory
            self.memory.add_to_working_memory(f"step_{step_number}_result", result)
            
            # Record error in memory
            self.memory.add_interaction(
                role="agent",
                content=f"Error in step {step_number}: {str(e)}",
                metadata={"error": str(e)}
            )
            
            return result
            
    async def _generate_result_summary(self, task: str, step_results: List[Dict[str, Any]], 
                                     success: bool) -> str:
        """Generate a summary of the task results"""
        if self.use_claude and self.anthropic_client:
            return await self._generate_summary_with_claude(task, step_results, success)
        elif self.openai_client:
            return await self._generate_summary_with_openai(task, step_results, success)
        else:
            return self._generate_summary_basic(task, step_results, success)
            
    async def _generate_summary_with_claude(self, task: str, step_results: List[Dict[str, Any]], success: bool) -> str:
        """Generate a summary using Claude"""
        try:
            # Create a summary of the steps and their results
            steps_summary = ""
            for step in step_results:
                step_number = step.get("step_number", 0)
                description = step.get("description", "")
                success_status = "✓" if step.get("success", False) else "✗"
                error = step.get("error", "")
                
                steps_summary += f"Step {step_number}: {description} [{success_status}]\n"
                if error:
                    steps_summary += f"  Error: {error}\n"
                    
            # Create prompt for Claude
            system_prompt = """You are an assistant that summarizes task execution results clearly and concisely.
Focus on what was accomplished, what failed, and the overall outcome.
Keep the summary brief but informative, highlighting the most important aspects of the task execution."""

            user_prompt = f"""Task: {task}

Overall status: {'Successful' if success else 'Failed'}

Steps summary:
{steps_summary}

Please provide a concise summary of the task execution results, focusing on what was accomplished, what failed (if anything), and the overall outcome."""
            
            # Call Claude API
            response = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Get summary text
            summary = response.content[0].text.strip()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary with Claude: {str(e)}")
            # Fall back to basic summary
            return self._generate_summary_basic(task, step_results, success)
            
    async def _generate_summary_with_openai(self, task: str, step_results: List[Dict[str, Any]], success: bool) -> str:
        """Generate a summary using OpenAI"""
        try:
            # Create a summary of the steps and their results
            steps_summary = ""
            for step in step_results:
                step_number = step.get("step_number", 0)
                description = step.get("description", "")
                success_status = "✓" if step.get("success", False) else "✗"
                error = step.get("error", "")
                
                steps_summary += f"Step {step_number}: {description} [{success_status}]\n"
                if error:
                    steps_summary += f"  Error: {error}\n"
                    
            # Create prompt for OpenAI
            system_prompt = """You are an assistant that summarizes task execution results clearly and concisely.
Focus on what was accomplished, what failed, and the overall outcome.
Keep the summary brief but informative, highlighting the most important aspects of the task execution."""
            
            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"""Task: {task}

Overall status: {'Successful' if success else 'Failed'}

Steps summary:
{steps_summary}

Please provide a concise summary of the task execution results, focusing on what was accomplished, what failed (if anything), and the overall outcome."""}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            # Get summary text
            summary = response.choices[0].message.content.strip()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary with OpenAI: {str(e)}")
            # Fall back to basic summary
            return self._generate_summary_basic(task, step_results, success)
            
    def _generate_summary_basic(self, task: str, step_results: List[Dict[str, Any]], success: bool) -> str:
        """Generate a basic summary without AI models"""
        # Count successful steps
        total_steps = len(step_results)
        successful_steps = sum(1 for step in step_results if step.get("success", False))
        
        # List any errors
        errors = [f"Step {step.get('step_number')}: {step.get('error')}" 
                for step in step_results if not step.get("success", False) and step.get("error")]
        
        # Generate summary text
        if success:
            summary = f"Successfully completed task: {task}. "
            summary += f"Completed {successful_steps}/{total_steps} steps successfully."
        else:
            summary = f"Failed to complete task: {task}. "
            summary += f"Completed {successful_steps}/{total_steps} steps successfully."
            
            if errors:
                summary += f" Encountered {len(errors)} errors:"
                for error in errors[:3]:  # Show first 3 errors
                    summary += f"\n- {error}"
                if len(errors) > 3:
                    summary += f"\n- And {len(errors) - 3} more errors..."
                    
        return summary
        
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the agent"""
        uptime = time.time() - self.start_time
        
        # Gather component status information
        components_status = {
            "web_agent": self.web_agent is not None and getattr(self.web_agent, "initialized", False),
            "vision_system": self.vision_system is not None,
            "decision_system": True,  # Always available
            "openai_client": self.openai_client is not None,
            "anthropic_client": self.anthropic_client is not None,
            "quantum_available": QUANTUM_AVAILABLE,
            "azure_quantum_available": AZURE_QUANTUM_AVAILABLE
        }
        
        # Gather memory statistics
        memory_stats = {
            "short_term_items": len(self.memory.short_term),
            "working_memory_items": len(self.memory.working_memory),
            "long_term_items": len(self.memory.long_term),
            "web_history_items": len(self.memory.web_history),
            "screen_observations": len(self.memory.screen_observations)
        }
        
        # Gather quantum capabilities
        quantum_capabilities = None
        if QUANTUM_AVAILABLE:
            quantum_capabilities = {
                "provider": "Azure Quantum" if AZURE_QUANTUM_AVAILABLE else "Local Simulation",
                "n_qubits": self.decision_system.n_qubits if hasattr(self.decision_system, "n_qubits") else 0,
                "device_type": "quantum" if QUANTUM_AVAILABLE else "classical"
            }
            
        # Gather performance metrics
        performance = {
            "tasks_processed": self.tasks_processed,
            "successful_tasks": self.successful_tasks,
            "success_rate": self.successful_tasks / self.tasks_processed if self.tasks_processed > 0 else 0,
            "average_task_time": self.average_task_time,
            "uptime": uptime
        }
        
        # Web metrics if available
        web_metrics = None
        if self.web_agent:
            web_metrics = self.web_agent.get_metrics()
            
        return {
            "initialized": self.initialized,
            "current_task": self.current_task,
            "components": components_status,
            "memory": memory_stats,
            "quantum": quantum_capabilities,
            "performance": performance,
            "web_metrics": web_metrics,
            "available_tools": list(self.tools.keys()),
            "timestamp": datetime.now().isoformat()
        }
        
    async def close(self):
        """Close all resources used by the agent"""
        # Close web agent if available
        if self.web_agent:
            await self.web_agent.close()
            
        logger.info("Enhanced autonomous agent closed")


# Helper function for running async functions in Streamlit
def run_async(func):
    """Run an async function"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(func)
    loop.close()
    return result