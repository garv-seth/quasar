"""Quantum-Accelerated AI Agent (Q3A) - Core Agent Architecture"""

import asyncio
import logging
import time
import os
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
import json

import pennylane as qml
import numpy as np
from openai import AsyncOpenAI

from quantum_agent_framework.quantum.optimizer import QuantumOptimizer
from quantum_agent_framework.integration.hybrid_computation import HybridComputation
from quantum_agent_framework.agents.web_agent import QuantumWebAgent
from quantum_agent_framework.quantum.factorization_manager import FactorizationManager


@dataclass
class AgentMemory:
    """Agent's memory store for maintaining context across tasks."""
    short_term: List[Dict[str, Any]] = None
    working_memory: Dict[str, Any] = None
    long_term: List[Dict[str, Any]] = None
    max_short_term: int = 10  # Maximum number of recent interactions to store
    
    def __post_init__(self):
        """Initialize memory stores if not provided."""
        if self.short_term is None:
            self.short_term = []
        if self.working_memory is None:
            self.working_memory = {}
        if self.long_term is None:
            self.long_term = []
    
    def add_interaction(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add an interaction to short-term memory."""
        entry = {"role": role, "content": content, "timestamp": time.time()}
        if metadata:
            entry["metadata"] = metadata
        self.short_term.append(entry)
        if len(self.short_term) > self.max_short_term:
            self.short_term.pop(0)  # Remove oldest entry
    
    def add_working_memory(self, key: str, value: Any):
        """Add to working memory."""
        self.working_memory[key] = value
    
    def get_working_memory(self, key: str) -> Any:
        """Retrieve from working memory."""
        return self.working_memory.get(key)
    
    def add_to_long_term(self, data: Dict[str, Any]):
        """Add important information to long-term memory."""
        data["timestamp"] = time.time()
        self.long_term.append(data)
    
    def get_formatted_memory(self) -> List[Dict[str, str]]:
        """Format memory for use in prompts."""
        formatted = []
        for item in self.short_term:
            formatted.append({"role": item["role"], "content": item["content"]})
        return formatted


@dataclass
class Tool:
    """Tool definition for agent use."""
    name: str
    description: str
    function: Callable
    required_params: List[str]
    optional_params: Dict[str, Any] = None
    uses_quantum: bool = False
    quantum_advantage: Optional[str] = None
    
    def __post_init__(self):
        if self.optional_params is None:
            self.optional_params = {}


class Q3AAgent:
    """Quantum-Accelerated AI Agent (Q3A) - Core Agent Implementation"""
    
    def __init__(self, n_qubits: int = 8, use_quantum: bool = True, use_azure: bool = True):
        """Initialize the Q3A agent with quantum capabilities."""
        # Core components
        self.memory = AgentMemory()
        self.n_qubits = n_qubits
        self.use_quantum = use_quantum
        self.use_azure = use_azure
        
        # Set up OpenAI client
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Quantum components for computation acceleration
        self.quantum_optimizer = None
        self.hybrid_computer = None
        self.web_agent = None
        self.factorization_manager = None
        
        # Initialize performance metrics
        self.performance_metrics = {
            "tasks_completed": 0,
            "quantum_accelerated_tasks": 0,
            "classical_tasks": 0,
            "avg_quantum_speedup": 0,
            "total_execution_time": 0,
        }
        
        # Available tools
        self.tools = []
        
        # Setup quantum resources if enabled
        if self.use_quantum:
            self._initialize_quantum_components()
            
        # Setup available tools
        self._register_tools()
        
        logging.info(f"Q3A Agent initialized with {n_qubits} qubits, quantum {'enabled' if use_quantum else 'disabled'}")
        
    async def _initialize_quantum_components(self):
        """Initialize quantum computing components."""
        try:
            logging.info("Initializing quantum components...")
            
            # Initialize quantum optimizer
            self.quantum_optimizer = QuantumOptimizer(
                n_qubits=self.n_qubits,
                use_azure=self.use_azure
            )
            
            # Initialize hybrid computation manager
            self.hybrid_computer = HybridComputation(
                n_qubits=self.n_qubits,
                use_quantum=self.use_quantum,
                use_azure=self.use_azure
            )
            
            # Initialize quantum web agent
            self.web_agent = QuantumWebAgent(
                n_qubits=self.n_qubits,
                use_quantum=self.use_quantum
            )
            await self.web_agent.initialize()
            
            # Initialize factorization manager
            self.factorization_manager = FactorizationManager(
                quantum_optimizer=self.quantum_optimizer
            )
            
            logging.info(f"Successfully initialized quantum components with {self.n_qubits} qubits")
            return True
            
        except Exception as e:
            logging.error(f"Error initializing quantum components: {str(e)}")
            self.use_quantum = False
            logging.warning("Falling back to classical computation")
            return False
    
    def _register_tools(self):
        """Register available tools for the agent to use."""
        self.tools = [
            Tool(
                name="web_search",
                description="Search the web for information with quantum-accelerated relevance ranking",
                function=self._web_search_tool,
                required_params=["query"],
                optional_params={"max_results": 5},
                uses_quantum=True,
                quantum_advantage="Provides quadratic speedup for unstructured search using quantum computing"
            ),
            Tool(
                name="factorize",
                description="Factorize large numbers with quantum acceleration using Shor's algorithm",
                function=self._factorize_tool,
                required_params=["number"],
                uses_quantum=True,
                quantum_advantage="Provides exponential speedup for integer factorization"
            ),
            Tool(
                name="optimize_resources",
                description="Optimize resource allocation with quantum algorithms",
                function=self._optimize_tool,
                required_params=["resources", "constraints"],
                uses_quantum=True,
                quantum_advantage="Provides quadratic speedup for optimization problems"
            ),
            Tool(
                name="analyze_data",
                description="Analyze data with quantum pattern recognition techniques",
                function=self._analyze_data_tool,
                required_params=["data", "objective"],
                uses_quantum=True,
                quantum_advantage="Enhances pattern recognition with quantum amplitude amplification"
            ),
            Tool(
                name="execute_task",
                description="Plan and execute a complex task using a combination of tools",
                function=self._execute_task_tool,
                required_params=["task_description"],
                optional_params={"use_tools": []},
                uses_quantum=True
            )
        ]
    
    async def _web_search_tool(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search the web with quantum-enhanced algorithms."""
        if not self.web_agent:
            # Initialize if not already done
            self.web_agent = QuantumWebAgent(n_qubits=self.n_qubits, use_quantum=self.use_quantum)
            await self.web_agent.initialize()
        
        start_time = time.time()
        try:
            # First, enhance the search query with intelligence
            enhanced_query = await self._enhance_search_query(query)
            
            # Perform the actual search using quantum-enhanced relevance
            results = await self.web_agent.search(enhanced_query, max_results=max_results)
            
            # Extract relevant information from results and summarize
            processed_results = {
                "query": query,
                "enhanced_query": enhanced_query,
                "results": results.get("results", []),
                "summary": await self._summarize_search_results(results),
                "execution_time": time.time() - start_time,
                "quantum_advantage": results.get("quantum_advantage", False)
            }
            
            if self.use_quantum:
                self.performance_metrics["quantum_accelerated_tasks"] += 1
            else:
                self.performance_metrics["classical_tasks"] += 1
                
            self.performance_metrics["tasks_completed"] += 1
            self.performance_metrics["total_execution_time"] += processed_results["execution_time"]
            
            # Add to agent memory
            self.memory.add_working_memory(f"search_{int(time.time())}", processed_results)
            
            return processed_results
        
        except Exception as e:
            logging.error(f"Error in web search: {str(e)}")
            return {"error": str(e), "query": query}
    
    async def _factorize_tool(self, number: int) -> Dict[str, Any]:
        """Factorize a number using quantum acceleration when beneficial."""
        if not self.factorization_manager:
            if self.quantum_optimizer:
                self.factorization_manager = FactorizationManager(quantum_optimizer=self.quantum_optimizer)
            else:
                self.factorization_manager = FactorizationManager()
        
        start_time = time.time()
        try:
            result = await self.factorization_manager.factorize(number)
            
            processed_result = {
                "number": number,
                "factors": result.factors,
                "method_used": result.method_used,
                "computation_time": result.computation_time,
                "success": result.success,
                "details": result.details,
                "execution_time": time.time() - start_time
            }
            
            if "quantum" in result.method_used:
                self.performance_metrics["quantum_accelerated_tasks"] += 1
            else:
                self.performance_metrics["classical_tasks"] += 1
                
            self.performance_metrics["tasks_completed"] += 1
            self.performance_metrics["total_execution_time"] += processed_result["execution_time"]
            
            # Add to agent memory
            self.memory.add_working_memory(f"factorize_{number}", processed_result)
            
            return processed_result
        
        except Exception as e:
            logging.error(f"Error in factorization: {str(e)}")
            return {"error": str(e), "number": number}
    
    async def _optimize_tool(self, resources: Dict[str, Any], constraints: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize resource allocation with quantum algorithms."""
        if not self.quantum_optimizer:
            self.quantum_optimizer = QuantumOptimizer(n_qubits=self.n_qubits, use_azure=self.use_azure)
        
        start_time = time.time()
        try:
            # Format the resources for optimization
            optimization_input = {
                "resources": resources,
                "constraints": constraints
            }
            
            # Perform optimization
            result = self.quantum_optimizer.optimize_resource_allocation(optimization_input)
            
            processed_result = {
                "original_resources": resources,
                "original_constraints": constraints,
                "optimized_allocation": result.get("allocation", []),
                "objective_value": result.get("objective_value", 0),
                "computation_time": result.get("computation_time", 0),
                "method_used": result.get("method_used", "unknown"),
                "quantum_advantage": result.get("quantum_advantage", "N/A"),
                "execution_time": time.time() - start_time
            }
            
            if self.use_quantum:
                self.performance_metrics["quantum_accelerated_tasks"] += 1
            else:
                self.performance_metrics["classical_tasks"] += 1
                
            self.performance_metrics["tasks_completed"] += 1
            self.performance_metrics["total_execution_time"] += processed_result["execution_time"]
            
            # Add to agent memory
            self.memory.add_working_memory(f"optimize_{int(time.time())}", processed_result)
            
            return processed_result
        
        except Exception as e:
            logging.error(f"Error in optimization: {str(e)}")
            return {"error": str(e), "resources": resources}
    
    async def _analyze_data_tool(self, data: Dict[str, Any], objective: str) -> Dict[str, Any]:
        """Analyze data with quantum pattern recognition techniques."""
        if not self.hybrid_computer:
            self.hybrid_computer = HybridComputation(
                n_qubits=self.n_qubits,
                use_quantum=self.use_quantum,
                use_azure=self.use_azure
            )
        
        start_time = time.time()
        try:
            # Format task for hybrid computation
            task = f"Analyze data for {objective}. Data: {json.dumps(data)}"
            
            # Perform hybrid analysis
            result = await self.hybrid_computer.process_task(task)
            
            processed_result = {
                "data": data,
                "objective": objective,
                "analysis": result,
                "execution_time": time.time() - start_time
            }
            
            if self.use_quantum:
                self.performance_metrics["quantum_accelerated_tasks"] += 1
            else:
                self.performance_metrics["classical_tasks"] += 1
                
            self.performance_metrics["tasks_completed"] += 1
            self.performance_metrics["total_execution_time"] += processed_result["execution_time"]
            
            # Add to agent memory
            self.memory.add_working_memory(f"analyze_{int(time.time())}", processed_result)
            
            return processed_result
        
        except Exception as e:
            logging.error(f"Error in data analysis: {str(e)}")
            return {"error": str(e), "objective": objective}
    
    async def _execute_task_tool(self, task_description: str, use_tools: List[str] = None) -> Dict[str, Any]:
        """Plan and execute a complex task using a combination of tools."""
        start_time = time.time()
        try:
            # Get available tools
            available_tools = [t.name for t in self.tools if use_tools is None or t.name in use_tools]
            
            # Create task execution plan using AI
            plan = await self._create_task_plan(task_description, available_tools)
            
            # Execute each step in the plan
            results = []
            for step in plan["steps"]:
                step_result = await self._execute_step(step)
                results.append({
                    "step": step,
                    "result": step_result
                })
                
                # Update plan with results for subsequent steps
                step["result"] = step_result
            
            # Analyze results and create summary
            summary = await self._create_task_summary(task_description, results)
            
            processed_result = {
                "task": task_description,
                "plan": plan,
                "steps_executed": len(results),
                "results": results,
                "summary": summary,
                "execution_time": time.time() - start_time
            }
            
            self.performance_metrics["tasks_completed"] += 1
            self.performance_metrics["total_execution_time"] += processed_result["execution_time"]
            
            # Add to agent memory
            self.memory.add_working_memory(f"task_{int(time.time())}", processed_result)
            self.memory.add_to_long_term({
                "type": "task_execution",
                "task": task_description,
                "summary": summary
            })
            
            return processed_result
        
        except Exception as e:
            logging.error(f"Error executing task: {str(e)}")
            return {"error": str(e), "task": task_description}
    
    async def _create_task_plan(self, task: str, available_tools: List[str]) -> Dict[str, Any]:
        """Create a plan for executing a complex task."""
        # Prepare prompt for planning
        tools_descriptions = []
        for tool in self.tools:
            if tool.name in available_tools:
                tools_descriptions.append(f"- {tool.name}: {tool.description}")
        
        planning_prompt = f"""
        Task: {task}
        
        Available Tools:
        {chr(10).join(tools_descriptions)}
        
        Create a detailed step-by-step plan to execute this task efficiently.
        For each step, specify:
        1. The tool to use
        2. The parameters to pass to the tool
        3. How this step contributes to the overall task
        
        Format your response as valid JSON with the following structure:
        {{
            "plan_name": "Short descriptive name for the plan",
            "steps": [
                {{
                    "step_number": 1,
                    "tool": "tool_name",
                    "parameters": {{ ... }},
                    "purpose": "Description of what this step accomplishes"
                }},
                ...
            ]
        }}
        """
        
        # Get planning response from AI
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": planning_prompt}],
            response_format={"type": "json_object"}
        )
        
        # Parse plan from JSON
        plan_text = response.choices[0].message.content
        plan = json.loads(plan_text)
        
        return plan
    
    async def _execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step in the task plan."""
        tool_name = step.get("tool")
        parameters = step.get("parameters", {})
        
        # Find the tool
        tool = next((t for t in self.tools if t.name == tool_name), None)
        
        if not tool:
            return {"error": f"Tool '{tool_name}' not found"}
        
        # Validate required parameters
        missing_params = [p for p in tool.required_params if p not in parameters]
        if missing_params:
            return {"error": f"Missing required parameters: {', '.join(missing_params)}"}
        
        # Execute the tool
        try:
            result = await tool.function(**parameters)
            return result
        except Exception as e:
            logging.error(f"Error executing {tool_name}: {str(e)}")
            return {"error": str(e)}
    
    async def _create_task_summary(self, task: str, results: List[Dict[str, Any]]) -> str:
        """Create a summary of the task execution results."""
        # Prepare results for summary
        results_text = ""
        for i, result_item in enumerate(results):
            step = result_item["step"]
            result = result_item["result"]
            
            success = "error" not in result
            status = "✅ Successful" if success else "❌ Failed"
            
            results_text += f"\nStep {i+1}: {step['tool']} - {status}\n"
            if not success:
                results_text += f"Error: {result['error']}\n"
            else:
                # Summarize successful results
                if step['tool'] == "web_search":
                    results_text += f"Found {len(result.get('results', []))} results for '{result.get('query', '')}'.\n"
                elif step['tool'] == "factorize":
                    results_text += f"Factorized {result.get('number', '')} using {result.get('method_used', '')}.\n"
                elif step['tool'] == "optimize_resources":
                    results_text += f"Optimized allocation with objective value {result.get('objective_value', '')}.\n"
                elif step['tool'] == "analyze_data":
                    results_text += f"Analyzed data for {result.get('objective', '')}.\n"
        
        # Create prompt for summary
        summary_prompt = f"""
        Task: {task}
        
        Results:
        {results_text}
        
        Please provide a concise summary of these results, highlighting:
        1. The overall success or failure of the task
        2. Key findings or outcomes
        3. Any significant quantum acceleration benefits observed
        4. Recommendations for follow-up actions based on these results
        
        Keep the summary clear and focused on the practical implications.
        """
        
        # Get summary from AI
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": summary_prompt}]
        )
        
        return response.choices[0].message.content
    
    async def _enhance_search_query(self, query: str) -> str:
        """Enhance a search query to improve results."""
        enhance_prompt = f"""
        Original search query: "{query}"
        
        Please enhance this search query to make it more specific and effective.
        Add relevant terms that would help find the most accurate information.
        Keep the enhanced query concise but comprehensive.
        """
        
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": enhance_prompt}],
            max_tokens=100
        )
        
        enhanced_query = response.choices[0].message.content.strip()
        if enhanced_query.startswith('"') and enhanced_query.endswith('"'):
            enhanced_query = enhanced_query[1:-1]
            
        return enhanced_query
    
    async def _summarize_search_results(self, results: Dict[str, Any]) -> str:
        """Summarize search results."""
        result_items = results.get("results", [])
        if not result_items:
            return "No results found."
            
        # Prepare results for summarization
        results_text = ""
        for i, result in enumerate(result_items[:5]):  # Limit to top 5
            results_text += f"{i+1}. {result.get('title', 'Untitled')}\n"
            results_text += f"   URL: {result.get('url', 'No URL')}\n"
            results_text += f"   Summary: {result.get('summary', 'No summary')[:200]}...\n\n"
        
        # Create prompt for summary
        summary_prompt = f"""
        Summarize the following search results in a concise paragraph:
        
        {results_text}
        
        Focus on extracting the most relevant information related to the query.
        """
        
        # Get summary from AI
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=200
        )
        
        return response.choices[0].message.content.strip()
    
    async def process_user_instruction(self, instruction: str) -> Dict[str, Any]:
        """Process a natural language instruction from the user."""
        # Add to memory
        self.memory.add_interaction("user", instruction)
        
        # First, analyze the instruction to understand the intent
        intent = await self._analyze_instruction_intent(instruction)
        
        # Plan execution based on intent
        if intent["type"] == "search":
            result = await self._web_search_tool(intent["query"])
            action = "search"
        elif intent["type"] == "factorize":
            result = await self._factorize_tool(intent["number"])
            action = "factorize"
        elif intent["type"] == "optimize":
            result = await self._optimize_tool(intent["resources"], intent["constraints"])
            action = "optimize"
        elif intent["type"] == "analyze":
            result = await self._analyze_data_tool(intent["data"], intent["objective"])
            action = "analyze"
        elif intent["type"] == "complex_task":
            result = await self._execute_task_tool(instruction)
            action = "execute_task"
        else:
            # General task, let the agent figure out the best approach
            result = await self._execute_task_tool(instruction)
            action = "execute_task"
        
        # Generate response to user
        response = await self._generate_user_response(instruction, action, result)
        
        # Add to memory
        self.memory.add_interaction("assistant", response, {"action": action, "result": result})
        
        return {
            "response": response,
            "action": action,
            "result": result
        }
    
    async def _analyze_instruction_intent(self, instruction: str) -> Dict[str, Any]:
        """Analyze the user's instruction to determine the intent and extract parameters."""
        # Prepare prompt for intent analysis
        tools_descriptions = []
        for tool in self.tools:
            tools_descriptions.append(f"- {tool.name}: {tool.description}")
        
        intent_prompt = f"""
        User instruction: "{instruction}"
        
        Available tools:
        {chr(10).join(tools_descriptions)}
        
        Analyze the user's instruction and determine which tool would be most appropriate to use.
        Extract any relevant parameters that would be needed for the tool.
        
        Format your response as valid JSON with the following structure:
        {{
            "type": "tool_name or complex_task",
            "confidence": 0.0-1.0,
            ... (extracted parameters based on the tool)
        }}
        
        For example, if it's a search query:
        {{
            "type": "search",
            "confidence": 0.95,
            "query": "the search query"
        }}
        
        Or if it's a factorization request:
        {{
            "type": "factorize",
            "confidence": 0.98,
            "number": 12345
        }}
        
        Or if it's an optimization request:
        {{
            "type": "optimize",
            "confidence": 0.85,
            "resources": {{ ... resource definition ... }},
            "constraints": [ ... constraint definitions ... ]
        }}
        
        Only include parameters that can be directly extracted from the user's instruction.
        """
        
        # Get intent analysis from AI
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": intent_prompt}],
            response_format={"type": "json_object"}
        )
        
        # Parse intent from JSON
        intent_text = response.choices[0].message.content
        intent = json.loads(intent_text)
        
        return intent
    
    async def _generate_user_response(self, instruction: str, action: str, result: Dict[str, Any]) -> str:
        """Generate a natural language response based on the action and result."""
        # Prepare prompt for response generation
        response_prompt = f"""
        User instruction: "{instruction}"
        
        Action performed: {action}
        
        Result: {json.dumps(result, indent=2)}
        
        Generate a natural, friendly response to the user that:
        1. Acknowledges their request
        2. Explains what was done (mentioning quantum acceleration if used)
        3. Presents the key findings or results in an easy-to-understand way
        4. Offers follow-up suggestions if relevant
        
        Keep the language conversational and helpful.
        """
        
        # Get response from AI
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": response_prompt}]
        )
        
        return response.choices[0].message.content.strip()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics for the agent."""
        metrics = self.performance_metrics.copy()
        
        # Calculate derived metrics
        if metrics["tasks_completed"] > 0:
            metrics["avg_execution_time"] = metrics["total_execution_time"] / metrics["tasks_completed"]
        else:
            metrics["avg_execution_time"] = 0
            
        if metrics["classical_tasks"] > 0 and metrics["quantum_accelerated_tasks"] > 0:
            metrics["quantum_usage_percentage"] = (metrics["quantum_accelerated_tasks"] / 
                                                 metrics["tasks_completed"]) * 100
        else:
            metrics["quantum_usage_percentage"] = 0 if not self.use_quantum else 100
            
        # Add quantum component metrics if available
        if self.quantum_optimizer:
            qo_metrics = self.quantum_optimizer.get_circuit_stats()
            metrics.update({"quantum_optimizer_" + k: v for k, v in qo_metrics.items()})
            
        if self.web_agent:
            wa_metrics = self.web_agent.get_performance_metrics()
            metrics.update({"web_agent_" + k: v for k, v in wa_metrics.items()})
            
        if self.hybrid_computer:
            hc_metrics = self.hybrid_computer.get_quantum_metrics()
            metrics.update({"hybrid_computer_" + k: v for k, v in hc_metrics.items()})
            
        return metrics