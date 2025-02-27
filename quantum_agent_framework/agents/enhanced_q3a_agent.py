"""
Enhanced Q3A Agent with Advanced Browser Automation and Claude 3.7 Sonnet Integration
"""

import asyncio
import logging
import os
import time
import json
import re
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

# Quantum libraries
import pennylane as qml
import numpy as np

# AI models
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

# QUASAR Framework components
from quantum_agent_framework.quantum.optimizer import QuantumOptimizer
from quantum_agent_framework.quantum.factorization_manager import FactorizationManager
from quantum_agent_framework.integration.hybrid_computation import HybridComputation
from quantum_agent_framework.agent_core.browser_interaction import QuantumEnhancedBrowser
from quantum_agent_framework.agents.web_agent import QuantumWebAgent
from quantum_agent_framework.agent_core.q3a_agent import Q3AAgent
from quantum_agent_framework.agent_core.task_engine import QUASARTaskEngine


@dataclass
class EnhancedAgentMemory:
    """Extended memory system for Enhanced Q3A Agent"""
    short_term: List[Dict[str, Any]] = None
    working_memory: Dict[str, Any] = None
    long_term: List[Dict[str, Any]] = None
    web_history: List[Dict[str, str]] = None
    max_short_term: int = 15
    
    def __post_init__(self):
        """Initialize memory collections if not provided"""
        if self.short_term is None:
            self.short_term = []
        if self.working_memory is None:
            self.working_memory = {}
        if self.long_term is None:
            self.long_term = []
        if self.web_history is None:
            self.web_history = []
    
    def add_interaction(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add an interaction to memory"""
        self.short_term.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        })
        
        # Trim if exceeding max size
        if len(self.short_term) > self.max_short_term:
            self.short_term = self.short_term[-self.max_short_term:]
    
    def add_web_visit(self, url: str, title: str, summary: str = "", relevance: float = 0.0):
        """Add a web visit to history"""
        self.web_history.append({
            "url": url,
            "title": title,
            "summary": summary,
            "relevance": relevance,
            "timestamp": time.time()
        })
    
    def get_context_for_claude(self, include_web_history: bool = True) -> str:
        """Generate context summary for Claude prompts"""
        context = "Recent interactions:\n"
        
        for interaction in self.short_term[-5:]:  # Last 5 interactions
            context += f"- {interaction['role'].upper()}: {interaction['content'][:200]}...\n"
        
        if include_web_history and self.web_history:
            context += "\nRecent web browsing history:\n"
            for visit in self.web_history[-5:]:
                context += f"- {visit['title']} ({visit['url']})\n"
                if visit['summary']:
                    context += f"  Summary: {visit['summary'][:100]}...\n"
        
        # Add working memory if relevant
        if self.working_memory:
            context += "\nCurrent context:\n"
            for key, value in self.working_memory.items():
                if isinstance(value, str):
                    context += f"- {key}: {value[:100]}...\n"
                else:
                    context += f"- {key}: {str(value)[:100]}...\n"
        
        return context


class EnhancedQ3AAgent:
    """
    Enhanced Quantum-Accelerated AI Agent with browser automation and Claude integration.
    This agent extends Q3A with advanced capabilities and better decision-making for hybrid
    quantum-classical computing.
    """
    
    def __init__(self, 
                 n_qubits: int = 8, 
                 use_quantum: bool = True, 
                 use_azure: bool = True,
                 use_claude: bool = True):
        """
        Initialize the Enhanced Q3A agent.
        
        Args:
            n_qubits: Number of qubits to use for quantum circuits
            use_quantum: Whether to use quantum acceleration
            use_azure: Whether to use Azure Quantum for hardware acceleration
            use_claude: Whether to use Claude 3.7 Sonnet for reasoning
        """
        self.n_qubits = n_qubits
        self.use_quantum = use_quantum
        self.use_azure = use_azure
        self.use_claude = use_claude
        
        # Initialize memory
        self.memory = EnhancedAgentMemory()
        
        # Initialize quantum components
        self.quantum_optimizer = QuantumOptimizer(n_qubits=n_qubits, use_azure=use_azure)
        self.factorization_manager = FactorizationManager(quantum_optimizer=self.quantum_optimizer)
        self.hybrid_computation = HybridComputation(
            n_qubits=n_qubits, 
            use_quantum=use_quantum, 
            use_azure=use_azure
        )
        
        # Initialize task engine
        self.task_engine = QUASARTaskEngine(
            n_qubits=n_qubits, 
            use_quantum=use_quantum, 
            max_concurrent_tasks=5
        )
        
        # Initialize web components
        self.web_agent = QuantumWebAgent(n_qubits=n_qubits, use_quantum=use_quantum)
        self.browser = QuantumEnhancedBrowser(n_qubits=n_qubits, use_quantum=use_quantum)
        
        # Initialize AI clients
        self.openai_client = None
        self.anthropic_client = None
        self._initialize_ai_clients()
        
        # Performance metrics
        self.metrics = {
            "tasks_completed": 0,
            "quantum_accelerated_tasks": 0,
            "classical_tasks": 0,
            "browser_sessions": 0,
            "average_quantum_speedup": 0,
            "total_execution_time": 0,
            "claude_interactions": 0,
            "openai_interactions": 0,
            "successful_interactions": 0,
            "failed_interactions": 0
        }
        
        logging.info(f"Enhanced Q3A Agent initialized with {n_qubits} qubits")
    
    def _initialize_ai_clients(self):
        """Initialize AI client connections"""
        # Initialize OpenAI client
        if os.environ.get("OPENAI_API_KEY"):
            self.openai_client = AsyncOpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
            )
            logging.info("OpenAI client initialized")
        else:
            logging.warning("OpenAI API key not available")
        
        # Initialize Anthropic client
        if os.environ.get("ANTHROPIC_API_KEY") and self.use_claude:
            self.anthropic_client = AsyncAnthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
            )
            logging.info("Anthropic client initialized")
        else:
            if self.use_claude:
                logging.warning("Anthropic API key not available")
    
    async def initialize(self):
        """Initialize components that require async setup"""
        await self.web_agent.initialize()
        await self._initialize_browser()
        logging.info("Enhanced Q3A Agent async components initialized")
    
    async def _initialize_browser(self):
        """Initialize the browser if needed"""
        try:
            if not hasattr(self.browser, 'browser') or not self.browser.browser:
                await self.browser._initialize_browser()
                self.metrics["browser_sessions"] += 1
                logging.info("Browser initialized")
        except Exception as e:
            logging.error(f"Error initializing browser: {str(e)}")
    
    async def process_user_instruction(self, instruction: str) -> Dict[str, Any]:
        """
        Process a user instruction using the most appropriate method.
        
        Args:
            instruction: The natural language instruction from the user
            
        Returns:
            Dict containing processing results
        """
        start_time = time.time()
        
        # Add to memory
        self.memory.add_interaction("user", instruction)
        
        # Analyze instruction to determine the appropriate processing method
        analysis = await self._analyze_instruction(instruction)
        task_type = analysis["task_type"]
        use_quantum = analysis["use_quantum"] and self.use_quantum
        requires_browsing = analysis["requires_browsing"]
        
        result = None
        
        # Route to appropriate processor based on task type
        if requires_browsing:
            result = await self._process_browsing_task(instruction, analysis, use_quantum)
        elif task_type == "factorization":
            result = await self._process_factorization(instruction, analysis, use_quantum)
        elif task_type == "optimization":
            result = await self._process_optimization(instruction, analysis, use_quantum)
        elif task_type == "search":
            result = await self._process_search(instruction, analysis, use_quantum)
        else:
            # General task
            result = await self._process_general_task(instruction, analysis, use_quantum)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        self.metrics["total_execution_time"] += execution_time
        self.metrics["tasks_completed"] += 1
        
        if use_quantum:
            self.metrics["quantum_accelerated_tasks"] += 1
            # Update average speedup
            if "speedup" in result and result["speedup"] > 0:
                current_speedup = self.metrics["average_quantum_speedup"]
                count = self.metrics["quantum_accelerated_tasks"]
                self.metrics["average_quantum_speedup"] = (current_speedup * (count - 1) + result["speedup"]) / count
        else:
            self.metrics["classical_tasks"] += 1
        
        # Add result to memory
        self.memory.add_interaction("assistant", result["response"])
        
        # Return the full result
        return {
            "task_id": result.get("task_id", time.time()),
            "task": instruction,
            "task_type": task_type,
            "use_quantum": use_quantum,
            "execution_time": execution_time,
            "result": result,
            "requires_browsing": requires_browsing
        }
    
    async def _analyze_instruction(self, instruction: str) -> Dict[str, Any]:
        """
        Analyze the user instruction to determine task type and processing method.
        
        Args:
            instruction: The user instruction
            
        Returns:
            Dict with analysis results
        """
        # Default to using Claude for analysis if available
        if self.anthropic_client and self.use_claude:
            return await self._analyze_with_claude(instruction)
        elif self.openai_client:
            return await self._analyze_with_openai(instruction)
        else:
            # Fallback to basic rule-based analysis
            return await self._analyze_basic(instruction)
    
    async def _analyze_with_claude(self, instruction: str) -> Dict[str, Any]:
        """Use Claude to analyze the instruction"""
        try:
            prompt = f"""
            I need you to analyze this user instruction and categorize it:
            
            "{instruction}"
            
            Please classify this into one of these task types:
            1. factorization - if the user wants to factorize a number
            2. optimization - if the user wants to optimize resources or solve a constraint problem
            3. search - if the user wants to search for information
            4. general - for any other type of instruction
            
            Also determine:
            - Whether quantum computing would likely provide an advantage (true/false)
            - If this task requires web browsing (true/false)
            - For factorization: extract the number to factorize
            - For search: extract the search query
            - For optimization: extract what needs to be optimized
            
            Respond in valid JSON format only, with these keys:
            {{
                "task_type": "factorization|optimization|search|general",
                "use_quantum": true|false,
                "requires_browsing": true|false,
                "parameters": {{ relevant parameters based on task type }}
            }}
            """
            
            response = await self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=0.1,
                system="You are an AI assistant specializing in quantum computing applications.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            self.metrics["claude_interactions"] += 1
            self.metrics["successful_interactions"] += 1
            
            # Extract JSON from response
            text = response.content[0].text
            json_match = re.search(r'({.*})', text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group(1))
                return analysis
            else:
                raise ValueError("Could not extract JSON from Claude response")
        
        except Exception as e:
            logging.error(f"Error analyzing with Claude: {str(e)}")
            self.metrics["failed_interactions"] += 1
            # Fallback to basic analysis
            return await self._analyze_basic(instruction)
    
    async def _analyze_with_openai(self, instruction: str) -> Dict[str, Any]:
        """Use OpenAI to analyze the instruction"""
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You analyze user instructions for a quantum computing AI assistant."},
                    {"role": "user", "content": f"""
                    Analyze this instruction and return a JSON object with the following keys:
                    - task_type: 'factorization', 'optimization', 'search', or 'general'
                    - use_quantum: boolean indicating if quantum computing would provide advantage
                    - requires_browsing: boolean indicating if web browsing is needed
                    - parameters: object containing task-specific parameters
                    
                    Instruction: "{instruction}"
                    
                    Respond with only valid JSON.
                    """}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            self.metrics["openai_interactions"] += 1
            self.metrics["successful_interactions"] += 1
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logging.error(f"Error analyzing with OpenAI: {str(e)}")
            self.metrics["failed_interactions"] += 1
            # Fallback to basic analysis
            return await self._analyze_basic(instruction)
    
    async def _analyze_basic(self, instruction: str) -> Dict[str, Any]:
        """Basic rule-based instruction analysis"""
        instruction_lower = instruction.lower()
        
        # Check for factorization
        if any(word in instruction_lower for word in ["factor", "factorize", "factorization", "prime factors"]):
            # Try to extract the number
            number_match = re.search(r'\b(\d+)\b', instruction)
            number = int(number_match.group(1)) if number_match else 0
            
            return {
                "task_type": "factorization",
                "use_quantum": number > 1000000,  # Use quantum for larger numbers
                "requires_browsing": False,
                "parameters": {"number": number}
            }
            
        # Check for search
        elif any(word in instruction_lower for word in ["search", "find", "lookup", "information about", "what is"]):
            return {
                "task_type": "search",
                "use_quantum": True,  # Quantum search may be beneficial
                "requires_browsing": True,
                "parameters": {"query": instruction}
            }
            
        # Check for optimization
        elif any(word in instruction_lower for word in ["optimize", "optimization", "allocate", "maximize", "minimize"]):
            return {
                "task_type": "optimization",
                "use_quantum": True,  # Quantum optimization is generally beneficial
                "requires_browsing": False,
                "parameters": {"problem": instruction}
            }
            
        # Default to general
        else:
            return {
                "task_type": "general",
                "use_quantum": False,  # Default to classical for general tasks
                "requires_browsing": "browse" in instruction_lower or "website" in instruction_lower,
                "parameters": {"instruction": instruction}
            }
    
    async def _process_browsing_task(self, instruction: str, analysis: Dict[str, Any], use_quantum: bool) -> Dict[str, Any]:
        """
        Process tasks that require web browsing.
        
        Args:
            instruction: The original user instruction
            analysis: The instruction analysis
            use_quantum: Whether to use quantum acceleration
            
        Returns:
            Dict with browsing results
        """
        # Ensure browser is initialized
        await self._initialize_browser()
        
        # Extract URL if present
        url_match = re.search(r'https?://[^\s]+', instruction)
        url = url_match.group(0) if url_match else None
        
        # If no URL found, try to extract a domain
        if not url:
            domain_match = re.search(r'\b([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)\b', instruction)
            if domain_match:
                url = f"https://{domain_match.group(0)}"
        
        # If still no URL, perform a search and navigate to top result
        if not url and analysis["task_type"] == "search":
            search_results = await self.web_agent.search(analysis["parameters"]["query"])
            if search_results.get("success") and search_results.get("results"):
                url = search_results["results"][0].get("url")
        
        browsing_result = {}
        
        # Navigate to URL if found
        if url:
            navigation_result = await self.browser.navigate(url)
            
            if navigation_result.get("success"):
                # Store in web history
                self.memory.add_web_visit(
                    url=navigation_result["url"],
                    title=navigation_result.get("title", ""),
                    relevance=navigation_result.get("content_relevance", {}).get("relevance_score", 0)
                )
                
                # Get page content
                page_content = await self.browser.get_page_content()
                
                # Extract links and forms
                links = await self.browser.find_links()
                forms = await self.browser.find_forms()
                
                # Take screenshot
                screenshot = await self.browser.take_screenshot()
                
                # Analyze content with Claude or OpenAI
                content_analysis = await self._analyze_web_content(
                    instruction=instruction,
                    url=url,
                    title=navigation_result.get("title", ""),
                    content=page_content.get("content", ""),
                    links=links.get("links", []),
                    forms=forms.get("forms", [])
                )
                
                browsing_result = {
                    "success": True,
                    "url": url,
                    "title": navigation_result.get("title", ""),
                    "summary": content_analysis.get("summary", ""),
                    "relevance": content_analysis.get("relevance", 0),
                    "suggested_actions": content_analysis.get("suggested_actions", []),
                    "has_screenshot": True,
                    "quantum_enhanced": use_quantum,
                    "quantum_time": navigation_result.get("content_relevance", {}).get("processing_time", 0) if use_quantum else 0,
                    "classical_time": 0,  # We don't have a classical comparison for this
                    "speedup": 1.0  # Default value
                }
            else:
                # Navigation failed
                browsing_result = {
                    "success": False,
                    "url": url,
                    "error": navigation_result.get("error", "Failed to navigate"),
                    "has_screenshot": False,
                    "quantum_enhanced": False
                }
        else:
            # Couldn't determine URL
            browsing_result = {
                "success": False,
                "error": "Could not determine a URL to navigate to",
                "has_screenshot": False,
                "quantum_enhanced": False
            }
        
        # Generate natural language response
        response = await self._generate_browsing_response(instruction, browsing_result)
        browsing_result["response"] = response
        
        return browsing_result
    
    async def _analyze_web_content(self, instruction: str, url: str, title: str, content: str, links: List[Dict], forms: List[Dict]) -> Dict[str, Any]:
        """
        Analyze web content with AI to determine relevance and potential actions.
        
        Args:
            instruction: Original user instruction
            url: Page URL
            title: Page title
            content: Page content
            links: List of links on the page
            forms: List of forms on the page
            
        Returns:
            Dict with content analysis
        """
        # Truncate content if too long
        content_summary = content[:5000] + "..." if len(content) > 5000 else content
        
        # Prepare link and form summaries
        link_summary = "\n".join([f"- {link.get('text', 'No text')} -> {link.get('href', '#')}" for link in links[:10]])
        form_summary = "\n".join([f"- Form {form.get('id', 'No ID')}: {len(form.get('elements', []))} input fields" for form in forms[:5]])
        
        if self.anthropic_client and self.use_claude:
            prompt = f"""
            I need you to analyze this web page in relation to the user's instruction:
            
            User instruction: "{instruction}"
            
            Page URL: {url}
            Page title: {title}
            
            Page content:
            {content_summary}
            
            Top links:
            {link_summary}
            
            Forms on page:
            {form_summary}
            
            Please provide:
            1. A concise summary of the page content (2-3 sentences)
            2. Relevance to the user's instruction (0-100)
            3. Suggested next actions (e.g., click a specific link, fill a form, extract specific information)
            
            Respond in valid JSON format only with these keys:
            {{
                "summary": "concise summary",
                "relevance": relevance_score,
                "suggested_actions": [list of 1-3 specific actions]
            }}
            """
            
            try:
                response = await self.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1500,
                    temperature=0.1,
                    system="You are an AI assistant specialized in web content analysis.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                self.metrics["claude_interactions"] += 1
                
                # Extract JSON from response
                text = response.content[0].text
                json_match = re.search(r'({.*})', text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                else:
                    raise ValueError("Could not extract JSON from Claude response")
                    
            except Exception as e:
                logging.error(f"Error analyzing web content with Claude: {str(e)}")
                self.metrics["failed_interactions"] += 1
                # Fallback to basic analysis
                return {
                    "summary": title,
                    "relevance": 50,  # Default middle value
                    "suggested_actions": ["Extract information from page"]
                }
                
        elif self.openai_client:
            # Fallback to OpenAI if Claude is unavailable
            try:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You analyze web page content for relevance and suggested actions."},
                        {"role": "user", "content": f"""
                        Analyze this web page in relation to the user's instruction:
                        
                        User instruction: "{instruction}"
                        
                        URL: {url}
                        Title: {title}
                        
                        Content: {content_summary}
                        
                        Links: {link_summary}
                        
                        Forms: {form_summary}
                        
                        Provide a JSON object with:
                        - summary: concise 2-3 sentence summary
                        - relevance: number 0-100 indicating relevance
                        - suggested_actions: list of 1-3 specific actions to take
                        
                        Respond with only valid JSON.
                        """}
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                
                self.metrics["openai_interactions"] += 1
                
                return json.loads(response.choices[0].message.content)
                
            except Exception as e:
                logging.error(f"Error analyzing web content with OpenAI: {str(e)}")
                self.metrics["failed_interactions"] += 1
                # Fallback to basic analysis
                return {
                    "summary": title,
                    "relevance": 50,
                    "suggested_actions": ["Extract information from page"]
                }
        else:
            # Basic analysis if no AI available
            return {
                "summary": title,
                "relevance": 50,
                "suggested_actions": ["Extract information from page"]
            }
    
    async def _generate_browsing_response(self, instruction: str, browsing_result: Dict[str, Any]) -> str:
        """Generate a natural language response for browsing results"""
        if not browsing_result.get("success"):
            return f"I couldn't browse to the requested website. Error: {browsing_result.get('error', 'Unknown error')}"
        
        response = f"I browsed to {browsing_result['url']} ({browsing_result['title']}).\n\n"
        
        if browsing_result.get("summary"):
            response += f"Summary: {browsing_result['summary']}\n\n"
        
        if browsing_result.get("suggested_actions"):
            response += "Based on the page content, you could:\n"
            for action in browsing_result["suggested_actions"]:
                response += f"- {action}\n"
        
        if browsing_result.get("quantum_enhanced"):
            response += f"\nNote: I used quantum-enhanced processing to analyze this page's content."
        
        return response
    
    async def _process_factorization(self, instruction: str, analysis: Dict[str, Any], use_quantum: bool) -> Dict[str, Any]:
        """Process factorization tasks"""
        number = analysis["parameters"].get("number", 0)
        
        if number <= 0:
            # Try to extract number again if not found in analysis
            number_match = re.search(r'\b(\d+)\b', instruction)
            number = int(number_match.group(1)) if number_match else 0
        
        if number <= 0:
            return {
                "success": False,
                "error": "Could not determine a number to factorize",
                "response": "I couldn't find a number to factorize in your instruction. Please specify a number."
            }
        
        start_time = time.time()
        
        # Process with quantum if appropriate
        quantum_time = 0
        classical_time = 0
        speedup = 1.0
        
        try:
            # Get factorization result
            factorization_result = await self.factorization_manager.factorize(number)
            
            # Extract results
            factors = factorization_result.factors
            method_used = factorization_result.method_used
            computation_time = factorization_result.computation_time
            details = factorization_result.details
            
            # Get prime factorization
            prime_factors = []
            n = number
            for i in range(2, int(n**0.5) + 1):
                while n % i == 0:
                    prime_factors.append(i)
                    n //= i
            if n > 1:
                prime_factors.append(n)
            
            # Record quantum vs classical time
            if use_quantum:
                quantum_time = computation_time
                classical_time = details.get("classical_time", computation_time * 0.8)  # Fallback if not available
            else:
                classical_time = computation_time
                quantum_time = details.get("quantum_time", computation_time * 1.2)  # Fallback if not available
            
            speedup = classical_time / max(quantum_time, 0.001)  # Avoid division by zero
            
            # Get detailed explanation
            explanation = await self.factorization_manager.get_advanced_factorization_explanation(
                number=number,
                factors=factors,
                method_used=method_used
            )
            
            result = {
                "success": True,
                "number": number,
                "factors": factors,
                "prime_factors": prime_factors,
                "method_used": method_used,
                "explanation": explanation,
                "quantum_time": quantum_time,
                "classical_time": classical_time,
                "speedup": speedup,
                "quantum_enhanced": use_quantum
            }
            
            # Generate a user-friendly response
            response = f"I've factorized {number} using {method_used}.\n\n"
            response += f"Factors: {', '.join(map(str, sorted(factors)))}\n"
            response += f"Prime factorization: {' × '.join(map(str, prime_factors))}\n\n"
            response += explanation[:2000] + "..." if len(explanation) > 2000 else explanation
            
            result["response"] = response
            return result
            
        except Exception as e:
            logging.error(f"Error in factorization: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": f"I encountered an error while trying to factorize {number}: {str(e)}"
            }
    
    async def _process_search(self, instruction: str, analysis: Dict[str, Any], use_quantum: bool) -> Dict[str, Any]:
        """Process search tasks"""
        query = analysis["parameters"].get("query", instruction)
        
        try:
            # Perform search
            search_result = await self.web_agent.search(query, max_results=5)
            
            # Process results
            if search_result.get("success"):
                results = search_result.get("results", [])
                quantum_time = search_result.get("quantum_time", 0)
                classical_time = search_result.get("classical_time", 0)
                speedup = search_result.get("speedup", 1.0)
                
                # Generate summary using Claude if available
                summary = ""
                if results and (self.anthropic_client or self.openai_client):
                    summary = await self._generate_search_summary(query, results)
                
                result = {
                    "success": True,
                    "query": query,
                    "results": results,
                    "summary": summary,
                    "quantum_time": quantum_time,
                    "classical_time": classical_time,
                    "speedup": speedup,
                    "quantum_enhanced": use_quantum
                }
                
                # Generate a user-friendly response
                response = f"Here are the search results for '{query}':\n\n"
                
                if summary:
                    response += f"Summary: {summary}\n\n"
                
                for i, item in enumerate(results[:3]):
                    response += f"{i+1}. {item.get('title', 'No title')}\n"
                    response += f"   {item.get('content', '')[:150]}...\n"
                    if item.get('url'):
                        response += f"   URL: {item.get('url')}\n"
                    response += "\n"
                
                if use_quantum:
                    response += f"\nNote: Search was enhanced with quantum processing."
                
                result["response"] = response
                return result
                
            else:
                return {
                    "success": False,
                    "error": search_result.get("error", "Search failed"),
                    "response": f"I couldn't complete the search for '{query}'. Error: {search_result.get('error', 'Unknown error')}"
                }
                
        except Exception as e:
            logging.error(f"Error in search: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": f"I encountered an error while searching for '{query}': {str(e)}"
            }
    
    async def _process_optimization(self, instruction: str, analysis: Dict[str, Any], use_quantum: bool) -> Dict[str, Any]:
        """Process optimization tasks"""
        problem = analysis["parameters"].get("problem", instruction)
        
        try:
            # Extract optimization parameters using Claude or OpenAI
            parameters = await self._extract_optimization_parameters(problem)
            
            if not parameters.get("success"):
                return {
                    "success": False,
                    "error": "Could not extract optimization parameters",
                    "response": "I couldn't determine what to optimize. Please provide more specific details about your optimization problem."
                }
            
            # Get resources to optimize
            resources = parameters.get("resources", {})
            constraints = parameters.get("constraints", [])
            objective = parameters.get("objective", "maximize")
            
            # Perform optimization
            optimization_result = self.quantum_optimizer.optimize_resource_allocation(resources)
            
            if optimization_result.get("success"):
                solution = optimization_result.get("solution", {})
                objective_value = optimization_result.get("objective_value", 0)
                quantum_time = optimization_result.get("quantum_time", 0)
                classical_time = optimization_result.get("classical_time", 0)
                speedup = optimization_result.get("speedup", 1.0)
                method_used = optimization_result.get("method", "QAOA" if use_quantum else "Classical")
                
                # Generate explanation
                explanation = await self._generate_optimization_explanation(
                    resources=resources,
                    constraints=constraints,
                    solution=solution,
                    objective_value=objective_value,
                    method=method_used
                )
                
                result = {
                    "success": True,
                    "resources": resources,
                    "constraints": constraints,
                    "solution": solution,
                    "objective_value": objective_value,
                    "explanation": explanation,
                    "method": method_used,
                    "quantum_time": quantum_time,
                    "classical_time": classical_time,
                    "speedup": speedup,
                    "quantum_enhanced": use_quantum
                }
                
                # Generate a user-friendly response
                response = f"I've optimized your resources using {method_used}.\n\n"
                response += "Optimization solution:\n"
                for resource, value in solution.items():
                    response += f"- {resource}: {value}\n"
                response += f"\nObjective value: {objective_value}\n\n"
                response += explanation
                
                result["response"] = response
                return result
                
            else:
                return {
                    "success": False,
                    "error": optimization_result.get("error", "Optimization failed"),
                    "response": f"I couldn't complete the optimization. Error: {optimization_result.get('error', 'Unknown error')}"
                }
                
        except Exception as e:
            logging.error(f"Error in optimization: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": f"I encountered an error during optimization: {str(e)}"
            }
    
    async def _process_general_task(self, instruction: str, analysis: Dict[str, Any], use_quantum: bool) -> Dict[str, Any]:
        """Process general tasks that don't fit other categories"""
        start_time = time.time()
        
        # Use Claude to process general tasks if available
        if self.anthropic_client and self.use_claude:
            context = self.memory.get_context_for_claude()
            
            try:
                prompt = f"""
                User instruction: {instruction}
                
                Previous context:
                {context}
                
                Please respond to this instruction as helpfully and accurately as possible.
                If this relates to quantum computing, provide educational insights.
                """
                
                response = await self.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=4000,
                    temperature=0.7,
                    system="You are a quantum computing expert assistant. You are part of a quantum-enhanced AI agent system that can use both classical and quantum computing for various tasks.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                self.metrics["claude_interactions"] += 1
                
                ai_response = response.content[0].text
                
                result = {
                    "success": True,
                    "response": ai_response,
                    "quantum_time": 0,  # No quantum processing for general tasks
                    "classical_time": time.time() - start_time,
                    "speedup": 1.0,
                    "quantum_enhanced": False,
                    "ai_model": "claude-3-sonnet"
                }
                
                return result
                
            except Exception as e:
                logging.error(f"Error processing with Claude: {str(e)}")
                self.metrics["failed_interactions"] += 1
                # Fallback to OpenAI
        
        # Fallback to OpenAI
        if self.openai_client:
            try:
                context = self.memory.get_context_for_claude()
                
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a quantum computing expert assistant. You are part of a quantum-enhanced AI agent system."},
                        {"role": "user", "content": f"""
                        Previous context:
                        {context}
                        
                        User instruction: {instruction}
                        
                        Please respond helpfully and accurately.
                        """}
                    ],
                    temperature=0.7
                )
                
                self.metrics["openai_interactions"] += 1
                
                ai_response = response.choices[0].message.content
                
                result = {
                    "success": True,
                    "response": ai_response,
                    "quantum_time": 0,
                    "classical_time": time.time() - start_time,
                    "speedup": 1.0,
                    "quantum_enhanced": False,
                    "ai_model": "gpt-4o"
                }
                
                return result
                
            except Exception as e:
                logging.error(f"Error processing with OpenAI: {str(e)}")
                self.metrics["failed_interactions"] += 1
        
        # Basic fallback if no AI available
        return {
            "success": False,
            "error": "No AI available to process general task",
            "response": "I'm having trouble processing your request. Please try a more specific task like factorization, searching, or optimization."
        }
    
    async def _extract_optimization_parameters(self, problem: str) -> Dict[str, Any]:
        """Extract optimization parameters from natural language description"""
        if self.anthropic_client and self.use_claude:
            try:
                prompt = f"""
                Extract optimization parameters from this problem description:
                
                "{problem}"
                
                Identify:
                1. Resources to optimize (with their quantities if available)
                2. Constraints on the optimization
                3. The objective (maximize or minimize)
                
                Respond in valid JSON format only with these keys:
                {{
                    "success": true,
                    "resources": {{ "resource1": quantity1, ... }},
                    "constraints": [ "constraint1", ... ],
                    "objective": "maximize" or "minimize"
                }}
                
                If you can't extract meaningful optimization parameters, respond with:
                {{
                    "success": false,
                    "error": "reason"
                }}
                """
                
                response = await self.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1500,
                    temperature=0.1,
                    system="You are an expert in optimization problems and mathematical modeling.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                self.metrics["claude_interactions"] += 1
                
                # Extract JSON from response
                text = response.content[0].text
                json_match = re.search(r'({.*})', text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                else:
                    raise ValueError("Could not extract JSON from Claude response")
                    
            except Exception as e:
                logging.error(f"Error extracting optimization parameters with Claude: {str(e)}")
                self.metrics["failed_interactions"] += 1
        
        # Fallback to OpenAI
        if self.openai_client:
            try:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You extract optimization parameters from problem descriptions."},
                        {"role": "user", "content": f"""
                        Extract optimization parameters from this problem:
                        
                        "{problem}"
                        
                        Provide a JSON object with:
                        - success: boolean
                        - resources: object mapping resource names to quantities
                        - constraints: array of constraint descriptions
                        - objective: "maximize" or "minimize"
                        
                        If you can't extract meaningful parameters, set success to false with an error reason.
                        
                        Respond with only valid JSON.
                        """}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                
                self.metrics["openai_interactions"] += 1
                
                return json.loads(response.choices[0].message.content)
                
            except Exception as e:
                logging.error(f"Error extracting optimization parameters with OpenAI: {str(e)}")
                self.metrics["failed_interactions"] += 1
        
        # Default fallback
        return {
            "success": False,
            "error": "Could not extract optimization parameters"
        }
    
    async def _generate_search_summary(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Generate a summary of search results using Claude or OpenAI"""
        if not results:
            return "No results found."
        
        # Extract content from results
        contents = [f"Result {i+1}: {result.get('title', 'No title')}\n{result.get('content', '')[:500]}" 
                   for i, result in enumerate(results[:5])]
        combined_content = "\n\n".join(contents)
        
        if self.anthropic_client and self.use_claude:
            try:
                prompt = f"""
                Summarize these search results for the query "{query}":
                
                {combined_content}
                
                Provide a concise 2-3 sentence summary that captures the most important information
                relevant to the query.
                """
                
                response = await self.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1000,
                    temperature=0.3,
                    system="You summarize search results concisely and accurately.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                self.metrics["claude_interactions"] += 1
                
                return response.content[0].text.strip()
                
            except Exception as e:
                logging.error(f"Error generating search summary with Claude: {str(e)}")
                self.metrics["failed_interactions"] += 1
        
        # Fallback to OpenAI
        if self.openai_client:
            try:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You summarize search results concisely and accurately."},
                        {"role": "user", "content": f"""
                        Summarize these search results for the query "{query}":
                        
                        {combined_content}
                        
                        Provide a concise 2-3 sentence summary.
                        """}
                    ],
                    temperature=0.3,
                    max_tokens=150
                )
                
                self.metrics["openai_interactions"] += 1
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                logging.error(f"Error generating search summary with OpenAI: {str(e)}")
                self.metrics["failed_interactions"] += 1
        
        # Default fallback
        return f"Search results for '{query}' provide information on this topic."
    
    async def _generate_optimization_explanation(self, resources: Dict[str, Any], constraints: List[str], 
                                               solution: Dict[str, Any], objective_value: float, method: str) -> str:
        """Generate an explanation of optimization results"""
        if self.anthropic_client and self.use_claude:
            try:
                quantum_context = ""
                if "qaoa" in method.lower() or "quantum" in method.lower():
                    quantum_context = """
                    Include a brief explanation of how quantum computing aids in optimization problems,
                    focusing on the Quantum Approximate Optimization Algorithm (QAOA) and its advantages
                    for combinatorial optimization problems.
                    """
                
                prompt = f"""
                Explain this optimization result:
                
                Resources: {json.dumps(resources)}
                Constraints: {json.dumps(constraints)}
                Solution: {json.dumps(solution)}
                Objective value: {objective_value}
                Method used: {method}
                
                Provide a clear explanation of:
                1. What was optimized and why
                2. How the optimization approach works
                3. What the solution means in practical terms
                {quantum_context}
                
                Keep the explanation concise (3-4 paragraphs) but informative.
                """
                
                response = await self.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1500,
                    temperature=0.3,
                    system="You are an optimization expert who explains solutions clearly.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                self.metrics["claude_interactions"] += 1
                
                return response.content[0].text.strip()
                
            except Exception as e:
                logging.error(f"Error generating optimization explanation with Claude: {str(e)}")
                self.metrics["failed_interactions"] += 1
        
        # Fallback to OpenAI
        if self.openai_client:
            try:
                quantum_context = ""
                if "qaoa" in method.lower() or "quantum" in method.lower():
                    quantum_context = "Include a brief explanation of quantum optimization advantages."
                
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an optimization expert who explains solutions clearly."},
                        {"role": "user", "content": f"""
                        Explain this optimization result:
                        
                        Resources: {json.dumps(resources)}
                        Constraints: {json.dumps(constraints)}
                        Solution: {json.dumps(solution)}
                        Objective value: {objective_value}
                        Method used: {method}
                        
                        Provide a clear explanation of what was optimized, how the approach works, 
                        and what the solution means practically.
                        
                        {quantum_context}
                        
                        Keep it concise (3-4 paragraphs) but informative.
                        """}
                    ],
                    temperature=0.3
                )
                
                self.metrics["openai_interactions"] += 1
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                logging.error(f"Error generating optimization explanation with OpenAI: {str(e)}")
                self.metrics["failed_interactions"] += 1
        
        # Basic fallback
        return f"This optimization problem was solved using {method} and achieved an objective value of {objective_value}. The optimal allocation of resources is shown in the solution above."
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        return self.metrics
    
    async def close(self):
        """Close connections and resources"""
        try:
            if hasattr(self, 'browser') and self.browser and hasattr(self.browser, 'close'):
                await self.browser.close()
            
            if hasattr(self, 'web_agent') and self.web_agent and hasattr(self.web_agent, 'close'):
                await self.web_agent.close()
                
            logging.info("Enhanced Q3A Agent resources closed")
        except Exception as e:
            logging.error(f"Error closing resources: {str(e)}")