"""
Enhanced QA³ Agent: Quantum-Accelerated AI Agent with Advanced Capabilities

This enhanced version of the QA³ agent integrates:
1. Deep search with 20+ sources
2. Browser automation with natural language processing
3. Task history tracking
4. PWA capabilities

It provides a complete intelligent agent with quantum-accelerated search
and comprehensive task execution capabilities.
"""

import os
import json
import time
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qa3-agent-enhanced")

class QA3AgentEnhanced:
    """
    Enhanced Quantum-Accelerated AI Agent with comprehensive capabilities
    
    This agent integrates:
    1. Deep search across 20+ sources with quantum acceleration
    2. Browser automation with natural language task processing
    3. Task history tracking and analysis
    4. PWA support for offline functionality
    """
    
    def __init__(self, use_quantum: bool = True, n_qubits: int = 8, 
                use_claude: bool = True, use_openai: bool = True):
        """
        Initialize the enhanced QA³ agent
        
        Args:
            use_quantum: Whether to use quantum acceleration
            n_qubits: Number of qubits for quantum operations
            use_claude: Whether to use Claude for AI processing
            use_openai: Whether to use OpenAI for AI processing
        """
        self.use_quantum = use_quantum
        self.n_qubits = n_qubits
        self.use_claude = use_claude
        self.use_openai = use_openai
        
        # Initialize components
        self.search_integration = None
        self.browser_integration = None
        self.task_history = None
        self.running = False
        self.initialized = False
        
        # Initialize AI clients
        self.claude_client = None
        self.openai_client = None
        self._initialize_ai_clients()
        
        # Status tracking
        self.status = {
            "agent_state": "initializing",
            "last_update": datetime.now().isoformat(),
            "components": {},
            "tasks_completed": 0,
            "searches_performed": 0,
            "web_interactions": 0,
            "quantum_operations": 0
        }
        
        logger.info("Enhanced QA³ agent created, ready for initialization")
    
    def _initialize_ai_clients(self):
        """Initialize AI clients for Claude and OpenAI"""
        # Initialize OpenAI client
        if self.use_openai and os.environ.get("OPENAI_API_KEY"):
            try:
                from openai import AsyncOpenAI
                self.openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                logger.info("OpenAI client initialized")
                self.status["components"]["openai"] = "available"
            except ImportError:
                logger.warning("OpenAI SDK not available. OpenAI features will be disabled.")
                self.status["components"]["openai"] = "unavailable"
                self.openai_client = None
        else:
            self.status["components"]["openai"] = "disabled"
            self.openai_client = None
        
        # Initialize Claude client
        if self.use_claude and os.environ.get("ANTHROPIC_API_KEY"):
            try:
                import anthropic
                self.claude_client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
                logger.info("Claude client initialized")
                self.status["components"]["claude"] = "available"
            except ImportError:
                logger.warning("Anthropic SDK not available. Claude features will be disabled.")
                self.status["components"]["claude"] = "unavailable"
                self.claude_client = None
        else:
            self.status["components"]["claude"] = "disabled"
            self.claude_client = None
    
    async def initialize(self) -> bool:
        """
        Initialize the agent components
        
        Returns:
            Initialization success
        """
        if self.initialized:
            logger.info("Agent already initialized")
            return True
        
        self.status["agent_state"] = "initializing"
        self.status["last_update"] = datetime.now().isoformat()
        
        try:
            # Initialize search integration
            from quantum_agent_framework.search_integration import SearchIntegration
            self.search_integration = SearchIntegration(
                use_quantum=self.use_quantum,
                n_qubits=self.n_qubits
            )
            logger.info("Search integration initialized")
            self.status["components"]["search"] = "available"
            
            # Initialize browser integration
            from quantum_agent_framework.browser_integration import BrowserIntegration
            self.browser_integration = BrowserIntegration()
            await self.browser_integration.initialize(
                headless=False,
                use_quantum=self.use_quantum,
                n_qubits=self.n_qubits
            )
            logger.info("Browser integration initialized")
            self.status["components"]["browser"] = "available"
            
            # Initialize task history
            from quantum_agent_framework.task_history import TaskHistoryManager
            self.task_history = TaskHistoryManager()
            logger.info("Task history initialized")
            self.status["components"]["task_history"] = "available"
            
            # Mark as initialized
            self.initialized = True
            self.status["agent_state"] = "initialized"
            self.status["last_update"] = datetime.now().isoformat()
            
            logger.info("Agent initialization completed successfully")
            return True
            
        except ImportError as e:
            logger.error(f"Initialization failed: {str(e)}")
            self.status["agent_state"] = "initialization_failed"
            self.status["last_update"] = datetime.now().isoformat()
            self.status["error"] = str(e)
            return False
        
        except Exception as e:
            logger.error(f"Initialization failed with unexpected error: {str(e)}")
            self.status["agent_state"] = "initialization_failed"
            self.status["last_update"] = datetime.now().isoformat()
            self.status["error"] = str(e)
            return False
    
    async def start_agent_loop(self) -> bool:
        """
        Start the agent loop for background processing
        
        Returns:
            Success status
        """
        if not self.initialized:
            success = await self.initialize()
            if not success:
                logger.error("Cannot start agent loop: initialization failed")
                return False
        
        if self.running:
            logger.warning("Agent loop already running")
            return True
        
        self.running = True
        self.status["agent_state"] = "running"
        self.status["last_update"] = datetime.now().isoformat()
        
        logger.info("Agent loop started")
        
        # Start background task for agent loop
        asyncio.create_task(self._agent_loop())
        
        return True
    
    async def _agent_loop(self):
        """Background agent loop for continuous processing"""
        logger.info("Agent loop running in background")
        
        try:
            while self.running:
                # Simple keepalive loop
                await asyncio.sleep(5)
                
                # Update status
                self.status["last_update"] = datetime.now().isoformat()
        
        except Exception as e:
            logger.error(f"Error in agent loop: {str(e)}")
            self.running = False
            self.status["agent_state"] = "error"
            self.status["error"] = str(e)
    
    async def stop_agent_loop(self) -> bool:
        """
        Stop the agent loop
        
        Returns:
            Success status
        """
        if not self.running:
            logger.info("Agent loop already stopped")
            return True
        
        self.running = False
        self.status["agent_state"] = "stopped"
        self.status["last_update"] = datetime.now().isoformat()
        
        logger.info("Agent loop stopped")
        return True
    
    async def process_task(self, task: str) -> Dict[str, Any]:
        """
        Process a user task
        
        Args:
            task: Natural language task description
            
        Returns:
            Task result
        """
        if not self.initialized:
            success = await self.initialize()
            if not success:
                return {
                    "success": False,
                    "error": "Agent initialization failed",
                    "task": task
                }
        
        # Track task start time
        start_time = time.time()
        
        logger.info(f"Processing task: {task}")
        
        # Determine task type
        task_type = await self._analyze_task_type(task)
        logger.info(f"Determined task type: {task_type}")
        
        # Process task based on type
        try:
            if task_type == "search":
                result = await self._process_search_task(task)
            elif task_type == "web_task":
                result = await self._process_web_task(task)
            elif task_type == "job_search":
                result = await self._process_job_search_task(task)
            else:
                result = await self._process_general_task(task)
                
            # Add task metadata
            execution_time = time.time() - start_time
            result["task"] = task
            result["task_type"] = task_type
            result["execution_time"] = execution_time
            result["timestamp"] = datetime.now().isoformat()
            result["quantum_enhanced"] = self.use_quantum
            
            # Add to task history
            if self.task_history:
                result["task_id"] = self.task_history.add_task({
                    "task": task,
                    "task_type": task_type,
                    "status": "success" if result.get("success", False) else "failure",
                    "execution_time": execution_time,
                    "quantum_enhanced": self.use_quantum,
                    "result": result
                })
            
            # Update status counters
            self.status["tasks_completed"] += 1
            if task_type == "search":
                self.status["searches_performed"] += 1
            elif task_type == "web_task":
                self.status["web_interactions"] += 1
            
            if result.get("quantum_enhanced", False):
                self.status["quantum_operations"] += 1
            
            # Generate summary if not present
            if "summary" not in result:
                result["summary"] = await self._generate_task_summary(task, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing task: {str(e)}", exc_info=True)
            
            error_result = {
                "success": False,
                "error": str(e),
                "task": task,
                "task_type": task_type,
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add to task history
            if self.task_history:
                error_result["task_id"] = self.task_history.add_task({
                    "task": task,
                    "task_type": task_type,
                    "status": "failure",
                    "execution_time": time.time() - start_time,
                    "error": str(e)
                })
            
            # Generate error summary
            error_result["summary"] = f"I encountered an error while processing your task: {str(e)}"
            
            return error_result
    
    async def _analyze_task_type(self, task: str) -> str:
        """
        Analyze task to determine its type
        
        Args:
            task: Task description
            
        Returns:
            Task type ('search', 'web_task', 'job_search', or 'general')
        """
        task_lower = task.lower()
        
        # Check for explicit task type indicators
        if task_lower.startswith("search:") or task_lower.startswith("deep search:"):
            return "search"
        
        if task_lower.startswith("browse:") or task_lower.startswith("web:"):
            return "web_task"
        
        # Check for search-related keywords
        if any(keyword in task_lower for keyword in [
            "search for", "find information about", "look up", "research",
            "tell me about", "what is", "who is", "what are"
        ]):
            return "search"
        
        # Check for web interaction keywords
        if any(keyword in task_lower for keyword in [
            "go to", "navigate to", "visit", "open website", "browse", "click",
            "fill", "submit", "download", "check website"
        ]):
            return "web_task"
        
        # Check for job search
        if ("job" in task_lower or "career" in task_lower or "position" in task_lower) and (
            "find" in task_lower or "search" in task_lower or "look" in task_lower
        ):
            return "job_search"
        
        # Default to general task
        return "general"
    
    async def _process_search_task(self, task: str) -> Dict[str, Any]:
        """
        Process a search task
        
        Args:
            task: Search task description
            
        Returns:
            Search results
        """
        if not self.search_integration:
            return {
                "success": False,
                "error": "Search integration not available",
                "task": task
            }
        
        # Extract query from task
        query = self._extract_search_query(task)
        
        logger.info(f"Executing search for query: {query}")
        
        # Execute deep search
        search_results = await self.search_integration.search(
            query=query,
            max_results=20,
            deep_search=True
        )
        
        # Generate summary
        if search_results.get("success", True) and not "comprehensive_summary" in search_results:
            summary = await self._generate_search_summary(query, search_results)
            search_results["comprehensive_summary"] = summary
        
        return search_results
    
    async def _process_web_task(self, task: str) -> Dict[str, Any]:
        """
        Process a web interaction task
        
        Args:
            task: Web task description
            
        Returns:
            Task result
        """
        if not self.browser_integration:
            return {
                "success": False,
                "error": "Browser integration not available",
                "task": task
            }
        
        logger.info(f"Executing web task: {task}")
        
        # Process natural language task with browser
        result = await self.browser_integration.process_natural_language_task(task)
        
        return result
    
    async def _process_job_search_task(self, task: str) -> Dict[str, Any]:
        """
        Process a job search task
        
        Args:
            task: Job search task description
            
        Returns:
            Job search results
        """
        if not self.search_integration:
            return {
                "success": False,
                "error": "Search integration not available",
                "task": task
            }
        
        # Extract job search query
        query = self._extract_job_search_query(task)
        
        # Extract companies from task
        companies = self._extract_companies_from_task(task)
        
        logger.info(f"Executing job search for query: {query} (companies: {companies})")
        
        # Execute job search
        job_results = await self.search_integration.get_job_search_results(
            query=query,
            companies=companies
        )
        
        return job_results
    
    async def _process_general_task(self, task: str) -> Dict[str, Any]:
        """
        Process a general task
        
        Args:
            task: General task description
            
        Returns:
            Task result
        """
        # For general tasks, use AI to determine the best course of action
        task_analysis = await self._analyze_task_with_ai(task)
        
        # Based on analysis, choose how to handle the task
        if task_analysis.get("suggested_action") == "search":
            logger.info("Analysis suggests search action for general task")
            return await self._process_search_task(task)
        
        elif task_analysis.get("suggested_action") == "web_task":
            logger.info("Analysis suggests web action for general task")
            return await self._process_web_task(task)
        
        elif task_analysis.get("suggested_action") == "job_search":
            logger.info("Analysis suggests job search for general task")
            return await self._process_job_search_task(task)
        
        # Default response for general tasks
        return {
            "success": True,
            "task": task,
            "task_analysis": task_analysis,
            "summary": task_analysis.get("response", "I've analyzed your request and determined the best way to help.")
        }
    
    def _extract_search_query(self, task: str) -> str:
        """
        Extract search query from task description
        
        Args:
            task: Task description
            
        Returns:
            Search query
        """
        task_lower = task.lower()
        
        # Handle explicit search prefixes
        prefixes = ["search:", "deep search:", "find:", "lookup:"]
        for prefix in prefixes:
            if task_lower.startswith(prefix):
                return task[len(prefix):].strip()
        
        # Handle search phrases
        phrases = ["search for", "find information about", "look up", 
                  "find", "tell me about", "what is", "who is"]
        
        for phrase in phrases:
            if phrase in task_lower:
                # Get text after the phrase
                query = task_lower.split(phrase, 1)[1].strip()
                return query
        
        # Default: use the whole task as query
        return task
    
    def _extract_job_search_query(self, task: str) -> str:
        """
        Extract job search query from task description
        
        Args:
            task: Task description
            
        Returns:
            Job search query
        """
        task_lower = task.lower()
        
        # Remove common job search phrases
        phrases = ["find jobs", "search for jobs", "job search", "find positions",
                  "look for jobs", "career search", "find job openings"]
        
        query = task_lower
        for phrase in phrases:
            if phrase in query:
                query = query.replace(phrase, "")
        
        # Remove company references
        companies = ["microsoft", "google", "amazon", "apple", "facebook", "meta"]
        for company in companies:
            if company in query:
                query = query.replace(company, "")
        
        # Remove common filler words
        fillers = ["at", "for", "in", "related to", "about", "regarding", "concerning"]
        for filler in fillers:
            if f" {filler} " in query:
                query = query.replace(f" {filler} ", " ")
        
        # Clean up and return
        query = query.strip()
        if not query:
            # Default to generic job search if nothing specific found
            query = "software jobs"
        
        return query
    
    def _extract_companies_from_task(self, task: str) -> List[str]:
        """
        Extract mentioned companies from task description
        
        Args:
            task: Task description
            
        Returns:
            List of companies
        """
        task_lower = task.lower()
        companies = []
        
        # Check for common companies
        company_list = ["microsoft", "google", "amazon", "apple", "facebook", "meta", 
                      "twitter", "netflix", "uber", "airbnb", "linkedin", "ibm"]
        
        for company in company_list:
            if company in task_lower:
                companies.append(company)
        
        return companies
    
    async def _analyze_task_with_ai(self, task: str) -> Dict[str, Any]:
        """
        Analyze a task using AI to determine best action
        
        Args:
            task: Task description
            
        Returns:
            Analysis results
        """
        # Try Claude first if available
        if self.claude_client:
            try:
                return await self._analyze_task_with_claude(task)
            except Exception as e:
                logger.warning(f"Claude analysis failed: {str(e)}")
        
        # Fall back to OpenAI if available
        if self.openai_client:
            try:
                return await self._analyze_task_with_openai(task)
            except Exception as e:
                logger.warning(f"OpenAI analysis failed: {str(e)}")
        
        # Simple fallback if AI not available
        return self._analyze_task_simple(task)
    
    async def _analyze_task_with_claude(self, task: str) -> Dict[str, Any]:
        """Analyze task using Claude"""
        system_prompt = """You are an AI assistant that analyzes user requests to determine the best way to help.
        For each request, determine whether it should be handled as:
        1. A search task (looking up information)
        2. A web browser task (navigating to websites, interacting with web pages)
        3. A job search task (finding job listings)
        4. A general conversation task (answering questions directly)
        
        Also provide a brief, helpful response to the user's request.
        
        Respond in JSON format with the following fields:
        {
            "suggested_action": "search" | "web_task" | "job_search" | "conversation",
            "response": "Your helpful response to the user's request",
            "explanation": "Brief explanation of why you chose this action"
        }
        """
        
        user_message = f"Analyze this user request and determine the best way to help: {task}"
        
        response = await self.claude_client.messages.create(
            model="claude-3-opus-20240229",
            system=system_prompt,
            max_tokens=1000,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        
        # Parse JSON response
        try:
            content = response.content[0].text
            # Find JSON object in the content
            import re
            json_match = re.search(r'```json\s*({.*?})\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
            else:
                json_match = re.search(r'{.*}', content, re.DOTALL)
                if json_match:
                    content = json_match.group(0)
            
            analysis = json.loads(content)
            analysis["ai_provider"] = "claude"
            return analysis
        except Exception as e:
            logger.error(f"Error parsing Claude response: {str(e)}")
            # Fallback to simple structure
            return {
                "suggested_action": "search",
                "response": response.content[0].text,
                "explanation": "Fallback response due to parsing error",
                "ai_provider": "claude",
                "error": str(e)
            }
    
    async def _analyze_task_with_openai(self, task: str) -> Dict[str, Any]:
        """Analyze task using OpenAI"""
        system_prompt = """You are an AI assistant that analyzes user requests to determine the best way to help.
        For each request, determine whether it should be handled as:
        1. A search task (looking up information)
        2. A web browser task (navigating to websites, interacting with web pages)
        3. A job search task (finding job listings)
        4. A general conversation task (answering questions directly)
        
        Also provide a brief, helpful response to the user's request.
        
        Respond in JSON format with the following fields:
        {
            "suggested_action": "search" | "web_task" | "job_search" | "conversation",
            "response": "Your helpful response to the user's request",
            "explanation": "Brief explanation of why you chose this action"
        }
        """
        
        user_message = f"Analyze this user request and determine the best way to help: {task}"
        
        response = await self.openai_client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=1000
        )
        
        # Parse JSON response
        try:
            content = response.choices[0].message.content
            analysis = json.loads(content)
            analysis["ai_provider"] = "openai"
            return analysis
        except Exception as e:
            logger.error(f"Error parsing OpenAI response: {str(e)}")
            # Fallback to simple structure
            return {
                "suggested_action": "search",
                "response": response.choices[0].message.content,
                "explanation": "Fallback response due to parsing error",
                "ai_provider": "openai",
                "error": str(e)
            }
    
    def _analyze_task_simple(self, task: str) -> Dict[str, Any]:
        """Simple task analysis without AI"""
        task_lower = task.lower()
        
        # Check for search indicators
        if any(term in task_lower for term in ["what", "who", "how", "why", "where", "when", "find", "search"]):
            return {
                "suggested_action": "search",
                "response": f"I'll search for information about '{task}'",
                "explanation": "This appears to be an information-seeking query"
            }
        
        # Check for web indicators
        if any(term in task_lower for term in ["website", "browse", "go to", "visit", "open", "navigate"]):
            return {
                "suggested_action": "web_task",
                "response": f"I'll browse the web to help with '{task}'",
                "explanation": "This appears to be a web navigation or interaction request"
            }
        
        # Check for job search
        if any(term in task_lower for term in ["job", "career", "position", "work", "employment"]):
            return {
                "suggested_action": "job_search",
                "response": f"I'll look for job opportunities related to '{task}'",
                "explanation": "This appears to be a job search request"
            }
        
        # Default to search
        return {
            "suggested_action": "search",
            "response": f"I'll find information related to '{task}'",
            "explanation": "Using search as the default action for this request"
        }
    
    async def _generate_search_summary(self, query: str, results: Dict[str, Any]) -> str:
        """
        Generate a summary of search results
        
        Args:
            query: Search query
            results: Search results
            
        Returns:
            Summary text
        """
        # If results already have a comprehensive summary, use it
        if "comprehensive_summary" in results:
            return results["comprehensive_summary"]
        
        # Check if we have search results
        if not results.get("success", False) or "search_results" not in results:
            return f"I couldn't find any relevant information about '{query}'."
        
        # Try with Claude
        if self.claude_client:
            try:
                return await self._generate_search_summary_with_claude(query, results)
            except Exception as e:
                logger.warning(f"Claude summary generation failed: {str(e)}")
        
        # Try with OpenAI
        if self.openai_client:
            try:
                return await self._generate_search_summary_with_openai(query, results)
            except Exception as e:
                logger.warning(f"OpenAI summary generation failed: {str(e)}")
        
        # Fallback to simple summary
        return self._generate_simple_search_summary(query, results)
    
    async def _generate_search_summary_with_claude(self, query: str, results: Dict[str, Any]) -> str:
        """Generate search summary using Claude"""
        search_results = results.get("search_results", [])
        results_str = json.dumps(search_results[:5], indent=2)  # Limit to first 5 results
        
        system_prompt = """You are a professional research assistant skilled at summarizing search results.
        Create a comprehensive, accurate summary that answers the query using only the provided search results.
        
        Guidelines:
        1. Include ONLY information directly supported by the search results.
        2. Cite sources using [1], [2], etc. corresponding to the result numbers.
        3. Be factual, precise, and concise.
        4. Structure information logically with clear headings.
        5. If results have conflicting information, acknowledge different perspectives.
        6. Highlight the most important and relevant findings first.
        
        Begin with a direct answer to the query, then provide supporting details.
        Do not include disclaimers about the limitations of your knowledge or the search results.
        """
        
        user_message = f"Query: {query}\n\nSearch Results:\n{results_str}\n\nPlease provide a comprehensive summary that answers the query using only the information in these search results with proper citations."
        
        response = await self.claude_client.messages.create(
            model="claude-3-opus-20240229",
            system=system_prompt,
            max_tokens=1500,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        
        return response.content[0].text
    
    async def _generate_search_summary_with_openai(self, query: str, results: Dict[str, Any]) -> str:
        """Generate search summary using OpenAI"""
        search_results = results.get("search_results", [])
        results_str = json.dumps(search_results[:5], indent=2)  # Limit to first 5 results
        
        system_prompt = """You are a professional research assistant skilled at summarizing search results.
        Create a comprehensive, accurate summary that answers the query using only the provided search results.
        
        Guidelines:
        1. Include ONLY information directly supported by the search results.
        2. Cite sources using [1], [2], etc. corresponding to the result numbers.
        3. Be factual, precise, and concise.
        4. Structure information logically with clear headings.
        5. If results have conflicting information, acknowledge different perspectives.
        6. Highlight the most important and relevant findings first.
        
        Begin with a direct answer to the query, then provide supporting details.
        Do not include disclaimers about the limitations of your knowledge or the search results.
        """
        
        user_message = f"Query: {query}\n\nSearch Results:\n{results_str}\n\nPlease provide a comprehensive summary that answers the query using only the information in these search results with proper citations."
        
        response = await self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=1500
        )
        
        return response.choices[0].message.content
    
    def _generate_simple_search_summary(self, query: str, results: Dict[str, Any]) -> str:
        """Generate a simple search summary"""
        search_results = results.get("search_results", [])
        
        if not search_results:
            return f"I couldn't find any relevant information about '{query}'."
        
        summary = [f"# Search Results for: {query}"]
        
        # Add result count and method
        result_count = len(search_results)
        method = "quantum-enhanced" if results.get("quantum_enhanced", False) else "classical"
        summary.append(f"\nFound {result_count} results using {method} search.\n")
        
        # Add top results
        summary.append("## Top Results\n")
        
        for i, result in enumerate(search_results[:3]):  # Top 3 results
            title = result.get("title", "Untitled")
            source = result.get("source", "Unknown source")
            snippet = result.get("snippet", "No description available")
            
            summary.append(f"**[{i+1}] {title}**")
            summary.append(f"Source: {source}")
            summary.append(f"{snippet[:200]}..." if len(snippet) > 200 else snippet)
            summary.append("")
        
        # Add conclusion
        summary.append("\n## Summary")
        summary.append(f"The search results provide information about {query}. Multiple sources were found with varying levels of relevance.")
        
        return "\n".join(summary)
    
    async def _generate_task_summary(self, task: str, result: Dict[str, Any]) -> str:
        """
        Generate a summary of task results
        
        Args:
            task: Task description
            result: Task result
            
        Returns:
            Summary text
        """
        # If the result already has a summary, use it
        if "summary" in result:
            return result["summary"]
        
        task_type = result.get("task_type", "general")
        
        # Generate different summaries based on task type
        if task_type == "search":
            if "comprehensive_summary" in result:
                return result["comprehensive_summary"]
            else:
                return await self._generate_search_summary(task, result)
        
        elif task_type == "web_task":
            return self._generate_web_task_summary(task, result)
        
        elif task_type == "job_search":
            return self._generate_job_search_summary(task, result)
        
        else:
            # General task
            return f"I've processed your request: '{task}'. " + \
                  (f"The task was completed successfully." if result.get("success", False) else 
                   f"I encountered an issue: {result.get('error', 'Unknown error')}")
    
    def _generate_web_task_summary(self, task: str, result: Dict[str, Any]) -> str:
        """Generate summary for web task"""
        if not result.get("success", False):
            return f"I was unable to complete the web task: '{task}'. The error was: {result.get('error', 'Unknown error')}"
        
        if "url" in result:
            summary = f"I successfully browsed to {result['url']}"
            if "title" in result:
                summary += f" (Page title: {result['title']})"
            summary += "."
        elif "element" in result:
            summary = f"I successfully clicked on '{result['element']}'."
        else:
            summary = f"I successfully completed the web task: '{task}'."
        
        # Add analysis if available
        if "analysis" in result and isinstance(result["analysis"], dict):
            analysis = result["analysis"]
            if "key_topics" in analysis and analysis["key_topics"]:
                topics = ", ".join(analysis["key_topics"][:3])
                summary += f" The page contains information about: {topics}."
            
            if "suggested_actions" in analysis and analysis["suggested_actions"]:
                next_action = analysis["suggested_actions"][0]["description"]
                summary += f" You might want to {next_action.lower()}."
        
        return summary
    
    def _generate_job_search_summary(self, task: str, result: Dict[str, Any]) -> str:
        """Generate summary for job search task"""
        if not result.get("success", False):
            return f"I was unable to complete the job search: '{task}'. The error was: {result.get('error', 'Unknown error')}"
        
        job_results = result.get("job_listings", result.get("search_results", []))
        count = len(job_results)
        
        if count == 0:
            return f"I searched for jobs related to '{task}' but didn't find any matching positions."
        
        company = result.get("company", "various companies")
        
        summary = [f"I found {count} job opportunities related to '{task}' at {company}:"]
        
        # Add top jobs
        for i, job in enumerate(job_results[:3]):  # Top 3 jobs
            title = job.get("title", "Unnamed position")
            employer = job.get("company", company)
            location = job.get("location", "Unknown location")
            
            summary.append(f"\n**{i+1}. {title}**")
            summary.append(f"* Employer: {employer}")
            summary.append(f"* Location: {location}")
            
            if "description" in job:
                desc = job["description"]
                summary.append(f"* {desc[:100]}..." if len(desc) > 100 else f"* {desc}")
        
        if count > 3:
            summary.append(f"\nThere are {count - 3} more job listings available.")
        
        return "\n".join(summary)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the agent
        
        Returns:
            Status information
        """
        # Update component statuses
        if self.search_integration:
            self.status["components"]["search"] = self.search_integration.get_status()
        
        if self.task_history:
            self.status["components"]["task_history"] = self.task_history.get_metrics()
        
        return self.status
    
    def get_task_history(self) -> List[Dict[str, Any]]:
        """
        Get task history
        
        Returns:
            List of tasks with results
        """
        if self.task_history:
            return self.task_history.get_tasks(limit=50)
        return []
    
    async def close(self):
        """Close the agent and release resources"""
        if self.running:
            await self.stop_agent_loop()
        
        if self.browser_integration:
            await self.browser_integration.close()
        
        self.initialized = False
        logger.info("Agent closed and resources released")