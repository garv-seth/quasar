"""
QUASAR Framework: AI Helpers Module
Provides AI integration capabilities for the QUASAR framework
"""

import os
import logging
import time
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for Anthropic's Claude API availability
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
    logger.info("Anthropic's Claude API available.")
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic's Claude API not available. Some AI features will be limited.")

# Check for OpenAI API availability
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    logger.info("OpenAI API available.")
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI API not available. Some AI features will be limited.")

class AIEngine:
    """AI Engine for the QUASAR framework using Claude or alternatives"""
    
    def __init__(self, 
                 use_claude: bool = False,  # Default to False since we're having issues
                 use_openai: bool = True,
                 claude_model: str = "claude-3-opus-20240229",  # Updated to latest model
                 openai_model: str = "gpt-4o"):
        """
        Initialize the AI Engine.
        
        Args:
            use_claude: Whether to use Claude (if available)
            use_openai: Whether to use OpenAI (if available and Claude not available)
            claude_model: Claude model to use
            openai_model: OpenAI model to use
        """
        # Check if API keys are available and valid
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        
        self.use_claude = use_claude and ANTHROPIC_AVAILABLE and anthropic_key and anthropic_key != "sk-..."
        self.use_openai = use_openai and OPENAI_AVAILABLE and openai_key and openai_key != "sk-..."
        
        self.claude_model = claude_model
        self.openai_model = openai_model
        
        self.clients = {}
        
        if self.use_claude:
            try:
                self.clients["claude"] = Anthropic(api_key=anthropic_key)
                logger.info(f"Initialized Claude client with model {self.claude_model}")
            except Exception as e:
                logger.error(f"Error initializing Claude client: {e}")
                self.use_claude = False
        
        if self.use_openai:
            try:
                self.clients["openai"] = OpenAI(api_key=openai_key)
                logger.info(f"Initialized OpenAI client with model {self.openai_model}")
            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {e}")
                self.use_openai = False
    
    async def analyze_task(self, task: str) -> Dict[str, Any]:
        """
        Analyze a task to determine the best processing approach.
        
        Args:
            task: The task description
            
        Returns:
            Dict containing analysis results
        """
        start_time = time.time()
        
        # Prepare prompt
        prompt = f"""
        Analyze the following computational task and determine:
        1. Task category (search, factorization, optimization, general)
        2. Whether quantum computing would provide an advantage
        3. Specific parameters needed for processing
        4. Complexity estimate
        
        Task: {task}
        
        Provide your analysis in JSON format with the following keys:
        - task_type: string (search/factorization/optimization/general)
        - quantum_advantage: boolean
        - reason: string (brief explanation)
        - parameters: object (task-specific parameters)
        - complexity_estimate: string
        """
        
        result = None
        
        # Try Claude first if available
        if self.use_claude:
            try:
                response = self.clients["claude"].messages.create(
                    model=self.claude_model,
                    max_tokens=1000,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                result = {
                    "analysis": response.content[0].text,
                    "model": f"Claude ({self.claude_model})",
                    "response_time": time.time() - start_time
                }
                logger.info(f"Task analyzed using Claude in {result['response_time']:.2f}s")
            except Exception as e:
                logger.error(f"Error using Claude for task analysis: {e}")
        
        # Fall back to OpenAI if Claude failed or not available
        if result is None and self.use_openai:
            try:
                response = self.clients["openai"].chat.completions.create(
                    model=self.openai_model,
                    max_tokens=1000,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                result = {
                    "analysis": response.choices[0].message.content,
                    "model": f"OpenAI ({self.openai_model})",
                    "response_time": time.time() - start_time
                }
                logger.info(f"Task analyzed using OpenAI in {result['response_time']:.2f}s")
            except Exception as e:
                logger.error(f"Error using OpenAI for task analysis: {e}")
        
        # If both failed, provide a basic analysis
        if result is None:
            result = {
                "analysis": {
                    "task_type": "general",
                    "quantum_advantage": False,
                    "reason": "Basic analysis only - AI services unavailable",
                    "parameters": {},
                    "complexity_estimate": "unknown"
                },
                "model": "basic rule-based analysis",
                "response_time": time.time() - start_time
            }
            logger.warning("Using basic rule-based task analysis due to AI service unavailability")
        
        return result
    
    async def explain_quantum_advantage(self, 
                                       task_type: str, 
                                       problem_size: int,
                                       speedup: float,
                                       quantum_time: float,
                                       classical_time: float) -> str:
        """
        Generate an explanation of the quantum advantage for a task.
        
        Args:
            task_type: Type of task (search, factorization, optimization)
            problem_size: Size of the problem
            speedup: Speedup factor achieved
            quantum_time: Time taken with quantum processing
            classical_time: Time taken with classical processing
            
        Returns:
            Explanation string
        """
        start_time = time.time()
        
        # Prepare prompt
        prompt = f"""
        Generate a clear, educational explanation of the quantum advantage observed in this computation:
        
        - Task type: {task_type}
        - Problem size: {problem_size}
        - Observed speedup: {speedup:.2f}x
        - Quantum processing time: {quantum_time:.6f} seconds
        - Classical processing time: {classical_time:.6f} seconds
        
        Your explanation should:
        1. Explain the relevant quantum algorithm (Grover's for search, Shor's for factorization, QAOA for optimization)
        2. Compare the asymptotic complexity (Big O notation) of quantum vs classical approaches
        3. Clarify how the advantage scales with problem size
        4. Use accessible language that a technical but non-quantum-specialist could understand
        
        Limit your response to 3-4 paragraphs.
        """
        
        result = None
        
        # Try Claude first if available
        if self.use_claude:
            try:
                response = self.clients["claude"].messages.create(
                    model=self.claude_model,
                    max_tokens=1000,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                result = response.content[0].text
                logger.info(f"Quantum advantage explanation generated using Claude in {time.time() - start_time:.2f}s")
            except Exception as e:
                logger.error(f"Error using Claude for explanation generation: {e}")
        
        # Fall back to OpenAI if Claude failed or not available
        if result is None and self.use_openai:
            try:
                response = self.clients["openai"].chat.completions.create(
                    model=self.openai_model,
                    max_tokens=1000,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                result = response.choices[0].message.content
                logger.info(f"Quantum advantage explanation generated using OpenAI in {time.time() - start_time:.2f}s")
            except Exception as e:
                logger.error(f"Error using OpenAI for explanation generation: {e}")
        
        # If both failed, provide a basic explanation
        if result is None:
            if task_type == "search":
                result = (
                    f"Quantum search using Grover's algorithm achieved a {speedup:.2f}x speedup over classical search. "
                    f"This demonstrates the quadratic advantage (O(√N) vs O(N)) that quantum computing can provide for unstructured search problems. "
                    f"As the database size increases, this advantage becomes more significant."
                )
            elif task_type == "factorization":
                result = (
                    f"Quantum factorization using Shor's algorithm achieved a {speedup:.2f}x speedup over classical factorization. "
                    f"This demonstrates the exponential advantage that quantum computing can provide for integer factorization. "
                    f"As the bit length of numbers increases, this advantage becomes dramatically more significant."
                )
            elif task_type == "optimization":
                result = (
                    f"Quantum optimization using QAOA achieved a {speedup:.2f}x speedup over classical optimization. "
                    f"This demonstrates the polynomial advantage that quantum computing can provide for certain NP-hard optimization problems. "
                    f"As the problem size increases, this advantage becomes more significant."
                )
            else:
                result = (
                    f"Quantum computing achieved a {speedup:.2f}x speedup over classical computing for this {task_type} task. "
                    f"The quantum approach took {quantum_time:.6f}s compared to {classical_time:.6f}s for the classical approach."
                )
            logger.warning("Using template-based explanation due to AI service unavailability")
        
        return result
    
    async def summarize_results(self, task_type: str, results: Dict[str, Any]) -> str:
        """
        Generate a summary of task results.
        
        Args:
            task_type: Type of task
            results: Task results
            
        Returns:
            Summary string
        """
        start_time = time.time()
        
        # Prepare prompt based on task type
        prompt = f"Summarize the following {task_type} results in 2-3 sentences, highlighting the key findings:\n\n"
        prompt += str(results)
        
        result = None
        
        # Try Claude first if available
        if self.use_claude:
            try:
                response = self.clients["claude"].messages.create(
                    model=self.claude_model,
                    max_tokens=300,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                result = response.content[0].text
                logger.info(f"Results summary generated using Claude in {time.time() - start_time:.2f}s")
            except Exception as e:
                logger.error(f"Error using Claude for summary generation: {e}")
        
        # Fall back to OpenAI if Claude failed or not available
        if result is None and self.use_openai:
            try:
                response = self.clients["openai"].chat.completions.create(
                    model=self.openai_model,
                    max_tokens=300,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                result = response.choices[0].message.content
                logger.info(f"Results summary generated using OpenAI in {time.time() - start_time:.2f}s")
            except Exception as e:
                logger.error(f"Error using OpenAI for summary generation: {e}")
        
        # If both failed, provide a basic summary
        if result is None:
            if task_type == "search":
                result = f"Found {len(results.get('results', []))} results for the query '{results.get('query', 'unknown')}' with {results.get('speedup', 1.0):.2f}x quantum speedup."
            elif task_type == "factorization":
                number = results.get("number", 0)
                factors = results.get("factors", [])
                result = f"Factorized {number} into {len(factors)} factors with {results.get('speedup', 1.0):.2f}x quantum speedup."
            elif task_type == "optimization":
                result = f"Optimized a {results.get('problem_type', 'unknown')} problem with {results.get('speedup', 1.0):.2f}x quantum speedup."
            else:
                result = f"Completed task with {results.get('speedup', 1.0):.2f}x quantum speedup."
            logger.warning("Using template-based summary due to AI service unavailability")
        
        return result
    
    def check_api_keys(self) -> Dict[str, bool]:
        """
        Check if API keys are available.
        
        Returns:
            Dict with key availability status
        """
        return {
            "claude": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "openai": bool(os.environ.get("OPENAI_API_KEY"))
        }