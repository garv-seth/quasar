"""
QA³ Agent Core: True Autonomous Web Interaction Module

This module provides the core functionality for true autonomous web browsing and
interaction capabilities for the Quantum Accelerated Agent Architecture.
"""

import os
import time
import json
import logging
import asyncio
import traceback
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from urllib.parse import urlparse, urljoin

# Browser interaction libraries
import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Page, Browser, BrowserContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qa3-agent-core")


class AgentMemory:
    """Memory system for the autonomous agent with short and long-term storage"""
    
    def __init__(self, max_short_term_items: int = 20):
        """
        Initialize agent memory systems
        
        Args:
            max_short_term_items: Maximum items in short-term memory
        """
        self.short_term = []  # Recent interactions and context
        self.long_term = []   # Important information to retain
        self.max_short_term_items = max_short_term_items
        self.working_memory = {}  # Current task state
        self.web_history = []  # History of web interactions
        
    def add_interaction(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add an interaction to short-term memory with proper timestamp"""
        interaction = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.short_term.append(interaction)
        
        # Trim if exceeding max size
        if len(self.short_term) > self.max_short_term_items:
            self.short_term = self.short_term[-self.max_short_term_items:]
            
    def add_to_working_memory(self, key: str, value: Any):
        """Store information in working memory"""
        self.working_memory[key] = value
        
    def get_from_working_memory(self, key: str, default: Any = None) -> Any:
        """Retrieve information from working memory"""
        return self.working_memory.get(key, default)
        
    def add_to_long_term(self, item: Dict[str, Any]):
        """Store important information in long-term memory"""
        if "timestamp" not in item:
            item["timestamp"] = datetime.now().isoformat()
        self.long_term.append(item)
        
    def add_web_visit(self, url: str, title: str, content_summary: str, metadata: Optional[Dict[str, Any]] = None):
        """Record web browsing history with metadata"""
        web_item = {
            "url": url,
            "title": title,
            "content_summary": content_summary,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.web_history.append(web_item)
        
    def get_conversation_context(self, include_web_history: bool = True, max_items: int = 10) -> str:
        """Generate a context summary for decision making"""
        context = "Short-term memory:\n"
        
        for item in self.short_term[-max_items:]:
            context += f"- {item['role']}: {item['content'][:200]}...\n"
            
        if include_web_history and self.web_history:
            context += "\nRecent web browsing:\n"
            for item in self.web_history[-3:]:
                context += f"- Visited: {item['title']} ({item['url']})\n"
                
        return context
        
    def clear_working_memory(self):
        """Clear working memory for new tasks"""
        self.working_memory = {}
        
    def get_relevant_memories(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Simple retrieval of relevant memories based on keyword matching"""
        results = []
        query_terms = query.lower().split()
        
        # Search in long-term memory
        for memory in self.long_term:
            score = 0
            memory_text = memory.get("content", "").lower()
            
            for term in query_terms:
                if term in memory_text:
                    score += 1
                    
            if score > 0:
                results.append({"memory": memory, "relevance": score})
                
        # Sort by relevance and return limited results
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return [item["memory"] for item in results[:max_results]]


class WebInteractionAgent:
    """
    Handles autonomous web browsing and interactions
    
    This is the core component that enables true agency by allowing the AI
    to interact with web interfaces, execute searches, fill forms, and
    extract information directly from web pages.
    """
    
    def __init__(self):
        """Initialize the web interaction agent"""
        self.browser = None
        self.context = None
        self.page = None
        self.playwright = None
        
        # Store browsing state
        self.current_url = None
        self.current_page_content = None
        self.current_page_title = None
        
        # Metrics
        self.total_interactions = 0
        self.successful_interactions = 0
        self.navigation_history = []
        
    async def initialize(self, headless: bool = True):
        """Initialize browser with Playwright for automation"""
        if self.browser is not None:
            return  # Already initialized
            
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=headless,
                args=["--no-sandbox", "--disable-setuid-sandbox"]
            )
            self.context = await self.browser.new_context(
                viewport={"width": 1280, "height": 800},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
            )
            self.page = await self.context.new_page()
            
            # Set up event handlers
            self.page.on("console", lambda msg: logger.debug(f"Browser console: {msg.text}"))
            logger.info("Web interaction agent initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize browser: {str(e)}")
            traceback.print_exc()
            return False
            
    async def close(self):
        """Close browser and clean up resources"""
        try:
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
                
            self.browser = None
            self.context = None
            self.page = None
            self.playwright = None
            logger.info("Browser resources released")
        except Exception as e:
            logger.error(f"Error closing browser: {str(e)}")
            
    async def navigate(self, url: str) -> Dict[str, Any]:
        """Navigate to a URL and extract page information"""
        if self.page is None:
            await self.initialize()
            
        self.total_interactions += 1
        start_time = time.time()
        
        try:
            # Make sure URL has proper scheme
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                
            # Navigate to URL with timeout
            response = await self.page.goto(url, timeout=30000, wait_until="networkidle")
            
            # Extract page information
            self.current_url = self.page.url
            self.current_page_title = await self.page.title()
            self.current_page_content = await self.page.content()
            
            # Extract main content
            content_text = await self.page.evaluate("""() => {
                // Simple content extraction targeting main content elements
                const elements = Array.from(document.querySelectorAll('article, main, #content, .content, .main, [role="main"]'));
                if (elements.length > 0) {
                    return elements.map(el => el.innerText).join('\\n\\n');
                }
                
                // Fallback to paragraphs and headings
                const textElements = Array.from(document.querySelectorAll('p, h1, h2, h3, h4, h5, h6'));
                return textElements.map(el => el.innerText).join('\\n\\n');
            }""")
            
            # Extract links
            links = await self.page.evaluate("""() => {
                return Array.from(document.querySelectorAll('a[href]'))
                    .filter(a => a.href && a.href.startsWith('http') && a.innerText.trim())
                    .map(a => ({ url: a.href, text: a.innerText.trim() }))
                    .slice(0, 20); // Limit to 20 links
            }""")
            
            # Record in navigation history
            navigation_record = {
                "url": self.current_url,
                "title": self.current_page_title,
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            self.navigation_history.append(navigation_record)
            
            # Update metrics
            self.successful_interactions += 1
            execution_time = time.time() - start_time
            
            # Return comprehensive page data
            return {
                "url": self.current_url,
                "title": self.current_page_title,
                "content_text": content_text[:10000],  # Limit size
                "links": links,
                "status_code": response.status if response else None,
                "execution_time": execution_time,
                "success": True
            }
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Navigation error for {url}: {str(e)}")
            
            error_record = {
                "url": url,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "success": False
            }
            self.navigation_history.append(error_record)
            
            return {
                "url": url,
                "error": str(e),
                "execution_time": execution_time,
                "success": False
            }
            
    async def search(self, query: str, search_engine: str = "google") -> Dict[str, Any]:
        """Perform a web search using a search engine"""
        if search_engine.lower() == "google":
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        elif search_engine.lower() == "bing":
            search_url = f"https://www.bing.com/search?q={query.replace(' ', '+')}"
        else:
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            
        # Navigate to search page
        result = await self.navigate(search_url)
        
        if not result["success"]:
            return result
            
        # Extract search results
        try:
            search_results = await self.page.evaluate("""() => {
                // This is a simplified extraction and will need to be adapted based on the search engine
                const results = [];
                
                // For Google
                const elements = document.querySelectorAll('div.g');
                elements.forEach(el => {
                    const titleEl = el.querySelector('h3');
                    const linkEl = el.querySelector('a');
                    const snippetEl = el.querySelector('div.VwiC3b, span.st');
                    
                    if (titleEl && linkEl && linkEl.href) {
                        results.push({
                            title: titleEl.innerText,
                            url: linkEl.href,
                            snippet: snippetEl ? snippetEl.innerText : ''
                        });
                    }
                });
                
                // If no results found with that selector, try a more generic approach
                if (results.length === 0) {
                    document.querySelectorAll('a[href^="http"]').forEach(a => {
                        if (a.innerText.trim() && !a.href.includes('google.com')) {
                            const snippet = a.closest('div') ? a.closest('div').innerText : '';
                            results.push({
                                title: a.innerText.trim(),
                                url: a.href,
                                snippet: snippet
                            });
                        }
                    });
                }
                
                return results.slice(0, 10); // Return top 10 results
            }""")
            
            return {
                "query": query,
                "search_engine": search_engine,
                "results": search_results,
                "url": result["url"],
                "execution_time": result["execution_time"],
                "success": True
            }
        except Exception as e:
            logger.error(f"Error extracting search results: {str(e)}")
            return {
                "query": query,
                "search_engine": search_engine,
                "error": str(e),
                "url": result["url"],
                "execution_time": result["execution_time"],
                "success": False
            }
            
    async def click_element(self, selector: str) -> Dict[str, Any]:
        """Click on an element on the current page"""
        if self.page is None:
            return {"success": False, "error": "Browser not initialized"}
            
        self.total_interactions += 1
        start_time = time.time()
        
        try:
            # Wait for the element to be available
            await self.page.wait_for_selector(selector, timeout=5000)
            
            # Click the element
            await self.page.click(selector)
            
            # Wait for navigation if it occurs
            await self.page.wait_for_load_state("networkidle")
            
            # Get updated page information
            new_url = self.page.url
            new_title = await self.page.title()
            
            # Update state
            self.current_url = new_url
            self.current_page_title = new_title
            
            # Update metrics
            self.successful_interactions += 1
            execution_time = time.time() - start_time
            
            return {
                "action": "click",
                "selector": selector,
                "previous_url": self.current_url,
                "new_url": new_url,
                "new_title": new_title,
                "execution_time": execution_time,
                "success": True
            }
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed to click element {selector}: {str(e)}")
            
            return {
                "action": "click",
                "selector": selector,
                "error": str(e),
                "execution_time": execution_time,
                "success": False
            }
            
    async def fill_form(self, selector: str, value: str) -> Dict[str, Any]:
        """Fill a form field on the current page"""
        if self.page is None:
            return {"success": False, "error": "Browser not initialized"}
            
        self.total_interactions += 1
        start_time = time.time()
        
        try:
            # Wait for the element to be available
            await self.page.wait_for_selector(selector, timeout=5000)
            
            # Fill the form field
            await self.page.fill(selector, value)
            
            # Update metrics
            self.successful_interactions += 1
            execution_time = time.time() - start_time
            
            return {
                "action": "fill",
                "selector": selector,
                "value": value,
                "execution_time": execution_time,
                "success": True
            }
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed to fill form field {selector}: {str(e)}")
            
            return {
                "action": "fill",
                "selector": selector,
                "value": value,
                "error": str(e),
                "execution_time": execution_time,
                "success": False
            }
            
    async def submit_form(self, form_selector: str) -> Dict[str, Any]:
        """Submit a form on the current page"""
        if self.page is None:
            return {"success": False, "error": "Browser not initialized"}
            
        self.total_interactions += 1
        start_time = time.time()
        
        try:
            # Submit the form
            await self.page.evaluate(f"""() => {{
                const form = document.querySelector('{form_selector}');
                if (form) {{
                    form.submit();
                    return true;
                }}
                return false;
            }}""")
            
            # Wait for navigation to complete
            await self.page.wait_for_load_state("networkidle")
            
            # Get updated page information
            new_url = self.page.url
            new_title = await self.page.title()
            
            # Update state
            self.current_url = new_url
            self.current_page_title = new_title
            
            # Update metrics
            self.successful_interactions += 1
            execution_time = time.time() - start_time
            
            return {
                "action": "submit_form",
                "form_selector": form_selector,
                "new_url": new_url,
                "new_title": new_title,
                "execution_time": execution_time,
                "success": True
            }
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed to submit form {form_selector}: {str(e)}")
            
            return {
                "action": "submit_form",
                "form_selector": form_selector,
                "error": str(e),
                "execution_time": execution_time,
                "success": False
            }
            
    async def extract_structured_data(self) -> Dict[str, Any]:
        """Extract structured data from the current page (tables, lists, metadata)"""
        if self.page is None:
            return {"success": False, "error": "Browser not initialized"}
            
        try:
            # Extract tables
            tables = await self.page.evaluate("""() => {
                const tables = [];
                document.querySelectorAll('table').forEach((table, tableIndex) => {
                    const headers = [];
                    const headerRow = table.querySelector('thead tr, tr:first-child');
                    if (headerRow) {
                        headerRow.querySelectorAll('th, td').forEach(cell => {
                            headers.push(cell.innerText.trim());
                        });
                    }
                    
                    const rows = [];
                    table.querySelectorAll('tbody tr, tr:not(:first-child)').forEach(row => {
                        const cells = [];
                        row.querySelectorAll('td').forEach(cell => {
                            cells.push(cell.innerText.trim());
                        });
                        if (cells.length > 0) {
                            rows.push(cells);
                        }
                    });
                    
                    tables.push({
                        id: tableIndex,
                        headers: headers,
                        rows: rows
                    });
                });
                return tables;
            }""")
            
            # Extract lists
            lists = await self.page.evaluate("""() => {
                const lists = [];
                document.querySelectorAll('ul, ol').forEach((list, listIndex) => {
                    const items = [];
                    list.querySelectorAll('li').forEach(item => {
                        items.push(item.innerText.trim());
                    });
                    
                    if (items.length > 0) {
                        lists.push({
                            id: listIndex,
                            type: list.tagName.toLowerCase(),
                            items: items
                        });
                    }
                });
                return lists;
            }""")
            
            # Extract metadata
            metadata = await self.page.evaluate("""() => {
                const meta = {};
                
                // Title and description
                meta.title = document.title;
                
                const descriptionTag = document.querySelector('meta[name="description"]');
                meta.description = descriptionTag ? descriptionTag.getAttribute('content') : '';
                
                // Open Graph metadata
                document.querySelectorAll('meta[property^="og:"]').forEach(tag => {
                    const property = tag.getAttribute('property');
                    meta[property] = tag.getAttribute('content');
                });
                
                return meta;
            }""")
            
            return {
                "tables": tables,
                "lists": lists,
                "metadata": metadata,
                "url": self.current_url,
                "success": True
            }
        except Exception as e:
            logger.error(f"Failed to extract structured data: {str(e)}")
            return {
                "error": str(e),
                "url": self.current_url,
                "success": False
            }
            
    async def take_screenshot(self) -> Dict[str, Any]:
        """Take a screenshot of the current page"""
        if self.page is None:
            return {"success": False, "error": "Browser not initialized"}
            
        try:
            # Create a timestamp-based filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            
            # Take the screenshot
            await self.page.screenshot(path=filename)
            
            return {
                "action": "screenshot",
                "filename": filename,
                "url": self.current_url,
                "success": True
            }
        except Exception as e:
            logger.error(f"Failed to take screenshot: {str(e)}")
            return {
                "action": "screenshot",
                "error": str(e),
                "success": False
            }
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get web interaction metrics"""
        success_rate = (self.successful_interactions / self.total_interactions * 100) if self.total_interactions > 0 else 0
        
        return {
            "total_interactions": self.total_interactions,
            "successful_interactions": self.successful_interactions,
            "success_rate": success_rate,
            "navigation_history_count": len(self.navigation_history),
            "is_browser_active": self.browser is not None
        }


class AutomationTool:
    """
    A tool that can be executed by the agent to perform specific actions
    """
    
    def __init__(self, name: str, description: str, func):
        self.name = name
        self.description = description
        self.func = func
        self.metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_execution_time": 0
        }
        
    async def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """Execute the tool function with metrics tracking"""
        start_time = time.time()
        self.metrics["total_calls"] += 1
        
        try:
            if asyncio.iscoroutinefunction(self.func):
                result = await self.func(*args, **kwargs)
            else:
                result = self.func(*args, **kwargs)
                
            execution_time = time.time() - start_time
            self.metrics["successful_calls"] += 1
            self.metrics["total_execution_time"] += execution_time
            
            return {
                "tool": self.name,
                "result": result,
                "execution_time": execution_time,
                "success": True
            }
        except Exception as e:
            execution_time = time.time() - start_time
            self.metrics["failed_calls"] += 1
            self.metrics["total_execution_time"] += execution_time
            
            logger.error(f"Tool {self.name} execution failed: {str(e)}")
            traceback.print_exc()
            
            return {
                "tool": self.name,
                "error": str(e),
                "execution_time": execution_time,
                "success": False
            }


class AutonomousAgent:
    """
    Complete autonomous agent system tying together all components
    
    This agent is capable of:
    1. Autonomous web browsing and data extraction
    2. Complex decision making based on observations
    3. Memory management for maintaining context
    4. Tool usage for specialized operations
    """
    
    def __init__(self, 
                use_quantum: bool = True,
                n_qubits: int = 8,
                use_claude: bool = True,
                use_browser: bool = True):
        """Initialize the autonomous agent with all its components"""
        self.memory = AgentMemory()
        self.web_agent = WebInteractionAgent() if use_browser else None
        self.tools = {}
        self.quantum_processor = None  # Will be initialized later
        self.ai_processor = None  # Will be initialized later
        
        # Configuration
        self.use_quantum = use_quantum
        self.n_qubits = n_qubits
        self.use_claude = use_claude
        
        # Metrics
        self.start_time = time.time()
        self.total_tasks = 0
        self.completed_tasks = 0
        self.task_history = []
        
    async def initialize(self):
        """Initialize all agent components"""
        # Initialize web interaction capabilities if enabled
        if self.web_agent:
            await self.web_agent.initialize()
            
        # Register built-in tools
        self._register_tools()
        
        logger.info("Autonomous agent initialized successfully")
        
    async def _register_tools(self):
        """Register all available tools for the agent"""
        # Web browsing tools
        if self.web_agent:
            self.register_tool(
                "browse_web",
                "Navigate to a web page and extract its content",
                self.web_agent.navigate
            )
            
            self.register_tool(
                "search_web",
                "Search the web for information using search engines",
                self.web_agent.search
            )
            
            self.register_tool(
                "click_element",
                "Click on an element on a web page",
                self.web_agent.click_element
            )
            
            self.register_tool(
                "fill_form",
                "Fill a form field on a web page",
                self.web_agent.fill_form
            )
            
            self.register_tool(
                "submit_form",
                "Submit a form on a web page",
                self.web_agent.submit_form
            )
            
            self.register_tool(
                "extract_data",
                "Extract structured data from the current web page",
                self.web_agent.extract_structured_data
            )
            
            self.register_tool(
                "take_screenshot",
                "Take a screenshot of the current web page",
                self.web_agent.take_screenshot
            )
        
    def register_tool(self, name: str, description: str, func):
        """Register a new tool for the agent to use"""
        self.tools[name] = AutomationTool(name, description, func)
        logger.info(f"Registered tool: {name}")
        
    async def execute_tool(self, tool_name: str, *args, **kwargs) -> Dict[str, Any]:
        """Execute a registered tool by name"""
        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found"
            }
            
        tool = self.tools[tool_name]
        result = await tool.execute(*args, **kwargs)
        
        # Record tool execution in memory
        self.memory.add_interaction(
            role="agent",
            content=f"Executed tool: {tool_name}",
            metadata={
                "tool": tool_name,
                "args": args,
                "kwargs": kwargs,
                "success": result["success"]
            }
        )
        
        return result
        
    async def process_task(self, task: str) -> Dict[str, Any]:
        """
        Process a user task autonomously
        
        This is the main entry point for handling tasks. The agent will:
        1. Analyze the task
        2. Plan necessary actions
        3. Execute actions using tools
        4. Report results
        """
        self.total_tasks += 1
        start_time = time.time()
        
        # Record the task in memory
        self.memory.add_interaction(role="user", content=task)
        
        # Clear working memory for new task
        self.memory.clear_working_memory()
        
        try:
            # Analyze the task to determine required actions
            task_analysis = await self._analyze_task(task)
            
            # Store analysis in working memory
            self.memory.add_to_working_memory("task_analysis", task_analysis)
            
            # Create a plan
            plan = await self._create_plan(task, task_analysis)
            
            # Store plan in working memory
            self.memory.add_to_working_memory("plan", plan)
            
            # Execute the plan steps
            step_results = []
            for i, step in enumerate(plan["steps"]):
                self.memory.add_interaction(
                    role="agent",
                    content=f"Executing step {i+1}: {step['description']}"
                )
                
                # Execute the step
                step_result = await self._execute_step(step)
                step_results.append(step_result)
                
                # If a critical step failed, abort the plan
                if not step_result["success"] and step.get("critical", False):
                    self.memory.add_interaction(
                        role="agent",
                        content=f"Critical step failed, aborting plan: {step_result['error']}"
                    )
                    break
            
            # Generate result summary
            execution_time = time.time() - start_time
            success = all(step["success"] for step in step_results)
            
            if success:
                self.completed_tasks += 1
                
            # Generate a summary of the results
            summary = await self._generate_result_summary(task, step_results, success)
            
            # Record the entire task in history
            task_record = {
                "task": task,
                "analysis": task_analysis,
                "plan": plan,
                "results": step_results,
                "summary": summary,
                "success": success,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }
            self.task_history.append(task_record)
            
            # Add the agent's response to memory
            self.memory.add_interaction(role="agent", content=summary)
            
            return {
                "task": task,
                "summary": summary,
                "success": success,
                "execution_time": execution_time,
                "step_results": step_results
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error processing task: {str(e)}")
            traceback.print_exc()
            
            error_summary = f"I encountered an error while processing your task: {str(e)}"
            self.memory.add_interaction(role="agent", content=error_summary)
            
            return {
                "task": task,
                "error": str(e),
                "success": False,
                "execution_time": execution_time
            }
            
    async def _analyze_task(self, task: str) -> Dict[str, Any]:
        """
        Analyze a task to determine appropriate action plan
        This would use an LLM in a real implementation
        """
        # For now, use keyword-based classification
        task_lower = task.lower()
        
        # Detect web browsing intent
        web_terms = ["browse", "website", "web page", "visit", "go to", "open", ".com", ".org", ".edu", "http"]
        is_web_task = any(term in task_lower for term in web_terms)
        
        # Detect search intent
        search_terms = ["search", "find", "look up", "information about", "what is", "who is"]
        is_search_task = any(term in task_lower for term in search_terms)
        
        # Detect quantum computing intent
        quantum_terms = ["quantum", "factorize", "factor", "optimize", "qubit"]
        is_quantum_task = any(term in task_lower for term in quantum_terms)
        
        # Extract potential URLs
        urls = []
        import re
        url_pattern = r'https?://\S+|www\.\S+|\S+\.(com|org|edu|io|net|gov)\S*'
        matches = re.findall(url_pattern, task)
        if matches:
            for match in matches:
                if isinstance(match, tuple):
                    continue
                url = match
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url
                urls.append(url)
        
        # Extract potential search queries
        search_query = None
        if is_search_task:
            for term in search_terms:
                if term in task_lower:
                    parts = task_lower.split(term, 1)
                    if len(parts) > 1:
                        query = parts[1].strip()
                        if query:
                            search_query = query
                            break
        
        return {
            "task_type": "web_browsing" if is_web_task else "search" if is_search_task else "quantum_computing" if is_quantum_task else "general",
            "requires_web": is_web_task or is_search_task,
            "requires_quantum": is_quantum_task,
            "extracted_urls": urls,
            "search_query": search_query,
            "confidence": 0.8 if (is_web_task or is_search_task or is_quantum_task) else 0.5
        }
        
    async def _create_plan(self, task: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create a plan for executing the task based on analysis"""
        steps = []
        
        if analysis["task_type"] == "web_browsing":
            if analysis["extracted_urls"]:
                for url in analysis["extracted_urls"]:
                    steps.append({
                        "tool": "browse_web",
                        "description": f"Browse to {url}",
                        "params": {"url": url},
                        "critical": True
                    })
                    
                    steps.append({
                        "tool": "extract_data",
                        "description": "Extract structured data from the page",
                        "params": {},
                        "critical": False
                    })
            else:
                # No specific URL found, try to infer from context
                domain_hints = [
                    word for word in task.split() 
                    if (".com" in word or ".org" in word or ".edu" in word or ".io" in word)
                ]
                
                if domain_hints:
                    inferred_url = domain_hints[0]
                    if not inferred_url.startswith(('http://', 'https://')):
                        inferred_url = 'https://' + inferred_url
                        
                    steps.append({
                        "tool": "browse_web",
                        "description": f"Browse to inferred URL: {inferred_url}",
                        "params": {"url": inferred_url},
                        "critical": True
                    })
                    
                    steps.append({
                        "tool": "extract_data",
                        "description": "Extract structured data from the page",
                        "params": {},
                        "critical": False
                    })
                    
        elif analysis["task_type"] == "search":
            if analysis["search_query"]:
                steps.append({
                    "tool": "search_web",
                    "description": f"Search for: {analysis['search_query']}",
                    "params": {"query": analysis["search_query"]},
                    "critical": True
                })
                
                # Add a step to visit the first result
                steps.append({
                    "tool": "custom_action",
                    "description": "Visit the first relevant search result",
                    "action": "visit_search_result",
                    "params": {"index": 0},
                    "critical": False
                })
                
        elif analysis["task_type"] == "quantum_computing":
            # For quantum tasks, we'll delegate to specialized handlers
            steps.append({
                "tool": "custom_action",
                "description": f"Process quantum computing task",
                "action": "process_quantum_task",
                "params": {"task": task},
                "critical": True
            })
            
        # Always include a summarization step
        steps.append({
            "tool": "custom_action",
            "description": "Generate response summary",
            "action": "generate_summary",
            "params": {},
            "critical": True
        })
        
        return {
            "task": task,
            "task_type": analysis["task_type"],
            "steps": steps
        }
        
    async def _execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step in the plan"""
        if step.get("tool") in self.tools:
            # Execute a registered tool
            return await self.execute_tool(step["tool"], **step["params"])
            
        elif step.get("action") == "visit_search_result":
            # Custom action to visit a search result
            try:
                search_results = self.memory.get_from_working_memory("search_results")
                if not search_results or not isinstance(search_results, list) or len(search_results) <= step["params"]["index"]:
                    return {
                        "success": False,
                        "error": "No search results available or index out of range"
                    }
                    
                result = search_results[step["params"]["index"]]
                if "url" in result:
                    return await self.execute_tool("browse_web", url=result["url"])
                else:
                    return {
                        "success": False,
                        "error": "Selected search result has no URL"
                    }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Error visiting search result: {str(e)}"
                }
                
        elif step.get("action") == "process_quantum_task":
            # Handle quantum computing task
            if not self.quantum_processor:
                return {
                    "success": False, 
                    "error": "Quantum processor not initialized"
                }
                
            # This is a placeholder - would delegate to quantum_processor
            return {
                "success": True,
                "message": "Quantum task processing would happen here",
                "note": "This is simulated since this method focuses on web agency"
            }
            
        elif step.get("action") == "generate_summary":
            # Generate a summary of the task results
            # In a real implementation, this would use an LLM
            collected_data = []
            
            # Get browsing results from memory
            web_history = self.memory.web_history
            if web_history:
                last_pages = web_history[-3:]  # Last 3 pages
                collected_data.append(f"Browsed {len(web_history)} web pages.")
                for page in last_pages:
                    collected_data.append(f"- {page['title']} ({page['url']})")
            
            # Get search results from memory
            search_results = self.memory.get_from_working_memory("search_results")
            if search_results:
                collected_data.append(f"Found {len(search_results)} search results.")
                
            # Get extracted data
            structured_data = self.memory.get_from_working_memory("structured_data")
            if structured_data:
                if "tables" in structured_data and structured_data["tables"]:
                    collected_data.append(f"Extracted {len(structured_data['tables'])} tables.")
                if "lists" in structured_data and structured_data["lists"]:
                    collected_data.append(f"Extracted {len(structured_data['lists'])} lists.")
                    
            summary = "I've completed the task and collected the following information:\n\n"
            summary += "\n".join(collected_data) if collected_data else "No significant data was collected."
            
            return {
                "success": True,
                "summary": summary
            }
            
        else:
            # Unknown step type
            return {
                "success": False,
                "error": f"Unknown step type: {step.get('tool')} or {step.get('action')}"
            }
            
    async def _generate_result_summary(self, task: str, step_results: List[Dict[str, Any]], 
                                     success: bool) -> str:
        """Generate a summary of the task results"""
        # In a real implementation, this would use an LLM based on the memory
        # and step results
        
        if not success:
            # Find the first failed step
            for step in step_results:
                if not step["success"]:
                    return f"I wasn't able to complete the task due to an error: {step.get('error', 'Unknown error')}"
        
        # For successful tasks, gather relevant information
        summary_parts = []
        
        # Look for web browsing results
        browse_results = [s for s in step_results if s.get("tool") == "browse_web" and s["success"]]
        if browse_results:
            last_browse = browse_results[-1]
            summary_parts.append(f"I browsed to {last_browse.get('result', {}).get('url')} and viewed the page.")
            
            # If we have a title, add it
            if "result" in last_browse and "title" in last_browse["result"]:
                summary_parts.append(f"The page title is: {last_browse['result']['title']}")
                
        # Look for search results
        search_results = [s for s in step_results if s.get("tool") == "search_web" and s["success"]]
        if search_results:
            last_search = search_results[-1]
            query = last_search.get("result", {}).get("query", "")
            result_count = len(last_search.get("result", {}).get("results", []))
            
            if query and result_count > 0:
                summary_parts.append(f"I searched for '{query}' and found {result_count} results.")
                
                # List the top 3 results
                results = last_search.get("result", {}).get("results", [])
                if results:
                    summary_parts.append("Top results:")
                    for i, result in enumerate(results[:3]):
                        summary_parts.append(f"{i+1}. {result.get('title', 'Untitled')} - {result.get('url', 'No URL')}")
                        
        # Look for extracted data
        extract_results = [s for s in step_results if s.get("tool") == "extract_data" and s["success"]]
        if extract_results:
            last_extract = extract_results[-1]
            
            # Check for tables
            tables = last_extract.get("result", {}).get("tables", [])
            if tables:
                summary_parts.append(f"I found {len(tables)} tables on the page.")
                
            # Check for lists
            lists = last_extract.get("result", {}).get("lists", [])
            if lists:
                summary_parts.append(f"I found {len(lists)} lists on the page.")
                
        # If we have custom summaries from steps, include them
        for step in step_results:
            if "summary" in step:
                summary_parts.append(step["summary"])
                
        # Combine all parts into final summary
        if summary_parts:
            return "\n\n".join(summary_parts)
        else:
            return "I completed the task successfully, but don't have specific details to report."
            
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the agent"""
        uptime = time.time() - self.start_time
        success_rate = (self.completed_tasks / self.total_tasks * 100) if self.total_tasks > 0 else 0
        
        status = {
            "uptime_seconds": uptime,
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "success_rate": success_rate,
            "memory_stats": {
                "short_term_items": len(self.memory.short_term),
                "long_term_items": len(self.memory.long_term),
                "web_history_items": len(self.memory.web_history)
            }
        }
        
        # Add web metrics if available
        if self.web_agent:
            status["web_metrics"] = self.web_agent.get_metrics()
            
        # Add tool metrics
        tool_metrics = {}
        for name, tool in self.tools.items():
            tool_metrics[name] = tool.metrics
            
        status["tool_metrics"] = tool_metrics
        
        return status
        
    async def close(self):
        """Close all resources used by the agent"""
        if self.web_agent:
            await self.web_agent.close()
            
        logger.info("Agent resources released")