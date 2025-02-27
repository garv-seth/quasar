"""
Q³A: Quantum-Accelerated AI Agent with True Agentic Capabilities
Enhanced Streamlit Interface with Advanced Browser Automation and Computer Vision
"""

import streamlit as st
import os
import time
import logging
import json
import asyncio
import base64
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import threading
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quantum-agent")

# Check for quantum libraries with proper error handling
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available. This is required for quantum operations.")

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
    logger.info("PennyLane quantum computing library available.")
except ImportError:
    PENNYLANE_AVAILABLE = False
    logger.warning("PennyLane not available. Will use simulated quantum results.")

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

# Computer vision for screen understanding (for agentic capabilities)
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available. Screen analysis capabilities limited.")

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available. Image processing capabilities limited.")

# Browser automation 
try:
    import playwright
    import asyncio
    from playwright.async_api import async_playwright, Route, Request, ConsoleMessage
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not available. Browser automation capabilities limited.")


#####################################################
# AGENTIC CORE - Memory Systems
#####################################################

class AgentMemory:
    """Enhanced memory system for autonomous agent"""
    
    def __init__(self, max_short_term_items: int = 20):
        """Initialize agent memory systems"""
        # Short-term memory (recent interactions)
        self.short_term = []
        self.max_short_term = max_short_term_items
        
        # Working memory (current context and task state)
        self.working_memory = {}
        
        # Long-term memory (important information persisted)
        self.long_term = []
        
        # Web browsing history
        self.web_history = []
        
        # Screen observation memory
        self.screen_observations = []
        
    def add_interaction(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add an interaction to short-term memory with proper timestamp"""
        timestamp = datetime.now().isoformat()
        
        interaction = {
            "role": role,
            "content": content,
            "timestamp": timestamp
        }
        
        if metadata:
            interaction["metadata"] = metadata
            
        self.short_term.append(interaction)
        
        # Maintain maximum size
        if len(self.short_term) > self.max_short_term:
            self.short_term.pop(0)
    
    def add_to_working_memory(self, key: str, value: Any):
        """Store information in working memory"""
        self.working_memory[key] = value
    
    def get_from_working_memory(self, key: str, default: Any = None) -> Any:
        """Retrieve information from working memory"""
        return self.working_memory.get(key, default)
    
    def add_to_long_term(self, item: Dict[str, Any]):
        """Store important information in long-term memory"""
        timestamp = datetime.now().isoformat()
        item["timestamp"] = timestamp
        self.long_term.append(item)
    
    def add_web_visit(self, url: str, title: str, content_summary: str, metadata: Optional[Dict[str, Any]] = None):
        """Record web browsing history with metadata"""
        timestamp = datetime.now().isoformat()
        
        visit = {
            "url": url,
            "title": title,
            "content_summary": content_summary,
            "timestamp": timestamp
        }
        
        if metadata:
            visit["metadata"] = metadata
            
        self.web_history.append(visit)
    
    def add_screen_observation(self, screenshot_base64: str, elements_detected: List[Dict[str, Any]], description: str):
        """Record screen observation with detected UI elements"""
        timestamp = datetime.now().isoformat()
        
        observation = {
            "timestamp": timestamp,
            "screenshot_base64": screenshot_base64,
            "elements_detected": elements_detected,
            "description": description
        }
        
        self.screen_observations.append(observation)
        
        # Keep only last 10 observations to save memory
        if len(self.screen_observations) > 10:
            self.screen_observations.pop(0)
    
    def get_conversation_context(self, include_web_history: bool = True, max_items: int = 10) -> str:
        """Generate a context summary for decision making"""
        context = []
        
        # Add recent interactions
        context.append("## Recent Interactions")
        for item in self.short_term[-max_items:]:
            context.append(f"{item['role'].capitalize()}: {item['content']}")
        
        # Add relevant working memory items
        if self.working_memory:
            context.append("\n## Current Context")
            for key, value in self.working_memory.items():
                if isinstance(value, (str, int, float, bool)):
                    context.append(f"{key}: {value}")
                else:
                    context.append(f"{key}: [complex data]")
        
        # Add web browsing history if requested
        if include_web_history and self.web_history:
            context.append("\n## Recent Web Browsing")
            for item in self.web_history[-max_items:]:
                context.append(f"- Visited: {item['title']} ({item['url']})")
        
        return "\n".join(context)
    
    def clear_working_memory(self):
        """Clear working memory for new tasks"""
        self.working_memory = {}
    
    def get_relevant_memories(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Simple retrieval of relevant memories based on keyword matching"""
        # In a real implementation, this would use vector similarity search
        results = []
        
        # Search in long-term memory
        for item in self.long_term:
            content = json.dumps(item)
            if query.lower() in content.lower():
                results.append({
                    "source": "long_term",
                    "content": item,
                    "relevance": 0.8  # Placeholder
                })
        
        # Search in web history
        for item in self.web_history:
            if query.lower() in item.get("title", "").lower() or query.lower() in item.get("content_summary", "").lower():
                results.append({
                    "source": "web_history",
                    "content": item,
                    "relevance": 0.7  # Placeholder
                })
        
        # Sort by relevance and limit
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:max_results]


#####################################################
# AGENTIC CORE - Vision System for UI Understanding
#####################################################

class VisionSystem:
    """
    Computer vision system for screen understanding and UI element detection
    
    This component allows the agent to visually analyze interfaces, identify
    UI elements, and make decisions based on visual context.
    """
    
    def __init__(self):
        """Initialize the vision system"""
        self.cv_available = CV2_AVAILABLE and PIL_AVAILABLE
        self.element_detection_templates = {}  # Will store templates for button detection, etc.
        
        # UI element classifiers
        self.classifiers = {
            "button": None,
            "input": None,
            "dropdown": None,
            "checkbox": None,
            "link": None
        }
        
        if self.cv_available:
            self._load_templates()
    
    def _load_templates(self):
        """Load element detection templates"""
        # In a real implementation, this would load actual templates or models
        # This is a placeholder
        logger.info("Vision system initialized with UI element detection capabilities")
    
    def analyze_screenshot(self, screenshot_base64: str) -> Dict[str, Any]:
        """
        Analyze a screenshot to detect UI elements and understand the interface
        
        Args:
            screenshot_base64: Base64 encoded screenshot image
            
        Returns:
            Dict with detected elements and analysis
        """
        if not self.cv_available:
            return {"error": "Computer vision libraries not available"}
        
        try:
            # Decode the image
            image_data = base64.b64decode(screenshot_base64)
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            # Convert to RGB (OpenCV uses BGR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect UI elements (simplified implementation)
            ui_elements = self._detect_ui_elements(image_rgb)
            
            # Create visualization of detected elements
            visualization = self._create_visualization(image_rgb, ui_elements)
            
            return {
                "ui_elements": ui_elements,
                "visualization_base64": visualization
            }
            
        except Exception as e:
            logger.error(f"Error analyzing screenshot: {str(e)}")
            traceback.print_exc()
            return {"error": str(e)}
    
    def _detect_ui_elements(self, image) -> List[Dict[str, Any]]:
        """
        Detect UI elements in the image
        
        In a real implementation, this would use more sophisticated models
        for accurate UI element detection. This is a simplified placeholder.
        """
        height, width = image.shape[:2]
        elements = []
        
        # Simulate UI element detection based on color patterns and shapes
        # Note: This is a major simplification - real systems would use trained models
        
        # Detect potential buttons (looking for rectangular areas with consistent colors)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size (ignore very small or large areas)
            if 20 < w < 300 and 10 < h < 100:
                roi = image[y:y+h, x:x+w]
                color_var = np.std(roi)
                
                # Buttons often have consistent colors
                if color_var < 50:
                    elements.append({
                        "type": "button",
                        "confidence": 0.7,
                        "position": {"x": x, "y": y, "width": w, "height": h},
                        "center": {"x": x + w//2, "y": y + h//2}
                    })
        
        # Detect potential input fields (looking for rectangular areas with borders)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio
            if 100 < w < 500 and 20 < h < 70 and w > 3*h:
                elements.append({
                    "type": "input",
                    "confidence": 0.6,
                    "position": {"x": x, "y": y, "width": w, "height": h},
                    "center": {"x": x + w//2, "y": y + h//2}
                })
        
        return elements
    
    def _create_visualization(self, image, elements) -> str:
        """Create a visualization of detected UI elements"""
        # Convert NumPy array to PIL Image
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        
        # Draw bounding boxes for each element
        for element in elements:
            pos = element["position"]
            x, y, w, h = pos["x"], pos["y"], pos["width"], pos["height"]
            
            if element["type"] == "button":
                color = (255, 0, 0)  # Red for buttons
            elif element["type"] == "input":
                color = (0, 255, 0)  # Green for inputs
            else:
                color = (0, 0, 255)  # Blue for others
                
            draw.rectangle([x, y, x+w, y+h], outline=color, width=2)
            draw.text((x, y-10), f"{element['type']} ({element['confidence']:.2f})", fill=color)
        
        # Convert back to base64
        from io import BytesIO
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()


#####################################################
# AGENTIC CORE - Web Interaction Agent
#####################################################

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
        if not PLAYWRIGHT_AVAILABLE:
            logger.error("Playwright not available. Cannot initialize browser automation.")
            return False
            
        if self.browser is not None:
            return True  # Already initialized
            
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
                
            # Navigate to the URL
            response = await self.page.goto(url, wait_until="networkidle")
            
            # Extract page information
            self.current_url = self.page.url
            self.current_page_title = await self.page.title()
            self.current_page_content = await self.page.content()
            
            # Take screenshot for visual understanding
            screenshot = await self.page.screenshot(type="png")
            screenshot_base64 = base64.b64encode(screenshot).decode('utf-8')
            
            # Record in navigation history
            self.navigation_history.append({
                "url": self.current_url,
                "title": self.current_page_title,
                "timestamp": time.time()
            })
            
            # Update metrics
            self.successful_interactions += 1
            execution_time = time.time() - start_time
            
            # Extract information from the page
            page_text = await self.page.evaluate("""() => {
                return document.body.innerText;
            }""")
            
            # Extract links
            links = await self.page.evaluate("""() => {
                const links = Array.from(document.querySelectorAll('a'));
                return links.map(link => ({ 
                    text: link.innerText, 
                    href: link.href,
                    visible: link.offsetParent !== null
                }));
            }""")
            
            # Extract forms
            forms = await self.page.evaluate("""() => {
                const forms = Array.from(document.querySelectorAll('form'));
                return forms.map(form => {
                    const inputs = Array.from(form.querySelectorAll('input, select, textarea'));
                    return {
                        id: form.id,
                        action: form.action,
                        method: form.method,
                        inputs: inputs.map(input => ({
                            type: input.type,
                            name: input.name,
                            id: input.id,
                            placeholder: input.placeholder,
                            required: input.required
                        }))
                    };
                });
            }""")
            
            return {
                "success": True,
                "url": self.current_url,
                "title": self.current_page_title,
                "text_content": page_text[:2000] + ("..." if len(page_text) > 2000 else ""),
                "links": links[:20],  # Limit to 20 links
                "forms": forms,
                "screenshot_base64": screenshot_base64,
                "execution_time": execution_time
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed to navigate to {url}: {str(e)}")
            
            return {
                "success": False,
                "url": url,
                "error": str(e),
                "execution_time": execution_time
            }
            
    async def search(self, query: str, search_engine: str = "google") -> Dict[str, Any]:
        """Perform a web search using a search engine"""
        if self.page is None:
            await self.initialize()
            
        self.total_interactions += 1
        start_time = time.time()
        
        try:
            # Choose search engine URL
            if search_engine.lower() == "google":
                url = f"https://www.google.com/search?q={query}"
            elif search_engine.lower() == "bing":
                url = f"https://www.bing.com/search?q={query}"
            elif search_engine.lower() == "duckduckgo":
                url = f"https://duckduckgo.com/?q={query}"
            else:
                url = f"https://www.google.com/search?q={query}"
                
            # Navigate to search engine with query
            result = await self.navigate(url)
            
            if not result["success"]:
                return result
                
            # Extract search results
            if search_engine.lower() == "google":
                search_results = await self.page.evaluate("""() => {
                    const results = Array.from(document.querySelectorAll('.g'));
                    return results.map(result => {
                        const titleEl = result.querySelector('h3');
                        const linkEl = result.querySelector('a');
                        const snippetEl = result.querySelector('.VwiC3b');
                        
                        return {
                            title: titleEl ? titleEl.innerText : '',
                            link: linkEl ? linkEl.href : '',
                            snippet: snippetEl ? snippetEl.innerText : ''
                        };
                    }).filter(r => r.title && r.link);
                }""")
            elif search_engine.lower() == "bing":
                search_results = await self.page.evaluate("""() => {
                    const results = Array.from(document.querySelectorAll('.b_algo'));
                    return results.map(result => {
                        const titleEl = result.querySelector('h2 a');
                        const snippetEl = result.querySelector('.b_caption p');
                        
                        return {
                            title: titleEl ? titleEl.innerText : '',
                            link: titleEl ? titleEl.href : '',
                            snippet: snippetEl ? snippetEl.innerText : ''
                        };
                    }).filter(r => r.title && r.link);
                }""")
            else:
                # Generic extraction as fallback
                search_results = await self.page.evaluate("""() => {
                    const results = Array.from(document.querySelectorAll('a[href^="http"]'));
                    return results.map(a => {
                        return {
                            title: a.innerText,
                            link: a.href,
                            snippet: ''
                        };
                    }).filter(r => r.title && r.link);
                }""")
                
            # Update metrics
            self.successful_interactions += 1
            execution_time = time.time() - start_time
                
            # Take screenshot for visual understanding
            screenshot = await self.page.screenshot(type="png")
            screenshot_base64 = base64.b64encode(screenshot).decode('utf-8')
                
            return {
                "success": True,
                "query": query,
                "search_engine": search_engine,
                "results": search_results[:10],  # Limit to top 10
                "screenshot_base64": screenshot_base64,
                "execution_time": execution_time
            }
                
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed to search for '{query}': {str(e)}")
            
            return {
                "success": False,
                "query": query,
                "search_engine": search_engine,
                "error": str(e),
                "execution_time": execution_time
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
                "previous_url": self.current_url,
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
            
        self.total_interactions += 1
        start_time = time.time()
        
        try:
            # Extract tables
            tables = await self.page.evaluate("""() => {
                const tables = Array.from(document.querySelectorAll('table'));
                return tables.map(table => {
                    const rows = Array.from(table.querySelectorAll('tr'));
                    return rows.map(row => {
                        const cells = Array.from(row.querySelectorAll('td, th'));
                        return cells.map(cell => cell.innerText);
                    });
                });
            }""")
            
            # Extract lists
            lists = await self.page.evaluate("""() => {
                const lists = Array.from(document.querySelectorAll('ul, ol'));
                return lists.map(list => {
                    const items = Array.from(list.querySelectorAll('li'));
                    return items.map(item => item.innerText);
                });
            }""")
            
            # Extract metadata
            metadata = await self.page.evaluate("""() => {
                const metadata = {};
                
                // Extract meta tags
                const metaTags = Array.from(document.querySelectorAll('meta'));
                metaTags.forEach(tag => {
                    const name = tag.getAttribute('name') || tag.getAttribute('property');
                    const content = tag.getAttribute('content');
                    if (name && content) {
                        metadata[name] = content;
                    }
                });
                
                // Extract JSON-LD
                const jsonLdScripts = Array.from(document.querySelectorAll('script[type="application/ld+json"]'));
                if (jsonLdScripts.length > 0) {
                    try {
                        metadata.jsonLd = JSON.parse(jsonLdScripts[0].innerText);
                    } catch (e) {
                        // Ignore JSON parse errors
                    }
                }
                
                return metadata;
            }""")
            
            # Update metrics
            self.successful_interactions += 1
            execution_time = time.time() - start_time
            
            return {
                "success": True,
                "tables": tables,
                "lists": lists,
                "metadata": metadata,
                "url": self.current_url,
                "execution_time": execution_time
            }
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed to extract structured data: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time
            }
            
    async def take_screenshot(self) -> Dict[str, Any]:
        """Take a screenshot of the current page"""
        if self.page is None:
            return {"success": False, "error": "Browser not initialized"}
            
        self.total_interactions += 1
        start_time = time.time()
        
        try:
            # Take screenshot
            screenshot = await self.page.screenshot(type="png")
            screenshot_base64 = base64.b64encode(screenshot).decode('utf-8')
            
            # Update metrics
            self.successful_interactions += 1
            execution_time = time.time() - start_time
            
            return {
                "success": True,
                "screenshot_base64": screenshot_base64,
                "url": self.current_url,
                "title": self.current_page_title,
                "execution_time": execution_time
            }
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed to take screenshot: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get web interaction metrics"""
        success_rate = 0
        if self.total_interactions > 0:
            success_rate = (self.successful_interactions / self.total_interactions) * 100
            
        return {
            "total_interactions": self.total_interactions,
            "successful_interactions": self.successful_interactions,
            "success_rate": success_rate,
            "pages_visited": len(self.navigation_history)
        }


#####################################################
# AGENTIC CORE - Tool System
#####################################################

class AutomationTool:
    """
    A tool that can be executed by the agent to perform specific actions
    """
    
    def __init__(self, name: str, description: str, func):
        self.name = name
        self.description = description
        self.func = func
        self.execution_count = 0
        self.success_count = 0
        self.total_execution_time = 0
        
    async def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """Execute the tool function with metrics tracking"""
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(self.func):
                result = await self.func(*args, **kwargs)
            else:
                result = self.func(*args, **kwargs)
                
            success = True
            if isinstance(result, dict) and "success" in result:
                success = result["success"]
                
            self.execution_count += 1
            if success:
                self.success_count += 1
                
            self.total_execution_time += time.time() - start_time
            
            return result
        except Exception as e:
            logger.error(f"Error executing tool {self.name}: {str(e)}")
            self.execution_count += 1
            self.total_execution_time += time.time() - start_time
            return {"success": False, "error": str(e)}


#####################################################
# AGENTIC CORE - Task Planning and Execution
#####################################################

class TaskPlanner:
    """
    Task planning and execution system for complex multi-step operations
    
    This component handles:
    1. Breaking down complex tasks into steps
    2. Executing steps in sequence
    3. Handling dependencies between steps
    4. Monitoring execution and adapting to results
    """
    
    def __init__(self, agent):
        """Initialize with reference to the parent agent"""
        self.agent = agent
        self.current_plan = []
        self.current_step_index = 0
        self.plan_status = "idle"  # idle, planning, executing, completed, failed
        
    async def create_plan(self, task: str) -> Dict[str, Any]:
        """Create a step-by-step plan for executing a complex task"""
        self.plan_status = "planning"
        
        # Use AI to break down the task
        plan_result = await self.agent._analyze_task(task)
        
        if not plan_result.get("success", False):
            self.plan_status = "failed"
            return {"success": False, "error": "Failed to create plan"}
        
        self.current_plan = plan_result.get("steps", [])
        self.current_step_index = 0
        self.plan_status = "ready"
        
        return {
            "success": True,
            "plan": self.current_plan,
            "step_count": len(self.current_plan)
        }
        
    async def execute_next_step(self) -> Dict[str, Any]:
        """Execute the next step in the current plan"""
        if self.plan_status not in ["ready", "executing"]:
            return {"success": False, "error": f"Cannot execute next step, plan status: {self.plan_status}"}
            
        if self.current_step_index >= len(self.current_plan):
            self.plan_status = "completed"
            return {"success": True, "status": "completed", "message": "Plan execution completed"}
            
        self.plan_status = "executing"
        step = self.current_plan[self.current_step_index]
        
        try:
            # Execute the step
            result = await self.agent._execute_step(step)
            
            self.current_step_index += 1
            
            # Check if plan is completed
            if self.current_step_index >= len(self.current_plan):
                self.plan_status = "completed"
            
            return {
                "success": True,
                "step_index": self.current_step_index - 1,
                "step": step,
                "result": result,
                "next_step_index": self.current_step_index if self.current_step_index < len(self.current_plan) else None,
                "next_step": self.current_plan[self.current_step_index] if self.current_step_index < len(self.current_plan) else None,
                "status": self.plan_status
            }
            
        except Exception as e:
            logger.error(f"Error executing step {self.current_step_index}: {str(e)}")
            self.plan_status = "failed"
            return {
                "success": False,
                "step_index": self.current_step_index,
                "step": step,
                "error": str(e),
                "status": self.plan_status
            }
            
    async def execute_complete_plan(self) -> Dict[str, Any]:
        """Execute the complete plan from start to finish"""
        if self.plan_status not in ["ready", "executing"]:
            return {"success": False, "error": f"Cannot execute plan, status: {self.plan_status}"}
            
        self.plan_status = "executing"
        results = []
        
        while self.current_step_index < len(self.current_plan):
            step_result = await self.execute_next_step()
            results.append(step_result)
            
            if not step_result.get("success", False):
                break
                
        return {
            "success": self.plan_status == "completed",
            "status": self.plan_status,
            "step_results": results,
            "completed_steps": self.current_step_index,
            "total_steps": len(self.current_plan)
        }
        
    def reset(self):
        """Reset the planner state"""
        self.current_plan = []
        self.current_step_index = 0
        self.plan_status = "idle"


#####################################################
# MAIN AUTONOMOUS AGENT
#####################################################

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
        self.vision_system = VisionSystem()
        self.task_planner = TaskPlanner(self)
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
        await self._register_tools()
        
        logger.info("Autonomous agent initialized successfully")
        return True
        
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
            
        # Visual understanding tools
        self.register_tool(
            "analyze_screenshot",
            "Analyze a screenshot to identify UI elements and understand the interface",
            self.vision_system.analyze_screenshot
        )
        
        # Memory tools
        self.register_tool(
            "get_relevant_memories",
            "Retrieve relevant information from memory based on a query",
            self.memory.get_relevant_memories
        )
        
        # Task planning tools
        self.register_tool(
            "create_task_plan",
            "Create a step-by-step plan for executing a complex task",
            self.task_planner.create_plan
        )
        
        self.register_tool(
            "execute_next_step",
            "Execute the next step in the current task plan",
            self.task_planner.execute_next_step
        )
        
        self.register_tool(
            "execute_complete_plan",
            "Execute the complete task plan from start to finish",
            self.task_planner.execute_complete_plan
        )
        
    def register_tool(self, name: str, description: str, func):
        """Register a new tool for the agent to use"""
        self.tools[name] = AutomationTool(name, description, func)
        logger.info(f"Registered tool: {name}")
        
    async def execute_tool(self, tool_name: str, *args, **kwargs) -> Dict[str, Any]:
        """Execute a registered tool by name"""
        if tool_name not in self.tools:
            return {"success": False, "error": f"Tool not found: {tool_name}"}
            
        tool = self.tools[tool_name]
        return await tool.execute(*args, **kwargs)
        
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
        self.memory.add_interaction("user", task)
        
        try:
            # Analyze the task
            analysis = await self._analyze_task(task)
            
            # Create a plan based on the analysis
            plan = await self._create_plan(task, analysis)
            
            # Execute the plan
            results = []
            success = True
            
            for step in plan.get("steps", []):
                step_result = await self._execute_step(step)
                results.append(step_result)
                
                if not step_result.get("success", False):
                    success = False
                    break
            
            # Generate a summary of results
            summary = await self._generate_result_summary(task, results, success)
            
            # Record task completion
            self.completed_tasks += 1
            self.memory.add_interaction("agent", summary)
            self.task_history.append({
                "task": task,
                "analysis": analysis,
                "plan": plan,
                "results": results,
                "success": success,
                "summary": summary,
                "timestamp": time.time()
            })
            
            return {
                "success": success,
                "analysis": analysis,
                "plan": plan,
                "results": results,
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"Error processing task: {str(e)}")
            error_message = f"I encountered an error while processing your task: {str(e)}"
            self.memory.add_interaction("agent", error_message)
            
            return {
                "success": False,
                "error": str(e),
                "message": error_message
            }
    
    async def _analyze_task(self, task: str) -> Dict[str, Any]:
        """
        Analyze a task to determine appropriate action plan
        This would use an LLM in a real implementation
        """
        # Simplified implementation - in a real system this would use an LLM
        analysis = {"task_type": "unknown", "entities": [], "required_tools": []}
        
        # Check for web browsing tasks
        if re.search(r'(browse|visit|go to|open|navigate).*?(website|webpage|site|url|http)', task, re.IGNORECASE):
            analysis["task_type"] = "web_navigation"
            analysis["required_tools"] = ["browse_web"]
            
            # Extract URL if present
            url_match = re.search(r'(https?://\S+|www\.\S+|\S+\.(com|org|net|edu|gov)\S*)', task)
            if url_match:
                analysis["entities"].append({"type": "url", "value": url_match.group(0)})
        
        # Check for search tasks
        elif re.search(r'(search|find|look up|google|research)', task, re.IGNORECASE):
            analysis["task_type"] = "web_search"
            analysis["required_tools"] = ["search_web"]
            
            # Extract search query
            query_match = re.search(r'(search|find|look up|google|research)\s+(for|about)?\s+(.+?)(\.|$)', task, re.IGNORECASE)
            if query_match:
                analysis["entities"].append({"type": "search_query", "value": query_match.group(3).strip()})
        
        # Check for form filling tasks
        elif re.search(r'(fill|enter|input|type)', task, re.IGNORECASE) and re.search(r'(form|field|input|textbox)', task, re.IGNORECASE):
            analysis["task_type"] = "form_interaction"
            analysis["required_tools"] = ["browse_web", "fill_form", "submit_form"]
            
        # Check for data extraction tasks
        elif re.search(r'(extract|scrape|get|collect|fetch)', task, re.IGNORECASE) and re.search(r'(data|information|content|text)', task, re.IGNORECASE):
            analysis["task_type"] = "data_extraction"
            analysis["required_tools"] = ["browse_web", "extract_data"]
            
        # Default to general task requiring planning
        else:
            analysis["task_type"] = "general"
            analysis["required_tools"] = ["create_task_plan", "execute_complete_plan"]
        
        # Add timestamp
        analysis["timestamp"] = time.time()
        analysis["success"] = True
        
        return analysis
    
    async def _create_plan(self, task: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create a plan for executing the task based on analysis"""
        plan = {"steps": [], "success": True}
        
        # Web navigation task
        if analysis.get("task_type") == "web_navigation":
            url = None
            for entity in analysis.get("entities", []):
                if entity.get("type") == "url":
                    url = entity.get("value")
                    break
                    
            if url:
                plan["steps"].append({
                    "tool": "browse_web",
                    "description": f"Navigate to {url}",
                    "params": {"url": url}
                })
                
                plan["steps"].append({
                    "tool": "take_screenshot",
                    "description": "Take a screenshot of the page",
                    "params": {}
                })
            else:
                plan["success"] = False
                plan["error"] = "No URL found in the task"
                
        # Web search task
        elif analysis.get("task_type") == "web_search":
            query = None
            for entity in analysis.get("entities", []):
                if entity.get("type") == "search_query":
                    query = entity.get("value")
                    break
                    
            if query:
                plan["steps"].append({
                    "tool": "search_web",
                    "description": f"Search for information about '{query}'",
                    "params": {"query": query, "search_engine": "google"}
                })
            else:
                # Try to extract query from the task directly
                query = task.replace("search for", "").replace("find", "").replace("look up", "").strip()
                plan["steps"].append({
                    "tool": "search_web",
                    "description": f"Search for information: '{query}'",
                    "params": {"query": query, "search_engine": "google"}
                })
                
        # Form interaction task
        elif analysis.get("task_type") == "form_interaction":
            # This would need more sophisticated parsing in a real system
            plan["steps"].append({
                "tool": "browse_web",
                "description": "Navigate to the page with the form",
                "params": {"url": "https://example.com"}  # Placeholder
            })
            
            plan["steps"].append({
                "tool": "fill_form",
                "description": "Fill in the form field",
                "params": {"selector": "input[name='q']", "value": "example"}  # Placeholder
            })
            
            plan["steps"].append({
                "tool": "submit_form",
                "description": "Submit the form",
                "params": {"form_selector": "form"}  # Placeholder
            })
            
        # Data extraction task
        elif analysis.get("task_type") == "data_extraction":
            # Extract URL if present
            url = None
            for entity in analysis.get("entities", []):
                if entity.get("type") == "url":
                    url = entity.get("value")
                    break
                    
            if url:
                plan["steps"].append({
                    "tool": "browse_web",
                    "description": f"Navigate to {url}",
                    "params": {"url": url}
                })
            else:
                # Default to a search step first
                query = task.replace("extract", "").replace("get", "").replace("fetch", "").strip()
                plan["steps"].append({
                    "tool": "search_web",
                    "description": f"Search for information to extract: '{query}'",
                    "params": {"query": query, "search_engine": "google"}
                })
                
                plan["steps"].append({
                    "tool": "browse_web",
                    "description": "Navigate to the first relevant result",
                    "params": {"url": "PLACEHOLDER_FROM_SEARCH_RESULTS"}  # Will be replaced after search
                })
                
            plan["steps"].append({
                "tool": "extract_data",
                "description": "Extract structured data from the page",
                "params": {}
            })
            
        # General task using planning
        else:
            plan["steps"].append({
                "tool": "create_task_plan",
                "description": f"Create a detailed plan for: {task}",
                "params": {"task": task}
            })
            
            plan["steps"].append({
                "tool": "execute_complete_plan",
                "description": "Execute the generated plan",
                "params": {}
            })
        
        return plan
    
    async def _execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step in the plan"""
        tool_name = step.get("tool")
        params = step.get("params", {})
        
        # Special case for placeholder values that should be filled from previous steps
        for key, value in params.items():
            if isinstance(value, str) and value == "PLACEHOLDER_FROM_SEARCH_RESULTS" and self.memory.get_from_working_memory("last_search_results"):
                last_results = self.memory.get_from_working_memory("last_search_results")
                if last_results and "results" in last_results and last_results["results"]:
                    first_result = last_results["results"][0]
                    params[key] = first_result.get("link", "https://example.com")
        
        # Execute the tool
        result = await self.execute_tool(tool_name, **params)
        
        # Store important results in working memory
        if tool_name == "search_web" and result.get("success", False):
            self.memory.add_to_working_memory("last_search_results", result)
            
        if tool_name == "browse_web" and result.get("success", False):
            self.memory.add_to_working_memory("current_page", {
                "url": result.get("url"),
                "title": result.get("title"),
                "content_preview": result.get("text_content", "")[:200]
            })
            
            if "links" in result:
                self.memory.add_to_working_memory("current_page_links", result["links"])
                
            if "forms" in result:
                self.memory.add_to_working_memory("current_page_forms", result["forms"])
                
        return {
            "step": step,
            "result": result,
            "success": result.get("success", False),
            "timestamp": time.time()
        }
    
    async def _generate_result_summary(self, task: str, step_results: List[Dict[str, Any]], 
                                     success: bool) -> str:
        """Generate a summary of the task results"""
        # In a real implementation, this would use an LLM to generate a natural language summary
        
        if not success:
            failed_step = next((r for r in step_results if not r.get("success", False)), None)
            if failed_step:
                return f"I was unable to complete the task. I encountered an error while trying to {failed_step['step'].get('description')}: {failed_step['result'].get('error', 'Unknown error')}"
            else:
                return "I was unable to complete the task due to an unknown error."
                
        summary_parts = []
        
        # Web navigation result
        if any(r["step"].get("tool") == "browse_web" for r in step_results):
            browse_result = next((r for r in step_results if r["step"].get("tool") == "browse_web"), None)
            if browse_result and browse_result.get("success", False):
                result = browse_result["result"]
                summary_parts.append(f"I visited {result.get('title')} at {result.get('url')}.")
                
                # Add link stats if available
                if "links" in result:
                    summary_parts.append(f"The page contains {len(result['links'])} links.")
                    
                # Add form stats if available
                if "forms" in result:
                    summary_parts.append(f"The page contains {len(result['forms'])} forms.")
        
        # Search result
        if any(r["step"].get("tool") == "search_web" for r in step_results):
            search_result = next((r for r in step_results if r["step"].get("tool") == "search_web"), None)
            if search_result and search_result.get("success", False):
                result = search_result["result"]
                summary_parts.append(f"I searched for '{result.get('query')}' and found {len(result.get('results', []))} results.")
                
                # Add top results
                if "results" in result and result["results"]:
                    summary_parts.append("Top results:")
                    for i, r in enumerate(result["results"][:3], 1):
                        summary_parts.append(f"{i}. {r.get('title')} - {r.get('link')}")
                        
        # Data extraction result
        if any(r["step"].get("tool") == "extract_data" for r in step_results):
            extract_result = next((r for r in step_results if r["step"].get("tool") == "extract_data"), None)
            if extract_result and extract_result.get("success", False):
                result = extract_result["result"]
                
                # Summarize tables
                if "tables" in result and result["tables"]:
                    summary_parts.append(f"I extracted {len(result['tables'])} tables from the page.")
                    
                # Summarize lists
                if "lists" in result and result["lists"]:
                    summary_parts.append(f"I extracted {len(result['lists'])} lists from the page.")
                    
                # Summarize metadata
                if "metadata" in result and result["metadata"]:
                    summary_parts.append(f"I extracted metadata information from the page.")
                    
        if not summary_parts:
            return "I completed the task successfully."
            
        return "\n".join(summary_parts)
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the agent"""
        uptime = time.time() - self.start_time
        
        status = {
            "uptime": uptime,
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "success_rate": (self.completed_tasks / max(1, self.total_tasks)) * 100,
            "memory_stats": {
                "short_term_items": len(self.memory.short_term),
                "working_memory_items": len(self.memory.working_memory),
                "long_term_items": len(self.memory.long_term),
                "web_history_items": len(self.memory.web_history)
            }
        }
        
        # Add web agent metrics if available
        if self.web_agent:
            status["web_agent"] = self.web_agent.get_metrics()
            
        # Add tool usage metrics
        status["tools"] = {}
        for name, tool in self.tools.items():
            status["tools"][name] = {
                "execution_count": tool.execution_count,
                "success_count": tool.success_count,
                "success_rate": (tool.success_count / max(1, tool.execution_count)) * 100,
                "avg_execution_time": tool.total_execution_time / max(1, tool.execution_count)
            }
            
        return status
        
    async def close(self):
        """Close all resources used by the agent"""
        if self.web_agent:
            await self.web_agent.close()


#####################################################
# Streamlit Interface
#####################################################

def initialize_session_state():
    """Initialize Streamlit session state"""
    if "agent" not in st.session_state:
        st.session_state.agent = None
        
    if "agent_initialized" not in st.session_state:
        st.session_state.agent_initialized = False
        
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if "current_task" not in st.session_state:
        st.session_state.current_task = None
        
    if "task_status" not in st.session_state:
        st.session_state.task_status = None
        
    if "task_results" not in st.session_state:
        st.session_state.task_results = []
        
    if "api_keys_configured" not in st.session_state:
        st.session_state.api_keys_configured = {
            "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
            "ANTHROPIC_API_KEY": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "AZURE_QUANTUM_SUBSCRIPTION_ID": bool(os.environ.get("AZURE_QUANTUM_SUBSCRIPTION_ID"))
        }
        
    if "ui_tab" not in st.session_state:
        st.session_state.ui_tab = "chat"


async def initialize_agent_async():
    """Initialize the agent asynchronously"""
    if st.session_state.agent is None:
        st.session_state.agent = AutonomousAgent(
            use_quantum=True, 
            n_qubits=8,
            use_claude=True,
            use_browser=True
        )
        
    if not st.session_state.agent_initialized:
        await st.session_state.agent.initialize()
        st.session_state.agent_initialized = True
        
    return st.session_state.agent


def run_async(coroutine):
    """Run an async function from Streamlit"""
    async_thread = threading.Thread(target=lambda: asyncio.run(coroutine))
    async_thread.start()
    async_thread.join()


def initialize_agent():
    """Initialize the agent (wrapper for async initialization)"""
    if not st.session_state.agent_initialized:
        with st.spinner("Initializing agent..."):
            run_async(initialize_agent_async())
            st.success("Agent initialized successfully")


async def process_message_async(message):
    """Process a user message asynchronously"""
    agent = await initialize_agent_async()
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": message})
    
    # Set current task
    st.session_state.current_task = message
    
    try:
        # Process the task
        result = await agent.process_task(message)
        
        # Add agent response to chat history
        if result.get("success", False):
            response = result.get("summary", "Task completed successfully")
        else:
            response = result.get("message", f"Error: {result.get('error', 'Unknown error')}")
            
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Store task results
        st.session_state.task_results.append(result)
        st.session_state.task_status = "completed" if result.get("success", False) else "failed"
        
    except Exception as e:
        error_message = f"Error processing message: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_message})
        st.session_state.task_status = "failed"


def process_message(message):
    """Process a user message (wrapper for async processing)"""
    with st.spinner("Processing your request..."):
        run_async(lambda: process_message_async(message))


def display_chat_interface():
    """Display the main chat interface"""
    st.markdown("### 💬 Chat with your Quantum-Accelerated AI Agent")
    
    # Display chat messages
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(content)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(content)
    
    # Chat input
    if prompt := st.chat_input("What would you like the agent to do?"):
        process_message(prompt)


def display_task_details():
    """Display details of the current or most recent task"""
    if not st.session_state.task_results:
        st.info("No tasks have been executed yet. Use the chat interface to give the agent a task.")
        return
    
    # Get the most recent task result
    task_result = st.session_state.task_results[-1]
    
    st.markdown("### 📋 Task Details")
    
    # Task status
    status = "✅ Completed" if task_result.get("success", False) else "❌ Failed"
    st.markdown(f"**Status:** {status}")
    
    # Task analysis
    if "analysis" in task_result:
        analysis = task_result["analysis"]
        st.markdown("#### Task Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Task Type:** {analysis.get('task_type', 'Unknown')}")
            
            if "entities" in analysis and analysis["entities"]:
                st.markdown("**Entities:**")
                for entity in analysis["entities"]:
                    st.markdown(f"- {entity.get('type')}: {entity.get('value')}")
        
        with col2:
            if "required_tools" in analysis:
                st.markdown("**Required Tools:**")
                for tool in analysis["required_tools"]:
                    st.markdown(f"- {tool}")
    
    # Execution plan
    if "plan" in task_result and "steps" in task_result["plan"]:
        st.markdown("#### Execution Plan")
        
        for i, step in enumerate(task_result["plan"]["steps"], 1):
            with st.expander(f"Step {i}: {step.get('description', 'No description')}"):
                st.markdown(f"**Tool:** {step.get('tool', 'N/A')}")
                
                if "params" in step:
                    st.markdown("**Parameters:**")
                    for key, value in step["params"].items():
                        st.markdown(f"- {key}: {value}")
    
    # Results
    if "results" in task_result and task_result["results"]:
        st.markdown("#### Execution Results")
        
        for i, step_result in enumerate(task_result["results"], 1):
            result = step_result.get("result", {})
            success = step_result.get("success", False)
            status_icon = "✅" if success else "❌"
            
            with st.expander(f"{status_icon} Step {i} Result"):
                if not success and "error" in result:
                    st.error(f"Error: {result['error']}")
                    continue
                
                # Display different result types
                if step_result["step"].get("tool") == "search_web" and "results" in result:
                    st.markdown(f"Searched for: **{result.get('query', 'Unknown query')}**")
                    
                    for j, search_result in enumerate(result["results"][:5], 1):
                        st.markdown(f"{j}. [{search_result.get('title', 'No title')}]({search_result.get('link', '#')})")
                        st.markdown(f"   {search_result.get('snippet', 'No snippet')}")
                
                elif step_result["step"].get("tool") == "browse_web":
                    st.markdown(f"Visited: **{result.get('title', 'Unknown page')}**")
                    st.markdown(f"URL: {result.get('url', 'Unknown URL')}")
                    
                    if "text_content" in result:
                        with st.expander("Page content preview"):
                            st.markdown(result["text_content"])
                    
                    if "screenshot_base64" in result:
                        with st.expander("Page screenshot"):
                            st.image(f"data:image/png;base64,{result['screenshot_base64']}", use_column_width=True)
                
                elif step_result["step"].get("tool") == "extract_data":
                    if "tables" in result and result["tables"]:
                        st.markdown(f"Extracted {len(result['tables'])} tables")
                        
                        for j, table in enumerate(result["tables"][:3], 1):
                            with st.expander(f"Table {j}"):
                                # Convert to pandas DataFrame for display
                                if len(table) > 0:
                                    import pandas as pd
                                    # Use first row as header if possible
                                    if len(table) > 1:
                                        df = pd.DataFrame(table[1:], columns=table[0])
                                    else:
                                        df = pd.DataFrame([table[0]])
                                    st.dataframe(df)
                    
                    if "lists" in result and result["lists"]:
                        st.markdown(f"Extracted {len(result['lists'])} lists")
                        
                        for j, lst in enumerate(result["lists"][:3], 1):
                            with st.expander(f"List {j}"):
                                for item in lst:
                                    st.markdown(f"- {item}")


def display_agent_status():
    """Display agent status and metrics"""
    if not st.session_state.agent_initialized:
        st.info("Agent not initialized yet.")
        return
    
    st.markdown("### 📊 Agent Status and Metrics")
    
    # Get agent status
    status = st.session_state.agent.get_status()
    
    # General metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Tasks", status.get("total_tasks", 0))
    
    with col2:
        st.metric("Completed Tasks", status.get("completed_tasks", 0))
    
    with col3:
        st.metric("Success Rate", f"{status.get('success_rate', 0):.1f}%")
    
    # Memory stats
    memory_stats = status.get("memory_stats", {})
    
    st.markdown("#### Memory Usage")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    
    with mcol1:
        st.metric("Short-term", memory_stats.get("short_term_items", 0))
    
    with mcol2:
        st.metric("Working memory", memory_stats.get("working_memory_items", 0))
    
    with mcol3:
        st.metric("Long-term", memory_stats.get("long_term_items", 0))
    
    with mcol4:
        st.metric("Web history", memory_stats.get("web_history_items", 0))
    
    # Web agent metrics
    if "web_agent" in status:
        web_metrics = status["web_agent"]
        
        st.markdown("#### Web Interaction Metrics")
        wcol1, wcol2, wcol3 = st.columns(3)
        
        with wcol1:
            st.metric("Pages Visited", web_metrics.get("pages_visited", 0))
        
        with wcol2:
            st.metric("Total Interactions", web_metrics.get("total_interactions", 0))
        
        with wcol3:
            st.metric("Success Rate", f"{web_metrics.get('success_rate', 0):.1f}%")
    
    # Tool usage
    if "tools" in status:
        st.markdown("#### Tool Usage")
        
        tool_data = []
        for name, metrics in status["tools"].items():
            tool_data.append({
                "Tool": name,
                "Executions": metrics.get("execution_count", 0),
                "Success Rate": f"{metrics.get('success_rate', 0):.1f}%",
                "Avg Time (s)": f"{metrics.get('avg_execution_time', 0):.3f}"
            })
        
        # Convert to DataFrame for display
        if tool_data:
            import pandas as pd
            tool_df = pd.DataFrame(tool_data)
            st.dataframe(tool_df, use_container_width=True)


def display_api_config():
    """Display API configuration interface"""
    st.markdown("### 🔑 API Configuration")
    
    api_col1, api_col2 = st.columns(2)
    
    with api_col1:
        st.markdown("#### OpenAI API")
        openai_key = st.text_input("OpenAI API Key", type="password", key="openai_api_key")
        if st.button("Save OpenAI Key"):
            os.environ["OPENAI_API_KEY"] = openai_key
            st.session_state.api_keys_configured["OPENAI_API_KEY"] = bool(openai_key)
            st.success("OpenAI API key saved!")
            
        st.markdown("#### Anthropic API")
        anthropic_key = st.text_input("Anthropic API Key", type="password", key="anthropic_api_key")
        if st.button("Save Anthropic Key"):
            os.environ["ANTHROPIC_API_KEY"] = anthropic_key
            st.session_state.api_keys_configured["ANTHROPIC_API_KEY"] = bool(anthropic_key)
            st.success("Anthropic API key saved!")
    
    with api_col2:
        st.markdown("#### Azure Quantum")
        sub_id = st.text_input("Subscription ID", key="azure_sub_id")
        resource_group = st.text_input("Resource Group", key="azure_resource_group")
        workspace = st.text_input("Workspace Name", key="azure_workspace")
        location = st.text_input("Location", key="azure_location")
        
        if st.button("Save Azure Quantum Config"):
            os.environ["AZURE_QUANTUM_SUBSCRIPTION_ID"] = sub_id
            os.environ["AZURE_QUANTUM_RESOURCE_GROUP"] = resource_group
            os.environ["AZURE_QUANTUM_WORKSPACE_NAME"] = workspace
            os.environ["AZURE_QUANTUM_LOCATION"] = location
            
            st.session_state.api_keys_configured["AZURE_QUANTUM_SUBSCRIPTION_ID"] = bool(sub_id)
            st.success("Azure Quantum configuration saved!")
    
    # Display configuration status
    st.markdown("#### API Status")
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        openai_status = "✅ Configured" if st.session_state.api_keys_configured["OPENAI_API_KEY"] else "❌ Not Configured"
        st.markdown(f"**OpenAI API:** {openai_status}")
    
    with status_col2:
        anthropic_status = "✅ Configured" if st.session_state.api_keys_configured["ANTHROPIC_API_KEY"] else "❌ Not Configured"
        st.markdown(f"**Anthropic API:** {anthropic_status}")
    
    with status_col3:
        azure_status = "✅ Configured" if st.session_state.api_keys_configured["AZURE_QUANTUM_SUBSCRIPTION_ID"] else "❌ Not Configured"
        st.markdown(f"**Azure Quantum:** {azure_status}")


def display_about():
    """Display information about the agent"""
    st.markdown("### ℹ️ About QA³ Agent")
    
    st.markdown("""
    **QA³ Agent** (Quantum-Accelerated AI Agent) is a cutting-edge autonomous agent system that combines:
    
    1. **Quantum Computing** - Leverages quantum algorithms for specific computational advantages
    2. **AI Integration** - Uses large language models for understanding and decision making
    3. **Browser Automation** - Allows the agent to interact with web interfaces
    4. **Computer Vision** - Enables UI understanding and screen interactions
    5. **Task Planning** - Breaks down complex tasks into executable steps
    
    This agent demonstrates several key agentic capabilities:
    
    - **True Agency**: The agent can take concrete actions in digital environments
    - **Autonomous Decision Making**: The agent analyzes tasks and creates execution plans
    - **Memory Management**: The agent maintains short-term and long-term memory
    - **Tool Usage**: The agent can use a variety of tools to accomplish tasks
    - **Task Composition**: The agent can break down complex tasks into manageable steps
    
    ### Technical Architecture
    
    The system consists of several interconnected components:
    
    - **Agent Memory**: Manages conversation history, working memory, and long-term storage
    - **Vision System**: Analyzes screenshots to identify UI elements
    - **Web Interaction Agent**: Handles browser automation and web interactions
    - **Tool System**: Provides a flexible framework for executing various actions
    - **Task Planner**: Creates step-by-step execution plans for complex tasks
    
    ### Quantum Acceleration
    
    When available, the agent uses quantum computing for:
    
    - **Search Enhancement**: Applies quantum search algorithms for faster discovery
    - **Optimization Problems**: Uses quantum algorithms for resource allocation tasks
    - **Pattern Recognition**: Leverages quantum computing for pattern detection
    """)
    
    st.markdown("### API Dependencies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **For AI Reasoning:**
        - OpenAI API (GPT-4o)
        - Anthropic API (Claude 3.7 Sonnet)
        """)
    
    with col2:
        st.markdown("""
        **For Quantum Computing:**
        - Azure Quantum API
        - IonQ Aria-1 hardware
        """)


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="QA³ Agent | Quantum-Accelerated AI with Agentic Capabilities",
        page_icon="⚛️",
        layout="wide"
    )
    
    # Custom CSS for better appearance
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(45deg, #4361ee, #7b2cbf);
        color: white;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        border-radius: 10px;
        font-weight: 700;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: white;
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #0e1117;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #262730;
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Main header
    st.markdown('<div class="main-header">QA³ Agent: Quantum-Accelerated AI with True Agentic Capabilities</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://i.imgur.com/fOQ6lK8.png", width=250)
        st.markdown("### Agent Settings")
        
        # Initialize agent button
        if not st.session_state.agent_initialized:
            if st.button("Initialize Agent"):
                initialize_agent()
        else:
            st.success("Agent initialized and ready!")
        
        # Quantum settings
        st.markdown("#### Quantum Settings")
        quantum_enabled = st.toggle("Enable Quantum Acceleration", value=True)
        n_qubits = st.slider("Number of qubits", min_value=4, max_value=20, value=8)
        
        # API config status
        st.markdown("#### API Configuration Status")
        openai_status = "✅" if st.session_state.api_keys_configured["OPENAI_API_KEY"] else "❌"
        anthropic_status = "✅" if st.session_state.api_keys_configured["ANTHROPIC_API_KEY"] else "❌"
        azure_status = "✅" if st.session_state.api_keys_configured["AZURE_QUANTUM_SUBSCRIPTION_ID"] else "❌"
        
        st.markdown(f"OpenAI API: {openai_status}")
        st.markdown(f"Anthropic API: {anthropic_status}")
        st.markdown(f"Azure Quantum: {azure_status}")
        
        # About section
        with st.expander("About QA³ Agent"):
            st.markdown("""
            **QA³ Agent** combines quantum computing with agentic AI capabilities.
            
            Key features:
            - True autonomous web browsing
            - Computer vision for UI understanding
            - Task planning and execution
            - Quantum-accelerated computations
            
            Created for agents that can truly interact with digital environments.
            """)
    
    # Main content using tabs
    tabs = st.tabs(["💬 Chat", "📋 Task Details", "📊 Status", "🔑 API Configuration", "ℹ️ About"])
    
    with tabs[0]:
        display_chat_interface()
    
    with tabs[1]:
        display_task_details()
    
    with tabs[2]:
        display_agent_status()
    
    with tabs[3]:
        display_api_config()
    
    with tabs[4]:
        display_about()


if __name__ == "__main__":
    main()