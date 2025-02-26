"""Browser interaction module enabling Q3A agents to interact with web content."""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
import os
import json
import time
import re
from dataclasses import dataclass
import base64

# Playwright for browser automation
from playwright.async_api import async_playwright, Page, Browser, BrowserContext, ElementHandle

# Quantum enhanced processing
import pennylane as qml
import numpy as np


@dataclass
class BrowserState:
    """Maintains the current state of the browser session."""
    current_url: str = ""
    page_title: str = ""
    tabs: List[Dict[str, str]] = None
    navigation_history: List[str] = None
    screenshots: Dict[str, str] = None  # URL -> base64 screenshot
    dom_structure: Dict[str, Any] = None
    forms_found: List[Dict[str, Any]] = None
    links_found: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize collections if not provided."""
        if self.tabs is None:
            self.tabs = []
        if self.navigation_history is None:
            self.navigation_history = []
        if self.screenshots is None:
            self.screenshots = {}
        if self.dom_structure is None:
            self.dom_structure = {}
        if self.forms_found is None:
            self.forms_found = []
        if self.links_found is None:
            self.links_found = []


class QuantumEnhancedBrowser:
    """Browser interface with quantum-enhanced content processing capabilities."""
    
    def __init__(self, n_qubits: int = 8, use_quantum: bool = True, headless: bool = True):
        """Initialize the browser interface."""
        self.n_qubits = n_qubits
        self.use_quantum = use_quantum
        self.headless = headless
        
        # Browser components
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        
        # State tracking
        self.state = BrowserState()
        
        # Quantum processing
        if self.use_quantum:
            self._setup_quantum_circuits()
        
        # Performance metrics
        self.metrics = {
            "pages_visited": 0,
            "forms_submitted": 0,
            "screenshots_taken": 0,
            "quantum_enhancements_applied": 0,
            "navigation_time": 0,
            "processing_time": 0
        }
        
        logging.info(f"QuantumEnhancedBrowser initialized with {n_qubits} qubits, quantum {'enabled' if use_quantum else 'disabled'}")
    
    def _setup_quantum_circuits(self):
        """Setup quantum circuits for enhanced web content processing."""
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
        @qml.qnode(self.dev)
        def element_priority_circuit(features, weights):
            """Quantum circuit for prioritizing web elements."""
            # Normalize inputs
            features = features / np.linalg.norm(features)
            
            # Amplitude encoding
            for i, feature in enumerate(features):
                if i < self.n_qubits:
                    qml.RY(feature * np.pi, wires=i)
            
            # Apply entangling layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
            
            # Apply weighted rotation layer
            for i, weight in enumerate(weights):
                if i < self.n_qubits:
                    qml.RZ(weight * np.pi, wires=i)
            
            # Measurement in computational basis
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        @qml.qnode(self.dev)
        def content_relevance_circuit(page_vector, query_vector):
            """Quantum circuit for determining content relevance."""
            # Normalize vectors
            page_vector = page_vector / np.linalg.norm(page_vector)
            query_vector = query_vector / np.linalg.norm(query_vector)
            
            # Encode page content in first half of qubits
            for i, val in enumerate(page_vector):
                if i < self.n_qubits // 2:
                    qml.RY(val * np.pi, wires=i)
            
            # Encode query in second half of qubits
            for i, val in enumerate(query_vector):
                if i < self.n_qubits // 2:
                    qml.RY(val * np.pi, wires=i + self.n_qubits // 2)
            
            # Apply entangling operations between corresponding qubits
            for i in range(self.n_qubits // 2):
                qml.CNOT(wires=[i, i + self.n_qubits // 2])
            
            # Apply Hadamard to measure interference
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            
            # Measure
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        self.element_priority_circuit = element_priority_circuit
        self.content_relevance_circuit = content_relevance_circuit
        
        logging.info("Quantum circuits initialized for web content processing")
    
    async def _initialize_browser(self):
        """Initialize the browser if not already initialized."""
        if not self.playwright:
            self.playwright = await async_playwright().start()
            
            # Launch browser
            self.browser = await self.playwright.chromium.launch(headless=self.headless)
            
            # Create a new browser context
            self.context = await self.browser.new_context(
                viewport={"width": 1280, "height": 800},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Q3A Quantum Browser/1.0"
            )
            
            # Create a new page
            self.page = await self.context.new_page()
            
            # Setup event listeners
            self.page.on("load", self._on_page_load)
            
            logging.info("Browser initialized successfully")
    
    async def _on_page_load(self):
        """Handle page load events."""
        await self._update_state()
    
    async def _update_state(self):
        """Update browser state after navigation or interaction."""
        if not self.page:
            return
        
        # Update current URL and title
        self.state.current_url = self.page.url
        self.state.page_title = await self.page.title()
        
        # Update navigation history
        if self.state.current_url and self.state.current_url not in self.state.navigation_history:
            self.state.navigation_history.append(self.state.current_url)
        
        # Take screenshot
        screenshot = await self.page.screenshot(type="jpeg", quality=50)
        self.state.screenshots[self.state.current_url] = base64.b64encode(screenshot).decode("utf-8")
        self.metrics["screenshots_taken"] += 1
        
        # Extract page structure and elements
        await self._extract_page_structure()
    
    async def _extract_page_structure(self):
        """Extract and analyze page structure."""
        if not self.page:
            return
        
        # Get links
        links = await self.page.query_selector_all("a")
        self.state.links_found = [
            {
                "text": await link.text_content() or "No text", 
                "href": await link.get_attribute("href") or "#",
                "visible": await link.is_visible()
            } 
            for link in links
        ]
        
        # Get forms
        forms = await self.page.query_selector_all("form")
        self.state.forms_found = []
        
        for form in forms:
            form_elements = []
            
            # Get inputs
            inputs = await form.query_selector_all("input, textarea, select")
            for input_el in inputs:
                input_type = await input_el.get_attribute("type") or "text"
                name = await input_el.get_attribute("name") or ""
                placeholder = await input_el.get_attribute("placeholder") or ""
                
                form_elements.append({
                    "type": input_type,
                    "name": name,
                    "placeholder": placeholder,
                    "required": await input_el.get_attribute("required") == "true"
                })
            
            self.state.forms_found.append({
                "id": await form.get_attribute("id") or "",
                "action": await form.get_attribute("action") or "",
                "method": await form.get_attribute("method") or "get",
                "elements": form_elements
            })
        
        # Simplified DOM structure for state tracking
        self.state.dom_structure = await self.page.evaluate("""() => {
            const getBasicStructure = (el, depth = 0, maxDepth = 3) => {
                if (depth > maxDepth) return { type: el.tagName.toLowerCase() };
                
                const children = Array.from(el.children).map(child => 
                    getBasicStructure(child, depth + 1, maxDepth)
                );
                
                return {
                    type: el.tagName.toLowerCase(),
                    id: el.id || undefined,
                    class: el.className || undefined,
                    children: children.length > 0 ? children : undefined
                };
            };
            
            return getBasicStructure(document.body);
        }""")
    
    async def navigate(self, url: str) -> Dict[str, Any]:
        """Navigate to a URL."""
        start_time = time.time()
        
        try:
            # Initialize browser if needed
            if not self.browser:
                await self._initialize_browser()
            
            # Add http:// prefix if missing
            if not url.startswith("http"):
                url = "https://" + url
            
            # Navigate to the page
            response = await self.page.goto(url, wait_until="networkidle")
            
            # Update state
            await self._update_state()
            
            self.metrics["pages_visited"] += 1
            self.metrics["navigation_time"] += time.time() - start_time
            
            # Check for error responses
            if response and response.status >= 400:
                return {
                    "success": False,
                    "url": url,
                    "status": response.status,
                    "error": f"HTTP error: {response.status}",
                    "state": {
                        "current_url": self.state.current_url,
                        "page_title": self.state.page_title,
                    }
                }
            
            # Process page with quantum enhancement if enabled
            content_relevance = None
            if self.use_quantum:
                content_relevance = await self._quantum_analyze_page()
            
            return {
                "success": True,
                "url": url,
                "status": response.status if response else 200,
                "title": self.state.page_title,
                "links_count": len(self.state.links_found),
                "forms_count": len(self.state.forms_found),
                "content_relevance": content_relevance
            }
            
        except Exception as e:
            logging.error(f"Navigation error to {url}: {str(e)}")
            return {
                "success": False,
                "url": url,
                "error": str(e)
            }
    
    async def _quantum_analyze_page(self) -> Dict[str, Any]:
        """Apply quantum processing to analyze page content."""
        if not self.use_quantum:
            return None
            
        start_time = time.time()
        
        try:
            # Extract page text content
            page_text = await self.page.evaluate("""() => {
                return document.body.innerText;
            }""")
            
            # Tokenize and vectorize the content (simplified version)
            tokens = re.findall(r'\b\w+\b', page_text.lower())
            word_counts = {}
            for word in tokens:
                if len(word) > 2:  # Filter out very short words
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            # Create a simple frequency vector (up to n_qubits dimensions)
            top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:self.n_qubits]
            if not top_words:
                return None
                
            # Create word frequency vector
            frequency_vector = [count / max(1, len(tokens)) for _, count in top_words]
            
            # Pad vector if needed
            while len(frequency_vector) < self.n_qubits:
                frequency_vector.append(0.0)
            
            # Truncate if too long
            frequency_vector = frequency_vector[:self.n_qubits]
            
            # Create a "general relevance" vector (could be customized based on user query)
            relevance_vector = np.ones(self.n_qubits) / self.n_qubits
            
            # Apply quantum circuit for content analysis
            quantum_result = self.content_relevance_circuit(
                np.array(frequency_vector), 
                np.array(relevance_vector)
            )
            
            # Calculate overall relevance score (normalized between 0 and 1)
            relevance_score = (np.mean(quantum_result) + 1) / 2
            
            processing_time = time.time() - start_time
            self.metrics["processing_time"] += processing_time
            self.metrics["quantum_enhancements_applied"] += 1
            
            return {
                "relevance_score": float(relevance_score),
                "top_words": [word for word, _ in top_words[:10]],
                "processing_time": processing_time
            }
            
        except Exception as e:
            logging.error(f"Error in quantum page analysis: {str(e)}")
            return None
    
    async def _quantum_prioritize_elements(self, elements: List[Dict[str, Any]], objective: str) -> List[Dict[str, Any]]:
        """Use quantum processing to prioritize web elements based on an objective."""
        if not self.use_quantum or not elements:
            return elements
            
        try:
            # Create feature vectors for each element
            feature_vectors = []
            
            for element in elements:
                features = [
                    int(element.get("visible", False)) * 0.8,  # Visibility is important
                    min(len(element.get("text", "")), 100) / 100 * 0.6,  # Text length
                    0.5,  # Base relevance
                    0.7 if objective.lower() in element.get("text", "").lower() else 0.2  # Objective matching
                ]
                
                # Pad to n_qubits
                while len(features) < self.n_qubits:
                    features.append(0.0)
                
                # Truncate if needed
                features = features[:self.n_qubits]
                feature_vectors.append(features)
            
            # Apply quantum circuit to each element
            weights = np.ones(self.n_qubits) / self.n_qubits  # Uniform weights for now
            
            prioritized_elements = []
            for i, element in enumerate(elements):
                priority_score = np.mean(self.element_priority_circuit(
                    np.array(feature_vectors[i]), 
                    np.array(weights)
                ))
                
                # Normalize score between 0 and 1
                priority_score = (priority_score + 1) / 2
                
                prioritized_elements.append({
                    **element,
                    "priority_score": float(priority_score)
                })
            
            # Sort by priority score
            prioritized_elements.sort(key=lambda x: x["priority_score"], reverse=True)
            
            self.metrics["quantum_enhancements_applied"] += 1
            return prioritized_elements
            
        except Exception as e:
            logging.error(f"Error in quantum element prioritization: {str(e)}")
            return elements
    
    async def search_element(self, selector: str) -> Dict[str, Any]:
        """Find an element on the page using a CSS selector."""
        if not self.page:
            return {"success": False, "error": "Browser not initialized"}
            
        try:
            element = await self.page.query_selector(selector)
            if not element:
                return {"success": False, "error": f"Element not found: {selector}"}
                
            # Get element properties
            properties = await element.evaluate("""(el) => {
                const rect = el.getBoundingClientRect();
                return {
                    tagName: el.tagName.toLowerCase(),
                    text: el.innerText || el.textContent || "",
                    value: el.value || "",
                    visible: window.getComputedStyle(el).display !== 'none',
                    attributes: Array.from(el.attributes).reduce((obj, attr) => {
                        obj[attr.name] = attr.value;
                        return obj;
                    }, {}),
                    position: {
                        x: rect.x,
                        y: rect.y,
                        width: rect.width,
                        height: rect.height
                    }
                };
            }""")
            
            return {
                "success": True,
                "selector": selector,
                "properties": properties
            }
            
        except Exception as e:
            logging.error(f"Error searching for element {selector}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def fill_form(self, form_selector: str, data: Dict[str, str]) -> Dict[str, Any]:
        """Fill form fields with provided data."""
        if not self.page:
            return {"success": False, "error": "Browser not initialized"}
            
        try:
            # Find the form
            form = await self.page.query_selector(form_selector)
            if not form:
                return {"success": False, "error": f"Form not found: {form_selector}"}
            
            filled_fields = []
            failed_fields = []
            
            # Fill each field
            for field_name, field_value in data.items():
                try:
                    # Try to find the field in various ways
                    field_selector = f"{form_selector} [name='{field_name}']"
                    await self.page.fill(field_selector, field_value)
                    filled_fields.append(field_name)
                except Exception as field_error:
                    try:
                        # Try alternative selector with ID
                        alt_selector = f"{form_selector} #{field_name}"
                        await self.page.fill(alt_selector, field_value)
                        filled_fields.append(field_name)
                    except Exception:
                        failed_fields.append({
                            "field": field_name,
                            "error": str(field_error)
                        })
            
            self.metrics["forms_submitted"] += 1
            
            return {
                "success": len(failed_fields) == 0,
                "form_selector": form_selector,
                "filled_fields": filled_fields,
                "failed_fields": failed_fields
            }
            
        except Exception as e:
            logging.error(f"Error filling form {form_selector}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def submit_form(self, form_selector: str, wait_for_navigation: bool = True) -> Dict[str, Any]:
        """Submit a form and wait for navigation if specified."""
        if not self.page:
            return {"success": False, "error": "Browser not initialized"}
            
        try:
            # Click the submit button within the form
            submit_selector = f"{form_selector} [type='submit']"
            
            try:
                if wait_for_navigation:
                    async with self.page.expect_navigation():
                        await self.page.click(submit_selector)
                else:
                    await self.page.click(submit_selector)
            except Exception as click_error:
                # Try to submit the form directly
                try:
                    await self.page.evaluate(f"""() => {{
                        document.querySelector("{form_selector}").submit();
                    }}""")
                except Exception as submit_error:
                    return {"success": False, "error": f"Failed to submit form: {str(submit_error)}"}
            
            # Update state
            await self._update_state()
            
            return {
                "success": True,
                "current_url": self.state.current_url,
                "page_title": self.state.page_title
            }
            
        except Exception as e:
            logging.error(f"Error submitting form {form_selector}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def click(self, selector: str, wait_for_navigation: bool = False) -> Dict[str, Any]:
        """Click on an element."""
        if not self.page:
            return {"success": False, "error": "Browser not initialized"}
            
        try:
            # Click the element
            if wait_for_navigation:
                async with self.page.expect_navigation():
                    await self.page.click(selector)
            else:
                await self.page.click(selector)
            
            # Update state
            await self._update_state()
            
            return {
                "success": True,
                "selector": selector,
                "current_url": self.state.current_url,
                "page_title": self.state.page_title
            }
            
        except Exception as e:
            logging.error(f"Error clicking element {selector}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def get_page_content(self) -> Dict[str, Any]:
        """Get the current page content."""
        if not self.page:
            return {"success": False, "error": "Browser not initialized"}
            
        try:
            # Get page text content
            text_content = await self.page.evaluate("""() => document.body.innerText""")
            
            # Get page HTML (limited to avoid huge responses)
            html_content = await self.page.content()
            
            # Take screenshot
            screenshot = await self.page.screenshot(type="jpeg", quality=50)
            screenshot_base64 = base64.b64encode(screenshot).decode("utf-8")
            
            # Apply quantum content analysis if enabled
            content_analysis = None
            if self.use_quantum:
                content_analysis = await self._quantum_analyze_page()
                
            return {
                "success": True,
                "url": self.state.current_url,
                "title": self.state.page_title,
                "text_content": text_content[:10000] + "..." if len(text_content) > 10000 else text_content,
                "html_preview": html_content[:5000] + "..." if len(html_content) > 5000 else html_content,
                "screenshot_base64": screenshot_base64,
                "links_count": len(self.state.links_found),
                "forms_count": len(self.state.forms_found),
                "quantum_analysis": content_analysis
            }
            
        except Exception as e:
            logging.error(f"Error getting page content: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def find_links(self, query: str = None) -> Dict[str, Any]:
        """Find links on the page, optionally filtered by a search query."""
        if not self.page:
            return {"success": False, "error": "Browser not initialized"}
            
        try:
            links = self.state.links_found
            
            # Filter links if query provided
            if query:
                filtered_links = [
                    link for link in links 
                    if query.lower() in link.get("text", "").lower() or 
                       query.lower() in link.get("href", "").lower()
                ]
                
                # Use quantum prioritization if enabled
                if self.use_quantum:
                    filtered_links = await self._quantum_prioritize_elements(filtered_links, query)
                else:
                    # Simple classical prioritization
                    for link in filtered_links:
                        score = 0
                        if query.lower() in link.get("text", "").lower():
                            score += 0.6
                        if query.lower() in link.get("href", "").lower():
                            score += 0.4
                        link["priority_score"] = score
                    filtered_links.sort(key=lambda x: x.get("priority_score", 0), reverse=True)
            else:
                filtered_links = links
                
            return {
                "success": True,
                "total_links": len(links),
                "found_links": len(filtered_links),
                "links": filtered_links[:20]  # Limit to 20 links to avoid huge responses
            }
            
        except Exception as e:
            logging.error(f"Error finding links: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def find_forms(self) -> Dict[str, Any]:
        """Find forms on the page."""
        if not self.page:
            return {"success": False, "error": "Browser not initialized"}
            
        try:
            return {
                "success": True,
                "forms_count": len(self.state.forms_found),
                "forms": self.state.forms_found
            }
            
        except Exception as e:
            logging.error(f"Error finding forms: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def extract_tables(self) -> Dict[str, Any]:
        """Extract tables from the page."""
        if not self.page:
            return {"success": False, "error": "Browser not initialized"}
            
        try:
            # Execute JavaScript to extract tables
            tables = await self.page.evaluate("""() => {
                const extractTable = (table) => {
                    const rows = Array.from(table.rows);
                    const headers = Array.from(rows[0]?.cells || []).map(cell => cell.innerText.trim());
                    
                    const data = rows.slice(1).map(row => {
                        const cells = Array.from(row.cells).map(cell => cell.innerText.trim());
                        return headers.reduce((obj, header, i) => {
                            obj[header || `Column${i+1}`] = cells[i] || '';
                            return obj;
                        }, {});
                    });
                    
                    return {
                        headers,
                        data,
                        rowCount: rows.length - 1,
                        columnCount: headers.length
                    };
                };
                
                return Array.from(document.querySelectorAll('table')).map((table, index) => ({
                    tableIndex: index,
                    caption: table.caption?.innerText || `Table ${index+1}`,
                    content: extractTable(table)
                }));
            }""")
            
            return {
                "success": True,
                "tables_count": len(tables),
                "tables": tables
            }
            
        except Exception as e:
            logging.error(f"Error extracting tables: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def take_screenshot(self, full_page: bool = False) -> Dict[str, Any]:
        """Take a screenshot of the current page."""
        if not self.page:
            return {"success": False, "error": "Browser not initialized"}
            
        try:
            screenshot = await self.page.screenshot(full_page=full_page, type="jpeg", quality=70)
            screenshot_base64 = base64.b64encode(screenshot).decode("utf-8")
            
            self.metrics["screenshots_taken"] += 1
            
            return {
                "success": True,
                "full_page": full_page,
                "url": self.state.current_url,
                "title": self.state.page_title,
                "screenshot_base64": screenshot_base64
            }
            
        except Exception as e:
            logging.error(f"Error taking screenshot: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def get_browser_state(self) -> Dict[str, Any]:
        """Get the current browser state."""
        if not self.page:
            return {"success": False, "error": "Browser not initialized"}
            
        return {
            "success": True,
            "current_url": self.state.current_url,
            "page_title": self.state.page_title,
            "navigation_history": self.state.navigation_history[-10:],  # Last 10 URLs
            "tabs_count": len(self.state.tabs),
            "forms_count": len(self.state.forms_found),
            "links_count": len(self.state.links_found)
        }
    
    async def close(self):
        """Close the browser and clean up resources."""
        try:
            if self.page:
                await self.page.close()
                
            if self.context:
                await self.context.close()
                
            if self.browser:
                await self.browser.close()
                
            if self.playwright:
                await self.playwright.stop()
                
            logging.info("Browser closed successfully")
            
        except Exception as e:
            logging.error(f"Error closing browser: {str(e)}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the browser."""
        return self.metrics.copy()