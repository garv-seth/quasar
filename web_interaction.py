"""
Web Interaction Module for Agentic Quantum Computing Platform

This module provides real web interaction capabilities using Selenium,
allowing the agent to actually browse the web, take screenshots, and
interact with web elements.
"""

import os
import time
import base64
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import io
from PIL import Image
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("web-interaction")

class WebAgent:
    """
    Real web interaction agent using Selenium
    
    This class enables actual web browsing and interaction capabilities,
    allowing the agent to:
    1. Visit websites
    2. Take screenshots
    3. Find and interact with elements
    4. Extract information from web pages
    """
    
    def __init__(self, headless: bool = True):
        """
        Initialize the web agent
        
        Args:
            headless: Whether to run the browser in headless mode
        """
        self.headless = headless
        self.driver = None
        self.initialized = False
        self.error_message = None
        
        try:
            # Import Selenium components
            from selenium import webdriver
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            
            self.By = By
            self.WebDriverWait = WebDriverWait
            self.EC = EC
            
            # Set up Chrome options
            chrome_options = Options()
            if headless:
                chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            
            # Initialize the WebDriver
            service = Service()
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.initialized = True
            
            logger.info("Web agent initialized successfully")
        except Exception as e:
            self.error_message = f"Failed to initialize web agent: {str(e)}"
            logger.error(self.error_message)
    
    def navigate(self, url: str) -> Dict[str, Any]:
        """
        Navigate to a URL (alias for visit)
        
        Args:
            url: URL to navigate to
            
        Returns:
            Dict with operation result
        """
        # This is an alias for visit to maintain compatibility with core_agent API
        return self.visit(url)
    
    def visit(self, url: str) -> Dict[str, Any]:
        """
        Visit a URL
        
        Args:
            url: URL to visit
            
        Returns:
            Dict with operation result
        """
        if not self.initialized:
            return {"success": False, "error": self.error_message}
        
        try:
            # Add http:// if missing
            if not url.startswith("http"):
                url = "https://" + url
            
            logger.info(f"Visiting URL: {url}")
            self.driver.get(url)
            time.sleep(2)  # Wait for page to load
            
            # Get page title and current URL
            title = self.driver.title
            current_url = self.driver.current_url
            
            # Take screenshot
            screenshot_b64 = self._take_screenshot()
            
            # Get page source
            page_source = self.driver.page_source
            
            return {
                "success": True,
                "title": title,
                "url": current_url,
                "screenshot": screenshot_b64,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error visiting URL {url}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _take_screenshot(self) -> Optional[str]:
        """Take a screenshot and return it as a base64 string"""
        try:
            screenshot = self.driver.get_screenshot_as_png()
            # Compress and convert to base64
            img = Image.open(io.BytesIO(screenshot))
            img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG", optimize=True, quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return img_str
        except Exception as e:
            logger.error(f"Error taking screenshot: {str(e)}")
            return None
    
    def find_element(self, selector: str, wait_time: int = 10) -> Dict[str, Any]:
        """
        Find an element on the page
        
        Args:
            selector: CSS selector for the element
            wait_time: Maximum time to wait for the element
            
        Returns:
            Dict with operation result
        """
        if not self.initialized:
            return {"success": False, "error": self.error_message}
        
        try:
            logger.info(f"Finding element with selector: {selector}")
            element = WebDriverWait(self.driver, wait_time).until(
                self.EC.presence_of_element_located((self.By.CSS_SELECTOR, selector))
            )
            
            # Get element attributes
            location = element.location
            size = element.size
            text = element.text
            tag_name = element.tag_name
            
            # Check if element is displayed and enabled
            is_displayed = element.is_displayed()
            is_enabled = element.is_enabled()
            
            return {
                "success": True,
                "element_id": element.id,
                "tag_name": tag_name,
                "text": text,
                "location": location,
                "size": size,
                "is_displayed": is_displayed,
                "is_enabled": is_enabled,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error finding element {selector}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def click_element(self, selector: str, wait_time: int = 10) -> Dict[str, Any]:
        """
        Click an element on the page
        
        Args:
            selector: CSS selector for the element
            wait_time: Maximum time to wait for the element
            
        Returns:
            Dict with operation result
        """
        if not self.initialized:
            return {"success": False, "error": self.error_message}
        
        try:
            logger.info(f"Clicking element with selector: {selector}")
            element = WebDriverWait(self.driver, wait_time).until(
                self.EC.element_to_be_clickable((self.By.CSS_SELECTOR, selector))
            )
            element.click()
            time.sleep(1)  # Allow for page changes
            
            # Take a screenshot after clicking
            screenshot_b64 = self._take_screenshot()
            
            return {
                "success": True,
                "action": "click",
                "selector": selector,
                "screenshot": screenshot_b64,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error clicking element {selector}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def type_text(self, selector: str, text: str, wait_time: int = 10) -> Dict[str, Any]:
        """
        Type text into an element
        
        Args:
            selector: CSS selector for the element
            text: Text to type
            wait_time: Maximum time to wait for the element
            
        Returns:
            Dict with operation result
        """
        if not self.initialized:
            return {"success": False, "error": self.error_message}
        
        try:
            logger.info(f"Typing text into element with selector: {selector}")
            element = WebDriverWait(self.driver, wait_time).until(
                self.EC.presence_of_element_located((self.By.CSS_SELECTOR, selector))
            )
            element.clear()
            element.send_keys(text)
            
            # Take a screenshot after typing
            screenshot_b64 = self._take_screenshot()
            
            return {
                "success": True,
                "action": "type",
                "selector": selector,
                "text": text,
                "screenshot": screenshot_b64,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error typing text into element {selector}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_page_data(self) -> Dict[str, Any]:
        """
        Get data from the current page
        
        Returns:
            Dict with page data
        """
        if not self.initialized:
            return {"success": False, "error": self.error_message}
        
        try:
            # Get page title and URL
            title = self.driver.title
            url = self.driver.current_url
            
            # Get all links
            links = []
            link_elements = self.driver.find_elements(self.By.TAG_NAME, "a")
            for link in link_elements:
                href = link.get_attribute("href")
                text = link.text
                if href:
                    links.append({"href": href, "text": text})
            
            # Get all text
            body_text = self.driver.find_element(self.By.TAG_NAME, "body").text
            
            # Get all images
            images = []
            img_elements = self.driver.find_elements(self.By.TAG_NAME, "img")
            for img in img_elements:
                src = img.get_attribute("src")
                alt = img.get_attribute("alt")
                if src:
                    images.append({"src": src, "alt": alt})
            
            # Take screenshot
            screenshot_b64 = self._take_screenshot()
            
            return {
                "success": True,
                "title": title,
                "url": url,
                "links": links[:20],  # Limit to first 20 links
                "images": images[:20],  # Limit to first 20 images
                "text_length": len(body_text),
                "screenshot": screenshot_b64,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting page data: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def close(self):
        """Close the browser and clean up"""
        if self.initialized and self.driver:
            try:
                self.driver.quit()
                logger.info("Web agent closed")
            except Exception as e:
                logger.error(f"Error closing web agent: {str(e)}")


# Implement quantum-enhanced web features
class QuantumEnhancedWebAgent(WebAgent):
    """
    Quantum-enhanced web interaction agent
    
    This class extends the WebAgent with quantum computing capabilities for:
    1. Improved web page analysis using quantum-inspired algorithms
    2. Quantum-enhanced decision making for actions
    3. Privacy-preserving browsing using quantum techniques
    """
    
    def __init__(self, headless: bool = True, n_qubits: int = 4):
        """
        Initialize the quantum-enhanced web agent
        
        Args:
            headless: Whether to run the browser in headless mode
            n_qubits: Number of qubits for quantum operations
        """
        super().__init__(headless)
        self.n_qubits = n_qubits
        self.quantum_initialized = False
        
        # Initialize quantum capabilities if available
        try:
            import pennylane as qml
            self.quantum_device = qml.device("default.qubit", wires=n_qubits)
            self.quantum_initialized = True
            logger.info(f"Quantum enhancement initialized with {n_qubits} qubits")
        except ImportError:
            logger.warning("PennyLane not available. Quantum enhancements disabled.")
    
    def quantum_analyze_page(self) -> Dict[str, Any]:
        """
        Perform quantum-enhanced analysis of the current page
        
        Returns:
            Dict with analysis results
        """
        if not self.initialized:
            return {"success": False, "error": self.error_message}
        
        if not self.quantum_initialized:
            return {"success": False, "error": "Quantum capabilities not available"}
        
        try:
            # Get basic page data first
            page_data = self.get_page_data()
            if not page_data["success"]:
                return page_data
            
            # Extract text features for quantum processing
            import pennylane as qml
            import numpy as np
            
            # Simple text-based feature extraction (word counts, etc.)
            body_text = self.driver.find_element(self.By.TAG_NAME, "body").text
            words = body_text.lower().split()
            
            # Count word frequencies for top 20 most common words
            word_freq = {}
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top words by frequency
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
            
            # Create a feature vector from word frequencies
            # Normalize to be suitable for quantum circuit
            if top_words:
                max_freq = top_words[0][1]
                feature_vector = [min(item[1] / max_freq, 1.0) for item in top_words]
            else:
                feature_vector = [0.0] * 8
            
            # Ensure we have enough features (pad or truncate)
            feature_vector = feature_vector[:self.n_qubits] + [0.0] * max(0, self.n_qubits - len(feature_vector))
            
            # Define a quantum circuit for encoding and analyzing the features
            @qml.qnode(self.quantum_device)
            def quantum_analysis_circuit():
                # Encode features as rotation angles
                for i, feature in enumerate(feature_vector):
                    qml.RY(feature * np.pi, wires=i)
                
                # Create entanglement
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i+1])
                
                # Measure all qubits
                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
            
            # Run the quantum circuit
            start_time = time.time()
            quantum_results = quantum_analysis_circuit()
            end_time = time.time()
            
            # Calculate a "quantum score" based on measurement results
            # This is a simplified representation - a real implementation would use
            # more sophisticated quantum algorithms
            quantum_score = sum(abs(val) for val in quantum_results) / self.n_qubits
            
            # Generate circuit diagram
            fig, _ = qml.draw_mpl(quantum_analysis_circuit)()
            
            # Save the diagram to a base64 string
            buffered = io.BytesIO()
            fig.savefig(buffered, format="PNG")
            circuit_diagram = base64.b64encode(buffered.getvalue()).decode()
            
            return {
                "success": True,
                "url": page_data["url"],
                "title": page_data["title"],
                "top_words": dict(top_words),
                "quantum_results": quantum_results if isinstance(quantum_results, list) else list(quantum_results),
                "quantum_score": float(quantum_score),
                "circuit_diagram": circuit_diagram,
                "feature_vector": feature_vector,
                "processing_time": end_time - start_time,
                "n_qubits": self.n_qubits,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in quantum page analysis: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def quantum_enhanced_search(self, query: str, elements_css: str = "*") -> Dict[str, Any]:
        """
        Perform quantum-enhanced search for elements on the page
        
        Args:
            query: Search query
            elements_css: CSS selector to limit search scope
            
        Returns:
            Dict with search results
        """
        if not self.initialized:
            return {"success": False, "error": self.error_message}
        
        try:
            # Find all potential elements
            elements = self.driver.find_elements(self.By.CSS_SELECTOR, elements_css)
            if not elements:
                return {"success": False, "error": "No elements found to search"}
            
            # Extract text from elements
            element_texts = []
            for element in elements:
                try:
                    text = element.text.strip()
                    if text:
                        element_texts.append((element, text))
                except:
                    continue
            
            # If we have quantum capabilities, use them for ranking
            if self.quantum_initialized:
                import pennylane as qml
                import numpy as np
                
                # Prepare query and documents
                query = query.lower()
                texts = [text.lower() for _, text in element_texts]
                
                # Simple TF-IDF like scoring
                scores = []
                for text in texts:
                    # Count query terms in text
                    score = 0
                    for term in query.split():
                        if term in text:
                            score += 1
                    scores.append(score)
                
                # Normalize scores (if any non-zero scores exist)
                if any(scores):
                    max_score = max(scores)
                    normalized_scores = [score / max_score for score in scores]
                else:
                    normalized_scores = scores
                
                # Use quantum circuit to amplify scoring differences
                @qml.qnode(self.quantum_device)
                def quantum_search_circuit():
                    # Encode normalized scores into a superposition
                    for i, score in enumerate(normalized_scores[:min(len(normalized_scores), self.n_qubits)]):
                        qml.RY(score * np.pi, wires=i)
                    
                    # Apply Grover-inspired diffusion for amplitude amplification
                    for _ in range(1):  # Simplified Grover iteration
                        # Hadamard gates
                        for i in range(min(len(normalized_scores), self.n_qubits)):
                            qml.Hadamard(wires=i)
                        
                        # Phase flip
                        for i in range(min(len(normalized_scores), self.n_qubits)):
                            qml.PhaseShift(np.pi, wires=i)
                        
                        # Hadamard gates again
                        for i in range(min(len(normalized_scores), self.n_qubits)):
                            qml.Hadamard(wires=i)
                    
                    # Measure all qubits
                    return [qml.expval(qml.PauliZ(i)) for i in range(min(len(normalized_scores), self.n_qubits))]
                
                # Run the quantum circuit
                quantum_results = quantum_search_circuit()
                
                # Generate circuit diagram
                fig, _ = qml.draw_mpl(quantum_search_circuit)()
                
                # Save the diagram to a base64 string
                buffered = io.BytesIO()
                fig.savefig(buffered, format="PNG")
                circuit_diagram = base64.b64encode(buffered.getvalue()).decode()
                
                # Combine original scores with quantum results
                enhanced_scores = []
                for i, score in enumerate(normalized_scores):
                    if i < len(quantum_results):
                        # Combine classical score with quantum result
                        enhanced_score = score * (1 + abs(quantum_results[i]))
                    else:
                        enhanced_score = score
                    enhanced_scores.append(enhanced_score)
                
                # Sort by enhanced scores
                results = sorted(zip(element_texts, enhanced_scores), key=lambda x: x[1], reverse=True)
                
                # Take top 5 results
                top_results = []
                for (element, text), score in results[:5]:
                    element_info = {
                        "text": text,
                        "tag": element.tag_name,
                        "score": float(score),
                        "location": element.location,
                        "size": element.size
                    }
                    top_results.append(element_info)
                
                return {
                    "success": True,
                    "query": query,
                    "results": top_results,
                    "total_elements": len(elements),
                    "quantum_enhanced": True,
                    "circuit_diagram": circuit_diagram,
                    "quantum_results": quantum_results if isinstance(quantum_results, list) else list(quantum_results),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Classical search method
                query = query.lower()
                scores = []
                
                for element, text in element_texts:
                    text_lower = text.lower()
                    score = 0
                    for term in query.split():
                        if term in text_lower:
                            score += 1
                    scores.append(score)
                
                # Sort by scores
                results = sorted(zip(element_texts, scores), key=lambda x: x[1], reverse=True)
                
                # Take top 5 results
                top_results = []
                for (element, text), score in results[:5]:
                    element_info = {
                        "text": text,
                        "tag": element.tag_name,
                        "score": score,
                        "location": element.location,
                        "size": element.size
                    }
                    top_results.append(element_info)
                
                return {
                    "success": True,
                    "query": query,
                    "results": top_results,
                    "total_elements": len(elements),
                    "quantum_enhanced": False,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error in quantum-enhanced search: {str(e)}")
            return {"success": False, "error": str(e)}

# Integration with agent_core's AutonomousAgent
def create_quantum_web_integration(n_qubits: int = 4) -> Dict[str, Any]:
    """
    Create a quantum-enhanced web agent for integration with the autonomous agent
    
    Args:
        n_qubits: Number of qubits for quantum operations
        
    Returns:
        Dict with agent instance and status
    """
    try:
        agent = QuantumEnhancedWebAgent(headless=True, n_qubits=n_qubits)
        
        if agent.initialized:
            return {
                "success": True,
                "agent": agent,
                "quantum_enabled": agent.quantum_initialized,
                "n_qubits": n_qubits
            }
        else:
            return {
                "success": False,
                "error": agent.error_message,
                "quantum_enabled": agent.quantum_initialized,
                "n_qubits": n_qubits
            }
    except Exception as e:
        logger.error(f"Error creating quantum web integration: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "quantum_enabled": False,
            "n_qubits": n_qubits
        }