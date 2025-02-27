"""
Web Interaction Agent for Autonomous Computing

This module provides true agentic capabilities through autonomous web browsing
and interaction with web interfaces.
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
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("web-interaction-agent")

# Playwright is used for web automation
try:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not available. Web interaction capabilities disabled.")

# BeautifulSoup for HTML parsing
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logger.warning("BeautifulSoup not available. HTML parsing capabilities limited.")

class WebInteractionAgent:
    """
    Handles autonomous web browsing and interactions
    
    This is the core component that enables true agency by allowing the AI
    to interact with web interfaces, execute searches, fill forms, and
    extract information directly from web pages.
    """
    
    def __init__(self):
        """Initialize the web interaction agent"""
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        
        # Status tracking
        self.initialized = False
        self.current_url = None
        self.navigation_history = []
        
        # Metrics tracking
        self.metrics = {
            "pages_visited": 0,
            "interactions": 0,
            "searches": 0,
            "form_fills": 0,
            "errors": 0,
            "start_time": None,
            "last_action_time": None
        }
        
    async def initialize(self, headless: bool = True):
        """Initialize browser with Playwright for automation"""
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError("Playwright is required for web interaction but is not available.")
            
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(headless=headless)
            self.context = await self.browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
            self.page = await self.context.new_page()
            
            # Set up event handlers
            self.page.on("dialog", self._handle_dialog)
            
            # Initialize metrics
            self.metrics["start_time"] = time.time()
            self.metrics["last_action_time"] = time.time()
            
            self.initialized = True
            logger.info("Web interaction agent initialized successfully")
            
            return {
                "success": True,
                "message": "Web interaction agent initialized successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize web interaction agent: {str(e)}")
            self.metrics["errors"] += 1
            
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to initialize web interaction agent"
            }
            
    async def _handle_dialog(self, dialog):
        """Handle browser dialogs automatically"""
        dialog_type = dialog.type
        message = dialog.message
        
        logger.info(f"Dialog detected: {dialog_type} - {message}")
        
        # Automatically dismiss alerts and confirmations
        if dialog_type in ["alert", "confirm"]:
            await dialog.accept()
        # Automatically dismiss prompts with empty string
        elif dialog_type == "prompt":
            await dialog.accept("")
        # Automatically dismiss others
        else:
            await dialog.dismiss()
            
    async def close(self):
        """Close browser and clean up resources"""
        if self.browser:
            await self.browser.close()
            
        if self.playwright:
            await self.playwright.stop()
            
        self.initialized = False
        logger.info("Web interaction agent closed")
        
    async def navigate(self, url: str) -> Dict[str, Any]:
        """Navigate to a URL and extract page information"""
        if not self.initialized:
            return {"success": False, "error": "Web interaction agent not initialized"}
            
        self.metrics["last_action_time"] = time.time()
        
        try:
            # Ensure URL has protocol
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                
            # Navigate to the URL
            response = await self.page.goto(url, wait_until="networkidle", timeout=30000)
            
            # Wait for page to be fully loaded
            await self.page.wait_for_load_state("networkidle")
            
            # Get current URL (might be different after redirects)
            self.current_url = self.page.url
            
            # Add to navigation history
            self.navigation_history.append({
                "url": self.current_url,
                "timestamp": datetime.now().isoformat(),
                "title": await self.page.title()
            })
            
            # Update metrics
            self.metrics["pages_visited"] += 1
            self.metrics["interactions"] += 1
            
            # Extract page information
            title = await self.page.title()
            
            # Take screenshot
            screenshot = await self.page.screenshot()
            screenshot_base64 = base64.b64encode(screenshot).decode('utf-8')
            
            # Extract HTML content
            html_content = await self.page.content()
            
            # Parse HTML to extract useful information
            page_info = self._parse_html(html_content)
            
            return {
                "success": True,
                "url": self.current_url,
                "title": title,
                "status": response.status,
                "screenshot": screenshot_base64,
                "links": page_info["links"],
                "forms": page_info["forms"],
                "headings": page_info["headings"],
                "main_content": page_info["main_content"]
            }
            
        except PlaywrightTimeoutError:
            self.metrics["errors"] += 1
            return {
                "success": False,
                "error": "Navigation timed out",
                "url": url
            }
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"Error navigating to {url}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "url": url
            }
            
    def _parse_html(self, html_content: str) -> Dict[str, Any]:
        """
        Parse HTML content to extract structured information
        
        Returns:
            Dict containing extracted links, forms, headings, and main content
        """
        if not BS4_AVAILABLE:
            # Simple regex-based extraction as fallback
            links = []
            for match in re.finditer(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html_content):
                links.append({
                    "href": match.group(1),
                    "text": re.sub(r'<[^>]+>', '', match.group(2)).strip()
                })
                
            return {
                "links": links[:20],  # Limit to first 20 links
                "forms": [],
                "headings": [],
                "main_content": html_content[:1000]  # First 1000 chars as preview
            }
            
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract links
            links = []
            for a in soup.find_all('a', href=True):
                link_text = a.get_text().strip()
                if link_text and len(link_text) < 200:  # Exclude empty or overly long link text
                    links.append({
                        "href": a['href'],
                        "text": link_text
                    })
                    
            # Extract forms
            forms = []
            for idx, form in enumerate(soup.find_all('form')):
                form_inputs = []
                for input_tag in form.find_all(['input', 'textarea', 'select']):
                    input_type = input_tag.get('type', input_tag.name)
                    input_name = input_tag.get('name', '')
                    input_id = input_tag.get('id', '')
                    input_placeholder = input_tag.get('placeholder', '')
                    
                    form_inputs.append({
                        "type": input_type,
                        "name": input_name,
                        "id": input_id,
                        "placeholder": input_placeholder
                    })
                    
                form_action = form.get('action', '')
                form_method = form.get('method', 'get')
                form_id = form.get('id', f'form_{idx}')
                
                forms.append({
                    "id": form_id,
                    "action": form_action,
                    "method": form_method,
                    "inputs": form_inputs
                })
                
            # Extract headings
            headings = []
            for h in soup.find_all(['h1', 'h2', 'h3']):
                heading_text = h.get_text().strip()
                if heading_text:
                    headings.append({
                        "level": int(h.name[1]),
                        "text": heading_text
                    })
                    
            # Extract main content
            main_content = ""
            main_tag = soup.find('main')
            if main_tag:
                main_content = main_tag.get_text(separator=' ', strip=True)
            else:
                # Fallback: try article or main content area
                article = soup.find('article')
                if article:
                    main_content = article.get_text(separator=' ', strip=True)
                else:
                    # Fallback: use body content
                    body = soup.find('body')
                    if body:
                        # Remove navigation, footer, etc.
                        for tag in body.find_all(['nav', 'footer', 'aside', 'header']):
                            tag.decompose()
                        main_content = body.get_text(separator=' ', strip=True)
                        
            # Limit main content length for practicality
            main_content = main_content[:5000] if main_content else ""
            
            return {
                "links": links[:50],  # Limit to 50 links for practicality
                "forms": forms,
                "headings": headings,
                "main_content": main_content
            }
            
        except Exception as e:
            logger.error(f"Error parsing HTML: {str(e)}")
            return {
                "links": [],
                "forms": [],
                "headings": [],
                "main_content": html_content[:1000]  # First 1000 chars as fallback
            }
            
    async def search(self, query: str, search_engine: str = "google") -> Dict[str, Any]:
        """Perform a web search using a search engine"""
        if not self.initialized:
            return {"success": False, "error": "Web interaction agent not initialized"}
            
        self.metrics["last_action_time"] = time.time()
        self.metrics["searches"] += 1
        
        search_urls = {
            "google": "https://www.google.com/search?q=",
            "bing": "https://www.bing.com/search?q=",
            "duckduckgo": "https://duckduckgo.com/?q="
        }
        
        if search_engine not in search_urls:
            search_engine = "google"
            
        search_url = search_urls[search_engine] + query.replace(' ', '+')
        
        try:
            # Navigate to search page
            navigation_result = await self.navigate(search_url)
            
            if not navigation_result["success"]:
                return navigation_result
                
            # Extract search results
            results = await self._extract_search_results(search_engine)
            
            return {
                "success": True,
                "search_engine": search_engine,
                "query": query,
                "results": results,
                "url": self.current_url,
                "screenshot": navigation_result["screenshot"]
            }
            
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"Error performing search: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "search_engine": search_engine,
                "query": query
            }
            
    async def _extract_search_results(self, search_engine: str) -> List[Dict[str, Any]]:
        """Extract search results based on the search engine structure"""
        results = []
        
        try:
            if search_engine == "google":
                # Extract Google search results
                result_elements = await self.page.query_selector_all("div.g")
                
                for idx, elem in enumerate(result_elements):
                    if idx >= 10:  # Limit to first 10 results
                        break
                        
                    # Extract title and link
                    title_elem = await elem.query_selector("h3")
                    link_elem = await elem.query_selector("a")
                    snippet_elem = await elem.query_selector("div.VwiC3b")
                    
                    title = await title_elem.text_content() if title_elem else "No title"
                    href = await link_elem.get_attribute("href") if link_elem else ""
                    snippet = await snippet_elem.text_content() if snippet_elem else ""
                    
                    results.append({
                        "position": idx + 1,
                        "title": title,
                        "url": href,
                        "snippet": snippet
                    })
                    
            elif search_engine == "bing":
                # Extract Bing search results
                result_elements = await self.page.query_selector_all("li.b_algo")
                
                for idx, elem in enumerate(result_elements):
                    if idx >= 10:  # Limit to first 10 results
                        break
                        
                    # Extract title and link
                    title_elem = await elem.query_selector("h2")
                    link_elem = await elem.query_selector("a")
                    snippet_elem = await elem.query_selector("p")
                    
                    title = await title_elem.text_content() if title_elem else "No title"
                    href = await link_elem.get_attribute("href") if link_elem else ""
                    snippet = await snippet_elem.text_content() if snippet_elem else ""
                    
                    results.append({
                        "position": idx + 1,
                        "title": title,
                        "url": href,
                        "snippet": snippet
                    })
                    
            elif search_engine == "duckduckgo":
                # Extract DuckDuckGo search results
                result_elements = await self.page.query_selector_all("div.result")
                
                for idx, elem in enumerate(result_elements):
                    if idx >= 10:  # Limit to first 10 results
                        break
                        
                    # Extract title and link
                    title_elem = await elem.query_selector("h2")
                    link_elem = await elem.query_selector("a.result__a")
                    snippet_elem = await elem.query_selector("div.result__snippet")
                    
                    title = await title_elem.text_content() if title_elem else "No title"
                    href = await link_elem.get_attribute("href") if link_elem else ""
                    snippet = await snippet_elem.text_content() if snippet_elem else ""
                    
                    results.append({
                        "position": idx + 1,
                        "title": title,
                        "url": href,
                        "snippet": snippet
                    })
                    
            # Fallback: generic extraction
            if not results:
                # Try to find any search results with generic selectors
                links = await self.page.query_selector_all("a[href^='http']")
                
                for idx, link in enumerate(links):
                    if idx >= 20:  # Limit to first 20 links
                        break
                        
                    href = await link.get_attribute("href")
                    title = await link.text_content()
                    
                    # Filter out navigation links, ads, etc.
                    if (href and title and len(title.strip()) > 10 and
                        not re.match(r'.*(login|sign in|register|about|contact|help|support).*', title.lower())):
                        results.append({
                            "position": idx + 1,
                            "title": title.strip(),
                            "url": href,
                            "snippet": ""
                        })
                        
            return results
            
        except Exception as e:
            logger.error(f"Error extracting search results: {str(e)}")
            return []
            
    async def click_element(self, selector: str) -> Dict[str, Any]:
        """Click on an element on the current page"""
        if not self.initialized:
            return {"success": False, "error": "Web interaction agent not initialized"}
            
        self.metrics["last_action_time"] = time.time()
        self.metrics["interactions"] += 1
        
        try:
            # Wait for the element to be available
            await self.page.wait_for_selector(selector, timeout=5000)
            
            # Click the element
            await self.page.click(selector)
            
            # Wait for page to load after click
            await self.page.wait_for_load_state("networkidle")
            
            # Take screenshot after click
            screenshot = await self.page.screenshot()
            screenshot_base64 = base64.b64encode(screenshot).decode('utf-8')
            
            # Get updated URL and title
            current_url = self.page.url
            title = await self.page.title()
            
            # Update navigation history if URL changed
            if current_url != self.current_url:
                self.current_url = current_url
                self.navigation_history.append({
                    "url": self.current_url,
                    "timestamp": datetime.now().isoformat(),
                    "title": title
                })
                self.metrics["pages_visited"] += 1
                
            return {
                "success": True,
                "selector": selector,
                "url": current_url,
                "title": title,
                "screenshot": screenshot_base64
            }
            
        except PlaywrightTimeoutError:
            self.metrics["errors"] += 1
            return {
                "success": False,
                "error": f"Element with selector '{selector}' not found",
                "selector": selector
            }
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"Error clicking element with selector '{selector}': {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "selector": selector
            }
            
    async def fill_form(self, selector: str, value: str) -> Dict[str, Any]:
        """Fill a form field on the current page"""
        if not self.initialized:
            return {"success": False, "error": "Web interaction agent not initialized"}
            
        self.metrics["last_action_time"] = time.time()
        self.metrics["interactions"] += 1
        self.metrics["form_fills"] += 1
        
        try:
            # Wait for the form field to be available
            await self.page.wait_for_selector(selector, timeout=5000)
            
            # Clear the field first
            await self.page.fill(selector, "")
            
            # Fill the field
            await self.page.fill(selector, value)
            
            # Take screenshot after filling
            screenshot = await self.page.screenshot()
            screenshot_base64 = base64.b64encode(screenshot).decode('utf-8')
            
            return {
                "success": True,
                "selector": selector,
                "value": value,
                "screenshot": screenshot_base64
            }
            
        except PlaywrightTimeoutError:
            self.metrics["errors"] += 1
            return {
                "success": False,
                "error": f"Form field with selector '{selector}' not found",
                "selector": selector
            }
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"Error filling form field with selector '{selector}': {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "selector": selector,
                "value": value
            }
            
    async def submit_form(self, form_selector: str) -> Dict[str, Any]:
        """Submit a form on the current page"""
        if not self.initialized:
            return {"success": False, "error": "Web interaction agent not initialized"}
            
        self.metrics["last_action_time"] = time.time()
        self.metrics["interactions"] += 1
        
        try:
            # Try to find and click the submit button within the form
            form = await self.page.query_selector(form_selector)
            
            if not form:
                return {
                    "success": False,
                    "error": f"Form with selector '{form_selector}' not found",
                    "selector": form_selector
                }
                
            # Look for submit button within form
            submit_button = await form.query_selector("input[type='submit'], button[type='submit'], button")
            
            if submit_button:
                # Click the submit button
                await submit_button.click()
            else:
                # Fallback: try to submit the form using JavaScript
                await self.page.evaluate(f"document.querySelector('{form_selector}').submit()")
                
            # Wait for page to load after form submission
            await self.page.wait_for_load_state("networkidle")
            
            # Take screenshot after submission
            screenshot = await self.page.screenshot()
            screenshot_base64 = base64.b64encode(screenshot).decode('utf-8')
            
            # Get updated URL and title
            current_url = self.page.url
            title = await self.page.title()
            
            # Update navigation history if URL changed
            if current_url != self.current_url:
                self.current_url = current_url
                self.navigation_history.append({
                    "url": self.current_url,
                    "timestamp": datetime.now().isoformat(),
                    "title": title
                })
                self.metrics["pages_visited"] += 1
                
            return {
                "success": True,
                "form_selector": form_selector,
                "url": current_url,
                "title": title,
                "screenshot": screenshot_base64
            }
            
        except PlaywrightTimeoutError:
            self.metrics["errors"] += 1
            return {
                "success": False,
                "error": "Form submission timed out",
                "selector": form_selector
            }
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"Error submitting form with selector '{form_selector}': {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "selector": form_selector
            }
            
    async def extract_structured_data(self) -> Dict[str, Any]:
        """Extract structured data from the current page (tables, lists, metadata)"""
        if not self.initialized:
            return {"success": False, "error": "Web interaction agent not initialized"}
            
        try:
            # Get HTML content
            html_content = await self.page.content()
            
            if not BS4_AVAILABLE:
                return {
                    "success": False,
                    "error": "BeautifulSoup not available for structured data extraction"
                }
                
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract tables
            tables = []
            for idx, table in enumerate(soup.find_all('table')):
                table_data = []
                
                # Extract headers
                headers = []
                for th in table.find_all('th'):
                    headers.append(th.get_text().strip())
                    
                # Extract rows
                for tr in table.find_all('tr'):
                    row = []
                    for td in tr.find_all('td'):
                        row.append(td.get_text().strip())
                    if row:  # Skip empty rows
                        table_data.append(row)
                        
                tables.append({
                    "id": f"table_{idx}",
                    "headers": headers,
                    "rows": table_data
                })
                
            # Extract lists
            lists = []
            for idx, list_element in enumerate(soup.find_all(['ul', 'ol'])):
                list_type = list_element.name  # 'ul' or 'ol'
                items = []
                
                for li in list_element.find_all('li'):
                    items.append(li.get_text().strip())
                    
                lists.append({
                    "id": f"list_{idx}",
                    "type": list_type,
                    "items": items
                })
                
            # Extract metadata
            metadata = {}
            
            # Meta tags
            for meta in soup.find_all('meta'):
                name = meta.get('name', meta.get('property', ''))
                content = meta.get('content', '')
                if name and content:
                    metadata[name] = content
                    
            # Schema.org structured data
            structured_data = []
            for script in soup.find_all('script', {"type": "application/ld+json"}):
                try:
                    data = json.loads(script.string)
                    structured_data.append(data)
                except (json.JSONDecodeError, TypeError):
                    pass
                    
            return {
                "success": True,
                "tables": tables,
                "lists": lists,
                "metadata": metadata,
                "structured_data": structured_data
            }
            
        except Exception as e:
            logger.error(f"Error extracting structured data: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
            
    async def take_screenshot(self) -> Dict[str, Any]:
        """Take a screenshot of the current page"""
        if not self.initialized:
            return {"success": False, "error": "Web interaction agent not initialized"}
            
        try:
            # Take screenshot
            screenshot = await self.page.screenshot(full_page=True)
            screenshot_base64 = base64.b64encode(screenshot).decode('utf-8')
            
            return {
                "success": True,
                "screenshot": screenshot_base64,
                "url": self.current_url,
                "title": await self.page.title()
            }
            
        except Exception as e:
            logger.error(f"Error taking screenshot: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get web interaction metrics"""
        current_time = time.time()
        elapsed_time = current_time - self.metrics["start_time"] if self.metrics["start_time"] else 0
        
        return {
            "pages_visited": self.metrics["pages_visited"],
            "interactions": self.metrics["interactions"],
            "searches": self.metrics["searches"],
            "form_fills": self.metrics["form_fills"],
            "errors": self.metrics["errors"],
            "elapsed_time": elapsed_time,
            "time_since_last_action": current_time - self.metrics["last_action_time"] if self.metrics["last_action_time"] else 0,
            "initialized": self.initialized,
            "current_url": self.current_url,
            "navigation_history_count": len(self.navigation_history)
        }
        
    async def execute_javascript(self, script: str) -> Dict[str, Any]:
        """Execute JavaScript on the current page"""
        if not self.initialized:
            return {"success": False, "error": "Web interaction agent not initialized"}
            
        try:
            result = await self.page.evaluate(script)
            
            return {
                "success": True,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error executing JavaScript: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "script": script
            }
            
    async def wait_for_element(self, selector: str, timeout: int = 10000) -> Dict[str, Any]:
        """Wait for an element to appear on the page"""
        if not self.initialized:
            return {"success": False, "error": "Web interaction agent not initialized"}
            
        try:
            element = await self.page.wait_for_selector(selector, timeout=timeout)
            
            if element:
                return {
                    "success": True,
                    "selector": selector,
                    "found": True
                }
            else:
                return {
                    "success": True,
                    "selector": selector,
                    "found": False
                }
                
        except PlaywrightTimeoutError:
            return {
                "success": False,
                "error": f"Timeout waiting for element '{selector}'",
                "selector": selector,
                "found": False
            }
        except Exception as e:
            logger.error(f"Error waiting for element '{selector}': {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "selector": selector,
                "found": False
            }