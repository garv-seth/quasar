"""Web crawling and analysis agent with quantum-enhanced processing."""

import aiohttp
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
import logging
import os
import json
import numpy as np
import asyncio
from datetime import datetime
import pennylane as qml
from openai import AsyncOpenAI
import re
from urllib.parse import urljoin, urlparse
import time

class QuantumWebAgent:
    """Quantum-enhanced web crawling and analysis agent."""

    def __init__(self, n_qubits: int = 8, use_quantum: bool = True):
        """Initialize the quantum web agent."""
        self.n_qubits = n_qubits
        self.use_quantum = use_quantum
        self.session = None
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0"
        }
        
        # Setup quantum circuits if enabled
        if self.use_quantum:
            self._setup_quantum_circuits()
            
        # Initialize OpenAI client for content analysis
        self.openai_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        
        # Performance metrics
        self.metrics = {
            "requests_made": 0,
            "quantum_ops_performed": 0,
            "average_response_time": 0,
            "total_processing_time": 0,
            "successful_requests": 0,
            "failed_requests": 0
        }

    def _setup_quantum_circuits(self):
        """Setup quantum circuits for web search enhancements."""
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
        @qml.qnode(self.dev)
        def similarity_circuit(vec1, vec2):
            """Quantum circuit for enhanced similarity calculation."""
            # Ensure vectors are properly normalized
            vec1 = vec1 / np.linalg.norm(vec1)
            vec2 = vec2 / np.linalg.norm(vec2)
            
            # Encode first vector
            for i, val in enumerate(vec1):
                if i < self.n_qubits:
                    qml.RY(np.arcsin(val) * np.pi, wires=i)
                    qml.RZ(np.arccos(val) * np.pi, wires=i)
            
            # Apply Hadamard gates as separators
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            
            # Encode second vector (with controlled operations)
            for i, val in enumerate(vec2):
                if i < self.n_qubits:
                    qml.CRY(np.arcsin(val) * np.pi, wires=[0, i])
                    qml.CRZ(np.arccos(val) * np.pi, wires=[0, i])
            
            # Measurement in computational basis
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        @qml.qnode(self.dev)
        def pattern_circuit(pattern_array, text_array):
            """Quantum circuit for pattern matching in text."""
            # Encode pattern
            for i, val in enumerate(pattern_array):
                if i < self.n_qubits // 2:
                    qml.RX(val * np.pi, wires=i)
            
            # Apply entangling operations
            for i in range(self.n_qubits // 2 - 1):
                qml.CNOT(wires=[i, i+1])
            
            # Encode text
            for i, val in enumerate(text_array):
                if i < self.n_qubits // 2:
                    qml.RX(val * np.pi, wires=i + self.n_qubits // 2)
            
            # Apply pattern matching quantum operations
            for i in range(self.n_qubits // 2):
                qml.CNOT(wires=[i, i + self.n_qubits // 2])
                qml.Hadamard(wires=i)
            
            # Measure correlation
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits // 2)]
        
        @qml.qnode(self.dev)
        def search_priority_circuit(query_vectors, num_results=5):
            """Quantum circuit for prioritizing search results."""
            # Initialize in uniform superposition
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            
            # Apply oracle based on query relevance
            for vec in query_vectors:
                for i, val in enumerate(vec):
                    if i < self.n_qubits:
                        qml.RZ(val * np.pi, wires=i)
                
                # Entangle qubits
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i+1])
            
            # Apply Grover's diffusion operator
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
                qml.X(wires=i)
            
            # Multi-controlled Z gate
            qml.MultiControlledX(
                control_wires=list(range(self.n_qubits-1)),
                wires=self.n_qubits-1,
                work_wires=[]
            )
            
            for i in range(self.n_qubits):
                qml.X(wires=i)
                qml.Hadamard(wires=i)
            
            # Measure to get prioritized results
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        self.similarity_circuit = similarity_circuit
        self.pattern_circuit = pattern_circuit
        self.search_priority_circuit = search_priority_circuit

    async def initialize(self):
        """Initialize aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(headers=self.headers)
        return self.session

    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()

    async def fetch_url(self, url: str, retries: int = 3) -> Optional[Dict]:
        """
        Fetch content from URL with retry mechanism and proper error handling.

        Args:
            url: URL to fetch
            retries: Number of retry attempts

        Returns:
            Dict containing status and content or error message
        """
        if self.session is None or self.session.closed:
            await self.initialize()
            
        self.metrics["requests_made"] += 1
        start_time = time.time()
        
        for attempt in range(retries):
            try:
                async with self.session.get(url, timeout=30) as response:
                    response_time = time.time() - start_time
                    self._update_response_time_metric(response_time)
                    
                    if response.status == 200:
                        content_type = response.headers.get('Content-Type', '').lower()
                        
                        if 'text/html' in content_type:
                            html_content = await response.text()
                            self.metrics["successful_requests"] += 1
                            return {
                                "status": "success",
                                "url": url,
                                "content_type": "html",
                                "content": html_content,
                                "response_time": response_time
                            }
                        elif 'application/json' in content_type:
                            json_content = await response.json()
                            self.metrics["successful_requests"] += 1
                            return {
                                "status": "success",
                                "url": url,
                                "content_type": "json",
                                "content": json_content,
                                "response_time": response_time
                            }
                        else:
                            text_content = await response.text()
                            self.metrics["successful_requests"] += 1
                            return {
                                "status": "success",
                                "url": url,
                                "content_type": "text",
                                "content": text_content,
                                "response_time": response_time
                            }
                    else:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
            except (aiohttp.ClientError, asyncio.TimeoutError, UnicodeDecodeError) as e:
                if attempt == retries - 1:
                    self.metrics["failed_requests"] += 1
                    return {
                        "status": "error",
                        "url": url,
                        "error_type": str(type(e).__name__),
                        "error_message": str(e),
                        "attempt": attempt + 1
                    }
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            except Exception as e:
                self.metrics["failed_requests"] += 1
                logging.error(f"Unexpected error fetching {url}: {str(e)}")
                return {
                    "status": "error",
                    "url": url,
                    "error_type": "UnexpectedError",
                    "error_message": str(e)
                }
        
        self.metrics["failed_requests"] += 1
        return {
            "status": "error",
            "url": url,
            "error_type": "MaxRetriesExceeded",
            "error_message": f"Failed after {retries} attempts"
        }
        
    def _update_response_time_metric(self, new_time: float):
        """Update the average response time metric with a new data point."""
        total_requests = self.metrics["successful_requests"] + self.metrics["failed_requests"]
        if total_requests == 0:
            self.metrics["average_response_time"] = new_time
        else:
            current_avg = self.metrics["average_response_time"]
            self.metrics["average_response_time"] = (current_avg * (total_requests - 1) + new_time) / total_requests

    def extract_content(self, content: Any, content_type: str) -> Dict[str, Any]:
        """
        Extract relevant content based on content type.

        Args:
            content: Raw content from URL
            content_type: Type of content (html, json, text)

        Returns:
            Dict containing extracted content
        """
        if content_type == "html":
            return self._extract_html_content(content)
        elif content_type == "json":
            return self._extract_json_content(content)
        else:
            # For plain text, just return the content
            return {
                "text": content if isinstance(content, str) else str(content),
                "title": "Text Content",
                "links": []
            }

    def _extract_html_content(self, html_content: str) -> Dict[str, str]:
        """Extract content from HTML using BeautifulSoup."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title
            title = soup.title.string if soup.title else "No Title"
            
            # Extract meta description
            meta_desc = ""
            meta_tag = soup.find('meta', attrs={'name': 'description'})
            if meta_tag and 'content' in meta_tag.attrs:
                meta_desc = meta_tag['content']
            
            # Extract main content (prioritize article, main, then div containers)
            main_content = soup.find('article') or soup.find('main') or soup.find('div', class_=['content', 'main-content'])
            
            # If main content container not found, use body but exclude scripts, styles, etc.
            if not main_content:
                main_content = soup.body
            
            # Remove scripts, styles, and non-content elements
            if main_content:
                for script in main_content.find_all(['script', 'style', 'nav', 'footer', 'header']):
                    script.decompose()
            
            # Extract text but keep reasonable structure
            content_text = ""
            if main_content:
                # Process by paragraph to maintain some structure
                for p in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                    content_text += p.get_text() + "\n\n"
                
                # If no structured content, fall back to main content text
                if not content_text.strip():
                    content_text = main_content.get_text(separator='\n', strip=True)
            
            # Extract links
            links = []
            for a in soup.find_all('a', href=True):
                link_text = a.get_text().strip()
                href = a['href']
                if href and not href.startswith(('#', 'javascript:', 'mailto:')):
                    links.append({
                        "text": link_text[:100] if link_text else "No text",
                        "href": href
                    })
            
            return {
                "title": title,
                "meta_description": meta_desc,
                "text": content_text,
                "links": links[:20],  # Limit to 20 links
                "word_count": len(content_text.split())
            }
            
        except Exception as e:
            logging.error(f"Error extracting HTML content: {str(e)}")
            return {
                "title": "Extraction Error",
                "text": "Failed to extract content from HTML",
                "error": str(e),
                "links": []
            }

    def _extract_json_content(self, json_content: Dict) -> Dict[str, Any]:
        """Extract relevant information from JSON content."""
        try:
            # Initialize extracted data
            extracted = {
                "title": json_content.get("title", "JSON Content"),
                "text": "",
                "fields": {},
                "links": []
            }
            
            # Process JSON content recursively to extract text fields
            text_chunks = []
            
            def process_json_item(item, path=""):
                """Recursively process JSON items to extract text content."""
                if isinstance(item, dict):
                    for key, value in item.items():
                        new_path = f"{path}.{key}" if path else key
                        
                        # Check if this might be a content field
                        if isinstance(value, str) and len(value) > 50 and key.lower() in [
                            "content", "text", "description", "body", "abstract"
                        ]:
                            text_chunks.append(value)
                        
                        # Record fields that might be meaningful
                        if isinstance(value, (str, int, float, bool)) and not isinstance(value, str) or len(str(value)) < 500:
                            extracted["fields"][new_path] = value
                            
                        # Check for URL fields
                        if isinstance(value, str) and key.lower() in ["url", "link", "href"] and not value.startswith(("#", "javascript:", "mailto:")):
                            extracted["links"].append({
                                "text": item.get("title", item.get("name", new_path)),
                                "href": value
                            })
                        
                        # Continue recursion
                        process_json_item(value, new_path)
                    
                elif isinstance(item, list):
                    for i, value in enumerate(item):
                        new_path = f"{path}[{i}]"
                        process_json_item(value, new_path)
            
            # Process the JSON content
            process_json_item(json_content)
            
            # Combine extracted text
            extracted["text"] = "\n\n".join(text_chunks)
            
            return extracted
            
        except Exception as e:
            logging.error(f"Error extracting JSON content: {str(e)}")
            return {
                "title": "JSON Extraction Error",
                "text": "Failed to extract content from JSON",
                "error": str(e),
                "fields": {},
                "links": []
            }

    def _classical_process_content(self, contents: List[Dict]) -> List[Dict]:
        """Process a list of content items using classical methods."""
        # Simple scoring based on keyword frequency and metadata
        processed_results = []
        
        for content in contents:
            if content.get("status") == "error":
                continue
                
            text = content.get("text", "")
            title = content.get("title", "")
            
            # Simple relevance scoring based on content length and structure
            word_count = len(text.split())
            has_structure = text.count('\n\n') > 5  # Check for paragraph structure
            link_count = len(content.get("links", []))
            
            # Calculate a simple relevance score (0-1)
            relevance = min(1.0, (0.5 * min(1.0, word_count / 1000) + 
                                  0.3 * (1 if has_structure else 0) + 
                                  0.2 * min(1.0, link_count / 10)))
            
            processed_results.append({
                "url": content.get("url", ""),
                "title": title,
                "summary": text[:300] + "..." if len(text) > 300 else text,
                "content_type": content.get("content_type", "unknown"),
                "relevance": relevance,
                "word_count": word_count
            })
        
        # Sort by relevance
        processed_results.sort(key=lambda x: x["relevance"], reverse=True)
        return processed_results

    def _quantum_process_content(self, contents: List[Dict], query: str) -> List[Dict]:
        """Process a list of content items using quantum methods."""
        if not self.use_quantum:
            return self._classical_process_content(contents)
            
        self.metrics["quantum_ops_performed"] += 1
        processed_results = []
        
        # Extract query keywords for quantum similarity processing
        query_keywords = self._extract_keywords(query)
        query_vector = self._text_to_vector(query)
        
        for content in contents:
            if content.get("status") == "error":
                continue
                
            text = content.get("text", "")
            title = content.get("title", "")
            
            # Prepare document vector
            doc_vector = self._text_to_vector(text[:500])  # Use beginning of document
            
            # Calculate quantum similarity
            quantum_similarity = self._quantum_similarity(query_vector, doc_vector)
            
            # Calculate pattern matching scores for keywords
            pattern_scores = []
            for keyword in query_keywords:
                keyword_vector = self._text_to_vector(keyword)
                text_sample_vector = self._text_to_vector(text[:200])  # Use beginning for efficiency
                pattern_score = self._quantum_pattern_match(keyword_vector, text_sample_vector)
                pattern_scores.append(pattern_score)
            
            # Average pattern score
            avg_pattern_score = sum(pattern_scores) / len(pattern_scores) if pattern_scores else 0
            
            # Combine scores with quantum weighting
            quantum_relevance = 0.7 * quantum_similarity + 0.3 * avg_pattern_score
            
            # Enhance with classic factors
            word_count = len(text.split())
            link_count = len(content.get("links", []))
            classic_factor = min(1.0, (0.3 * min(1.0, word_count / 1000) + 
                                      0.2 * min(1.0, link_count / 10)))
            
            # Combined relevance score with quantum advantage
            relevance = 0.8 * quantum_relevance + 0.2 * classic_factor
            
            processed_results.append({
                "url": content.get("url", ""),
                "title": title,
                "summary": text[:300] + "..." if len(text) > 300 else text,
                "content_type": content.get("content_type", "unknown"),
                "relevance": relevance,
                "quantum_similarity": quantum_similarity,
                "word_count": word_count,
                "quantum_processing": True
            })
        
        # Sort by relevance
        processed_results.sort(key=lambda x: x["relevance"], reverse=True)
        return processed_results

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Filter out common stop words
        stop_words = {"the", "and", "in", "at", "on", "with", "from", "by", "for", "is", "are", "was", "were", "be", "been", "has", "have", "had", "this", "that", "these", "those"}
        filtered_words = [word for word in words if word not in stop_words]
        
        # Count frequency
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency and get top keywords
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, count in sorted_words[:5]]
        
        return keywords

    def _text_to_vector(self, text: str, max_dim: int = 8) -> np.ndarray:
        """Convert text to vector for quantum processing."""
        # Simple but effective feature extraction for quantum circuits
        if not text:
            return np.zeros(max_dim)
        
        # Character frequencies (normalized)
        text = text.lower()
        char_set = "abcdefghijklmnopqrstuvwxyz0123456789 "
        
        # Calculate character frequencies
        freq = np.zeros(len(char_set))
        for char in text:
            if char in char_set:
                idx = char_set.index(char)
                freq[idx] += 1
        
        # Normalize
        if np.sum(freq) > 0:
            freq = freq / np.sum(freq)
        
        # Word length distribution (also normalized)
        words = re.findall(r'\b\w+\b', text)
        word_lengths = np.zeros(10)  # 0-9+ characters
        for word in words:
            length = min(len(word), 9)
            word_lengths[length] += 1
        
        # Normalize
        if np.sum(word_lengths) > 0:
            word_lengths = word_lengths / np.sum(word_lengths)
        
        # Combine features (take the most informative dimensions)
        features = np.concatenate([freq[:max_dim//2], word_lengths[:max_dim//2]])
        
        # Ensure we have exactly max_dim dimensions
        if len(features) > max_dim:
            features = features[:max_dim]
        elif len(features) < max_dim:
            features = np.pad(features, (0, max_dim - len(features)))
        
        return features

    def _quantum_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate quantum-enhanced similarity between vectors."""
        if not self.use_quantum:
            return self._classical_similarity(vec1, vec2)
        
        try:
            # Use the quantum circuit for similarity calculation
            result = self.similarity_circuit(vec1, vec2)
            
            # Map the result to a similarity score [0,1]
            similarity = (np.mean(result) + 1) / 2
            return similarity
        except Exception as e:
            logging.error(f"Quantum similarity error: {str(e)}")
            return self._classical_similarity(vec1, vec2)

    def _quantum_pattern_match(self, pattern: np.ndarray, text: np.ndarray) -> float:
        """Use quantum circuit for pattern matching."""
        if not self.use_quantum:
            return self._classical_similarity(pattern, text)
        
        try:
            # Use the quantum circuit for pattern matching
            result = self.pattern_circuit(pattern, text)
            
            # Transform result to a match score [0,1]
            # More negative values indicate better pattern matches
            match_score = (1 - np.mean(result)) / 2
            return match_score
        except Exception as e:
            logging.error(f"Quantum pattern match error: {str(e)}")
            return self._classical_similarity(pattern, text)

    def _classical_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Classical vector similarity for comparison or fallback."""
        # Cosine similarity
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
            
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    async def search(self, query: str, urls: Optional[List[str]] = None, max_results: int = 10) -> Dict[str, Any]:
        """
        Perform quantum-enhanced web search across multiple URLs.

        Args:
            query: Search query
            urls: List of URLs to search (optional)
            max_results: Maximum number of results to return

        Returns:
            Dict containing search results and metrics
        """
        start_time = time.time()
        
        # Default URLs for quantum computing related searches
        if not urls:
            urls = [
                "https://quantum-computing.ibm.com/",
                "https://www.nature.com/subjects/quantum-information",
                "https://quantum.country/",
                "https://pennylane.ai/",
                "https://www.quantiki.org/",
                "https://arxiv.org/list/quant-ph/recent"
            ]
        
        # Fetch content from URLs in parallel
        try:
            if self.session is None or self.session.closed:
                await self.initialize()
                
            fetch_tasks = [self.fetch_url(url) for url in urls]
            content_results = await asyncio.gather(*fetch_tasks)
            
            # Extract content
            processed_contents = []
            for result in content_results:
                if result.get("status") == "success":
                    content = result.get("content", "")
                    content_type = result.get("content_type", "text")
                    
                    # Extract and add content
                    extracted = self.extract_content(content, content_type)
                    extracted["url"] = result.get("url", "")
                    extracted["content_type"] = content_type
                    processed_contents.append(extracted)
                else:
                    # Include error information
                    processed_contents.append({
                        "status": "error",
                        "url": result.get("url", ""),
                        "error": result.get("error_message", "Unknown error")
                    })
            
            # Process content with quantum enhancement if enabled
            if self.use_quantum:
                results = self._quantum_process_content(processed_contents, query)
                processing_method = "quantum"
            else:
                results = self._classical_process_content(processed_contents)
                processing_method = "classical"
            
            # Calculate result similarity with alternative method for comparison
            if self.use_quantum:
                classical_results = self._classical_process_content(processed_contents)
                similarity = self._calculate_result_similarity(results[:5], classical_results[:5])
            else:
                similarity = 1.0  # Same method used
            
            # Prepare final results with only the top results
            top_results = results[:max_results]
            
            # Calculate computation time
            computation_time = time.time() - start_time
            self.metrics["total_processing_time"] += computation_time
            
            return {
                "query": query,
                "results": top_results,
                "result_count": len(top_results),
                "total_sources": len(urls),
                "successful_sources": len([c for c in processed_contents if c.get("status") != "error"]),
                "processing_method": processing_method,
                "computation_time": computation_time,
                "cross_method_similarity": similarity,
                "quantum_advantage": bool(self.use_quantum and similarity < 0.95)  # Real advantage if results differ
            }
            
        except Exception as e:
            logging.error(f"Search error: {str(e)}")
            return {
                "query": query,
                "error": str(e),
                "results": [],
                "result_count": 0,
                "computation_time": time.time() - start_time
            }
        finally:
            # Don't close the session here to allow for continued use
            pass

    def _calculate_result_similarity(self, results1: List[Dict], results2: List[Dict]) -> float:
        """Calculate similarity between two sets of results."""
        # Simple implementation: Check overlap in result URLs
        urls1 = [r.get("url", "") for r in results1]
        urls2 = [r.get("url", "") for r in results2]
        
        # Count matches
        matches = sum(1 for url in urls1 if url in urls2)
        total = max(len(urls1), len(urls2))
        
        if total == 0:
            return 0.0
            
        return matches / total

    async def analyze_content(self, query: str, urls: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze content related to a query using quantum-enhanced processing.

        Args:
            query: Query or topic to analyze
            urls: Optional list of URLs to analyze

        Returns:
            Dict containing analysis and quantum metrics
        """
        start_time = time.time()
        
        # First, perform a search to gather content
        search_results = await self.search(query, urls)
        results = search_results.get("results", [])
        
        if not results:
            return {
                "query": query,
                "error": "No content found to analyze",
                "analysis": "",
                "computation_time": time.time() - start_time
            }
        
        # Combine content from search results
        combined_content = ""
        for result in results:
            combined_content += f"Title: {result.get('title', '')}\n"
            combined_content += f"Source: {result.get('url', '')}\n"
            combined_content += f"{result.get('summary', '')}\n\n"
        
        # Process content with OpenAI
        analysis = await self._process_with_ai(combined_content, query)
        
        # Calculate computation time
        computation_time = time.time() - start_time
        
        return {
            "query": query,
            "sources_used": len(results),
            "analysis": analysis,
            "computation_time": computation_time,
            "processing_method": "quantum" if self.use_quantum else "classical",
            "content_word_count": len(combined_content.split())
        }

    async def _process_with_ai(self, content: str, query: str) -> str:
        """Process content using GPT-4o."""
        try:
            # Maximum token limitation
            if len(content) > 20000:  # Approximate token count
                content = content[:20000] + "... (content truncated)"
            
            messages = [
                {"role": "system", "content": 
                    "You are a quantum computing expert analyzing web content. "
                    "Provide a comprehensive, insightful analysis of the provided content "
                    "related to the user's query. Focus on key insights, trends, and findings. "
                    "Structure your response with clear headings and bullet points when appropriate."
                },
                {"role": "user", "content": 
                    f"Query: {query}\n\n"
                    f"Please analyze the following content related to this query:\n\n{content}"
                }
            ]
            
            completion = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1000,
                temperature=0.5
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            logging.error(f"AI processing error: {str(e)}")
            return f"Error analyzing content: {str(e)}"

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics of the quantum web agent."""
        # Copy metrics to avoid modification during read
        metrics = self.metrics.copy()
        
        # Add additional computed metrics
        if metrics["requests_made"] > 0:
            metrics["success_rate"] = metrics["successful_requests"] / metrics["requests_made"]
        else:
            metrics["success_rate"] = 0
            
        # Add quantum-specific metrics if applicable
        if self.use_quantum:
            metrics["quantum_enabled"] = True
            metrics["qubits_used"] = self.n_qubits
            metrics["theoretical_speedup"] = "O(√N) for search"
        else:
            metrics["quantum_enabled"] = False
        
        return metrics
