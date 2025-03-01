"""
Search Integration Module for QA³ (Quantum-Accelerated AI Agent)

This module provides integration with both the quantum-enhanced search and deep search modules,
enabling unified search capabilities across 20+ sources with quantum acceleration.
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
logger = logging.getLogger("search-integration")

class SearchIntegration:
    """
    Integration with search modules for comprehensive search capabilities
    
    This class provides a unified interface to both quantum-enhanced search and deep search,
    with automatic fallback mechanisms and result aggregation.
    """
    
    def __init__(self, use_quantum: bool = True, n_qubits: int = 8):
        """
        Initialize search integration
        
        Args:
            use_quantum: Whether to use quantum acceleration
            n_qubits: Number of qubits for quantum operations
        """
        self.use_quantum = use_quantum
        self.n_qubits = n_qubits
        self.quantum_search = None
        self.deep_search = None
        self.search_history = []
        
        # Initialize search engines
        self._initialize_search_engines()
        
        logger.info(f"Search integration initialized (quantum_enabled={use_quantum}, n_qubits={n_qubits})")
    
    def _initialize_search_engines(self):
        """Initialize search engines"""
        try:
            # Try to import and initialize quantum-enhanced search
            from quantum_agent_framework.search import QuantumEnhancedSearch
            self.quantum_search = QuantumEnhancedSearch(
                n_qubits=self.n_qubits,
                use_quantum=self.use_quantum
            )
            logger.info("Quantum-enhanced search initialized successfully")
        except ImportError as e:
            logger.warning(f"QuantumEnhancedSearch not available: {str(e)}")
            self.quantum_search = None
        
        try:
            # Try to import and initialize deep search
            from quantum_agent_framework.search import DeepSearch
            self.deep_search = DeepSearch(
                n_qubits=self.n_qubits,
                use_quantum=self.use_quantum
            )
            logger.info("Deep search initialized successfully")
        except ImportError as e:
            logger.warning(f"DeepSearch not available: {str(e)}")
            
            # Try to import directly
            try:
                from quantum_agent_framework.search.deep_search import DeepSearch
                self.deep_search = DeepSearch(
                    n_qubits=self.n_qubits,
                    use_quantum=self.use_quantum
                )
                logger.info("Deep search initialized successfully (direct import)")
            except ImportError as e:
                logger.warning(f"Direct DeepSearch import failed: {str(e)}")
                self.deep_search = None
    
    async def search(self, query: str, source_types: Optional[List[str]] = None, 
                   max_results: int = 10, deep_search: bool = True) -> Dict[str, Any]:
        """
        Perform a search with either quantum-enhanced search or deep search
        
        Args:
            query: The search query
            source_types: Optional list of source types to search (for deep search)
            max_results: Maximum number of results to return
            deep_search: Whether to use deep search (if available)
            
        Returns:
            Search results
        """
        start_time = time.time()
        search_id = len(self.search_history) + 1
        
        logger.info(f"Performing search for query: {query} (deep_search={deep_search})")
        
        try:
            # Determine which search engine to use
            use_deep = deep_search and self.deep_search is not None
            
            if use_deep:
                # Use deep search for comprehensive results
                logger.info("Using deep search")
                raw_results = await self.deep_search.execute_deep_search(
                    query=query,
                    source_types=source_types,
                    max_total_results=max_results
                )
            elif self.quantum_search is not None:
                # Fall back to quantum-enhanced search
                logger.info("Using quantum-enhanced search")
                raw_results = await self.quantum_search.deep_search(
                    query=query,
                    max_results=max_results
                )
            else:
                # No search engines available
                logger.error("No search engines available")
                return {
                    "success": False,
                    "error": "No search engines available",
                    "search_id": search_id,
                    "query": query,
                    "execution_time": time.time() - start_time
                }
            
            # Record search in history
            search_record = {
                "id": search_id,
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "deep_search": use_deep,
                "quantum_enhanced": self.use_quantum,
                "execution_time": time.time() - start_time,
                "result_count": len(raw_results.get("search_results", [])) if raw_results.get("success", False) else 0,
                "success": raw_results.get("success", False)
            }
            self.search_history.append(search_record)
            
            # Add search record ID to results
            if raw_results.get("success", False):
                raw_results["search_id"] = search_id
            
            return raw_results
        
        except Exception as e:
            logger.error(f"Search failed: {str(e)}", exc_info=True)
            
            # Record failed search
            failed_search = {
                "id": search_id,
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "deep_search": deep_search,
                "quantum_enhanced": self.use_quantum,
                "execution_time": time.time() - start_time,
                "success": False,
                "error": str(e)
            }
            self.search_history.append(failed_search)
            
            return {
                "success": False,
                "error": str(e),
                "search_id": search_id,
                "query": query,
                "execution_time": time.time() - start_time
            }
    
    async def get_job_search_results(self, query: str, companies: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform a job search
        
        Args:
            query: Job search query
            companies: Optional list of companies to filter by
            
        Returns:
            Job search results
        """
        # Default companies if none provided
        if not companies:
            companies = ["microsoft", "google", "amazon", "apple", "meta", "linkedin"]
        
        # Construct a job-specific query
        job_query = f"job {query}"
        
        # Use deep search with job sources
        source_types = ["jobs", "jobs_company"]
        
        results = await self.search(
            query=job_query,
            source_types=source_types,
            max_results=20,
            deep_search=True
        )
        
        # Filter by companies if needed
        if results.get("success", False) and "search_results" in results:
            # If companies specified, filter results
            if companies:
                filtered_results = []
                
                for result in results["search_results"]:
                    # Check if result is from one of the specified companies
                    result_company = result.get("company", "").lower()
                    title = result.get("title", "").lower()
                    snippet = result.get("snippet", "").lower()
                    
                    # Check company in various fields
                    match = False
                    for company in companies:
                        if (company.lower() in result_company or 
                            company.lower() in title or 
                            company.lower() in snippet):
                            match = True
                            break
                    
                    if match:
                        filtered_results.append(result)
                
                # Update results with filtered set
                results["search_results"] = filtered_results
                results["result_count"] = len(filtered_results)
                results["filtered_by_companies"] = companies
        
        return results
    
    async def get_research_results(self, topic: str, academic: bool = True) -> Dict[str, Any]:
        """
        Perform a research search for academic or technical content
        
        Args:
            topic: Research topic
            academic: Whether to focus on academic sources
            
        Returns:
            Research results
        """
        # Construct a research-specific query
        research_query = f"research {topic}"
        
        # Select source types based on academic flag
        source_types = ["academic"] if academic else ["tech", "academic", "news"]
        
        # Use deep search with appropriate sources
        results = await self.search(
            query=research_query,
            source_types=source_types,
            max_results=15,
            deep_search=True
        )
        
        # Add additional metadata
        if results.get("success", False):
            results["research_type"] = "academic" if academic else "general"
            results["topic"] = topic
        
        return results
    
    async def get_trending_topics(self, category: Optional[str] = None) -> Dict[str, Any]:
        """
        Get trending topics in a specific category
        
        Args:
            category: Optional category to filter by
            
        Returns:
            Trending topics
        """
        # Construct a trending topics query
        trending_query = "trending topics"
        if category:
            trending_query += f" in {category}"
        
        # Select news and social media sources
        source_types = ["news", "social"]
        
        # Use deep search
        results = await self.search(
            query=trending_query,
            source_types=source_types,
            max_results=10,
            deep_search=True
        )
        
        # Extract trending topics from results
        if results.get("success", False) and "search_results" in results:
            # Try to extract topics from the comprehensive summary
            if "comprehensive_summary" in results:
                results["trending_topics"] = self._extract_topics_from_summary(
                    results["comprehensive_summary"])
            else:
                # Extract from individual results
                results["trending_topics"] = self._extract_topics_from_results(
                    results["search_results"])
        
        return results
    
    def _extract_topics_from_summary(self, summary: str) -> List[Dict[str, Any]]:
        """Extract trending topics from search summary"""
        # Simple extraction for demo purposes
        lines = summary.split('\n')
        topics = []
        
        for line in lines:
            # Look for bullet points or numbered lists
            if line.strip().startswith('•') or line.strip().startswith('-') or line.strip().startswith('*'):
                topic_text = line.strip()[1:].strip()
                topics.append({
                    "topic": topic_text,
                    "source": "summary"
                })
            elif len(line.strip()) > 0 and (line[0].isdigit() and line[1:3] in ['. ', ') ']):
                topic_text = line.strip()[3:].strip()
                topics.append({
                    "topic": topic_text,
                    "source": "summary"
                })
        
        # If no topics found, use the first few lines
        if not topics and len(lines) > 2:
            for i in range(min(5, len(lines))):
                if len(lines[i].strip()) > 10:  # Minimum length check
                    topics.append({
                        "topic": lines[i].strip(),
                        "source": "summary"
                    })
        
        return topics
    
    def _extract_topics_from_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract trending topics from search results"""
        topics = []
        
        for result in results:
            title = result.get("title", "")
            source = result.get("source", "unknown")
            
            # Only use titles of reasonable length
            if 15 <= len(title) <= 100:
                topics.append({
                    "topic": title,
                    "source": source
                })
        
        return topics[:10]  # Limit to top 10
    
    def get_search_history(self) -> List[Dict[str, Any]]:
        """
        Get search history
        
        Returns:
            List of search records
        """
        return self.search_history
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status information
        
        Returns:
            Dict with status information
        """
        return {
            "quantum_enabled": self.use_quantum,
            "n_qubits": self.n_qubits,
            "quantum_search_available": self.quantum_search is not None,
            "deep_search_available": self.deep_search is not None,
            "searches_performed": len(self.search_history),
            "search_engines": [
                "quantum_search" if self.quantum_search is not None else None,
                "deep_search" if self.deep_search is not None else None
            ]
        }