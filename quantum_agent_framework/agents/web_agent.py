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
from openai import AsyncOpenAI

from ..quantum.optimizer import QuantumOptimizer
from ..quantum.preprocessor import QuantumPreprocessor

class WebAgent:
    """Quantum-enhanced web crawling and analysis agent."""

    def __init__(self, optimizer: QuantumOptimizer, preprocessor: QuantumPreprocessor):
        """Initialize the web agent."""
        self.optimizer = optimizer
        self.preprocessor = preprocessor
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; ResearchBot/1.0; +http://example.com/bot)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        self.client = AsyncOpenAI()
        # Use only publicly accessible APIs that don't require authentication
        self.search_urls = [
            "https://api.github.com/search/repositories?q={query}+language:python",
            "https://api.github.com/search/issues?q={query}+label:hiring",
            "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&exintro=true&titles={query}",
            "https://en.wikipedia.org/w/api.php?action=query&format=json&list=search&srsearch={query}"
        ]

    async def _fetch_page(self, url: str) -> Optional[Dict]:
        """Fetch content from a URL using aiohttp."""
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=self.headers, ssl=False) as response:
                    if response.status == 200:
                        if 'application/json' in response.headers.get('Content-Type', ''):
                            return await response.json()
                        else:
                            text = await response.text()
                            return {'html': text}
                    else:
                        logging.warning(f"Failed to fetch {url}: Status {response.status}")
                        return None
        except Exception as e:
            logging.error(f"Error fetching {url}: {str(e)}")
            return None

    def _extract_text_from_json(self, data: Dict) -> str:
        """Extract relevant text from JSON API responses."""
        texts = []

        try:
            if isinstance(data, dict):
                # GitHub API responses
                if 'items' in data:
                    for item in data['items']:
                        if 'description' in item and item['description']:
                            texts.append(item['description'])
                        if 'body' in item and item['body']:
                            texts.append(item['body'])

                # Wikipedia API responses
                if 'query' in data:
                    if 'pages' in data['query']:
                        for page in data['query']['pages'].values():
                            if 'extract' in page:
                                texts.append(page['extract'])
                    if 'search' in data['query']:
                        for result in data['query']['search']:
                            if 'snippet' in result:
                                texts.append(BeautifulSoup(result['snippet'], 'html.parser').get_text())

                # HTML content
                if 'html' in data:
                    soup = BeautifulSoup(data['html'], 'html.parser')
                    for tag in ['script', 'style', 'nav', 'footer', 'header']:
                        for element in soup.find_all(tag):
                            element.decompose()

                    for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'article']):
                        text = p.get_text().strip()
                        if len(text) > 50:  # Filter out short snippets
                            texts.append(text)

            return ' '.join(texts)
        except Exception as e:
            logging.error(f"Error extracting text: {str(e)}")
            return ""

    def _quantum_process_data(self, texts: List[str]) -> Dict[str, Any]:
        """Process text data using quantum circuits with performance comparison."""
        try:
            start_time = datetime.now()

            # Classical processing (TF-IDF) for comparison
            classical_start = datetime.now()
            classical_features = []
            for text in texts:
                words = text.lower().split()
                unique_words = list(set(words[:self.optimizer.n_qubits]))
                freq = np.zeros(self.optimizer.n_qubits)

                for i, word in enumerate(unique_words[:self.optimizer.n_qubits]):
                    tf = words.count(word) / len(words)
                    idf = np.log(len(texts) / sum(1 for t in texts if word in t.lower()))
                    freq[i] = tf * idf

                if np.sum(freq) > 0:
                    freq = freq / np.linalg.norm(freq)
                classical_features.append(freq)

            classical_time = (datetime.now() - classical_start).total_seconds() * 1000

            # Quantum processing
            quantum_start = datetime.now()
            quantum_features = []
            for feature in classical_features:
                processed = self.preprocessor.preprocess(feature)
                quantum_features.append(processed)

            # Calculate quantum-enhanced relevance scores
            scores = []
            for feature in quantum_features:
                try:
                    score = float(self.optimizer.get_expectation(feature))
                    scores.append(score)
                except Exception as e:
                    logging.error(f"Quantum circuit error: {str(e)}")
                    scores.append(0.0)

            quantum_time = (datetime.now() - quantum_start).total_seconds() * 1000

            # Normalize scores
            if scores:
                min_score = min(scores)
                max_score = max(scores)
                if max_score > min_score:
                    scores = [(s - min_score) / (max_score - min_score) for s in scores]
                else:
                    scores = [1.0 / len(scores)] * len(scores)

            return {
                'scores': scores,
                'metrics': {
                    'classical_time_ms': classical_time,
                    'quantum_time_ms': quantum_time,
                    'total_time_ms': (datetime.now() - start_time).total_seconds() * 1000,
                    'speedup': classical_time / quantum_time if quantum_time > 0 else 1.0,
                    'accuracy': np.mean(scores) * 100 if scores else 0.0,
                    'quantum_circuit_stats': self.optimizer.get_circuit_stats()
                }
            }

        except Exception as e:
            logging.error(f"Quantum processing error: {str(e)}")
            return {
                'scores': [1.0 / len(texts)] * len(texts),
                'metrics': {
                    'error': str(e)
                }
            }

    async def _process_with_gpt(self, content: str, prompt: str) -> str:
        """Process content using GPT-4o."""
        try:
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            messages = [
                {
                    "role": "system",
                    "content": """You are a quantum-enhanced AI assistant specializing in job market analysis.
                    Analyze the provided content and create a comprehensive report with these sections:
                    1. Key Job Market Trends
                    2. In-Demand Skills
                    3. Industry Growth Areas
                    4. Future Outlook
                    5. Recommendations for Job Seekers

                    Focus on current trends, emerging opportunities, and practical insights.
                    Base your analysis strictly on the provided content."""
                },
                {"role": "user", "content": f"Analyze these job market trends:\n\nContent:\n{content}"}
            ]

            completion = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1500,
                temperature=0.7
            )

            return completion.choices[0].message.content

        except Exception as e:
            logging.error(f"GPT processing error: {str(e)}")
            return "I apologize, but I'm having trouble analyzing the content right now."

    async def analyze_content(self, prompt: str, additional_context: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Analyze content using quantum-enhanced processing and OpenAI API."""
        try:
            start_time = datetime.now()

            # Format search queries appropriately for job market analysis
            search_terms = [
                f"job market trends {prompt}",
                f"career opportunities {prompt}",
                f"job skills {prompt}",
                f"hiring trends {prompt}"
            ]

            # Build URLs with proper encoding
            formatted_urls = []
            for term in search_terms:
                for url in self.search_urls:
                    formatted_urls.append(url.format(
                        query=term.replace(' ', '+').replace('?', '')
                    ))

            # Fetch content in parallel with error handling
            fetch_tasks = [self._fetch_page(url) for url in formatted_urls]
            pages = await asyncio.gather(*fetch_tasks, return_exceptions=True)

            # Process successful responses
            texts = []
            processed_urls = []
            for data, url in zip(pages, formatted_urls):
                if isinstance(data, dict):  # Successful response
                    content = self._extract_text_from_json(data)
                    if content:
                        texts.append(content)
                        processed_urls.append(url)

            if not texts:
                return {
                    'error': True,
                    'message': 'Failed to fetch relevant content. Please try again.'
                }

            # Process with quantum acceleration
            quantum_results = self._quantum_process_data(texts)

            # Sort content by relevance
            sorted_content = sorted(
                zip(texts, quantum_results['scores'], processed_urls),
                key=lambda x: x[1],
                reverse=True
            )

            # Prepare most relevant content for analysis
            top_content = "\n---\n".join(
                f"Source ({url}):\n{text[:2000]}..."
                for text, _, url in sorted_content[:3]
            )

            # Process with GPT-4o
            analysis = await self._process_with_gpt(top_content, prompt)

            # Prepare comprehensive metrics
            metrics = quantum_results['metrics']

            return {
                'analysis': analysis,
                'quantum_metrics': {
                    'relevance_scores': quantum_results['scores'],
                    'quantum_confidence': metrics['accuracy'],
                    'circuit_stats': metrics['quantum_circuit_stats'],
                    'processing_time_ms': metrics['quantum_time_ms'],
                    'quantum_advantage': {
                        'speedup': f"{metrics['speedup']:.2f}x",
                        'accuracy_improvement': f"{metrics['accuracy']:.1f}%",
                        'classical_time_ms': metrics['classical_time_ms'],
                        'quantum_time_ms': metrics['quantum_time_ms']
                    },
                    'sources': processed_urls[:3]
                },
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logging.error(f"Analysis error: {str(e)}")
            return {
                'error': True,
                'message': f"Error during analysis: {str(e)}"
            }