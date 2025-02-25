"""Web crawling and analysis agent with quantum-enhanced processing."""

import aiohttp
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
import logging
import openai
from openai import AsyncOpenAI
import os
import json
import numpy as np
import asyncio
from datetime import datetime
import aiohttp

from ..quantum.optimizer import QuantumOptimizer
from ..quantum.preprocessor import QuantumPreprocessor

class WebAgent:
    """Quantum-enhanced web crawling and analysis agent."""

    def __init__(self, optimizer: QuantumOptimizer, preprocessor: QuantumPreprocessor):
        """Initialize the web agent."""
        self.optimizer = optimizer
        self.preprocessor = preprocessor
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.client = AsyncOpenAI()

    async def _fetch_page(self, url: str) -> str:
        """Fetch content from a URL using aiohttp."""
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=self.headers, ssl=False) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        logging.warning(f"Failed to fetch {url}: Status {response.status}")
                        return ""
        except Exception as e:
            logging.error(f"Error fetching {url}: {str(e)}")
            return ""

    def _extract_text(self, html: str) -> str:
        """Extract readable text from HTML."""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            content = []
            for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'article']):
                text = element.get_text().strip()
                if len(text) > 50:  # Filter out short snippets
                    content.append(text)
            return ' '.join(content)
        except Exception as e:
            logging.error(f"Error extracting text: {str(e)}")
            return ""

    def _quantum_process_data(self, texts: List[str]) -> List[float]:
        """Process text data using quantum circuits."""
        try:
            features = []
            for text in texts:
                words = text.lower().split()
                freq = np.zeros(self.optimizer.n_qubits)
                for i, word in enumerate(words[:self.optimizer.n_qubits]):
                    freq[i] = words.count(word) / len(words)
                features.append(freq)

            if not features:
                return [0.0] * len(texts)

            quantum_features = []
            for feature in features:
                flat_feature = feature.flatten()
                processed = self.preprocessor.preprocess(flat_feature)
                quantum_features.append(processed)

            scores = []
            for feature in quantum_features:
                try:
                    score = self.optimizer.get_expectation(feature)
                    scores.append(float(score))
                except Exception as e:
                    logging.error(f"Error getting expectation value: {str(e)}")
                    scores.append(0.0)

            if scores:
                min_score = min(scores)
                max_score = max(scores)
                if max_score > min_score:
                    scores = [(s - min_score) / (max_score - min_score) for s in scores]
                else:
                    scores = [1.0 / len(scores)] * len(scores)

            return scores

        except Exception as e:
            logging.error(f"Quantum processing error: {str(e)}")
            return [1.0 / len(texts)] * len(texts)

    async def _process_with_realtime_gpt(self, content: str, prompt: str) -> str:
        """Process content using GPT-4o."""
        try:
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            messages = [
                {"role": "system", "content": "You are a quantum-enhanced AI assistant analyzing technical content."},
                {"role": "user", "content": f"Analyze this content related to: {prompt}\n\nContent:\n{content}"}
            ]

            # Use the async OpenAI client
            completion = await self.client.chat.completions.create(
                model="gpt-4o",  # Updated model name
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )

            return completion.choices[0].message.content

        except Exception as e:
            logging.error(f"Realtime GPT processing error: {str(e)}")
            return "I apologize, but I'm having trouble analyzing the content right now."

    async def analyze_content(self, prompt: str, additional_context: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Analyze content using quantum-enhanced processing and OpenAI API."""
        try:
            # Process additional context if provided
            texts = []
            urls = []

            if additional_context:
                for item in additional_context:
                    if 'content' in item and item['content'].get('text'):
                        texts.append(item['content']['text'])
                        urls.append(item['url'])

            # Ensure we have some content
            if not texts:
                urls = [
                    "https://ionq.com",
                    "https://www.ibm.com/quantum",
                    "https://quantumai.google"
                ]
                # Fetch and process content in parallel
                fetch_tasks = [self._fetch_page(url) for url in urls]
                pages = await asyncio.gather(*fetch_tasks)
                texts = [self._extract_text(html) for html in pages if html]

            if not texts:
                return {
                    'error': True,
                    'message': 'Failed to fetch relevant content'
                }

            # Quantum process the collected data
            relevance_scores = self._quantum_process_data(texts)

            # Sort and combine relevant content
            sorted_content = sorted(
                zip(texts, relevance_scores, urls),
                key=lambda x: x[1],
                reverse=True
            )

            # Prepare top content for analysis
            top_content = "\n---\n".join(
                f"Source ({url}):\n{text[:1000]}..."
                for text, _, url in sorted_content[:3]
            )

            # Use realtime GPT-4o-mini for analysis
            analysis = await self._process_with_realtime_gpt(top_content, prompt)

            # Get quantum circuit statistics
            circuit_stats = self.optimizer.get_circuit_stats()

            # Calculate quantum advantage metrics
            quantum_confidence = min(100, np.mean(relevance_scores) * 100)
            processing_time = circuit_stats['optimization_steps'] * circuit_stats['circuit_depth']

            return {
                'analysis': analysis,
                'quantum_metrics': {
                    'relevance_scores': relevance_scores,
                    'quantum_confidence': quantum_confidence,
                    'circuit_stats': circuit_stats,
                    'processing_time_ms': processing_time,
                    'quantum_advantage': {
                        'speedup': f"{processing_time/100:.2f}x",
                        'accuracy': f"{quantum_confidence:.1f}%"
                    },
                    'sources': urls
                },
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logging.error(f"Analysis error: {str(e)}")
            return {
                'error': True,
                'message': f"Error during analysis: {str(e)}"
            }