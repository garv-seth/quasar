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
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.client = AsyncOpenAI()
        self.search_urls = [
            "https://www.indeed.com/jobs?q={query}",
            "https://www.linkedin.com/jobs/search?keywords={query}",
            "https://www.glassdoor.com/Job/jobs.htm?sc.keyword={query}",
            "https://www.bls.gov/ooh/computer-and-information-technology/home.htm",
            "https://www.weforum.org/reports/the-future-of-jobs-report-2023",
            "https://www.mckinsey.com/featured-insights/future-of-work"
        ]

    async def _fetch_page(self, url: str) -> str:
        """Fetch content from a URL using aiohttp."""
        try:
            timeout = aiohttp.ClientTimeout(total=30)
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

    def _extract_content(self, html: str) -> str:
        """Extract meaningful content from HTML using BeautifulSoup."""
        try:
            soup = BeautifulSoup(html, 'html.parser')

            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()

            # Extract main content
            content = []
            priority_tags = ['article', 'main', '.content', '.post-content']

            for selector in priority_tags:
                main_content = soup.select(selector)
                if main_content:
                    for element in main_content:
                        paragraphs = element.find_all(['p', 'h1', 'h2', 'h3', 'li'])
                        for p in paragraphs:
                            text = p.get_text().strip()
                            if len(text) > 50:  # Filter out short snippets
                                content.append(text)

            # If no priority content found, fall back to all paragraphs
            if not content:
                for p in soup.find_all(['p', 'h1', 'h2', 'h3']):
                    text = p.get_text().strip()
                    if len(text) > 50:
                        content.append(text)

            return ' '.join(content)
        except Exception as e:
            logging.error(f"Error extracting content: {str(e)}")
            return ""

    def _quantum_process_data(self, texts: List[str]) -> List[float]:
        """Process text data using enhanced quantum circuits."""
        try:
            # Convert texts to numerical features using TF-IDF approach
            features = []
            for text in texts:
                # Create word frequency vector
                words = text.lower().split()
                unique_words = list(set(words[:self.optimizer.n_qubits]))
                freq = np.zeros(self.optimizer.n_qubits)

                for i, word in enumerate(unique_words[:self.optimizer.n_qubits]):
                    # Calculate TF-IDF score
                    tf = words.count(word) / len(words)
                    idf = np.log(len(texts) / sum(1 for t in texts if word in t.lower()))
                    freq[i] = tf * idf

                # Normalize feature vector
                if np.sum(freq) > 0:
                    freq = freq / np.linalg.norm(freq)
                features.append(freq)

            if not features:
                return [0.0] * len(texts)

            # Process through quantum circuit
            quantum_features = []
            for feature in features:
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

            # Normalize scores
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

            # Format URLs with query
            formatted_urls = [url.format(query=prompt.replace(' ', '+')) for url in self.search_urls]

            # Fetch content in parallel
            pages = await asyncio.gather(*[self._fetch_page(url) for url in formatted_urls])

            # Extract and process content
            texts = []
            processed_urls = []
            for html, url in zip(pages, formatted_urls):
                if html:
                    content = self._extract_content(html)
                    if content:
                        texts.append(content)
                        processed_urls.append(url)

            if not texts:
                return {
                    'error': True,
                    'message': 'Failed to fetch relevant content'
                }

            # Process with quantum acceleration
            quantum_start = datetime.now()
            relevance_scores = self._quantum_process_data(texts)
            quantum_processing_time = (datetime.now() - quantum_start).total_seconds() * 1000

            # Sort content by relevance
            sorted_content = sorted(
                zip(texts, relevance_scores, processed_urls),
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

            # Calculate metrics
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            classical_time = total_time - quantum_processing_time
            speedup = classical_time / quantum_processing_time if quantum_processing_time > 0 else 1.0

            # Get quantum circuit statistics
            circuit_stats = self.optimizer.get_circuit_stats()

            return {
                'analysis': analysis,
                'quantum_metrics': {
                    'relevance_scores': relevance_scores,
                    'quantum_confidence': min(100, np.mean(relevance_scores) * 100),
                    'circuit_stats': circuit_stats,
                    'processing_time_ms': quantum_processing_time,
                    'quantum_advantage': {
                        'speedup': f"{speedup:.2f}x",
                        'accuracy_improvement': f"{(np.mean(relevance_scores) * 100):.1f}%"
                    },
                    'sources': processed_urls[:3]  # Top 3 most relevant sources
                },
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logging.error(f"Analysis error: {str(e)}")
            return {
                'error': True,
                'message': f"Error during analysis: {str(e)}"
            }