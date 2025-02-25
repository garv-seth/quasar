"""Web crawling and analysis agent with quantum-enhanced processing."""

import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
import logging
import openai
import os
import json
import numpy as np
import asyncio
from datetime import datetime

from ..quantum.optimizer import QuantumOptimizer
from ..quantum.preprocessor import QuantumPreprocessor

class WebAgent:
    """Quantum-enhanced web crawling and analysis agent."""

    def __init__(self, optimizer: QuantumOptimizer, preprocessor: QuantumPreprocessor):
        """
        Initialize the web agent.

        Args:
            optimizer: Quantum optimizer for processing
            preprocessor: Quantum preprocessor for data
        """
        self.optimizer = optimizer
        self.preprocessor = preprocessor
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    async def _fetch_page(self, url: str) -> str:
        """Fetch content from a URL."""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logging.error(f"Error fetching {url}: {str(e)}")
            return ""

    def _extract_text(self, html: str) -> str:
        """Extract readable text from HTML."""
        try:
            soup = BeautifulSoup(html, 'html.parser')

            # Remove script and style elements
            for script in soup(['script', 'style']):
                script.decompose()

            # Get text and clean it
            text = ' '.join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'article'])])
            return ' '.join(text.split())  # Clean whitespace
        except Exception as e:
            logging.error(f"Error extracting text: {str(e)}")
            return ""

    def _quantum_process_data(self, texts: List[str]) -> List[float]:
        """
        Process text data using quantum circuits.

        This uses quantum computing to:
        1. Optimize feature extraction from text
        2. Find patterns in the data
        3. Score relevance using quantum measurements
        """
        try:
            # Convert texts to numerical features using TF-IDF-like approach
            features = []
            for text in texts:
                # Simple feature extraction (can be enhanced)
                words = text.lower().split()
                freq = {}
                for word in words:
                    freq[word] = freq.get(word, 0) + 1

                # Get top 100 words by frequency
                top_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:100]
                feature_vector = [count for _, count in top_words]
                features.append(feature_vector)

            # Convert to numpy array and pad if needed
            max_len = max(len(f) for f in features)
            padded_features = []
            for f in features:
                padded = f + [0] * (max_len - len(f))
                padded_features.append(padded)

            features = np.array(padded_features)
            features = features / np.max(features)  # Normalize

            # Quantum preprocessing of features
            processed_features = []
            for feature in features:
                processed = self.preprocessor.preprocess(feature)
                processed_features.append(processed)

            # Optimize using quantum circuits
            processed_features = np.array(processed_features)
            opt_params, _ = self.optimizer.optimize(
                processed_features.flatten(),
                steps=50  # Reduced steps for faster response
            )

            # Calculate relevance scores
            scores = np.mean(opt_params.reshape(len(texts), -1), axis=1)
            return scores.tolist()

        except Exception as e:
            logging.error(f"Quantum processing error: {str(e)}")
            # Fallback to basic scoring
            return [1.0 / len(texts)] * len(texts)

    async def analyze_market_trends(self, sector: str = "tech") -> Dict[str, Any]:
        """
        Analyze market trends using quantum-enhanced web crawling.

        Args:
            sector: Market sector to analyze

        Returns:
            Dict with analysis results and quantum metrics
        """
        # List of financial and tech news sources
        sources = [
            "https://techcrunch.com",
            "https://www.theverge.com",
            "https://www.wired.com",
            "https://www.reuters.com/technology",
            "https://www.cnbc.com/technology"
        ]

        # Collect data
        texts = []
        urls = []
        for url in sources:
            html = await self._fetch_page(url)
            if html:
                text = self._extract_text(html)
                if text:
                    texts.append(text)
                    urls.append(url)

        if not texts:
            return {
                'error': True,
                'message': 'Failed to fetch data from sources'
            }

        # Quantum process the collected data
        relevance_scores = self._quantum_process_data(texts)

        # Sort and combine relevant content
        sorted_content = sorted(zip(texts, relevance_scores, urls), 
                              key=lambda x: x[1],
                              reverse=True)

        # Prepare content for LLM
        top_articles = "\n---\n".join(
            f"Source ({url}):\n{text[:1500]}..."  # First 1500 chars of each
            for text, _, url in sorted_content[:3]  # Top 3 most relevant
        )

        try:
            # Use OpenAI to analyze the content
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": f"""
                    You are a market analysis expert. Analyze the provided news data 
                    for {sector} sector trends. Focus on:
                    1. Key market movements and trends
                    2. Emerging technologies and innovations
                    3. Investment opportunities
                    4. Potential risks and challenges

                    Provide a comprehensive yet concise analysis with actionable insights.
                    """},
                    {"role": "user", "content": top_articles}
                ],
                temperature=0.7
            )

            analysis = response.choices[0].message.content

            # Calculate quantum confidence from relevance scores
            quantum_confidence = min(100, np.mean(relevance_scores) * 100)

            return {
                'analysis': analysis,
                'quantum_metrics': {
                    'relevance_scores': relevance_scores,
                    'processed_sources': len(texts),
                    'quantum_confidence': quantum_confidence,
                    'sources': urls
                },
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logging.error(f"LLM processing error: {str(e)}")
            return {
                'error': True,
                'message': f"Error processing content: {str(e)}"
            }