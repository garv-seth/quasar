"""Web crawler implementation for Q3A framework."""

import aiohttp
import logging
from typing import List, Dict, Optional
import json
from bs4 import BeautifulSoup
import asyncio
from urllib.parse import urljoin, urlparse

class WebCrawler:
    """Advanced web crawler with fallback mechanisms and proper error handling."""

    def __init__(self):
        self.session = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.quantum_urls = [
            "https://ionq.com/quantum",
            "https://azure.microsoft.com/solutions/quantum-computing/",
            "https://quantum.google/",
            "https://www.ibm.com/quantum"
        ]

    async def initialize(self):
        """Initialize aiohttp session with custom settings."""
        if not self.session:
            self.session = aiohttp.ClientSession(headers=self.headers)

    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None

    async def fetch_url(self, url: str, retries: int = 3) -> Optional[Dict]:
        """
        Fetch content from URL with retry mechanism and proper error handling.

        Args:
            url: URL to fetch
            retries: Number of retry attempts

        Returns:
            Dict containing status and content or error message
        """
        if not self.session:
            await self.initialize()

        for attempt in range(retries):
            try:
                async with self.session.get(url, timeout=30) as response:
                    if response.status == 200:
                        content = await response.text()
                        return {
                            'status': 'success',
                            'content': content,
                            'url': url
                        }
                    elif response.status == 403:
                        logging.warning(f"Access forbidden for {url}, might need authentication")
                        return {
                            'status': 'error',
                            'error': 'Access forbidden',
                            'url': url
                        }
                    else:
                        logging.error(f"Failed to fetch {url}: Status {response.status}")

            except asyncio.TimeoutError:
                logging.warning(f"Timeout while fetching {url}, attempt {attempt + 1}/{retries}")
            except Exception as e:
                logging.error(f"Error fetching {url}: {str(e)}")

            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

        return {
            'status': 'error',
            'error': 'Max retries exceeded',
            'url': url
        }

    def extract_content(self, html_content: str) -> Dict[str, str]:
        """
        Extract relevant content from HTML using BeautifulSoup.

        Args:
            html_content: Raw HTML content

        Returns:
            Dict containing extracted content
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove unwanted elements
            for element in soup.find_all(['script', 'style', 'nav', 'footer']):
                element.decompose()

            # Extract main content
            content = {
                'title': soup.title.string if soup.title else '',
                'text': ' '.join([p.get_text().strip() for p in soup.find_all('p') if p.get_text().strip()]),
                'headings': [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3'])]
            }

            return content

        except Exception as e:
            logging.error(f"Error extracting content: {str(e)}")
            return {'error': str(e)}

    async def crawl(self, urls: Optional[List[str]] = None) -> List[Dict]:
        """
        Crawl multiple URLs concurrently with quantum-relevant content prioritization.

        Args:
            urls: Optional list of URLs to crawl. If not provided, uses quantum-relevant URLs.

        Returns:
            List of dictionaries containing crawled content
        """
        results = []
        tasks = []

        try:
            # If no URLs provided, use quantum-relevant URLs
            urls_to_crawl = urls if urls else self.quantum_urls

            for url in urls_to_crawl:
                task = asyncio.create_task(self.fetch_url(url))
                tasks.append(task)

            responses = await asyncio.gather(*tasks, return_exceptions=True)

            for response in responses:
                if isinstance(response, dict) and response.get('status') == 'success':
                    content = self.extract_content(response['content'])
                    results.append({
                        'url': response['url'],
                        'content': content
                    })
                else:
                    logging.warning(f"Failed to process URL: {response}")

        except Exception as e:
            logging.error(f"Error during crawling: {str(e)}")

        return results

# Create a singleton instance
crawler = WebCrawler()