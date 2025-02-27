"""
QA³ Agent Core: True Autonomous Agency Implementation

This module provides the core functionality for a true agentic system that can:
1. Perceive its environment through computer vision
2. Make autonomous decisions based on observations
3. Take actions through UI automation
4. Learn from interactions
5. Manage goals and prioritize tasks

Inspired by principles from Microsoft Omniparser and Claude Computer Use
"""

import os
import json
import time
import base64
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
import asyncio
import numpy as np
from datetime import datetime
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("agent-core")

class AgentMemory:
    """
    Advanced memory system for the agent with episodic, semantic, and procedural memory
    
    This system enables the agent to:
    1. Remember past interactions (episodic)
    2. Store factual knowledge (semantic)
    3. Remember how to perform tasks (procedural)
    4. Retrieve relevant information based on context
    """
    
    def __init__(self, max_episodes: int = 100):
        """Initialize agent memory systems"""
        # Episodic memory - sequences of interactions and observations
        self.episodic_memory = []
        self.max_episodes = max_episodes
        
        # Semantic memory - facts, conceptual knowledge
        self.semantic_memory = {}
        
        # Procedural memory - how to perform tasks
        self.procedural_memory = []
        
        # Working memory - current context and state
        self.working_memory = {}
        
        # Observation history
        self.observation_history = []
        
        logger.info("Agent memory system initialized")
    
    def add_episode(self, episode: Dict[str, Any]):
        """Add an episode to episodic memory"""
        episode["timestamp"] = datetime.now().isoformat()
        self.episodic_memory.append(episode)
        
        # Maintain maximum size
        if len(self.episodic_memory) > self.max_episodes:
            self.episodic_memory.pop(0)
    
    def add_observation(self, observation: Dict[str, Any]):
        """Add an observation to history"""
        observation["timestamp"] = datetime.now().isoformat()
        self.observation_history.append(observation)
        
        # Maintain reasonable size
        if len(self.observation_history) > self.max_episodes:
            self.observation_history.pop(0)
    
    def add_semantic_knowledge(self, key: str, value: Any):
        """Store factual knowledge"""
        self.semantic_memory[key] = {
            "value": value,
            "last_updated": datetime.now().isoformat()
        }
    
    def add_procedural_knowledge(self, task_name: str, steps: List[Dict[str, Any]]):
        """Store knowledge about how to perform a task"""
        # Check if task already exists
        for i, proc in enumerate(self.procedural_memory):
            if proc["task_name"] == task_name:
                # Update existing procedure
                self.procedural_memory[i] = {
                    "task_name": task_name,
                    "steps": steps,
                    "last_updated": datetime.now().isoformat()
                }
                return
        
        # Add new procedure
        self.procedural_memory.append({
            "task_name": task_name,
            "steps": steps,
            "last_updated": datetime.now().isoformat()
        })
    
    def update_working_memory(self, key: str, value: Any):
        """Update working memory with new information"""
        self.working_memory[key] = value
    
    def get_semantic_knowledge(self, key: str, default: Any = None) -> Any:
        """Retrieve factual knowledge"""
        return self.semantic_memory.get(key, {}).get("value", default)
    
    def get_procedural_knowledge(self, task_name: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieve knowledge about how to perform a task"""
        for proc in self.procedural_memory:
            if proc["task_name"] == task_name:
                return proc["steps"]
        return None
    
    def get_relevant_episodes(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant episodes based on semantic similarity
        
        In a full implementation, this would use embedding similarity.
        This is a simplified version using keyword matching.
        """
        query_terms = query.lower().split()
        scored_episodes = []
        
        for episode in self.episodic_memory:
            # Calculate a simple relevance score
            score = 0
            episode_text = str(episode).lower()
            
            for term in query_terms:
                if term in episode_text:
                    score += 1
            
            if score > 0:
                scored_episodes.append((episode, score))
        
        # Sort by score and return top results
        scored_episodes.sort(key=lambda x: x[1], reverse=True)
        return [episode for episode, _ in scored_episodes[:max_results]]
    
    def get_relevant_observations(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant observations based on query"""
        query_terms = query.lower().split()
        scored_observations = []
        
        for obs in self.observation_history:
            # Calculate a simple relevance score
            score = 0
            obs_text = str(obs).lower()
            
            for term in query_terms:
                if term in obs_text:
                    score += 1
            
            if score > 0:
                scored_observations.append((obs, score))
        
        # Sort by score and return top results
        scored_observations.sort(key=lambda x: x[1], reverse=True)
        return [obs for obs, _ in scored_observations[:max_results]]
    
    def get_working_memory(self, key: str, default: Any = None) -> Any:
        """Retrieve information from working memory"""
        return self.working_memory.get(key, default)
    
    def clear_working_memory(self):
        """Clear working memory for new tasks"""
        self.working_memory = {}
    
    def clear_episodic_memory(self):
        """Clear episodic memory"""
        self.episodic_memory = []
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage"""
        return {
            "episodic_memory_count": len(self.episodic_memory),
            "semantic_memory_count": len(self.semantic_memory),
            "procedural_memory_count": len(self.procedural_memory),
            "working_memory_count": len(self.working_memory),
            "observation_history_count": len(self.observation_history)
        }


class PerceptionSystem:
    """
    Visual perception system for screen understanding
    
    This system enables the agent to:
    1. Capture and analyze screen contents
    2. Identify UI elements and their functions
    3. Extract text and structural information
    4. Build a semantic understanding of the interface
    
    Similar to systems used in Microsoft Omniparser, RPA tools, and Claude Computer Use
    """
    
    def __init__(self):
        """Initialize the perception system"""
        # Detection models would be initialized here in a full implementation
        self.initialized = False
        self.error_message = None
        
        try:
            # Attempt to import necessary libraries
            import cv2
            import PIL
            from PIL import Image
            self.initialized = True
        except ImportError as e:
            self.error_message = f"Failed to initialize perception system: {str(e)}"
            logger.error(self.error_message)
    
    def capture_screen(self) -> Dict[str, Any]:
        """
        Capture the current screen for analysis
        
        In a real implementation, this might use:
        - Selenium for web browsers
        - PyAutoGUI or similar for desktop applications
        - Platform-specific APIs
        
        Here we'll simulate this with a placeholder implementation.
        """
        if not self.initialized:
            return {"success": False, "error": self.error_message}
        
        try:
            # Simulated screen capture
            # In a real implementation, this would capture an actual screenshot
            return {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "screen_width": 1920,
                "screen_height": 1080,
                "image_data": None  # This would be actual image data in a real implementation
            }
        except Exception as e:
            logger.error(f"Error capturing screen: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def analyze_screen(self, screen_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze screen contents to detect UI elements and extract information
        
        This would use computer vision and OCR techniques to:
        1. Detect buttons, text fields, dropdowns, etc.
        2. Extract text content
        3. Identify interactive elements
        4. Build a semantic representation of the screen
        """
        if not self.initialized:
            return {"success": False, "error": self.error_message}
        
        if not screen_data.get("success", False):
            return {"success": False, "error": "Invalid screen data"}
        
        try:
            # Simulated analysis
            # In a real implementation, this would use CV models to analyze the screenshot
            
            # Example detected elements (simulated)
            ui_elements = [
                {
                    "type": "button",
                    "text": "Submit",
                    "bounds": [100, 200, 200, 250],
                    "confidence": 0.95,
                    "is_enabled": True
                },
                {
                    "type": "text_field",
                    "placeholder": "Enter your query",
                    "bounds": [100, 100, 500, 150],
                    "confidence": 0.92,
                    "is_enabled": True
                }
            ]
            
            # Extract text (simulated)
            text_elements = [
                {
                    "text": "Quantum Computing Interface",
                    "bounds": [400, 50, 800, 80],
                    "is_header": True,
                    "confidence": 0.98
                }
            ]
            
            return {
                "success": True,
                "ui_elements": ui_elements,
                "text_elements": text_elements,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error analyzing screen: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def identify_element_by_text(self, analysis: Dict[str, Any], text: str) -> Optional[Dict[str, Any]]:
        """Find a UI element by its text content"""
        if not analysis.get("success", False):
            return None
        
        text = text.lower()
        
        # Check UI elements
        for element in analysis.get("ui_elements", []):
            if "text" in element and text in element["text"].lower():
                return element
        
        # Check text elements
        for element in analysis.get("text_elements", []):
            if "text" in element and text in element["text"].lower():
                return element
        
        return None
    
    def identify_element_by_type(self, analysis: Dict[str, Any], element_type: str) -> List[Dict[str, Any]]:
        """Find all UI elements of a specific type"""
        if not analysis.get("success", False):
            return []
        
        element_type = element_type.lower()
        matching_elements = []
        
        for element in analysis.get("ui_elements", []):
            if "type" in element and element_type in element["type"].lower():
                matching_elements.append(element)
        
        return matching_elements
    
    def get_element_at_position(self, analysis: Dict[str, Any], x: int, y: int) -> Optional[Dict[str, Any]]:
        """Find the UI element at the given screen coordinates"""
        if not analysis.get("success", False):
            return None
        
        # Check all elements
        all_elements = analysis.get("ui_elements", []) + analysis.get("text_elements", [])
        
        for element in all_elements:
            if "bounds" in element:
                bounds = element["bounds"]
                if bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]:
                    return element
        
        return None


class ActionSystem:
    """
    System for taking actions within the environment
    
    This system enables the agent to:
    1. Click on UI elements
    2. Type text
    3. Navigate between screens
    4. Execute commands
    5. Interact with applications
    
    Inspired by RPA frameworks and Claude Computer Use capabilities
    """
    
    def __init__(self):
        """Initialize the action system"""
        self.initialized = False
        self.error_message = None
        
        try:
            # Attempt to import necessary libraries
            # In a full implementation, this would use selenium, pyautogui, or similar
            import selenium
            self.initialized = True
        except ImportError as e:
            self.error_message = f"Failed to initialize action system: {str(e)}"
            logger.error(self.error_message)
    
    def click(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """
        Click on a UI element
        
        Args:
            element: UI element with bounds information
        
        Returns:
            Dict with operation result
        """
        if not self.initialized:
            return {"success": False, "error": self.error_message}
        
        try:
            # Ensure element has bounds
            if "bounds" not in element:
                return {"success": False, "error": "Element has no bounds"}
            
            bounds = element["bounds"]
            center_x = (bounds[0] + bounds[2]) // 2
            center_y = (bounds[1] + bounds[3]) // 2
            
            # Simulated click
            # In a real implementation, this would perform an actual click
            logger.info(f"Clicking at position ({center_x}, {center_y})")
            
            return {
                "success": True,
                "action": "click",
                "position": (center_x, center_y),
                "element_type": element.get("type", "unknown"),
                "element_text": element.get("text", ""),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error clicking element: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def type_text(self, element: Dict[str, Any], text: str) -> Dict[str, Any]:
        """
        Type text into a UI element
        
        Args:
            element: UI element with bounds information
            text: Text to type
        
        Returns:
            Dict with operation result
        """
        if not self.initialized:
            return {"success": False, "error": self.error_message}
        
        try:
            # Ensure element has bounds
            if "bounds" not in element:
                return {"success": False, "error": "Element has no bounds"}
            
            # Ensure element is a text field
            if element.get("type", "") != "text_field":
                return {"success": False, "error": "Element is not a text field"}
            
            bounds = element["bounds"]
            center_x = (bounds[0] + bounds[2]) // 2
            center_y = (bounds[1] + bounds[3]) // 2
            
            # Simulated typing
            # In a real implementation, this would type actual text
            logger.info(f"Typing '{text}' at position ({center_x}, {center_y})")
            
            return {
                "success": True,
                "action": "type",
                "position": (center_x, center_y),
                "text": text,
                "element_type": element.get("type", "unknown"),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error typing text: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def navigate(self, url: str) -> Dict[str, Any]:
        """
        Navigate to a URL
        
        Args:
            url: URL to navigate to
        
        Returns:
            Dict with operation result
        """
        if not self.initialized:
            return {"success": False, "error": self.error_message}
        
        try:
            # Simulated navigation
            # In a real implementation, this would use a browser automation library
            logger.info(f"Navigating to {url}")
            
            return {
                "success": True,
                "action": "navigate",
                "url": url,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error navigating to URL: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def execute_command(self, command: str) -> Dict[str, Any]:
        """
        Execute a shell command
        
        Args:
            command: Shell command to execute
        
        Returns:
            Dict with operation result
        """
        if not self.initialized:
            return {"success": False, "error": self.error_message}
        
        try:
            # Simulated command execution
            # In a real implementation, this would use subprocess
            logger.info(f"Executing command: {command}")
            
            return {
                "success": True,
                "action": "execute",
                "command": command,
                "output": "Simulated command output",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error executing command: {str(e)}")
            return {"success": False, "error": str(e)}


class DecisionSystem:
    """
    Autonomous decision making system
    
    This system enables the agent to:
    1. Analyze the current situation
    2. Generate and evaluate options
    3. Select the best course of action
    4. Learn from successes and failures
    
    This is the core of true agency, combining perception and action capabilities
    with autonomous decision making.
    """
    
    def __init__(self, use_llm: bool = True):
        """
        Initialize the decision system
        
        Args:
            use_llm: Whether to use an LLM for decision making
        """
        self.use_llm = use_llm
        self.initialized = False
        self.error_message = None
        
        # Initialize LLM clients if available
        self.openai_client = None
        self.anthropic_client = None
        
        try:
            if use_llm:
                self._initialize_llm_clients()
            self.initialized = True
        except Exception as e:
            self.error_message = f"Failed to initialize decision system: {str(e)}"
            logger.error(self.error_message)
    
    def _initialize_llm_clients(self):
        """Initialize LLM clients if API keys are available"""
        # Initialize OpenAI client if available
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if openai_api_key:
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
                logger.info("OpenAI client initialized for decision making")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {str(e)}")
        
        # Initialize Anthropic client if available
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
                logger.info("Anthropic client initialized for decision making")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {str(e)}")
    
    async def analyze_situation(self, 
                              perception_data: Dict[str, Any], 
                              memory: AgentMemory,
                              goals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the current situation to inform decision making
        
        Args:
            perception_data: Current perception data
            memory: Agent memory
            goals: Current goals
            
        Returns:
            Dict with situation analysis
        """
        if not self.initialized:
            return {"success": False, "error": self.error_message}
        
        if not perception_data.get("success", False):
            return {"success": False, "error": "Invalid perception data"}
        
        try:
            if self.use_llm and (self.openai_client or self.anthropic_client):
                return await self._analyze_with_llm(perception_data, memory, goals)
            else:
                return self._analyze_with_rules(perception_data, memory, goals)
        except Exception as e:
            logger.error(f"Error analyzing situation: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _analyze_with_llm(self, 
                              perception_data: Dict[str, Any], 
                              memory: AgentMemory,
                              goals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use an LLM to analyze the situation"""
        # Create a prompt for the LLM
        prompt = self._create_analysis_prompt(perception_data, memory, goals)
        
        # Try with OpenAI first, then fall back to Anthropic
        if self.openai_client:
            try:
                response = await self._call_openai_analysis(prompt)
                return self._parse_llm_analysis(response)
            except Exception as e:
                logger.error(f"OpenAI analysis failed: {str(e)}")
                
        if self.anthropic_client:
            try:
                response = await self._call_anthropic_analysis(prompt)
                return self._parse_llm_analysis(response)
            except Exception as e:
                logger.error(f"Anthropic analysis failed: {str(e)}")
        
        # Fall back to rule-based analysis if both LLMs fail
        return self._analyze_with_rules(perception_data, memory, goals)
    
    async def _call_openai_analysis(self, prompt: str) -> str:
        """Call OpenAI API for situation analysis"""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        
        response = await asyncio.to_thread(
            self.openai_client.chat.completions.create,
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI agent analyzing a situation to make autonomous decisions. Provide a detailed, structured analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        return response.choices[0].message.content
    
    async def _call_anthropic_analysis(self, prompt: str) -> str:
        """Call Anthropic API for situation analysis"""
        if not self.anthropic_client:
            raise ValueError("Anthropic client not initialized")
        
        response = await asyncio.to_thread(
            self.anthropic_client.messages.create,
            model="claude-3-sonnet-20240229",
            max_tokens=2000,
            system="You are an AI agent analyzing a situation to make autonomous decisions. Provide a detailed, structured analysis.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text
    
    def _create_analysis_prompt(self, 
                              perception_data: Dict[str, Any], 
                              memory: AgentMemory,
                              goals: List[Dict[str, Any]]) -> str:
        """Create a prompt for LLM-based situation analysis"""
        # Format perception data
        ui_elements = perception_data.get("ui_elements", [])
        text_elements = perception_data.get("text_elements", [])
        
        ui_elements_str = "\n".join([
            f"- {element.get('type', 'unknown')}: '{element.get('text', '')}' at position {element.get('bounds', 'unknown')}" 
            for element in ui_elements
        ])
        
        text_elements_str = "\n".join([
            f"- '{element.get('text', '')}' at position {element.get('bounds', 'unknown')}"
            for element in text_elements
        ])
        
        # Format goals
        goals_str = "\n".join([
            f"- {goal.get('description', 'unknown')} (priority: {goal.get('priority', 'unknown')})"
            for goal in goals
        ])
        
        # Format working memory
        working_memory = memory.working_memory
        working_memory_str = "\n".join([
            f"- {key}: {str(value)}"
            for key, value in working_memory.items()
        ])
        
        # Create the prompt
        prompt = f"""
        # Current Situation Analysis
        
        ## UI Elements Detected:
        {ui_elements_str}
        
        ## Text Elements Detected:
        {text_elements_str}
        
        ## Current Goals:
        {goals_str}
        
        ## Working Memory:
        {working_memory_str}
        
        Please analyze this situation and provide:
        1. A summary of what's happening on the screen
        2. Identification of the most relevant UI elements for the current goals
        3. Potential obstacles or challenges in the current state
        4. Recommended focus areas for decision making
        
        Structure your analysis in JSON format with the following fields:
        - "screen_summary": A concise description of what's displayed
        - "relevant_elements": Array of the most important UI elements for current goals
        - "obstacles": Array of potential challenges
        - "focus_areas": Array of recommended focus areas for decision making
        """
        
        return prompt
    
    def _parse_llm_analysis(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into a structured analysis"""
        try:
            # Try to extract JSON from the response
            json_str = response
            
            # If the response contains markdown code blocks, extract JSON from them
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            
            analysis = json.loads(json_str)
            
            return {
                "success": True,
                "screen_summary": analysis.get("screen_summary", ""),
                "relevant_elements": analysis.get("relevant_elements", []),
                "obstacles": analysis.get("obstacles", []),
                "focus_areas": analysis.get("focus_areas", []),
                "raw_response": response
            }
        except Exception as e:
            logger.error(f"Error parsing LLM analysis: {str(e)}")
            
            # Return a simplified analysis based on the raw text
            return {
                "success": True,
                "screen_summary": "Unable to parse structured analysis",
                "relevant_elements": [],
                "obstacles": ["Analysis parsing failed"],
                "focus_areas": [],
                "raw_response": response
            }
    
    def _analyze_with_rules(self, 
                          perception_data: Dict[str, Any], 
                          memory: AgentMemory,
                          goals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use rule-based approaches to analyze the situation when LLMs aren't available"""
        ui_elements = perception_data.get("ui_elements", [])
        text_elements = perception_data.get("text_elements", [])
        
        # Simple rule-based analysis
        screen_summary = "Screen contains " + str(len(ui_elements)) + " UI elements and " + str(len(text_elements)) + " text elements"
        
        # Find interactive elements as relevant
        relevant_elements = [
            element for element in ui_elements
            if element.get("type") in ["button", "text_field", "checkbox", "dropdown"]
        ]
        
        # Simple obstacle detection
        obstacles = []
        if not relevant_elements:
            obstacles.append("No interactive elements detected")
        
        # Basic focus areas
        focus_areas = ["Identify main task", "Locate interactive elements"]
        
        return {
            "success": True,
            "screen_summary": screen_summary,
            "relevant_elements": relevant_elements,
            "obstacles": obstacles,
            "focus_areas": focus_areas
        }
    
    async def generate_options(self, 
                             situation_analysis: Dict[str, Any],
                             memory: AgentMemory,
                             goals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate possible action options based on situation analysis
        
        Args:
            situation_analysis: Analysis of the current situation
            memory: Agent memory
            goals: Current goals
            
        Returns:
            List of possible action options
        """
        if not self.initialized:
            return [{"success": False, "error": self.error_message}]
        
        if not situation_analysis.get("success", False):
            return [{"success": False, "error": "Invalid situation analysis"}]
        
        try:
            if self.use_llm and (self.openai_client or self.anthropic_client):
                return await self._generate_options_with_llm(situation_analysis, memory, goals)
            else:
                return self._generate_options_with_rules(situation_analysis, memory, goals)
        except Exception as e:
            logger.error(f"Error generating options: {str(e)}")
            return [{"success": False, "error": str(e)}]
    
    async def _generate_options_with_llm(self, 
                                       situation_analysis: Dict[str, Any],
                                       memory: AgentMemory,
                                       goals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use an LLM to generate decision options"""
        # Create a prompt for the LLM
        prompt = self._create_options_generation_prompt(situation_analysis, memory, goals)
        
        # Try with OpenAI first, then fall back to Anthropic
        if self.openai_client:
            try:
                response = await self._call_openai_options(prompt)
                return self._parse_llm_options(response)
            except Exception as e:
                logger.error(f"OpenAI options generation failed: {str(e)}")
                
        if self.anthropic_client:
            try:
                response = await self._call_anthropic_options(prompt)
                return self._parse_llm_options(response)
            except Exception as e:
                logger.error(f"Anthropic options generation failed: {str(e)}")
        
        # Fall back to rule-based generation if both LLMs fail
        return self._generate_options_with_rules(situation_analysis, memory, goals)
    
    async def _call_openai_options(self, prompt: str) -> str:
        """Call OpenAI API for options generation"""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        
        response = await asyncio.to_thread(
            self.openai_client.chat.completions.create,
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI agent generating action options based on a situation analysis. Provide detailed, concrete actions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        
        return response.choices[0].message.content
    
    async def _call_anthropic_options(self, prompt: str) -> str:
        """Call Anthropic API for options generation"""
        if not self.anthropic_client:
            raise ValueError("Anthropic client not initialized")
        
        response = await asyncio.to_thread(
            self.anthropic_client.messages.create,
            model="claude-3-sonnet-20240229",
            max_tokens=2000,
            system="You are an AI agent generating action options based on a situation analysis. Provide detailed, concrete actions.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text
    
    def _create_options_generation_prompt(self, 
                                        situation_analysis: Dict[str, Any],
                                        memory: AgentMemory,
                                        goals: List[Dict[str, Any]]) -> str:
        """Create a prompt for LLM-based options generation"""
        # Format situation analysis
        screen_summary = situation_analysis.get("screen_summary", "")
        
        relevant_elements_str = "\n".join([
            f"- {element.get('type', 'unknown')}: '{element.get('text', '')}' at position {element.get('bounds', 'unknown')}" 
            for element in situation_analysis.get("relevant_elements", [])
        ])
        
        obstacles_str = "\n".join([
            f"- {obstacle}" 
            for obstacle in situation_analysis.get("obstacles", [])
        ])
        
        focus_areas_str = "\n".join([
            f"- {focus}" 
            for focus in situation_analysis.get("focus_areas", [])
        ])
        
        # Format goals
        goals_str = "\n".join([
            f"- {goal.get('description', 'unknown')} (priority: {goal.get('priority', 'unknown')})"
            for goal in goals
        ])
        
        # Format working memory
        working_memory = memory.working_memory
        working_memory_str = "\n".join([
            f"- {key}: {str(value)}"
            for key, value in working_memory.items()
        ])
        
        # Create the prompt
        prompt = f"""
        # Situation Analysis
        
        ## Screen Summary:
        {screen_summary}
        
        ## Relevant UI Elements:
        {relevant_elements_str}
        
        ## Potential Obstacles:
        {obstacles_str}
        
        ## Recommended Focus Areas:
        {focus_areas_str}
        
        ## Current Goals:
        {goals_str}
        
        ## Working Memory:
        {working_memory_str}
        
        Based on this analysis, generate 3-5 possible action options. For each option, provide:
        1. A clear description of the action
        2. The specific UI element to interact with (if applicable)
        3. The type of interaction (click, type, etc.)
        4. Expected outcome
        5. Alignment with current goals
        
        Structure your response in JSON format with an array of options, each containing:
        - "description": Description of the action
        - "element_id": ID or description of the UI element to interact with
        - "interaction_type": Type of interaction (click, type, navigate, etc.)
        - "interaction_params": Additional parameters for the interaction (e.g., text to type)
        - "expected_outcome": What should happen after the action
        - "goal_alignment": How this aligns with current goals (high, medium, low)
        """
        
        return prompt
    
    def _parse_llm_options(self, response: str) -> List[Dict[str, Any]]:
        """Parse the LLM response into a structured list of options"""
        try:
            # Try to extract JSON from the response
            json_str = response
            
            # If the response contains markdown code blocks, extract JSON from them
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            
            data = json.loads(json_str)
            
            # Handle different possible structures
            if isinstance(data, list):
                options = data
            elif isinstance(data, dict) and "options" in data:
                options = data["options"]
            else:
                options = [data]
            
            # Add success field
            for option in options:
                option["success"] = True
            
            return options
            
        except Exception as e:
            logger.error(f"Error parsing LLM options: {str(e)}")
            
            # Return a single generic option based on the raw text
            return [{
                "success": True,
                "description": "Fallback option due to parsing error",
                "element_id": "unknown",
                "interaction_type": "click",
                "interaction_params": {},
                "expected_outcome": "Progress to next step",
                "goal_alignment": "medium",
                "raw_response": response
            }]
    
    def _generate_options_with_rules(self, 
                                   situation_analysis: Dict[str, Any],
                                   memory: AgentMemory,
                                   goals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use rule-based approaches to generate options when LLMs aren't available"""
        options = []
        
        # Get relevant elements from the analysis
        relevant_elements = situation_analysis.get("relevant_elements", [])
        
        # Generate options for each interactive element
        for element in relevant_elements:
            element_type = element.get("type", "")
            
            if element_type == "button":
                options.append({
                    "success": True,
                    "description": f"Click the '{element.get('text', 'unknown')}' button",
                    "element_id": str(element),
                    "interaction_type": "click",
                    "interaction_params": {},
                    "expected_outcome": "Button action triggered",
                    "goal_alignment": "medium"
                })
            
            elif element_type == "text_field":
                options.append({
                    "success": True,
                    "description": f"Enter text in the '{element.get('placeholder', 'unknown')}' field",
                    "element_id": str(element),
                    "interaction_type": "type",
                    "interaction_params": {"text": "sample input"},
                    "expected_outcome": "Text entered in field",
                    "goal_alignment": "medium"
                })
            
            elif element_type == "checkbox":
                options.append({
                    "success": True,
                    "description": f"Toggle the '{element.get('text', 'unknown')}' checkbox",
                    "element_id": str(element),
                    "interaction_type": "click",
                    "interaction_params": {},
                    "expected_outcome": "Checkbox toggled",
                    "goal_alignment": "medium"
                })
        
        # Always add a fallback option
        options.append({
            "success": True,
            "description": "Wait and observe",
            "element_id": "none",
            "interaction_type": "wait",
            "interaction_params": {"duration": 2},
            "expected_outcome": "Better understanding of the interface",
            "goal_alignment": "low"
        })
        
        return options
    
    async def select_best_option(self, options: List[Dict[str, Any]], goals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select the best option from the available choices
        
        Args:
            options: List of available options
            goals: Current goals
            
        Returns:
            The selected option
        """
        if not options:
            return {"success": False, "error": "No options available"}
        
        # Filter out unsuccessful options
        valid_options = [opt for opt in options if opt.get("success", False)]
        
        if not valid_options:
            return {"success": False, "error": "No valid options available"}
        
        try:
            # Score each option based on goal alignment
            scored_options = []
            
            for option in valid_options:
                goal_alignment = option.get("goal_alignment", "medium")
                
                # Convert text alignment to numerical score
                alignment_score = {
                    "high": 3,
                    "medium": 2,
                    "low": 1
                }.get(goal_alignment.lower(), 1)
                
                scored_options.append((option, alignment_score))
            
            # Sort by score (descending)
            scored_options.sort(key=lambda x: x[1], reverse=True)
            
            # Return the highest-scoring option
            selected_option = scored_options[0][0]
            selected_option["selection_reason"] = f"Highest goal alignment (score: {scored_options[0][1]})"
            
            return selected_option
            
        except Exception as e:
            logger.error(f"Error selecting best option: {str(e)}")
            
            # Fall back to first option
            return valid_options[0]


class GoalManagementSystem:
    """
    System for managing agent goals and tracking progress
    
    This system enables the agent to:
    1. Maintain a hierarchy of goals and subgoals
    2. Track progress toward goals
    3. Adjust priorities dynamically
    4. Handle goal conflicts
    5. Generate new goals based on observations
    
    This is a critical component for true autonomy and agency.
    """
    
    def __init__(self):
        """Initialize the goal management system"""
        self.goals = []
        self.next_goal_id = 1
    
    def add_goal(self, description: str, priority: int = 5) -> int:
        """
        Add a new top-level goal
        
        Args:
            description: Description of the goal
            priority: Priority level (1-10, with 10 being highest)
            
        Returns:
            Goal ID
        """
        goal_id = self.next_goal_id
        self.next_goal_id += 1
        
        goal = {
            "id": goal_id,
            "description": description,
            "priority": priority,
            "status": "active",
            "progress": 0,
            "subgoals": [],
            "parent_id": None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        self.goals.append(goal)
        return goal_id
    
    def add_subgoal(self, parent_id: int, description: str, priority: int = None) -> int:
        """
        Add a subgoal to an existing goal
        
        Args:
            parent_id: ID of the parent goal
            description: Description of the subgoal
            priority: Priority level (if None, inherits from parent)
            
        Returns:
            Goal ID
        """
        # Find the parent goal
        parent_goal = self._find_goal(parent_id)
        if not parent_goal:
            raise ValueError(f"Parent goal with ID {parent_id} not found")
        
        # Use parent's priority if not specified
        if priority is None:
            priority = parent_goal["priority"]
        
        # Create the subgoal
        goal_id = self.next_goal_id
        self.next_goal_id += 1
        
        subgoal = {
            "id": goal_id,
            "description": description,
            "priority": priority,
            "status": "active",
            "progress": 0,
            "subgoals": [],
            "parent_id": parent_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Add to parent's subgoals
        parent_goal["subgoals"].append(subgoal)
        
        return goal_id
    
    def update_goal_progress(self, goal_id: int, progress: int) -> bool:
        """
        Update the progress of a goal
        
        Args:
            goal_id: ID of the goal
            progress: Progress percentage (0-100)
            
        Returns:
            Success status
        """
        goal = self._find_goal(goal_id)
        if not goal:
            return False
        
        goal["progress"] = max(0, min(100, progress))
        goal["updated_at"] = datetime.now().isoformat()
        
        # If progress is 100%, mark as completed
        if goal["progress"] >= 100:
            goal["status"] = "completed"
        
        # Update parent goal progress based on subgoals
        if goal["parent_id"]:
            self._update_parent_progress(goal["parent_id"])
        
        return True
    
    def _update_parent_progress(self, parent_id: int):
        """Update a parent goal's progress based on its subgoals"""
        parent = self._find_goal(parent_id)
        if not parent or not parent["subgoals"]:
            return
        
        # Calculate average progress of subgoals
        total_progress = sum(sg["progress"] for sg in parent["subgoals"])
        avg_progress = total_progress / len(parent["subgoals"])
        
        parent["progress"] = int(avg_progress)
        parent["updated_at"] = datetime.now().isoformat()
        
        # If progress is 100%, mark as completed
        if parent["progress"] >= 100:
            parent["status"] = "completed"
        
        # Recursively update parent's parent
        if parent["parent_id"]:
            self._update_parent_progress(parent["parent_id"])
    
    def complete_goal(self, goal_id: int) -> bool:
        """
        Mark a goal as completed
        
        Args:
            goal_id: ID of the goal
            
        Returns:
            Success status
        """
        return self.update_goal_progress(goal_id, 100)
    
    def fail_goal(self, goal_id: int, reason: str) -> bool:
        """
        Mark a goal as failed
        
        Args:
            goal_id: ID of the goal
            reason: Reason for failure
            
        Returns:
            Success status
        """
        goal = self._find_goal(goal_id)
        if not goal:
            return False
        
        goal["status"] = "failed"
        goal["failure_reason"] = reason
        goal["updated_at"] = datetime.now().isoformat()
        
        return True
    
    def update_goal_priority(self, goal_id: int, priority: int) -> bool:
        """
        Update the priority of a goal
        
        Args:
            goal_id: ID of the goal
            priority: New priority level (1-10)
            
        Returns:
            Success status
        """
        goal = self._find_goal(goal_id)
        if not goal:
            return False
        
        goal["priority"] = max(1, min(10, priority))
        goal["updated_at"] = datetime.now().isoformat()
        
        return True
    
    def get_active_goals(self) -> List[Dict[str, Any]]:
        """Get all active top-level goals"""
        return [goal for goal in self.goals if goal["status"] == "active" and goal["parent_id"] is None]
    
    def get_all_goals(self) -> List[Dict[str, Any]]:
        """Get all goals"""
        return self.goals
    
    def get_highest_priority_goal(self) -> Optional[Dict[str, Any]]:
        """Get the active goal with the highest priority"""
        active_goals = [goal for goal in self.goals if goal["status"] == "active"]
        if not active_goals:
            return None
        
        # Sort by priority (descending)
        active_goals.sort(key=lambda g: g["priority"], reverse=True)
        return active_goals[0]
    
    def _find_goal(self, goal_id: int) -> Optional[Dict[str, Any]]:
        """Find a goal by ID (including subgoals)"""
        # Check top-level goals
        for goal in self.goals:
            if goal["id"] == goal_id:
                return goal
            
            # Check subgoals recursively
            subgoal = self._find_subgoal(goal, goal_id)
            if subgoal:
                return subgoal
        
        return None
    
    def _find_subgoal(self, parent: Dict[str, Any], goal_id: int) -> Optional[Dict[str, Any]]:
        """Recursively find a subgoal within a parent goal"""
        for subgoal in parent.get("subgoals", []):
            if subgoal["id"] == goal_id:
                return subgoal
            
            # Check deeper levels
            nested = self._find_subgoal(subgoal, goal_id)
            if nested:
                return nested
        
        return None


class LearningSystem:
    """
    System for learning from experiences and improving over time
    
    This system enables the agent to:
    1. Learn from successes and failures
    2. Improve decision making based on past experiences
    3. Adapt to new situations by generalizing from previous encounters
    4. Store procedural knowledge about how to accomplish tasks
    
    This component is essential for true agency and long-term improvement.
    """
    
    def __init__(self):
        """Initialize the learning system"""
        # Store experiences
        self.experiences = []
        
        # Store task procedures
        self.procedures = {}
    
    def add_experience(self, experience: Dict[str, Any]):
        """
        Add a new experience for learning
        
        Args:
            experience: Dictionary with experience details
        """
        # Add timestamp
        experience["timestamp"] = datetime.now().isoformat()
        
        # Add to experiences
        self.experiences.append(experience)
    
    def add_successful_procedure(self, task_name: str, steps: List[Dict[str, Any]]):
        """
        Store a successful procedure for a task
        
        Args:
            task_name: Name of the task
            steps: List of steps to accomplish the task
        """
        self.procedures[task_name] = {
            "steps": steps,
            "success_count": self.procedures.get(task_name, {}).get("success_count", 0) + 1,
            "last_update": datetime.now().isoformat()
        }
    
    def get_relevant_experiences(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Get experiences relevant to a given query
        
        Args:
            query: Query string
            max_results: Maximum number of results
            
        Returns:
            List of relevant experiences
        """
        query_terms = query.lower().split()
        scored_experiences = []
        
        for exp in self.experiences:
            # Calculate a simple relevance score
            score = 0
            exp_text = str(exp).lower()
            
            for term in query_terms:
                if term in exp_text:
                    score += 1
            
            if score > 0:
                scored_experiences.append((exp, score))
        
        # Sort by score and return top results
        scored_experiences.sort(key=lambda x: x[1], reverse=True)
        return [exp for exp, _ in scored_experiences[:max_results]]
    
    def get_procedure(self, task_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get a stored procedure for a task
        
        Args:
            task_name: Name of the task
            
        Returns:
            List of steps, or None if not found
        """
        procedure = self.procedures.get(task_name)
        if procedure:
            return procedure["steps"]
        return None
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about the learning system"""
        return {
            "experience_count": len(self.experiences),
            "procedure_count": len(self.procedures),
            "most_successful_procedures": [
                {
                    "task_name": task,
                    "success_count": info["success_count"]
                }
                for task, info in sorted(
                    self.procedures.items(),
                    key=lambda x: x[1]["success_count"],
                    reverse=True
                )[:5]
            ]
        }


class AutonomousAgent:
    """
    Main agent class that integrates all components for true agency
    
    This agent is capable of:
    1. Perceiving its environment through computer vision
    2. Taking actions to achieve goals
    3. Making autonomous decisions
    4. Learning from experience
    5. Managing its own goals
    
    This implementation draws inspiration from:
    - Microsoft Omniparser's screen understanding capabilities
    - Claude Computer Use's action system
    - Advanced AI agent architectures
    """
    
    def __init__(self, use_quantum: bool = True, n_qubits: int = 8):
        """
        Initialize the autonomous agent
        
        Args:
            use_quantum: Whether to use quantum computing capabilities
            n_qubits: Number of qubits for quantum simulations
        """
        # Agent configuration
        self.use_quantum = use_quantum
        self.n_qubits = n_qubits
        self.agent_name = "QA³ Agent"
        
        # Initialize key components
        self.memory = AgentMemory()
        self.perception = PerceptionSystem()
        self.action = ActionSystem()
        self.decision = DecisionSystem(use_llm=True)
        self.goals = GoalManagementSystem()
        self.learning = LearningSystem()
        
        # Initialize quantum capabilities
        self.quantum_initialized = False
        if use_quantum:
            self._initialize_quantum()
        
        # Agent state
        self.is_running = False
        self.current_task = None
        self.startup_time = datetime.now().isoformat()
        
        # Action and task queues
        self.action_queue = queue.Queue()
        self.task_queue = queue.Queue()
        
        # Register core objectives
        self.goals.add_goal("Understand the user interface", priority=8)
        self.goals.add_goal("Accomplish user-specified tasks", priority=10)
        self.goals.add_goal("Learn from interactions", priority=6)
        
        logger.info(f"Autonomous agent initialized with {n_qubits} qubits")
    
    def _initialize_quantum(self):
        """Initialize quantum computing capabilities"""
        try:
            # Check for PennyLane
            import pennylane as qml
            
            # Create a simple device
            self.quantum_device = qml.device("default.qubit", wires=self.n_qubits)
            self.quantum_initialized = True
            
            # Store in memory
            self.memory.add_semantic_knowledge("quantum_device", "PennyLane default.qubit")
            self.memory.add_semantic_knowledge("n_qubits", self.n_qubits)
            
            logger.info(f"Quantum core initialized with {self.n_qubits} qubits")
        except ImportError:
            logger.warning("PennyLane not available. Quantum features will be disabled.")
            self.quantum_initialized = False
    
    async def process_task(self, task: str) -> Dict[str, Any]:
        """
        Process a user task with autonomous capabilities
        
        Args:
            task: Description of the task to accomplish
            
        Returns:
            Dict with task results
        """
        logger.info(f"Processing task: {task}")
        
        # Store task in memory
        self.memory.add_episode({
            "type": "task_received",
            "task": task
        })
        
        # Update working memory
        self.memory.update_working_memory("current_task", task)
        self.current_task = task
        
        # Add as a goal
        goal_id = self.goals.add_goal(f"Complete task: {task}", priority=9)
        
        # Capture current state
        perception_result = self.perception.capture_screen()
        if perception_result["success"]:
            screen_analysis = self.perception.analyze_screen(perception_result)
            self.memory.add_observation(screen_analysis)
        
        # Analyze the situation
        situation = await self.decision.analyze_situation(
            screen_analysis if perception_result["success"] else {"success": False, "error": "Perception failed"},
            self.memory,
            self.goals.get_active_goals()
        )
        
        # Generate action options
        options = await self.decision.generate_options(
            situation,
            self.memory,
            self.goals.get_active_goals()
        )
        
        # Select best option
        selected_option = await self.decision.select_best_option(
            options,
            self.goals.get_active_goals()
        )
        
        # Execute the selected action
        result = await self.execute_action(selected_option)
        
        # Store the results
        self.memory.add_episode({
            "type": "action_result",
            "task": task,
            "action": selected_option,
            "result": result
        })
        
        # Update goal progress based on result
        if result.get("success", False):
            self.goals.update_goal_progress(goal_id, 100)
            
            # Store successful procedure
            self.learning.add_successful_procedure(
                f"task_{task}", 
                [{"action": selected_option, "result": result}]
            )
        else:
            self.goals.update_goal_progress(goal_id, 25)
        
        # Return the results
        return {
            "task": task,
            "success": result.get("success", False),
            "action_taken": selected_option,
            "result": result,
            "agent_state": self.get_status()
        }
    
    async def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a selected action
        
        Args:
            action: Details of the action to execute
            
        Returns:
            Dict with action results
        """
        if not action.get("success", False):
            return {"success": False, "error": "Invalid action"}
        
        interaction_type = action.get("interaction_type", "").lower()
        element_id = action.get("element_id", "")
        params = action.get("interaction_params", {})
        
        try:
            # Different actions based on interaction type
            if interaction_type == "click":
                # Find the element (in a real implementation, this would use a proper reference)
                element = {"type": "button", "bounds": [100, 100, 200, 150], "text": "Sample"}
                
                # Execute the click
                result = self.action.click(element)
                
                # Add to memory
                self.memory.add_episode({
                    "type": "action",
                    "action": "click",
                    "element": element,
                    "result": result
                })
                
                return result
                
            elif interaction_type == "type":
                # Find the element
                element = {"type": "text_field", "bounds": [100, 100, 500, 150], "placeholder": "Sample"}
                text = params.get("text", "")
                
                # Execute the typing
                result = self.action.type_text(element, text)
                
                # Add to memory
                self.memory.add_episode({
                    "type": "action",
                    "action": "type",
                    "element": element,
                    "text": text,
                    "result": result
                })
                
                return result
                
            elif interaction_type == "navigate":
                url = params.get("url", "")
                
                # Execute the navigation
                result = self.action.navigate(url)
                
                # Add to memory
                self.memory.add_episode({
                    "type": "action",
                    "action": "navigate",
                    "url": url,
                    "result": result
                })
                
                return result
                
            elif interaction_type == "wait":
                duration = params.get("duration", 1)
                
                # Simulate waiting
                await asyncio.sleep(duration)
                
                # Add to memory
                self.memory.add_episode({
                    "type": "action",
                    "action": "wait",
                    "duration": duration
                })
                
                return {
                    "success": True,
                    "action": "wait",
                    "duration": duration,
                    "timestamp": datetime.now().isoformat()
                }
                
            else:
                return {"success": False, "error": f"Unknown interaction type: {interaction_type}"}
                
        except Exception as e:
            logger.error(f"Error executing action: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the agent"""
        return {
            "agent_name": self.agent_name,
            "initialized": True,
            "quantum_initialized": self.quantum_initialized,
            "n_qubits": self.n_qubits,
            "current_task": self.current_task,
            "goals": len(self.goals.get_active_goals()),
            "memory_stats": self.memory.get_memory_stats(),
            "learning_stats": self.learning.get_learning_stats(),
            "startup_time": self.startup_time,
            "current_time": datetime.now().isoformat()
        }
    
    async def start_agent_loop(self):
        """Start the autonomous agent loop"""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("Starting autonomous agent loop")
        
        try:
            while self.is_running:
                # Check for new tasks
                if not self.task_queue.empty():
                    task = self.task_queue.get()
                    await self.process_task(task)
                
                # Check for new actions
                if not self.action_queue.empty():
                    action = self.action_queue.get()
                    await self.execute_action(action)
                
                # Perform autonomous perception if no active tasks
                if self.task_queue.empty() and self.action_queue.empty():
                    # Capture screen state
                    perception_result = self.perception.capture_screen()
                    if perception_result["success"]:
                        screen_analysis = self.perception.analyze_screen(perception_result)
                        self.memory.add_observation(screen_analysis)
                
                # Small delay to prevent CPU hogging
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error in agent loop: {str(e)}")
            self.is_running = False
    
    def stop_agent_loop(self):
        """Stop the autonomous agent loop"""
        self.is_running = False
        logger.info("Stopping autonomous agent loop")
    
    def add_task(self, task: str):
        """
        Add a task to the queue
        
        Args:
            task: Description of the task
        """
        self.task_queue.put(task)
    
    def add_action(self, action: Dict[str, Any]):
        """
        Add an action to the queue
        
        Args:
            action: Action to execute
        """
        self.action_queue.put(action)


async def run_agent(task: str) -> Dict[str, Any]:
    """
    Run the autonomous agent on a specific task
    
    Args:
        task: Description of the task
        
    Returns:
        Dict with task results
    """
    # Create and initialize the agent
    agent = AutonomousAgent()
    
    # Process the task
    result = await agent.process_task(task)
    
    return result


def run_agent_sync(task: str) -> Dict[str, Any]:
    """
    Synchronous wrapper for run_agent
    
    Args:
        task: Description of the task
        
    Returns:
        Dict with task results
    """
    return asyncio.run(run_agent(task))