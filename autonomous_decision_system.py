"""
Autonomous Decision System for Quantum-Accelerated AI Agent

This module provides true agentic capabilities through autonomous decision-making,
goal management, and self-evaluation mechanisms.
"""

import os
import logging
import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import random
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("autonomous-decision-system")

# Optional dependencies with proper error handling
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available. Using fallback mechanisms.")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available. Using fallback similarity calculations.")

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available. Text processing capabilities limited.")

# AI models with proper error handling
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI API not available for decision making.")

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic API not available for decision making.")


#####################################################
# Autonomous Goal Management
#####################################################

class Goal:
    """
    Represents a goal that the agent is trying to achieve.
    
    Goals can be hierarchical, with high-level goals broken down into subgoals.
    Each goal has a priority, status, and criteria for success.
    """
    
    def __init__(self, 
                 description: str, 
                 goal_type: str, 
                 priority: int = 5,
                 parent_goal_id: Optional[str] = None):
        """
        Initialize a goal with its core attributes.
        
        Args:
            description: Human-readable description of the goal
            goal_type: Type of goal (task, information, exploration, etc.)
            priority: Priority level (1-10, with 10 being highest)
            parent_goal_id: ID of parent goal if this is a subgoal
        """
        self.id = f"goal_{int(time.time())}_{random.randint(1000, 9999)}"
        self.description = description
        self.goal_type = goal_type
        self.priority = priority
        self.parent_goal_id = parent_goal_id
        self.subgoals = []
        
        # Status tracking
        self.status = "active"  # active, completed, failed, paused, archived
        self.progress = 0.0  # 0.0 to 1.0
        self.created_at = datetime.now().isoformat()
        self.completed_at = None
        self.failed_at = None
        
        # Success criteria and constraints
        self.success_criteria = []
        self.constraints = []
        
        # Execution history
        self.attempts = []
        self.results = []
        
    def add_subgoal(self, description: str, goal_type: str, priority: Optional[int] = None) -> str:
        """Add a subgoal and return its ID"""
        if priority is None:
            priority = max(1, self.priority - 1)  # Default to one level lower priority
            
        subgoal = Goal(description, goal_type, priority, self.id)
        self.subgoals.append(subgoal)
        return subgoal.id
        
    def add_success_criterion(self, criterion: str):
        """Add a success criterion for this goal"""
        self.success_criteria.append(criterion)
        
    def add_constraint(self, constraint: str):
        """Add a constraint for this goal"""
        self.constraints.append(constraint)
        
    def record_attempt(self, attempt_description: str, result: Any, success: bool = False):
        """Record an attempt to achieve this goal"""
        timestamp = datetime.now().isoformat()
        
        attempt = {
            "timestamp": timestamp,
            "description": attempt_description,
            "result": result,
            "success": success
        }
        
        self.attempts.append(attempt)
        self.results.append(result)
        
        # Update progress based on success
        if success:
            # Simple progress model - can be made more sophisticated
            self.progress = min(1.0, self.progress + (1.0 / max(1, len(self.success_criteria))))
            
        # Check if goal is completed
        if self.progress >= 1.0:
            self.complete()
            
    def complete(self):
        """Mark the goal as completed"""
        self.status = "completed"
        self.completed_at = datetime.now().isoformat()
        
    def fail(self, reason: str):
        """Mark the goal as failed"""
        self.status = "failed"
        self.failed_at = datetime.now().isoformat()
        self.attempts.append({
            "timestamp": datetime.now().isoformat(),
            "description": f"Goal failed: {reason}",
            "result": None,
            "success": False
        })
        
    def pause(self):
        """Pause the goal"""
        self.status = "paused"
        
    def resume(self):
        """Resume the goal"""
        if self.status == "paused":
            self.status = "active"
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert goal to dictionary representation"""
        return {
            "id": self.id,
            "description": self.description,
            "goal_type": self.goal_type,
            "priority": self.priority,
            "parent_goal_id": self.parent_goal_id,
            "status": self.status,
            "progress": self.progress,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "failed_at": self.failed_at,
            "success_criteria": self.success_criteria,
            "constraints": self.constraints,
            "attempts_count": len(self.attempts),
            "subgoals_count": len(self.subgoals)
        }


class GoalManager:
    """
    Manages the agent's goals and tracks their progress.
    
    The GoalManager enables autonomous behavior by maintaining:
    1. Active goals the agent is pursuing
    2. Goal hierarchies (goals and subgoals)
    3. Goal priorities and statuses
    """
    
    def __init__(self):
        """Initialize the goal manager"""
        self.goals = {}  # id -> Goal
        self.active_goals = []  # List of active goal IDs
        self.completed_goals = []  # List of completed goal IDs
        self.failed_goals = []  # List of failed goal IDs
        
    def create_goal(self, description: str, goal_type: str, priority: int = 5) -> str:
        """Create a new top-level goal and return its ID"""
        goal = Goal(description, goal_type, priority)
        self.goals[goal.id] = goal
        self.active_goals.append(goal.id)
        return goal.id
        
    def create_subgoal(self, parent_goal_id: str, description: str, goal_type: str, priority: Optional[int] = None) -> str:
        """Create a subgoal under a parent goal"""
        if parent_goal_id not in self.goals:
            raise ValueError(f"Parent goal {parent_goal_id} not found")
            
        parent_goal = self.goals[parent_goal_id]
        subgoal_id = parent_goal.add_subgoal(description, goal_type, priority)
        self.goals[subgoal_id] = parent_goal.subgoals[-1]
        self.active_goals.append(subgoal_id)
        return subgoal_id
        
    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Get a goal by ID"""
        return self.goals.get(goal_id)
        
    def complete_goal(self, goal_id: str):
        """Mark a goal as completed"""
        if goal_id not in self.goals:
            raise ValueError(f"Goal {goal_id} not found")
            
        goal = self.goals[goal_id]
        goal.complete()
        
        if goal_id in self.active_goals:
            self.active_goals.remove(goal_id)
        self.completed_goals.append(goal_id)
        
        # Check if this completes the parent goal
        if goal.parent_goal_id and goal.parent_goal_id in self.goals:
            parent = self.goals[goal.parent_goal_id]
            all_subgoals_completed = all(
                self.goals[subgoal.id].status == "completed" 
                for subgoal in parent.subgoals
            )
            
            if all_subgoals_completed:
                self.complete_goal(parent.id)
                
    def fail_goal(self, goal_id: str, reason: str):
        """Mark a goal as failed"""
        if goal_id not in self.goals:
            raise ValueError(f"Goal {goal_id} not found")
            
        goal = self.goals[goal_id]
        goal.fail(reason)
        
        if goal_id in self.active_goals:
            self.active_goals.remove(goal_id)
        self.failed_goals.append(goal_id)
        
    def get_highest_priority_goal(self) -> Optional[Goal]:
        """Get the active goal with the highest priority"""
        if not self.active_goals:
            return None
            
        highest_priority = -1
        highest_priority_goal = None
        
        for goal_id in self.active_goals:
            goal = self.goals[goal_id]
            if goal.priority > highest_priority:
                highest_priority = goal.priority
                highest_priority_goal = goal
                
        return highest_priority_goal
        
    def get_all_goals(self, include_completed: bool = False, include_failed: bool = False) -> List[Dict[str, Any]]:
        """Get all goals as dictionary representations"""
        result = []
        
        for goal_id, goal in self.goals.items():
            if goal.status == "active" or \
               (include_completed and goal.status == "completed") or \
               (include_failed and goal.status == "failed"):
                result.append(goal.to_dict())
                
        # Sort by priority (highest first)
        result.sort(key=lambda x: x["priority"], reverse=True)
        return result
        
    def prune_completed_goals(self, max_age_days: int = 7):
        """Remove old completed goals to free up memory"""
        now = datetime.now()
        to_remove = []
        
        for goal_id in self.completed_goals:
            goal = self.goals.get(goal_id)
            if not goal:
                continue
                
            completed_at = datetime.fromisoformat(goal.completed_at) if goal.completed_at else None
            if completed_at and (now - completed_at).days > max_age_days:
                to_remove.append(goal_id)
                
        for goal_id in to_remove:
            self.completed_goals.remove(goal_id)
            del self.goals[goal_id]
            
    def get_goal_status_counts(self) -> Dict[str, int]:
        """Get counts of goals by status"""
        counts = {
            "active": 0,
            "completed": 0,
            "failed": 0,
            "paused": 0
        }
        
        for goal in self.goals.values():
            counts[goal.status] = counts.get(goal.status, 0) + 1
            
        return counts


#####################################################
# Decision Making System
#####################################################

class DecisionOptions:
    """
    Represents a set of possible decisions with their evaluations.
    
    This framework allows the agent to:
    1. Generate multiple possible courses of action
    2. Evaluate each option using multiple criteria
    3. Make a decision based on the evaluations
    """
    
    def __init__(self):
        """Initialize a set of decision options"""
        self.options = []  # List of possible decisions
        self.evaluations = {}  # option_id -> criteria evaluations
        self.criteria_weights = {}  # criterion -> weight
        
    def add_option(self, option_id: str, description: str, actions: List[Dict[str, Any]]) -> None:
        """
        Add a possible decision option.
        
        Args:
            option_id: Unique identifier for this option
            description: Human-readable description of the option
            actions: List of concrete actions to take if this option is chosen
        """
        self.options.append({
            "id": option_id,
            "description": description,
            "actions": actions
        })
        self.evaluations[option_id] = {}
        
    def add_criterion(self, criterion: str, weight: float = 1.0) -> None:
        """
        Add an evaluation criterion.
        
        Args:
            criterion: Name of the criterion
            weight: Weight of this criterion in the final decision (higher = more important)
        """
        self.criteria_weights[criterion] = weight
        
    def evaluate_option(self, option_id: str, criterion: str, score: float, explanation: str) -> None:
        """
        Evaluate an option based on a specific criterion.
        
        Args:
            option_id: ID of the option to evaluate
            criterion: Name of the criterion
            score: Score for this criterion (0.0 to 1.0)
            explanation: Explanation for this evaluation
        """
        if option_id not in self.evaluations:
            raise ValueError(f"Option {option_id} not found")
            
        if criterion not in self.criteria_weights:
            raise ValueError(f"Criterion {criterion} not defined")
            
        self.evaluations[option_id][criterion] = {
            "score": score,
            "explanation": explanation
        }
        
    def get_best_option(self) -> Tuple[dict, float]:
        """
        Get the best option based on weighted evaluations.
        
        Returns:
            Tuple of (best_option, score)
        """
        if not self.options:
            return None, 0.0
            
        best_score = -1.0
        best_option = None
        
        for option in self.options:
            option_id = option["id"]
            evaluation = self.evaluations.get(option_id, {})
            
            # Calculate weighted score
            weighted_score = 0.0
            total_weights = sum(self.criteria_weights.values())
            
            for criterion, weight in self.criteria_weights.items():
                if criterion in evaluation:
                    criterion_score = evaluation[criterion]["score"]
                    weighted_score += (criterion_score * weight) / total_weights
                    
            if weighted_score > best_score:
                best_score = weighted_score
                best_option = option
                
        return best_option, best_score
        
    def get_all_evaluations(self) -> Dict[str, Any]:
        """Get all evaluations in a structured format"""
        result = {
            "options": self.options,
            "evaluations": self.evaluations,
            "criteria_weights": self.criteria_weights
        }
        
        for option in self.options:
            option_id = option["id"]
            option["total_score"] = self._calculate_option_score(option_id)
            
        return result
        
    def _calculate_option_score(self, option_id: str) -> float:
        """Calculate the weighted score for an option"""
        evaluation = self.evaluations.get(option_id, {})
        
        weighted_score = 0.0
        total_weights = sum(self.criteria_weights.values())
        
        for criterion, weight in self.criteria_weights.items():
            if criterion in evaluation:
                criterion_score = evaluation[criterion]["score"]
                weighted_score += (criterion_score * weight) / total_weights
                
        return weighted_score


class DecisionMaker:
    """
    Core decision-making system for autonomous behavior.
    
    This system enables the agent to:
    1. Analyze situations
    2. Generate possible approaches
    3. Evaluate options based on multiple criteria
    4. Make decisions that align with goals and constraints
    5. Learn from the outcomes of decisions
    """
    
    def __init__(self, use_llm: bool = True):
        """
        Initialize the decision maker.
        
        Args:
            use_llm: Whether to use LLMs for decision making when available
        """
        self.use_llm = use_llm
        self.openai_client = None
        self.anthropic_client = None
        
        self.decisions_history = []
        self.decision_templates = self._load_decision_templates()
        
        # Initialize LLM clients if available
        self._initialize_llm_clients()
        
    def _initialize_llm_clients(self):
        """Initialize LLM clients if available"""
        if OPENAI_AVAILABLE and self.use_llm:
            try:
                self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                logger.info("OpenAI client initialized for decision making")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {str(e)}")
                
        if ANTHROPIC_AVAILABLE and self.use_llm:
            try:
                self.anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
                logger.info("Anthropic client initialized for decision making")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {str(e)}")
                
    def _load_decision_templates(self) -> Dict[str, Any]:
        """Load decision templates for different scenarios"""
        # In a real implementation, these would be loaded from files or a database
        templates = {
            "general": {
                "criteria": ["effectiveness", "efficiency", "risk", "alignment"],
                "weights": {"effectiveness": 2.0, "efficiency": 1.0, "risk": 1.5, "alignment": 2.0}
            },
            "web_navigation": {
                "criteria": ["relevance", "information_value", "navigation_complexity", "page_trustworthiness"],
                "weights": {"relevance": 2.0, "information_value": 1.5, "navigation_complexity": 1.0, "page_trustworthiness": 1.8}
            },
            "search": {
                "criteria": ["query_relevance", "result_diversity", "information_gain"],
                "weights": {"query_relevance": 2.0, "result_diversity": 1.2, "information_gain": 1.5}
            },
            "tool_selection": {
                "criteria": ["appropriateness", "success_probability", "efficiency", "side_effects"],
                "weights": {"appropriateness": 2.0, "success_probability": 1.5, "efficiency": 1.0, "side_effects": 1.2}
            }
        }
        
        return templates
        
    async def analyze_situation(self, 
                              context: Dict[str, Any], 
                              goals: List[Dict[str, Any]],
                              constraints: List[str]) -> Dict[str, Any]:
        """
        Analyze a situation to prepare for decision making.
        
        Args:
            context: Current context including observations and state
            goals: Current active goals
            constraints: Constraints that must be respected
            
        Returns:
            Dictionary with situation analysis
        """
        # Prepare the analysis based on available tools
        if self.use_llm and (self.openai_client or self.anthropic_client):
            return await self._analyze_with_llm(context, goals, constraints)
        else:
            return self._analyze_with_rules(context, goals, constraints)
            
    async def _analyze_with_llm(self, context: Dict[str, Any], goals: List[Dict[str, Any]], constraints: List[str]) -> Dict[str, Any]:
        """Use an LLM to analyze the situation"""
        if not self.openai_client and not self.anthropic_client:
            return self._analyze_with_rules(context, goals, constraints)
            
        # Format the input for the LLM
        prompt = self._create_analysis_prompt(context, goals, constraints)
        
        try:
            if self.openai_client:
                response = await self._call_openai_analysis(prompt)
            else:
                response = await self._call_anthropic_analysis(prompt)
                
            # Parse the response
            analysis = self._parse_llm_analysis(response)
            return analysis
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {str(e)}")
            # Fall back to rule-based analysis
            return self._analyze_with_rules(context, goals, constraints)
            
    async def _call_openai_analysis(self, prompt: str) -> str:
        """Call OpenAI API for situation analysis"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert decision analyst helping an autonomous agent analyze situations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise
            
    async def _call_anthropic_analysis(self, prompt: str) -> str:
        """Call Anthropic API for situation analysis"""
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                system="You are an expert decision analyst helping an autonomous agent analyze situations.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic API call failed: {str(e)}")
            raise
            
    def _create_analysis_prompt(self, context: Dict[str, Any], goals: List[Dict[str, Any]], constraints: List[str]) -> str:
        """Create a prompt for LLM-based situation analysis"""
        prompt = "## Situation Analysis Request\n\n"
        
        # Add context
        prompt += "### Current Context\n"
        for key, value in context.items():
            if isinstance(value, (str, int, float, bool)):
                prompt += f"- {key}: {value}\n"
            elif isinstance(value, dict):
                prompt += f"- {key}: {json.dumps(value)[:200]}...\n"
            elif isinstance(value, list):
                prompt += f"- {key}: List with {len(value)} items\n"
            else:
                prompt += f"- {key}: Complex data\n"
                
        # Add goals
        prompt += "\n### Current Goals\n"
        for i, goal in enumerate(goals, 1):
            prompt += f"{i}. {goal['description']} (Priority: {goal['priority']}, Status: {goal['status']})\n"
            
        # Add constraints
        prompt += "\n### Constraints\n"
        for i, constraint in enumerate(constraints, 1):
            prompt += f"{i}. {constraint}\n"
            
        # Request specific analysis
        prompt += "\n### Analysis Request\n"
        prompt += "Please analyze this situation and provide:\n\n"
        prompt += "1. Key factors that should influence decisions\n"
        prompt += "2. Potential approaches to address the situation\n"
        prompt += "3. Risks and opportunities in the current context\n"
        prompt += "4. Alignment assessment with current goals\n"
        prompt += "5. Recommended decision criteria and their relative importance\n\n"
        prompt += "Format your response as clear sections that can be easily parsed."
        
        return prompt
        
    def _parse_llm_analysis(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into a structured analysis"""
        analysis = {
            "key_factors": [],
            "potential_approaches": [],
            "risks": [],
            "opportunities": [],
            "alignment_assessment": "",
            "recommended_criteria": {}
        }
        
        # Extract key factors
        key_factors_match = re.search(r"Key factors[:\n]+(.*?)(?=\n\n|\n#)", response, re.DOTALL)
        if key_factors_match:
            factors_text = key_factors_match.group(1).strip()
            factors = [f.strip() for f in re.findall(r"[-\*]\s*(.*?)(?=\n[-\*]|\Z)", factors_text, re.DOTALL)]
            analysis["key_factors"] = factors
            
        # Extract potential approaches
        approaches_match = re.search(r"Potential approaches[:\n]+(.*?)(?=\n\n|\n#)", response, re.DOTALL)
        if approaches_match:
            approaches_text = approaches_match.group(1).strip()
            approaches = [a.strip() for a in re.findall(r"[-\*]\s*(.*?)(?=\n[-\*]|\Z)", approaches_text, re.DOTALL)]
            analysis["potential_approaches"] = approaches
            
        # Extract risks and opportunities
        risks_match = re.search(r"Risks[:\n]+(.*?)(?=\n\n|\n#|Opportunities)", response, re.DOTALL)
        if risks_match:
            risks_text = risks_match.group(1).strip()
            risks = [r.strip() for r in re.findall(r"[-\*]\s*(.*?)(?=\n[-\*]|\Z)", risks_text, re.DOTALL)]
            analysis["risks"] = risks
            
        opps_match = re.search(r"Opportunities[:\n]+(.*?)(?=\n\n|\n#)", response, re.DOTALL)
        if opps_match:
            opps_text = opps_match.group(1).strip()
            opps = [o.strip() for o in re.findall(r"[-\*]\s*(.*?)(?=\n[-\*]|\Z)", opps_text, re.DOTALL)]
            analysis["opportunities"] = opps
            
        # Extract alignment assessment
        alignment_match = re.search(r"Alignment assessment[:\n]+(.*?)(?=\n\n|\n#)", response, re.DOTALL)
        if alignment_match:
            analysis["alignment_assessment"] = alignment_match.group(1).strip()
            
        # Extract recommended criteria
        criteria_match = re.search(r"Recommended decision criteria[:\n]+(.*?)(?=\n\n|\n#|\Z)", response, re.DOTALL)
        if criteria_match:
            criteria_text = criteria_match.group(1).strip()
            criteria_items = re.findall(r"[-\*]\s*(.*?)(?:\((.*?)\))?(?=\n[-\*]|\Z)", criteria_text, re.DOTALL)
            
            for item in criteria_items:
                criterion = item[0].strip()
                weight_match = re.search(r"(?:importance|weight|priority)[:\s]*([\d\.]+)", item[1]) if len(item) > 1 else None
                
                if weight_match:
                    weight = float(weight_match.group(1))
                else:
                    weight = 1.0
                    
                analysis["recommended_criteria"][criterion] = weight
                
        return analysis
        
    def _analyze_with_rules(self, context: Dict[str, Any], goals: List[Dict[str, Any]], constraints: List[str]) -> Dict[str, Any]:
        """Use rule-based approaches to analyze the situation when LLMs aren't available"""
        analysis = {
            "key_factors": [],
            "potential_approaches": [],
            "risks": [],
            "opportunities": [],
            "alignment_assessment": "",
            "recommended_criteria": {}
        }
        
        # Extract basic context factors
        for key, value in context.items():
            if key in ["current_task", "location", "user_request", "environment"]:
                analysis["key_factors"].append(f"{key}: {str(value)}")
                
        # Identify basic approaches based on goals
        for goal in goals:
            if goal["status"] == "active":
                analysis["potential_approaches"].append(f"Address goal: {goal['description']}")
                
        # Identify basic risks based on constraints
        for constraint in constraints:
            analysis["risks"].append(f"Violation of constraint: {constraint}")
            
        # Add default criteria
        analysis["recommended_criteria"] = {
            "effectiveness": 1.0,
            "efficiency": 1.0,
            "risk": 1.0,
            "alignment": 1.0
        }
        
        return analysis
        
    async def generate_options(self, 
                             situation_analysis: Dict[str, Any], 
                             context: Dict[str, Any],
                             decision_type: str = "general") -> DecisionOptions:
        """
        Generate decision options based on situation analysis.
        
        Args:
            situation_analysis: Analysis of the current situation
            context: Current context information
            decision_type: Type of decision (general, web_navigation, search, tool_selection)
            
        Returns:
            DecisionOptions object with generated options
        """
        # Create options container
        options = DecisionOptions()
        
        # Add criteria based on the decision type and analysis
        template = self.decision_templates.get(decision_type, self.decision_templates["general"])
        
        for criterion in template["criteria"]:
            options.add_criterion(criterion, template["weights"].get(criterion, 1.0))
            
        # Add recommended criteria from the analysis
        for criterion, weight in situation_analysis.get("recommended_criteria", {}).items():
            if criterion not in template["criteria"]:
                options.add_criterion(criterion, weight)
                
        # Generate options based on available methods
        if self.use_llm and (self.openai_client or self.anthropic_client):
            await self._generate_options_with_llm(options, situation_analysis, context, decision_type)
        else:
            self._generate_options_with_rules(options, situation_analysis, context, decision_type)
            
        return options
        
    async def _generate_options_with_llm(self, 
                                       options: DecisionOptions, 
                                       analysis: Dict[str, Any], 
                                       context: Dict[str, Any],
                                       decision_type: str):
        """Use an LLM to generate decision options"""
        if not self.openai_client and not self.anthropic_client:
            self._generate_options_with_rules(options, analysis, context, decision_type)
            return
            
        # Create the prompt
        prompt = self._create_options_generation_prompt(analysis, context, decision_type)
        
        try:
            if self.openai_client:
                response = await self._call_openai_options(prompt)
            else:
                response = await self._call_anthropic_options(prompt)
                
            # Parse and add the options
            generated_options = self._parse_llm_options(response)
            
            for i, option in enumerate(generated_options, 1):
                option_id = f"option_{i}"
                options.add_option(
                    option_id=option_id,
                    description=option["description"],
                    actions=option["actions"]
                )
                
                # Add evaluations if provided
                for criterion in options.criteria_weights.keys():
                    if criterion in option.get("evaluations", {}):
                        eval_data = option["evaluations"][criterion]
                        options.evaluate_option(
                            option_id=option_id,
                            criterion=criterion,
                            score=eval_data.get("score", 0.5),
                            explanation=eval_data.get("explanation", "")
                        )
                        
        except Exception as e:
            logger.error(f"LLM options generation failed: {str(e)}")
            # Fall back to rule-based generation
            self._generate_options_with_rules(options, analysis, context, decision_type)
            
    async def _call_openai_options(self, prompt: str) -> str:
        """Call OpenAI API for options generation"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert decision analyst generating options for an autonomous agent."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=2000
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise
            
    async def _call_anthropic_options(self, prompt: str) -> str:
        """Call Anthropic API for options generation"""
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                system="You are an expert decision analyst generating options for an autonomous agent.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic API call failed: {str(e)}")
            raise
            
    def _create_options_generation_prompt(self, analysis: Dict[str, Any], context: Dict[str, Any], decision_type: str) -> str:
        """Create a prompt for LLM-based options generation"""
        prompt = "## Decision Options Generation Request\n\n"
        
        # Add decision type
        prompt += f"### Decision Type: {decision_type}\n\n"
        
        # Add situation analysis
        prompt += "### Situation Analysis\n"
        
        if analysis.get("key_factors"):
            prompt += "Key factors:\n"
            for factor in analysis["key_factors"]:
                prompt += f"- {factor}\n"
                
        if analysis.get("potential_approaches"):
            prompt += "\nPotential approaches:\n"
            for approach in analysis["potential_approaches"]:
                prompt += f"- {approach}\n"
                
        if analysis.get("risks"):
            prompt += "\nRisks:\n"
            for risk in analysis["risks"]:
                prompt += f"- {risk}\n"
                
        if analysis.get("opportunities"):
            prompt += "\nOpportunities:\n"
            for opp in analysis["opportunities"]:
                prompt += f"- {opp}\n"
                
        if analysis.get("alignment_assessment"):
            prompt += f"\nAlignment assessment: {analysis['alignment_assessment']}\n"
            
        # Add relevant context
        prompt += "\n### Relevant Context\n"
        relevant_keys = [
            "current_task", "current_url", "user_request", 
            "available_tools", "constraints", "web_content",
            "screen_elements"
        ]
        
        for key in relevant_keys:
            if key in context:
                value = context[key]
                if isinstance(value, (str, int, float, bool)):
                    prompt += f"- {key}: {value}\n"
                elif isinstance(value, dict):
                    prompt += f"- {key}: {json.dumps(value)[:200]}...\n"
                elif isinstance(value, list):
                    prompt += f"- {key}: List with {len(value)} items\n"
                else:
                    prompt += f"- {key}: Complex data\n"
                    
        # Add decision criteria
        template = self.decision_templates.get(decision_type, self.decision_templates["general"])
        prompt += "\n### Decision Criteria\n"
        
        for criterion, weight in template["weights"].items():
            prompt += f"- {criterion} (weight: {weight})\n"
            
        for criterion, weight in analysis.get("recommended_criteria", {}).items():
            if criterion not in template["criteria"]:
                prompt += f"- {criterion} (weight: {weight})\n"
                
        # Request format
        prompt += "\n### Request\n"
        prompt += "Generate 3-5 concrete decision options. For each option, provide:\n\n"
        prompt += "1. A clear description of the option\n"
        prompt += "2. Specific actions to execute this option (as an array of action objects)\n"
        prompt += "3. An evaluation for each decision criterion (score 0.0-1.0 and explanation)\n\n"
        
        prompt += "Format the response as a JSON array of options with this structure:\n"
        prompt += """
```json
[
  {
    "description": "Option description",
    "actions": [
      {"action_type": "web_navigation", "url": "https://example.com"},
      {"action_type": "click", "selector": "#button-id"}
    ],
    "evaluations": {
      "criterion1": {"score": 0.8, "explanation": "Explanation for score"},
      "criterion2": {"score": 0.5, "explanation": "Explanation for score"}
    }
  }
]
```
"""
        
        return prompt
        
    def _parse_llm_options(self, response: str) -> List[Dict[str, Any]]:
        """Parse the LLM response into a structured list of options"""
        # Extract JSON from the response
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON without the code blocks
            json_match = re.search(r"\[\s*\{[\s\S]*\}\s*\]", response)
            if json_match:
                json_str = json_match.group(0)
            else:
                logger.error("Could not extract JSON from LLM response")
                return []
                
        try:
            options = json.loads(json_str)
            return options
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {str(e)}")
            return []
            
    def _generate_options_with_rules(self, 
                                  options: DecisionOptions, 
                                  analysis: Dict[str, Any], 
                                  context: Dict[str, Any],
                                  decision_type: str):
        """Use rule-based approaches to generate options when LLMs aren't available"""
        # Generate basic options based on decision type
        if decision_type == "web_navigation":
            self._generate_web_navigation_options(options, analysis, context)
        elif decision_type == "search":
            self._generate_search_options(options, analysis, context)
        elif decision_type == "tool_selection":
            self._generate_tool_selection_options(options, analysis, context)
        else:
            self._generate_general_options(options, analysis, context)
            
    def _generate_web_navigation_options(self, options: DecisionOptions, analysis: Dict[str, Any], context: Dict[str, Any]):
        """Generate options for web navigation decisions"""
        # Option 1: Navigate to a specific URL if provided
        if "target_url" in context:
            options.add_option(
                option_id="nav_direct",
                description=f"Navigate directly to the specified URL: {context['target_url']}",
                actions=[
                    {"action_type": "web_navigation", "url": context["target_url"]}
                ]
            )
            
            # Add evaluations
            for criterion in options.criteria_weights.keys():
                if criterion == "relevance":
                    options.evaluate_option(
                        option_id="nav_direct",
                        criterion=criterion,
                        score=0.9,
                        explanation="Direct navigation to the specified URL is highly relevant to the task."
                    )
                elif criterion == "information_value":
                    options.evaluate_option(
                        option_id="nav_direct",
                        criterion=criterion,
                        score=0.7,
                        explanation="The specified URL likely contains valuable information for the task."
                    )
                elif criterion == "navigation_complexity":
                    options.evaluate_option(
                        option_id="nav_direct",
                        criterion=criterion,
                        score=0.9,
                        explanation="Direct navigation is simple and has low complexity."
                    )
                elif criterion == "page_trustworthiness":
                    # Simple heuristic: well-known domains are more trustworthy
                    domain = context['target_url'].split('/')[2] if '://' in context['target_url'] else context['target_url'].split('/')[0]
                    trusted_domains = ["wikipedia.org", "github.com", "google.com", "microsoft.com", "apple.com", "amazon.com"]
                    score = 0.8 if any(td in domain for td in trusted_domains) else 0.5
                    
                    options.evaluate_option(
                        option_id="nav_direct",
                        criterion=criterion,
                        score=score,
                        explanation=f"Domain '{domain}' trustworthiness assessment."
                    )
                    
        # Option 2: Search for information related to the task
        if "search_query" in context or "current_task" in context:
            query = context.get("search_query", context.get("current_task", ""))
            options.add_option(
                option_id="search_first",
                description=f"Search for information about: {query}",
                actions=[
                    {"action_type": "web_search", "query": query, "search_engine": "google"}
                ]
            )
            
            # Add evaluations
            for criterion in options.criteria_weights.keys():
                if criterion == "relevance":
                    options.evaluate_option(
                        option_id="search_first",
                        criterion=criterion,
                        score=0.8,
                        explanation="Searching provides access to relevant information but requires filtering."
                    )
                elif criterion == "information_value":
                    options.evaluate_option(
                        option_id="search_first",
                        criterion=criterion,
                        score=0.9,
                        explanation="Search results can provide valuable diverse information sources."
                    )
                elif criterion == "navigation_complexity":
                    options.evaluate_option(
                        option_id="search_first",
                        criterion=criterion,
                        score=0.7,
                        explanation="Searching requires additional steps to filter and select results."
                    )
                elif criterion == "page_trustworthiness":
                    options.evaluate_option(
                        option_id="search_first",
                        criterion=criterion,
                        score=0.6,
                        explanation="Search results vary in trustworthiness and require evaluation."
                    )
            
        # Option 3: If already on a page with links, navigate to a relevant link
        if context.get("current_url") and context.get("links"):
            links = context.get("links", [])
            if links:
                # Simple relevance matching - in a real implementation this would be more sophisticated
                most_relevant_link = links[0]  # Just pick the first one in this simple implementation
                
                options.add_option(
                    option_id="follow_link",
                    description=f"Follow a relevant link: {most_relevant_link.get('text', '')}",
                    actions=[
                        {"action_type": "click", "selector": f"a[href*='{most_relevant_link.get('href', '')}']"}
                    ]
                )
                
                # Add evaluations
                for criterion in options.criteria_weights.keys():
                    if criterion == "relevance":
                        options.evaluate_option(
                            option_id="follow_link",
                            criterion=criterion,
                            score=0.7,
                            explanation="Following a related link may lead to relevant information."
                        )
                    elif criterion == "information_value":
                        options.evaluate_option(
                            option_id="follow_link",
                            criterion=criterion,
                            score=0.6,
                            explanation="Link may provide additional context but relevance is uncertain."
                        )
                    elif criterion == "navigation_complexity":
                        options.evaluate_option(
                            option_id="follow_link",
                            criterion=criterion,
                            score=0.8,
                            explanation="Following a link is a simple navigation action."
                        )
                    elif criterion == "page_trustworthiness":
                        options.evaluate_option(
                            option_id="follow_link",
                            criterion=criterion,
                            score=0.5,
                            explanation="Link destination's trustworthiness is unknown without additional analysis."
                        )
                        
    def _generate_search_options(self, options: DecisionOptions, analysis: Dict[str, Any], context: Dict[str, Any]):
        """Generate options for search decisions"""
        query_base = context.get("search_query", context.get("current_task", ""))
        
        # Option 1: Basic search with the provided query
        options.add_option(
            option_id="basic_search",
            description=f"Basic search with query: {query_base}",
            actions=[
                {"action_type": "web_search", "query": query_base, "search_engine": "google"}
            ]
        )
        
        # Option 2: Enhanced search with additional keywords
        enhanced_query = f"{query_base} detailed tutorial documentation examples"
        options.add_option(
            option_id="enhanced_search",
            description=f"Enhanced search with additional keywords: {enhanced_query}",
            actions=[
                {"action_type": "web_search", "query": enhanced_query, "search_engine": "google"}
            ]
        )
        
        # Option 3: Search with site-specific qualifier
        if "domain_preference" in context:
            domain = context["domain_preference"]
            site_specific_query = f"{query_base} site:{domain}"
            options.add_option(
                option_id="site_specific_search",
                description=f"Search within {domain}: {site_specific_query}",
                actions=[
                    {"action_type": "web_search", "query": site_specific_query, "search_engine": "google"}
                ]
            )
            
        # Add evaluations
        for criterion in options.criteria_weights.keys():
            # Evaluate basic search
            if criterion == "query_relevance":
                options.evaluate_option(
                    option_id="basic_search",
                    criterion=criterion,
                    score=0.7,
                    explanation="Basic query directly related to the task."
                )
            elif criterion == "result_diversity":
                options.evaluate_option(
                    option_id="basic_search",
                    criterion=criterion,
                    score=0.6,
                    explanation="Basic search may return diverse but potentially unfocused results."
                )
            elif criterion == "information_gain":
                options.evaluate_option(
                    option_id="basic_search",
                    criterion=criterion,
                    score=0.5,
                    explanation="Moderate information gain expected from basic search."
                )
                
            # Evaluate enhanced search
            if criterion == "query_relevance":
                options.evaluate_option(
                    option_id="enhanced_search",
                    criterion=criterion,
                    score=0.8,
                    explanation="Enhanced query includes helpful qualifiers while staying relevant."
                )
            elif criterion == "result_diversity":
                options.evaluate_option(
                    option_id="enhanced_search",
                    criterion=criterion,
                    score=0.8,
                    explanation="Additional keywords should increase result diversity."
                )
            elif criterion == "information_gain":
                options.evaluate_option(
                    option_id="enhanced_search",
                    criterion=criterion,
                    score=0.7,
                    explanation="Higher information gain expected from enhanced search with tutorials and examples."
                )
                
            # Evaluate site-specific search if applicable
            if "domain_preference" in context:
                if criterion == "query_relevance":
                    options.evaluate_option(
                        option_id="site_specific_search",
                        criterion=criterion,
                        score=0.9,
                        explanation=f"Site-specific search targets the preferred domain {context['domain_preference']}."
                    )
                elif criterion == "result_diversity":
                    options.evaluate_option(
                        option_id="site_specific_search",
                        criterion=criterion,
                        score=0.4,
                        explanation="Limited to a single domain, reducing result diversity."
                    )
                elif criterion == "information_gain":
                    options.evaluate_option(
                        option_id="site_specific_search",
                        criterion=criterion,
                        score=0.8,
                        explanation=f"High-quality information expected from the preferred domain {context['domain_preference']}."
                    )
                    
    def _generate_tool_selection_options(self, options: DecisionOptions, analysis: Dict[str, Any], context: Dict[str, Any]):
        """Generate options for tool selection decisions"""
        available_tools = context.get("available_tools", [])
        current_task = context.get("current_task", "")
        
        # Generate options for each available tool that might be relevant
        for i, tool in enumerate(available_tools[:3]):  # Limit to 3 tools
            tool_name = tool.get("name", f"tool_{i}")
            tool_description = tool.get("description", "")
            
            options.add_option(
                option_id=f"use_{tool_name}",
                description=f"Use {tool_name}: {tool_description}",
                actions=[
                    {"action_type": "execute_tool", "tool_name": tool_name, "parameters": {}}
                ]
            )
            
            # Add evaluations
            for criterion in options.criteria_weights.keys():
                if criterion == "appropriateness":
                    # Simple keyword matching - in real implementation this would be more sophisticated
                    relevance = 0.5  # Default
                    if any(keyword in tool_description.lower() for keyword in current_task.lower().split()):
                        relevance = 0.8
                        
                    options.evaluate_option(
                        option_id=f"use_{tool_name}",
                        criterion=criterion,
                        score=relevance,
                        explanation=f"Tool purpose match assessment with current task: {current_task}"
                    )
                elif criterion == "success_probability":
                    options.evaluate_option(
                        option_id=f"use_{tool_name}",
                        criterion=criterion,
                        score=0.6,  # Default moderate probability
                        explanation="Estimated probability of successful execution based on available information."
                    )
                elif criterion == "efficiency":
                    options.evaluate_option(
                        option_id=f"use_{tool_name}",
                        criterion=criterion,
                        score=0.7,  # Default good efficiency
                        explanation="Estimated efficiency based on tool functionality."
                    )
                elif criterion == "side_effects":
                    options.evaluate_option(
                        option_id=f"use_{tool_name}",
                        criterion=criterion,
                        score=0.8,  # Default low side effects (high score is good)
                        explanation="Estimated minimal side effects on system state."
                    )
                    
    def _generate_general_options(self, options: DecisionOptions, analysis: Dict[str, Any], context: Dict[str, Any]):
        """Generate options for general decisions"""
        # Option 1: Execute the most direct approach identified in the analysis
        if analysis.get("potential_approaches"):
            direct_approach = analysis["potential_approaches"][0]
            options.add_option(
                option_id="direct_approach",
                description=f"Direct approach: {direct_approach}",
                actions=[
                    {"action_type": "custom", "description": direct_approach}
                ]
            )
            
        # Option 2: Gather more information before proceeding
        options.add_option(
            option_id="gather_info",
            description="Gather more information before proceeding",
            actions=[
                {"action_type": "web_search", "query": context.get("current_task", "")},
                {"action_type": "analyze_results"}
            ]
        )
        
        # Option 3: Break down the task into subtasks
        options.add_option(
            option_id="decompose_task",
            description="Break down the task into smaller subtasks",
            actions=[
                {"action_type": "task_decomposition", "task": context.get("current_task", "")}
            ]
        )
        
        # Add basic evaluations
        for option_id in ["direct_approach", "gather_info", "decompose_task"]:
            if option_id in [opt["id"] for opt in options.options]:
                for criterion in options.criteria_weights.keys():
                    # Simple defaults - in a real implementation these would be more context-dependent
                    if criterion == "effectiveness":
                        scores = {"direct_approach": 0.7, "gather_info": 0.5, "decompose_task": 0.8}
                        options.evaluate_option(
                            option_id=option_id,
                            criterion=criterion,
                            score=scores.get(option_id, 0.5),
                            explanation=f"Estimated effectiveness for {option_id}"
                        )
                    elif criterion == "efficiency":
                        scores = {"direct_approach": 0.8, "gather_info": 0.4, "decompose_task": 0.6}
                        options.evaluate_option(
                            option_id=option_id,
                            criterion=criterion,
                            score=scores.get(option_id, 0.5),
                            explanation=f"Estimated efficiency for {option_id}"
                        )
                    elif criterion == "risk":
                        scores = {"direct_approach": 0.6, "gather_info": 0.9, "decompose_task": 0.7}
                        options.evaluate_option(
                            option_id=option_id,
                            criterion=criterion,
                            score=scores.get(option_id, 0.5),
                            explanation=f"Estimated risk level for {option_id} (higher is better/lower risk)"
                        )
                    elif criterion == "alignment":
                        scores = {"direct_approach": 0.8, "gather_info": 0.7, "decompose_task": 0.9}
                        options.evaluate_option(
                            option_id=option_id,
                            criterion=criterion,
                            score=scores.get(option_id, 0.5),
                            explanation=f"Estimated alignment with goals for {option_id}"
                        )
                        
    async def make_decision(self, options: DecisionOptions) -> Dict[str, Any]:
        """
        Make a decision by selecting the best option based on evaluations.
        
        Args:
            options: DecisionOptions object with evaluated options
            
        Returns:
            Dictionary with the selected option and decision metadata
        """
        # Get the best option based on evaluations
        best_option, score = options.get_best_option()
        
        if not best_option:
            return {
                "success": False,
                "error": "No valid options available for decision",
                "decision_time": datetime.now().isoformat()
            }
            
        decision = {
            "success": True,
            "selected_option": best_option,
            "score": score,
            "decision_time": datetime.now().isoformat(),
            "all_options": options.get_all_evaluations()
        }
        
        # Add to decision history
        self.decisions_history.append(decision)
        
        return decision
        
    def record_decision_outcome(self, decision: Dict[str, Any], outcome: Dict[str, Any]) -> None:
        """
        Record the outcome of a decision for learning and improvement.
        
        Args:
            decision: The decision that was made
            outcome: The outcome information including success/failure and metrics
        """
        # Find the decision in history
        for past_decision in self.decisions_history:
            if past_decision.get("decision_time") == decision.get("decision_time"):
                past_decision["outcome"] = outcome
                break


#####################################################
# Self-Evaluation System
#####################################################

class SelfEvaluationSystem:
    """
    Enables the agent to evaluate its own performance and learn from experiences.
    
    This component provides:
    1. Performance tracking across different dimensions
    2. Success/failure analysis
    3. Strategy adjustment based on past experiences
    4. Metrics for ongoing improvement
    """
    
    def __init__(self):
        """Initialize the self-evaluation system"""
        self.evaluations = []
        self.dimension_scores = {
            "task_completion": [],     # How well tasks are completed
            "efficiency": [],          # Resource usage efficiency
            "accuracy": [],            # Information accuracy
            "adaptation": [],          # Ability to adapt to changing conditions
            "autonomy": []             # Level of independent operation
        }
        self.improvement_suggestions = []
        self.learning_experiences = []
        
    def evaluate_task(self, 
                     task: Dict[str, Any], 
                     result: Dict[str, Any], 
                     context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the agent's performance on a specific task.
        
        Args:
            task: The task that was attempted
            result: The result of the task execution
            context: Context information during execution
            
        Returns:
            Evaluation with scores and insights
        """
        evaluation = {
            "task_id": task.get("id", str(len(self.evaluations))),
            "task_description": task.get("description", "Unknown task"),
            "success": result.get("success", False),
            "timestamp": datetime.now().isoformat(),
            "scores": {},
            "insights": [],
            "improvement_suggestions": []
        }
        
        # Calculate scores for different dimensions
        scores = self._calculate_dimension_scores(task, result, context)
        evaluation["scores"] = scores
        
        # Update dimension score history
        for dimension, score in scores.items():
            if dimension in self.dimension_scores:
                self.dimension_scores[dimension].append(score)
                
        # Generate insights
        evaluation["insights"] = self._generate_insights(task, result, scores)
        
        # Generate improvement suggestions
        evaluation["improvement_suggestions"] = self._generate_improvement_suggestions(task, result, scores)
        
        # Add to evaluation history
        self.evaluations.append(evaluation)
        
        return evaluation
        
    def _calculate_dimension_scores(self, task: Dict[str, Any], result: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate scores for different evaluation dimensions"""
        scores = {}
        
        # Task completion score
        if "success" in result:
            scores["task_completion"] = 1.0 if result["success"] else 0.0
        elif "progress" in result:
            scores["task_completion"] = min(1.0, max(0.0, result["progress"]))
        else:
            scores["task_completion"] = 0.5  # Neutral score
            
        # Efficiency score
        if "execution_time" in result and "expected_time" in task:
            # Lower is better for execution time
            time_ratio = task["expected_time"] / max(0.001, result["execution_time"])
            scores["efficiency"] = min(1.0, max(0.0, time_ratio))
        else:
            steps_taken = len(result.get("steps", []))
            estimated_min_steps = task.get("estimated_min_steps", steps_taken)
            
            if steps_taken > 0 and estimated_min_steps > 0:
                step_ratio = estimated_min_steps / steps_taken
                scores["efficiency"] = min(1.0, max(0.0, step_ratio))
            else:
                scores["efficiency"] = 0.5  # Neutral score
                
        # Accuracy score
        if "accuracy" in result:
            scores["accuracy"] = result["accuracy"]
        else:
            # Estimate based on outcome
            scores["accuracy"] = 0.8 if result.get("success", False) else 0.4
            
        # Adaptation score
        if "changes_handled" in result:
            scores["adaptation"] = result["changes_handled"]
        else:
            # Estimate based on context changes
            initial_context = task.get("initial_context", {})
            context_changes = sum(1 for k, v in context.items() if k in initial_context and initial_context[k] != v)
            total_context = len(initial_context)
            
            if total_context > 0:
                adaptation_ratio = min(1.0, context_changes / total_context)
                scores["adaptation"] = 0.5 + (adaptation_ratio * 0.5)  # 0.5-1.0 range
            else:
                scores["adaptation"] = 0.5  # Neutral score
                
        # Autonomy score
        if "human_interventions" in result:
            interventions = result["human_interventions"]
            scores["autonomy"] = max(0.0, 1.0 - (interventions * 0.2))  # Each intervention reduces score by 0.2
        else:
            scores["autonomy"] = 0.8  # Default good autonomy
            
        return scores
        
    def _generate_insights(self, task: Dict[str, Any], result: Dict[str, Any], scores: Dict[str, float]) -> List[str]:
        """Generate insights based on task performance"""
        insights = []
        
        # Success or failure insight
        if result.get("success", False):
            insights.append(f"Successfully completed task: {task.get('description', 'Unknown')}")
        else:
            insights.append(f"Failed to complete task: {task.get('description', 'Unknown')}")
            if "error" in result:
                insights.append(f"Error encountered: {result['error']}")
                
        # Dimension-specific insights
        for dimension, score in scores.items():
            if score >= 0.8:
                insights.append(f"Strong performance in {dimension}: {score:.2f}")
            elif score <= 0.3:
                insights.append(f"Weak performance in {dimension}: {score:.2f}")
                
        # Complex task insight
        if task.get("complexity", 0) > 7:
            if result.get("success", False):
                insights.append("Successfully handled a highly complex task")
            else:
                insights.append("Struggled with a highly complex task")
                
        # Time performance insight
        if "execution_time" in result and "expected_time" in task:
            ratio = result["execution_time"] / task["expected_time"]
            if ratio < 0.8:
                insights.append(f"Completed task faster than expected: {ratio:.2f}x expected time")
            elif ratio > 1.5:
                insights.append(f"Completed task slower than expected: {ratio:.2f}x expected time")
                
        return insights
        
    def _generate_improvement_suggestions(self, task: Dict[str, Any], result: Dict[str, Any], scores: Dict[str, float]) -> List[str]:
        """Generate suggestions for improvement"""
        suggestions = []
        
        # Check for low scores and generate suggestions
        if scores.get("task_completion", 1.0) < 0.7:
            if "error" in result:
                suggestions.append(f"Improve error handling for: {result['error']}")
            suggestions.append("Break down tasks into smaller, more manageable subtasks")
            
        if scores.get("efficiency", 1.0) < 0.6:
            suggestions.append("Reduce the number of steps required to complete tasks")
            suggestions.append("Prioritize direct approaches over exploratory ones")
            
        if scores.get("accuracy", 1.0) < 0.7:
            suggestions.append("Verify information from multiple sources before making decisions")
            suggestions.append("Implement additional data validation steps")
            
        if scores.get("adaptation", 1.0) < 0.6:
            suggestions.append("Improve monitoring of environmental changes during task execution")
            suggestions.append("Develop fallback strategies for common failure scenarios")
            
        if scores.get("autonomy", 1.0) < 0.7:
            suggestions.append("Reduce dependency on human interventions for decision making")
            suggestions.append("Build confidence thresholds for autonomous action")
            
        # Task-specific suggestions
        if task.get("type") == "web_navigation" and not result.get("success", False):
            suggestions.append("Improve web page parsing and content extraction capabilities")
            
        if task.get("type") == "search" and not result.get("success", False):
            suggestions.append("Refine search query formulation for more relevant results")
            
        # Add to global improvement suggestions if not already present
        for suggestion in suggestions:
            if suggestion not in self.improvement_suggestions:
                self.improvement_suggestions.append(suggestion)
                
        return suggestions
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of overall agent performance"""
        if not self.evaluations:
            return {"message": "No evaluations available yet"}
            
        total_tasks = len(self.evaluations)
        successful_tasks = sum(1 for e in self.evaluations if e.get("success", False))
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0
        
        # Calculate average scores
        avg_scores = {}
        for dimension, scores in self.dimension_scores.items():
            if scores:
                avg_scores[dimension] = sum(scores) / len(scores)
            else:
                avg_scores[dimension] = 0.0
                
        # Calculate improvement over time (last 10 vs first 10)
        improvement = {}
        for dimension, scores in self.dimension_scores.items():
            if len(scores) >= 20:
                first_10_avg = sum(scores[:10]) / 10
                last_10_avg = sum(scores[-10:]) / 10
                improvement[dimension] = last_10_avg - first_10_avg
                
        summary = {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": success_rate,
            "average_scores": avg_scores,
            "improvement": improvement,
            "top_improvement_suggestions": self.improvement_suggestions[:5] if self.improvement_suggestions else []
        }
        
        return summary
        
    def add_learning_experience(self, experience: Dict[str, Any]) -> None:
        """
        Record a specific learning experience.
        
        Args:
            experience: Information about what was learned
        """
        experience["timestamp"] = datetime.now().isoformat()
        self.learning_experiences.append(experience)
        
    def get_relevant_learnings(self, task_description: str) -> List[Dict[str, Any]]:
        """
        Get relevant learning experiences for a specific task.
        
        Args:
            task_description: Description of the current task
            
        Returns:
            List of relevant learning experiences
        """
        if not self.learning_experiences:
            return []
            
        # Simple keyword matching - in a real system this would use semantic similarity
        relevant_experiences = []
        keywords = set(task_description.lower().split())
        
        for experience in self.learning_experiences:
            exp_keywords = set(experience.get("description", "").lower().split())
            if keywords.intersection(exp_keywords):
                relevant_experiences.append(experience)
                
        return relevant_experiences[:5]  # Return top 5