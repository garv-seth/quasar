"""QUASAR Task Execution Engine for managing complex Q3A workflows"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional, Callable, Union, Set
from dataclasses import dataclass, field
import os

from openai import AsyncOpenAI

from quantum_agent_framework.agent_core.q3a_agent import Q3AAgent


@dataclass
class Task:
    """Representation of a task in the QUASAR framework."""
    id: str
    description: str
    status: str = "pending"  # pending, in_progress, completed, failed
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    assigned_agent: Optional[str] = None
    priority: int = 1  # 1-5, with 5 being highest
    dependencies: Set[str] = field(default_factory=set)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def mark_started(self):
        """Mark the task as started."""
        self.status = "in_progress"
        self.started_at = time.time()
    
    def mark_completed(self, result: Dict[str, Any]):
        """Mark the task as completed."""
        self.status = "completed"
        self.completed_at = time.time()
        self.result = result
    
    def mark_failed(self, error: str):
        """Mark the task as failed."""
        self.status = "failed"
        self.completed_at = time.time()
        self.result = {"error": error}
    
    @property
    def execution_time(self) -> Optional[float]:
        """Get the execution time if completed or failed."""
        if self.completed_at and self.started_at:
            return self.completed_at - self.started_at
        return None
    
    @property
    def is_ready(self) -> bool:
        """Check if the task is ready to be executed (all dependencies completed)."""
        return self.status == "pending" and not self.dependencies
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization."""
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "execution_time": self.execution_time,
            "assigned_agent": self.assigned_agent,
            "priority": self.priority,
            "dependencies": list(self.dependencies),
            "tags": self.tags,
            "metadata": self.metadata,
            "result": self.result
        }


class QUASARTaskEngine:
    """Task execution engine for managing and executing complex workflows."""
    
    def __init__(self, n_qubits: int = 8, use_quantum: bool = True, max_concurrent_tasks: int = 5):
        """Initialize the task execution engine."""
        self.tasks = {}  # id -> Task
        self.agents = {}  # id -> Q3AAgent
        self.max_concurrent_tasks = max_concurrent_tasks
        self.n_qubits = n_qubits
        self.use_quantum = use_quantum
        
        # OpenAI client for task planning
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Create primary agent
        self.create_agent("primary", n_qubits=n_qubits, use_quantum=use_quantum)
        
        # Engine metrics
        self.metrics = {
            "tasks_created": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "avg_execution_time": 0,
            "total_execution_time": 0,
        }
        
        logging.info(f"QUASAR Task Engine initialized with {n_qubits} qubits, quantum {'enabled' if use_quantum else 'disabled'}")
    
    def create_agent(self, agent_id: str, n_qubits: int = None, use_quantum: bool = None) -> str:
        """Create a new agent in the system."""
        if agent_id in self.agents:
            logging.warning(f"Agent {agent_id} already exists, returning existing agent")
            return agent_id
        
        if n_qubits is None:
            n_qubits = self.n_qubits
            
        if use_quantum is None:
            use_quantum = self.use_quantum
            
        # Create new agent
        self.agents[agent_id] = Q3AAgent(n_qubits=n_qubits, use_quantum=use_quantum)
        
        logging.info(f"Created agent {agent_id} with {n_qubits} qubits, quantum {'enabled' if use_quantum else 'disabled'}")
        return agent_id
    
    def create_task(self, 
                   description: str, 
                   priority: int = 1, 
                   dependencies: List[str] = None,
                   tags: List[str] = None,
                   metadata: Dict[str, Any] = None) -> str:
        """Create a new task in the system."""
        task_id = f"task_{int(time.time())}_{len(self.tasks)}"
        
        # Setup dependencies
        task_dependencies = set()
        if dependencies:
            for dep_id in dependencies:
                if dep_id in self.tasks:
                    task_dependencies.add(dep_id)
                else:
                    logging.warning(f"Dependency {dep_id} not found, ignoring")
        
        # Create task
        self.tasks[task_id] = Task(
            id=task_id,
            description=description,
            priority=priority,
            dependencies=task_dependencies,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        self.metrics["tasks_created"] += 1
        
        logging.info(f"Created task {task_id}: {description}")
        return task_id
    
    async def process_user_request(self, request: str) -> Dict[str, Any]:
        """Process a natural language request from a user."""
        # Analyze the request to determine if it's a simple task or a complex workflow
        workflow_analysis = await self._analyze_workflow_complexity(request)
        
        if workflow_analysis["is_complex_workflow"]:
            # Create a workflow of multiple coordinated tasks
            task_definitions = await self._plan_complex_workflow(request, workflow_analysis)
            
            # Create the tasks with dependencies
            task_ids = []
            for task_def in task_definitions:
                task_id = self.create_task(
                    description=task_def["description"],
                    priority=task_def.get("priority", 3),
                    dependencies=task_def.get("dependencies", []),
                    tags=task_def.get("tags", []),
                    metadata=task_def.get("metadata", {})
                )
                task_ids.append(task_id)
            
            # Start executing the workflow
            asyncio.create_task(self._execute_pending_tasks())
            
            return {
                "request_type": "complex_workflow",
                "workflow_id": f"workflow_{int(time.time())}",
                "task_count": len(task_ids),
                "tasks": task_ids,
                "message": f"Created a workflow with {len(task_ids)} tasks based on your request. The tasks will be executed in the appropriate order."
            }
        else:
            # Handle as a single task
            primary_agent = self.agents.get("primary")
            if not primary_agent:
                primary_agent = Q3AAgent(n_qubits=self.n_qubits, use_quantum=self.use_quantum)
                self.agents["primary"] = primary_agent
            
            # Process with the primary agent
            result = await primary_agent.process_user_instruction(request)
            
            # Create a task record for tracking
            task_id = self.create_task(
                description=request,
                priority=2,
                tags=["user_request"]
            )
            
            task = self.tasks[task_id]
            task.mark_started()
            task.mark_completed(result)
            
            # Update metrics
            self.metrics["tasks_completed"] += 1
            self.metrics["total_execution_time"] += task.execution_time or 0
            if self.metrics["tasks_completed"] > 0:
                self.metrics["avg_execution_time"] = self.metrics["total_execution_time"] / self.metrics["tasks_completed"]
            
            return {
                "request_type": "single_task",
                "task_id": task_id,
                "response": result["response"],
                "action": result["action"]
            }
    
    async def _analyze_workflow_complexity(self, request: str) -> Dict[str, Any]:
        """Analyze a user request to determine if it requires a complex workflow."""
        prompt = f"""
        User request: "{request}"
        
        Analyze this request and determine if it should be handled as:
        1. A simple task (one operation with one agent)
        2. A complex workflow (multiple coordinated tasks potentially using multiple agents)
        
        Consider these factors:
        - Does it involve multiple distinct operations?
        - Does it require coordinating different types of data processing?
        - Would it benefit from parallel processing of sub-tasks?
        - Does it have dependencies between different operations?
        
        Format your response as valid JSON with the following structure:
        {{
            "is_complex_workflow": true/false,
            "reasoning": "Brief explanation of why this is or isn't a complex workflow",
            "estimated_task_count": number,
            "suggested_approach": "Brief description of how to approach this request"
        }}
        """
        
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        analysis = json.loads(response.choices[0].message.content)
        return analysis
    
    async def _plan_complex_workflow(self, request: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan a complex workflow of multiple coordinated tasks."""
        prompt = f"""
        User request: "{request}"
        
        This request has been identified as a complex workflow with approximately {analysis["estimated_task_count"]} tasks.
        
        Create a detailed workflow plan with individual tasks that together satisfy the user's request.
        For each task, include:
        1. A clear description of what the task should accomplish
        2. Priority level (1-5, with 5 being highest)
        3. Dependencies (list of task indices that must complete before this task)
        4. Tags for categorization
        5. Any other metadata useful for task execution
        
        Format your response as valid JSON with the following structure:
        [
            {{
                "task_index": 0,
                "description": "Detailed task description",
                "priority": 1-5,
                "dependencies": [], 
                "tags": ["tag1", "tag2"],
                "metadata": {{ additional useful information }}
            }},
            ...
        ]
        
        Ensure the tasks are ordered logically, with dependencies correctly specified.
        Tasks that depend on outputs from other tasks should list those tasks in their dependencies.
        """
        
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        tasks = json.loads(response.choices[0].message.content)
        
        # Convert task index references to actual task IDs that will be created
        task_id_mapping = {}
        for task in tasks:
            task_index = task["task_index"]
            task_id = f"task_{int(time.time())}_{task_index}"
            task_id_mapping[task_index] = task_id
        
        # Update dependencies to use task IDs instead of indices
        for task in tasks:
            if "dependencies" in task:
                task["dependencies"] = [
                    task_id_mapping[dep_index] 
                    for dep_index in task["dependencies"] 
                    if dep_index in task_id_mapping
                ]
        
        return tasks
    
    async def _execute_pending_tasks(self):
        """Execute all pending tasks that have no dependencies."""
        # Get all ready tasks
        ready_tasks = [
            task for task in self.tasks.values() 
            if task.is_ready and task.status == "pending"
        ]
        
        # Sort by priority (highest first)
        ready_tasks.sort(key=lambda t: t.priority, reverse=True)
        
        # Only execute up to max_concurrent_tasks
        tasks_to_execute = ready_tasks[:self.max_concurrent_tasks]
        
        # Execute each task
        execution_futures = []
        for task in tasks_to_execute:
            execution_futures.append(self._execute_task(task))
        
        # Wait for all to complete
        if execution_futures:
            await asyncio.gather(*execution_futures)
            
            # Check if there are more tasks to execute
            if any(task.status == "pending" for task in self.tasks.values()):
                # Continue executing tasks after a short delay
                await asyncio.sleep(0.1)
                asyncio.create_task(self._execute_pending_tasks())
    
    async def _execute_task(self, task: Task):
        """Execute a single task using an appropriate agent."""
        task.mark_started()
        logging.info(f"Executing task {task.id}: {task.description}")
        
        # Select an agent
        agent_id = task.assigned_agent or "primary"
        if agent_id not in self.agents:
            agent_id = "primary"
        
        agent = self.agents[agent_id]
        
        try:
            # Execute task with the agent
            result = await agent.process_user_instruction(task.description)
            
            # Update task
            task.mark_completed(result)
            
            # Update dependencies
            self._update_dependencies(task.id)
            
            # Update metrics
            self.metrics["tasks_completed"] += 1
            self.metrics["total_execution_time"] += task.execution_time or 0
            if self.metrics["tasks_completed"] > 0:
                self.metrics["avg_execution_time"] = self.metrics["total_execution_time"] / self.metrics["tasks_completed"]
            
            logging.info(f"Task {task.id} completed successfully in {task.execution_time:.2f}s")
            return result
            
        except Exception as e:
            logging.error(f"Error executing task {task.id}: {str(e)}")
            task.mark_failed(str(e))
            
            # Update dependencies (task failed, but dependencies should be updated)
            self._update_dependencies(task.id)
            
            # Update metrics
            self.metrics["tasks_failed"] += 1
            
            return {"error": str(e)}
    
    def _update_dependencies(self, completed_task_id: str):
        """Update dependencies for all tasks that depend on the completed task."""
        for task in self.tasks.values():
            if completed_task_id in task.dependencies:
                task.dependencies.remove(completed_task_id)
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get a task by ID."""
        task = self.tasks.get(task_id)
        if task:
            return task.to_dict()
        return None
    
    def get_all_tasks(self, status: Optional[str] = None, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get all tasks, optionally filtered by status and/or tags."""
        filtered_tasks = self.tasks.values()
        
        if status:
            filtered_tasks = [task for task in filtered_tasks if task.status == status]
        
        if tags:
            filtered_tasks = [
                task for task in filtered_tasks 
                if any(tag in task.tags for tag in tags)
            ]
        
        return [task.to_dict() for task in filtered_tasks]
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or in-progress task."""
        task = self.tasks.get(task_id)
        if not task or task.status in ["completed", "failed"]:
            return False
        
        task.mark_failed("Task cancelled by user")
        self._update_dependencies(task_id)
        return True
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get statistics about the task engine."""
        stats = self.metrics.copy()
        
        # Add current tasks stats
        stats.update({
            "pending_tasks": len([t for t in self.tasks.values() if t.status == "pending"]),
            "in_progress_tasks": len([t for t in self.tasks.values() if t.status == "in_progress"]),
            "completed_tasks": len([t for t in self.tasks.values() if t.status == "completed"]),
            "failed_tasks": len([t for t in self.tasks.values() if t.status == "failed"]),
            "total_tasks": len(self.tasks),
            "agent_count": len(self.agents)
        })
        
        # Add agent stats
        if "primary" in self.agents:
            primary_metrics = self.agents["primary"].get_performance_metrics()
            stats.update({
                "primary_agent_" + k: v for k, v in primary_metrics.items()
            })
        
        return stats