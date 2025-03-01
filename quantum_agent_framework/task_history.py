"""
Task History Management for QA³ (Quantum-Accelerated AI Agent)

This module provides task history tracking and management capabilities,
enabling the agent to remember past tasks, their outcomes, and related metrics.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("task-history")

class TaskHistoryManager:
    """
    Manages task history and related metrics for the QA³ agent
    """
    
    def __init__(self, storage_path: str = "./data/task_history", max_history: int = 100):
        """
        Initialize task history manager
        
        Args:
            storage_path: Path to store task history
            max_history: Maximum number of tasks to keep in memory
        """
        self.storage_path = storage_path
        self.max_history = max_history
        self.tasks = []
        self.metrics = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "search_tasks": 0,
            "navigation_tasks": 0,
            "interaction_tasks": 0,
            "general_tasks": 0,
            "quantum_enhanced_tasks": 0,
            "classical_tasks": 0,
            "average_execution_time": 0,
            "quantum_speedup_factor": 1.0
        }
        
        # Ensure storage directory exists
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        
        # Initialize from storage if available
        self._load_history()
        
        logger.info(f"Task history manager initialized with {len(self.tasks)} tasks")
    
    def _load_history(self) -> None:
        """Load task history from storage"""
        try:
            # Load task history
            history_file = os.path.join(self.storage_path, "tasks.json")
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    self.tasks = json.load(f)
                logger.info(f"Loaded {len(self.tasks)} tasks from history")
            
            # Load metrics
            metrics_file = os.path.join(self.storage_path, "metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    self.metrics = json.load(f)
                logger.info(f"Loaded metrics from storage")
        except Exception as e:
            logger.error(f"Error loading task history: {str(e)}")
    
    def _save_history(self) -> None:
        """Save task history to storage"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.storage_path, exist_ok=True)
            
            # Save task history
            history_file = os.path.join(self.storage_path, "tasks.json")
            with open(history_file, 'w') as f:
                json.dump(self.tasks, f, indent=2)
            
            # Save metrics
            metrics_file = os.path.join(self.storage_path, "metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            logger.info(f"Saved {len(self.tasks)} tasks to history")
        except Exception as e:
            logger.error(f"Error saving task history: {str(e)}")
    
    def add_task(self, task: Dict[str, Any]) -> str:
        """
        Add a task to history
        
        Args:
            task: Task data to add
            
        Returns:
            Task ID
        """
        # Generate task ID if not provided
        if "id" not in task:
            task["id"] = f"task_{int(time.time())}_{len(self.tasks)}"
        
        # Ensure timestamp is present
        if "timestamp" not in task:
            task["timestamp"] = datetime.now().isoformat()
        
        # Add task to history
        self.tasks.append(task)
        
        # Limit history size
        if len(self.tasks) > self.max_history:
            self.tasks = self.tasks[-self.max_history:]
        
        # Update metrics
        self._update_metrics_with_task(task)
        
        # Save to storage
        self._save_history()
        
        return task["id"]
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a task by ID
        
        Args:
            task_id: Task ID
            
        Returns:
            Task data if found, None otherwise
        """
        for task in self.tasks:
            if task.get("id") == task_id:
                return task
        return None
    
    def update_task(self, task_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update a task by ID
        
        Args:
            task_id: Task ID
            update_data: Data to update
            
        Returns:
            Success status
        """
        for i, task in enumerate(self.tasks):
            if task.get("id") == task_id:
                # Apply updates
                self.tasks[i].update(update_data)
                
                # Update metrics if status changed
                if "status" in update_data:
                    self._update_metrics()
                
                # Save to storage
                self._save_history()
                
                return True
        return False
    
    def get_tasks(self, limit: int = 10, offset: int = 0, 
                 task_type: Optional[str] = None,
                 status: Optional[str] = None,
                 sort_by: str = "timestamp",
                 sort_order: str = "desc") -> List[Dict[str, Any]]:
        """
        Get tasks with filtering and sorting
        
        Args:
            limit: Maximum number of tasks to return
            offset: Offset for pagination
            task_type: Filter by task type
            status: Filter by status
            sort_by: Field to sort by
            sort_order: Sort order ('asc' or 'desc')
            
        Returns:
            List of tasks
        """
        # Apply filters
        filtered_tasks = self.tasks
        
        if task_type:
            filtered_tasks = [t for t in filtered_tasks if t.get("task_type") == task_type]
        
        if status:
            filtered_tasks = [t for t in filtered_tasks if t.get("status") == status]
        
        # Apply sorting
        reverse = sort_order.lower() == "desc"
        sorted_tasks = sorted(filtered_tasks, 
                             key=lambda t: t.get(sort_by, ""),
                             reverse=reverse)
        
        # Apply pagination
        paginated_tasks = sorted_tasks[offset:offset + limit]
        
        return paginated_tasks
    
    def _update_metrics_with_task(self, task: Dict[str, Any]) -> None:
        """Update metrics based on a new task"""
        # Increment total tasks
        self.metrics["total_tasks"] += 1
        
        # Update task type counters
        task_type = task.get("task_type", "general")
        if task_type == "search":
            self.metrics["search_tasks"] += 1
        elif task_type == "navigation":
            self.metrics["navigation_tasks"] += 1
        elif task_type == "interaction":
            self.metrics["interaction_tasks"] += 1
        else:
            self.metrics["general_tasks"] += 1
        
        # Update success/failure counters
        status = task.get("status", "unknown")
        if status == "success":
            self.metrics["successful_tasks"] += 1
        elif status == "failure":
            self.metrics["failed_tasks"] += 1
        
        # Update quantum vs classical counters
        if task.get("quantum_enhanced", False):
            self.metrics["quantum_enhanced_tasks"] += 1
        else:
            self.metrics["classical_tasks"] += 1
        
        # Update execution time metrics
        if "execution_time" in task:
            current_avg = self.metrics["average_execution_time"]
            current_total = self.metrics["total_tasks"] - 1  # Exclude current task
            
            if current_total > 0:
                new_avg = (current_avg * current_total + task["execution_time"]) / self.metrics["total_tasks"]
                self.metrics["average_execution_time"] = new_avg
            else:
                self.metrics["average_execution_time"] = task["execution_time"]
        
        # Update quantum speedup metrics
        if task.get("quantum_enhanced", False) and task.get("classical_comparison", False):
            quantum_time = task.get("execution_time", 0)
            classical_time = task.get("classical_time", 0)
            
            if quantum_time > 0 and classical_time > 0:
                speedup = classical_time / quantum_time
                
                # Update rolling average of speedup
                current_speedup = self.metrics["quantum_speedup_factor"]
                quantum_tasks = self.metrics["quantum_enhanced_tasks"]
                
                if quantum_tasks > 1:  # More than one quantum task including this one
                    new_speedup = (current_speedup * (quantum_tasks - 1) + speedup) / quantum_tasks
                    self.metrics["quantum_speedup_factor"] = new_speedup
                else:
                    self.metrics["quantum_speedup_factor"] = speedup
    
    def _update_metrics(self) -> None:
        """Recalculate all metrics from task history"""
        # Reset metrics
        self.metrics = {
            "total_tasks": len(self.tasks),
            "successful_tasks": 0,
            "failed_tasks": 0,
            "search_tasks": 0,
            "navigation_tasks": 0,
            "interaction_tasks": 0,
            "general_tasks": 0,
            "quantum_enhanced_tasks": 0,
            "classical_tasks": 0,
            "average_execution_time": 0,
            "quantum_speedup_factor": 1.0
        }
        
        # Calculate metrics from tasks
        total_execution_time = 0
        total_speedup = 0
        speedup_count = 0
        
        for task in self.tasks:
            # Update task type counters
            task_type = task.get("task_type", "general")
            if task_type == "search":
                self.metrics["search_tasks"] += 1
            elif task_type == "navigation":
                self.metrics["navigation_tasks"] += 1
            elif task_type == "interaction":
                self.metrics["interaction_tasks"] += 1
            else:
                self.metrics["general_tasks"] += 1
            
            # Update success/failure counters
            status = task.get("status", "unknown")
            if status == "success":
                self.metrics["successful_tasks"] += 1
            elif status == "failure":
                self.metrics["failed_tasks"] += 1
            
            # Update quantum vs classical counters
            if task.get("quantum_enhanced", False):
                self.metrics["quantum_enhanced_tasks"] += 1
            else:
                self.metrics["classical_tasks"] += 1
            
            # Track execution time
            if "execution_time" in task:
                total_execution_time += task["execution_time"]
            
            # Track quantum speedup
            if task.get("quantum_enhanced", False) and task.get("classical_comparison", False):
                quantum_time = task.get("execution_time", 0)
                classical_time = task.get("classical_time", 0)
                
                if quantum_time > 0 and classical_time > 0:
                    speedup = classical_time / quantum_time
                    total_speedup += speedup
                    speedup_count += 1
        
        # Calculate averages
        if self.metrics["total_tasks"] > 0:
            self.metrics["average_execution_time"] = total_execution_time / self.metrics["total_tasks"]
        
        if speedup_count > 0:
            self.metrics["quantum_speedup_factor"] = total_speedup / speedup_count
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics
        
        Returns:
            Dict with metrics
        """
        return self.metrics
    
    def get_metrics_summary(self) -> str:
        """
        Get a human-readable summary of metrics
        
        Returns:
            Metrics summary string
        """
        m = self.metrics
        
        summary = []
        summary.append(f"Total tasks: {m['total_tasks']}")
        
        if m['total_tasks'] > 0:
            success_rate = (m['successful_tasks'] / m['total_tasks']) * 100
            summary.append(f"Success rate: {success_rate:.1f}%")
        
        summary.append(f"Task types: {m['search_tasks']} search, {m['navigation_tasks']} navigation, {m['interaction_tasks']} interaction, {m['general_tasks']} general")
        
        quantum_percentage = 0
        if m['total_tasks'] > 0:
            quantum_percentage = (m['quantum_enhanced_tasks'] / m['total_tasks']) * 100
        summary.append(f"Quantum-enhanced tasks: {m['quantum_enhanced_tasks']} ({quantum_percentage:.1f}%)")
        
        summary.append(f"Average execution time: {m['average_execution_time']:.3f} seconds")
        
        if m['quantum_enhanced_tasks'] > 0:
            summary.append(f"Average quantum speedup: {m['quantum_speedup_factor']:.2f}x")
        
        return "\n".join(summary)
    
    def clear_history(self) -> None:
        """Clear all task history"""
        self.tasks = []
        self._update_metrics()
        self._save_history()
        
        logger.info("Task history cleared")
    
    def export_history(self, format: str = "json") -> str:
        """
        Export task history in specified format
        
        Args:
            format: Export format ('json' or 'csv')
            
        Returns:
            Exported data as string
        """
        if format.lower() == "json":
            return json.dumps({
                "tasks": self.tasks,
                "metrics": self.metrics
            }, indent=2)
        
        elif format.lower() == "csv":
            # Simple CSV export for tasks
            header = "id,timestamp,task_type,status,quantum_enhanced,execution_time\n"
            rows = []
            
            for task in self.tasks:
                row = [
                    task.get("id", ""),
                    task.get("timestamp", ""),
                    task.get("task_type", ""),
                    task.get("status", ""),
                    str(task.get("quantum_enhanced", False)),
                    str(task.get("execution_time", ""))
                ]
                rows.append(",".join(row))
            
            return header + "\n".join(rows)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def import_history(self, data: str, format: str = "json") -> bool:
        """
        Import task history from string
        
        Args:
            data: Data to import
            format: Import format ('json' or 'csv')
            
        Returns:
            Success status
        """
        try:
            if format.lower() == "json":
                imported = json.loads(data)
                
                if "tasks" in imported:
                    self.tasks = imported["tasks"]
                
                if "metrics" in imported:
                    self.metrics = imported["metrics"]
                else:
                    self._update_metrics()
                
                self._save_history()
                return True
            
            elif format.lower() == "csv":
                # Simple CSV import
                lines = data.strip().split("\n")
                
                if len(lines) < 2:  # Header only
                    return False
                
                header = lines[0].split(",")
                self.tasks = []
                
                for line in lines[1:]:
                    fields = line.split(",")
                    
                    if len(fields) < len(header):
                        continue
                    
                    task = {}
                    for i, field_name in enumerate(header):
                        if field_name == "quantum_enhanced":
                            task[field_name] = fields[i].lower() == "true"
                        elif field_name == "execution_time":
                            try:
                                task[field_name] = float(fields[i])
                            except:
                                pass
                        else:
                            task[field_name] = fields[i]
                    
                    self.tasks.append(task)
                
                self._update_metrics()
                self._save_history()
                return True
            
            else:
                logger.error(f"Unsupported import format: {format}")
                return False
                
        except Exception as e:
            logger.error(f"Error importing task history: {str(e)}")
            return False