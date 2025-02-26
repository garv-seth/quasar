"""Enhanced CRUD operations for the QUASAR framework database."""

from sqlalchemy.orm import Session
from sqlalchemy import func, desc, asc
from datetime import datetime
import logging
from typing import List, Dict, Any, Optional, Tuple

from . import models


def create_task(db: Session,
                description: str,
                task_type: str = "unknown") -> models.Task:
    """
    Create a new task record.

    Args:
        db: Database session
        description: Task description
        task_type: Type of task (factorization, optimization, etc.)

    Returns:
        models.Task: Created task
    """
    try:
        db_task = models.Task(description=description,
                              task_type=task_type,
                              status="pending",
                              created_at=datetime.utcnow())
        db.add(db_task)
        db.commit()
        db.refresh(db_task)
        return db_task
    except Exception as e:
        db.rollback()
        logging.error(f"Error creating task: {str(e)}")
        raise


def update_task_status(db: Session, task_id: int,
                       status: str) -> Optional[models.Task]:
    """
    Update task status.

    Args:
        db: Database session
        task_id: Task ID
        status: New status

    Returns:
        models.Task: Updated task or None
    """
    task = db.query(models.Task).filter(models.Task.id == task_id).first()
    if task:
        task.status = status
        if status == "processing":
            task.started_at = datetime.utcnow()
        elif status in ["completed", "failed"]:
            task.completed_at = datetime.utcnow()
        db.commit()
        db.refresh(task)
        return task
    return None


def update_task_result(
        db: Session,
        task_id: int,
        result: dict,
        execution_time: float,
        processing_method: str = "unknown") -> Optional[models.Task]:
    """
    Update task with results.

    Args:
        db: Database session
        task_id: Task ID
        result: Task result data
        execution_time: Execution time in seconds
        processing_method: Method used (quantum, classical, etc.)

    Returns:
        models.Task: Updated task or None
    """
    task = db.query(models.Task).filter(models.Task.id == task_id).first()
    if task:
        task.result = result
        task.status = "completed"
        task.execution_time = execution_time
        task.processing_method = processing_method
        task.completed_at = datetime.utcnow()
        db.commit()
        db.refresh(task)
        return task
    return None


def create_quantum_metrics(db: Session, task_id: int,
                           metrics: dict) -> models.QuantumMetrics:
    """
    Create quantum metrics for a task.

    Args:
        db: Database session
        task_id: Task ID
        metrics: Quantum metrics data

    Returns:
        models.QuantumMetrics: Created metrics
    """
    try:
        db_metrics = models.QuantumMetrics(
            task_id=task_id,
            quantum_advantage=metrics.get("quantum_advantage"),
            memory_efficiency=metrics.get("memory_efficiency"),
            circuit_depth=metrics.get("circuit_depth"),
            qubit_count=metrics.get("qubit_count"),
            backend_provider=metrics.get("backend_provider"),
            backend_name=metrics.get("backend_name"),
            theoretical_speedup=metrics.get("theoretical_speedup"),
            gates_used=metrics.get("gates_used"),
            error_rate=metrics.get("error_rate"))
        db.add(db_metrics)
        db.commit()
        db.refresh(db_metrics)
        return db_metrics
    except Exception as e:
        db.rollback()
        logging.error(f"Error creating quantum metrics: {str(e)}")
        raise


def create_classical_metrics(db: Session, task_id: int,
                             metrics: dict) -> models.ClassicalMetrics:
    """
    Create classical metrics for a task.

    Args:
        db: Database session
        task_id: Task ID
        metrics: Classical metrics data

    Returns:
        models.ClassicalMetrics: Created metrics
    """
    try:
        db_metrics = models.ClassicalMetrics(
            task_id=task_id,
            processing_time=metrics.get("processing_time"),
            memory_usage=metrics.get("memory_usage"),
            cpu_usage=metrics.get("cpu_usage"),
            algorithm_used=metrics.get("algorithm_used"),
            complexity=metrics.get("complexity"))
        db.add(db_metrics)
        db.commit()
        db.refresh(db_metrics)
        return db_metrics
    except Exception as e:
        db.rollback()
        logging.error(f"Error creating classical metrics: {str(e)}")
        raise


def create_performance_comparison(
        db: Session, task_id: int,
        comparison: dict) -> models.PerformanceComparison:
    """
    Create performance comparison for a task.

    Args:
        db: Database session
        task_id: Task ID
        comparison: Performance comparison data

    Returns:
        models.PerformanceComparison: Created comparison
    """
    try:
        db_comparison = models.PerformanceComparison(
            task_id=task_id,
            task_type=comparison.get("task_type", "unknown"),
            quantum_time=comparison.get("quantum_time"),
            classical_time=comparison.get("classical_time"),
            speedup=comparison.get("speedup"),
            quantum_method=comparison.get("quantum_method"),
            classical_method=comparison.get("classical_method"),
            dataset_size=comparison.get("dataset_size"))
        db.add(db_comparison)
        db.commit()
        db.refresh(db_comparison)
        return db_comparison
    except Exception as e:
        db.rollback()
        logging.error(f"Error creating performance comparison: {str(e)}")
        raise


def create_circuit_execution(db: Session, task_id: int,
                             execution: dict) -> models.CircuitExecution:
    """
    Record a quantum circuit execution.

    Args:
        db: Database session
        task_id: Task ID
        execution: Circuit execution data

    Returns:
        models.CircuitExecution: Created record
    """
    try:
        db_execution = models.CircuitExecution(
            task_id=task_id,
            circuit_type=execution.get("circuit_type", "unknown"),
            execution_time=execution.get("execution_time"),
            n_qubits=execution.get("n_qubits"),
            n_gates=execution.get("n_gates"),
            success=execution.get("success", True),
            error_message=execution.get("error_message"),
            params=execution.get("params"),
            results=execution.get("results"))
        db.add(db_execution)
        db.commit()
        db.refresh(db_execution)
        return db_execution
    except Exception as e:
        db.rollback()
        logging.error(f"Error creating circuit execution: {str(e)}")
        raise


def get_task(db: Session, task_id: int) -> Optional[models.Task]:
    """
    Get task by ID.

    Args:
        db: Database session
        task_id: Task ID

    Returns:
        models.Task: Task or None
    """
    return db.query(models.Task).filter(models.Task.id == task_id).first()


def get_task_history(db: Session,
                     limit: int = 10,
                     offset: int = 0,
                     task_type: Optional[str] = None,
                     processing_method: Optional[str] = None,
                     status: Optional[str] = None,
                     sort_by: str = "created_at",
                     sort_desc: bool = True) -> List[models.Task]:
    """
    Get task history with filtering and sorting.

    Args:
        db: Database session
        limit: Maximum number of results
        offset: Offset for pagination
        task_type: Filter by task type
        processing_method: Filter by processing method
        status: Filter by status
        sort_by: Field to sort by
        sort_desc: Sort in descending order if true

    Returns:
        List[models.Task]: List of tasks
    """
    query = db.query(models.Task)

    # Apply filters
    if task_type:
        query = query.filter(models.Task.task_type == task_type)
    if processing_method:
        query = query.filter(
            models.Task.processing_method == processing_method)
    if status:
        query = query.filter(models.Task.status == status)

    # Apply sorting
    if hasattr(models.Task, sort_by):
        sort_field = getattr(models.Task, sort_by)
        if sort_desc:
            query = query.order_by(desc(sort_field))
        else:
            query = query.order_by(asc(sort_field))
    else:
        # Default sort by created_at
        query = query.order_by(desc(models.Task.created_at))

    # Apply pagination
    return query.offset(offset).limit(limit).all()


def get_performance_metrics(db: Session,
                            task_type: Optional[str] = None,
                            days: int = 30) -> Dict[str, Any]:
    """
    Get performance metrics comparing quantum vs classical.

    Args:
        db: Database session
        task_type: Filter by task type
        days: Number of days to include

    Returns:
        Dict with performance metrics
    """
    try:
        # Calculate date cutoff
        cutoff_date = datetime.utcnow().date()

        # Base query for performance comparisons
        query = db.query(models.PerformanceComparison)

        if task_type:
            query = query.filter(
                models.PerformanceComparison.task_type == task_type)

        # Get all comparisons within time period
        comparisons = query.filter(
            func.date(models.PerformanceComparison.created_at) >=
            cutoff_date).all()

        # Aggregate metrics
        total_comparisons = len(comparisons)
        if total_comparisons == 0:
            return {
                "task_type": task_type,
                "total_comparisons": 0,
                "days": days,
                "average_speedup": 0,
                "max_speedup": 0,
                "quantum_wins": 0,
                "quantum_win_percentage": 0
            }

        # Calculate metrics
        speedups = [c.speedup for c in comparisons if c.speedup is not None]
        average_speedup = sum(speedups) / len(speedups) if speedups else 0
        max_speedup = max(speedups) if speedups else 0

        # Count wins (speedup > 1 means quantum is faster)
        quantum_wins = sum(1 for s in speedups if s > 1)
        quantum_win_percentage = (quantum_wins /
                                  len(speedups)) * 100 if speedups else 0

        return {
            "task_type": task_type,
            "total_comparisons": total_comparisons,
            "days": days,
            "average_speedup": average_speedup,
            "max_speedup": max_speedup,
            "quantum_wins": quantum_wins,
            "quantum_win_percentage": quantum_win_percentage
        }
    except Exception as e:
        logging.error(f"Error getting performance metrics: {str(e)}")
        return {
            "error": str(e),
            "task_type": task_type,
            "total_comparisons": 0
        }


def get_quantum_statistics(db: Session) -> Dict[str, Any]:
    """
    Get statistics about quantum processing.

    Args:
        db: Database session

    Returns:
        Dict with statistics
    """
    try:
        # Get quantum metrics
        quantum_metrics = db.query(models.QuantumMetrics).all()

        if not quantum_metrics:
            return {
                "total_quantum_tasks": 0,
                "average_qubit_count": 0,
                "average_circuit_depth": 0,
                "average_quantum_advantage": 0,
                "provider_distribution": {}
            }

        # Calculate statistics
        total_quantum_tasks = len(quantum_metrics)

        avg_qubit_count = sum(qm.qubit_count for qm in quantum_metrics
                              if qm.qubit_count) / total_quantum_tasks
        avg_circuit_depth = sum(qm.circuit_depth for qm in quantum_metrics
                                if qm.circuit_depth) / total_quantum_tasks
        avg_quantum_advantage = sum(
            qm.quantum_advantage for qm in quantum_metrics
            if qm.quantum_advantage) / total_quantum_tasks

        # Count provider distribution
        provider_count = {}
        for qm in quantum_metrics:
            provider = qm.backend_provider or "Unknown"
            provider_count[provider] = provider_count.get(provider, 0) + 1

        return {
            "total_quantum_tasks": total_quantum_tasks,
            "average_qubit_count": avg_qubit_count,
            "average_circuit_depth": avg_circuit_depth,
            "average_quantum_advantage": avg_quantum_advantage,
            "provider_distribution": provider_count
        }
    except Exception as e:
        logging.error(f"Error getting quantum statistics: {str(e)}")
        return {"error": str(e), "total_quantum_tasks": 0}


def delete_task(db: Session, task_id: int) -> bool:
    """
    Delete a task and all related data.

    Args:
        db: Database session
        task_id: Task ID

    Returns:
        bool: Success
    """
    try:
        task = db.query(models.Task).filter(models.Task.id == task_id).first()
        if task:
            db.delete(task)
            db.commit()
            return True
        return False
    except Exception as e:
        db.rollback()
        logging.error(f"Error deleting task: {str(e)}")
        return False


def get_task_with_metrics(db: Session,
                          task_id: int) -> Optional[Dict[str, Any]]:
    """
    Get task with all related metrics.

    Args:
        db: Database session
        task_id: Task ID

    Returns:
        Dict with task and metrics
    """
    task = db.query(models.Task).filter(models.Task.id == task_id).first()
    if not task:
        return None

    # Get quantum metrics
    quantum_metrics = db.query(models.QuantumMetrics).filter(
        models.QuantumMetrics.task_id == task_id).first()

    # Get classical metrics
    classical_metrics = db.query(models.ClassicalMetrics).filter(
        models.ClassicalMetrics.task_id == task_id).first()

    # Get performance comparison
    comparison = db.query(models.PerformanceComparison).filter(
        models.PerformanceComparison.task_id == task_id).first()

    # Get circuit executions
    circuit_executions = db.query(models.CircuitExecution).filter(
        models.CircuitExecution.task_id == task_id).all()

    return {
        "task":
        task.to_dict(),
        "quantum_metrics":
        quantum_metrics.to_dict() if quantum_metrics else None,
        "classical_metrics":
        classical_metrics.to_dict() if classical_metrics else None,
        "performance_comparison": {
            "speedup": comparison.speedup,
            "quantum_time": comparison.quantum_time,
            "classical_time": comparison.classical_time,
            "quantum_method": comparison.quantum_method,
            "classical_method": comparison.classical_method,
        } if comparison else None,
        "circuit_executions": [{
            "circuit_type": ce.circuit_type,
            "execution_time": ce.execution_time,
            "n_qubits": ce.n_qubits,
            "n_gates": ce.n_gates,
            "success": ce.success
        } for ce in circuit_executions] if circuit_executions else []
    }
