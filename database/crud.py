from sqlalchemy.orm import Session
from . import models
from datetime import datetime
import json

def create_task(db: Session, description: str):
    db_task = models.Task(
        description=description,
        status="pending",
        created_at=datetime.utcnow()
    )
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    return db_task

def update_task_result(db: Session, task_id: int, result: dict, execution_time: float):
    task = db.query(models.Task).filter(models.Task.id == task_id).first()
    if task:
        task.result = result
        task.status = "completed"
        task.execution_time = execution_time
        db.commit()
        return task
    return None

def create_quantum_metrics(db: Session, task_id: int, metrics: dict):
    db_metrics = models.QuantumMetrics(
        task_id=task_id,
        quantum_advantage=metrics.get("quantum_advantage", 0),
        memory_efficiency=metrics.get("memory_efficiency", 0),
        circuit_depth=metrics.get("circuit_depth", 0),
        qubit_count=metrics.get("qubit_count", 0)
    )
    db.add(db_metrics)
    db.commit()
    db.refresh(db_metrics)
    return db_metrics

def get_task_history(db: Session, limit: int = 10):
    return (
        db.query(models.Task)
        .order_by(models.Task.created_at.desc())
        .limit(limit)
        .all()
    )

def get_quantum_metrics_for_task(db: Session, task_id: int):
    return (
        db.query(models.QuantumMetrics)
        .filter(models.QuantumMetrics.task_id == task_id)
        .first()
    )
