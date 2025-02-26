"""Enhanced database models for the QUASAR framework."""

import datetime
from sqlalchemy import (create_engine, Column, Integer, String, Float,
                        DateTime, JSON, ForeignKey, Boolean, Text, Table,
                        BigInteger, Enum, func)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
import os
import logging
import enum
from typing import List, Optional, Dict, Any

# Define connection string - use SQLite by default for local development
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./quasar.db')

# Create engine with appropriate settings
if DATABASE_URL.startswith('sqlite'):
    engine = create_engine(DATABASE_URL,
                           connect_args={"check_same_thread": False},
                           echo=False)
else:
    # For PostgreSQL or other database engines
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Define task processing methods as enum
class ProcessingMethod(enum.Enum):
    """Enumeration of processing methods."""
    CLASSICAL = "classical"
    QUANTUM = "quantum"
    QUANTUM_HYBRID = "quantum_hybrid"
    CLASSICAL_FALLBACK = "classical_fallback"
    ERROR = "error"


# Define task types as enum
class TaskType(enum.Enum):
    """Enumeration of task types."""
    FACTORIZATION = "factorization"
    OPTIMIZATION = "optimization"
    SEARCH = "search"
    QUANTUM_SIMULATION = "quantum_simulation"
    GENERAL = "general"
    UNKNOWN = "unknown"


class Task(Base):
    """Represents a user task processed by the QUASAR framework."""
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    description = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    task_type = Column(String, default="unknown")
    status = Column(
        String,
        default="pending")  # 'pending', 'processing', 'completed', 'failed'
    result = Column(JSON, nullable=True)
    execution_time = Column(Float, nullable=True)  # in seconds
    processing_method = Column(String, default="unknown")
    user_id = Column(String, nullable=True)  # For future multi-user support
    input_parameters = Column(JSON, nullable=True)

    # Relationships
    quantum_metrics = relationship("QuantumMetrics",
                                   back_populates="task",
                                   cascade="all, delete-orphan")
    classical_metrics = relationship("ClassicalMetrics",
                                     back_populates="task",
                                     cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation."""
        return {
            "id":
            self.id,
            "description":
            self.description,
            "created_at":
            self.created_at.isoformat() if self.created_at else None,
            "completed_at":
            self.completed_at.isoformat() if self.completed_at else None,
            "task_type":
            self.task_type,
            "status":
            self.status,
            "result":
            self.result,
            "execution_time":
            self.execution_time,
            "processing_method":
            self.processing_method,
            "quantum_metrics":
            [qm.to_dict()
             for qm in self.quantum_metrics] if self.quantum_metrics else None,
            "classical_metrics":
            [cm.to_dict() for cm in self.classical_metrics]
            if self.classical_metrics else None
        }


class QuantumMetrics(Base):
    """Stores quantum processing metrics for a task."""
    __tablename__ = "quantum_metrics"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"))
    quantum_advantage = Column(Float, nullable=True)  # percentage improvement
    memory_efficiency = Column(Float, nullable=True)  # percentage improvement
    circuit_depth = Column(Integer, nullable=True)
    qubit_count = Column(Integer, nullable=True)
    backend_provider = Column(String, nullable=True)  # 'Azure', 'IBM', etc.
    backend_name = Column(String,
                          nullable=True)  # 'IonQ Aria-1', 'Qiskit Aer', etc.
    theoretical_speedup = Column(
        String, nullable=True)  # 'Exponential', 'Quadratic', etc.
    gates_used = Column(Integer, nullable=True)
    error_rate = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationship
    task = relationship("Task", back_populates="quantum_metrics")

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary representation."""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "quantum_advantage": self.quantum_advantage,
            "memory_efficiency": self.memory_efficiency,
            "circuit_depth": self.circuit_depth,
            "qubit_count": self.qubit_count,
            "backend_provider": self.backend_provider,
            "backend_name": self.backend_name,
            "theoretical_speedup": self.theoretical_speedup,
            "gates_used": self.gates_used,
            "error_rate": self.error_rate,
            "created_at":
            self.created_at.isoformat() if self.created_at else None
        }


class ClassicalMetrics(Base):
    """Stores classical processing metrics for comparison."""
    __tablename__ = "classical_metrics"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"))
    processing_time = Column(Float, nullable=True)  # in seconds
    memory_usage = Column(Float, nullable=True)  # in MB
    cpu_usage = Column(Float, nullable=True)  # percentage
    algorithm_used = Column(String, nullable=True)
    complexity = Column(String, nullable=True)  # Big O notation
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationship
    task = relationship("Task", back_populates="classical_metrics")

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary representation."""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "processing_time": self.processing_time,
            "memory_usage": self.memory_usage,
            "cpu_usage": self.cpu_usage,
            "algorithm_used": self.algorithm_used,
            "complexity": self.complexity,
            "created_at":
            self.created_at.isoformat() if self.created_at else None
        }


class CircuitExecution(Base):
    """Stores information about quantum circuit executions."""
    __tablename__ = "circuit_executions"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"))
    circuit_type = Column(String,
                          nullable=False)  # 'factorization', 'search', etc.
    execution_time = Column(Float, nullable=True)  # in seconds
    n_qubits = Column(Integer, nullable=True)
    n_gates = Column(Integer, nullable=True)
    success = Column(Boolean, default=True)
    error_message = Column(String, nullable=True)
    params = Column(JSON, nullable=True)  # Circuit parameters
    results = Column(JSON, nullable=True)  # Measurement results
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationship
    task = relationship("Task")


class PerformanceComparison(Base):
    """Stores performance comparisons between quantum and classical methods."""
    __tablename__ = "performance_comparisons"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"))
    task_type = Column(String, nullable=False)
    quantum_time = Column(Float, nullable=True)
    classical_time = Column(Float, nullable=True)
    speedup = Column(Float, nullable=True)  # classical_time / quantum_time
    quantum_method = Column(String, nullable=True)
    classical_method = Column(String, nullable=True)
    dataset_size = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationship
    task = relationship("Task")


# Create tables
Base.metadata.create_all(bind=engine)


# Database dependency
def get_db():
    """Create database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Helper function to initialize database with default data
def initialize_database():
    """Initialize the database with default data if it's empty."""
    db = SessionLocal()
    try:
        # Check if the tasks table is empty
        task_count = db.query(func.count(Task.id)).scalar()

        if task_count == 0:
            logging.info("Initializing database with default data")

            # Create a sample task for demonstration
            sample_task = Task(description="Sample factorization task",
                               task_type="factorization",
                               status="completed",
                               execution_time=0.123,
                               processing_method="quantum",
                               result={
                                   "factors": [1, 2, 4, 8, 16],
                                   "number": 16,
                                   "success": True
                               },
                               created_at=datetime.datetime.utcnow(),
                               completed_at=datetime.datetime.utcnow())
            db.add(sample_task)
            db.commit()

            # Add quantum metrics
            quantum_metrics = QuantumMetrics(task_id=sample_task.id,
                                             quantum_advantage=1.5,
                                             memory_efficiency=0.8,
                                             circuit_depth=3,
                                             qubit_count=8,
                                             backend_provider="Azure",
                                             backend_name="IonQ Aria-1",
                                             theoretical_speedup="Exponential",
                                             gates_used=24,
                                             error_rate=0.02)
            db.add(quantum_metrics)

            # Add classical metrics
            classical_metrics = ClassicalMetrics(
                task_id=sample_task.id,
                processing_time=0.185,
                memory_usage=24.5,
                cpu_usage=15.3,
                algorithm_used="Trial division",
                complexity="O(sqrt(N))")
            db.add(classical_metrics)

            # Add performance comparison
            comparison = PerformanceComparison(
                task_id=sample_task.id,
                task_type="factorization",
                quantum_time=0.123,
                classical_time=0.185,
                speedup=1.5,
                quantum_method="Shor's algorithm",
                classical_method="Trial division",
                dataset_size=16)
            db.add(comparison)

            db.commit()
            logging.info("Database initialized with sample data")
    except Exception as e:
        db.rollback()
        logging.error(f"Error initializing database: {str(e)}")
    finally:
        db.close()
