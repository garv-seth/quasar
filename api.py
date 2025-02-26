"""API integration module for the QUASAR framework."""

import aiohttp
import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import numpy as np
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from quantum_agent_framework.agents.web_agent import QuantumWebAgent
from quantum_agent_framework.integration import HybridComputation
from database import get_db
from database.models import Task
from config import Config

# Initialize FastAPI app
app = FastAPI(
    title="Q3A: Quantum-Accelerated AI Agent API",
    description=
    "API for the Quantum-Accelerated AI Agent (Q3A) powered by QUASAR framework",
    version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize logger
logger = logging.getLogger("quasar-api")


# Pydantic models for request/response validation
class SearchQuery(BaseModel):
    query: str = Field(..., description="Search query")
    urls: Optional[List[str]] = Field(
        None, description="Optional list of URLs to search")
    use_quantum: bool = Field(
        True, description="Whether to use quantum acceleration")
    n_qubits: int = Field(8,
                          description="Number of qubits to use",
                          ge=4,
                          le=29)


class FactorizationQuery(BaseModel):
    number: int = Field(..., description="Number to factorize", gt=0)
    use_quantum: bool = Field(
        True, description="Whether to use quantum acceleration")
    n_qubits: int = Field(8,
                          description="Number of qubits to use",
                          ge=4,
                          le=29)


class OptimizationQuery(BaseModel):
    resources: Dict[str, Any] = Field(..., description="Resources to optimize")
    use_quantum: bool = Field(
        True, description="Whether to use quantum acceleration")
    n_qubits: int = Field(8,
                          description="Number of qubits to use",
                          ge=4,
                          le=29)


class TaskResponse(BaseModel):
    task_id: int = Field(..., description="Task ID")
    status: str = Field(..., description="Task status")
    created_at: str = Field(..., description="Task creation timestamp")


class TaskResult(BaseModel):
    task_id: int = Field(..., description="Task ID")
    description: str = Field(..., description="Task description")
    status: str = Field(..., description="Task status")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result")
    processing_method: str = Field(..., description="Processing method used")
    execution_time: Optional[float] = Field(
        None, description="Execution time in seconds")
    task_type: str = Field(..., description="Task type")
    created_at: str = Field(..., description="Task creation timestamp")
    completed_at: Optional[str] = Field(
        None, description="Task completion timestamp")


# API endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": Config.APP_NAME,
        "version": Config.APP_VERSION,
        "description": Config.APP_DESCRIPTION,
        "quantum_enabled": Config.QUANTUM_ENABLED,
        "endpoints": {
            "GET /": "This information",
            "GET /health": "Health check",
            "GET /tasks": "List recent tasks",
            "GET /tasks/{task_id}": "Get task details",
            "POST /search": "Perform quantum-enhanced search",
            "POST /factorize": "Factorize a number",
            "POST /optimize": "Optimize resource allocation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "quantum_enabled": Config.QUANTUM_ENABLED,
        "config_valid": all(Config.validate().values())
    }


@app.get("/tasks", response_model=List[TaskResult])
async def get_tasks(limit: int = Query(10, ge=1, le=100),
                    offset: int = Query(0, ge=0),
                    task_type: Optional[str] = Query(None),
                    status: Optional[str] = Query(None),
                    db=Depends(get_db)):
    """Get recent tasks."""
    from database import crud

    tasks = crud.get_task_history(db,
                                  limit=limit,
                                  offset=offset,
                                  task_type=task_type,
                                  status=status)

    return [
        TaskResult(task_id=task.id,
                   description=task.description,
                   status=task.status,
                   result=task.result,
                   processing_method=task.processing_method,
                   execution_time=task.execution_time,
                   task_type=task.task_type,
                   created_at=task.created_at.isoformat(),
                   completed_at=task.completed_at.isoformat()
                   if task.completed_at else None) for task in tasks
    ]


@app.get("/tasks/{task_id}", response_model=TaskResult)
async def get_task(task_id: int, db=Depends(get_db)):
    """Get task details."""
    from database import crud

    task = crud.get_task(db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return TaskResult(task_id=task.id,
                      description=task.description,
                      status=task.status,
                      result=task.result,
                      processing_method=task.processing_method,
                      execution_time=task.execution_time,
                      task_type=task.task_type,
                      created_at=task.created_at.isoformat(),
                      completed_at=task.completed_at.isoformat()
                      if task.completed_at else None)


@app.post("/search", response_model=TaskResponse)
async def search(query: SearchQuery,
                 background_tasks: BackgroundTasks,
                 db=Depends(get_db)):
    """Perform quantum-enhanced search."""
    from database import crud

    # Create task record
    task = crud.create_task(db,
                            description=f"Search: {query.query}",
                            task_type="search")

    # Update task status
    crud.update_task_status(db, task.id, "processing")

    # Process in background
    background_tasks.add_task(process_search,
                              task_id=task.id,
                              query=query.query,
                              urls=query.urls,
                              use_quantum=query.use_quantum,
                              n_qubits=query.n_qubits)

    return TaskResponse(task_id=task.id,
                        status="processing",
                        created_at=task.created_at.isoformat())


@app.post("/factorize", response_model=TaskResponse)
async def factorize(query: FactorizationQuery,
                    background_tasks: BackgroundTasks,
                    db=Depends(get_db)):
    """Factorize a number."""
    from database import crud

    # Create task record
    task = crud.create_task(db,
                            description=f"Factorize: {query.number}",
                            task_type="factorization")

    # Update task status
    crud.update_task_status(db, task.id, "processing")

    # Process in background
    background_tasks.add_task(process_factorization,
                              task_id=task.id,
                              number=query.number,
                              use_quantum=query.use_quantum,
                              n_qubits=query.n_qubits)

    return TaskResponse(task_id=task.id,
                        status="processing",
                        created_at=task.created_at.isoformat())


@app.post("/optimize", response_model=TaskResponse)
async def optimize(query: OptimizationQuery,
                   background_tasks: BackgroundTasks,
                   db=Depends(get_db)):
    """Optimize resource allocation."""
    from database import crud

    # Create task record
    task = crud.create_task(
        db,
        description=
        f"Optimize resource allocation with {len(query.resources.get('items', []))} items",
        task_type="optimization")

    # Update task status
    crud.update_task_status(db, task.id, "processing")

    # Process in background
    background_tasks.add_task(process_optimization,
                              task_id=task.id,
                              resources=query.resources,
                              use_quantum=query.use_quantum,
                              n_qubits=query.n_qubits)

    return TaskResponse(task_id=task.id,
                        status="processing",
                        created_at=task.created_at.isoformat())


@app.get("/metrics")
async def get_metrics(db=Depends(get_db)):
    """Get performance metrics."""
    from database import crud

    # Get quantum statistics
    quantum_stats = crud.get_quantum_statistics(db)

    # Get performance metrics for different task types
    factorization_metrics = crud.get_performance_metrics(
        db, task_type="factorization")
    search_metrics = crud.get_performance_metrics(db, task_type="search")
    optimization_metrics = crud.get_performance_metrics(
        db, task_type="optimization")

    return {
        "quantum_statistics": quantum_stats,
        "performance_metrics": {
            "factorization": factorization_metrics,
            "search": search_metrics,
            "optimization": optimization_metrics
        },
        "timestamp": datetime.utcnow().isoformat()
    }


# Background task processing functions
async def process_search(task_id: int, query: str, urls: Optional[List[str]],
                         use_quantum: bool, n_qubits: int):
    """Process search in background."""
    from database import crud
    from database.models import SessionLocal

    db = SessionLocal()
    try:
        # Initialize web agent
        web_agent = QuantumWebAgent(n_qubits=n_qubits, use_quantum=use_quantum)

        # Start timing
        start_time = time.time()

        # Also measure classical time for comparison
        classical_start = time.time()
        classical_result = await web_agent.search(query,
                                                  urls,
                                                  use_quantum=False)
        classical_time = time.time() - classical_start

        # Process search with quantum if enabled
        if use_quantum:
            search_result = await web_agent.search(query, urls)
            method_used = "quantum"
        else:
            search_result = classical_result
            method_used = "classical"

        # Calculate execution time
        execution_time = time.time() - start_time

        # Add classical time for comparison
        search_result["classical_time"] = classical_time

        # Update task with result
        crud.update_task_result(db, task_id, search_result, execution_time,
                                method_used)

        # Add performance comparison
        if use_quantum and "performance_comparison" in search_result:
            comparison = search_result["performance_comparison"]
            crud.create_performance_comparison(
                db, task_id, {
                    "task_type": "search",
                    "quantum_time": comparison.get("quantum_time", 0),
                    "classical_time": comparison.get("classical_time", 0),
                    "speedup": comparison.get("speedup", 0),
                    "quantum_method": "Quantum-enhanced search",
                    "classical_method": "Classical search",
                    "dataset_size": search_result.get("total_sources", 0)
                })

        # Add quantum metrics if available
        if use_quantum and "quantum_metrics" in search_result:
            metrics = search_result["quantum_metrics"]
            crud.create_quantum_metrics(
                db, task_id, {
                    "quantum_advantage":
                    float(metrics.get("quantum_advantage", "0").split("x")[0]),
                    "circuit_depth":
                    metrics.get("circuit_depth", 0),
                    "qubit_count":
                    n_qubits,
                    "backend_provider":
                    "Simulation",
                    "backend_name":
                    "Pennylane default.qubit",
                    "theoretical_speedup":
                    "Quadratic"
                })

    except Exception as e:
        logger.error(f"Error processing search task {task_id}: {str(e)}")
        # Update task with error
        crud.update_task_result(db, task_id, {"error": str(e)}, 0, "error")
    finally:
        db.close()


async def process_factorization(task_id: int, number: int, use_quantum: bool,
                                n_qubits: int):
    """Process factorization in background."""
    from database import crud
    from database.models import SessionLocal

    db = SessionLocal()
    try:
        # Initialize hybrid computation
        hybrid_comp = HybridComputation(n_qubits=n_qubits,
                                        use_quantum=use_quantum)

        # Process factorization
        result = await hybrid_comp.process_task(f"Factor {number}")

        # Update task with result
        crud.update_task_result(db, task_id, result,
                                result.get("computation_time", 0),
                                result.get("method_used", "unknown"))

        # Add performance comparison if available
        if "classical_time" in result:
            crud.create_performance_comparison(
                db, task_id, {
                    "task_type":
                    "factorization",
                    "quantum_time":
                    result.get("computation_time", 0),
                    "classical_time":
                    result.get("classical_time", 0),
                    "speedup":
                    result.get("classical_time", 0) /
                    result.get("computation_time", 1),
                    "quantum_method":
                    "Shor's algorithm",
                    "classical_method":
                    "Trial division",
                    "dataset_size":
                    number
                })

        # Add quantum metrics if available
        if use_quantum and "quantum_metrics" in result:
            metrics = result["quantum_metrics"]
            crud.create_quantum_metrics(
                db, task_id, {
                    "quantum_advantage":
                    float(
                        result.get("classical_time", 0) /
                        result.get("computation_time", 1)),
                    "circuit_depth":
                    metrics.get("circuit_depth", 0),
                    "qubit_count":
                    metrics.get("n_qubits", n_qubits),
                    "backend_provider":
                    "Simulation",
                    "backend_name":
                    metrics.get("quantum_backend", "Pennylane default.qubit"),
                    "theoretical_speedup":
                    "Exponential",
                    "gates_used":
                    metrics.get("total_gates", 0)
                })

    except Exception as e:
        logger.error(
            f"Error processing factorization task {task_id}: {str(e)}")
        # Update task with error
        crud.update_task_result(db, task_id, {"error": str(e)}, 0, "error")
    finally:
        db.close()


async def process_optimization(task_id: int, resources: Dict[str, Any],
                               use_quantum: bool, n_qubits: int):
    """Process optimization in background."""
    from database import crud
    from database.models import SessionLocal

    db = SessionLocal()
    try:
        # Initialize hybrid computation
        hybrid_comp = HybridComputation(n_qubits=n_qubits,
                                        use_quantum=use_quantum)

        # Convert resources to a task description
        task_description = f"Optimize distribution of {len(resources.get('items', []))} resources"

        # Process optimization
        start_time = time.time()

        # Measure classical time for comparison
        classical_start = time.time()
        classical_result = await hybrid_comp._classical_optimization(resources)
        classical_time = time.time() - classical_start

        if use_quantum:
            # Get quantum optimizer
            optimizer = hybrid_comp.quantum_optimizer

            # Process with quantum
            quantum_result = optimizer.optimize_resource_allocation(resources)
            result = quantum_result
            method_used = "quantum_optimization"
        else:
            result = classical_result
            method_used = "classical_optimization"

        # Calculate execution time
        execution_time = time.time() - start_time

        # Add classical time for comparison
        result["classical_time"] = classical_time

        # Update task with result
        crud.update_task_result(db, task_id, result, execution_time,
                                method_used)

        # Add performance comparison
        speedup = classical_time / execution_time if execution_time > 0 else 0
        crud.create_performance_comparison(
            db, task_id, {
                "task_type": "optimization",
                "quantum_time": execution_time,
                "classical_time": classical_time,
                "speedup": speedup,
                "quantum_method": "QAOA",
                "classical_method": "Greedy optimization",
                "dataset_size": len(resources.get("items", []))
            })

        # Add quantum metrics if using quantum
        if use_quantum:
            circuit_stats = optimizer.get_circuit_stats()
            crud.create_quantum_metrics(
                db, task_id, {
                    "quantum_advantage":
                    speedup,
                    "circuit_depth":
                    circuit_stats.get("circuit_depth", 0),
                    "qubit_count":
                    n_qubits,
                    "backend_provider":
                    "Simulation",
                    "backend_name":
                    circuit_stats.get("backend", "Pennylane default.qubit"),
                    "theoretical_speedup":
                    "Polynomial",
                    "gates_used":
                    circuit_stats.get("total_gates", 0)
                })

    except Exception as e:
        logger.error(f"Error processing optimization task {task_id}: {str(e)}")
        # Update task with error
        crud.update_task_result(db, task_id, {"error": str(e)}, 0, "error")
    finally:
        db.close()


def start_api():
    """Start the FastAPI server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    start_api()
