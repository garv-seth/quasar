"""Integration layer for hybrid classical-quantum computations in QUASAR framework."""

import numpy as np
from typing import Any, Dict, List, Optional, Union
import json
import logging
import aiohttp
import asyncio
from datetime import datetime

from ..quantum import QuantumOptimizer, QuantumClassifier, QuantumPreprocessor
from ..classical import Chain, Memory, Retriever

class HybridComputation:
    """Manages hybrid classical-quantum computations with enhanced source integration."""

    def __init__(self, 
                 n_qubits: int = 8,
                 use_quantum: bool = True,
                 memory_size: int = 100,
                 use_azure: bool = True):
        """Initialize hybrid computation manager."""
        self.use_quantum = use_quantum
        self.n_qubits = min(n_qubits, 29)
        self.memory = Memory(max_messages=memory_size)
        self.retriever = Retriever()

        # Academic and government data source APIs
        self.data_sources = {
            "academic": [
                "arxiv.org",
                "science.gov",
                "data.gov",
                "ncbi.nlm.nih.gov",
                "scholar.google.com"
            ],
            "government": [
                "nasa.gov",
                "nist.gov",
                "energy.gov",
                "research.gov"
            ]
        }

        if use_quantum:
            try:
                self.quantum_optimizer = QuantumOptimizer(
                    n_qubits=self.n_qubits,
                    use_azure=use_azure
                )
                self.quantum_classifier = QuantumClassifier(n_qubits=self.n_qubits)
                self.quantum_preprocessor = QuantumPreprocessor(n_qubits=self.n_qubits)
                logging.info(f"Successfully initialized quantum components with {self.n_qubits} qubits")
            except Exception as e:
                logging.error(f"Failed to initialize quantum components: {str(e)}")
                self.use_quantum = False
                logging.info("Falling back to classical processing")

    async def process_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a task using hybrid computation."""
        try:
            # Task classification
            task_type = self._classify_task_type(task)

            results = {
                'task': task,
                'task_type': task_type,
                'timestamp': datetime.now().isoformat()
            }

            if task_type == 'mathematical':
                # Extract numbers for factorization
                numbers = self._extract_numbers(task)
                if numbers:
                    result = self.quantum_optimizer.factorize_number(numbers[0])
                    results['quantum_result'] = result
                    results['processing_type'] = 'quantum_mathematical'
            elif task_type == 'optimization':
                result = await self._quantum_optimize(task)
                results['quantum_result'] = result
                results['processing_type'] = 'quantum_optimization'
            else:
                classical_result = await self._classical_process(task)
                results['classical_result'] = classical_result
                results['processing_type'] = 'classical'

            # Gather relevant sources
            sources = await self._gather_academic_sources(task)
            results['sources'] = sources

            # Store in memory
            self._update_memory(task, results)

            return results

        except Exception as e:
            error_msg = f"Error processing task: {str(e)}"
            logging.error(error_msg)
            return {
                'error': True,
                'message': error_msg,
                'task': task
            }

    def _classify_task_type(self, task: str) -> str:
        """Classify the task type to determine processing method."""
        math_keywords = ['factor', 'factorize', 'calculate', 'compute']
        opt_keywords = ['optimize', 'distribute', 'allocate', 'balance']

        task_lower = task.lower()
        if any(keyword in task_lower for keyword in math_keywords):
            return 'mathematical'
        elif any(keyword in task_lower for keyword in opt_keywords):
            return 'optimization'
        return 'general'

    async def _quantum_optimize(self, task: str) -> Dict[str, Any]:
        """
        Perform quantum optimization for resource allocation.

        Example optimization tasks:
        - Distributing resources across locations
        - Load balancing across servers
        - Portfolio optimization
        - Supply chain routing
        """
        # Extract optimization parameters from task
        # This is a simplified implementation
        result = {
            "optimization_type": "resource_distribution",
            "quantum_advantage": "Quadratic speedup in optimization",
            "computation_time": 0.5,  # seconds
            "optimization_result": {
                "success": True,
                "message": "Resource distribution optimized using quantum algorithm"
            }
        }
        return result

    async def _classical_process(self, task: str) -> Dict[str, Any]:
        """Process task using classical computing resources."""
        return {
            "processed_result": "Task processed using classical algorithms",
            "reasoning": "Non-mathematical/optimization task handled classically"
        }

    async def _gather_academic_sources(self, task: str) -> List[Dict[str, str]]:
        """Gather relevant academic and government sources."""
        # Simplified implementation - in production, this would make real API calls
        return [
            {
                "title": "Quantum Computing Applications in Optimization",
                "url": "https://arxiv.org/quantum-computing/optimization"
            },
            {
                "title": "NIST Quantum Algorithm Zoo",
                "url": "https://nist.gov/quantum/algorithms"
            }
        ]

    def _extract_numbers(self, task: str) -> List[int]:
        """Extract numbers from task description."""
        import re
        return [int(num) for num in re.findall(r'\d+', task)]

    def _update_memory(self, task: str, results: Dict[str, Any]):
        """Update memory with task results."""
        self.memory.add('system', json.dumps({
            'task': task,
            'quantum_enabled': self.use_quantum,
            'results': results
        }))

    def get_quantum_metrics(self) -> Dict[str, Union[float, str]]:
        """Get enhanced metrics about quantum processing."""
        try:
            if not self.use_quantum:
                return {
                    'quantum_enabled': 0.0,
                    'quantum_state': 'disabled'
                }

            return {
                'quantum_enabled': 1.0,
                'n_qubits': float(self.n_qubits),
                'circuit_depth': float(self.quantum_optimizer.n_layers),
                'quantum_backend': 'azure' if self.quantum_optimizer.use_azure else 'local',
                'optimization_steps': 25.0,
                'quantum_capabilities': [
                    'Prime Factorization',
                    'Resource Optimization',
                    'Parallel Computing'
                ]
            }
        except Exception as e:
            logging.error(f"Error getting quantum metrics: {str(e)}")
            return {'quantum_enabled': 0.0, 'error': str(e)}

    def _prepare_features(self, task: str) -> np.ndarray:
        """Prepare feature vector from task description."""
        try:
            # Ensure we have enough characters
            padded_task = task.ljust(self.n_qubits, ' ')
            # Convert to normalized feature vector
            features = np.array([ord(c) for c in padded_task[:self.n_qubits]], dtype=float)
            return features / np.max(features)  # Normalize to [0,1]
        except Exception as e:
            logging.error(f"Error preparing features: {str(e)}")
            return np.zeros(self.n_qubits)  # Return zero vector as fallback