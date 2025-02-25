"""Integration layer for hybrid classical-quantum computations in QUASAR framework."""

import numpy as np
from typing import Any, Dict, List, Optional
import json
import logging

from ..quantum import QuantumOptimizer, QuantumClassifier, QuantumPreprocessor
from ..classical import Chain, Memory, Retriever

class HybridComputation:
    """Manages hybrid classical-quantum computations."""

    def __init__(self, 
                 n_qubits: int = 4,
                 use_quantum: bool = True,
                 memory_size: int = 100,
                 use_azure: bool = True):
        """
        Initialize hybrid computation manager.

        Args:
            n_qubits (int): Number of qubits for quantum components
            use_quantum (bool): Whether to use quantum acceleration
            memory_size (int): Size of memory buffer
            use_azure (bool): Whether to use Azure Quantum (vs local simulation)
        """
        self.use_quantum = use_quantum
        self.n_qubits = n_qubits
        self.memory = Memory(max_messages=memory_size)
        self.retriever = Retriever()

        if use_quantum:
            try:
                self.quantum_optimizer = QuantumOptimizer(
                    n_qubits=n_qubits,
                    use_azure=use_azure
                )
                self.quantum_classifier = QuantumClassifier(n_qubits=n_qubits)
                self.quantum_preprocessor = QuantumPreprocessor(n_qubits=n_qubits)
                logging.info("Successfully initialized quantum components")
            except Exception as e:
                logging.error(f"Failed to initialize quantum components: {str(e)}")
                self.use_quantum = False

    def _prepare_features(self, task: str) -> np.ndarray:
        """Prepare feature vector from task description."""
        # Ensure we have enough characters
        padded_task = task.ljust(self.n_qubits, ' ')
        # Convert to normalized feature vector
        features = np.array([ord(c) for c in padded_task[:self.n_qubits]], dtype=float)
        return features / np.max(features)  # Normalize to [0,1]

    def process_task(self, 
                    task: str, 
                    context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a task using hybrid computation.

        Args:
            task: Task description
            context: Additional context for processing

        Returns:
            Dict[str, Any]: Processing results
        """
        try:
            results = {
                'task': task,
                'timestamp': self.memory.get_history()[-1]['timestamp'] if self.memory.get_history() else None
            }

            # Convert task to feature vector
            features = self._prepare_features(task)

            if self.use_quantum:
                try:
                    # Quantum preprocessing
                    processed_features = self.quantum_preprocessor.preprocess(features)
                    results['processed_features'] = processed_features.tolist()

                    # Optimize task parameters
                    opt_params, cost_history = self.quantum_optimizer.optimize(
                        processed_features,
                        steps=50  # Reduced steps for faster response
                    )
                    results['quantum_params'] = opt_params.tolist()
                    results['optimization_history'] = [float(c) for c in cost_history]

                    # Classify task type
                    task_class = self.quantum_classifier.predict(processed_features)
                    results['task_class'] = int(task_class)
                except Exception as e:
                    logging.error(f"Quantum processing failed: {str(e)}")
                    self.use_quantum = False
                    results['quantum_error'] = str(e)

            if not self.use_quantum:
                # Classical fallback
                results['classical_score'] = float(np.mean(features))
                results['quantum_enabled'] = False

            # Store in memory
            self.memory.add('system', json.dumps({
                'task': task,
                'quantum_enabled': self.use_quantum,
                'results': results
            }))

            # Add memory to results
            results['memory'] = [
                {
                    'role': m['role'],
                    'content': m['content'],
                    'timestamp': m['timestamp']
                }
                for m in self.memory.get_history()
            ]

            return results

        except Exception as e:
            error_msg = f"Error processing task: {str(e)}"
            logging.error(error_msg)
            return {
                'error': True,
                'message': error_msg,
                'task': task
            }

    def get_quantum_metrics(self) -> Dict[str, float]:
        """Get metrics about quantum processing."""
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
                'optimization_steps': 50.0
            }
        except Exception as e:
            logging.error(f"Error getting quantum metrics: {str(e)}")
            return {'quantum_enabled': 0.0, 'error': str(e)}