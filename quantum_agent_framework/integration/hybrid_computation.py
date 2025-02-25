"""Integration layer for hybrid classical-quantum computations in QUASAR framework."""

import numpy as np
from typing import Any, Dict, List, Optional

from ..quantum import QuantumOptimizer, QuantumClassifier, QuantumPreprocessor
from ..classical import Chain, Memory, Retriever

class HybridComputation:
    """Manages hybrid classical-quantum computations."""
    
    def __init__(self, 
                 n_qubits: int = 4,
                 use_quantum: bool = True,
                 memory_size: int = 100):
        """
        Initialize hybrid computation manager.
        
        Args:
            n_qubits (int): Number of qubits for quantum components
            use_quantum (bool): Whether to use quantum acceleration
            memory_size (int): Size of memory buffer
        """
        self.use_quantum = use_quantum
        self.memory = Memory(max_messages=memory_size)
        self.retriever = Retriever()
        
        if use_quantum:
            self.quantum_optimizer = QuantumOptimizer(n_qubits=n_qubits)
            self.quantum_classifier = QuantumClassifier(n_qubits=n_qubits)
            self.quantum_preprocessor = QuantumPreprocessor(n_qubits=n_qubits)
            
    async def process_task(self, 
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
        context = context or {}
        results = {}
        
        # Convert task to feature vector (simplified)
        features = np.array([ord(c) for c in task[:self.n_qubits]])
        features = features / np.max(features)  # Normalize
        
        if self.use_quantum:
            # Quantum preprocessing
            processed_features = self.quantum_preprocessor.preprocess(features)
            
            # Optimize task parameters
            opt_params, _ = self.quantum_optimizer.optimize(processed_features)
            results['quantum_params'] = opt_params.tolist()
            
            # Classify task type
            task_class = self.quantum_classifier.predict(processed_features)
            results['task_class'] = task_class
        else:
            # Classical fallback
            results['classical_score'] = float(np.mean(features))
            
        # Store in memory
        self.memory.add('system', f'Processed task: {task}')
        results['memory'] = self.memory.get_history()
        
        return results
        
    def get_quantum_metrics(self) -> Dict[str, float]:
        """Get metrics about quantum processing."""
        if not self.use_quantum:
            return {'quantum_enabled': 0.0}
            
        return {
            'quantum_enabled': 1.0,
            'n_qubits': float(self.quantum_optimizer.n_qubits),
            'circuit_depth': float(self.quantum_optimizer.n_layers)
        }
