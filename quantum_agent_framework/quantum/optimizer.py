"""Quantum optimization component for the QUASAR framework."""

import pennylane as qml
from pennylane import numpy as np
from typing import List, Tuple, Optional
import os
import logging

class QuantumOptimizer:
    """Manages quantum circuit optimization for enhanced decision making."""

    def __init__(self, n_qubits: int = 8, n_layers: int = 2, use_azure: bool = True):
        """
        Initialize the quantum optimizer.

        Args:
            n_qubits (int): Number of qubits in the circuit (default: 8 for optimal balance)
            n_layers (int): Number of circuit layers
            use_azure (bool): Whether to use Azure Quantum or local simulation
        """
        self.n_qubits = min(n_qubits, 29)  # Limit to IonQ simulator max
        self.n_layers = n_layers
        self.use_azure = use_azure and self._check_azure_credentials()

        try:
            # Always start with local simulator as fallback
            self.dev = qml.device("default.qubit", wires=self.n_qubits)
            logging.info("Initialized local quantum simulator as fallback")

            # Disable Azure if credentials missing
            if not self._check_azure_credentials():
                self.use_azure = False
                logging.info("Azure Quantum disabled - missing credentials")
                return

            # Only attempt Azure setup if explicitly enabled
            if self.use_azure:
                try:
                    # Configure Azure Quantum device
                    azure_dev = qml.device(
                        "qsharp.simulator",  # Using Q# simulator with IonQ backend
                        wires=self.n_qubits,
                        shots=1000,
                        target="ionq.simulator",
                        credentials={
                            'subscription_id': os.environ["AZURE_QUANTUM_SUBSCRIPTION_ID"],
                            'resource_group': os.environ["AZURE_QUANTUM_RESOURCE_GROUP"],
                            'workspace_name': os.environ["AZURE_QUANTUM_WORKSPACE_NAME"],
                            'location': os.environ["AZURE_QUANTUM_LOCATION"]
                        }
                    )
                    self.dev = azure_dev
                    logging.info(f"Successfully initialized Azure IonQ simulator with {self.n_qubits} qubits")
                except Exception as azure_err:
                    logging.warning(f"Azure initialization failed, using local simulator: {azure_err}")
                    self.use_azure = False

        except Exception as e:
            logging.error(f"Device initialization error: {str(e)}")
            self.use_azure = False

        # Initialize circuit parameters
        self.params = np.random.uniform(
            low=-np.pi, 
            high=np.pi,
            size=(n_layers, self.n_qubits, 3)
        )

        # Create quantum circuit
        self._quantum_cost = qml.QNode(self._circuit, self.dev, interface="autograd")

    def _check_azure_credentials(self) -> bool:
        """Check if all required Azure Quantum credentials are present."""
        required_env_vars = [
            "AZURE_QUANTUM_SUBSCRIPTION_ID",
            "AZURE_QUANTUM_RESOURCE_GROUP",
            "AZURE_QUANTUM_WORKSPACE_NAME",
            "AZURE_QUANTUM_LOCATION"
        ]
        return all(os.environ.get(var) for var in required_env_vars)

    def _circuit(self, params: np.ndarray, features: np.ndarray) -> float:
        """
        Define an optimized quantum circuit architecture.

        Args:
            params: Circuit parameters
            features: Input features to encode

        Returns:
            float: Expectation value
        """
        # Ensure features are 1D
        if len(features.shape) > 1:
            features = features.flatten()

        # Pad or truncate features to match qubit count
        if len(features) != self.n_qubits:
            features = np.pad(features, (0, self.n_qubits - len(features) % self.n_qubits))[:self.n_qubits]

        # Normalize features
        features = features / np.linalg.norm(features)

        # Apply feature encoding
        for i in range(self.n_qubits):
            qml.RY(features[i] * np.pi, wires=i)

        # Apply parameterized circuit layers
        for layer in range(self.n_layers):
            # Single-qubit rotations
            for qubit in range(self.n_qubits):
                qml.Rot(*params[layer, qubit], wires=qubit)

            # Entangling layer with native IonQ gates
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        # Return measurement expectation
        return qml.expval(qml.PauliZ(0))

    def optimize(self, features: np.ndarray, steps: int = 25) -> Tuple[np.ndarray, List[float]]:
        """
        Optimize the quantum circuit parameters.

        Args:
            features: Input features
            steps: Number of optimization steps (reduced for faster response)

        Returns:
            Tuple[np.ndarray, List[float]]: Optimized parameters and cost history
        """
        opt = qml.AdamOptimizer(stepsize=0.1)
        params = self.params.copy()
        cost_history = []

        try:
            for _ in range(steps):
                params = opt.step(lambda p: self._quantum_cost(p, features), params)
                cost = float(self._quantum_cost(params, features))
                cost_history.append(cost)

            self.params = params
            return params, cost_history

        except Exception as e:
            logging.error(f"Optimization failed: {str(e)}")
            return self.params, cost_history

    def get_expectation(self, features: np.ndarray) -> float:
        """Get the expectation value for given features."""
        try:
            return float(self._quantum_cost(self.params, features))
        except Exception as e:
            logging.error(f"Failed to get expectation value: {str(e)}")
            return 0.0

    def get_circuit_stats(self) -> dict:
        """Get quantum circuit statistics."""
        return {
            "n_qubits": self.n_qubits,
            "circuit_depth": self.n_layers,
            "backend": "Azure IonQ" if self.use_azure else "Local",
            "total_gates": self.n_layers * self.n_qubits * 4,  # Rotations + CNOTs
            "optimization_steps": 25  # Reduced for faster response
        }