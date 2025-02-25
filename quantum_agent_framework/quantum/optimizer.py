"""Quantum optimization component for the QUASAR framework."""

import pennylane as qml
from pennylane import numpy as np
from typing import List, Tuple, Optional
import os
import logging

class QuantumOptimizer:
    """Manages quantum circuit optimization for enhanced decision making."""

    def __init__(self, n_qubits: int = 29, n_layers: int = 3, use_azure: bool = True):
        """
        Initialize the quantum optimizer.

        Args:
            n_qubits (int): Number of qubits in the circuit (max 29 for IonQ simulator)
            n_layers (int): Number of circuit layers
            use_azure (bool): Whether to use Azure Quantum or local simulation
        """
        self.n_qubits = min(n_qubits, 29)  # Limit to IonQ simulator max
        self.n_layers = n_layers
        self.use_azure = use_azure and self._check_azure_credentials()

        try:
            # Start with local simulator by default
            self.dev = qml.device("default.qubit", wires=self.n_qubits)
            logging.info("Using local quantum simulator")

            # Try Azure only if explicitly configured
            if self.use_azure and all(k in os.environ for k in [
                "AZURE_QUANTUM_SUBSCRIPTION_ID",
                "AZURE_QUANTUM_RESOURCE_GROUP",
                "AZURE_QUANTUM_WORKSPACE_NAME",
                "AZURE_QUANTUM_LOCATION"
            ]):
                try:
                    azure_dev = qml.device(
                        "microsoft.ionq.simulator",
                        wires=self.n_qubits,
                        shots=1000,
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
            self.dev = qml.device("default.qubit", wires=self.n_qubits)
            self.use_azure = False

        # Initialize circuit parameters for the full qubit range
        self.params = np.random.uniform(
            low=-np.pi, 
            high=np.pi,
            size=(n_layers, self.n_qubits, 3)
        )

        # Create the quantum node with optimization
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
        # Encode input features using amplitude encoding
        features_pad = np.pad(features, (0, self.n_qubits - len(features) % self.n_qubits))
        features_normalized = features_pad / np.linalg.norm(features_pad)

        qml.AmplitudeEmbedding(features_normalized, wires=range(self.n_qubits), normalize=True)

        # Apply parameterized circuit with optimized layout
        for layer in range(self.n_layers):
            # Apply parallel single-qubit rotations
            for qubit in range(self.n_qubits):
                qml.Rot(*params[layer, qubit], wires=qubit)

            # Apply entangling gates in parallel where possible
            for i in range(0, self.n_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])
            for i in range(1, self.n_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])

        # Measure in parallel for speedup
        measurements = [qml.expval(qml.PauliZ(i)) for i in range(min(4, self.n_qubits))]
        return np.mean(measurements)

    def optimize(self, features: np.ndarray, 
                steps: int = 50) -> Tuple[np.ndarray, List[float]]:
        """
        Optimize the quantum circuit parameters with parallel processing.

        Args:
            features: Input features for optimization
            steps: Number of optimization steps

        Returns:
            Tuple[np.ndarray, List[float]]: Optimized parameters and cost history
        """
        opt = qml.AdamOptimizer(stepsize=0.1)  # Using Adam for better convergence
        params = self.params.copy()
        cost_history = []

        try:
            for _ in range(steps):
                # Compute gradients in parallel
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
            "optimization_steps": 50
        }