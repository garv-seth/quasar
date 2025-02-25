"""Quantum optimization component for the QUASAR framework."""

import pennylane as qml
from pennylane import numpy as np
from typing import List, Tuple, Optional
import os
import logging

class QuantumOptimizer:
    """Manages quantum circuit optimization for enhanced decision making."""

    def __init__(self, n_qubits: int = 4, n_layers: int = 2, use_azure: bool = True):
        """
        Initialize the quantum optimizer.

        Args:
            n_qubits (int): Number of qubits in the circuit
            n_layers (int): Number of circuit layers
            use_azure (bool): Whether to use Azure Quantum or local simulation
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.use_azure = use_azure and self._check_azure_credentials()

        try:
            if self.use_azure:
                # Set up Azure Quantum device using IonQ simulator
                self.dev = qml.device(
                    "ionq.simulator",  # IonQ simulator through Azure
                    wires=n_qubits,
                    shots=1000,  # Number of shots for measurement
                    azure_credentials={
                        'subscription_id': os.environ["AZURE_QUANTUM_SUBSCRIPTION_ID"],
                        'resource_group': os.environ["AZURE_QUANTUM_RESOURCE_GROUP"],
                        'workspace_name': os.environ["AZURE_QUANTUM_WORKSPACE_NAME"],
                        'location': os.environ["AZURE_QUANTUM_LOCATION"],
                        'credential': os.environ["AZURE_QUANTUM_ACCESS_KEY"]
                    }
                )
                logging.info("Successfully initialized Azure Quantum IonQ simulator")
            else:
                # Fallback to local simulator
                self.dev = qml.device("default.qubit", wires=n_qubits)
                logging.info("Using local quantum simulator")

        except Exception as e:
            logging.error(f"Failed to initialize Azure Quantum device: {str(e)}")
            logging.info("Falling back to local quantum simulator")
            self.dev = qml.device("default.qubit", wires=n_qubits)
            self.use_azure = False

        # Initialize circuit parameters
        self.params = np.random.uniform(low=-np.pi, high=np.pi,
                                      size=(n_layers, n_qubits, 3))

        # Create the quantum node
        self._quantum_cost = qml.QNode(self._circuit, self.dev)

    def _check_azure_credentials(self) -> bool:
        """Check if all required Azure Quantum credentials are present."""
        required_env_vars = [
            "AZURE_QUANTUM_SUBSCRIPTION_ID",
            "AZURE_QUANTUM_RESOURCE_GROUP",
            "AZURE_QUANTUM_WORKSPACE_NAME",
            "AZURE_QUANTUM_LOCATION",
            "AZURE_QUANTUM_ACCESS_KEY",
            "AZURE_QUANTUM_CONNECTION_STRING"
        ]
        return all(os.environ.get(var) for var in required_env_vars)

    def _circuit(self, params: np.ndarray, features: np.ndarray) -> float:
        """
        Define the quantum circuit architecture.

        Args:
            params: Circuit parameters
            features: Input features to encode

        Returns:
            float: Expectation value
        """
        # Encode input features
        for i in range(self.n_qubits):
            qml.RY(features[i], wires=i)

        # Apply parameterized circuit
        for layer in range(self.n_layers):
            # Single-qubit rotations
            for qubit in range(self.n_qubits):
                qml.RX(params[layer, qubit, 0], wires=qubit)
                qml.RY(params[layer, qubit, 1], wires=qubit)
                qml.RZ(params[layer, qubit, 2], wires=qubit)

            # Entangling layer using native gates
            for i in range(self.n_qubits - 1):
                if self.use_azure:
                    qml.CNOT(wires=[i, i + 1])  # IonQ native gate
                else:
                    qml.CZ(wires=[i, i + 1])

        return qml.expval(qml.PauliZ(0))

    def optimize(self, features: np.ndarray, 
                steps: int = 100) -> Tuple[np.ndarray, List[float]]:
        """
        Optimize the quantum circuit parameters.

        Args:
            features: Input features for optimization
            steps: Number of optimization steps

        Returns:
            Tuple[np.ndarray, List[float]]: Optimized parameters and cost history
        """
        opt = qml.GradientDescentOptimizer(stepsize=0.01)
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
            # Return current parameters and history if optimization fails
            return self.params, cost_history

    def get_expectation(self, features: np.ndarray) -> float:
        """
        Get the expectation value for given features.

        Args:
            features: Input features

        Returns:
            float: Expectation value
        """
        try:
            return float(self._quantum_cost(self.params, features))
        except Exception as e:
            logging.error(f"Failed to get expectation value: {str(e)}")
            return 0.0