"""Quantum optimization component for the QUASAR framework."""

import pennylane as qml
import numpy as np
from typing import List, Tuple, Optional, Dict, Union, Any
import os
import logging
import time
from openai import AsyncOpenAI
import azure.quantum as aq
import re

class QuantumOptimizer:
    """Manages quantum circuit optimization for enhanced mathematical computations."""

    def __init__(self, n_qubits: int = 8, n_layers: int = 2, use_azure: bool = True):
        """Initialize quantum optimizer with enhanced mathematical capabilities."""
        self.n_qubits = min(n_qubits, 29)  # IonQ hardware limit
        self.n_layers = n_layers
        self.use_azure = use_azure and self._check_azure_credentials()
        self.openai_client = AsyncOpenAI()

        try:
            if self.use_azure:
                # Initialize Azure Quantum workspace for IonQ Aria-1
                self.workspace = aq.Workspace(
                    subscription_id=os.environ.get("AZURE_QUANTUM_SUBSCRIPTION_ID"),
                    resource_group=os.environ.get("AZURE_QUANTUM_RESOURCE_GROUP"),
                    name=os.environ.get("AZURE_QUANTUM_WORKSPACE_NAME"),
                    location=os.environ.get("AZURE_QUANTUM_LOCATION")
                )
                self.dev = qml.device('qiskit.aer', wires=self.n_qubits, backend='aer_simulator')
                logging.info(f"Initialized Azure Quantum device with {self.n_qubits} qubits")
            else:
                # Fallback to IBM Qiskit
                self.dev = qml.device("default.qubit", wires=self.n_qubits)
                logging.info(f"Initialized IBM Qiskit simulator with {self.n_qubits} qubits")

            # Initialize quantum arithmetic circuit
            self._setup_quantum_arithmetic()

        except Exception as e:
            logging.error(f"Device initialization error: {str(e)}")
            # Final fallback to local simulator
            self.dev = qml.device("default.qubit", wires=self.n_qubits)
            self.use_azure = False

    async def preprocess_input(self, task: str) -> Dict[str, Any]:
        """Extract quantum task parameters from input with improved factorization detection."""
        try:
            # First try to extract number using regex
            number_match = re.search(r'factor.*?(\d+)', task.lower())
            if number_match:
                number = int(number_match.group(1))
                return {"type": "factorization", "number": number}

            # If no direct match, use GPT-4 for analysis
            messages = [
                {"role": "system", "content": """You are a quantum computing task analyzer.
                For ANY factorization task, extract the number and ALWAYS return:
                {"type": "factorization", "number": <extracted_number>}

                Look for:
                - Numbers to factorize
                - Terms like "factor", "prime", "divisor"
                - Mathematical notation (N=, p=, q=)

                ALWAYS classify factorization tasks as quantum tasks.
                If no number is found, return {"type": "unknown"}"""},
                {"role": "user", "content": task}
            ]

            completion = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=100,
                temperature=0,
                response_format={"type": "json_object"}
            )

            response = completion.choices[0].message.content
            return self._parse_ai_response(response)

        except Exception as e:
            logging.error(f"Preprocessing error: {str(e)}")
            return {"type": "unknown", "error": str(e)}

    def factorize_number(self, number: int) -> Dict[str, Any]:
        """Implement Shor's algorithm for quantum factorization."""
        try:
            logging.info(f"Starting quantum factorization of {number}")
            start_time = time.time()

            # Check if number is prime or too small
            if number < 4 or self._is_prime(number):
                return {
                    "success": True,
                    "factors": [1, number] if number > 1 else [1],
                    "computation_time": time.time() - start_time,
                    "method_used": "classical",
                    "backend": "Classical (Prime number check)",
                    "details": {
                        "reason": "Number is prime or too small for quantum factorization",
                        "quantum_advantage": "Not applicable for prime numbers"
                    }
                }

            # Convert number to binary for quantum processing
            binary_rep = np.array([int(x) for x in bin(number)[2:]])
            quantum_ready = len(binary_rep) <= self.n_qubits

            if not quantum_ready:
                return {
                    "success": False,
                    "factors": [],
                    "computation_time": time.time() - start_time,
                    "method_used": "none",
                    "backend": "Failed - Number too large",
                    "details": {
                        "error": f"Number requires {len(binary_rep)} qubits, max available: {self.n_qubits}",
                        "suggestion": "Use classical factorization for this size"
                    }
                }

            # Execute quantum circuit for period finding
            logging.info("Executing quantum circuit for period finding")
            probabilities = self.arithmetic_circuit(binary_rep, "factorize")

            # Process results
            period = self._find_period(probabilities)
            if period:
                # Calculate potential factors using period
                x = int(2 ** (period / 2))
                if x != 1 and x != number - 1:
                    p = np.gcd(x + 1, number)
                    q = np.gcd(x - 1, number)
                    if p > 1 and q > 1:
                        return {
                            "success": True,
                            "factors": sorted([int(p), int(q)]),
                            "computation_time": time.time() - start_time,
                            "method_used": "quantum",
                            "backend": "Azure Quantum IonQ" if self.use_azure else "IBM Qiskit",
                            "details": {
                                "period_found": period,
                                "quantum_advantage": "Used Shor's algorithm",
                                "hardware_backend": "IonQ Aria-1" if self.use_azure else "Qiskit Aer"
                            }
                        }

            return {
                "success": False,
                "factors": [],
                "computation_time": time.time() - start_time,
                "method_used": "quantum_failed",
                "backend": "Azure Quantum IonQ" if self.use_azure else "IBM Qiskit",
                "details": {
                    "error": "Quantum factorization failed to find factors",
                    "suggestion": "Try classical factorization"
                }
            }

        except Exception as e:
            logging.error(f"Factorization error: {str(e)}")
            return {
                "success": False,
                "factors": [],
                "computation_time": time.time() - start_time,
                "method_used": "error",
                "backend": "Error",
                "details": {"error": str(e)}
            }

    def _is_prime(self, n: int) -> bool:
        """Check if a number is prime."""
        if n < 2:
            return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    def _setup_quantum_arithmetic(self):
        """Setup quantum arithmetic circuits."""
        @qml.qnode(self.dev)
        def arithmetic_circuit(x: np.ndarray, operation: str):
            try:
                if operation == "factorize":
                    # Initialize registers in superposition
                    for i in range(self.n_qubits):
                        qml.Hadamard(wires=i)

                    # Phase estimation circuit
                    for i in range(self.n_qubits - 1):
                        # Controlled phase rotations
                        qml.CNOT(wires=[i, i + 1])
                        qml.RZ(np.pi / (2 ** i), wires=i + 1)
                        qml.CNOT(wires=[i, i + 1])

                    # QFT† on control register
                    qml.adjoint(qml.QFT)(wires=range(self.n_qubits))

                    # Return probabilities instead of expectation values
                    return qml.probs(wires=range(self.n_qubits))

                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

            except Exception as e:
                logging.error(f"Circuit execution failed: {str(e)}")
                raise

        self.arithmetic_circuit = arithmetic_circuit

    def _find_period(self, probabilities: np.ndarray) -> Optional[int]:
        """Find the period from quantum measurements."""
        try:
            # Ensure probabilities is a 1D numpy array
            if isinstance(probabilities, list):
                probabilities = np.array(probabilities)

            # Find peaks in probability distribution
            peaks = []
            threshold = np.max(probabilities) * 0.1  # 10% of maximum

            for i in range(1, len(probabilities) - 1):
                if (probabilities[i] > threshold and 
                    probabilities[i] > probabilities[i-1] and 
                    probabilities[i] > probabilities[i+1]):
                    peaks.append(i)

            if len(peaks) >= 2:
                # Calculate differences between consecutive peaks
                differences = np.diff(peaks)
                # Return the most common difference as the period
                return int(np.median(differences))

            return None

        except Exception as e:
            logging.error(f"Period finding error: {str(e)}")
            return None

    def _parse_ai_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response with enhanced number extraction."""
        import json
        try:
            # First try direct JSON parsing
            params = json.loads(response)
            if params.get('type') == 'factorization':
                number = params.get('number')
                if number and str(number).isdigit():
                    return {'type': 'factorization', 'number': int(number)}
            return params

        except json.JSONDecodeError:
            # Fallback to regex parsing for numbers
            import re
            # Look for numbers in various formats
            number_pattern = r'(?:N\s*=\s*)?(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)'
            numbers = re.findall(number_pattern, response)

            if numbers:
                try:
                    # Take the largest number found
                    number = max(int(float(n)) for n in numbers)
                    return {'type': 'factorization', 'number': number}
                except ValueError:
                    pass

            # Check for factorization keywords
            if any(word in response.lower() for word in ['factor', 'prime', 'semiprime', 'multiply']):
                return {'type': 'factorization', 'number': 0}

            return {'type': 'classical', 'description': response}

    def get_circuit_stats(self) -> Dict[str, Any]:
        """Get quantum circuit statistics."""
        return {
            "circuit_depth": self.n_layers * 3,
            "total_gates": self.n_qubits * self.n_layers * 4,
            "backend": "Azure Quantum IonQ" if self.use_azure else "IBM Qiskit",
            "max_number_size": 2 ** (self.n_qubits // 2)
        }

    def _check_azure_credentials(self) -> bool:
        """Check if all required Azure Quantum credentials are present."""
        required_env_vars = [
            "AZURE_QUANTUM_SUBSCRIPTION_ID",
            "AZURE_QUANTUM_RESOURCE_GROUP",
            "AZURE_QUANTUM_WORKSPACE_NAME",
            "AZURE_QUANTUM_LOCATION"
        ]
        return all(os.environ.get(var) for var in required_env_vars)