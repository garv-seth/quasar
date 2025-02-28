"""
Quantum Agent Bridge: Integrate Quantum Computing with Agentic Capabilities

This module bridges the quantum computing capabilities with agentic decision making,
enabling true quantum-enhanced agency. It provides:

1. Quantum-enhanced decision making for agent actions
2. Integration between quantum computing resources and agent perception
3. Conversion between agent goals and quantum optimization problems
4. Quantum circuit construction for agent decision processes
"""

import os
import time
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("quantum-agent-bridge")

class QuantumAgentBridge:
    """
    Bridge between quantum computing capabilities and agent systems
    
    This class enables:
    1. Quantum-enhanced decision making
    2. Quantum acceleration of agent perception
    3. Quantum optimization of agent planning
    4. Seamless integration with both real quantum hardware and simulation
    """
    
    def __init__(self, 
                 quantum_engine,
                 use_hardware: bool = True,
                 n_qubits: int = 8,
                 decision_enhancement_level: int = 2):
        """
        Initialize the quantum agent bridge
        
        Args:
            quantum_engine: Quantum computing engine to use
            use_hardware: Whether to use real quantum hardware when available
            n_qubits: Number of qubits to use
            decision_enhancement_level: Level of quantum enhancement for decisions (1-3)
        """
        self.quantum_engine = quantum_engine
        self.use_hardware = use_hardware
        self.n_qubits = n_qubits
        self.decision_enhancement_level = max(1, min(3, decision_enhancement_level))
        
        self.initialized = self.quantum_engine is not None
        self.perception_circuits = {}
        self.decision_circuits = {}
        self.learning_circuits = {}
        
        self.execution_history = []
        
        if self.initialized:
            self._precompile_common_circuits()
            logger.info(f"Quantum Agent Bridge initialized with {n_qubits} qubits and enhancement level {decision_enhancement_level}")
        else:
            logger.warning("Quantum engine not available. Quantum enhancement disabled.")
    
    def _precompile_common_circuits(self):
        """Precompile commonly used quantum circuits for efficiency"""
        try:
            # Precompile decision circuits
            self._precompile_decision_circuits()
            
            # Precompile perception circuits
            self._precompile_perception_circuits()
            
            # Precompile learning circuits
            self._precompile_learning_circuits()
            
        except Exception as e:
            logger.error(f"Error precompiling quantum circuits: {str(e)}")
    
    def _precompile_decision_circuits(self):
        """Precompile common decision circuits"""
        try:
            # Binary decision circuit
            self.decision_circuits["binary"] = self._create_binary_decision_circuit()
            
            # Multi-option decision circuit (for up to 8 options)
            self.decision_circuits["multi_option"] = self._create_multi_option_circuit()
            
            # Priority decision circuit (for goal prioritization)
            self.decision_circuits["prioritization"] = self._create_prioritization_circuit()
            
            logger.info("Decision circuits precompiled successfully")
        except Exception as e:
            logger.error(f"Error precompiling decision circuits: {str(e)}")
    
    def _precompile_perception_circuits(self):
        """Precompile common perception circuits"""
        try:
            # Image classification circuit
            self.perception_circuits["image_classifier"] = self._create_image_classifier_circuit()
            
            # Text relevance circuit
            self.perception_circuits["text_relevance"] = self._create_text_relevance_circuit()
            
            # Element detection circuit
            self.perception_circuits["element_detector"] = self._create_element_detector_circuit()
            
            logger.info("Perception circuits precompiled successfully")
        except Exception as e:
            logger.error(f"Error precompiling perception circuits: {str(e)}")
    
    def _precompile_learning_circuits(self):
        """Precompile common learning circuits"""
        try:
            # Reinforcement learning circuit
            self.learning_circuits["reinforcement"] = self._create_reinforcement_learning_circuit()
            
            # Memory optimization circuit
            self.learning_circuits["memory_optimization"] = self._create_memory_optimization_circuit()
            
            logger.info("Learning circuits precompiled successfully")
        except Exception as e:
            logger.error(f"Error precompiling learning circuits: {str(e)}")
    
    def _create_binary_decision_circuit(self):
        """Create a quantum circuit for binary decisions"""
        try:
            import pennylane as qml
            
            # Define the binary decision circuit
            def binary_decision_circuit(params):
                # Extract parameters
                confidence_a = max(0.0, min(1.0, params.get("confidence_a", 0.5)))
                confidence_b = max(0.0, min(1.0, params.get("confidence_b", 0.5)))
                bias = params.get("bias", 0.0)  # Bias towards option A (positive) or B (negative)
                
                # Initialize in superposition
                qml.Hadamard(wires=0)
                
                # Apply confidence-based rotations
                qml.RY(confidence_a * 1.5, wires=0)
                
                # Apply bias
                qml.RZ(bias, wires=0)
                
                # Return measurement
                return qml.sample(wires=0)
            
            return binary_decision_circuit
        except ImportError:
            logger.warning("PennyLane not available for binary decision circuit")
            return None
    
    def _create_multi_option_circuit(self):
        """Create a quantum circuit for multi-option decisions"""
        try:
            import pennylane as qml
            import numpy as np
            
            # Define the multi-option decision circuit
            def multi_option_circuit(params):
                # Get the number of options (power of 2)
                num_options = params.get("num_options", 4)
                n_qubits = max(1, int(np.ceil(np.log2(num_options))))
                n_qubits = min(n_qubits, self.n_qubits)
                
                # Get confidence values for each option
                confidences = params.get("confidences", [0.5] * num_options)
                confidences = [max(0.0, min(1.0, c)) for c in confidences]
                
                # Initialize in superposition
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
                
                # Enhance the amplitude of each option based on confidence
                for option_idx, confidence in enumerate(confidences[:2**n_qubits]):
                    if option_idx >= 2**n_qubits:
                        break
                        
                    # Convert option index to binary representation
                    binary = format(option_idx, f'0{n_qubits}b')
                    
                    # Setup the proper states to amplify
                    for i, bit in enumerate(binary):
                        if bit == '0':
                            qml.PauliX(wires=i)
                    
                    # Apply controlled rotation to amplify the desired option
                    if n_qubits > 1:
                        # Use multi-controlled rotation
                        qml.MultiControlledPhaseShift(
                            confidence * np.pi/2,
                            control_wires=list(range(n_qubits-1)),
                            wires=n_qubits-1
                        )
                    else:
                        # Single qubit case
                        qml.RY(confidence * np.pi, wires=0)
                    
                    # Restore to computational basis
                    for i, bit in enumerate(binary):
                        if bit == '0':
                            qml.PauliX(wires=i)
                
                # Return measurement
                return [qml.sample(wires=i) for i in range(n_qubits)]
            
            return multi_option_circuit
        except ImportError:
            logger.warning("PennyLane not available for multi-option decision circuit")
            return None
    
    def _create_prioritization_circuit(self):
        """Create a quantum circuit for prioritizing tasks or goals"""
        try:
            import pennylane as qml
            import numpy as np
            
            # Define the prioritization circuit
            def prioritization_circuit(params):
                # Get the priorities and constraints
                priorities = params.get("priorities", [0.5, 0.5, 0.5, 0.5])
                constraints = params.get("constraints", [[0, 1, 0.2], [2, 3, 0.3]])
                
                n_qubits = min(len(priorities), self.n_qubits)
                
                # Initialize in superposition
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
                
                # Apply priority rotations to each qubit
                for i, priority in enumerate(priorities[:n_qubits]):
                    qml.RY(priority * np.pi, wires=i)
                
                # Apply constraint entanglement
                for constraint in constraints:
                    if len(constraint) >= 3:
                        q1, q2, strength = constraint[0], constraint[1], constraint[2]
                        if q1 < n_qubits and q2 < n_qubits:
                            qml.CNOT(wires=[q1, q2])
                            qml.RZ(strength * np.pi, wires=q2)
                            qml.CNOT(wires=[q1, q2])
                
                # Return measurements
                return [qml.sample(wires=i) for i in range(n_qubits)]
            
            return prioritization_circuit
        except ImportError:
            logger.warning("PennyLane not available for prioritization circuit")
            return None
    
    def _create_image_classifier_circuit(self):
        """Create a simple quantum circuit for image feature classification"""
        try:
            import pennylane as qml
            import numpy as np
            
            # Define the image classifier circuit
            def image_classifier_circuit(params):
                # Get the image features
                features = params.get("features", [0.5, 0.5, 0.5, 0.5])
                n_qubits = min(len(features), self.n_qubits)
                
                # Data encoding
                for i, feature in enumerate(features[:n_qubits]):
                    qml.RY(feature * np.pi, wires=i)
                
                # Entanglement layer
                for i in range(n_qubits-1):
                    qml.CNOT(wires=[i, i+1])
                
                # Measurement layer
                return [qml.sample(wires=i) for i in range(n_qubits)]
            
            return image_classifier_circuit
        except ImportError:
            logger.warning("PennyLane not available for image classifier circuit")
            return None
    
    def _create_text_relevance_circuit(self):
        """Create a quantum circuit for text relevance assessment"""
        try:
            import pennylane as qml
            import numpy as np
            
            # Define the text relevance circuit
            def text_relevance_circuit(params):
                # Get text features
                query_features = params.get("query_features", [0.5, 0.5])
                text_features = params.get("text_features", [0.5, 0.5])
                
                # Use 4 qubits: 2 for query, 2 for text
                n_qubits = min(4, self.n_qubits)
                
                # Data encoding
                for i, feature in enumerate(query_features[:n_qubits//2]):
                    qml.RY(feature * np.pi, wires=i)
                
                for i, feature in enumerate(text_features[:n_qubits//2]):
                    qml.RY(feature * np.pi, wires=i + n_qubits//2)
                
                # Entanglement between query and text
                for i in range(n_qubits//2):
                    qml.CNOT(wires=[i, i + n_qubits//2])
                
                # Final relevance rotation
                qml.RY(np.pi/4, wires=0)
                
                # Return measurement
                return [qml.sample(wires=i) for i in range(n_qubits)]
            
            return text_relevance_circuit
        except ImportError:
            logger.warning("PennyLane not available for text relevance circuit")
            return None
    
    def _create_element_detector_circuit(self):
        """Create a quantum circuit for UI element detection"""
        try:
            import pennylane as qml
            import numpy as np
            
            # Define the element detector circuit
            def element_detector_circuit(params):
                # Get element features
                element_features = params.get("element_features", [0.5, 0.5, 0.5, 0.5])
                element_types = params.get("element_types", [0.3, 0.6, 0.1])  # Probabilities for different types
                
                n_features = min(len(element_features), self.n_qubits - 2)
                n_types = min(len(element_types), 2)  # Need logarithmic encoding
                
                n_qubits = n_features + n_types
                n_qubits = min(n_qubits, self.n_qubits)
                
                # Feature encoding
                for i, feature in enumerate(element_features[:n_features]):
                    qml.RY(feature * np.pi, wires=i)
                
                # Type encoding (binary encoding of types)
                for i, prob in enumerate(element_types[:n_types]):
                    qml.RY(prob * np.pi, wires=n_features + i)
                
                # Entanglement between features and type
                for i in range(n_features):
                    qml.CNOT(wires=[i, n_features])
                
                # Return measurement
                return [qml.sample(wires=i) for i in range(n_qubits)]
            
            return element_detector_circuit
        except ImportError:
            logger.warning("PennyLane not available for element detector circuit")
            return None
    
    def _create_reinforcement_learning_circuit(self):
        """Create a quantum circuit for reinforcement learning"""
        try:
            import pennylane as qml
            import numpy as np
            
            # Define the reinforcement learning circuit
            def reinforcement_learning_circuit(params):
                # Get state and action spaces
                state_features = params.get("state_features", [0.5, 0.5])
                actions = params.get("actions", [0.3, 0.7])
                reward = params.get("reward", 0.5)
                
                n_state = min(len(state_features), self.n_qubits - 1)
                n_qubits = n_state + 1  # +1 for action
                
                # State encoding
                for i, feature in enumerate(state_features[:n_state]):
                    qml.RY(feature * np.pi, wires=i)
                
                # Current policy
                qml.RY(actions[0] * np.pi, wires=n_state)
                
                # Entanglement between state and action
                for i in range(n_state):
                    qml.CNOT(wires=[i, n_state])
                
                # Reward feedback
                qml.RZ(reward * np.pi, wires=n_state)
                
                # Return measurement (action probabilities)
                return qml.sample(wires=n_state)
            
            return reinforcement_learning_circuit
        except ImportError:
            logger.warning("PennyLane not available for reinforcement learning circuit")
            return None
    
    def _create_memory_optimization_circuit(self):
        """Create a quantum circuit for memory optimization"""
        try:
            import pennylane as qml
            import numpy as np
            
            # Define the memory optimization circuit
            def memory_optimization_circuit(params):
                # Get memory items and importance scores
                memory_items = params.get("memory_items", [0.1, 0.5, 0.8, 0.3])
                capacity = params.get("capacity", 2)  # Number of items to keep
                
                n_qubits = min(len(memory_items), self.n_qubits)
                
                # Initialize to superposition
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
                
                # Encode importance scores
                for i, importance in enumerate(memory_items[:n_qubits]):
                    qml.RY(importance * np.pi, wires=i)
                
                # Apply capacity constraint (optimization)
                if self.decision_enhancement_level >= 2:
                    # Create entanglement based on capacity
                    for i in range(n_qubits):
                        for j in range(i+1, n_qubits):
                            qml.CNOT(wires=[i, j])
                            qml.RZ(-0.1 * np.pi, wires=j)  # Small penalty for selecting too many
                            qml.CNOT(wires=[i, j])
                
                # Return measurements (which items to keep)
                return [qml.sample(wires=i) for i in range(n_qubits)]
            
            return memory_optimization_circuit
        except ImportError:
            logger.warning("PennyLane not available for memory optimization circuit")
            return None
    
    async def enhance_decision(self, 
                             options: List[Dict[str, Any]], 
                             context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Enhance agent decision making with quantum computing
        
        Args:
            options: List of decision options with confidence scores
            context: Current decision context
            
        Returns:
            Enhanced list of options with quantum-adjusted confidence scores
        """
        if not self.initialized or not self.quantum_engine or len(options) == 0:
            return options
        
        try:
            start_time = time.time()
            
            # Choose the appropriate quantum circuit based on number of options
            if len(options) == 2:
                # Binary decision
                circuit_func = self.decision_circuits.get("binary")
                if not circuit_func:
                    return options
                
                # Prepare parameters
                params = {
                    "confidence_a": options[0].get("confidence", 0.5),
                    "confidence_b": options[1].get("confidence", 0.5),
                    "bias": context.get("bias", 0.0)
                }
                
                # Run quantum circuit
                result = await self.quantum_engine.run_quantum_circuit(
                    circuit_func, params, shots=100
                )
                
                if not result.get("success", False):
                    logger.warning(f"Quantum decision enhancement failed: {result.get('error', 'Unknown error')}")
                    return options
                
                # Process results
                quantum_results = result.get("results", {})
                measurements = quantum_results.get("counts", {"0": 50, "1": 50})
                
                # Convert to confidence adjustments
                total_shots = sum(measurements.values())
                prob_0 = measurements.get("0", 0) / total_shots
                
                # Update option confidences
                options[0]["confidence"] = prob_0
                options[0]["quantum_enhanced"] = True
                options[1]["confidence"] = 1.0 - prob_0
                options[1]["quantum_enhanced"] = True
                
            else:
                # Multi-option decision
                circuit_func = self.decision_circuits.get("multi_option")
                if not circuit_func:
                    return options
                
                # Determine number of qubits needed
                n_options = len(options)
                n_qubits = max(1, int(np.ceil(np.log2(n_options))))
                
                # Prepare parameters
                confidences = [opt.get("confidence", 0.5) for opt in options]
                params = {
                    "num_options": n_options,
                    "confidences": confidences
                }
                
                # Run quantum circuit
                result = await self.quantum_engine.run_quantum_circuit(
                    circuit_func, params, shots=100
                )
                
                if not result.get("success", False):
                    logger.warning(f"Quantum decision enhancement failed: {result.get('error', 'Unknown error')}")
                    return options
                
                # Process results
                quantum_results = result.get("results", {})
                measurements = quantum_results.get("counts", {})
                
                # Default measurements if empty
                if not measurements:
                    # Create equal distribution
                    measurements = {format(i, f'0{n_qubits}b'): 100//n_options for i in range(n_options)}
                
                # Convert to confidence adjustments
                total_shots = sum(measurements.values())
                
                # Update option confidences for options that fit in quantum register
                for i, option in enumerate(options[:2**n_qubits]):
                    binary = format(i, f'0{n_qubits}b')
                    prob = measurements.get(binary, 0) / total_shots
                    option["confidence"] = prob
                    option["quantum_enhanced"] = True
            
            # Track execution
            end_time = time.time()
            self.execution_history.append({
                "type": "decision_enhancement",
                "timestamp": datetime.now().isoformat(),
                "n_options": len(options),
                "processing_time": end_time - start_time,
                "quantum_utilized": result.get("metadata", {}).get("hardware_used", False)
            })
            
            return options
        
        except Exception as e:
            logger.error(f"Error in quantum decision enhancement: {str(e)}")
            return options
    
    async def enhance_perception(self, 
                               perception_data: Dict[str, Any], 
                               task: str) -> Dict[str, Any]:
        """
        Enhance agent perception with quantum processing
        
        Args:
            perception_data: Raw perception data from agent
            task: The specific perception task (image, text, element)
            
        Returns:
            Enhanced perception data with quantum-processed features
        """
        if not self.initialized or not self.quantum_engine:
            return perception_data
        
        try:
            start_time = time.time()
            
            # Choose the appropriate quantum circuit based on task
            if task == "image":
                circuit_func = self.perception_circuits.get("image_classifier")
                if not circuit_func:
                    return perception_data
                
                # Extract features from image
                features = perception_data.get("features", [0.5, 0.5, 0.5, 0.5])
                
                # Prepare parameters
                params = {"features": features}
                
                # Run quantum circuit
                result = await self.quantum_engine.run_quantum_circuit(
                    circuit_func, params, shots=100
                )
                
                if not result.get("success", False):
                    logger.warning(f"Quantum perception enhancement failed: {result.get('error', 'Unknown error')}")
                    return perception_data
                
                # Process results
                quantum_results = result.get("results", {})
                
                # Update perception data with quantum-enhanced classification
                perception_data["quantum_enhanced"] = True
                perception_data["quantum_classification"] = quantum_results
                
            elif task == "text":
                circuit_func = self.perception_circuits.get("text_relevance")
                if not circuit_func:
                    return perception_data
                
                # Extract features
                query_features = perception_data.get("query_features", [0.5, 0.5])
                text_features = perception_data.get("text_features", [0.5, 0.5])
                
                # Prepare parameters
                params = {
                    "query_features": query_features,
                    "text_features": text_features
                }
                
                # Run quantum circuit
                result = await self.quantum_engine.run_quantum_circuit(
                    circuit_func, params, shots=100
                )
                
                if not result.get("success", False):
                    logger.warning(f"Quantum perception enhancement failed: {result.get('error', 'Unknown error')}")
                    return perception_data
                
                # Process results
                quantum_results = result.get("results", {})
                
                # Update perception data with quantum-enhanced relevance
                perception_data["quantum_enhanced"] = True
                perception_data["quantum_relevance"] = quantum_results
                
            elif task == "element":
                circuit_func = self.perception_circuits.get("element_detector")
                if not circuit_func:
                    return perception_data
                
                # Extract features
                element_features = perception_data.get("element_features", [0.5, 0.5, 0.5, 0.5])
                element_types = perception_data.get("element_types", [0.3, 0.6, 0.1])
                
                # Prepare parameters
                params = {
                    "element_features": element_features,
                    "element_types": element_types
                }
                
                # Run quantum circuit
                result = await self.quantum_engine.run_quantum_circuit(
                    circuit_func, params, shots=100
                )
                
                if not result.get("success", False):
                    logger.warning(f"Quantum perception enhancement failed: {result.get('error', 'Unknown error')}")
                    return perception_data
                
                # Process results
                quantum_results = result.get("results", {})
                
                # Update perception data with quantum-enhanced detection
                perception_data["quantum_enhanced"] = True
                perception_data["quantum_detection"] = quantum_results
            
            # Track execution
            end_time = time.time()
            self.execution_history.append({
                "type": "perception_enhancement",
                "timestamp": datetime.now().isoformat(),
                "task": task,
                "processing_time": end_time - start_time,
                "quantum_utilized": result.get("metadata", {}).get("hardware_used", False)
            })
            
            return perception_data
        
        except Exception as e:
            logger.error(f"Error in quantum perception enhancement: {str(e)}")
            return perception_data
    
    async def optimize_goals(self, goals: List[Dict[str, Any]], constraints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize agent goals using quantum optimization
        
        Args:
            goals: List of agent goals with priority scores
            constraints: List of constraints between goals
            
        Returns:
            Optimized list of goals with quantum-adjusted priorities
        """
        if not self.initialized or not self.quantum_engine or len(goals) == 0:
            return goals
        
        try:
            start_time = time.time()
            
            # Use the prioritization circuit
            circuit_func = self.decision_circuits.get("prioritization")
            if not circuit_func:
                return goals
            
            # Extract priorities
            priorities = [g.get("priority", 5) / 10.0 for g in goals]  # Normalize to 0-1
            
            # Format constraints
            formatted_constraints = []
            for constraint in constraints:
                if "goal_a" in constraint and "goal_b" in constraint and "strength" in constraint:
                    goal_a_idx = next((i for i, g in enumerate(goals) if g.get("id") == constraint["goal_a"]), None)
                    goal_b_idx = next((i for i, g in enumerate(goals) if g.get("id") == constraint["goal_b"]), None)
                    
                    if goal_a_idx is not None and goal_b_idx is not None:
                        formatted_constraints.append([
                            goal_a_idx,
                            goal_b_idx,
                            constraint["strength"]
                        ])
            
            # Prepare parameters
            params = {
                "priorities": priorities,
                "constraints": formatted_constraints
            }
            
            # Run quantum circuit
            result = await self.quantum_engine.run_quantum_circuit(
                circuit_func, params, shots=100
            )
            
            if not result.get("success", False):
                logger.warning(f"Quantum goal optimization failed: {result.get('error', 'Unknown error')}")
                return goals
            
            # Process results
            quantum_results = result.get("results", {})
            measurements = quantum_results.get("counts", {})
            
            if not measurements:
                return goals
                
            # Calculate new priorities based on measurement frequencies
            total_shots = sum(measurements.values())
            
            # Update goal priorities
            for i, goal in enumerate(goals):
                if i >= self.n_qubits:
                    break
                    
                # Count frequency of 1 for this qubit position
                freq_1 = 0
                for bitstring, count in measurements.items():
                    if len(bitstring) > i and bitstring[i] == "1":
                        freq_1 += count
                
                # Calculate new priority (1-10 scale)
                new_priority = 1 + 9 * (freq_1 / total_shots)
                goal["priority"] = new_priority
                goal["quantum_enhanced"] = True
            
            # Track execution
            end_time = time.time()
            self.execution_history.append({
                "type": "goal_optimization",
                "timestamp": datetime.now().isoformat(),
                "n_goals": len(goals),
                "processing_time": end_time - start_time,
                "quantum_utilized": result.get("metadata", {}).get("hardware_used", False)
            })
            
            return goals
        
        except Exception as e:
            logger.error(f"Error in quantum goal optimization: {str(e)}")
            return goals
    
    async def enhance_memory(self, memory_items: List[Dict[str, Any]], capacity: int) -> List[Dict[str, Any]]:
        """
        Enhance memory management with quantum optimization
        
        Args:
            memory_items: List of memory items with importance scores
            capacity: Maximum number of items to keep
            
        Returns:
            Optimized list of memory items
        """
        if not self.initialized or not self.quantum_engine or len(memory_items) == 0:
            return memory_items[:capacity]
        
        try:
            start_time = time.time()
            
            # Use the memory optimization circuit
            circuit_func = self.learning_circuits.get("memory_optimization")
            if not circuit_func:
                # Fall back to classical sorting
                return sorted(memory_items, key=lambda x: x.get("importance", 0), reverse=True)[:capacity]
            
            # Extract importance scores
            importance_scores = [item.get("importance", 0.5) for item in memory_items]
            
            # Prepare parameters
            params = {
                "memory_items": importance_scores,
                "capacity": capacity
            }
            
            # Run quantum circuit
            result = await self.quantum_engine.run_quantum_circuit(
                circuit_func, params, shots=100
            )
            
            if not result.get("success", False):
                logger.warning(f"Quantum memory enhancement failed: {result.get('error', 'Unknown error')}")
                return sorted(memory_items, key=lambda x: x.get("importance", 0), reverse=True)[:capacity]
            
            # Process results
            quantum_results = result.get("results", {})
            measurements = quantum_results.get("counts", {})
            
            if not measurements:
                return sorted(memory_items, key=lambda x: x.get("importance", 0), reverse=True)[:capacity]
            
            # Calculate selection probabilities for each item
            total_shots = sum(measurements.values())
            selection_probs = [0] * len(memory_items)
            
            for i in range(min(len(memory_items), self.n_qubits)):
                freq_1 = 0
                for bitstring, count in measurements.items():
                    if len(bitstring) > i and bitstring[i] == "1":
                        freq_1 += count
                
                selection_probs[i] = freq_1 / total_shots
            
            # Tag items with selection probability
            for i, item in enumerate(memory_items):
                if i < len(selection_probs):
                    item["selection_probability"] = selection_probs[i]
                    item["quantum_enhanced"] = True
                else:
                    item["selection_probability"] = 0.0
            
            # Sort by selection probability and return top 'capacity' items
            selected_items = sorted(memory_items, key=lambda x: x.get("selection_probability", 0), reverse=True)[:capacity]
            
            # Track execution
            end_time = time.time()
            self.execution_history.append({
                "type": "memory_enhancement",
                "timestamp": datetime.now().isoformat(),
                "n_items": len(memory_items),
                "capacity": capacity,
                "processing_time": end_time - start_time,
                "quantum_utilized": result.get("metadata", {}).get("hardware_used", False)
            })
            
            return selected_items
        
        except Exception as e:
            logger.error(f"Error in quantum memory enhancement: {str(e)}")
            # Fall back to classical sorting
            return sorted(memory_items, key=lambda x: x.get("importance", 0), reverse=True)[:capacity]
    
    def get_bridge_status(self) -> Dict[str, Any]:
        """
        Get the current status of the quantum agent bridge
        
        Returns:
            Dict with bridge status information
        """
        # Get engine status if available
        engine_status = self.quantum_engine.get_engine_status() if self.quantum_engine else {}
        
        # Calculate execution statistics
        avg_time = 0.0
        quantum_utilization = 0.0
        
        if self.execution_history:
            times = [e.get("processing_time", 0) for e in self.execution_history]
            avg_time = sum(times) / len(times)
            
            quantum_count = sum(1 for e in self.execution_history if e.get("quantum_utilized", False))
            quantum_utilization = quantum_count / len(self.execution_history) if self.execution_history else 0
        
        return {
            "initialized": self.initialized,
            "n_qubits": self.n_qubits,
            "decision_enhancement_level": self.decision_enhancement_level,
            "quantum_engine": engine_status,
            "precompiled_circuits": {
                "decision": list(self.decision_circuits.keys()),
                "perception": list(self.perception_circuits.keys()),
                "learning": list(self.learning_circuits.keys())
            },
            "execution_stats": {
                "total_executions": len(self.execution_history),
                "avg_processing_time": avg_time,
                "quantum_utilization": quantum_utilization
            }
        }