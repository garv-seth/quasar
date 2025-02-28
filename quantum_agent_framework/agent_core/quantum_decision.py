"""
Quantum Decision System for Agent Framework

This module provides quantum-enhanced decision making capabilities for agents,
offering significant advantages over classical decision systems:
1. Superposition-based multi-criteria evaluation
2. Entanglement-based interdependent decision analysis
3. Quantum amplitude amplification for option prioritization

Author: Quantum Agent Framework Team
"""

import asyncio
import logging
import time
import math
import json
from typing import Dict, List, Any, Optional, Tuple, Callable
import os
from dataclasses import dataclass

try:
    import numpy as np
    import pennylane as qml
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False

try:
    import anthropic
    from openai import AsyncOpenAI
    AI_MODELS_AVAILABLE = True
except ImportError:
    AI_MODELS_AVAILABLE = False


@dataclass
class Decision:
    """
    Represents a decision with options, criteria, constraints, and the selected option.
    """
    decision_id: str
    context: Dict[str, Any]
    options: List[Dict[str, Any]]
    criteria: List[Dict[str, Any]]
    constraints: List[Dict[str, Any]]
    selected_option: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    processing_time: float = 0.0
    method_used: str = "classical"
    explanation: str = ""
    

@dataclass
class DecisionRequest:
    """
    Request for making a decision with context, options, criteria, and constraints.
    """
    context: Dict[str, Any]
    options: List[Dict[str, Any]]
    criteria: List[Dict[str, Any]]
    constraints: List[Dict[str, Any]]
    decision_type: str = "general"
    require_explanation: bool = True


class QuantumDecisionSystem:
    """
    Quantum-enhanced decision making system for autonomous agents.
    
    This system leverages quantum computing to enhance:
    1. Multi-criteria evaluation
    2. Handling of uncertainty and ambiguity
    3. Exploration of interconnected decision spaces
    4. Prioritization and ranking
    """
    
    def __init__(self, n_qubits: int = 8, use_quantum: bool = True, use_ai: bool = True):
        """
        Initialize the quantum decision system.
        
        Args:
            n_qubits: Number of qubits to use
            use_quantum: Whether to use quantum acceleration
            use_ai: Whether to use AI for explanation generation
        """
        self.n_qubits = n_qubits
        self.use_quantum = use_quantum and QUANTUM_AVAILABLE
        self.use_ai = use_ai and AI_MODELS_AVAILABLE
        
        # Quantum device
        self.device = None
        self.initialized = False
        
        # AI clients
        self.anthropic_client = None
        self.openai_client = None
        
        # Performance metrics
        self.metrics = {
            "decisions_made": 0,
            "quantum_decisions": 0,
            "classical_decisions": 0,
            "avg_decision_time": 0,
            "total_decision_time": 0
        }
        
        # Initialize components
        if self.use_quantum:
            self._initialize_quantum()
            
        if self.use_ai:
            self._initialize_ai_clients()
    
    def _initialize_quantum(self):
        """Initialize quantum computing device."""
        try:
            # Initialize quantum device
            self.device = qml.device("default.qubit", wires=self.n_qubits)
            self.initialized = True
            logging.info(f"Quantum decision system initialized with {self.n_qubits} qubits")
        except Exception as e:
            logging.error(f"Failed to initialize quantum device: {str(e)}")
            self.use_quantum = False
            self.initialized = False
    
    def _initialize_ai_clients(self):
        """Initialize AI clients for explanation generation."""
        try:
            # Initialize OpenAI client
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if openai_api_key:
                self.openai_client = AsyncOpenAI(api_key=openai_api_key)
                logging.info("OpenAI client initialized")
            
            # Initialize Anthropic client
            anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
            if anthropic_api_key:
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
                logging.info("Anthropic client initialized")
                
        except Exception as e:
            logging.error(f"Failed to initialize AI clients: {str(e)}")
            self.use_ai = False
    
    async def make_decision(self, request: DecisionRequest) -> Decision:
        """
        Make a decision based on the provided options, criteria, and constraints.
        
        Args:
            request: Decision request with context, options, criteria, and constraints
            
        Returns:
            Decision with the selected option and explanation
        """
        start_time = time.time()
        
        # Create unique decision ID
        decision_id = f"decision_{int(start_time)}_{hash(str(request.options)[:100])}"
        
        # Choose decision method based on availability and complexity
        if self.use_quantum and self.initialized and self._is_quantum_beneficial(request):
            decision = await self._quantum_decision(decision_id, request)
            self.metrics["quantum_decisions"] += 1
        else:
            decision = await self._classical_decision(decision_id, request)
            self.metrics["classical_decisions"] += 1
        
        # Generate explanation if requested
        if request.require_explanation and not decision.explanation:
            decision.explanation = await self._generate_explanation(decision)
        
        # Update processing time
        decision.processing_time = time.time() - start_time
        
        # Update metrics
        self.metrics["decisions_made"] += 1
        self.metrics["total_decision_time"] += decision.processing_time
        self.metrics["avg_decision_time"] = (
            self.metrics["total_decision_time"] / self.metrics["decisions_made"]
        )
        
        return decision
    
    def _is_quantum_beneficial(self, request: DecisionRequest) -> bool:
        """
        Determine if quantum decision making would be beneficial for this request.
        
        Args:
            request: Decision request
            
        Returns:
            True if quantum processing would be beneficial
        """
        # Check if we have enough options and criteria for quantum to be beneficial
        num_options = len(request.options)
        num_criteria = len(request.criteria)
        
        # Quantum is beneficial when:
        # 1. We have multiple interdependent criteria (>= 3)
        # 2. We have multiple options to evaluate (>= 4)
        # 3. Decision type is appropriate for quantum processing
        quantum_beneficial = (
            num_criteria >= 3 and
            num_options >= 4 and
            request.decision_type in ["general", "multi_criteria", "resource_allocation", "risk_assessment"]
        )
        
        return quantum_beneficial
    
    async def _quantum_decision(self, decision_id: str, request: DecisionRequest) -> Decision:
        """
        Make a decision using quantum computing.
        
        This uses a quantum circuit to:
        1. Encode options and criteria into a quantum state
        2. Apply constraints as phase operations
        3. Amplify optimal options using amplitude amplification
        4. Measure to obtain the best option
        
        Args:
            decision_id: Unique decision ID
            request: Decision request
            
        Returns:
            Decision with the selected option
        """
        options = request.options
        criteria = request.criteria
        constraints = request.constraints
        
        # Convert options to numerical features
        option_features = self._extract_option_features(options, criteria)
        
        # Convert criteria to weights
        criteria_weights = self._extract_criteria_weights(criteria)
        
        # Convert constraints to penalty terms
        constraint_operators = self._extract_constraint_operators(constraints, options)
        
        # Define the quantum circuit for decision making
        @qml.qnode(self.device)
        def decision_circuit(features, weights, constraint_ops):
            # Number of options and criteria
            num_options = len(features)
            num_criteria = len(weights)
            
            # We'll use log2(num_options) qubits to encode options
            option_qubits = min(int(np.ceil(np.log2(num_options))), self.n_qubits - 1)
            
            # Extra qubit for amplitude amplification
            ancilla_qubit = self.n_qubits - 1
            
            # Initialize in superposition of all options
            for q in range(option_qubits):
                qml.Hadamard(wires=q)
            
            # Encode option features and criteria weights
            for i, option_feature in enumerate(features):
                # Binary encoding of option index
                binary_i = format(i, f'0{option_qubits}b')
                
                # Only apply if we have enough qubits to encode this option
                if i < 2**option_qubits:
                    # Controlled rotation based on option's criteria values
                    for c, (feature_val, weight) in enumerate(zip(option_feature, weights)):
                        # Skip if we don't have enough criteria or weight is zero
                        if c >= num_criteria or weight == 0:
                            continue
                        
                        # Determine control wires and values for this option
                        controls = []
                        for q, bit in enumerate(binary_i):
                            if bit == '1':
                                controls.append((q, 1))  # Control on |1⟩
                            else:
                                controls.append((q, 0))  # Control on |0⟩
                        
                        # Apply controlled rotation to the ancilla qubit
                        # Rotation angle depends on feature value and criteria weight
                        angle = feature_val * weight * np.pi
                        
                        # Use multi-control gate with both |0⟩ and |1⟩ controls
                        with qml.ctrl(controls):
                            qml.RY(angle, wires=ancilla_qubit)
            
            # Apply constraint penalties
            for constraint_op in constraint_ops:
                op_type = constraint_op.get("type")
                option_idx = constraint_op.get("option_idx")
                penalty = constraint_op.get("penalty", 0.0)
                
                # Skip if penalty is zero
                if penalty == 0:
                    continue
                
                # Convert option index to binary
                binary_idx = format(option_idx, f'0{option_qubits}b')
                
                # Apply phase based on constraint type
                controls = []
                for q, bit in enumerate(binary_idx):
                    if bit == '1':
                        controls.append((q, 1))
                    else:
                        controls.append((q, 0))
                
                # Apply phase rotation based on constraint penalty
                if op_type == "penalty":
                    with qml.ctrl(controls):
                        qml.PhaseShift(-penalty * np.pi, wires=ancilla_qubit)
            
            # Amplify the optimal options
            # This is a simplified version of amplitude amplification
            # In a full implementation, you would iterate this process
            
            # First apply Hadamard to ancilla
            qml.Hadamard(wires=ancilla_qubit)
            
            # Then apply phase flip to |0⟩ state of ancilla
            qml.PhaseShift(np.pi, wires=ancilla_qubit)
            
            # Apply another Hadamard to ancilla
            qml.Hadamard(wires=ancilla_qubit)
            
            # Measure option qubits
            return [qml.probs(wires=i) for i in range(option_qubits)]
        
        # Execute quantum circuit
        try:
            result = decision_circuit(option_features, criteria_weights, constraint_operators)
            
            # Process results to get the selected option
            option_probs = self._process_quantum_results(result, len(options))
            
            # Get the option with highest probability
            max_prob_idx = np.argmax(option_probs)
            confidence = float(option_probs[max_prob_idx])
            
            # Create decision with selected option
            selected_option = options[max_prob_idx] if max_prob_idx < len(options) else options[0]
            
            decision = Decision(
                decision_id=decision_id,
                context=request.context,
                options=options,
                criteria=criteria,
                constraints=constraints,
                selected_option=selected_option,
                confidence=confidence,
                method_used="quantum"
            )
            
            return decision
            
        except Exception as e:
            logging.error(f"Error in quantum decision making: {str(e)}")
            # Fall back to classical method
            return await self._classical_decision(decision_id, request)
    
    async def _classical_decision(self, decision_id: str, request: DecisionRequest) -> Decision:
        """
        Make a decision using classical computing.
        
        This uses weighted sum model for multi-criteria decision making:
        1. Calculate weighted scores for each option based on criteria
        2. Apply constraint penalties
        3. Select option with highest score
        
        Args:
            decision_id: Unique decision ID
            request: Decision request
            
        Returns:
            Decision with the selected option
        """
        options = request.options
        criteria = request.criteria
        constraints = request.constraints
        
        # Convert options to numerical features
        option_features = self._extract_option_features(options, criteria)
        
        # Convert criteria to weights
        criteria_weights = self._extract_criteria_weights(criteria)
        
        # Calculate weighted scores for each option
        option_scores = []
        for i, option in enumerate(options):
            # Skip if we don't have features for this option
            if i >= len(option_features):
                option_scores.append(0.0)
                continue
            
            # Calculate weighted sum for criteria
            features = option_features[i]
            weighted_sum = sum(
                f * w for f, w in zip(features, criteria_weights) 
                if f is not None and w is not None
            )
            
            # Apply constraint penalties
            penalty = 0.0
            for constraint in constraints:
                # Skip constraints not applicable to this option
                if not self._is_constraint_applicable(constraint, option):
                    continue
                
                # Calculate penalty based on constraint type and severity
                constraint_type = constraint.get("type", "")
                severity = constraint.get("severity", 1.0)
                
                if constraint_type == "must_have" and not self._option_satisfies_constraint(constraint, option):
                    penalty += 10.0 * severity  # Large penalty for must-have constraints
                elif constraint_type == "nice_to_have" and not self._option_satisfies_constraint(constraint, option):
                    penalty += 0.5 * severity  # Smaller penalty for nice-to-have
                elif constraint_type == "avoid" and self._option_satisfies_constraint(constraint, option):
                    penalty += 2.0 * severity  # Medium penalty for avoid constraints
            
            # Final score with penalty
            score = weighted_sum - penalty
            option_scores.append(score)
        
        # Get the option with highest score
        if not option_scores:
            # No valid scores, return first option with 0 confidence
            selected_option = options[0] if options else None
            confidence = 0.0
        else:
            max_score_idx = np.argmax(option_scores)
            selected_option = options[max_score_idx]
            
            # Calculate confidence based on score distribution
            max_score = option_scores[max_score_idx]
            # Avoid division by zero
            if max_score == 0:
                confidence = 1.0 / len(options)
            else:
                # Normalize scores and use max as confidence
                # Add small epsilon to avoid division by zero
                sum_scores = sum(max(0, s) for s in option_scores) + 1e-6
                confidence = max(0, max_score) / sum_scores
        
        # Create decision with selected option
        decision = Decision(
            decision_id=decision_id,
            context=request.context,
            options=options,
            criteria=criteria,
            constraints=constraints,
            selected_option=selected_option,
            confidence=float(confidence),
            method_used="classical"
        )
        
        return decision
    
    def _extract_option_features(self, options: List[Dict[str, Any]], criteria: List[Dict[str, Any]]) -> List[List[float]]:
        """
        Extract numerical features from options based on criteria.
        
        Args:
            options: List of option dictionaries
            criteria: List of criteria dictionaries
            
        Returns:
            List of feature lists for each option
        """
        features = []
        
        for option in options:
            option_features = []
            
            for criterion in criteria:
                criterion_id = criterion.get("id", "")
                criterion_key = criterion.get("key", criterion_id)
                
                # Extract feature value from option
                feature_value = option.get(criterion_key)
                
                # Convert to numerical value
                if feature_value is None:
                    # Missing value, use default or 0
                    option_features.append(0.0)
                elif isinstance(feature_value, (int, float)):
                    # Already numerical
                    option_features.append(float(feature_value))
                elif isinstance(feature_value, bool):
                    # Boolean to 0.0 or 1.0
                    option_features.append(1.0 if feature_value else 0.0)
                elif isinstance(feature_value, str):
                    # Try to convert string to float, or use 0
                    try:
                        option_features.append(float(feature_value))
                    except ValueError:
                        # Non-numeric string, use default score from criterion if available
                        default_score = criterion.get("default_score", 0.5)
                        option_features.append(default_score)
                else:
                    # Unknown type, use default
                    option_features.append(0.0)
            
            features.append(option_features)
        
        return features
    
    def _extract_criteria_weights(self, criteria: List[Dict[str, Any]]) -> List[float]:
        """
        Extract weights from criteria.
        
        Args:
            criteria: List of criteria dictionaries
            
        Returns:
            List of weights
        """
        weights = []
        
        for criterion in criteria:
            # Extract weight, default to 1.0
            weight = criterion.get("weight", 1.0)
            
            # Normalize weight
            if weight < 0:
                # Negative weight for minimization criteria
                weights.append(-weight / 10.0)  # Normalize to reasonable range
            else:
                # Positive weight for maximization criteria
                weights.append(weight / 10.0)  # Normalize to reasonable range
        
        return weights
    
    def _extract_constraint_operators(self, constraints: List[Dict[str, Any]], options: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract constraint operators as quantum circuit parameters.
        
        Args:
            constraints: List of constraint dictionaries
            options: List of option dictionaries
            
        Returns:
            List of constraint operators
        """
        operators = []
        
        for constraint in constraints:
            constraint_type = constraint.get("type", "")
            severity = constraint.get("severity", 1.0)
            
            # Process each option for this constraint
            for i, option in enumerate(options):
                satisfies = self._option_satisfies_constraint(constraint, option)
                
                # Determine penalty based on constraint type and satisfaction
                penalty = 0.0
                
                if constraint_type == "must_have" and not satisfies:
                    penalty = severity  # Full penalty for must-have not satisfied
                elif constraint_type == "avoid" and satisfies:
                    penalty = 0.8 * severity  # Strong penalty for avoided feature
                elif constraint_type == "nice_to_have" and not satisfies:
                    penalty = 0.3 * severity  # Small penalty for missing nice-to-have
                
                # Add operator only if penalty is non-zero
                if penalty > 0:
                    operators.append({
                        "type": "penalty",
                        "option_idx": i,
                        "penalty": min(penalty, 1.0)  # Clamp to [0,1] range
                    })
        
        return operators
    
    def _is_constraint_applicable(self, constraint: Dict[str, Any], option: Dict[str, Any]) -> bool:
        """
        Check if a constraint is applicable to an option.
        
        Args:
            constraint: Constraint dictionary
            option: Option dictionary
            
        Returns:
            True if constraint is applicable to option
        """
        # Check if constraint has a condition
        condition = constraint.get("condition", {})
        if not condition:
            return True  # No condition, always applicable
        
        # Check if condition attribute exists in option
        attr = condition.get("attribute", "")
        if attr not in option:
            return False  # Attribute not present, not applicable
        
        return True
    
    def _option_satisfies_constraint(self, constraint: Dict[str, Any], option: Dict[str, Any]) -> bool:
        """
        Check if an option satisfies a constraint.
        
        Args:
            constraint: Constraint dictionary
            option: Option dictionary
            
        Returns:
            True if option satisfies constraint
        """
        # Extract constraint parameters
        condition = constraint.get("condition", {})
        if not condition:
            return True  # No condition, always satisfied
        
        # Extract condition attributes
        attr = condition.get("attribute", "")
        operator = condition.get("operator", "eq")
        value = condition.get("value")
        
        # Check if attribute exists in option
        if attr not in option:
            return False
        
        # Get option attribute value
        option_value = option[attr]
        
        # Compare based on operator
        if operator == "eq":
            return option_value == value
        elif operator == "neq":
            return option_value != value
        elif operator == "gt":
            return option_value > value if isinstance(option_value, (int, float)) else False
        elif operator == "lt":
            return option_value < value if isinstance(option_value, (int, float)) else False
        elif operator == "gte":
            return option_value >= value if isinstance(option_value, (int, float)) else False
        elif operator == "lte":
            return option_value <= value if isinstance(option_value, (int, float)) else False
        elif operator == "contains":
            return value in option_value if isinstance(option_value, (str, list, tuple)) else False
        elif operator == "startswith":
            return option_value.startswith(value) if isinstance(option_value, str) else False
        elif operator == "endswith":
            return option_value.endswith(value) if isinstance(option_value, str) else False
        else:
            return False  # Unknown operator
    
    def _process_quantum_results(self, results, num_options: int) -> List[float]:
        """
        Process quantum circuit results to get option probabilities.
        
        Args:
            results: Quantum measurement results
            num_options: Number of options
            
        Returns:
            List of probabilities for each option
        """
        # Initialize probabilities for all options
        option_probs = [0.0] * num_options
        
        # Process probability distribution for each option
        try:
            # Combine individual qubit measurements into full state probabilities
            # This is a simplified approach - in a real implementation you would
            # need to handle correlations between qubits properly
            
            # Number of qubits used for options
            num_qubits = len(results)
            max_states = 2**num_qubits
            
            # Combine probabilities into option probabilities
            for state_idx in range(min(max_states, num_options)):
                # Get binary representation of state index
                binary = format(state_idx, f'0{num_qubits}b')
                
                # Calculate probability of this state
                # For simplicity, we multiply the probabilities of each qubit being in the expected state
                # In a real implementation, you would use the full state vector
                prob = 1.0
                for q, bit in enumerate(binary):
                    qubit_prob = results[q]
                    prob *= qubit_prob[int(bit)]
                
                # Assign to option
                if state_idx < num_options:
                    option_probs[state_idx] = prob
            
            # Normalize probabilities to sum to 1
            sum_probs = sum(option_probs)
            if sum_probs > 0:
                option_probs = [p / sum_probs for p in option_probs]
            
        except Exception as e:
            logging.error(f"Error processing quantum results: {str(e)}")
            # Fallback to uniform distribution
            if num_options > 0:
                option_probs = [1.0 / num_options] * num_options
        
        return option_probs
    
    async def _generate_explanation(self, decision: Decision) -> str:
        """
        Generate an explanation for the decision.
        
        Args:
            decision: Decision to explain
            
        Returns:
            Explanation string
        """
        if not self.use_ai or (self.openai_client is None and self.anthropic_client is None):
            # Generate a simple explanation without AI
            return self._generate_simple_explanation(decision)
        
        try:
            # Prepare context for explanation
            explanation_context = {
                "selected_option": decision.selected_option,
                "all_options": decision.options,
                "criteria": decision.criteria,
                "constraints": decision.constraints,
                "decision_context": decision.context,
                "confidence": decision.confidence,
                "method_used": decision.method_used
            }
            
            # Convert to JSON string
            context_str = json.dumps(explanation_context, indent=2)
            
            # Prepare prompt
            prompt = f"""
            You are a quantum-enhanced decision system explaining your selection process.
            
            You have made a decision based on the following context:
            {context_str}
            
            Please provide a clear, concise explanation of why the selected option was chosen:
            1. Explain how the option performs on key criteria
            2. Mention any constraints that influenced the decision
            3. Note the confidence level
            4. If quantum computing was used, briefly explain how it benefited the decision process
            
            Explanation:
            """
            
            # Try OpenAI first if available
            if self.openai_client:
                return await self._generate_openai_explanation(prompt)
            # Otherwise use Anthropic
            elif self.anthropic_client:
                return await self._generate_anthropic_explanation(prompt)
            else:
                # Fallback to simple explanation
                return self._generate_simple_explanation(decision)
                
        except Exception as e:
            logging.error(f"Error generating explanation: {str(e)}")
            return self._generate_simple_explanation(decision)
    
    async def _generate_openai_explanation(self, prompt: str) -> str:
        """Generate explanation using OpenAI."""
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",  # or another available model
                messages=[
                    {"role": "system", "content": "You are a quantum-enhanced decision system explaining your selections."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error generating OpenAI explanation: {str(e)}")
            return "I selected this option based on the criteria and constraints provided."
    
    async def _generate_anthropic_explanation(self, prompt: str) -> str:
        """Generate explanation using Anthropic."""
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-opus-20240229",  # or another available model
                max_tokens=500,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            logging.error(f"Error generating Anthropic explanation: {str(e)}")
            return "I selected this option based on the criteria and constraints provided."
    
    def _generate_simple_explanation(self, decision: Decision) -> str:
        """
        Generate a simple explanation without using AI.
        
        Args:
            decision: Decision to explain
            
        Returns:
            Simple explanation string
        """
        selected = decision.selected_option
        if not selected:
            return "No option was selected due to insufficient information."
        
        # Get option name or ID
        option_name = selected.get("name", selected.get("id", "The selected option"))
        
        # Build explanation
        explanation = f"{option_name} was selected with {decision.confidence:.1%} confidence using {decision.method_used} processing."
        
        # Add criteria evaluation if available
        if decision.criteria:
            explanation += "\n\nKey criteria evaluation:"
            for criterion in decision.criteria[:3]:  # Top 3 criteria
                name = criterion.get("name", criterion.get("id", "Criterion"))
                key = criterion.get("key", "")
                if key and key in selected:
                    value = selected[key]
                    explanation += f"\n- {name}: {value}"
        
        # Add quantum advantage if applicable
        if decision.method_used == "quantum":
            explanation += "\n\nQuantum computing provided advantages in evaluating multiple interdependent criteria simultaneously."
        
        return explanation
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the decision system.
        
        Returns:
            Dictionary with performance metrics
        """
        return self.metrics


# Utility function to create decision request from options and criteria
def create_decision_request(
    options: List[Dict[str, Any]],
    criteria: List[Dict[str, Any]],
    context: Dict[str, Any] = None,
    constraints: List[Dict[str, Any]] = None,
    decision_type: str = "general"
) -> DecisionRequest:
    """
    Create a decision request from options and criteria.
    
    Args:
        options: List of option dictionaries
        criteria: List of criteria dictionaries
        context: Decision context (optional)
        constraints: List of constraint dictionaries (optional)
        decision_type: Type of decision
        
    Returns:
        DecisionRequest object
    """
    if context is None:
        context = {}
    if constraints is None:
        constraints = []
    
    return DecisionRequest(
        context=context,
        options=options,
        criteria=criteria,
        constraints=constraints,
        decision_type=decision_type
    )


# Example function for testing the quantum decision system
async def test_quantum_decision_system() -> Dict[str, Any]:
    """
    Test the quantum decision system with a sample decision problem.
    
    Returns:
        Dict with test results
    """
    # Create a quantum decision system
    decision_system = QuantumDecisionSystem(n_qubits=6, use_quantum=True)
    
    # Sample options (e.g., product choices)
    options = [
        {
            "id": "product_a",
            "name": "Product A",
            "price": 50,
            "quality": 0.8,
            "delivery_time": 2,
            "eco_friendly": True
        },
        {
            "id": "product_b",
            "name": "Product B",
            "price": 40,
            "quality": 0.7,
            "delivery_time": 3,
            "eco_friendly": False
        },
        {
            "id": "product_c",
            "name": "Product C",
            "price": 70,
            "quality": 0.9,
            "delivery_time": 5,
            "eco_friendly": True
        },
        {
            "id": "product_d",
            "name": "Product D",
            "price": 30,
            "quality": 0.6,
            "delivery_time": 1,
            "eco_friendly": False
        }
    ]
    
    # Sample criteria
    criteria = [
        {
            "id": "price_criterion",
            "name": "Price",
            "key": "price",
            "weight": -0.8,  # Negative weight for minimization
            "description": "Lower price is better"
        },
        {
            "id": "quality_criterion",
            "name": "Quality",
            "key": "quality",
            "weight": 1.0,
            "description": "Higher quality is better"
        },
        {
            "id": "delivery_criterion",
            "name": "Delivery Time",
            "key": "delivery_time",
            "weight": -0.5,  # Negative weight for minimization
            "description": "Shorter delivery time is better"
        },
        {
            "id": "eco_criterion",
            "name": "Eco-Friendly",
            "key": "eco_friendly",
            "weight": 0.7,
            "description": "Eco-friendly is preferred"
        }
    ]
    
    # Sample constraints
    constraints = [
        {
            "type": "must_have",
            "condition": {
                "attribute": "quality",
                "operator": "gte",
                "value": 0.7
            },
            "description": "Quality must be at least 0.7"
        },
        {
            "type": "nice_to_have",
            "condition": {
                "attribute": "eco_friendly",
                "operator": "eq",
                "value": True
            },
            "description": "Eco-friendly is preferred"
        }
    ]
    
    # Create decision request
    request = create_decision_request(
        options=options,
        criteria=criteria,
        constraints=constraints,
        context={"user_preference": "balanced"},
        decision_type="product_selection"
    )
    
    # Make decision
    start_time = time.time()
    decision = await decision_system.make_decision(request)
    
    return {
        "success": True,
        "decision": {
            "selected_option": decision.selected_option,
            "confidence": decision.confidence,
            "method_used": decision.method_used,
            "processing_time": decision.processing_time
        },
        "explanation": decision.explanation,
        "test_time": time.time() - start_time
    }
"""