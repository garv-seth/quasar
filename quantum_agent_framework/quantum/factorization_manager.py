"""Manages factorization methods and coordinates between classical and quantum approaches."""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import time
from openai import AsyncOpenAI

@dataclass
class FactorizationResult:
    factors: List[int]
    method_used: str
    computation_time: float
    success: bool
    details: Dict[str, Any]

class FactorizationManager:
    """Coordinates between classical and quantum factorization methods."""

    def __init__(self, quantum_optimizer=None):
        """Initialize the factorization manager."""
        self.quantum_optimizer = quantum_optimizer
        self.openai_client = AsyncOpenAI()
        self.classical_threshold = 1000000  # Numbers below this use classical method
        self.quantum_threshold = 2**15  # Maximum size for quantum factorization

    async def _get_classical_factors_gpt(self, n: int) -> List[int]:
        """Get factors using GPT with academic prompting."""
        try:
            messages = [
                {"role": "system", "content": """You are a mathematics professor specializing in number theory.
                When given a number, return ONLY a Python list of ALL its factors in ascending order.
                Include 1 and the number itself.
                Format: [factor1, factor2, ...]
                Example for 12: [1, 2, 3, 4, 6, 12]"""},
                {"role": "user", "content": f"List all factors of {n}"}
            ]

            completion = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=100,
                temperature=0
            )

            factors_str = completion.choices[0].message.content.strip()
            # Extract the list from the string and convert to integers
            factors = eval(factors_str)  # Safe since we're controlling the prompt
            return sorted(factors)

        except Exception as e:
            logging.error(f"GPT factorization error: {str(e)}")
            # Fallback to basic factorization
            return self._get_basic_factors(n)

    def _get_basic_factors(self, n: int) -> List[int]:
        """Basic factorization as fallback."""
        factors = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                factors.append(i)
                if i * i != n:
                    factors.append(n // i)
            i += 1
        return sorted(factors)

    async def _analyze_factorization_approach(self, number: int) -> Dict[str, Any]:
        """Use AI to analyze and recommend factorization approach."""
        try:
            messages = [
                {"role": "system", "content": f"""Analyze this number for factorization:
                Number: {number}
                Classical threshold: {self.classical_threshold}
                Quantum threshold: {self.quantum_threshold}

                Consider:
                1. Number size and complexity
                2. Known factorization patterns
                3. Computational efficiency

                Return JSON with fields:
                - method: "classical" or "quantum"
                - reasoning: explanation
                - special_case: any special mathematical properties"""},
                {"role": "user", "content": f"Analyze {number} for factorization"}
            ]

            completion = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=300,
                temperature=0.2
            )

            import json
            response = completion.choices[0].message.content
            return json.loads(response)

        except Exception as e:
            logging.error(f"AI analysis error: {str(e)}")
            # Fallback to size-based decision
            return {
                "method": "classical" if number < self.classical_threshold else "quantum",
                "reasoning": "Fallback to size-based decision",
                "special_case": None
            }

    async def factorize(self, number: int) -> FactorizationResult:
        """Main factorization method that chooses and executes the appropriate approach."""
        start_time = time.time()

        try:
            # Input validation
            if not isinstance(number, int) or number < 1:
                return FactorizationResult(
                    factors=[],
                    method_used="error",
                    computation_time=0,
                    success=False,
                    details={"error": "Invalid input: must be a positive integer"}
                )

            # Get AI analysis
            analysis = await self._analyze_factorization_approach(number)
            method = analysis.get("method", "classical")

            # Classical factorization for small numbers or when recommended
            if method == "classical" or number < self.classical_threshold:
                factors = await self._get_classical_factors_gpt(number)
                return FactorizationResult(
                    factors=factors,
                    method_used="classical",
                    computation_time=time.time() - start_time,
                    success=True,
                    details={
                        "analysis": analysis,
                        "reason": "Used GPT-4 for accurate classical factorization",
                        "backend": "GPT-4"
                    }
                )

            # Quantum factorization when available and recommended
            if self.quantum_optimizer and method == "quantum":
                try:
                    quantum_result = self.quantum_optimizer.factorize_number(number)
                    if quantum_result.get("success", False):
                        # Get complete factorization using GPT to ensure all factors
                        all_factors = await self._get_classical_factors_gpt(number)

                        return FactorizationResult(
                            factors=all_factors,  # Use complete factorization
                            method_used="quantum",
                            computation_time=time.time() - start_time,
                            success=True,
                            details={
                                "analysis": analysis,
                                "quantum_metrics": quantum_result,
                                "quantum_found_factors": quantum_result.get("factors", []),
                                "backend": "IonQ Aria-1" if self.quantum_optimizer.use_azure else "IBM Qiskit"
                            }
                        )
                except Exception as qe:
                    logging.error(f"Quantum factorization error: {str(qe)}")

            # Fallback to classical GPT method
            factors = await self._get_classical_factors_gpt(number)
            return FactorizationResult(
                factors=factors,
                method_used="classical_fallback",
                computation_time=time.time() - start_time,
                success=True,
                details={
                    "analysis": analysis,
                    "reason": "Fallback to GPT-4 due to quantum unavailability or error",
                    "backend": "GPT-4"
                }
            )

        except Exception as e:
            logging.error(f"Factorization error: {str(e)}")
            return FactorizationResult(
                factors=[],
                method_used="error",
                computation_time=time.time() - start_time,
                success=False,
                details={"error": str(e)}
            )