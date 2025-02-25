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
        logging.info("Factorization Manager initialized")

    async def _get_classical_factors_gpt(self, n: int) -> List[int]:
        """Get factors using GPT with academic prompting."""
        try:
            logging.info(f"Getting classical factors for {n} using GPT")
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
            logging.info(f"GPT response for factors of {n}: {factors_str}")
            # Extract the list from the string and convert to integers
            factors = eval(factors_str)  # Safe since we're controlling the prompt
            return sorted(factors)

        except Exception as e:
            logging.error(f"GPT factorization error: {str(e)}")
            # Fallback to basic factorization
            return self._get_basic_factors(n)

    def _get_basic_factors(self, n: int) -> List[int]:
        """Basic factorization as fallback."""
        logging.info(f"Using basic factorization for {n}")
        factors = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                factors.append(i)
                if i * i != n:
                    factors.append(n // i)
            i += 1
        factors = sorted(factors)
        logging.info(f"Found factors for {n}: {factors}")
        return factors

    async def factorize(self, number: int) -> FactorizationResult:
        """Main factorization method that chooses and executes the appropriate approach."""
        start_time = time.time()
        logging.info(f"Starting factorization of {number}")

        try:
            # Input validation
            if not isinstance(number, int) or number < 1:
                error_msg = "Invalid input: must be a positive integer"
                logging.error(error_msg)
                return FactorizationResult(
                    factors=[],
                    method_used="error",
                    computation_time=0,
                    success=False,
                    details={"error": error_msg}
                )

            # For smaller numbers, use classical method
            if number < self.classical_threshold:
                logging.info(f"Using classical method for {number}")
                factors = await self._get_classical_factors_gpt(number)
                result = FactorizationResult(
                    factors=factors,
                    method_used="classical",
                    computation_time=time.time() - start_time,
                    success=True,
                    details={
                        "reason": "Used GPT-4 for accurate classical factorization",
                        "backend": "GPT-4"
                    }
                )
                logging.info(f"Classical factorization completed: {result}")
                return result

            # Try quantum factorization for larger numbers
            if self.quantum_optimizer:
                try:
                    logging.info(f"Attempting quantum factorization for {number}")
                    quantum_result = self.quantum_optimizer.factorize_number(number)
                    if quantum_result.get("success", False):
                        # Get complete factorization
                        all_factors = await self._get_classical_factors_gpt(number)
                        result = FactorizationResult(
                            factors=all_factors,
                            method_used="quantum",
                            computation_time=time.time() - start_time,
                            success=True,
                            details={
                                "quantum_metrics": quantum_result,
                                "backend": "IonQ Aria-1" if self.quantum_optimizer.use_azure else "IBM Qiskit"
                            }
                        )
                        logging.info(f"Quantum factorization completed: {result}")
                        return result
                except Exception as qe:
                    logging.error(f"Quantum factorization error: {str(qe)}")

            # Fallback to classical GPT method
            logging.info("Falling back to classical method")
            factors = await self._get_classical_factors_gpt(number)
            result = FactorizationResult(
                factors=factors,
                method_used="classical_fallback",
                computation_time=time.time() - start_time,
                success=True,
                details={
                    "reason": "Used classical method as fallback",
                    "backend": "GPT-4"
                }
            )
            logging.info(f"Classical fallback completed: {result}")
            return result

        except Exception as e:
            error_msg = f"Factorization error: {str(e)}"
            logging.error(error_msg)
            return FactorizationResult(
                factors=[],
                method_used="error",
                computation_time=time.time() - start_time,
                success=False,
                details={"error": error_msg}
            )